import argparse
from math import ceil
from multiprocessing import Process
from time import time

import numpy as np
import requests
import sentencepiece as spm
import torch
from torch.nn import functional as F

from model.transformer import Transformer


def main(proc_id, args):
    trg_sp = spm.SentencePieceProcessor()
    trg_sp.Load(args.spm_trg_path)
    trg_vocab_num = trg_sp.piece_size()
    bos_id = trg_sp.bos_id()
    eos_id = trg_sp.eos_id()
    pad_id = trg_sp.pad_id()
    src_vocab = requests.get(f'{args.api_url}/getMetaData').json()['src_vocab']
    unk_id = src_vocab['<unk>']

    device = torch.device(f"cuda:{proc_id}")
    model = Transformer(len(src_vocab), trg_vocab_num, pad_idx=pad_id, bos_idx=bos_id, eos_idx=eos_id,
                src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                d_model=args.d_model, d_embedding=args.d_embedding, n_head=args.n_head, 
                dim_feedforward=args.dim_feedforward, num_encoder_layer=args.num_encoder_layer, 
                num_decoder_layer=args.num_decoder_layer, num_mask_layer=args.num_mask_layer)

    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device)['model'])
    model.src_output_linear = None
    model.src_output_linear2 = None
    model.src_output_norm = None
    model.mask_encoders = None
    model = model.to(device)
    model = model.eval()

    tgt_masks = {l: model.generate_square_subsequent_mask(l, device) for l in range(1, args.trg_max_len + 1)}

    while True:
        data = requests.get(f'{args.api_url}/getData').json()
        pred_data = {'file': data['file'], 'content': []}
        parsed_ids = []
        for d in data['content']:
            parsed_id = [src_vocab.get(c, unk_id) for c in d['hanja']]
            if args.min_len <= len(parsed_id) <= args.src_max_len:
                input_id = np.zeros(args.src_max_len, dtype=np.int64)
                input_id[:len(parsed_id)] = parsed_id
                parsed_ids.append(input_id)
                pred_data['content'].append(d)

        num_iter = ceil(len(parsed_ids)/args.batch_size)
        batch_size_ = args.batch_size
        predicted_num = 0

        with torch.no_grad():
            batch_indices = torch.arange(0, args.beam_size * args.batch_size, args.beam_size, device=device)
            for iter_ in range(num_iter):
                iter_time = time()
                src_sequences = parsed_ids[iter_*args.batch_size: (iter_ + 1) * args.batch_size]

                scores_save = torch.zeros(args.beam_size * args.batch_size, 1, device=device)
                top_k_scores = torch.zeros(args.beam_size * args.batch_size, 1, device=device)
                complete_seqs = dict()
                complete_ind = set()
                if len(src_sequences) < args.batch_size:
                    batch_size_ = len(src_sequences)
                    batch_indices = torch.arange(0, args.beam_size * batch_size_, args.beam_size, device=device)
                    scores_save = torch.zeros(args.beam_size * batch_size_, 1, device=device)
                    top_k_scores = torch.zeros(args.beam_size * batch_size_, 1, device=device)

                src_sequences = torch.cat([torch.cuda.LongTensor(seq, device=device) for seq in src_sequences])
                src_sequences = src_sequences.view(batch_size_, args.src_max_len)

                # Encoding
                # encoder_out: (src_seq, batch_size, d_model), src_key_padding_mask: (batch_size, src_seq)
                encoder_out = model.src_embedding(src_sequences).transpose(0, 1) 
                src_key_padding_mask = (src_sequences == pad_id)
                for encoder in model.encoders:
                    encoder_out = encoder(encoder_out, src_key_padding_mask=src_key_padding_mask)

                # Expanding
                # encoder_out: (src_seq, batch_size*k, d_model), src_key_padding_mask: (batch_size*k, src_seq)
                src_seq_size = encoder_out.size(0)
                src_key_padding_mask = src_key_padding_mask.view(
                    batch_size_, 1, -1).repeat(1, args.beam_size, 1)
                src_key_padding_mask = src_key_padding_mask.view(-1, src_seq_size)
                encoder_out = encoder_out.view(-1, batch_size_, 1, args.d_model).repeat(1, 1, args.beam_size, 1)
                encoder_out = encoder_out.view(src_seq_size, -1, args.d_model)

                # Decoding start token setting
                seqs = torch.tensor([[bos_id]], dtype=torch.long, device=device) 
                seqs = seqs.repeat(args.beam_size*batch_size_, 1).contiguous()

                for step in range(model.trg_max_len):
                    # Decoder setting
                    # tgt_mask: (out_seq), tgt_key_padding_mask: (batch_size * k, out_seq)
                    tgt_mask = tgt_masks[seqs.size(1)]
                    tgt_key_padding_mask = (seqs == pad_id)

                    # Decoding sentence
                    # decoder_out: (out_seq, batch_size * k, d_model)
                    decoder_out = model.trg_embedding(seqs).transpose(0, 1)
                    for decoder in model.decoders:
                        decoder_out = decoder(decoder_out, encoder_out, tgt_mask=tgt_mask, 
                            memory_key_padding_mask=src_key_padding_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask)

                    # Score calculate
                    # scores: (batch_size * k, vocab_num)
                    scores = F.gelu(model.trg_output_linear(decoder_out[-1]))
                    scores = model.trg_output_linear2(model.trg_output_norm(scores))
                    scores = F.log_softmax(scores, dim=1) 

                    # Repetition Penalty
                    if step > 0 and args.repetition_penalty > 0:
                        prev_ix = next_word_inds.view(-1)
                        for index, prev_token_id in enumerate(prev_ix):
                            scores[index][prev_token_id] *= args.repetition_penalty

                    # Add score
                    scores = top_k_scores.expand_as(scores) + scores 
                    if step == 0:
                        # scores: (batch_size, vocab_num)
                        # top_k_scores: (batch_size, k)
                        scores = scores[::args.beam_size] 
                        # set eos token probability zero in first step
                        scores[:, eos_id] = float('-inf')
                        top_k_scores, top_k_words = scores.topk(args.beam_size, 1, True, True)  
                    else:
                        # top_k_scores: (batch_size * k, out_seq)
                        top_k_scores, top_k_words = scores.view(
                            batch_size_, -1).topk(args.beam_size, 1, True, True)

                    # Previous and Next word extract
                    # seqs: (batch_size * k, out_seq + 1)
                    prev_word_inds = top_k_words // trg_vocab_num
                    next_word_inds = top_k_words % trg_vocab_num
                    top_k_scores = top_k_scores.view(batch_size_ * args.beam_size, -1)
                    top_k_words = top_k_words.view(batch_size_ * args.beam_size, -1)
                    seqs = seqs[prev_word_inds.view(-1) + batch_indices.unsqueeze(1).repeat(1, args.beam_size).view(-1)]
                    seqs = torch.cat([seqs, next_word_inds.view(args.beam_size * batch_size_, -1)], dim=1) 

                    # Find and Save Complete Sequences Score
                    eos_ind = torch.where(next_word_inds.view(-1) == eos_id)[0]
                    if len(eos_ind) > 0:
                        eos_ind = eos_ind.tolist()
                        complete_ind_add = set(eos_ind) - complete_ind
                        complete_ind_add = list(complete_ind_add)
                        complete_ind.update(eos_ind)
                        if len(complete_ind_add) > 0:
                            scores_save[complete_ind_add] = top_k_scores[complete_ind_add]
                            for ix in complete_ind_add:
                                complete_seqs[ix] = seqs[ix].tolist()

                # If eos token doesn't exist in sequence
                score_save_pos = torch.where(scores_save == 0)
                if len(score_save_pos[0]) > 0:
                    for ix in score_save_pos[0].tolist():
                        complete_seqs[ix] = seqs[ix].tolist()
                    scores_save[score_save_pos] = top_k_scores[score_save_pos]

                # Beam Length Normalization
                lp = torch.tensor(
                    [len(complete_seqs[i]) for i in range(batch_size_ * args.beam_size)], device=device)
                lp = (((lp + args.beam_size) ** args.beam_alpha) / ((args.beam_size + 1) ** args.beam_alpha))
                scores_save = scores_save / lp.unsqueeze(1)

                # Predicted and Label processing
                ind = scores_save.view(batch_size_, args.beam_size, -1).argmax(dim=1)
                ind = (ind.view(-1) + batch_indices).tolist()
                for i in ind:
                    predicted_sequence = trg_sp.decode_ids(complete_seqs[i])
                    pred_data['content'][predicted_num]['predicted_sequence'] = predicted_sequence
                    predicted_num += 1

                iter_time = time() - iter_time
                print(f"{proc_id} - iter: {iter_ + 1}/{num_iter}, {iter_time:.2f}")

        res = requests.post(f'{args.api_url}/commitData', json=pred_data).json()
        print(f"{proc_id} - Progress: {res['progress']}, {pred_data['file']}")
        if res['progress'] == 'finish':
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test machine translation.')
    parser.add_argument('--spm_trg_path', 
        default='/home/nas1_userC/rudvlf0413/joseon_translation/dataset/preprocessed/m_korean.model', type=str)
    parser.add_argument('--checkpoint_path', default='./models/model_12_6_12_ckpt.pt', type=str)

    parser.add_argument('--min_len', default=3, type=int)
    parser.add_argument('--src_max_len', default=300, type=int)
    parser.add_argument('--trg_max_len', default=350, type=int)

    parser.add_argument('--d_model', default=768, type=int)
    parser.add_argument('--d_embedding', default=256, type=int)
    parser.add_argument('--n_head', default=12, type=int)
    parser.add_argument('--dim_feedforward', default=768*4, type=int)
    parser.add_argument('--num_encoder_layer', default=12, type=int)
    parser.add_argument('--num_decoder_layer', default=12, type=int)
    parser.add_argument('--num_mask_layer', default=6, type=int)
    
    parser.add_argument('--beam_size', default=3, type=int)
    parser.add_argument('--batch_size', default=80, type=int)
    parser.add_argument('--beam_alpha', default=0.7, type=float, help='beam search length normalization alpha')
    parser.add_argument('--repetition_penalty', default=1.3, type=float, help='repetition penalty')

    parser.add_argument('--gpu_workers', default='0,1,2,3,4,5,6,7', type=str, help='gpu workers')
    parser.add_argument('--api_url', default='http://0.0.0.0:5010', type=str, help='parallel api url')
    args = parser.parse_args()

    procs = []
    gpu_workers = args.gpu_workers.split(',')
    torch.multiprocessing.set_start_method('spawn')
    for i in gpu_workers:
        procs.append(Process(target=main, args=(i, args)))

    for proc in procs:
        proc.start()

    for proc in procs:
        proc.join()
