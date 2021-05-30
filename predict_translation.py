import argparse
import gc
import logging
import os
import pickle

import sentencepiece as spm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import HanjaKoreanDataset
from model.transformer import Transformer
from utils import TqdmLoggingHandler, write_log


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, "Load data")
    def load_data(args):
        gc.disable()
        with open(f"{args.preprocessed_data_path}/hanja_korean_word2id.pkl", "rb") as f:
            data = pickle.load(f)
            hanja_word2id = data['hanja_word2id']
            korean_word2id = data['korean_word2id']

        with open(f"{args.preprocessed_data_path}/preprocessed_test.pkl", "rb") as f:
            data = pickle.load(f)
            test_hanja_indices = data['hanja_indices']
            test_korean_indices = data['korean_indices']

        gc.enable()
        write_log(logger, "Finished loading data!")
        return hanja_word2id, korean_word2id, test_hanja_indices, test_korean_indices

    hanja_word2id, korean_word2id, test_hanja_indices, test_korean_indices = load_data(args)
    hanja_vocab_num = len(hanja_word2id)
    korean_vocab_num = len(korean_word2id)

    hk_dataset = HanjaKoreanDataset(test_hanja_indices, test_korean_indices, 
                    min_len=args.min_len, src_max_len=args.src_max_len, trg_max_len=args.trg_max_len)
    hk_loader = DataLoader(hk_dataset, drop_last=True, batch_size=args.hk_batch_size, 
                    num_workers=4, prefetch_factor=4, pin_memory=True)
    write_log(logger, f"hanja-korean: {len(hk_dataset)}, {len(hk_loader)}")
    del test_hanja_indices, test_korean_indices

    write_log(logger, "Build model")
    model = Transformer(hanja_vocab_num, korean_vocab_num, pad_idx=args.pad_idx, bos_idx=args.bos_idx,
        eos_idx=args.eos_idx, src_max_len=args.src_max_len, trg_max_len=args.trg_max_len, 
        d_model=args.d_model, d_embedding=args.d_embedding, n_head=args.n_head, 
        dim_feedforward=args.dim_feedforward, num_encoder_layer=args.num_encoder_layer, 
        num_decoder_layer=args.num_decoder_layer, num_mask_layer=args.num_mask_layer)

    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device)['model'])
    model.src_output_linear = None
    model.src_output_linear2 = None
    model.src_output_norm = None
    model.mask_encoders = None
    model = model.to(device)
    model.eval()

    write_log(logger, "Load SentencePiece model")
    parser = spm.SentencePieceProcessor()
    parser.Load(os.path.join(args.preprocessed_data_path, 'm_korean.model'))

    predicted_list = list()
    label_list = list()
    every_batch = torch.arange(0, args.beam_size * args.hk_batch_size, args.beam_size, device=device)
    tgt_masks = {l: model.generate_square_subsequent_mask(l, device) for l in range(1, args.trg_max_len + 1)}

    with torch.no_grad():
        for src_sequences, trg_sequences in tqdm(hk_loader):
            src_sequences = src_sequences.to(device)
            label_list.extend(trg_sequences.tolist())

            # Encoding
            # encoder_out: (src_seq, batch_size, d_model)
            # src_key_padding_mask: (batch_size, src_seq)
            encoder_out = model.src_embedding(src_sequences).transpose(0, 1) 
            src_key_padding_mask = (src_sequences == model.pad_idx)
            for encoder in model.encoders:
                encoder_out = encoder(encoder_out, src_key_padding_mask=src_key_padding_mask)

            # Expanding
            # encoder_out: (src_seq, batch_size * k, d_model)
            # src_key_padding_mask: (batch_size * k, src_seq)
            src_seq_size = encoder_out.size(0)
            src_key_padding_mask = src_key_padding_mask.view(
                args.hk_batch_size, 1, -1).repeat(1, args.beam_size, 1)
            src_key_padding_mask = src_key_padding_mask.view(-1, src_seq_size)
            encoder_out = encoder_out.view(
                -1, args.hk_batch_size, 1, args.d_model).repeat(1, 1, args.beam_size, 1)
            encoder_out = encoder_out.view(src_seq_size, -1, args.d_model)

            # Scores save vector & decoding list setting
            scores_save = torch.zeros(args.beam_size * args.hk_batch_size, 1, device=device)
            top_k_scores = torch.zeros(args.beam_size * args.hk_batch_size, 1, device=device)
            complete_seqs = dict()
            complete_ind = set()

            # Decoding start token setting
            seqs = torch.tensor([[model.bos_idx]], dtype=torch.long, device=device) 
            seqs = seqs.repeat(args.beam_size*args.hk_batch_size, 1).contiguous()

            for step in range(model.trg_max_len):
                # Decoder setting
                # tgt_mask: (out_seq)
                # tgt_key_padding_mask: (batch_size * k, out_seq)
                tgt_mask = tgt_masks[seqs.size(1)]
                tgt_key_padding_mask = (seqs == model.pad_idx)

                # Decoding sentence
                # decoder_out: (out_seq, batch_size * k, d_model)
                decoder_out = model.trg_embedding(seqs).transpose(0, 1)
                for decoder in model.decoders:
                    decoder_out = decoder(decoder_out, encoder_out, tgt_mask=tgt_mask, 
                        memory_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)

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
                    scores[:, model.eos_idx] = float('-inf') # set eos token probability zero in first step
                    top_k_scores, top_k_words = scores.topk(args.beam_size, 1, True, True)
                else:
                    # top_k_scores: (batch_size * k, out_seq)
                    top_k_scores, top_k_words = scores.view(
                        args.hk_batch_size, -1).topk(args.beam_size, 1, True, True)

                # Previous and Next word extract
                # seqs: (batch_size * k, out_seq + 1)
                prev_word_inds = top_k_words // korean_vocab_num
                next_word_inds = top_k_words % korean_vocab_num
                top_k_scores = top_k_scores.view(args.hk_batch_size * args.beam_size, -1)
                top_k_words = top_k_words.view(args.hk_batch_size * args.beam_size, -1)
                seqs = seqs[prev_word_inds.view(-1) + every_batch.unsqueeze(1).repeat(1, args.beam_size).view(-1)]
                seqs = torch.cat([seqs, next_word_inds.view(args.beam_size * args.hk_batch_size, -1)], dim=1) 

                # Find and Save Complete Sequences Score
                eos_ind = torch.where(next_word_inds.view(-1) == model.eos_idx)[0]
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
                [len(complete_seqs[i]) for i in range(args.hk_batch_size * args.beam_size)], device=device)
            lp = (((lp + args.beam_size) ** args.beam_alpha) / ((args.beam_size + 1) ** args.beam_alpha))
            scores_save = scores_save / lp.unsqueeze(1)

            # Predicted and Label processing
            ind = scores_save.view(args.hk_batch_size, args.beam_size, -1).argmax(dim=1)
            ind_expand = ind.view(-1) + every_batch
            predicted_list.extend([complete_seqs[i] for i in ind_expand.tolist()])

    with open(f'./results_beam_{args.beam_size}_{args.beam_alpha}_{args.repetition_penalty}.pkl', 'wb') as f:
        pickle.dump({
            'prediction': predicted_list, 
            'label': label_list,
            'prediction_decode': [parser.DecodeIds(pred) for pred in predicted_list],
            'label_decode': [parser.DecodeIds(label) for label in label_list]
        }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test machine translation.')
    parser.add_argument('--preprocessed_data_path',
        default='/home/nas1_userC/rudvlf0413/joseon_translation/dataset/preprocessed/', type=str, 
        help='path of data pickle file (train)')
    parser.add_argument('--checkpoint_path', default='./models/model_12_6_12_ckpt.pt', type=str)

    parser.add_argument('--min_len', default=3, type=int)
    parser.add_argument('--src_max_len', default=300, type=int, help='max length of the source sentence')
    parser.add_argument('--trg_max_len', default=350, type=int, help='max length of the target sentence')
    parser.add_argument('--pad_idx', default=0, type=int)
    parser.add_argument('--bos_idx', default=1, type=int)
    parser.add_argument('--eos_idx', default=2, type=int)

    parser.add_argument('--d_model', default=768, type=int)
    parser.add_argument('--d_embedding', default=256, type=int)
    parser.add_argument('--n_head', default=12, type=int)
    parser.add_argument('--dim_feedforward', default=3072, type=int)
    parser.add_argument('--num_encoder_layer', default=12, type=int)
    parser.add_argument('--num_mask_layer', default=6, type=int)
    parser.add_argument('--num_decoder_layer', default=12, type=int)
    
    parser.add_argument('--hk_batch_size', default=80, type=int)
    parser.add_argument('--beam_size', default=3, type=int)
    parser.add_argument('--beam_alpha', default=0.7, type=float, help='length normalization for beam search')
    parser.add_argument('--repetition_penalty', default=1.3, type=float, help='repetition penalty')
    parser.add_argument('--print_freq', default=30, type=int)
    args = parser.parse_args()
    main(args)
