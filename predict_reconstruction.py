import argparse
import gc
import pickle
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from model.dataset import HanjaDataset
from model.transformer import Transformer


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_data(args):
        gc.disable()
        with open(f"{args.preprocessed_data_path}/hanja_korean_word2id.pkl", "rb") as f:
            data = pickle.load(f)
            hanja_word2id = data['hanja_word2id']
            korean_word2id = data['korean_word2id']

        with open(f"{args.preprocessed_data_path}/preprocessed_test.pkl", "rb") as f:
            data = pickle.load(f)
            test_hanja_indices = data['hanja_indices']
            test_additional_hanja_indices = data['additional_hanja_indices']

        gc.enable()
        return hanja_word2id, korean_word2id, test_hanja_indices, test_additional_hanja_indices

    hanja_word2id, korean_word2id, hanja_indices, additional_hanja_indices = load_data(args)
    hanja_vocab_num = len(hanja_word2id)
    korean_vocab_num = len(korean_word2id)

    print('Loader and Model Setting...')
    h_dataset = HanjaDataset(hanja_indices, additional_hanja_indices, hanja_word2id, 
                    min_len=args.min_len, src_max_len=args.src_max_len)
    h_loader = DataLoader(h_dataset, drop_last=True, batch_size=args.batch_size, 
                    num_workers=4, prefetch_factor=4)

    model = Transformer(hanja_vocab_num, korean_vocab_num, pad_idx=args.pad_idx, bos_idx=args.bos_idx, 
                eos_idx=args.eos_idx, src_max_len=args.src_max_len, trg_max_len=args.trg_max_len, 
                d_model=args.d_model, d_embedding=args.d_embedding, n_head=args.n_head, 
                dim_feedforward=args.dim_feedforward, num_encoder_layer=args.num_encoder_layer, 
                num_decoder_layer=args.num_decoder_layer, num_mask_layer=args.num_mask_layer)

    model.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu')['model'])
    model.decoders = None
    model.trg_embedding = None
    model.trg_output_linear = None
    model.trg_output_linear2 = None
    model.trg_output_norm = None
    model = model.to(device)
    model.eval()

    masking_acc = defaultdict(float)

    with torch.no_grad():
        for inputs, labels in h_loader:
            # Setting
            inputs = inputs.to(device)
            labels = labels.to(device)
            masked_position = labels != args.pad_idx
            masked_labels = labels[masked_position].contiguous().view(-1).unsqueeze(1)
            total_mask_count = masked_labels.size(0)
            
            # Prediction, output: Batch * Length * Vocab
            pred = model.reconstruct_predict(inputs, masked_position=masked_position)
            _, pred = pred.topk(10, 1, True, True)
            
            # Top1, 5, 10
            masking_acc[1] += (torch.sum(masked_labels == pred[:, :1]).item()) / total_mask_count
            masking_acc[5] += (torch.sum(masked_labels == pred[:, :5]).item()) / total_mask_count
            masking_acc[10] += (torch.sum(masked_labels == pred).item()) / total_mask_count

    for key in masking_acc.keys():
        masking_acc[key] /= len(h_loader)

    for key, value in masking_acc.items():
        print(f'Top {key} Accuracy: {value:.4f}')

    with open('./mask_result.pkl', 'wb') as f:
        pickle.dump(masking_acc, f)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Test masking model.')
    parser.add_argument('--preprocessed_data_path', 
        default='/home/nas1_userC/rudvlf0413/joseon_translation/dataset/preprocessed/', type=str,
        help='path of data pickle file (test)')
    parser.add_argument('--checkpoint_path', default='./models/model_12_6_12_ckpt.pt', type=str)
    parser.add_argument('--min_len', default=4, type=int)
    parser.add_argument('--src_max_len', default=300, type=int)
    parser.add_argument('--trg_max_len', default=384, type=int)
    parser.add_argument('--pad_idx', default=0, type=int)
    parser.add_argument('--bos_idx', default=1, type=int)
    parser.add_argument('--eos_idx', default=2, type=int)
    parser.add_argument('--mask_idx', default=4, type=int)

    parser.add_argument('--d_model', default=768, type=int)
    parser.add_argument('--d_embedding', default=256, type=int)
    parser.add_argument('--n_head', default=12, type=int)
    parser.add_argument('--dim_feedforward', default=768*4, type=int)
    parser.add_argument('--num_encoder_layer', default=10, type=int)
    parser.add_argument('--num_mask_layer', default=4, type=int)
    parser.add_argument('--num_decoder_layer', default=10, type=int)
   
    parser.add_argument('--batch_size', default=80, type=int)

    args = parser.parse_args()
    main(args)
