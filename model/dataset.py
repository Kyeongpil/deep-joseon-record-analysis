from random import random, randrange

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class HanjaKoreanDataset(Dataset):
    def __init__(self, hanja_list, korean_list, min_len=4, src_max_len=300, trg_max_len=360):
        self.tensor_list = []
        for h, k in zip(hanja_list, korean_list):
            if min_len <= len(h) <= src_max_len and min_len <= len(k) <= trg_max_len:
                h_tensor = torch.zeros(src_max_len, dtype=torch.long)
                h_tensor[:len(h)] = torch.tensor(h, dtype=torch.long)
                k_tensor = torch.zeros(trg_max_len, dtype=torch.long)
                k_tensor[:len(k)] = torch.tensor(k, dtype=torch.long)
                self.tensor_list.append((h_tensor, k_tensor))

        self.tensor_list = tuple(self.tensor_list)
        self.num_data = len(self.tensor_list)

    def __getitem__(self, index):
        return self.tensor_list[index]

    def __len__(self):
        return self.num_data


class HanjaDataset(Dataset):
    def __init__(self, hanja_list, additional_hanja, word2id, min_len=4, src_max_len=512):
        
        self.src_max_len = src_max_len
        self.pad_idx = word2id['<pad>']
        self.mask_idx = word2id['[MASK]']
        self.blank_idx = word2id['_']
        unk_idx = word2id['<unk>']

        data = []
        for hanja in hanja_list:
            if min_len <= len(hanja) <= src_max_len:
                data.append(np.array(hanja, dtype=np.long))

        for hanja in additional_hanja:
            if min_len <= len(hanja) <= src_max_len:
                data.append(np.array(hanja, dtype=np.long))

        self.data = tuple(data)  
        self.num_data = len(self.data) 

        self.stopword_ids = set([self.pad_idx, self.mask_idx, self.blank_idx, unk_idx])
        self.vocab_indices = np.arange(len(word2id))
        self.vocab_prob = np.ones(len(word2id))
        self.vocab_prob[[self.pad_idx, self.mask_idx, self.blank_idx, unk_idx]] = 0
        self.vocab_prob /= self.vocab_prob.sum()

    def __getitem__(self, index):
        sentence = self.data[index]
        masked_sentence, label = self.get_random_tokens(sentence)
        sentence_tensor = torch.zeros(self.src_max_len, dtype=torch.long)
        sentence_tensor[:len(masked_sentence)] = torch.tensor(masked_sentence, dtype=torch.long)
        labelce_tensor = torch.zeros(self.src_max_len, dtype=torch.long)
        labelce_tensor[:len(label)] = torch.tensor(label, dtype=torch.long)
        return sentence_tensor, labelce_tensor

    def __len__(self):
        return self.num_data

    def get_random_tokens(self, sequence, mask_ratio=0.16):
        masked_sentence = np.array(sequence, copy=True)
        output_label = np.ones(len(sequence), dtype=np.long) * self.pad_idx
        num_mask_pos = max(1, round(len(masked_sentence) * mask_ratio))
        n, i, trial = 0, 0, 0
        
        while trial < 50 and n < num_mask_pos:
            trial += 1
            if output_label[i] != self.pad_idx or sequence[i] == self.blank_idx:
                i += 1
                continue

            # unigram: 0.6, bigram: 0.3, trigram: 1
            # 85%: masking, 10%: random token, 5%: original token
            p1 = random()
            if p1 <= mask_ratio:
                # mask
                p2 = random()
                if p2 >= 0.9 and i < len(sequence) - 2:
                    # trigram
                    if sequence[i + 1] not in self.stopword_ids and sequence[i + 2] not in self.stopword_ids:
                        output_label[i: i + 3] = sequence[i: i + 3]
                        
                        p3 = random()
                        if p3 < 0.85:
                            masked_sentence[i: i + 3] = self.mask_idx
                        elif p3 < 0.95:
                            # replace tokens to random tokens
                            masked_sentence[i: i + 3] = np.random.choice(
                                self.vocab_indices, size=3, p=self.vocab_prob)

                        n += 3
                    i += 4

                elif 0.6 < p2 < 0.9 and i < len(sequence) - 1:
                    # bigram
                    if sequence[i + 1] not in self.stopword_ids:
                        output_label[i: i + 2] = sequence[i: i + 2]
                        p3 = random()
                        if p3 < 0.85:
                            masked_sentence[i: i + 2] = self.mask_idx
                        elif p3 < 0.95:
                            # replace tokens to random tokens
                            masked_sentence[i: i + 2] = np.random.choice(
                                self.vocab_indices, size=2, p=self.vocab_prob)

                        n += 2
                    i += 3

                else:
                    # unigram
                    output_label[i] = sequence[i]
                    p3 = random()
                    if p3 < 0.85:
                        masked_sentence[i] = self.mask_idx
                    elif p3 < 0.95:
                        masked_sentence[i] = np.random.choice(self.vocab_indices, p=self.vocab_prob)
                    n += 1
                    i += 2

            else:
                # mask input 말고 나머지는 pad index로 해서 training loss에서 제외하도록
                i += 1

            if i >= len(sequence):
                break

        if n == 0:
            # 아무것도 마스킹 되지 않았으면 한 토큰에 대해서 마스킹 하거나 다른 토큰으로 대체
            while True:
                i = randrange(len(sequence))
                if sequence[i] not in self.stopword_ids:
                    output_label[i] = sequence[i]
                    p3 = random()
                    if p3 < 0.85:
                        masked_sentence[i] = self.mask_idx
                    elif p3 < 0.95:
                        masked_sentence[i] = np.random.choice(self.vocab_indices, p=self.vocab_prob)

                    break

        return masked_sentence, output_label
