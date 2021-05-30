import numpy as np
import sentencepiece as spm
from collections import Counter


def hanja_vocab_processing(args, hanja_list, additional_hanja_list):
    # Hanja data train & test split
    train_hanja, valid_hanja, test_hanja = hanja_list
    train_add_hanja, valid_add_hanja, test_add_hanja = additional_hanja_list

    # Processing the blank token
    train_hanja = tuple(x.replace(' ', '_') for x in train_hanja)
    valid_hanja = tuple(x.replace(' ', '_') for x in valid_hanja)
    test_hanja = tuple(x.replace(' ', '_') for x in test_hanja)
    train_add_hanja = tuple(x.replace(' ', '_') for x in train_add_hanja)
    valid_add_hanja = tuple(x.replace(' ', '_') for x in valid_add_hanja)
    test_add_hanja = tuple(x.replace(' ', '_') for x in test_add_hanja)
    
    word_counter = Counter()
    hanja_word2id = ['<pad>', '<unk>', '[MASK]']

    for sentence in train_hanja:
        for word in sentence:
            word_counter.update(word)

    for sentence in train_add_hanja:
        for word in sentence:
            word_counter.update(word)

    hanja_word2id.extend([w for w, freq in word_counter.items() if freq >= 10])
    hanja_word2id = {w: i for i, w in enumerate(hanja_word2id)}

    train_hanja_indices = tuple([hanja_word2id.get(w, args.unk_id) for w in hanja] for hanja in train_hanja)
    valid_hanja_indices = tuple([hanja_word2id.get(w, args.unk_id) for w in hanja] for hanja in valid_hanja)
    test_hanja_indices = tuple([hanja_word2id.get(w, args.unk_id) for w in hanja] for hanja in test_hanja)

    train_additional_hanja_indices = tuple(
        [hanja_word2id.get(w, args.unk_id) for w in hanja] for hanja in train_add_hanja
    )
    valid_additional_hanja_indices = tuple(
        [hanja_word2id.get(w, args.unk_id) for w in hanja] for hanja in valid_add_hanja
    )
    test_additional_hanja_indices = tuple(
        [hanja_word2id.get(w, args.unk_id) for w in hanja] for hanja in test_add_hanja
    )

    return (hanja_word2id,
            train_hanja_indices, valid_hanja_indices, test_hanja_indices,
            train_additional_hanja_indices, valid_additional_hanja_indices, test_additional_hanja_indices)


def korean_vocab_train(args, train_korean, valid_korean, test_korean):
    with open(f'{args.output_path}/korean.txt', 'w') as f:
        for korean in train_korean:
            f.write(f'{korean}\n')

    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={args.output_path}/korean.txt --model_prefix={args.output_path}/m_korean '
        f'--vocab_size={args.vocab_size} --character_coverage=0.9999 --split_by_whitespace=true '
        f'--pad_id={args.pad_id} --unk_id={args.unk_id} --bos_id={args.bos_id} --eos_id={args.eos_id} '
        f'--max_sentence_length=10000000')

    korean_vocab = list()
    with open(f'{args.output_path}/m_korean.vocab') as f:
        for line in f:
            korean_vocab.append(line[:-1].split('\t')[0])

    korean_word2id = {w: i for i, w in enumerate(korean_vocab)}
    sp_kr = spm.SentencePieceProcessor()
    sp_kr.Load(f"{args.output_path}/m_korean.model")

    train_korean_indices = tuple(
        [args.bos_id] + sp_kr.encode(
                            korean, enable_sampling=True, alpha=0.1, nbest_size=-1, out_type=int) + \
        [args.eos_id] for korean in train_korean
    )
    valid_korean_indices = tuple(
        [args.bos_id] + sp_kr.encode(korean, out_type=int) + [args.eos_id] for korean in valid_korean
    )
    test_korean_indices = tuple(
        [args.bos_id] + sp_kr.encode(korean, out_type=int) + [args.eos_id] for korean in test_korean
    )

    return korean_word2id, train_korean_indices, valid_korean_indices, test_korean_indices


def train_test_split(record_list, additional_record_list, valid_num=20000, test_num=30000):
    # Paired data split
    paired_data_len = len(record_list)

    valid_index = np.random.choice(paired_data_len, valid_num, replace=False)
    train_index = list(set(range(paired_data_len)) - set(valid_index))
    test_index = np.random.choice(train_index, test_num, replace=False)
    train_index = list(set(train_index) - set(test_index))

    train_record_list = tuple(record_list[i] for i in train_index)
    valid_record_list = tuple(record_list[i] for i in valid_index)
    test_record_list = tuple(record_list[i] for i in test_index)

    split_record = {'train': train_record_list, 'valid': valid_record_list,  'test': test_record_list}

    # Additional data split
    additional_data_len = len(additional_record_list)

    valid_index = np.random.choice(additional_data_len, valid_num, replace=False)
    train_index = list(set(range(additional_data_len)) - set(valid_index))
    test_index = np.random.choice(train_index, test_num, replace=False)
    train_index = list(set(train_index) - set(test_index))

    train_additional_record_list = tuple(additional_record_list[i] for i in train_index)
    valid_additional_record_list = tuple(additional_record_list[i] for i in valid_index)
    test_additional_record_list = tuple(additional_record_list[i] for i in test_index)

    split_additional_record = {
        'train': train_additional_record_list,
        'valid': valid_additional_record_list,
        'test': test_additional_record_list
    }

    return split_record, split_additional_record
    