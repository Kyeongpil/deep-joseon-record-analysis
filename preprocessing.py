# Import Module
import argparse
import os
import pickle
import re
from glob import glob

import numpy as np
import ujson as json

from vocab_utils import (hanja_vocab_processing, korean_vocab_train,
                         train_test_split)


def main(args):
    np.random.seed(0)
    
    j_path = glob(f'{args.joseon_record_path}/*/*.txt')
    s_path = glob(f'{args.sjw_path}/*/*.json')
    s_hj_path = glob(f'{args.sjw_hj_path}/*/*.json')
    # Preprocessing data path make
    if not os.path.exists(f'{args.output_path}'):
        os.mkdir(f'{args.output_path}')

    ## Missing processing
    # sjw_hanja_korean 기준 전체 262732 중 □: 2380,
    # 자 빠짐: 7, 원문 빠짐: 10414, 자 결락: 19, 원문 결락: 571, 원문 훼손: 114, 자 훼손: 4
    damage_words = ['자 빠짐', '원문 빠짐', '□', '자 결락', '원문 결락', '원문 훼손', '자 훼손']

    # Pre-processing
    record_list = []
    print('Joseon dynasty data pre-processing...', end=' ')
    for filename in j_path:
        with open(filename) as f:
            text = f.read()

        for record in text.split('\n\n=====\n\n')[:-1]:
            korean_content, hanja_content = record.split('\n\n-----\n\n')
            date, title, *korean_content = korean_content.split("\n")
            korean_content = " ".join(korean_content).strip()
            
            if '。' in korean_content:
                continue

            hanja_partition = hanja_content.partition('。')
            if '日' in hanja_partition[0] and len(hanja_partition[0]) <= 10:
                hanja_content = hanja_partition[2]

            if len(hanja_content) > 0:
                if '/' in hanja_content.split()[0]:
                    hanja_content = ' '.join(hanja_content.split('/')[1:])
            
            hanja_content = hanja_content.replace('○', ' ')
            hanja_content = hanja_content.replace('。', '。 ')
            hanja_content = hanja_content.replace(',', ', ')
            hanja_content = " ".join(hanja_content.split()).strip()

            is_damaged = False
            for w in damage_words:
                if w in korean_content:
                    is_damaged = True
                    break
            
            if is_damaged:
                continue

            korean_content = " ".join(korean_content.split()).strip()
            korean_content = " ".join(re.sub(r'\(\w+\)', '', korean_content).split()).strip()
            korean_content = re.sub(r'\d{3}\)', '', korean_content)
            
            if korean_content != '' and hanja_content != '':
                record_list.append({
                    'date': date, 'title': title, 'korean': korean_content, 'hanja': hanja_content
                })
    
    print(len(record_list))

    print('Seungjeongwon data pre-processing...', end=' ')
    for json_data in s_path:
        # Json file load
        with open(json_data) as json_file:
            record_json = json.load(json_file)
        
        for i in range(len(record_json)):
            # Hanja Preprocessing
            hanja_content = record_json[i]['hanja']
            if '□' in hanja_content:
                continue
            if '◆' in hanja_content:
                continue
            
            hanja_content = hanja_content.replace('○', ' ')
            hanja_content = hanja_content.replace('。', '。 ')
            hanja_content = hanja_content.replace(',', ', ')
            hanja_content = " ".join(hanja_content.split()).strip()
            
            # Korean Preprocessing
            korean_content = record_json[i]['korean']
            if '。' in korean_content:
                continue
            
            is_damaged = False
            for w in damage_words:
                if w in korean_content:
                    is_damaged = True
                    break
            if is_damaged:
                continue
            
            korean_content = korean_content.replace("“", ' "')
            korean_content = korean_content.replace("”", '" ')
            korean_content = " ".join(re.sub(r'\(\w+\)', ' ', korean_content).split()).strip()
            
            if hanja_content != '' and korean_content != '':
                record_list.append({
                    'date': record_json[i]['date'], 'korean': korean_content, 'hanja': hanja_content
                })
    
    print(len(record_list))

    print('Seungjeongwon hanja only data pre-processing...')
    additional_record_list = []
    for json_data in s_hj_path:
        with open(json_data, 'rb') as json_file:
            record_json = json.load(json_file)
        
        for i in range(len(record_json)):
            # Hanja Preprocessing
            hanja_content = record_json[i]['hanja']
            if '□' in hanja_content:
                continue
            if '◆' in hanja_content:
                continue
            
            hanja_content = hanja_content.replace('○', '')
            hanja_content = hanja_content.replace('〔○〕', '')
            hanja_content = hanja_content.strip()
            
            if hanja_content != '':
                additional_record_list.append({
                    'date': record_json[i]['date'], 'hanja': hanja_content
                })
    
    print(len(additional_record_list))

    # Train, test split. Codes in vocab_train.py
    split_record, split_additional_record = train_test_split(record_list, additional_record_list)

    print('Paired data num:')
    print(f"\ttrain: {len(split_record['train'])}")
    print(f"\tvalid: {len(split_record['valid'])}")
    print(f"\ttest: {len(split_record['test'])}")

    print('Additional data num:')
    print(f"\ttrain: {len(split_additional_record['train'])}")
    print(f"\tvalid: {len(split_additional_record['valid'])}")
    print(f"\ttest: {len(split_additional_record['test'])}")

    train_hanja_list = tuple(r['hanja'] for r in split_record['train'])
    train_korean_list = tuple(r['korean'] for r in split_record['train'])
    
    valid_hanja_list = tuple(r['hanja'] for r in split_record['valid'])
    valid_korean_list = tuple(r['korean'] for r in split_record['valid'])
    
    test_hanja_list = tuple(r['hanja'] for r in split_record['test'])
    test_korean_list = tuple(r['korean'] for r in split_record['test'])
    
    train_additional_hanja = tuple(r['hanja'] for r in split_additional_record['train'])
    valid_additional_hanja = tuple(r['hanja'] for r in split_additional_record['valid'])
    test_additional_hanja = tuple(r['hanja'] for r in split_additional_record['test'])
    
    hanja_list = (train_hanja_list, valid_hanja_list, test_hanja_list)
    additional_hanja_list = (train_additional_hanja, valid_additional_hanja, test_additional_hanja)
    
    print('Hanja Vocab Training...')
    (
        hanja_word2id, 
        train_hanja_indices, valid_hanja_indices, test_hanja_indices, 
        train_additional_hanja_indices, valid_additional_hanja_indices, test_additional_hanja_indices
    ) = hanja_vocab_processing(args, hanja_list, additional_hanja_list)
    
    print('Korean Vocab Training...')
    (
        korean_word2id, train_korean_indices, valid_korean_indices, test_korean_indices
    ) = korean_vocab_train(args, train_korean_list, valid_korean_list, test_korean_list)

    with open(f"{args.output_path}/hanja_korean_word2id.pkl", "wb") as f:
        pickle.dump({
            'hanja_word2id': hanja_word2id,
            'korean_word2id': korean_word2id
        }, f ,protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(f"{args.output_path}/preprocessed_train.pkl", "wb") as f:
        pickle.dump({
            'hanja_indices': train_hanja_indices,
            'korean_indices': train_korean_indices,
            'additional_hanja_indices': train_additional_hanja_indices
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{args.output_path}/preprocessed_valid.pkl', 'wb') as f:
        pickle.dump({
            'hanja_indices': valid_hanja_indices,
            'korean_indices': valid_korean_indices,
            'additional_hanja_indices': valid_additional_hanja_indices
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{args.output_path}/preprocessed_test.pkl', 'wb') as f:
        pickle.dump({
            'hanja_indices': test_hanja_indices,
            'korean_indices': test_korean_indices,
            'additional_hanja_indices': test_additional_hanja_indices
        }, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    parser.add_argument('--vocab_size', type=int, default=24000, help='Korean Vocabulary Size')
    parser.add_argument('--joseon_record_path', type=str, help='path of Joseon record',
                        default='/home/nas1_userC/rudvlf0413/joseon_translation/dataset/joseon_record')
    parser.add_argument('--sjw_path', type=str, help='path of Seungjeongwon diary',
                        default='/home/nas1_userC/rudvlf0413/joseon_translation/dataset/sjw_hanja_korean')
    parser.add_argument('--sjw_hj_path', type=str, help='path of Seungjeongwon diary of hanja',
                        default='/home/nas1_userC/rudvlf0413/joseon_translation/dataset/sjw_hanja')
    parser.add_argument('--output_path', type=str, help='path of preprocessed results',
                        default='/home/nas1_userC/rudvlf0413/joseon_translation/dataset/preprocessed')
    parser.add_argument('--pad_id', default=0, type=int, help='pad index')
    parser.add_argument('--bos_id', default=1, type=int, help='index of bos token')
    parser.add_argument('--eos_id', default=2, type=int, help='index of eos token')
    parser.add_argument('--unk_id', default=3, type=int, help='index of unk token')
    args = parser.parse_args()
    
    main(args)
