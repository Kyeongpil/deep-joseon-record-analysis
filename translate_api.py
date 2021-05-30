import argparse
import os
import pickle
from glob import glob
from random import randrange

import ujson as json
from flask import Flask, request

parser = argparse.ArgumentParser("Translate hanja corpus using multiple nodes")
parser.add_argument("--untranslated_corpus_dir", type=str, 
    default="/home/nas1_userC/rudvlf0413/joseon_translation/dataset/sjw_hanja")
parser.add_argument("--save_output_dir", type=str,
    default="/home/nas1_userC/rudvlf0413/joseon_translation/dataset/sjw_hanja_translated")
parser.add_argument("--hanja_korean_vocab_path", type=str, 
    default="/home/nas1_userC/rudvlf0413/joseon_translation/dataset/preprocessed/hanja_korean_word2id.pkl")
args = parser.parse_args()


# App config
app = Flask(__name__, static_url_path='')
app.config.from_object(__name__)

if not os.path.exists(args.save_output_dir):
    os.mkdir(args.save_output_dir)

with open(args.hanja_korean_vocab_path, 'rb') as f:
    data = pickle.load(f)
    src_vocab = data['hanja_word2id']
    del data


@app.route("/getMetaData", methods=['GET'])
def getSrcVocab():
    return json.dumps({'src_vocab': src_vocab})


@app.route("/getData", methods=['GET'])
def getData():
    # get random file
    files = glob(f'{args.untranslated_corpus_dir}/*/*.json')
    rand_file = files[randrange(len(files))]
    try:
        with open(rand_file) as f:
            content = json.load(f)
    except FileNotFoundError:
        files.pop(rand_file)
    
    data = {'file': rand_file, 'content': content}
    print(len(files))
    return json.dumps(data)


@app.route("/commitData", methods=['POST'])
def commitData():
    # save translated data
    data = request.get_json()
    filename = data['file']
    content = data['content']

    filename_ = filename.split('sjw_hanja/')[1]
    translated_filename = f"{args.save_output_dir}/{filename_}"
    translated_filename_dir = os.path.dirname(translated_filename)
    if not os.path.exists(translated_filename_dir):
        os.mkdir(translated_filename_dir)
    
    with open(translated_filename, 'w') as f:
        json.dump(content, f)

    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    files = glob(f'{args.untranslated_corpus_dir}/*/*.json')
    
    if len(files) == 0:
        data = {'progress': 'finish'}
    else:
        total_file_num = len(files)
        print(total_file_num)
        data = {'progress': total_file_num}
    
    return json.dumps(data)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5010, threaded=False, debug=True)
