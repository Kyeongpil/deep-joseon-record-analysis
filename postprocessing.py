import argparse
import pickle


def text_processing(text):
    text = text.replace(' 이 ', '이 ')
    text = text.replace(' 이라고 ', '이라고 ')
    text = text.replace(' 를 ', '를 ')
    text = text.replace(' 을 ', '을 ')
    text = text.replace(' 들을 ', '들을 ')
    text = text.replace(' 들의 ', '들의 ')
    text = text.replace(' 의 ', '의 ')
    text = text.replace(' 에게 ', '에게')
    text = text.replace(' 은 ', '은 ')
    text = text.replace(' 는 ', '는 ')
    text = text.replace(' 가 ', '가 ')
    text = text.replace(' 에 ', '에 ')
    text = text.replace(' 와 ', '와 ')
    text = text.replace(' 과 ', '과 ')
    text = text.replace(' 에는 ', '에는 ')
    text = text.replace(' 하는 ', '하는 ')
    text = text.replace(' 으로 ', '으로 ')
    text = text.replace(' 하여 ', '하여 ')
    text = text.replace(' 하소서', '하소서')
    text = text.replace(' 하라고 ', '하라고 ')
    text = text.replace(',', ', ')
    text = text.replace('.', '. ')
    text = text.replace(". '", ".'")
    text = text.replace(' , ', ', ')
    text = " ".join(text.split())
    return text


def main(args):
    model_config_str = f"{args.beam_size}_{args.beam_alpha}_{args.repetition_penalty}"
    with open(f'./results_beam_{model_config_str}.pkl', 'rb') as f:
        data = pickle.load(f)
        prediction_decode = data['prediction_decode']
        label_decode = data['label_decode']

    data['prediction_decode_post'] = [text_processing(x) for x in prediction_decode]
    data['label_decode_post'] = [text_processing(x) for x in label_decode]

    with open(f'./results_beam_{model_config_str}.pkl', 'wb') as f:
        pickle.dump(data, f)

    # For NLG-Eval
    with open(f'./prediction_text_{model_config_str}.txt', 'w') as f:
        for pred in data['prediction_decode_post']:
            f.write(pred + '\n')
    
    with open(f'./label_text_{model_config_str}.txt', 'w') as f:
        for label in data['label_decode_post']:
            f.write(label + '\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Post-processing')
    parser.add_argument('--beam_size', default=3, type=int, help='beam size')
    parser.add_argument('--beam_alpha', default=0.7, type=float, help='beam alpha')
    parser.add_argument('--repetition_penalty', default=1.3, type=float, help='repetition penalty')
    args = parser.parse_args()
    main(args)
