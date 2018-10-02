import argparse
import io
import math
import os

import numpy as np
import tqdm as tqdm
from gensim.models import Word2Vec
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import config
from constants import RELS_LIST
from data import parse_data, get_conll
from model import create_model
from postprocessing import resolve_roots, check_cycles
from utils import chunks, save_conll


def load_embeddings(embeddings_path):
    if os.path.isfile(embeddings_path + '.model'):
        model = Word2Vec.load(embeddings_path + ".model")
    return model


def add_padding_feature(array3d):
    new_array3d = []
    for idx, array2d in tqdm.tqdm(enumerate(array3d), total=len(array3d)):
        zeros = np.zeros((len(array2d), 1))
        for idx, array in enumerate(array2d):
            if not array.any():
                zeros[idx] = 1
        new_array2d = np.concatenate((array2d, zeros), axis=1)
        new_array3d.append(new_array2d)
    new_array3d = np.array(new_array3d)
    return new_array3d


def create_embeddings(emb, word2idx):
    unk_count = 0
    vocab_size = len(word2idx)
    index2vec = np.zeros((vocab_size + 1, emb.vector_size), dtype="float32")
    index2vec[0] = np.zeros(emb.vector_size)
    for word in word2idx:
        index = word2idx[word]
        try:
            index2vec[index] = emb[word]
        except KeyError:
            index2vec[index] = np.random.rand(emb.vector_size)
            unk_count += 1

    # print("emb vocab size: ", len(emb.vocabulary))
    print("unknown words count: ", unk_count)
    print("index2vec size: ", len(index2vec))
    print("words  ", len(word2idx))
    return index2vec


def get_embs(orths_padded, idx2vec, max_len):
    array_3d = []
    for id_s, orths_pad in enumerate(orths_padded):
        array_2d = np.zeros((max_len, emb.vector_size))
        for id_w, orth in enumerate(orths_pad):
            array_2d[id_w] = idx2vec[orth]
        array_3d.append(array_2d)
    orths_arr = np.array(array_3d)
    orths_padded = orths_arr
    return orths_padded


def prepare_data(path, test=0.1, max_len=None, split=False):
    poses, parents, rels, _ = parse_data(path, max_len)

    poses = add_padding_feature(pad_sequences(poses, maxlen=max_len, padding='post'))
    new_parents = []
    for sentence in parents:
        sentence = pad_sequences(sentence, maxlen=max_len + 1, padding='post')
        new_parents.append(sentence)

    parents = add_padding_feature(pad_sequences(new_parents, maxlen=max_len, padding='post'))
    rels = add_padding_feature(pad_sequences(rels, maxlen=max_len, padding='post'))

    if split:
        poses_train, poses_test, parents_train, parents_test, rels_train, rels_test = train_test_split(
            poses,
            parents,
            rels,
            test_size=test,
            shuffle=False)

        return poses_train, poses_test, parents_train, parents_test, rels_train, rels_test, max_len
    else:
        return poses, parents, rels, max_len


def train_eval(values, config=config.params):
    poses_train, poses_test, parents_train, parents_test, rels_train, rels_test, max_len = values
    model = create_model(maxlen=max_len, params=config)
    model.fit(poses_train, [parents_train, rels_train], epochs=config['epochs'],
              validation_split=0.1,
              verbose=2)

    loss, parents_loss, rels_loss, parents_accuracy, rels_accuracy = model.evaluate(poses_test,
                                                                                    [parents_test, rels_test],
                                                                                    verbose=0)
    print('Parents accuracy train: %f' % (parents_accuracy * 100))
    print('Relations accuracy train: %f' % (rels_accuracy * 100))

    return model


def train(path_train, path_test, config=config.params, max_len=None):
    poses_train, parents_train, rels_train, _ = prepare_data(path_train, max_len=max_len)
    poses_test, parents_test, rels_test, _ = prepare_data(path_test, max_len=max_len)
    values = poses_train, poses_test, parents_train, parents_test, rels_train, rels_test, max_len
    model = train_eval(values, config)
    return max_len, model, poses_test


def merge_long_sentences(predicted_conll):
    new_predicted_conll = []
    for sentence in predicted_conll:
        first_token_id = sentence[0].split("\t")[0]
        if len(first_token_id) > 1 and not "-" in first_token_id:
            new_predicted_conll[-1].extend(sentence)
        else:
            new_predicted_conll.append(sentence)
    sentences_len = [len(sentence) for sentence in new_predicted_conll]
    return new_predicted_conll, sentences_len


def unpad(flat_predictions, sentences_len):
    new_predictions = []
    sent_counter = 0
    begin = 0
    while len(new_predictions) < len(sentences_len):
        new_predictions.append(flat_predictions[begin: begin + sentences_len[sent_counter]])
        begin += 50 * math.ceil(sentences_len[sent_counter] / 50)
        sent_counter += 1
    return new_predictions


def write_predicted_output_to_conll(flat_predictions_parents, flat_predictions_rels, test_data, max_len, output_path):
    predict_parents_grouped = chunks(flat_predictions_parents, max_len)
    predict_rels_grouped = chunks(flat_predictions_rels, max_len)

    test_data_grouped, sentences_len = merge_long_sentences(test_data)

    predict_parents_grouped = unpad(flat_predictions_parents, sentences_len)
    predict_rels_grouped = unpad(flat_predictions_rels, sentences_len)

    predicted_conll = []
    for gold_sentence, parents, rels in zip(test_data_grouped, predict_parents_grouped, predict_rels_grouped):
        predicted_sentence = []
        idx = 1
        for gold_token, parent, rel in zip(gold_sentence, parents, rels):
            parent_id = np.argmax(parent)
            rel_id = np.argmax(rel)
            cats = gold_token.strip().split("\t")

            cats[6] = str(parent_id) if parent_id <= len(gold_sentence) and parent_id <= int(
                gold_sentence[-1].split("\t")[0]) else "_"
            cats[7] = RELS_LIST[rel_id] if rel_id < len(RELS_LIST) else RELS_LIST[-1]
            if "-" not in cats[0]:
                if idx > max_len:
                    cats[0] = str(idx)
                idx += 1
            else:
                pass
            # predicted_sentence.append("\t".join(cats))
            predicted_sentence.append(cats)
        predicted_sentence, root_id_list = resolve_roots(predicted_sentence, parents)
        predicted_sentence = check_cycles(predicted_sentence, root_id_list)
        predicted_conll.append(predicted_sentence)
    # predicted_conll, _ = merge_long_sentences(predicted_conll)
    save_conll(predicted_conll, output_path)


def predict(model, poses_test):
    print("Sentences in test: " + str(len(poses_test)))
    predictions_parents, predictions_rels = model.predict(poses_test, verbose=0)
    flat_predictions_parents = [i for x in predictions_parents for i in x]

    flat_predictions_rels = [i for x in predictions_rels for i in x]

    return flat_predictions_parents, flat_predictions_rels


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    fin = open(fname, "r", encoding='utf-8', newline='\n', errors='ignore').readlines()
    # n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm.tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full cycle of training and evaluating')
    parser.add_argument('input_train', help='Path to CONLL train file', type=str)
    parser.add_argument('input_test', help='Path to CONLL test file', type=str)
    parser.add_argument('--max_len', help='Maximal no of tokens in sentence', type=int, default=50)
    parser.add_argument('-p', '--predict', help='Set this flag to perform only prediction without training.',
                        action='store_true', default=False)
    args = parser.parse_args()

    if args.predict:
        model = load_model(config.params['model_name'])
        poses_test, _, _, _ = prepare_data(args.input_test, max_len=args.max_len)
    else:
        _, model, poses_test = train(args.input_train, args.input_test, max_len=args.max_len)

    flat_predictions_parents, flat_predictions_rels = predict(model, poses_test)
    test_data = get_conll(args.input_test, max_len=args.max_len)
    write_predicted_output_to_conll(flat_predictions_parents, flat_predictions_rels, test_data, args.max_len,
                                    config.params['predict_output_path'])
