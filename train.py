import argparse

import io
import os
import pickle
import numpy as np
import tqdm as tqdm
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import config
from constants import RELS_LIST
from data import parse_data, get_conll
from model import create_model
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


def prepare_data(path, emb, test=0.1, max_len=None, split=False, vocab=False):
    poses, parents, rels, orths, word2idx, _ = parse_data(path, max_len, vocab)

    if not vocab:
        idx2vec = create_embeddings(emb, word2idx)
        with open("generated/embeddings.pkl", "wb") as f:
            pickle.dump(idx2vec, f)
    else:
        with open("generated/embeddings.pkl", "rb") as f:
            idx2vec = pickle.load(f)

    poses = add_padding_feature(pad_sequences(poses, maxlen=max_len, padding='post'))
    new_parents = []
    for sentence in parents:
        sentence = pad_sequences(sentence, maxlen=max_len, padding='post')
        new_parents.append(sentence)

    parents = add_padding_feature(pad_sequences(new_parents, maxlen=max_len, padding='post'))
    rels = add_padding_feature(pad_sequences(rels, maxlen=max_len, padding='post'))
    orths_padded = pad_sequences(orths, maxlen=max_len, padding='post')

    orths_padded = get_embs(orths_padded, idx2vec, max_len)

    if split:
        poses_train, poses_test, parents_train, parents_test, rels_train, rels_test = train_test_split(
            poses,
            parents,
            rels,
            test_size=test,
            shuffle=False)

        return poses_train, poses_test, parents_train, parents_test, rels_train, rels_test, max_len
    else:
        return poses, parents, rels, orths_padded, max_len, idx2vec


def train_eval(values, config=config.params):
    poses_train, poses_test, parents_train, parents_test, rels_train, rels_test, orths_train, orths_test, idx2vec, max_len = values
    model = create_model(idx2vec, maxlen=max_len, params=config)
    model.fit([orths_train, poses_train], [parents_train, rels_train], epochs=config['epochs'],
              validation_split=0.1,
              verbose=2)

    loss, parents_loss, rels_loss, parents_accuracy, rels_accuracy = model.evaluate([orths_test, poses_test],
                                                                                    [parents_test, rels_test],
                                                                                    verbose=0)
    print('Parents accuracy train: %f' % (parents_accuracy * 100))
    print('Relations accuracy train: %f' % (rels_accuracy * 100))

    return model


def train(path_train, path_test, emb, config=config.params, max_len=None):
    poses_train, parents_train, rels_train, orths_train, _, idx2vec = prepare_data(path_train, emb, max_len=max_len)
    poses_test, parents_test, rels_test, orths_test, _, _ = prepare_data(path_test, emb, max_len=max_len, vocab=True)
    values = poses_train, poses_test, parents_train, parents_test, rels_train, rels_test, orths_train, orths_test, idx2vec, max_len
    model = train_eval(values, config)
    return max_len, model, poses_test, orths_test


def merge_long_sentences(predicted_conll, max_len):
    new_predicted_conll = []
    for sentence in predicted_conll:
        first_token_id = sentence[0].split("\t")[0]
        if len(first_token_id) > 1 and not "-" in first_token_id:
            new_predicted_conll[-1].extend(sentence)
        else:
            new_predicted_conll.append(sentence)
    return new_predicted_conll


def write_predicted_output_to_conll(flat_predictions_parents, flat_predictions_rels, test_data, max_len, output_path):
    predict_parents_grouped = chunks(flat_predictions_parents, max_len)
    predict_rels_grouped = chunks(flat_predictions_rels, max_len)

    predicted_conll = []
    for gold_sentence, parents, rels in zip(test_data, predict_parents_grouped, predict_rels_grouped):
        predicted_sentence = []
        idx = 1
        for gold_token, parent, rel in zip(gold_sentence, parents, rels):
            cats = gold_token.strip().split("\t")
            cats[0] = str(idx) if idx > max_len else cats[0]
            cats[6] = str(parent)
            cats[7] = RELS_LIST[rel] if int(rel) < len(RELS_LIST) else RELS_LIST[-1]
            predicted_sentence.append("\t".join(cats))
            idx += 1
        predicted_conll.append(predicted_sentence)
    predicted_conll = merge_long_sentences(predicted_conll, max_len)
    save_conll(predicted_conll, output_path)


def predict(model, poses_test, orths_test):
    print ("Sentences in test: " + str(len(poses_test)))
    predictions_parents, predictions_rels = model.predict([orths_test, poses_test], verbose=0)
    flat_predictions_parents = [np.argmax(i) for x in predictions_parents for i in x]

    flat_predictions_rels = [np.argmax(i) for x in predictions_rels for i in x]

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
    parser.add_argument('emb', type=str, help='Embedding file path')
    parser.add_argument('--max_len', help='Maximal no of tokens in sentence', type=int, default=50)
    args = parser.parse_args()

    emb = load_embeddings(args.emb)

    _, model, poses_test, orths_test = train(args.input_train, args.input_test, emb, max_len=args.max_len)

    model.save(config.params['model_name'])

    flat_predictions_parents, flat_predictions_rels = predict(model, poses_test, orths_test)
    test_data = get_conll(args.input_test, max_len=args.max_len)
    write_predicted_output_to_conll(flat_predictions_parents, flat_predictions_rels, test_data, args.max_len, config.params['predict_output_path'])
