import argparse

import io
import os
import pickle
import math
from random import randint

import numpy as np
import tqdm as tqdm
from collections import defaultdict
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


def prepare_data(path, test=0.1, max_len=None, split=False):
    poses, parents, rels, _ = parse_data(path, max_len)

    poses = add_padding_feature(pad_sequences(poses, maxlen=max_len, padding='post'))
    new_parents = []
    for sentence in parents:
        sentence = pad_sequences(sentence, maxlen=max_len, padding='post')
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
        begin += 50 * math.ceil(sentences_len[sent_counter]/50)
        sent_counter += 1
    return new_predictions


def resolve_roots(predicted_sentence, parents):
    roots_list_id = []
    root = 0
    for id, word in enumerate(predicted_sentence):
        if word[6] == "0":
            roots_list_id.append(id)
    if len(roots_list_id) == 0:
        scores = np.array([word[0] for word in parents])
        id_of_root = np.argmax(scores)
        if "-" in predicted_sentence[id_of_root][0]:
            scores[id_of_root] = 0.0
            id_of_root = np.argmax(scores)
        predicted_sentence[id_of_root][6] = "0"
        predicted_sentence[id_of_root][7] = "root"
        root = predicted_sentence[id_of_root][0]
    elif len(roots_list_id) > 1:
        # scores = []
        # for potential_root in count_roots:
        #     score = parents[potential_root][0]
        #     scores.append(score)
        scores = np.array([parents[potential_root][0] for potential_root in roots_list_id])
        on_roots_list = np.argmax(scores)
        id_of_root = roots_list_id[on_roots_list]
        if "-" in predicted_sentence[id_of_root][0]:
            scores[on_roots_list] = 0.0
            on_roots_list = np.argmax(scores)
            id_of_root = roots_list_id[on_roots_list]
            if "-" in predicted_sentence[id_of_root][0]:
                scores[on_roots_list] = 0.0
                on_roots_list = np.argmax(scores)
                id_of_root = roots_list_id[on_roots_list]
                if "-" in predicted_sentence[id_of_root][0]:
                    scores[on_roots_list] = 0.0
                    on_roots_list = np.argmax(scores)
                    id_of_root = roots_list_id[on_roots_list]
                    if "-" in predicted_sentence[id_of_root][0]:
                        scores[on_roots_list] = 0.0
                        id_of_root = randint(0, len(predicted_sentence))
        try:
            root = predicted_sentence[id_of_root][0]
        except IndexError:
            print(predicted_sentence)
        predicted_sentence[id_of_root][7] = "root"
        if id_of_root in roots_list_id:
            roots_list_id.remove(id_of_root)
        for fake_root_id in roots_list_id:
            predicted_sentence[fake_root_id][6] = str(root)
            predicted_sentence[fake_root_id][7] = "_" if predicted_sentence[fake_root_id][7] in ["_", "root"] else predicted_sentence[fake_root_id][7]
    else:
        if "-" in predicted_sentence[roots_list_id[0]][0]:
            scores = np.array([word[0] for word in parents])
            id_of_root = np.argmax(scores)
            if "-" in predicted_sentence[id_of_root][0]:
                scores[id_of_root] = 0.0
                id_of_root = np.argmax(scores)
            predicted_sentence[id_of_root][6] = "0"
            predicted_sentence[id_of_root][7] = "root"
            root = predicted_sentence[id_of_root][0]
        else:
            id_of_root = roots_list_id[0]
            root = predicted_sentence[id_of_root][0]
    for word in predicted_sentence:
        if word[6] == "_":
            word[6] = str(root)

    # return ["\t".join(word) for word in predicted_sentence]
    return predicted_sentence, id_of_root


def check_cycles(predicted_sentence, root_id):
    graph = dict()
    idxs = dict()
    for id, word in enumerate(predicted_sentence):
        idxs[word[0]] = id
    for word in predicted_sentence:
        if word[6] != "0":
            if idxs[word[0]] in graph.keys():
                krotka = graph[idxs[word[0]]]
                graph[idxs[word[0]]] = krotka + (idxs[word[6]],)
            else:
                graph[idxs[word[0]]] = (idxs[word[6]],)

    if cyclic(graph):
        root_id_conll = predicted_sentence[root_id][0]
        for id, word in enumerate(predicted_sentence):
            if id != root_id:
                predicted_sentence[id][6] = root_id_conll

    return ["\t".join(word) for word in predicted_sentence]

def cyclic(g):
    """Return True if the directed graph g has a cycle.
    g must be represented as a dictionary mapping vertices to
    iterables of neighbouring vertices. For example:

    >>> cyclic({1: (2,), 2: (3,), 3: (1,)})
    True
    >>> cyclic({1: (2,), 2: (3,), 3: (4,)})
    False

    """
    path = set()

    def visit(vertex):
        path.add(vertex)
        for neighbour in g.get(vertex, ()):
            if neighbour in path or visit(neighbour):
                return True
        path.remove(vertex)
        return False

    return any(visit(v) for v in g)


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
            if cats[1] == "Mądzika":
                print ("tutaj")

            cats[6] = str(parent_id) if parent_id <= len(gold_sentence) and parent_id <= int(gold_sentence[-1].split("\t")[0]) else "_"
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
    # parser.add_argument('emb', type=str, help='Embedding file path')
    parser.add_argument('--max_len', help='Maximal no of tokens in sentence', type=int, default=50)
    args = parser.parse_args()

    _, model, poses_test = train(args.input_train, args.input_test, max_len=args.max_len)

    model.save(config.params['model_name'])

    flat_predictions_parents, flat_predictions_rels = predict(model, poses_test)
    test_data = get_conll(args.input_test, max_len=args.max_len)
    write_predicted_output_to_conll(flat_predictions_parents, flat_predictions_rels, test_data, args.max_len, config.params['predict_output_path'])
