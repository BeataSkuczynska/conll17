import argparse

import numpy as np
import tqdm as tqdm
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import config
from constants import RELS_LIST
from data import parse_data, get_conll
from model import create_model
from utils import chunks, save_conll


def add_padding_feature(array3d):
    new_array3d = []
    for idx, array2d in tqdm.tqdm(enumerate(array3d)):
        zeros = np.zeros((len(array2d), 1))
        for idx, array in enumerate(array2d):
            if not array.any():
                zeros[idx] = 1
        new_array2d = np.concatenate((array2d, zeros), axis=1)
        new_array3d.append(new_array2d)
    new_array3d = np.array(new_array3d)
    return new_array3d


def prepare_data(path, test=0.1, max_len=None, split=True):
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
    model.fit(poses_train, [parents_train, rels_train], epochs=1,
              validation_split=0.1,
              verbose=1)

    loss, parents_loss, rels_loss, parents_accuracy, rels_accuracy = model.evaluate(poses_train,
                                                                                    [parents_train, rels_train],
                                                                                    verbose=0)
    print('Parents accuracy train: %f' % (parents_accuracy * 100))
    print('Relations accuracy train: %f' % (rels_accuracy * 100))

    predictions_parents, predictions_rels = model.predict(poses_test, verbose=0)
    flat_predictions_parents = [np.argmax(i) for x in predictions_parents for i in x]
    flat_test_parents = [np.argmax(i) for x in parents_test for i in x]
    confusion_m = confusion_matrix(flat_test_parents, flat_predictions_parents)
    print(confusion_m)

    flat_predictions_rels = [np.argmax(i) for x in predictions_rels for i in x]
    flat_test_rels = [np.argmax(i) for x in rels_test for i in x]
    confusion_m = confusion_matrix(flat_test_rels, flat_predictions_rels)
    print(confusion_m)
    return flat_predictions_parents, flat_predictions_rels


def train(path_train, path_test, config=config.params, max_len=None):
    poses_train, parents_train, rels_train, _ = prepare_data(path_train, test=0.1, max_len=max_len, split=False)
    poses_test, parents_test, rels_test, _ = prepare_data(path_test, test=0.1, max_len=max_len, split=False)
    values = poses_train, poses_test, parents_train, parents_test, rels_train, rels_test, max_len
    flat_predictions_parents, flat_predictions_rels = train_eval(values, config)
    return max_len, flat_predictions_parents, flat_predictions_rels


def write_predicted_output_to_conll(flat_predictions_parents, flat_predictions_rels, test_data):
    predict_parents_grouped = chunks(flat_predictions_parents, max_len)
    predict_rels_grouped = chunks(flat_predictions_rels, max_len)

    predicted_conll = []
    for gold_sentence, parents, rels in zip(test_data, predict_parents_grouped, predict_rels_grouped):
        predicted_sentence = []
        for gold_token, parent, rel in zip(gold_sentence, parents[1:], rels[1:]):
            cats = gold_token.strip().split("\t")
            cats[6] = str(parent)
            cats[7] = RELS_LIST[rel] if int(rel) < len(RELS_LIST) else RELS_LIST[-1]
            predicted_sentence.append("\t".join(cats))
        predicted_conll.append(predicted_sentence)
    save_conll(predicted_conll, 'generated/output.conllu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full cycle  of training and evaluating')
    parser.add_argument('input_train', help='Path to CONLL train file', type=str)
    parser.add_argument('input_test', help='Path to CONLL test file', type=str)
    parser.add_argument('--max_len', help='Maximal no of tokens in sentence', type=int, default=50)
    args = parser.parse_args()

    max_len = 50
    _, flat_predictions_parents, flat_predictions_rels = train(args.input_train, args.input_test, max_len=args.max_len)
    test_data = get_conll(args.input_test)
    write_predicted_output_to_conll(flat_predictions_parents, flat_predictions_rels, test_data)
