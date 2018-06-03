import argparse
import os

import numpy as np

from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from data import parse_data
from model import create_model
import config


def prepare_data(path, test=0.1, max_len=None):
    poses, parents, labels, max_len = parse_data(path)
    poses = pad_sequences(poses, maxlen=max_len)
    new_parents = []
    for sentence in parents:
        sentence = pad_sequences(sentence, maxlen=max_len)
        new_parents.append(sentence)

    parents = pad_sequences(new_parents, maxlen=max_len)
    labels = pad_sequences(labels, maxlen=max_len)
    poses_train, poses_test, parents_train, parents_test, labels_train, labels_test = train_test_split(
        poses,
        parents,
        labels,
        test_size=test,
        shuffle=False)

    return poses_train, poses_test, parents_train, parents_test, labels_train, labels_test, max_len


def train_eval(values, config=config.params):
    poses_train, poses_test, parents_train, parents_test, labels_train, labels_test, max_len = values
    model = create_model(maxlen=max_len, params=config)
    model.fit(poses_train, [parents_train, labels_train], epochs=10,
              validation_split=0.1,
              verbose=1)

    loss, parents_loss, rels_loss, parents_accuracy, rels_accuracy = model.evaluate(poses_train, [parents_train, labels_train], verbose=0)
    print('Parents accuracy train: %f' % (parents_accuracy * 100))
    print('Relations accuracy train: %f' % (rels_accuracy * 100))

    predictions_parents, predictions_labels = model.predict(poses_test, verbose=0)
    # unpad = []
    # for idx, val in enumerate(targets_test):
    #     unpad.append(np.argmax(predictions[idx], axis=1).tolist())
    flat_predictions_parents = [np.argmax(i) for x in predictions_parents for i in x]
    flat_test_parents = [np.argmax(i) for x in parents_test for i in x]
    confusion_m = confusion_matrix(flat_test_parents, flat_predictions_parents)
    print(confusion_m)

    flat_predictions_labels = [np.argmax(i) for x in predictions_labels for i in x]
    flat_test_labels = [np.argmax(i) for x in labels_test for i in x]
    confusion_m = confusion_matrix(flat_test_labels, flat_predictions_labels)
    print(confusion_m)


def train(path, config=config.params, max_len=None):
    values = prepare_data(path, test=0.1, max_len=max_len)
    return train_eval(values, config)


if __name__ == '__main__':
    train("resources/ud-treebanks-v2.1/UD_Polish/pl-ud-dev.conllu")
