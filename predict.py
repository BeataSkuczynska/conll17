import argparse

import numpy as np
from keras.models import load_model

from data import get_conll
from train import prepare_data, write_predicted_output_to_conll
from utils import chunks

MAX_LEN = 50


def postprocess(flat_parents, flat_rels, parents, rels, test_data):
    flat_parents_grouped = chunks(flat_parents, MAX_LEN)
    flat_rels_grouped = chunks(flat_rels, MAX_LEN)

    iterator = 0
    for sentence_flat_parents, sentence_flat_rels, sentence in zip(flat_parents_grouped, flat_parents_grouped, test_data):
        relevant_sentence_flat_parents = sentence_flat_parents[:len(sentence)]
        if 0 not in relevant_sentence_flat_parents:
            parents_not_argmax = parents[iterator]
            rels_not_argmax = rels[iterator]
            id_of_potential_root = np.argmax(parents_not_argmax[:len(sentence), 0])
            print(id_of_potential_root)
            flat_parents[iterator][id_of_potential_root]    # TO DO

        iterator += 1

    return flat_parents, flat_rels


def predict(model, poses_test, test_data):
    predictions_parents, predictions_rels = model.predict(poses_test, verbose=0)
    flat_predictions_parents = [np.argmax(i) for x in predictions_parents for i in x]

    flat_predictions_rels = [np.argmax(i) for x in predictions_rels for i in x]

    return flat_predictions_parents, flat_predictions_rels
    # return postprocess(flat_predictions_parents, flat_predictions_rels, predictions_parents, predictions_rels, test_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict parents & relations.')
    # parser.add_argument('model_path', help='Path to file with saved model', type=str)
    parser.add_argument('path_test', help='Path to CONLL test file', type=str)
    args = parser.parse_args()

    model = load_model('generated/30_polevalTrainData_newRels_model.h5')

    poses_test, parents_test, rels_test, _ = prepare_data(args.path_test, max_len=MAX_LEN)
    test_data = get_conll(args.path_test, max_len=MAX_LEN)
    flat_predictions_parents, flat_predictions_rels = predict(model, poses_test, test_data)

    write_predicted_output_to_conll(flat_predictions_parents, flat_predictions_rels, test_data, MAX_LEN, 'generated/output_predict_script_2.conllu')
