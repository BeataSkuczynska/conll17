import json

import numpy as np

from constants import POS_DICT, RELS_DICT
from utils import chunks


def get_conll(path, max_len=None):
    with open(path) as f:
        parsed_all = []
        parsed_sent = []
        for line in f.readlines():
            if line.startswith("#"):
                pass
            elif len(line) > 2:
                parsed_sent.append(line)
                if max_len and len(parsed_sent) == max_len:
                    parsed_all.append(parsed_sent)
                    parsed_sent = []
            else:
                if parsed_sent:
                    parsed_all.append(parsed_sent)
                parsed_sent = []
    return parsed_all


def parse_data(path, max_len=None):
    sentences = get_conll(path)

    max_len_count = 0
    count_words = 0
    word2idx = dict()
    poses, parents, rels, orths = [], [], [], []

    for sentence in sentences:
        sentence_length = len(sentence) + 1

        max_len_count = max(max_len_count, sentence_length) if max_len is None else 0
        for sentence_cut in chunks(sentence, max_len):
            poses_s, parents_s, rels_s, orths_s = [], [], [], []
            for word in sentence_cut:
                zeros_vector_pos = np.zeros(len(POS_DICT))
                zeros_vector_rels = np.zeros(len(RELS_DICT))
                zeros_vector_parents = np.zeros(sentence_length)

                word_splitted = word.split("\t")
                pos = word_splitted[3]
                parent = word_splitted[6]
                parent = parent if parent != "_" and int(parent) < max_len else 0
                rel = word_splitted[7]
                orth = word_splitted[1]

                zeros_vector_pos[POS_DICT[pos]] = 1
                poses_s.append(zeros_vector_pos)

                zeros_vector_rels[RELS_DICT[rel]] = 1
                rels_s.append(zeros_vector_rels)

                zeros_vector_parents[int(parent)] = 1
                parents_s.append(zeros_vector_parents)

                if orth not in word2idx:
                    word2idx[orth] = count_words
                    count_words +=1
                orths_s.append(word2idx[orth])

            poses.append(poses_s)
            parents.append(parents_s)
            rels.append(rels_s)
            orths.append(orths_s)

    with open("generated/vocab.json", "w+") as f:
        json.dump(word2idx, f)

    if max_len is None:
        max_len = max_len_count

    return poses, parents, rels, orths, word2idx, max_len
