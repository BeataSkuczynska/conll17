import numpy as np

from constants import POS_DICT, RELS_DICT


def get_conll(path):
    with open(path) as f:
        parsed_all = []
        parsed_sent = []
        for line in f.readlines():
            if line.startswith("#"):
                pass
            elif len(line) > 2:
                parsed_sent.append(line)
            else:
                parsed_all.append(parsed_sent)
                parsed_sent = []
    return parsed_all


def parse_data(path, max_len=None):
    sentences = get_conll(path)

    max_len_count = 0
    poses, parents, rels = [], [], []

    for sentence in sentences:
        sentence_length = len(sentence) + 1
        poses_s, parents_s, rels_s = [], [], []
        max_len_count = max(max_len_count, sentence_length)
        zeros_vector_parents = np.zeros(sentence_length + 1)
        zeros_vector_parents[-1] = 1
        parents_s.append(zeros_vector_parents)
        for word in sentence:
            zeros_vector_pos = np.zeros(len(POS_DICT))
            zeros_vector_rels = np.zeros(len(RELS_DICT))
            zeros_vector_parents = np.zeros(sentence_length)
            word_splitted = word.split("\t")
            pos = word_splitted[3]
            parent = word_splitted[6]
            parent = parent if parent != "_" else 0
            rel = word_splitted[7]
            zeros_vector_pos[POS_DICT[pos]] = 1
            poses_s.append(zeros_vector_pos)
            rel = rel.split(":")[0] if ":" in rel else rel
            zeros_vector_rels[RELS_DICT[rel]] = 1
            rels_s.append(zeros_vector_rels)

            zeros_vector_parents[int(parent)] = 1

            parents_s.append(zeros_vector_parents)
        poses.append(poses_s)
        parents.append(parents_s)
        rels.append(rels_s)
    if max_len is None:
        max_len = max_len_count

    return poses, parents, rels, max_len
