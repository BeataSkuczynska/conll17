from random import randint

import numpy as np


def resolve_roots(predicted_sentence, parents):
    roots_list_id = []
    root = 0
    for id, word in enumerate(predicted_sentence):
        if word[6] == "0":
            roots_list_id.append(id)
    if len(roots_list_id) == 0:
        scores = np.array([word[0] for word in parents])
        id_of_root = np.argmax(scores)
        if id_of_root > len(predicted_sentence) or "-" in predicted_sentence[id_of_root][0]:
            scores[id_of_root] = 0.0
            id_of_root = np.argmax(scores)
        predicted_sentence[id_of_root][6] = "0"
        predicted_sentence[id_of_root][7] = "root"
        root = predicted_sentence[id_of_root][0]
    elif len(roots_list_id) > 1:
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
            predicted_sentence[fake_root_id][7] = "_" if predicted_sentence[fake_root_id][7] in ["_", "root"] else \
                predicted_sentence[fake_root_id][7]
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
