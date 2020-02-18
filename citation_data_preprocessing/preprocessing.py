"""
    change the data form x / tx / allx into the same type as
    Graph -- .cites:
        node.id -- node.id

    Attributes -- .content:
        node.id -- features -- label

"""

import sys
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import networkx as nx
import os
import pandas as pd

CITATION_DATA_PATH = '../data/citation_data'
CORA_LABEL = ["Case_Based", "Genetic_Algorithms", "Neural_Networks", "Probabilistic_Methods",
              "Reinforcement_Learning", "Rule_Learning", "Theory"]
CITESEER_LABEL = ["Agents", "AI", "DB", "IR", "ML", "HCI"]


def label_preprocessing(dataset, labels):
    def _data_output(LABEL, _labels):
        hash_table = dict([(i, l) for i, l in enumerate(LABEL)])
        _output = []
        for i in range(len(labels)):
            classes = np.where(labels[i] == 1)
            if len(classes[0]) == 0:
                _output.append(np.random.choice(LABEL))
            else:
                _output.append(hash_table[classes[0][0]])
        return np.array(_output)

    if dataset == 'cora':
        output = _data_output(CORA_LABEL, labels)
    else:
        output = _data_output(CITESEER_LABEL, labels)

    return output


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(CITATION_DATA_PATH + "/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file(CITATION_DATA_PATH + "/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        # set zero if no label
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = np.vstack((np.array(allx.todense()), np.array(tx.todense()))).astype('int')
    labels = np.vstack((ally, ty))

    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = label_preprocessing(dataset, labels)

    content_data = np.column_stack((features, labels))

    if not os.path.exists('./{}'.format(dataset)):
        os.mkdir('./{}'.format(dataset))

    nx.write_edgelist(nx.from_dict_of_lists(graph), './{}/{}.cites'.format(dataset, dataset), data=False,
                      delimiter='\t')
    df = pd.DataFrame(content_data)
    df.to_csv('./{}/{}.content'.format(dataset, dataset), sep='\t', header=False)


if __name__ == '__main__':
    load_data('citeseer')
