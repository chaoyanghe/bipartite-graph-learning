#!/usr/bin/env python
# coding=utf-8
import networkx as nx
import numpy as np
import scipy.sparse as sp

EDGE_LIST_PATH = "./Tencent-QQ/edgelist"
NODE_LIST_PATH = "./Tencent-QQ/node_list"
NODE_ATTR_PATH = "./Tencent-QQ/node_attr"
NODE_LABEL_PATH = "./Tencent-QQ/node_true"

edgef = open(EDGE_LIST_PATH, 'rb')
nodein = open(NODE_LIST_PATH)
nodeattr = open(NODE_ATTR_PATH)
node_label = open(NODE_LABEL_PATH)

node_list = []


def load_node_attr(fname, node_idx_name, node_list):
    # Load the node attributes vector. 
    # If there is no attribute vector, ignore it.

    # node_list: id
    fin = open(node_idx_name)
    u2i_dict = {}
    for l in fin:
        l = l.strip().split("\t")
        u2i_dict[l[0]] = int(l[0])
    fin = open(fname, 'r')

    def decode_helper(s):
        if s == "":
            return 0
        return float(s)

    converters = {0: lambda s: u2i_dict[s.decode("utf-8")], 1: decode_helper, 4: decode_helper, 5: decode_helper,
                  6: decode_helper,
                  7: decode_helper, 8: decode_helper, 9: decode_helper, 10: decode_helper}

    # This part is different from differe data.
    # data: [[id, feature1, ..., feature10], ..., [id, feature1, ..., feature10]]
    data = np.loadtxt(fin, delimiter='\t', converters=converters, usecols=(0, 1, 4, 5, 6, 7, 8, 9, 10))

    # normalize per dim
    data[:, 1:] = data[:, 1:] / data[:, 1:].max(axis=0)

    # attr_dict: {id : [feature1, ..., feature10]}
    attr_dict = {}
    for d in data:
        attr_dict[d[0]] = d[1:]

    # res: [[feature1, ..., feature10], [feature1, ..., feature10], ...]
    res = []
    for n in node_list:
        res.append(attr_dict[n])
    return data, np.vstack(res)


# Load nodelist
for l in nodein:
    l = l.strip().split('\t')
    node_list.append(int(l[-1]))

# Load feature. len(node_attr) > len(node_list)
data, feature = load_node_attr(NODE_ATTR_PATH, NODE_LIST_PATH, node_list)
logging.info("data = %s" % data[0])
logging.info("features = %s" % feature[0])
logging.info(data.shape)
logging.info(feature.shape)

# Load adj
G = nx.read_weighted_edgelist(edgef)
logging.info(len(G))
adj = nx.adjacency_matrix(G)
logging.info(adj.shape)

adj = adj * adj.T
node_list = np.asarray(node_list)
adj = adj[node_list, :]
adj = adj[:, node_list]

logging.info(adj.shape)

adj[adj > 0] = 1


# load label
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


labelfin = open(NODE_LABEL_PATH)
true_set = set([int(x.strip()) for x in labelfin])
data_y = []
for n in node_list:
    if n in true_set:
        data_y.append(1)
    else:
        data_y.append(0)

# labels = encode_onehot(data_y)
labels = np.asarray(data_y)
nnz = len(node_list)
perm = np.random.permutation(nnz)
train_idx = perm[0:int(0.8 * nnz)]
val_idx = perm[int(0.8 * nnz):int(0.9 * nnz)]
test_idx = perm[int(0.9 * nnz):]
logging.info(train_idx)
logging.info(val_idx)
logging.info(test_idx)
logging.info("%d: %d: %d" % (len(train_idx), len(val_idx), len(test_idx)))
np.savez_compressed("anping.npz", feature=feature, label=labels, train_idx=train_idx, val_idx=val_idx,
                    test_idx=test_idx, node_list=node_list)
sp.save_npz("anping_adj.npz", adj)

EDGE_LIST_TEST_PATH = "./Tencent-QQ/edgelist_test"
edgef_test = open(EDGE_LIST_TEST_PATH, 'rb')
adjU = [[1, 1],
        [1, 0],
        [1, 0]]
adjV = [[1, 1, 1],
        [1, 0, 0]]
featuresU = np.random.rand(12).reshape(3, 4)
featuresV = np.random.rand(10).reshape(2, 5)

G_test = nx.read_weighted_edgelist(edgef_test)

adj = nx.adjacency_matrix(G_test)
# logging.info(adj[[0,1], :])

logging.info(adj[:, [0, 1]])
logging.info(adj.shape)