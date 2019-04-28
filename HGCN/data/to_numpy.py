#!/usr/bin/env python
# coding=utf-8
import numpy as np
import pickle as pl
import networkx as nx
import scipy.sparse as sp

edgef = open("./edgelist", 'rb')
nodein = open("./node_list")
nodeattr = open("./node_attr")
node_label = open("./node_true")

node_list = []


def load_node_attr(fname, node_idx_name, node_list):
    # Load the node attributes vector. 
    # If there is no attribute vector, ignore it.  
    fin = open(node_idx_name)
    u2i_dict = {}
    for l in fin:
        l = l.strip().split("\t")
        u2i_dict[l[0]] = int(l[1])
    fin = open(fname, 'r')

    def decode_helper(s):
        if s == "":
            return 0
        return float(s)

    converters = {0: lambda s: u2i_dict[s.decode("utf-8")], 1: decode_helper, 4: decode_helper, 5: decode_helper,
                  6: decode_helper,
                  7: decode_helper, 8: decode_helper, 9: decode_helper, 10: decode_helper}

    # This part is different from differe data.
    data = np.loadtxt(fin, delimiter='\t', converters=converters, usecols=(0, 1, 4, 5, 6, 7, 8, 9, 10))

    # normalize per dim
    data[:, 1:] = data[:, 1:] / data[:, 1:].max(axis=0)

    attr_dict = {}
    for d in data:
        attr_dict[d[0]] = d[1:]
    res = []
    for n in node_list:
        res.append(attr_dict[n])
    return data, np.vstack(res)


# Load nodelist
for l in nodein:
    l = l.strip().split('\t')
    node_list.append(int(l[-1]))

# Load feature
data, feature = load_node_attr("./node_attr", "./node_list", node_list)
print(data.shape)
print(feature.shape)
# Load adj
G = nx.read_weighted_edgelist(edgef)
print(len(G))
adj = nx.adjacency_matrix(G)
adj = adj * adj.T
node_list = np.asarray(node_list)
adj = adj[node_list, :]
adj = adj[:, node_list]

print(adj.shape)

adj[adj > 0] = 1


# load label
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


labelfin = open("./node_true")
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
print(train_idx)
print(val_idx)
print(test_idx)
print("%d: %d: %d" % (len(train_idx), len(val_idx), len(test_idx)))
np.savez_compressed("anping.npz", feature=feature, label=labels, train_idx=train_idx, val_idx=val_idx,
                    test_idx=test_idx, node_list=node_list)
sp.save_npz("anping_adj.npz", adj)
