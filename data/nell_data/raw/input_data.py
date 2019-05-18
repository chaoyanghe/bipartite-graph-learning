import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

from data.bipartite_graph_data_loader_tencent import BipartiteGraphDataLoaderTencent


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def __BipartiteToSingle(graph):
    """
    transfer the bipartite graph to single graph
    :param graph: sparse matrix
    :return: sparse adjacent matrix
    """
    single_graph = np.dot(graph.toarray(), graph.toarray().T)
    indice = np.where(single_graph != 0)
    val = [1 for i in range(len(indice[0]))]
    return sp.csr_matrix((val, indice))


def load_data_for_tencent(flags, device):
    NODE_LIST_PATH = "../data/Tencent-QQ/node_list"
    NODE_ATTR_PATH = "../data/Tencent-QQ/node_attr"
    NODE_LABEL_PATH = "../data/Tencent-QQ/node_true"
    EDGE_LIST_PATH = "../data/Tencent-QQ/edgelist"
    GROUP_LIST_PATH = "../data/Tencent-QQ/group_list"
    GROUP_ATTR_PATH = "../data/Tencent-QQ/group_attr"

    batch_size = 512
    device = 'cpu'
    # batch_size = flags.batch_size
    bipartite_graph_data_loader = BipartiteGraphDataLoaderTencent(batch_size, NODE_LIST_PATH, NODE_ATTR_PATH,
																  NODE_LABEL_PATH,
																  EDGE_LIST_PATH,
																  GROUP_LIST_PATH, GROUP_ATTR_PATH, device=device)
    bipartite_graph_data_loader.load()
    u_attr = bipartite_graph_data_loader.get_u_attr_array()
    u_adj = bipartite_graph_data_loader.get_u_adj()
    print(type(u_attr))
    print(type(u_adj))

if __name__ == "__main__":
    load_data_for_tencent(12, 21)
