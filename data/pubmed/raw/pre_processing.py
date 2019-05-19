import numpy as np
import pickle
import networkx as nx
import itertools
from matplotlib import pyplot as plt
from networkx.algorithms import is_bipartite


def load_data():
    extensions = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for extension in extensions:
        with open("ind.pubmed.{}".format(extension), 'rb') as input_file:
            objects.append(pickle.load(input_file, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph_dict = tuple(objects)
    x = np.asarray(x.todense())
    tx = np.asarray(tx.todense())
    allx = np.asarray(allx.todense())

    with open("ind.pubmed.test.index", mode="r") as input_file:
        file_content = input_file.readlines()
        tx_indices = [int(idx_str) for idx_str in file_content]
        tx_idx_set = set(tx_indices)
    allx_idx_set = set(graph_dict.keys()) - tx_idx_set
    allx_indices = sorted(allx_idx_set)

    features = np.zeros((len(tx) + len(allx), allx.shape[1]))
    features[allx_indices] = allx
    features[tx_indices] = tx

    return features, ally, graph_dict


def split_features(features: np.ndarray, labels: np.ndarray):
    pass


def build_graph(features: np.ndarray, graph_dict: dict):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(features)))

    # TODO: check index in graph
    for key in graph_dict.keys():
        # for node_idx in graph_dict[key]:
        #     graph.add_edge(key, node_idx)

        edges = list(itertools.product([key], graph_dict[key]))
        graph.add_edges_from(edges)

    # print("drawing...")
    # nx.draw_networkx(graph)
    # print("saving...")
    # plt.savefig("graph.pdf")
    print("is the graph bipartite? => {}".format(is_bipartite(graph)))
    A = np.asarray(nx.adjacency_matrix(graph).todense())

    return graph


if __name__ == '__main__':
    allFeature, allLabel, graphDict = load_data()

    dataGraph = build_graph(allFeature, graphDict)
