import numpy as np
import pickle
import networkx as nx
import itertools
from matplotlib import pyplot as plt
from networkx.algorithms import is_bipartite


def load_data():
    extensions = ['x', 'tx', 'allx', 'graph']
    objects = []

    for extension in extensions:
        with open("ind.nell.0.001.{}".format(extension), 'rb') as input_file:
            objects.append(pickle.load(input_file, encoding='latin1'))

    x, tx, allx, graph_dict = tuple(objects)
    labeled_feature = np.asarray(x.todense())
    test_feature = np.asarray(tx.todense())
    all_feature = np.asarray(allx.todense())

    return labeled_feature, test_feature, all_feature, graph_dict


def write_attr(output_dir, all_feature):
    # format: ID - feature
    with open("{}/node_attr".format(output_dir), mode="w"):
        for idx, row in enumerate(all_feature):
            pass


def build_graph(all_feature: np.ndarray, graph_dict: dict):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(all_feature)))

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
    print(is_bipartite(graph))
    A = np.asarray(nx.adjacency_matrix(graph).todense())

    return graph


if __name__ == '__main__':
    labeledFeature, testFeature, allFeature, graphDict = load_data()

    graph = build_graph(allFeature, graphDict)
