import numpy as np
import pickle
import networkx as nx
import itertools
from matplotlib import pyplot as plt
from networkx.algorithms import is_bipartite

import csv


def load_data(filename):
    # TODO: update parsing for tasks
    data_dict = dict()
    idx_key_map = dict()

    with open(filename) as input_file:
        file_reader = csv.reader(input_file, delimiter=',')
        line_count = 0
        for row in file_reader:
            for idx, value in enumerate(row):
                if line_count == 0:
                    idx_key_map[idx] = value
                    data_dict[value] = list()
                else:
                    data_dict[idx_key_map[idx]].append(value)
            line_count += 1
        print(f'Processed {line_count} lines.')


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
    load_data("ratings.csv")
