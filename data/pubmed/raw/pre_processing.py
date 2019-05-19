import numpy as np
import pickle
import networkx as nx


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
        test_indices = [int(idx_str) for idx_str in file_content]
    train_idx_set = set(graph_dict.keys()) - set(test_indices)
    train_indices = sorted(train_idx_set)

    features = np.zeros((len(tx) + len(allx), allx.shape[1]))
    features[train_indices] = allx
    features[test_indices] = tx

    labels_one_hot = np.zeros((len(ty) + len(ally), ally.shape[1]))
    labels_one_hot[train_indices] = ally
    labels_one_hot[test_indices] = ty

    return features, labels_one_hot, graph_dict


def split_features(features: np.ndarray, labels_one_hot: np.ndarray):
    feat_idx_to_label = np.argmax(labels_one_hot, axis=1)

    # two sets of vertices
    u_idx_set = set()
    v_idx_set = set()

    feat_idx_to_class_size = np.sum(labels_one_hot, axis=0)
    class_element_count = np.zeros(labels_one_hot.shape[1], dtype=int)

    for feat_idx in range(len(features)):
        label = feat_idx_to_label[feat_idx]

        # split each class by half
        if class_element_count[label] <= feat_idx_to_class_size[label] // 2:
            u_idx_set.add(feat_idx)
        else:
            v_idx_set.add(feat_idx)

        class_element_count[label] += 1

    return u_idx_set, v_idx_set, feat_idx_to_label


def build_graph(features: np.ndarray, u_idx_set: set, v_idx_set: set, feat_idx_to_label, graph_dict: dict):
    edges = list()
    for feat_idx in graph_dict.keys():
        vertex0 = feat_idx
        for vertex1 in graph_dict.get(feat_idx, []):
            if (vertex0 in u_idx_set) and (vertex1 in v_idx_set):
                edges.append((vertex0, vertex1))
            elif (vertex0 in v_idx_set) and (vertex1 in u_idx_set):
                edges.append((vertex1, vertex0))

    graph = nx.Graph()
    graph.add_nodes_from(u_idx_set, bipartite=0)
    graph.add_nodes_from(v_idx_set, bipartite=1)
    graph.add_edges_from(edges)
    assert nx.is_bipartite(graph), "Error: graph should be bipartite!"

    # remove disconnected components
    print("before: total of {} vertices".format(graph.number_of_nodes()))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    print("after: total of {} vertices".format(graph.number_of_nodes()))

    # A = np.asarray(nx.adjacency_matrix(graph).todense())
    # print(A.shape)

    return graph


def save_vertices_and_attr(features: np.ndarray, u_idx_set: set, v_idx_set: set):
    vertex_set_names = ["node", "group"]
    vertex_sets = [u_idx_set, v_idx_set]

    for i in range(len(vertex_sets)):
        attr_output_str = ""
        vertices_output_str = ""

        for vertex_idx in vertex_sets[i]:
            attr_str = '\t'.join(map(str, features[vertex_idx]))
            attr_output_str += "{}\t{}\n".format(vertex_idx, attr_str)

            vertices_output_str += "{}\n".format(vertex_idx)

        print("saving {}_attr...".format(vertex_set_names[i]))
        with open("../{}_attr".format(vertex_set_names[i]), "w") as output_file:
            output_file.write(attr_output_str)

        print("saving {}_list...".format(vertex_set_names[i]))
        with open("../{}_list".format(vertex_set_names[i]), "w") as output_file:
            output_file.write(vertices_output_str)


def save_edges(graph: nx.Graph):
    print("saving edgelist...")
    with open("../edgelist", "w") as output_file:
        for edge in graph.edges:
            output_file.write("{}\t{}\n".format(*edge))


def save_labels(labels_one_hot: np.ndarray, u_idx_set: set):
    labels = np.argmax(labels_one_hot, axis=1)
    print("saving node_true...")
    with open("../node_true", "w") as output_file:
        for u_idx in u_idx_set:
            output_file.write("{}\t{}\n".format(u_idx, labels[u_idx]))


if __name__ == '__main__':
    allFeature, allLabelOneHot, graphDict = load_data()

    uIdxSet, vIdxSet, featureIdxToLabel = split_features(allFeature, allLabelOneHot)

    dataGraph = build_graph(allFeature, uIdxSet, vIdxSet, featureIdxToLabel, graphDict)

    save_vertices_and_attr(allFeature, uIdxSet, vIdxSet)
    save_labels(allLabelOneHot, uIdxSet)
    save_edges(dataGraph)
