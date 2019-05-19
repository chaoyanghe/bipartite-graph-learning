import numpy as np
import pickle
import networkx as nx
import itertools

randomSeed = 2019
numDimKept = 400
splitRate = 6 / 10


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


def split_features_alt(features: np.ndarray, labels_one_hot: np.ndarray, split_rate=0.5):
    feat_idx_to_label = np.argmax(labels_one_hot, axis=1)

    # two sets of vertices
    u_idx_set = set()
    v_idx_set = set()

    feat_idx_to_class_size = np.sum(labels_one_hot, axis=0)
    class_element_count = np.zeros(labels_one_hot.shape[1], dtype=int)
    class_element_limit = np.asarray(feat_idx_to_class_size * split_rate, dtype=int)

    for feat_idx in range(len(features)):
        label = feat_idx_to_label[feat_idx]

        # split each class by half
        if class_element_count[label] <= class_element_limit[label]:
            u_idx_set.add(feat_idx)
        else:
            v_idx_set.add(feat_idx)

        class_element_count[label] += 1

    return u_idx_set, v_idx_set


def split_features(graph_dict: dict, v_set_size: int = 1000):
    # two sets of vertices
    u_idx_set = set()
    v_idx_set = set()

    original_graph = nx.Graph()
    original_graph.add_nodes_from(graph_dict.keys())

    for key in graph_dict.keys():
        # order doesn't matter here
        edges = list(itertools.product([key], graph_dict[key]))
        original_graph.add_edges_from(edges)

    # sort vertices in original graph by degree in descending order
    sorted_nodes_degree_list = sorted(list(original_graph.degree), key=lambda x: x[1], reverse=True)

    # first v_set_size vertices are stored in set V
    [v_idx_set.add(vertex_idx) for vertex_idx, degree in sorted_nodes_degree_list[:v_set_size]]
    [u_idx_set.add(vertex_idx) for vertex_idx, degree in sorted_nodes_degree_list[v_set_size:]]
    print("before filtering, |U| = {}, |V| = {}".format(len(u_idx_set), len(v_idx_set)))

    return u_idx_set, v_idx_set


def build_graph(u_idx_set: set, v_idx_set: set, graph_dict: dict):
    edges = list()

    for feat_idx in graph_dict.keys():
        # order doesn't matter here
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
    assert nx.is_bipartite(graph), "Error: graph should be bipartite!"

    # A = np.asarray(nx.adjacency_matrix(graph).todense())
    # print(A.shape)
    new_u_idx_set = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
    new_v_idx_set = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 1}

    return new_u_idx_set, new_v_idx_set, graph


def reduce_dim(features: np.ndarray, num_dim_kept: int):
    dim_indices_kept = np.arange(features.shape[1])
    np.random.shuffle(dim_indices_kept)
    dim_indices_kept = np.sort(dim_indices_kept[:num_dim_kept])

    reduced_features = features[:, dim_indices_kept]
    assert (len(reduced_features) == len(features)) and \
           (reduced_features.shape[1] == num_dim_kept), "Error: incorrect dimension"

    return reduced_features


def save_vertices_and_attr(features: np.ndarray, u_idx_set: set, v_idx_set: set):
    vertex_set_names = ["node", "group"]
    vertex_sets = [u_idx_set, v_idx_set]

    for i in range(len(vertex_sets)):
        attr_output_str = ""
        vertices_output_str = ""

        for vertex_idx in vertex_sets[i]:
            # already in sorted order since all elements of vertex_sets are sets
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


def print_stat(features: np.ndarray, labels_one_hot: np.ndarray, u_idx_set: set, v_idx_set: set, graph: nx.Graph):
    filter_indices = sorted(u_idx_set.union(v_idx_set))
    filter_features = features[filter_indices]
    filter_labels_one_hot = labels_one_hot[filter_indices]

    print("\n============ Stats ============")
    print("|U| = {}".format(len(u_idx_set)))
    print("|V| = {}".format(len(v_idx_set)))
    print("# vertices in total: {}".format(len(graph)))
    print("# edges in total: {}".format(len(graph.edges)))
    print("feature shape: {}".format(" * ".join(map(str, filter_features.shape))))
    print("# total label: {}".format(len(filter_indices)))


if __name__ == '__main__':
    # set random seed
    np.random.seed(randomSeed)

    allFeature, labelsOneHot, graphDict = load_data()

    # uIdxSet, vIdxSet = split_features_alt(allFeature, labelsOneHot, split_rate=splitRate)
    uIdxSet, vIdxSet = split_features(graphDict, v_set_size=6000)

    uIdxSet, vIdxSet, dataGraph = build_graph(uIdxSet, vIdxSet, graphDict)
    reducedFeatures = reduce_dim(allFeature, num_dim_kept=numDimKept)

    save_vertices_and_attr(reducedFeatures, uIdxSet, vIdxSet)
    save_labels(labelsOneHot, uIdxSet)
    save_edges(dataGraph)

    print_stat(reducedFeatures, labelsOneHot, uIdxSet, vIdxSet, dataGraph)
