import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms.bipartite import biadjacency_matrix
from sklearn import preprocessing


class BipartiteGraphDataLoaderPubMed:
    def __init__(self, batch_size, group_u_list_file_path, group_u_attr_file_path, group_u_label_file_path,
                 edge_list_file_path,
                 group_v_list_file_path, group_v_attr_file_path, group_v_label_file_path=None, device='cpu'):

        logging.info("BipartiteGraphDataLoader __init__().")

        self.device = device

        self.batch_size = batch_size
        self.batch_num_u = 0
        self.batch_num_v = 0
        logging.info("group_u_list_file_path = %s" % group_u_list_file_path)
        logging.info("group_u_attr_file_path = %s" % group_u_attr_file_path)
        logging.info("group_u_label_file_path = %s" % group_u_label_file_path)

        logging.info("edge_list_file_path = %s" % edge_list_file_path)

        logging.info("group_v_list_file_path = %s" % group_v_list_file_path)
        logging.info("group_v_attr_file_path = %s" % group_v_attr_file_path)
        logging.info("group_v_label_file_path = %s" % group_v_label_file_path)

        self.group_u_list_file_path = group_u_list_file_path
        self.group_u_attr_file_path = group_u_attr_file_path
        self.group_u_label_file_path = group_u_label_file_path
        self.edge_list_file_path = edge_list_file_path
        self.group_v_list_file_path = group_v_list_file_path
        self.group_v_attr_file_path = group_v_attr_file_path
        self.group_v_label_file_path = group_v_label_file_path

        self.u_node_list = []
        self.u_attr_dict = {}
        self.u_attr_array = []

        self.v_node_list = []
        self.v_attr_dict = {}
        self.v_attr_array = []

        self.edge_list = []
        self.u_adjacent_matrix = []
        self.v_adjacent_matrix = []

        self.u_label = []

        self.batches_u = []
        self.batches_v = []
        logging.info("BipartiteGraphDataLoader __init__(). END")

    def test(self):
        adjU = [[1, 1],
                [1, 0],
                [1, 0],
                [1, 1],
                [1, 1],
                [0, 1],
                [0, 1]]
        adjV = [[1, 1, 1, 1, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 1]]
        featuresU = np.random.rand(14).reshape(7, 2)
        featuresV = np.random.rand(8).reshape(2, 4)
        self.gernerate_mini_batch(featuresU, featuresV, np.array(adjU), np.array(adjV))
        logging.info("")

    def load(self):
        logging.info("##### generate_adjacent_matrix_feature_and_labels. START")
        u_list = self.__load_u_list()
        # print("####u_list = %d" % len(u_list))
        u_attr_dict, u_attr_array = self.__load_u_attribute(u_list)
        # print("####u_attr_dict = %d" % len(u_attr_dict))

        v_list = self.__load_v_list()
        # print("v_list = %d" % len(v_list))
        v_attr_dict, v_attr_array = self.__load_v_attribute(v_list)
        # print("v_attr_dict = %d" % len(v_attr_dict))
        #       logging.info("v_attribute = %s: %s" % (v_attr_array.shape, v_attr_array[0::50000]))  # 90047

        # choose the edge whose nodes have attribute
        f_edge_list = open(self.edge_list_file_path, 'r')
        edge_count = 0
        # print("####u keys = %s" % u_attr_dict.keys())
        # print("####v keys = %s" % v_attr_dict.keys())
        for l in f_edge_list:
            items = l.strip('\n').split("\t")
            u = int(items[0])
            v = int(items[1])
            if v in v_attr_dict.keys() and u in u_attr_dict.keys():
                edge_count += 1

                self.edge_list.append((u, v))

        # print("raw edge_list len = %d" % edge_count)  # 1979756
        # print("edge_list len = %d" % len(self.edge_list))  # 991734

        # load all the nodes without duplicate
        self.u_node_list = u_list
        self.v_node_list = v_list

        self.u_attr_dict = u_attr_dict
        self.v_attr_dict = v_attr_dict

        self.u_attr_array = u_attr_array
        self.v_attr_array = v_attr_array

        self.u_adjacent_matrix, self.v_adjacent_matrix = self.__generate_adjacent_matrix(self.u_node_list,
                                                                                         self.v_node_list,
                                                                                         self.edge_list)
        # print(self.u_node_list)
        self.u_label = self.__generate_u_labels(self.u_node_list)

        # for mini-batch
        self.gernerate_mini_batch(self.u_attr_array, self.v_attr_array,
                                  self.u_adjacent_matrix, self.v_adjacent_matrix)

        logging.info("#### generate_adjacent_matrix_feature_and_labels. END")

    def __load_u_list(self):
        u_list = []
        f_group_u_list = open(self.group_u_list_file_path)
        for l in f_group_u_list:
            u_list.append(int(l))
        return u_list

    def __load_u_attribute(self, u_list):
        u_attr = []
        f_u_attr = open(self.group_u_attr_file_path, 'r')

        dimension = 0
        for l in f_u_attr:
            l = l.strip('\n').split("\t")
            attribute_item = []
            dimension = len(l)
            for idx in range(dimension):
                attribute_item.append(float(l[idx]))
            u_attr.append(attribute_item)
        print("u dimension = %s" % str(dimension - 1))
        # print("u_attr = %d" % len(u_attr))
        # normalize per dim
        u_attr_np = np.array(u_attr, dtype=np.float64, copy=False)

        # follow the best practice here:
        # https://github.com/soumith/talks/blob/master/2017-ICCV_Venice/How_To_Train_a_GAN.pdf
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        u_attr_np[:, 1:] = min_max_scaler.fit_transform(u_attr_np[:, 1:])

        u_attr_np = u_attr_np.tolist()
        # print("u_attr_np = %d" % len(u_attr_np))

        # attr_dict: {id : [feature1, ..., feature10]}
        temp_attr_dict = {}
        for u_t in u_attr_np:
            temp_attr_dict[u_t[0]] = u_t[1:]

        # print("temp_attr_dict = %d" % len(temp_attr_dict))
        # merge with the v_list
        # print("before merging with v_list, the len is = %d" % len(u_attr))
        u_attr_dict = {}
        u_attr_array = []
        for u in u_list:
            if u in temp_attr_dict.keys():
                u_attr_dict[int(u)] = temp_attr_dict[u]
                u_attr_array.append(temp_attr_dict[u])

        # print("after merging with u_attr_dict, the len is = %d" % len(u_attr_dict))

        return u_attr_dict, u_attr_array

    def __load_v_list(self):
        v_list = []
        f_group_v_list = open(self.group_v_list_file_path)
        for l in f_group_v_list:
            v_list.append(int(l))
        return v_list

    def __load_v_attribute(self, v_list):
        v_attr = []
        f_v_attr = open(self.group_v_attr_file_path, 'r')

        dimension = 0
        for l in f_v_attr:
            l = l.strip('\n').split("\t")
            attribute_item = []
            dimension = len(l)
            for idx in range(dimension):
                attribute_item.append(float(l[idx]))
            v_attr.append(attribute_item)
        print("v dimension = %s" % str(dimension - 1))
        # print("v_attr = %d" % len(v_attr))
        # normalize per dim
        v_attr_np = np.array(v_attr, dtype=np.float64, copy=False)

        # follow the best practice here:
        # https://github.com/soumith/talks/blob/master/2017-ICCV_Venice/How_To_Train_a_GAN.pdf
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        v_attr_np[:, 1:] = min_max_scaler.fit_transform(v_attr_np[:, 1:])

        v_attr_np = v_attr_np.tolist()

        # attr_dict: {id : [feature1, ..., feature10]}
        temp_attr_dict = {}
        for v_t in v_attr_np:
            temp_attr_dict[v_t[0]] = v_t[1:]

        # merge with the v_list
        logging.info("before merging with v_list, the len is = %d" % len(v_attr))
        v_attr_dict = {}
        v_attr_array = []
        for v in v_list:
            if v in temp_attr_dict.keys():
                v_attr_dict[int(v)] = temp_attr_dict[v]
                v_attr_array.append(temp_attr_dict[v])

        logging.info("after merging with v_attr_dict, the len is = %d" % len(v_attr_dict))

        return v_attr_dict, v_attr_array

    def __filter_illegal_nodes(self, attr_dict, unique_node_list):
        ret_attr_dict = {}
        ret_attr_array = []
        logging.info("before filter, the len is = %d" % len(attr_dict))
        for node in unique_node_list:
            ret_attr_dict[node] = attr_dict[node]
            ret_attr_array.append(attr_dict[node])
        logging.info("after filter, the len is = %d" % len(ret_attr_array))
        return ret_attr_dict, ret_attr_array

    def __generate_adjacent_matrix(self, u_node_list, v_node_list, edge_list):
        logging.info("__generate_adjacent_matrix START")

        logging.info("u_node_list = %d" % len(u_node_list))
        logging.info("v_node_list = %d" % len(v_node_list))
        logging.info("edge_list = %d" % len(edge_list))  # 1979756(after filter); 991734(after filter)

        logging.info("start to load bipartite for u")
        B_u = nx.Graph()
        # Add nodes with the node attribute "bipartite"
        B_u.add_nodes_from(u_node_list, bipartite=0)
        B_u.add_nodes_from(v_node_list, bipartite=1)

        # Add edges only between nodes of opposite node sets
        B_u.add_edges_from(edge_list)

        u_adjacent_matrix_np = biadjacency_matrix(B_u, u_node_list, v_node_list)
        logging.info(u_adjacent_matrix_np.shape)
        # u_adjacent_matrix_np = u_adjacent_matrix.todense().A
        B_u.clear()
        logging.info("end to load bipartite for u")

        logging.info("start to load bipartite for u")
        B_v = nx.Graph()
        # Add nodes with the node attribute "bipartite"
        B_v.add_nodes_from(v_node_list, bipartite=0)
        B_v.add_nodes_from(u_node_list, bipartite=1)

        # Add edges only between nodes of opposite node sets
        B_v.add_edges_from(edge_list)

        v_adjacent_matrix_np = biadjacency_matrix(B_v, v_node_list, u_node_list)
        logging.info(v_adjacent_matrix_np.shape)
        # v_adjacent_matrix_np = v_adjacent_matrix.todense().A
        B_v.clear()
        logging.info("end to load bipartite for u")
        return u_adjacent_matrix_np, v_adjacent_matrix_np

    def plot_neighborhood_number_distribution(self):
        u_adjacent_matrix_np = self.u_adjacent_matrix.todense().A
        count_list = np.sum(u_adjacent_matrix_np, axis=1)
        print(count_list)
        u_adj_ner_count_dict = {}
        for idx in range(len(count_list)):
            neigher_num = count_list[idx]
            if neigher_num not in u_adj_ner_count_dict.keys():
                u_adj_ner_count_dict[neigher_num] = 0
            u_adj_ner_count_dict[neigher_num] += 1

        logging.info(len(u_adj_ner_count_dict))
        plot_x = []
        plot_y = []
        for neigher_num in sorted(u_adj_ner_count_dict.keys()):
            if neigher_num == 0 or u_adj_ner_count_dict[neigher_num] == 0:
                continue
            plot_x.append(neigher_num)
            plot_y.append(u_adj_ner_count_dict[neigher_num])

        # plt.plot(plot_x, plot_y, color="red", linewidth=2)
        # plt.xlabel("Neighborhood Number")
        # plt.ylabel("Count")
        # plt.title("Neighborhood Number Distribution (PubMed)")
        # plt.axis([0, 10, 0, 10000])
        # plt.show()

        plt.figure(figsize=(10, 7))
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.plot(plot_x, plot_y, color="red", linewidth=4)
        plt.xlabel("Nodes degree", fontsize=28)
        plt.ylabel("Count", fontsize=28)
        plt.title("Degree Distribution (pubmed)", fontsize=28)
        plt.xticks(np.arange(11, step=2), np.arange(11, step=2))
        ax = plt.gca()
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        plt.axis([0, 10, 0, 10000])
        plt.savefig('./distribution_pubmed.eps', format='eps')

    def __generate_u_labels(self, u_node_list):
        u_label_dict = {}
        f_label = open(self.group_u_label_file_path)
        for l in f_label:
            l = l.strip('\n').split("\t")
            id = int(l[0])
            label = l[1]
            u_label_dict[id] = label
        # print(u_label_dict)
        u_label = []
        for n in u_node_list:
            # print(n)
            u_label.append(u_label_dict[n])
        return u_label

    def gernerate_mini_batch(self, u_attr_array, v_attr_array, u_adjacent_matrix, v_adjacent_matrix):
        u_num = len(u_attr_array)
        logging.info("u number: " + str(u_num))
        logging.info("u_adjacent_matrix: " + str(u_adjacent_matrix.shape))

        v_num = len(v_attr_array)
        logging.info("v number: " + str(v_num))
        logging.info("v_adjacent_matrix: " + str(v_adjacent_matrix.shape))

        self.batch_num_u = int(u_num / self.batch_size) + 1
        logging.info("batch_num_u = %d" % self.batch_num_u)

        self.batch_num_v = int(v_num / self.batch_size) + 1
        logging.info("batch_num_v = %d" % self.batch_num_v)

        for batch_index in range(self.batch_num_u):
            start_index = self.batch_size * batch_index
            end_index = self.batch_size * (batch_index + 1)
            if batch_index == self.batch_num_u - 1:
                end_index = u_num
            tup = (u_attr_array[start_index:end_index], u_adjacent_matrix[start_index:end_index])
            self.batches_u.append(tup)
        # print(self.batches_u)

        for batch_index in range(self.batch_num_v):
            start_index = self.batch_size * batch_index
            end_index = self.batch_size * (batch_index + 1)
            if batch_index == self.batch_num_v - 1:
                end_index = v_num
            tup = (v_attr_array[start_index:end_index], v_adjacent_matrix[start_index:end_index])
            self.batches_v.append(tup)

    # print(self.batches_v)

    def get_u_attr_dimensions(self):
        return len(self.u_attr_array[0])

    def get_v_attr_dimensions(self):
        return len(self.v_attr_array[0])

    def get_batch_num_u(self):
        return self.batch_num_u

    def get_batch_num_v(self):
        return self.batch_num_v

    def get_u_attr_array(self):
        """
        :return: list
        """
        return self.u_attr_array

    def get_v_attr_array(self):
        """
        :return: list
        """
        return self.v_attr_array

    def get_u_adj(self):
        """
        :return: sparse csr_matrix
        """
        return self.u_adjacent_matrix

    def get_v_adj(self):
        return self.v_adjacent_matrix

    def get_u_list(self):
        return self.u_node_list

    def get_v_list(self):
        return self.v_node_list


if __name__ == "__main__":
    logging.basicConfig(filename='bipartite_graph_data_loading.log_embedding', filemode='w',
                        format='%(asctime)s  %(filename)s : %(lineno)d : %(levelname)s  %(message)s',
                        datefmt='%Y-%m-%d %A %H:%M:%S',
                        level=logging.INFO)

    NODE_LIST_PATH = "./../../data/pubmed/node_list"
    NODE_ATTR_PATH = "./../../data/pubmed/node_attr"
    NODE_LABEL_PATH = "./../../data/pubmed/node_true"

    EDGE_LIST_PATH = "./../../data/pubmed/edgelist"

    GROUP_LIST_PATH = "./../../data/pubmed/group_list"
    GROUP_ATTR_PATH = "./../../data/pubmed/group_attr"
    bipartite_graph_data_loader = BipartiteGraphDataLoaderPubMed(3, NODE_LIST_PATH, NODE_ATTR_PATH, NODE_LABEL_PATH,
                                                                 EDGE_LIST_PATH,
                                                                 GROUP_LIST_PATH, GROUP_ATTR_PATH)
    # bipartite_graph_data_loader.test()
    bipartite_graph_data_loader.load()
    bipartite_graph_data_loader.plot_neighborhood_number_distribution()
    u_attr = bipartite_graph_data_loader.get_u_attr_array()

# for i in range(len(u_attr)):
# len(data)
# #data[:, 1:] = data[:, 1:] / (data[:, 1:].max(axis=0) - data[:, 1:].min(axis=0))
# print(data[:, 1:])
# print(data[:, 1:].min(axis=0) / (data[:, 1:].max(axis=0)- data[:, 1:].min(axis=0)))
# print(data[0])
# u_attr_batch, u_adaj_batch = bipartite_graph_data_loader.get_one_batch_group_u_with_adjacent(1)
# count_list = np.sum(u_adaj_batch, axis=1)
# logging.info(u_adaj_batch[0])
# logging.info(count_list)
