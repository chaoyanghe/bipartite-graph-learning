import json
import logging

import networkx as nx
import numpy as np
import scipy.sparse as sp
from networkx.algorithms.bipartite import biadjacency_matrix
from sklearn import preprocessing


class BipartiteGraphDataLoaderTencent:
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
        u_attr_dict, u_attr_array = self.__load_u_attribute(u_list)
        logging.info("u_attribute = %s: %s" % (u_attr_array, u_attr_array[0::100000]))  # 1089436

        logging.info("u_attribute = %s: %s" % (u_attr_array, u_attr_array[0::100000]))

        v_list = self.__load_v_list()
        v_attr_dict, v_attr_array = self.__load_v_attribute(v_list)
        #       logging.info("v_attribute = %s: %s" % (v_attr_array.shape, v_attr_array[0::50000]))  # 90047

        # choose the edge whose nodes have attribute
        f_edge_list = open(self.edge_list_file_path, 'r')
        edge_count = 0
        for l in f_edge_list:
            items = l.strip('\n').split(" ")
            v = int(items[0])
            u = int(items[1])
            edge_count += 1
            if int(v) in v_attr_dict.keys() and int(u) in u_attr_dict.keys():
                self.edge_list.append((u, v))

        logging.info("raw edge_list len = %d" % edge_count)  # 1979756
        logging.info("edge_list len = %d" % len(self.edge_list))  # 991734

        # load all the nodes without duplicate
        self.u_node_list, self.v_node_list = self.__load_unique_node_in_edge_list(self.edge_list)

        # sample and print several value to evaluate the correctness
        logging.info("u_list len = %d. %s" % (len(self.u_node_list), str(self.u_node_list[0::50000])))  # 619030
        logging.info("v_list len = %d. %s" % (len(self.v_node_list), str(self.v_node_list[0::10000])))  # 90044

        # delete the nodes which do not have the attribute
        # delete the nodes which have attribute but are not in the edge list (isolated)
        self.u_attr_dict, self.u_attr_array = self.__filter_illegal_nodes(u_attr_dict, self.u_node_list)
        self.v_attr_dict, self.v_attr_array = self.__filter_illegal_nodes(v_attr_dict, self.v_node_list)
        # logging.info("u feature shape = %s" % str(self.u_attr_array.shape))
        # logging.info("v feature shape = %s" % str(self.v_attr_array.shape))

        self.u_adjacent_matrix, self.v_adjacent_matrix = self.__generate_adjacent_matrix(self.u_node_list,
                                                                                         self.v_node_list,
                                                                                         self.edge_list)

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
        """ Load the node (u) attributes vector.
            If there is no attribute vector, ignore it.
        """

        def decode_helper(s):
            if s == "":
                return 0
            return float(s)

        # node_list: id
        f_u_list = open(self.group_u_list_file_path)
        u2i_dict = {}
        for l in f_u_list:
            l = l.strip().split("\t")
            u2i_dict[l[0]] = int(l[0])

        f_u_attr = open(self.group_u_attr_file_path, 'r')

        converters = {0: lambda s: u2i_dict[s.decode("utf-8")], 1: decode_helper, 4: decode_helper, 5: decode_helper,
                      6: decode_helper,
                      7: decode_helper, 8: decode_helper, 9: decode_helper, 10: decode_helper}

        # This part is different from differe data.
        # data: [[id, feature1, ..., feature10], ..., [id, feature1, ..., feature10]]
        data = np.loadtxt(f_u_attr, delimiter='\t', converters=converters, usecols=(0, 1, 4, 5, 6, 7, 8, 9, 10))

        # normalize per dim
        # follow the best practice here:
        # https://github.com/soumith/talks/blob/master/2017-ICCV_Venice/How_To_Train_a_GAN.pdf
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        data[:, 1:] = min_max_scaler.fit_transform(data[:, 1:])

        data = data.tolist()

        # attr_dict: {id : [feature1, ..., feature10]}
        temp_attr_dict = {}
        for u_t in data:
            temp_attr_dict[u_t[0]] = u_t[1:]

        # merge with the v_list
        logging.info("before merging with u_list, the len is = %d" % len(temp_attr_dict))
        u_attr_dict = {}
        u_attr_array = []
        for u in u_list:
            if u in temp_attr_dict.keys():
                u_attr_dict[int(u)] = temp_attr_dict[u]
                u_attr_array.append(temp_attr_dict[u])

        logging.info("after merging with u_list, the len is = %d" % len(u_attr_dict))

        return u_attr_dict, u_attr_array

    def __load_v_list(self):
        v_list = []
        f_group_v_list = open(self.group_v_list_file_path)
        for l in f_group_v_list:
            v_list.append(int(l))
        return v_list

    def __load_v_attribute(self, v_list):
        v_attr = []
        count_no_attribute = 0
        count_10 = 0
        count_14 = 0
        count_15 = 0
        count_16 = 0
        count_17 = 0
        count_more_than_17 = 0
        count_all = 0
        f_v_attr = open(self.group_v_attr_file_path, 'r')

        for l in f_v_attr:
            count_all += 1

            l = l.strip('\n').split("\t")
            dimension = len(l)

            # skip all the nodes which do not have the attribute
            # TODO: evaluate the performance when keep these non-attribute nodes.
            if dimension == 1:
                count_no_attribute += 1
                continue

            if dimension == 10:
                count_10 += 1
                continue
            attribute_item = []
            if dimension >= 14:
                for idx in range(14):
                    if l[idx] == '':
                        attribute_item.append(0)
                    else:
                        attribute_item.append(float(l[idx]))

            if dimension == 14:
                count_14 += 1
                attribute_item.append(0)
                attribute_item.append(0)
                attribute_item.append(0)
            if dimension == 15:
                attribute_item.append(float(l[14]) if l[14] != '' else 0)
                attribute_item.append(0)
                attribute_item.append(0)
                count_15 += 1
            if dimension == 16:
                attribute_item.append(float(l[14]) if l[14] != '' else 0)
                attribute_item.append(float(l[15]) if l[15] != '' else 0)
                attribute_item.append(float(0))
                count_16 += 1
            if dimension == 17:
                attribute_item.append(float(l[14]) if l[14] != '' else 0)
                attribute_item.append(float(l[15]) if l[15] != '' else 0)
                attribute_item.append(float(l[16]) if l[16] != '' else 0)
                count_17 += 1
            if dimension > 17 or dimension < 10:
                count_more_than_17 += 1
            # logging.info(attribute_item)
            v_attr.append(attribute_item)
        logging.info("count_no_attribute = %d" % count_no_attribute)
        logging.info("count_10 = %d" % count_10)
        logging.info("count_14 = %d" % count_14)
        logging.info("count_15 = %d" % count_15)
        logging.info("count_16 = %d" % count_16)
        logging.info("count_17 = %d" % count_17)
        logging.info("count_more_than_17 = %d" % count_more_than_17)
        logging.info("count_all = %d" % count_all)

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

        logging.info("after merging with v_list, the len is = %d" % len(v_attr_dict))

        return v_attr_dict, v_attr_array

    def __load_unique_node_in_edge_list(self, edge_list):
        u_unique_dict = {}
        v_unique_dict = {}
        for (u, v) in edge_list:
            if v not in v_unique_dict.keys():
                v_unique_dict[int(v)] = v

            if u not in u_unique_dict.keys():
                u_unique_dict[int(u)] = u

        logging.info("group U length = " + str(len(u_unique_dict)))
        logging.info("group V length = " + str(len(v_unique_dict)))
        return [u for u in u_unique_dict.keys()], [v for v in v_unique_dict.keys()]

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

    #
    # def plot_neighborhood_number_distribution(self):
    #     count_list = np.sum(self.u_adjacent_matrix[0:100000], axis=1)
    #     u_adj_ner_count_dict = {}
    #     for idx in range(len(count_list)):
    #         neigher_num = count_list[idx]
    #         if neigher_num not in u_adj_ner_count_dict.keys():
    #             u_adj_ner_count_dict[neigher_num] = 0
    #         u_adj_ner_count_dict[neigher_num] += 1
    #
    #     logging.info(len(u_adj_ner_count_dict))
    #     plot_x = []
    #     plot_y = []
    #     for neigher_num in sorted(u_adj_ner_count_dict.keys()):
    #         if neigher_num == 0 or u_adj_ner_count_dict[neigher_num] == 0:
    #             continue
    #         plot_x.append(neigher_num)
    #         plot_y.append(u_adj_ner_count_dict[neigher_num])
    #
    #     plt.plot(plot_x, plot_y, color="red", linewidth=2)
    #     plt.xlabel("Neighborhood Number")
    #     plt.ylabel("Count")
    #     plt.title("Neighborhood Number Distribution")
    #     plt.axis([0, 50, 0, 5000])
    #     plt.show()

    def __generate_u_labels(self, u_node_list):
        f_label = open(self.group_u_label_file_path)
        true_set = set([int(x.strip()) for x in f_label])
        u_label = []
        for n in u_node_list:
            if n in true_set:
                u_label.append(1)
            else:
                u_label.append(0)
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

    def get_one_batch_group_u_with_adjacent(self, batch_index):
        """
        :param batch_index: batch index, iterate from batch_num_u
        :return: Tensor
        """
        if batch_index >= self.batch_num_u:
            raise Exception("batch_index is larger than the batch number")
        (u_attr_batch, u_adaj_batch) = self.batches_u[batch_index]
        return np.copy(u_attr_batch), np.copy(u_adaj_batch)

    def get_one_batch_group_v_with_adjacent(self, batch_index):
        """
        :param batch_index: batch index, iterate from batch_num_v
        :return: Tensor
        """
        if batch_index >= self.batch_num_v:
            raise Exception("batch_index is larger than the batch number")
        (v_attr_batch, v_adaj_batch) = self.batches_v[batch_index]
        return np.copy(v_attr_batch), np.copy(v_adaj_batch)

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

    def get_edge_list(self):
        return self.edge_list


class GraphSageDataLoader:

    def __init__(self, u_list, v_list, u_attr, v_attr, edge_list, node_true):
        self.u_attr = u_attr
        self.v_attr = v_attr
        self.u_list = u_list
        self.v_list = v_list
        self.edge_list = edge_list
        self.node_true = node_true

        self.u_map_id = {}
        self.id_map_u = {}
        self.v_map_id = {}
        self.id_map_v = {}
        self.label = {}
        self.node_map_id = {}

        self.graph = {}
        self.graph['directed'] = False
        self.graph['graph'] = {'name': 'bipartite graph'}
        self.graph['multigraph'] = False

    def node_id_map(self):
        u_list_length = len(self.u_list)
        self.u_map_id = dict([(self.u_list[i], i) for i in range(u_list_length)])
        self.id_map_u = dict([(i, self.u_list[i]) for i in range(u_list_length)])  # id to u
        self.v_map_id = dict([(self.v_list[i], i + u_list_length) for i in range(len(self.v_list))])
        self.id_map_v = dict([(i + u_list_length, self.v_list[i]) for i in range(len(self.v_list))])  # id to v

    def nodes_form(self):
        nodes = []
        length = len(self.u_list) + len(self.v_list)
        for i in range(length):
            temp_node = {}
            temp_node['id'] = i
            temp_node['test'] = False
            if i < len(self.u_list):
                temp_node['feature'] = self.u_attr[i]
                if self.id_map_u[i] in self.node_true:
                    temp_node['label'] = [1, 0]
                else:
                    temp_node['label'] = [0, 1]
            else:
                temp_node['feature'] = self.v_attr[i - len(self.u_list)]
                temp_node['label'] = [0, 1]
            if np.random.choice([0, 1], p=[0.3, 0.7]):
                temp_node['val'] = False
            else:
                temp_node['val'] = True
            nodes.append(temp_node)
        self.graph['nodes'] = nodes

    def link_form(self):
        links = []
        for edge in self.edge_list:
            temp_link = {}
            temp_link['source'] = self.u_map_id[edge[0]]
            temp_link['target'] = self.v_map_id[edge[1]]
            temp_link['test_removed'] = False
            temp_link['train_removed'] = False
            links.append(temp_link)
        self.graph['links'] = links

    def class_form(self):
        self.label = {}
        length = len(self.u_list) + len(self.v_list)
        for i in range(length):
            if i < len(self.u_list):
                if self.id_map_u[i] in self.node_true:
                    self.label[i] = [1, 0]
                else:
                    self.label[i] = [0, 1]
            else:
                self.label[i] = [0, 1]

    def write_to_json(self):
        with open('./bipartite-G.json', 'w') as outfile1:
            json.dump(self.graph, outfile1)

        length = len(self.u_list) + len(self.v_list)
        node_map_id = dict([(i, i) for i in range(length)])
        with open('./bipartite-id_map.json', 'w') as outfile2:
            json.dump(node_map_id, outfile2)

        with open('./bipartite-class_map.json', 'w') as outfile3:
            json.dump(self.label, outfile3)

    def get_id_map_node(self, write_to_file=False):
        id_map_node = self.id_map_u.copy()
        id_map_node.update(self.id_map_v)
        id_map_node['u_num'] = len(self.u_list)
        id_map_node['v_num'] = len(self.v_list)
        if write_to_file:
            with open('./bipartite-id_map_node.json', 'w') as outfile:
                json.dump(id_map_node, outfile)
            outfile.close()


class GraphSageSingleGraphDataLoader:
    def __init__(self, u_adj, u_list, u_attr, node_true):
        self.u_adj = u_adj
        self.u_list = u_list
        self.u_attr = u_attr
        self.node_true = node_true
        self.id_map_node = {}
        self.features = []

        self.adj = self.__BipartiteToSingle(u_adj)  # the adjacent connection only of set U
        self.label = {}
        self.graph = {}
        self.graph['directed'] = False
        self.graph['graph'] = {'name': 'bipartite graph'}
        self.graph['multigraph'] = False

    def data_loader(self):
        logging.info('ID map node')
        self.get_id_map_node()
        logging.info('Nodes Form')
        self.nodes_form()
        logging.info('Link Form')
        self.link_form()
        logging.info('Class Form')
        self.class_form()
        logging.info('Feature Form')
        self.feature_form()
        logging.info('Write to Json')
        self.write_to_json()
        logging.info('Done written')

    def get_id_map_node(self):
        self.id_map_node = dict([(i, self.u_list[i]) for i in range(len(u_list))])
        self.id_map_node['u_num'] = len(self.u_list)

    def nodes_form(self):
        nodes = []
        length = len(self.u_list)
        for i in range(length):
            temp_node = {}
            temp_node['id'] = i
            temp_node['test'] = False
            # temp_node['feature'] = self.u_attr[i]
            if self.u_list[i] in self.node_true:
                temp_node['label'] = [1, 0]
            else:
                temp_node['label'] = [0, 1]
            temp_node['val'] = False
            nodes.append(temp_node)
        self.graph['nodes'] = nodes

    def __sparse_to_tuple(self, sparse_mx):
        if not sp.isspmatrix_coo(sparse_mx):
            sparse_mx = sparse_mx.tocoo()
        coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
        return coords

    def link_form(self):
        triu = sp.triu(self.adj, 0)
        edges = self.__sparse_to_tuple(triu)
        links = []
        for i in range(len(edges)):
            s, t = edges[i][:2]
            temp_link = {}
            temp_link['source'] = int(s)
            temp_link['target'] = int(t)
            temp_link['test_removed'] = False
            temp_link['train_removed'] = False
            links.append(temp_link)
        self.graph['links'] = links

    def class_form(self):
        for i in range(len(self.u_list)):
            if self.u_list[i] in self.node_true:
                self.label[i] = [1, 0]
            else:
                self.label[i] = [0, 1]

    def feature_form(self):
        nodes_features = []
        length = len(self.u_list)
        for i in range(length):
            nodes_features.append(self.u_attr[i])
        self.features = np.array(nodes_features)

    def write_to_json(self):
        with open('./tencent/bipartite-G.json', 'w') as outfile1:
            json.dump(self.graph, outfile1)
        outfile1.close()

        length = len(self.u_list)
        node_map_id = dict([(i, i) for i in range(length)])
        with open('./tencent/bipartite-id_map.json', 'w') as outfile2:
            json.dump(node_map_id, outfile2)
        outfile2.close()

        with open('./tencent/bipartite-class_map.json', 'w') as outfile3:
            json.dump(self.label, outfile3)
        outfile3.close()

        with open('./tencent/bipartite-id_map_node.json', 'w') as outfile:
            json.dump(self.id_map_node, outfile)
        outfile.close()

        np.save('./tencent/bipartite-feats.npy', self.features)

    def __BipartiteToSingle(self, graph):
        """
        transfer the bipartite graph to single graph
        :param graph: sparse csr_matrix
        :return: sparse adjacent csr_matrix
        """
        single_graph = graph.dot(graph.T)
        single_graph[single_graph != 0] = 1
        single_graph -= sp.identity(graph.shape[0])
        return single_graph


if __name__ == "__main__":
    logging.basicConfig(filename='bipartite_graph_data_loading.log', filemode='w',
                        format='%(asctime)s  %(filename)s : %(lineno)d : %(levelname)s  %(message)s',
                        datefmt='%Y-%m-%d %A %H:%M:%S',
                        level=logging.INFO)

    # load the bipartite graph data
    dataset = 'tencent'
    NODE_LIST_PATH = "../../data/%s/node_list" % dataset
    NODE_ATTR_PATH = "../../data/%s/node_attr" % dataset
    NODE_LABEL_PATH = "../../data/%s/node_true" % dataset
    EDGE_LIST_PATH = "../../data/%s/edgelist" % dataset
    GROUP_LIST_PATH = "../../data/%s/group_list" % dataset
    GROUP_ATTR_PATH = "../../data/%s/group_attr" % dataset
    NODE_TRUE = "../../data/%s/node_true" % dataset
    bipartite_graph_data_loader = BipartiteGraphDataLoaderTencent(100, NODE_LIST_PATH, NODE_ATTR_PATH,
                                                                  NODE_LABEL_PATH,
                                                                  EDGE_LIST_PATH,
                                                                  GROUP_LIST_PATH, GROUP_ATTR_PATH)
    bipartite_graph_data_loader.load()
    u_attr = bipartite_graph_data_loader.get_u_attr_array()
    v_attr = bipartite_graph_data_loader.get_v_attr_array()
    u_list = bipartite_graph_data_loader.get_u_list()
    v_list = bipartite_graph_data_loader.get_v_list()
    edge_list = bipartite_graph_data_loader.get_edge_list()
    u_adj = bipartite_graph_data_loader.get_u_adj()  # csr_matrix

    with open(NODE_TRUE, 'r') as file:
        node_true = file.readlines()
    node_true = list(map(lambda x: int(x.strip()), node_true))

    # # test the code
    # u_list = [1, 3, 5, 7, 9]
    # v_list = [1, 3, 6, 8]
    # u_attr = np.random.rand(5, 4).tolist()
    # v_attr = np.random.rand(4, 3).tolist()
    # u_adj = sp.csr_matrix([[1, 1, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    # edge_list = [(1, 1), (1, 3), (1, 6), (3, 3), (5, 1), (5, 6), (7, 8), (9, 6)]
    # node_true = [1, 5, 9]

    # define whether to use the one hop graph or two hop graph
    bipartite_graph = False
    if bipartite_graph:
        logging.info('Start graphsage data loader')
        graphsage_loader = GraphSageDataLoader(u_list, v_list, u_attr, v_attr, edge_list, node_true)

        logging.info('node id map')
        graphsage_loader.node_id_map()

        logging.info('nodes form')
        graphsage_loader.nodes_form()

        logging.info('link form')
        graphsage_loader.link_form()
        graphsage_loader.class_form()

        logging.info('writing data to file')
        graphsage_loader.write_to_json()
        graphsage_loader.get_id_map_node(True)
    else:
        logging.info('Start graphsage two hop graph loader')
        graphsage_loader = GraphSageSingleGraphDataLoader(u_adj, u_list, u_attr, node_true)
        graphsage_loader.data_loader()

    # logging.info('######### U adjacent matrix ##########\n' + str(u_adj))
    # logging.info('######### V adjacent matrix ##########\n' + str(v_adj))
    # logging.info('######### U attribute matrix ##########\n' + str(u_attr))
    # logging.info('######### V attribute matrix ##########\n' + str(v_attr))
    # logging.info('######### U list ##########\n' + str(u_list))
    # logging.info('######### V list ##########\n' + str(v_list))
