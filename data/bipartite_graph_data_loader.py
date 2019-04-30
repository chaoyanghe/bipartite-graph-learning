import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms.bipartite import biadjacency_matrix


class BipartiteGraphDataLoader:
    def __init__(self, batch_size, group_u_list_file_path, group_u_attr_file_path, group_u_label_file_path,
                 edge_list_file_path,
                 group_v_list_file_path, group_v_attr_file_path, group_v_label_file_path=None):

        logging.basicConfig(filename='bipartite_graph_data_loading.log', filemode='w',
                            format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
                            datefmt='%Y-%m-%d %A %H:%M:%S',
                            level=logging.INFO)

        logging.info("BipartiteGraphDataLoader __init__().")

        self.batch_size = batch_size
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

        self.u_list = []
        self.u_attr_dict = {}
        self.u_attr_array = []

        self.v_list = []
        self.v_attr_dict = {}
        self.v_attr_array = []

        self.edge_list = []
        self.u_adjacent_matrix = []
        self.v_adjacent_matrix = []

        self.u_label = []

        logging.info("BipartiteGraphDataLoader __init__(). END")

    def test(self):
        plot_x = [i for i in range(10000000)]
        plot_y = [2 * i for i in range(10000000)]
        plt.plot(plot_x, plot_y, color="red", linewidth=2)
        plt.ylabel("Neighborhood Number")
        plt.xlabel("Count")
        plt.title("Neighborhood Number Distribution")
        plt.show()
        print("")

    def load(self):
        logging.info("##### generate_adjacent_matrix_feature_and_labels. START")
        self.u_list = self.__load_u_list()
        # sample and print several value to evaluate the correctness
        logging.info("u_list len = %d. %s" % (len(self.u_list), str(self.u_list[0::100000])))

        u_attr_dict, u_attr_array = self.__load_u_attribute()
        logging.info("u_attribute = %s: %s" % (u_attr_array.shape, u_attr_array[0::100000]))

        self.v_list = self.__load_v_list()
        # sample and print several value to evaluate the correctness
        logging.info("v_list len = %d. %s" % (len(self.v_list), str(self.v_list[0::50000])))

        v_attr_dict, v_attr_array = self.__load_v_attribute()
        logging.info("v_attribute = %s: %s" % (v_attr_array.shape, v_attr_array[0::50000]))

        group_u_dict, group_v_dict = self.__load_adjacent_list()
        self.u_attr_dict, self.u_attr_array, u_node_list = self.__filter_unused_nodes(u_attr_dict, group_u_dict)
        self.v_attr_dict, self.v_attr_array, v_node_list = self.__filter_unused_nodes(v_attr_dict, group_v_dict)

        f_edge_list = open(self.edge_list_file_path, 'r')
        for l in f_edge_list:
            items = l.strip('\n').split(" ")
            v = int(items[0])
            u = int(items[1])
            if int(v) in self.v_attr_dict.keys() and int(u) in self.u_attr_dict.keys():
                self.edge_list.append((u, v))

        self.__generate_adjacent_matrix(self.u_attr_array, self.v_attr_array, u_node_list, v_node_list)
        logging.info("#### generate_adjacent_matrix_feature_and_labels. END")

    def __load_u_list(self):
        f_group_u_list = open(self.group_u_list_file_path)
        for l in f_group_u_list:
            self.u_list.append(int(l))
        return self.u_list

    def __load_u_attribute(self):
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
        data[:, 1:] = data[:, 1:] / data[:, 1:].max(axis=0)

        # attr_dict: {id : [feature1, ..., feature10]}
        temp_attr_dict = {}
        for u_t in data:
            temp_attr_dict[u_t[0]] = u_t[1:]

        # merge with the v_list
        logging.info("before merging with u_list, the len is = %d" % len(temp_attr_dict))
        u_attr_dict = {}
        u_attr_array = []
        for u in self.u_list:
            if u in temp_attr_dict.keys():
                u_attr_dict[int(u)] = temp_attr_dict[u]
                u_attr_array.append((u, temp_attr_dict[u]))

        logging.info("after merging with u_list, the len is = %d" % len(u_attr_dict))

        return u_attr_dict, np.array(u_attr_array)

    def __load_v_list(self):
        f_group_v_list = open(self.group_v_list_file_path)
        for l in f_group_v_list:
            self.v_list.append(int(l))
        return self.v_list

    def __load_v_attribute(self):
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
            # print(attribute_item)
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
        v_attr_np = np.array(v_attr, dtype=np.float64)
        v_attr_np[:, 1:] = v_attr_np[:, 1:] / v_attr_np[:, 1:].max(axis=0)

        # attr_dict: {id : [feature1, ..., feature10]}
        temp_attr_dict = {}
        for v_t in v_attr_np:
            temp_attr_dict[v_t[0]] = v_t[1:]

        # merge with the v_list
        logging.info("before merging with v_list, the len is = %d" % len(v_attr))
        v_attr_dict = {}
        v_attr_array = []
        for v in self.v_list:
            if v in temp_attr_dict.keys():
                v_attr_dict[int(v)] = temp_attr_dict[v]
                v_attr_array.append((v, temp_attr_dict[v]))

        logging.info("after merging with v_list, the len is = %d" % len(v_attr_dict))

        return v_attr_dict, np.array(v_attr_array)

    def __load_adjacent_list(self):
        f_edge_list = open(self.edge_list_file_path, 'r')
        group_u_dict = {}
        group_v_dict = {}
        for l in f_edge_list:
            items = l.strip('\n').split(" ")
            v = items[0]
            if v not in group_v_dict.keys():
                group_v_dict[int(v)] = v

            u = items[1]
            if u not in group_u_dict.keys():
                group_u_dict[int(u)] = u

        logging.info("group U length = " + str(len(group_u_dict)))
        logging.info("group V length = " + str(len(group_v_dict)))
        return group_u_dict, group_v_dict

    def __filter_unused_nodes(self, attr_dict, group_dict):
        ret_attr_dict = {}
        ret_attr_array = []
        ret_node_list = []
        logging.info("before filter, the len is = %d" % len(attr_dict))
        for node in attr_dict.keys():
            if int(node) in group_dict.keys():
                ret_attr_dict[node] = attr_dict[node]
                ret_attr_array.append(attr_dict[node])
                ret_node_list.append(node)

        logging.info("after filter, the len is = %d" % len(ret_attr_array))
        return ret_attr_dict, ret_attr_array, ret_node_list

    def __generate_adjacent_matrix(self, u_attr_array, v_attr_array, u_node_list, v_node_list):
        logging.info("__generate_adjacent_matrix START")
        dimension_u = len(u_attr_array)
        dimension_v = len(v_attr_array)

        print("u_node_list = %d" % len(u_node_list))
        print("v_node_list = %d" % len(v_node_list))
        print("edge_list = %d" % len(self.edge_list))  # 1979756(after filter); 991734(after filter)

        B_u = nx.Graph()
        # Add nodes with the node attribute "bipartite"
        B_u.add_nodes_from(u_node_list, bipartite=0)
        B_u.add_nodes_from(v_node_list, bipartite=1)

        # Add edges only between nodes of opposite node sets
        B_u.add_edges_from(self.edge_list)

        u_adjacent_matrix = biadjacency_matrix(B_u, u_node_list, v_node_list)
        logging.info(u_adjacent_matrix)
        print(u_adjacent_matrix.shape)
        self.u_adjacent_matrix = u_adjacent_matrix.toarray()

        B_v = nx.Graph()
        # Add nodes with the node attribute "bipartite"
        B_v.add_nodes_from(v_node_list, bipartite=0)
        B_v.add_nodes_from(u_node_list, bipartite=1)

        # Add edges only between nodes of opposite node sets
        B_v.add_edges_from(self.edge_list)

        v_adjacent_matrix = biadjacency_matrix(B_v, v_node_list, u_node_list)
        logging.info(v_adjacent_matrix)
        print(v_adjacent_matrix.shape)
        self.v_adjacent_matrix = v_adjacent_matrix.toarray()

    def plot_neighborhood_number_distribution(self):
        count_list = np.sum(self.u_adjacent_matrix[0:100000], axis=1)
        u_adj_ner_count_dict = {}
        for idx in range(len(count_list)):
            neigher_num = count_list[idx]
            if neigher_num not in u_adj_ner_count_dict.keys():
                u_adj_ner_count_dict[neigher_num] = 0
            u_adj_ner_count_dict[neigher_num] += 1

        print(len(u_adj_ner_count_dict))
        plot_x = []
        plot_y = []
        for neigher_num in sorted(u_adj_ner_count_dict.keys()):
            if neigher_num == 0 or u_adj_ner_count_dict[neigher_num] == 0:
                continue
            plot_x.append(neigher_num)
            plot_y.append(u_adj_ner_count_dict[neigher_num])

        plt.plot(plot_x, plot_y, color="red", linewidth=2)
        plt.xlabel("Neighborhood Number")
        plt.ylabel("Count")
        plt.title("Neighborhood Number Distribution")
        plt.show()


    def __generate_features_and_labels(self):
        logging.info("__generate_features_and_labels. START")

        logging.info("__generate_features_and_labels. END")


    def get_batch_num(self):
        pass


    def get_one_batch_group_u_with_adjacent(self, batch_index):
        start_index = self.batch_size * batch_index
        end_index = self.batch_size * (batch_index + 1)
        if end_index >= len(self.u_list):
            end_index = len(self.u_list) - 1


        return None, None, None


    def get_one_batch_group_v_with_adjacent(self, batch_index):
        return None, None, None


if __name__ == "__main__":
    NODE_LIST_PATH = "./Tencent-QQ/node_list"
    NODE_ATTR_PATH = "./Tencent-QQ/node_attr"
    NODE_LABEL_PATH = "./Tencent-QQ/node_true"

    EDGE_LIST_PATH = "./Tencent-QQ/edgelist"

    GROUP_LIST_PATH = "./Tencent-QQ/group_list"
    GROUP_ATTR_PATH = "./Tencent-QQ/group_attr"
    bipartite_graph_data_loader = BipartiteGraphDataLoader(10, NODE_LIST_PATH, NODE_ATTR_PATH, NODE_LABEL_PATH,
                                                           EDGE_LIST_PATH,
                                                           GROUP_LIST_PATH, GROUP_ATTR_PATH)
    # bipartite_graph_data_loader.test()
    bipartite_graph_data_loader.load()
    bipartite_graph_data_loader.plot_neighborhood_number_distribution()
