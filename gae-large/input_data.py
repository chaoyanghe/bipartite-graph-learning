import logging
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp

from data.bipartite_graph_data_loader_tencent import BipartiteGraphDataLoader


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

	return adj, features  # features csr_matrix.tolil()


def __BipartiteToSingle(graph):
	"""
	transfer the bipartite graph to single graph
	:param graph: sparse csr_matrix
	:return: sparse adjacent csr_matrix
	"""
	single_graph = graph.dot(graph.T)
	single_graph[single_graph != 0] = 1
	single_graph -= sp.identity(graph.shape[0])
	return single_graph


def load_data_for_tencent(flags, device):
	NODE_LIST_PATH = "./../data/Tencent-QQ/node_list"
	NODE_ATTR_PATH = "./../data/Tencent-QQ/node_attr"
	NODE_LABEL_PATH = "./../data/Tencent-QQ/node_true"
	EDGE_LIST_PATH = "./../data/Tencent-QQ/edgelist"
	GROUP_LIST_PATH = "./../data/Tencent-QQ/group_list"
	GROUP_ATTR_PATH = "./../data/Tencent-QQ/group_attr"

	batch_size = flags.batch_size
	bipartite_graph_data_loader = BipartiteGraphDataLoader(batch_size, NODE_LIST_PATH, NODE_ATTR_PATH,
														   NODE_LABEL_PATH,
														   EDGE_LIST_PATH,
														   GROUP_LIST_PATH, GROUP_ATTR_PATH, device=device)
	bipartite_graph_data_loader.load()
	logging.info('####### Done bipartite graph loader #########')
	u_attr = bipartite_graph_data_loader.get_u_attr_array()  # list
	u_adj = bipartite_graph_data_loader.get_u_adj()  # csr_matrix
	u_list = bipartite_graph_data_loader.get_u_list()  # line indice -- node id

	logging.info("u_attr len = %d" % len(u_attr))
	logging.info("u_adj shape = %s" % str(u_adj.shape))
	logging.info("u_list len = %d" % len(u_list))

	# # # test the code
	# row_num = 20000 # 619030
	# col_num = 10000 # 90044
	#
	# logging.info("1")
	# u_list = [i for i in range(row_num)]
	# logging.info("2")
	# u_attr = np.random.rand(row_num, 8).tolist()
	# logging.info("3")
	# #u_adj = sp.csr_matrix([np.random.randint(2, size=90044) for i in range(619030)])
	# #u_adj = [np.random.randint(2, size=90044) for i in range(619030)]
	# u_adj = sp.csr_matrix((row_num, col_num), dtype=np.int8)
	# logging.info("u_adj shape = %s" % str(u_adj.shape))
	# logging.info("4")

	#edge_list = [(1, 1), (1, 3), (1, 6), (3, 3), (5, 1), (5, 6), (7, 8), (9, 6)]
	#node_true = [1, 5, 9]

	adj = __BipartiteToSingle(u_adj)
	adj_batch = adj[0:10]
	logging.info("adj_batch.shape = %s" % str(adj_batch.shape))

	features = sp.csr_matrix(u_attr).tolil()
	logging.info("len(features) = %s" % str(features.shape[0]))

	logging.info('######### Data loaded ###########')
	return adj, features, u_list
