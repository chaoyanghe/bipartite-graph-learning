import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
	if not sp.isspmatrix_coo(sparse_mx):
		sparse_mx = sparse_mx.tocoo()
	coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
	values = sparse_mx.data
	shape = sparse_mx.shape
	return coords, values, shape


adj_batch = sp.csr_matrix([[0, 2, 0, 0], [0, 5, 0, 0], [3, 0, 0, 0], [0, 0, 4, 0]])
adj_label = adj_batch + sp.eye(adj_batch.shape[0])
adj_label = sparse_to_tuple(adj_label)
print(adj_label)

adj_label = adj_label[0:2]
print("adj_label = %s" % str(adj_label))
#
# adj_mask = adj_batch.sum(axis=0)
# print("adj_mask = %s" % adj_mask)
# adj_mask[adj_mask >= 1] = 1
# print("adj_mask = %s" % adj_mask)
# adj_mask = adj_mask.transpose()
# print("adj_mask = %s" % adj_mask)
# adj_mask = sp.csr_matrix(adj_mask)
# print("adj_mask = %s" % adj_mask)
#
#
#
# features = sp.csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
#
# features_masked = adj_mask.multiply(features)
# print("features_masked = %s" % features_masked.toarray())
