import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import logging
import os
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from scipy.sparse.linalg import norm as sparsenorm
from scipy.linalg import qr
from sparse_tensor_utils import *
import json
from networkx.readwrite import json_graph


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


#
# def calc_f1(y_true, y_pred):
#     y_true = np.argmax(y_true, axis=1)
#     y_pred = np.argmax(y_pred, axis=1)
#     return f1_score(y_true, y_pred, average="micro"), f1_score(y_true, y_pred, average="macro")
#

#
# def load_data(dataset_str):
#     """Load data."""
#     names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
#     objects = []
#     for i in range(len(names)):
#         with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))
#             else:
#                 objects.append(pkl.load(f))
#
#     x, y, tx, ty, allx, ally, graph = tuple(objects)
#     test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
#     test_idx_range = np.sort(test_idx_reorder)
#
#     if dataset_str == 'citeseer':
#         # Fix citeseer dataset (there are some isolated nodes in the graph)
#         # Find isolated nodes, add them as zero-vecs into the right position
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range-min(test_idx_range), :] = tx
#         tx = tx_extended
#         ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
#         ty_extended[test_idx_range-min(test_idx_range), :] = ty
#         ty = ty_extended
#
#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
#
#     labels = np.vstack((ally, ty))
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]
#
#     idx_test = test_idx_range.tolist()
#     idx_train = range(len(y))
#     idx_val = range(len(y), len(y)+500)
#
#     train_mask = sample_mask(idx_train, labels.shape[0])
#     val_mask = sample_mask(idx_val, labels.shape[0])
#     test_mask = sample_mask(idx_test, labels.shape[0])
#
#     y_train = np.zeros(labels.shape)
#     y_val = np.zeros(labels.shape)
#     y_test = np.zeros(labels.shape)
#     y_train[train_mask, :] = labels[train_mask, :]
#     y_val[val_mask, :] = labels[val_mask, :]
#     y_test[test_mask, :] = labels[test_mask, :]
#
#     return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
#


def load_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        logging.info('*******Name: {}'.format(names[i]))
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    # add for tencent
    logging.info('********test.index')
    if dataset_str == 'tencent':
        with open('data/ind.{}.test.index'.format(dataset_str), 'rb') as f:
            test_idx_reorder = pkl.load(f)
    else:
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
    logging.info('*******all loaded')
    features = sp.vstack((allx, tx)).tolil()  # stack 'allx' and 'tx', then convert to linked list format
    features[test_idx_reorder, :] = features[test_idx_range, :]  # must follow the same order as train : test
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  # this is the reason why the code is slow!!!!

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    logging.info('*******test preparation')
    idx_test = test_idx_range.tolist()
    idx_train = range(len(ally) - 500)
    idx_val = range(len(ally) - 500, len(ally))
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y)+500)
    logging.info('********mask preparation')
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    logging.info('********y preparation')
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    logging.info('********load data function done')
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

# save the training output for asgcn
def save_embedding_to_file(gcn_merge_output, node_id_list):
    """ embedding file format:
        line1: number of the node, dimension of the embedding vector
        line2: node_id, embedding vector
        line3: ...
        lineN: node_id, embedding vector

    :param gcn_merge_output:
    :param node_id_list:
    :return:
    """
    logging.info("Start to save embedding file")
    # print(gcn_merge_output)
    node_num = gcn_merge_output.shape[0]
    logging.info("node_num = %s" % node_num)
    dimension_embedding = gcn_merge_output.shape[1]
    logging.info("dimension_embedding = %s" % dimension_embedding)
    output_folder = "./out/abcgraph-adv/" + str(dataset)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    f_emb = open(output_folder + '/abcgraph.emb', 'w')
    f_node_list = open(output_folder + '/node_list', 'w')

    str_first_line = str(node_num) + " " + str(dimension_embedding) + "\n"
    f_emb.write(str_first_line)
    for n_idx in range(node_num):
        f_emb.write(str(node_id_list[n_idx]) + ' ')
        f_node_list.write(str(node_id_list[n_idx]))
        emb_vec = gcn_merge_output[n_idx]
        for d_idx in range(dimension_embedding):
            if d_idx != dimension_embedding - 1:
                f_emb.write(str(emb_vec[d_idx]) + ' ')
            else:
                f_emb.write(str(emb_vec[d_idx]))
        if n_idx != node_num - 1:
            f_emb.write('\n')
            f_node_list.write('\n')
    f_emb.close()
    f_node_list.close()
    logging.info("Saved embedding file")


def load_data_original(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def nontuple_preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def nontuple_preprocess_adj(adj):
    adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)
    # adj_normalized = sp.eye(adj.shape[0]) + normalize_adj(adj)
    return adj_normalized.tocsr()


def column_prop(adj):
    # column_norm = sparsenorm(adj, ord=1, axis=0)
    column_norm = sparsenorm(adj, axis=0)
    # column_norm = pow(sparsenorm(adj, axis=0),2)
    norm_sum = sum(column_norm)
    return column_norm / norm_sum


def mix_prop(adj, features, sparseinputs=False):
    adj_column_norm = sparsenorm(adj, axis=0)
    if sparseinputs:
        features_row_norm = sparsenorm(features, axis=1)
    else:
        features_row_norm = np.linalg.norm(features, axis=1)
    mix_norm = adj_column_norm * features_row_norm

    norm_sum = sum(mix_norm)
    return mix_norm / norm_sum


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # adj_appr = np.array(sp.csr_matrix.todense(adj))
    # # adj_appr = dense_lanczos(adj_appr, 100)
    # adj_appr = dense_RandomSVD(adj_appr, 100)
    # if adj_appr.sum(1).min()<0:
    #     adj_appr = adj_appr- (adj_appr.sum(1).min()-0.5)*sp.eye(adj_appr.shape[0])
    # else:
    #     adj_appr = adj_appr + sp.eye(adj_appr.shape[0])
    # adj_normalized = normalize_adj(adj_appr)

    # adj_normalized = normalize_adj(adj+sp.eye(adj.shape[0]))
    # adj_appr = np.array(sp.coo_matrix.todense(adj_normalized))
    # # adj_normalized = dense_RandomSVD(adj_appr,100)
    # adj_normalized = dense_lanczos(adj_appr, 100)

    adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)
    # adj_normalized = sp.eye(adj.shape[0]) + normalize_adj(adj)
    return sparse_to_tuple(adj_normalized)


from lanczos import lanczos


def dense_lanczos(A, K):
    q = np.random.randn(A.shape[0], )
    Q, sigma = lanczos(A, K, q)
    A2 = np.dot(Q[:, :K], np.dot(sigma[:K, :K], Q[:, :K].T))
    return sp.csr_matrix(A2)


def sparse_lanczos(A, k):
    q = sp.random(A.shape[0], 1)
    n = A.shape[0]
    Q = sp.lil_matrix(np.zeros((n, k + 1)))
    A = sp.lil_matrix(A)

    Q[:, 0] = q / sparsenorm(q)

    alpha = 0
    beta = 0

    for i in range(k):
        if i == 0:
            q = A * Q[:, i]
        else:
            q = A * Q[:, i] - beta * Q[:, i - 1]
        alpha = q.T * Q[:, i]
        q = q - Q[:, i] * alpha
        q = q - Q[:, :i] * Q[:, :i].T * q  # full reorthogonalization
        beta = sparsenorm(q)
        Q[:, i + 1] = q / beta
        print(i)

    Q = Q[:, :k]

    Sigma = Q.T * A * Q
    A2 = Q[:, :k] * Sigma[:k, :k] * Q[:, :k].T
    return A2
    # return Q, Sigma


def dense_RandomSVD(A, K):
    G = np.random.randn(A.shape[0], K)
    B = np.dot(A, G)
    Q, R = qr(B, mode='economic')
    M = np.dot(Q, np.dot(Q.T, A))
    return sp.csr_matrix(M)


def construct_feed_dict(features, supports, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: supports[i] for i in range(len(supports))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def construct_feed_dict_with_prob(features_inputs, supports, probs, labels, labels_mask, placeholders):
    """Construct feed dictionary with adding sampling prob."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features_inputs'][i]: features_inputs[i] for i in range(len(features_inputs))})
    feed_dict.update({placeholders['support'][i]: supports[i] for i in range(len(supports))})
    feed_dict.update({placeholders['prob'][i]: probs[i] for i in range(len(probs))})
    # feed_dict.update({placeholders['prob_norm'][i]: probs_norm[i] for i in range(len(probs_norm))})
    feed_dict.update({placeholders['num_features_nonzero']: features_inputs[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def iterate_minibatches_listinputs(inputs, batchsize, shuffle=False):
    assert inputs is not None
    numSamples = inputs[0].shape[0]
    indices = np.arange(numSamples)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, numSamples - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [input[excerpt] for input in inputs]


def loadRedditFromG(dataset_dir, inputfile):
    f = open(dataset_dir + inputfile)
    objects = []
    for _ in range(pkl.load(f)):
        objects.append(pkl.load(f))
    adj, train_labels, val_labels, test_labels, train_index, val_index, test_index = tuple(objects)
    feats = np.load(dataset_dir + "/reddit-feats.npy")
    return sp.csr_matrix(adj), sp.lil_matrix(
        feats), train_labels, val_labels, test_labels, train_index, val_index, test_index


def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir + "_adj.npz")
    data = np.load(dataset_dir + ".npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], \
           data['test_index']


def transferRedditDataFormat(dataset_dir, output_file):
    G = json_graph.node_link_graph(json.load(open(dataset_dir + "-G.json")))
    labels = json.load(open(dataset_dir + "-class_map.json"))

    train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids = [n for n in G.nodes() if G.node[n]['test']]
    val_ids = [n for n in G.nodes() if G.node[n]['val']]
    train_labels = [labels[i] for i in train_ids]
    test_labels = [labels[i] for i in test_ids]
    val_labels = [labels[i] for i in val_ids]
    feats = np.load(dataset_dir + "-feats.npy")
    ## Logistic gets thrown off by big counts, so log transform num comments and score
    feats[:, 0] = np.log(feats[:, 0] + 1.0)
    feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))
    feat_id_map = json.load(open(dataset_dir + "-id_map.json"))
    feat_id_map = {id: val for id, val in feat_id_map.iteritems()}

    # train_feats = feats[[feat_id_map[id] for id in train_ids]]
    # test_feats = feats[[feat_id_map[id] for id in test_ids]]

    # numNode = len(feat_id_map)
    # adj = sp.lil_matrix(np.zeros((numNode,numNode)))
    # for edge in G.edges():
    #     adj[feat_id_map[edge[0]], feat_id_map[edge[1]]] = 1

    train_index = [feat_id_map[id] for id in train_ids]
    val_index = [feat_id_map[id] for id in val_ids]
    test_index = [feat_id_map[id] for id in test_ids]
    np.savez(output_file, feats=feats, y_train=train_labels, y_val=val_labels, y_test=test_labels,
             train_index=train_index,
             val_index=val_index, test_index=test_index)


def transferLabel2Onehot(labels, N):
    y = np.zeros((len(labels), N))
    for i in range(len(labels)):
        pos = labels[i]
        y[i, pos] = 1
    return y


def construct_feeddict_forMixlayers(AXfeatures, support, labels, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['AXfeatures']: AXfeatures})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['num_features_nonzero']: AXfeatures[1].shape})
    return feed_dict


def prepare_pubmed(dataset, max_degree):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)
    # features dimensions (allx + tx, features) -- sparse
    logging.info('********data loaded!')
    train_index = np.where(train_mask)[0]
    adj_train = adj[train_index, :][:, train_index]
    y_train = y_train[train_index]
    val_index = np.where(val_mask)[0]
    y_val = y_val[val_index]
    test_index = np.where(test_mask)[0]
    y_test = y_test[test_index]

    num_train = adj_train.shape[0]
    input_dim = features.shape[1]

    # the features becomes a dense matrix
    features = nontuple_preprocess_features(features).todense()
    train_features = features[train_index]

    norm_adj_train = nontuple_preprocess_adj(adj_train)
    norm_adj = nontuple_preprocess_adj(adj)

    if dataset == 'pubmed':
        norm_adj = 1 * sp.diags(np.ones(norm_adj.shape[0])) + norm_adj
        norm_adj_train = 1 * sp.diags(np.ones(num_train)) + norm_adj_train

    # adj_train, adj_val_train = norm_adj_train, norm_adj_train
    adj_train, adj_val_train = compute_adjlist(norm_adj_train, max_degree)
    train_features = np.concatenate((train_features, np.zeros((1, input_dim))))

    return norm_adj, adj_train, adj_val_train, features, train_features, y_train, y_test, test_index


def prepare_reddit(max_degree):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/reddit")
    adj = adj + adj.T

    y_train = transferLabel2Onehot(y_train, 41)
    y_val = transferLabel2Onehot(y_val, 41)
    y_test = transferLabel2Onehot(y_test, 41)

    features = sp.lil_matrix(features)
    adj_train = adj[train_index, :][:, train_index]
    num_train = adj_train.shape[0]
    input_dim = features.shape[1]

    mask = []

    norm_adj_train = nontuple_preprocess_adj(adj_train)
    norm_adj = nontuple_preprocess_adj(adj)

    # Some preprocessing
    features = nontuple_preprocess_features(features).todense()
    train_features = norm_adj_train.dot(features[train_index])
    features = norm_adj.dot(features)

    adj_train, adj_val_train = compute_adjlist(norm_adj_train, max_degree)
    train_features = np.concatenate((train_features, np.zeros((1, input_dim))))

    return norm_adj, adj_train, adj_val_train, features, train_features, y_train, y_test, test_index
