from __future__ import division
from __future__ import print_function

import time
import os
import logging

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from optimizer import OptimizerAE, OptimizerVAE

from input_data import load_data, load_data_for_tencent

from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

logging.basicConfig(filename="./gae.log",
                    level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('batch_size', 512, 'The minibatch size')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Load data
adj, features, u_list = load_data_for_tencent(FLAGS, 'cpu')  # u_list is the hash table

# Store original adjacency matrix (without diagonal entries) for later
# adj_orig = adj
# adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
# adj_orig.eliminate_zeros()

adj_train = adj

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

logging.info('preprocessing data')
# Some preprocessing
adj_norm = preprocess_graph(adj)
logging.info('done preprocessing data')

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj.shape[0]

# feature statistics
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

logging.info('create model')

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
logging.info('optimizer')
# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)
logging.info('initialize session')
# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []


# write the embedding to file
def get_emb(vae=True):
    feed_dict.update({placeholders['dropout']: 0})
    if vae:
        emb = sess.run(model.z, feed_dict=feed_dict)
    else:
        emb = sess.run(model.z_mean, feed_dict=feed_dict)
    # TODO: check the data type of emb
    output = ''
    output_node_list = ''
    output += '%d %d' % (len(u_list), len(emb[0])) + '\n'
    output_file_path = './out/'
    for i in range(len(emb)):
        output_node_list += str(u_list[i]) + '\n'
        embedding_output = [str(j) for j in emb[i]]
        output += '%d ' % u_list[i] + ' '.join(embedding_output) + '\n'
    with open(output_file_path + 'graphsage.emb', 'w') as file1, open(output_file_path + 'node_list', 'w') as file2:
        file1.write(output)
        file2.write(output_node_list)
    file1.close()
    file2.close()

#
# def get_roc_score(edges_pos, edges_neg, emb=None):
#     if emb is None:
#         feed_dict.update({placeholders['dropout']: 0})
#         emb = sess.run(model.z_mean, feed_dict=feed_dict)
#
#     def sigmoid(x):
#         return 1 / (1 + np.exp(-x))
#
#     # Predict on test set of edges
#     adj_rec = np.dot(emb, emb.T)
#     preds = []
#     pos = []
#     for e in edges_pos:
#         preds.append(sigmoid(adj_rec[e[0], e[1]]))
#         pos.append(adj_orig[e[0], e[1]])
#
#     preds_neg = []
#     neg = []
#     for e in edges_neg:
#         preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
#         neg.append(adj_orig[e[0], e[1]])
#
#     preds_all = np.hstack([preds, preds_neg])
#     labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
#     roc_score = roc_auc_score(labels_all, preds_all)
#     ap_score = average_precision_score(labels_all, preds_all)
#
#     return roc_score, ap_score
#

# cost_val = []
# acc_val = []
val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)
logging.info('train model')
# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()
    logging.info('construct dictionary')
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    logging.info('update')
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update
    logging.info('run the model')
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

    logging.info('average loss')
    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]
    get_emb(vae=True)
    # roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    # val_roc_score.append(roc_curr)

    # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
    #       "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
    #       "val_ap=", "{:.5f}".format(ap_curr),
    #       "time=", "{:.5f}".format(time.time() - t))
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")

# roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
# print('Test ROC score: ' + str(roc_score))
# print('Test AP score: ' + str(ap_score))
