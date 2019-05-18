from __future__ import division
from __future__ import print_function

import logging
import os
import time

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from optimizer import OptimizerAE, OptimizerVAE

from input_data import load_data_for_cora

from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple

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
adj, features, u_list = load_data_for_cora(FLAGS, 'cpu')  # u_list is the hash table

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

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

	print("Epoch: %d, train_loss = %s, train_acc = %s, time cost = %s" % (
	epoch, str(avg_cost), str(avg_accuracy), str(time.time() - t)))

vae = True
# write the embedding to file
feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
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
with open(output_file_path + 'gae.emb', 'w') as file1, open(output_file_path + 'node_list', 'w') as file2:
	file1.write(output)
	file2.write(output_node_list)
file1.close()
file2.close()
print("Optimization Finished!")
