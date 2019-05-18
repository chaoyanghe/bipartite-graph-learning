from __future__ import division
from __future__ import print_function

import logging
import os
import time

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import scipy.sparse as sp

from optimizer import OptimizerAE, OptimizerVAE

from input_data import load_data_for_tencent

from model import GCNModelAE, GCNModelVAE
from preprocessing import construct_feed_dict, sparse_to_tuple, preprocess_graph_without_tuple

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

if FLAGS.features == 0:
	features = sp.identity(features.shape[0])  # featureless

num_features = features.shape[1]
logging.info('num_features = %s' % str(num_features))

# Define placeholders
placeholders = {
	'features': tf.sparse_placeholder(tf.float32, name="features"),
	'adj': tf.sparse_placeholder(tf.float32, name="adj"),
	'adj_orig': tf.sparse_placeholder(tf.float32, name="adj_orig"),
	'dropout': tf.placeholder_with_default(0., shape=(), name="dropout"),
	'features_nonzero': tf.placeholder_with_default(0, shape=(), name="features_nonzero")
}

logging.info('create model')

# Create model
model = None

if model_str == 'gcn_ae':
	model = GCNModelAE(placeholders, num_features)
elif model_str == 'gcn_vae':
	model = GCNModelVAE(placeholders, num_features, FLAGS.batch_size)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
logging.info("pos_weight = %s" % str(pos_weight))

norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
logging.info("norm = %s" % str(norm))

# Optimizer
# optimizer should put before the global_variables_initializer()
# https://stackoverflow.com/questions/47765595/tensorflow-attempting-to-use-uninitialized-value-beta1-power?rq=1
logging.info('optimizer')
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
						   model=model, num_nodes=FLAGS.batch_size,
						   pos_weight=pos_weight,
						   norm=norm)

logging.info('initialize session')
# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


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
	logging.info("len(emb) = %s" % str(len(emb)))
	for i in range(len(emb)):
		output_node_list += str(u_list[i]) + '\n'
		embedding_output = [str(j) for j in emb[i]]
		output += '%d ' % u_list[i] + ' '.join(embedding_output) + '\n'
	with open(output_file_path + 'graphsage.emb', 'w') as file1, open(output_file_path + 'node_list', 'w') as file2:
		file1.write(output)
		file2.write(output_node_list)
	file1.close()
	file2.close()


logging.info('preprocessing data')
adj_norm_without_tuple = preprocess_graph_without_tuple(adj)
adj_label_without_tuple = adj + sp.eye(adj.shape[0])

logging.info('done preprocessing data.')


def get_feature_batch(features, adj_batch):
	adj_mask = adj_batch.sum(axis=0)
	adj_mask[adj_mask >= 1] = 1
	adj_mask = adj_mask.transpose()
	adj_mask = sp.csr_matrix(adj_mask)

	features_masked = adj_mask.multiply(features)

	features_masked = sparse_to_tuple(features_masked.tocoo())
	logging.info("features[2] = %s" % str(features_masked[2]))  # (619030, 8)
	num_features = features_masked[2][1]
	logging.info("num_features = %s" % str(num_features))  # 8
	features_nonzero = features_masked[1].shape[0]
	logging.info("features_nonzero = %s" % str(features_nonzero))  # 619030 * 8 - 4947442 = 4978
	return features_masked, num_features, features_nonzero


sparse_feature_batch, num_features, features_nonzero = get_feature_batch(features, adj[0:FLAGS.batch_size])

node_number = adj.shape[0]
batch_size = FLAGS.batch_size
batch_num = int(node_number / batch_size) + 1
logging.info("batch_num = %d" % batch_num)


logging.info('train model')
for epoch in range(FLAGS.epochs):
	t = time.time()

	hidden1 = []
	for batch_index in range(batch_num):
		start_index = batch_size * batch_index
		end_index = batch_size * (batch_index + 1)
		if batch_index == batch_num - 1:
			end_index = node_number

		adj_norm_batch = adj_norm_without_tuple[start_index:end_index]
		adj_norm_batch = sparse_to_tuple(adj_norm_batch)

		adj_label_batch = adj_label_without_tuple[start_index:FLAGS.end_index]
		adj_label_batch = sparse_to_tuple(adj_label_batch)

		feed_dict = construct_feed_dict(adj_norm_batch, adj_label_batch, sparse_feature_batch,
										features_nonzero, placeholders)
		logging.info('update')
		feed_dict.update({placeholders['dropout']: FLAGS.dropout})

		# Run a batch to get the hidden1 batch vectors
		logging.info('run the model')
		hidden1_batch = sess.run([model.hidden1], feed_dict=feed_dict)
		hidden1.append(hidden1_batch)

	logging.info("outs = %s" % outs)

	# outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

	logging.info('average loss')
	# Compute average loss
	avg_cost = outs[1]
	avg_accuracy = outs[2]

	print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
		  "train_acc=", "{:.5f}".format(avg_accuracy),
		  "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")

logging.info("generate embedding file...")
get_emb(vae=True)
logging.info("end.")
