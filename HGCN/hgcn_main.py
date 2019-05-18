import argparse
import logging
from collections import namedtuple

import numpy as np
import torch

from cascaded_adversarial_hgcn_with_decoder import DecoderGCNLayer
from cascaded_adversarial_hgcn_with_gan import CascadedAdversarialHGCN
from conf import (MODEL, RANDOM_SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE,
				  WEIGHT_DECAY, DROPOUT, HIDDEN_DIMENSIONS, GCN_OUTPUT_DIM)
from data.bipartite_graph_data_loader import BipartiteGraphDataLoader
from data.bipartite_graph_data_loader_citeseer import BipartiteGraphDataLoaderCiteseer
from data.bipartite_graph_data_loader_cora import BipartiteGraphDataLoaderCora
from variational_hgcn import VariationalGCNLayer


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='tencent', required=True)
	parser.add_argument('--model', type=str, default='gan_gcn', choices=MODEL, required=True)
	parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed.')
	parser.add_argument('--epochs', type=int, default=EPOCHS,
						help='Number of epochs to train.')
	parser.add_argument('--lr', type=float, default=LEARNING_RATE,
						help='Initial learning rate.')
	parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY,
						help='Weight decay (L2 loss on parameters).')
	parser.add_argument('--dis_hidden', type=int, default=HIDDEN_DIMENSIONS,
						help='Number of hidden units for discriminator in GAN model.')
	parser.add_argument('--dropout', type=float, default=DROPOUT,
						help='Dropout rate (1 - keep probability).')
	parser.add_argument('--gpu', type=bool, default=False,
						help='Whether to use CPU or GPU')
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
						help='batch size')
	parser.add_argument('--gcn_output_dim', type=int, default=GCN_OUTPUT_DIM,
						help='The output dimensions of GCN.')
	parser.add_argument('--rank', type=int, default=-1,
						help='process ID for MPI Simple AutoML')
	# parser.add_argument('--encoder_hidfeat', type=int, default=ENCODER_HIDDEN_DIMENSIONS,
	# 					help='Number of hidden units for encoder in VAE')
	# parser.add_argument('--decoder_hidfeat', type=int, default=DECODER_HIDDEN_DIMENSIONS,
	# 					help='Number of hidden units for decoder in VAE')
	# parser.add_argument('--vae_hidfeat', type=int, default=VAE_HIDDEN_DIMENSIONS,
	# 					help='Number of hidden units for latent representation in VAE')
	# parser.add_argument('--latent_hidfeat', type=int, default=LATENT_DIMENSIONS,
	#                     help='Number of latent units for latent representation in VAE')

	return parser.parse_args()


def get_the_bipartite_graph_loader(args, data_path, dataset, device):
	print(data_path)
	bipartite_graph_data_loader = None
	if dataset == "tencent":
		NODE_LIST_PATH = data_path + "data/Tencent-QQ/node_list"
		NODE_ATTR_PATH = data_path + "data/Tencent-QQ/node_attr"
		NODE_LABEL_PATH = data_path + "data/Tencent-QQ/node_true"
		EDGE_LIST_PATH = data_path + "data/Tencent-QQ/edgelist"
		GROUP_LIST_PATH = data_path + "data/Tencent-QQ/group_list"
		GROUP_ATTR_PATH = data_path + "data/Tencent-QQ/group_attr"

		bipartite_graph_data_loader = BipartiteGraphDataLoader(args.batch_size, NODE_LIST_PATH, NODE_ATTR_PATH,
															   NODE_LABEL_PATH,
															   EDGE_LIST_PATH,
															   GROUP_LIST_PATH, GROUP_ATTR_PATH, device=device)
	elif dataset == "cora":
		NODE_LIST_PATH = data_path + "data/cora/node_list"
		NODE_ATTR_PATH = data_path + "data/cora/node_attr"
		NODE_LABEL_PATH = data_path + "data/cora/node_true"
		EDGE_LIST_PATH = data_path + "data/cora/edgelist"
		GROUP_LIST_PATH = data_path + "data/cora/group_list"
		GROUP_ATTR_PATH = data_path + "data/cora/group_attr"

		bipartite_graph_data_loader = BipartiteGraphDataLoaderCora(args.batch_size, NODE_LIST_PATH, NODE_ATTR_PATH,
																   NODE_LABEL_PATH,
																   EDGE_LIST_PATH,
																   GROUP_LIST_PATH, GROUP_ATTR_PATH, device=device)
	elif dataset == "citeseer":
		NODE_LIST_PATH = data_path + "data/citeseer/node_list"
		NODE_ATTR_PATH = data_path + "data/citeseer/node_attr"
		NODE_LABEL_PATH = data_path + "data/citeseer/node_true"
		EDGE_LIST_PATH = data_path + "data/citeseer/edgelist"
		GROUP_LIST_PATH = data_path + "data/citeseer/group_list"
		GROUP_ATTR_PATH = data_path + "data/citeseer/group_attr"

		bipartite_graph_data_loader = BipartiteGraphDataLoaderCiteseer(args.batch_size, NODE_LIST_PATH, NODE_ATTR_PATH,
																   NODE_LABEL_PATH,
																   EDGE_LIST_PATH,
																   GROUP_LIST_PATH, GROUP_ATTR_PATH, device=device)

	return bipartite_graph_data_loader


def main():
	args = parse_args()
	rank = args.rank
	print("batch_size = " + str(args.batch_size))
	print("epochs = " + str(args.epochs))
	print("lr = " + str(args.lr))
	print("weight_decay = " + str(args.weight_decay))
	print("dis_hidden = " + str(args.dis_hidden))
	print("dropout = " + str(args.dropout))
	print("rank = " + str(rank))

	# log_embedding configuration
	logging.basicConfig(filename="./HGCN.log_embedding",
						level=logging.DEBUG,
						format=str(rank) + '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
						datefmt='%a, %d %b %Y %H:%M:%S')

	# initialization
	# https://pytorch.org/docs/stable/notes/randomness.html
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
	logging.info("device = %s" % device)

	torch.autograd.set_detect_anomaly(True)

	# load the bipartite graph data
	data_path = "./"
	if rank != -1:
		data_path = "/mnt/shared/home/bipartite-graph-learning/"

	dataset = args.dataset
	bipartite_graph_data_loader = get_the_bipartite_graph_loader(args, data_path, dataset, device)
	bipartite_graph_data_loader.load()

	# start the adversarial learning (output the embedding result into ./out directory)
	hgcn = CascadedAdversarialHGCN(bipartite_graph_data_loader, args, device, rank)
	print("hgcn = %s" % str(hgcn))
	if args.model == 'gan_gcn':
		# start training
		print("adversarial_learning START")
		hgcn.adversarial_learning()
		print("adversarial_learning END")
	elif args.model == 'decoder_gcn':
		hgcn = DecoderGCNLayer(bipartite_graph_data_loader, args, device, rank)
		hgcn.relation_learning()
	elif args.model == 'vae_gcn':
		layerInfo = namedtuple('LayerInfo', ['vae_hidfeat', 'gcn1_input_dim', 'gcn1_output_dim', 'gcn2_input_dim',
											 'gcn2_output_dim'])
		hgcn = VariationalGCNLayer(bipartite_graph_data_loader, args, device, layerInfo, rank)


if __name__ == '__main__':
	main()
