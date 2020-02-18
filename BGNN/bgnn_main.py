import argparse
import logging
import os
import random

import numpy as np
import torch
import conf
import setproctitle
from bgnn_mlp import BGNNMLP
from bgnn_adv import BGNNAdversarial
from conf import (MODEL, BATCH_SIZE, EPOCHS, LEARNING_RATE,
                  WEIGHT_DECAY, DROPOUT, HIDDEN_DIMENSIONS, GCN_OUTPUT_DIM, ENCODER_HIDDEN_DIMENSIONS,
                  DECODER_HIDDEN_DIMENSIONS, MLP_HIDDEN_DIMENSIONS, LATENT_DIMENSIONS)
from data.bipartite_graph_data_loader import BipartiteGraphDataLoader
from data.bipartite_graph_data_loader_citeseer import BipartiteGraphDataLoaderCiteseer
from data.bipartite_graph_data_loader_cora import BipartiteGraphDataLoaderCora
from data.bipartite_graph_data_loader_pubmed import BipartiteGraphDataLoaderPubMed

import calendar
import time
setproctitle.setproctitle('BGNN')
os.environ['CUDA_VISIBLE_DEVICES'] = ' '

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tencent', required=True)
    parser.add_argument('--model', type=str, default='adv', choices=MODEL, required=True)
    parser.add_argument('--seed', type=int, help='Random seed.')
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
    parser.add_argument('--gpu', type=int, default=0,
                        help='Whether to use CPU or GPU')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='batch size')
    parser.add_argument('--gcn_output_dim', type=int, default=GCN_OUTPUT_DIM,
                        help='The output dimensions of GCN.')
    parser.add_argument('--rank', type=int, default=-1,
                        help='process ID for MPI Simple AutoML')
    parser.add_argument('--encoder_hidfeat', type=int, default=ENCODER_HIDDEN_DIMENSIONS,
                        help='Number of hidden units for encoder in VAE')
    parser.add_argument('--decoder_hidfeat', type=int, default=DECODER_HIDDEN_DIMENSIONS,
                        help='Number of hidden units for mlp in VAE')
    parser.add_argument('--vae_hidfeat', type=int, default=MLP_HIDDEN_DIMENSIONS,
                        help='Number of hidden units for latent representation in VAE')
    parser.add_argument('--latent_hidfeat', type=int, default=LATENT_DIMENSIONS,
                        help='Number of latent units for latent representation in VAE')
    parser.add_argument('--layer_depth', type=int, default=3, help='Number of layer depth')
    parser.add_argument('--learning_type', type=str, default='inference', required=False)

    return parser.parse_args()


def get_the_bipartite_graph_loader(args, data_path, dataset, device):
    bipartite_graph_data_loader = None
    if dataset == "tencent":
        NODE_LIST_PATH = data_path + "data/tencent/node_list"
        NODE_ATTR_PATH = data_path + "data/tencent/node_attr"
        NODE_LABEL_PATH = data_path + "data/tencent/node_true"
        EDGE_LIST_PATH = data_path + "data/tencent/edgelist"
        GROUP_LIST_PATH = data_path + "data/tencent/group_list"
        GROUP_ATTR_PATH = data_path + "data/tencent/group_attr"

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
    elif dataset == "pubmed":
        NODE_LIST_PATH = data_path + "data/pubmed/node_list"
        NODE_ATTR_PATH = data_path + "data/pubmed/node_attr"
        NODE_LABEL_PATH = data_path + "data/pubmed/node_true"
        EDGE_LIST_PATH = data_path + "data/pubmed/edgelist"
        GROUP_LIST_PATH = data_path + "data/pubmed/group_list"
        GROUP_ATTR_PATH = data_path + "data/pubmed/group_attr"

        bipartite_graph_data_loader = BipartiteGraphDataLoaderPubMed(args.batch_size, NODE_LIST_PATH, NODE_ATTR_PATH,
                                                                     NODE_LABEL_PATH,
                                                                     EDGE_LIST_PATH,
                                                                     GROUP_LIST_PATH, GROUP_ATTR_PATH, device=device)

    return bipartite_graph_data_loader


def main():
    args = parse_args()

    dataset = args.dataset
    model_name = args.model
    rank = args.rank

    args.seed = random.randint(0, 1000000)

    logging.info("###############random seed = %s #########" % str(args.seed))

    output_folder = None
    if model_name == "adv":
        output_folder = conf.output_folder_bgnn_adv + "/" + str(dataset)
    elif model_name == "mlp":
        output_folder = conf.output_folder_bgnn_mlp + "/" + str(dataset)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    seed_file = output_folder + "/random_seed.txt"
    logging.info(seed_file)
    fs = open(seed_file, 'w')
    wstr = "%s" % str(args.seed)
    fs.write(wstr + "\n")
    fs.close()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info("device = %s" % device)

    torch.autograd.set_detect_anomaly(True)

    # load the bipartite graph data
    data_path = "./"

    bipartite_graph_data_loader = get_the_bipartite_graph_loader(args, data_path, dataset, device)
    bipartite_graph_data_loader.load()

    bgnn = BGNNAdversarial(bipartite_graph_data_loader, args, device, args.layer_depth, rank, dataset)
    if args.model == 'adv':
        # start training
        bgnn.adversarial_learning()
    elif args.model == 'mlp':
        bgnn = BGNNMLP(bipartite_graph_data_loader, args, device, rank, dataset)
        bgnn.relation_learning()


if __name__ == '__main__':
    main()
