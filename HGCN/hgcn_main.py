import argparse
import logging

import torch

from data.bipartite_graph_data_loader import BipartiteGraphDataLoader
from train import HeterogeneousGCN
from utils import (MODEL, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DROPOUT, HIDDEN_DIMENSIONS)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='gan_gcn', choices=MODEL, required=True)
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=HIDDEN_DIMENSIONS,
                        help='Number of hidden units in GCN.')
    parser.add_argument('--dropout', type=float, default=DROPOUT,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--gpu', type=bool, default=False,
                        help='Whether to use CPU or GPU')

    return parser.parse_args()


def main():
    args = parse_args()
    # hidden_dimensions = args.hidden
    # dropout = args.dropout

    logging.basicConfig(filename="./HGCN.log",
                        level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    NODE_LIST_PATH = "./../data/Tencent-QQ/node_list"
    NODE_ATTR_PATH = "./../data/Tencent-QQ/node_attr"
    NODE_LABEL_PATH = "./../data/Tencent-QQ/node_true"

    EDGE_LIST_PATH = "./../data/Tencent-QQ/edgelist"

    GROUP_LIST_PATH = "./../data/Tencent-QQ/group_list"
    GROUP_ATTR_PATH = "./../data/Tencent-QQ/group_attr"
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info("device = %s" % device)
    bipartite_graph_data_loader = BipartiteGraphDataLoader(100, NODE_LIST_PATH, NODE_ATTR_PATH, NODE_LABEL_PATH,
                                                           EDGE_LIST_PATH,
                                                           GROUP_LIST_PATH, GROUP_ATTR_PATH, device=device)
    bipartite_graph_data_loader.load()

    hgcn = HeterogeneousGCN(bipartite_graph_data_loader, device)
    if args.model == 'gan_gcn':
        hgcn.adversarial_train()
    else:
        hgcn.decoder_train()


if __name__ == '__main__':
    main()
