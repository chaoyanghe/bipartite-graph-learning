import argparse
import logging

import torch

from cascaded_adversarial_hgcn_with_gan import CascadedAdversarialHGCN
from conf import (MODEL, RANDOM_SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DROPOUT, HIDDEN_DIMENSIONS)
from data.bipartite_graph_data_loader import BipartiteGraphDataLoader


def parse_args():
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--rank', type=int, default=-1,
                        help='process ID for MPI Simple AutoML')

    return parser.parse_args()


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

    # log configuration
    logging.basicConfig(filename="./HGCN.log",
                        level=logging.DEBUG,
                        format=str(rank) + '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    # initialization
    # np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info("device = %s" % device)
    # torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.gpu:
        torch.cuda.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    # load the bipartite graph data
    data_path = "./"
    if rank != -1:
        data_path = "/mnt/shared/home/bipartite-graph-learning/"

    print(data_path)
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
    bipartite_graph_data_loader.load()

    # start the adversarial learning (output the embedding result into ./out directory)
    hgcn = CascadedAdversarialHGCN(bipartite_graph_data_loader, args, device)
    if args.model == 'gan_gcn':
        # start training
        hgcn.adversarial_learning()


if __name__ == '__main__':
    main()
