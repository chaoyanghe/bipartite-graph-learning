import argparse
import torch
from train import HeterogeneousGCN
from data.utils import load_data
from utils import (MODEL, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DROPOUT)


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
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=DROPOUT,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--gpu', type=bool, default=False,
                        help='Whether to use CPU or GPU')

    return parser.parse_args()


def main():
    args = parse_args()

    adjU, adjV, featuresU, featuresV = load_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    hgcn = HeterogeneousGCN(adjU, adjV, featuresU, featuresV, device)
    if args.model == 'gan_gcn':
        hgcn.adversarial_train()
    else:
        hgcn.decoder_train()


if __name__ == '__main__':
    main()
