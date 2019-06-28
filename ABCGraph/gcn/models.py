import torch
import torch.nn as nn
from gcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, infeat, outfeat):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(infeat, outfeat)

    def forward(self, x, adj):
        # follow the best practice here:
        # https://github.com/soumith/talks/blob/master/2017-ICCV_Venice/How_To_Train_a_GAN.pdf
        x = torch.tanh(self.gc1(x, adj))
        return x

    def aggregation(self, x, adj):
        x = self.gc1(x, adj)
        return x
