import torch.nn as nn
import torch.nn.functional as F

from gcn.layers import GraphConvolution


# one layer GCN
class GCN(nn.Module):
    def __init__(self, infeat, outfeat):
        super(GCN, self).__init__()

        hidden_dimension = 24
        self.gc1 = GraphConvolution(infeat, hidden_dimension)
        # self.gc2 = GraphConvolution(hidfeat, outfeat)
        # self.dropout = dropout
        self.fc1 = nn.Linear(hidden_dimension, outfeat)

    def forward(self, x, adj):
        # follow the best practice here:
        # https://github.com/soumith/talks/blob/master/2017-ICCV_Venice/How_To_Train_a_GAN.pdf
        x = F.leaky_relu(self.gc1(x, adj))
        x = self.fc1(x)

        # follow the best practice here:
        # https://github.com/soumith/talks/blob/master/2017-ICCV_Venice/How_To_Train_a_GAN.pdf
        x = F.tanh(x)

        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)

        return x

    def aggregation(self, x, adj):
        x = self.gc1(x, adj)
        return x
