import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn.layers import GraphConvolution


# one layer GCN
class GCN(nn.Module):
    def __init__(self, infeat, outfeat):
        super(GCN, self).__init__()

        # hidden_dimension1 = 32
        # hidden_dimension2 = 24
        self.gc1 = GraphConvolution(infeat, outfeat)
        # self.gc2 = GraphConvolution(hidfeat, outfeat)
        # self.dropout = dropout
        # self.fc1 = nn.Linear(hidden_dimension1, hidden_dimension2)
        # self.fc2 = nn.Linear(hidden_dimension2, outfeat)

    def forward(self, x, adj):
        # follow the best practice here:
        # https://github.com/soumith/talks/blob/master/2017-ICCV_Venice/How_To_Train_a_GAN.pdf
        x = torch.tanh(self.gc1(x, adj))
        # x = self.fc1(x)
        #
        # x = F.leaky_relu(x)
        # x = self.fc2(x)
        #
        # # follow the best practice here:
        # # https://github.com/soumith/talks/blob/master/2017-ICCV_Venice/How_To_Train_a_GAN.pdf
        # x = torch.tanh(x)

        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)

        return x

    def aggregation(self, x, adj):
        x = self.gc1(x, adj)
        return x


class GCNForVAE(nn.Module):
    def __init__(self, infeat, outfeat):
        super(GCNForVAE, self).__init__()

        hidden_dimension2 = 100
        self.gc1 = GraphConvolution(infeat, outfeat)
        # self.gc2 = GraphConvolution(hidfeat, outfeat)
        # self.dropout = dropout
        self.fc1 = nn.Linear(outfeat, hidden_dimension2)
        self.fc2 = nn.Linear(hidden_dimension2, outfeat)

    def forward(self, x, adj):
        # follow the best practice here:
        # https://github.com/soumith/talks/blob/master/2017-ICCV_Venice/How_To_Train_a_GAN.pdf
        x = self.gc1(x, adj)
        x = self.fc1(x)

        x = F.leaky_relu(x)
        x = self.fc2(x)

        # follow the best practice here:
        # https://github.com/soumith/talks/blob/master/2017-ICCV_Venice/How_To_Train_a_GAN.pdf
        x = torch.tanh(x)

        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)

        return x

    def aggregation(self, x, adj):
        x = self.gc1(x, adj)
        return x
