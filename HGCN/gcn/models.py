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
        x = F.relu(self.gc1(x, adj))
        x = self.fc1(x)

        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)

        return x

    def aggregation(self, x, adj):
        x = self.gc1(x, adj)
        return x
