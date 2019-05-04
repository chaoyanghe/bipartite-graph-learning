import torch.nn as nn
import torch.nn.functional as F

from gcn.layers import GraphConvolution


# one layer GCN
class GCN(nn.Module):
    def __init__(self, infeat, outfeat):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(infeat, outfeat)
        # self.gc2 = GraphConvolution(hidfeat, outfeat)
        # self.dropout = dropout

    def forward(self, x, adj):
        x = F.leaky_relu(self.gc1(x, adj), inplace=True)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)

        return x
