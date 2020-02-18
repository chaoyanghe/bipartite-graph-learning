import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, infeat, outfeat):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(infeat, outfeat)

    def forward(self, x, adj):
        x = torch.tanh(self.gc1(x, adj))
        return x

    def aggregation(self, x, adj):
        x = self.gc1(x, adj)
        return x

class twoLayersGCN(nn.Module):
    def __init__(self, infeat, outfeat):
        super(twoLayersGCN, self).__init__()
        self.gc1 = GraphConvolution(infeat, outfeat)
        self.gc2 = GraphConvolution(outfeat * 2, infeat)
        self.gc3 = GraphConvolution(infeat * 2, outfeat)
        self.gc4 = GraphConvolution(outfeat * 3, infeat)


    def forward(self, real_x, real_adj, fake_x, fake_adj):
        gc1_output = F.relu(self.gc1(fake_x, real_adj))
        gc1_concat = torch.cat((real_x, gc1_output), 1)

        gc2_output = F.relu(self.gc2(gc1_concat, fake_adj))
        gc2_concat = torch.cat((fake_x, gc2_output), 1)

        gc3_output = F.relu(self.gc3(gc2_concat, real_adj))
        gc3_concat = torch.cat((gc1_concat, gc3_output), 1)

        gc4_output = torch.tanh(self.gc4(gc3_concat, fake_adj))

        return gc3_output, gc4_output
