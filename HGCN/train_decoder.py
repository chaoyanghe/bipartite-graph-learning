from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch

from pygcn.models import GCN
from decoder.models import HGCNDecoder


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class DecoderGCNLayer(object):
    def __init__(self, bipartite_graph_data_loader, u_attr_dimensions, v_attr_dimensions,
                 decoder_hidden_dim, epochs, learning_rate, weight_decay, dropout, device,
                 gcn_output_dim):
        """For decoder layer, we can define any output dimension from GCN layer"""

        self.gcn_explicit = GCN(v_attr_dimensions, u_attr_dimensions).to(device)
        self.gcn_implicit = GCN(u_attr_dimensions, v_attr_dimensions).to(device)
        self.gcn_merge = GCN(v_attr_dimensions, u_attr_dimensions).to(device)
        self.gcn_opposite = GCN(u_attr_dimensions, v_attr_dimensions).to(device)

        self.decoder_explicit = HGCNDecoder(self.gcn_explicit, gcn_output_dim, u_attr_dimensions, decoder_hidden_dim,
                                            learning_rate, weight_decay, dropout, device)
        self.decoder_implicit = HGCNDecoder(self.gcn_implicit, gcn_output_dim, u_attr_dimensions, decoder_hidden_dim,
                                            learning_rate, weight_decay, dropout, device)
        self.decoder_merge = HGCNDecoder(self.gcn_merge, gcn_output_dim, u_attr_dimensions, decoder_hidden_dim,
                                         learning_rate, weight_decay, dropout, device)
        self.decoder_opposite = HGCNDecoder(self.gcn_opposite, gcn_output_dim, u_attr_dimensions, decoder_hidden_dim,
                                            learning_rate, weight_decay, dropout, device)

        self.batch_size = bipartite_graph_data_loader.batch_size
        self.device = device
        self.epochs = epochs

        self.batch_num_u = bipartite_graph_data_loader.get_batch_num_u()
        self.batch_num_v = bipartite_graph_data_loader.get_batch_num_v()
        self.u_attr = bipartite_graph_data_loader.get_u_attr_array()
        self.v_attr = bipartite_graph_data_loader.get_v_attr_array()
        self.u_adj = bipartite_graph_data_loader.get_u_adj()
        self.v_adj = bipartite_graph_data_loader.get_v_adj()
        self.u_num = len(self.u_attr)
        self.v_num = len(self.v_attr)

    def relation_learning(self):
        # explicit
        logging.info('Step 1: Explicit relation learning')
        U_explicit_attr = torch.FlaotTensosr([]).to(self.device)
        for i in range(self.epochs):
            for iter in range(self.batch_num_u):
                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                if iter == self.batch_num_u - 1:
                    end_index = self.u_num
                u_attr_batch = self.u_attr[start_index:end_index]
                u_adj_batch = self.u_adj[start_index:end_index]

                u_attr_tensor = torch.as_tensor(u_attr_batch, dtype=torch.float, device=self.device)
                u_adj_tensor = sparse_mx_to_torch_sparse_tensor(u_adj_batch).to(device=self.device)

                # training
                gcn_explicit_output = self.gcn_explicit(torch.as_tensor(self.v_attr, device=self.device), u_adj_tensor)
                if i == self.epochs - 1:
                    u_explicit_attr = torch.cat((u_explicit_attr, gcn_explicit_output.detach()), 0)
                self.gan_explicit.forward_backward(u_attr_tensor, gcn_explicit_output, step=1, epoch=i, iter=iter)

        # implicit
        logging.info('Step 2: Implicit relation learning')
        v_implicit_attr = torch.FloatTensor([]).to(self.device)
        for i in range(self.epochs):
            for iter in range(self.batch_num_v):
                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                if iter == self.batch_num_v - 1:
                    end_index = self.v_num
                v_attr_batch = self.v_attr[start_index:end_index]
                v_adj_batch = self.v_adj[start_index:end_index]

                v_attr_tensor = torch.as_tensor(v_attr_batch, dtype=torch.float, device=self.device)
                v_adj_tensor = sparse_mx_to_torch_sparse_tensor(v_adj_batch).to(device=self.device)

                # training
                gcn_implicit_output = self.gcn_implicit(u_explicit_attr, v_adj_tensor)

                # record the last epoch output from gcn as new hidden representation
                if i == self.epochs - 1:
                    v_implicit_attr = torch.cat((v_implicit_attr, gcn_implicit_output.detach()), 0)
                self.gan_implicit.forward_backward(v_attr_tensor, gcn_implicit_output, step=2, epoch=i, iter=iter)

        # merge
        u_merge_attr = torch.FloatTensor([]).to(self.device)
        for i in range(self.epochs):
            for iter in range(self.batch_num_u):
                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                if iter == self.batch_num_u - 1:
                    end_index = self.u_num
                u_adj_batch = self.u_adj[start_index:end_index]
                u_adj_tensor = sparse_mx_to_torch_sparse_tensor(u_adj_batch).to(device=self.device)

                gcn_merge_output = self.gcn_merge(v_implicit_attr, u_adj_tensor)
                if i == self.epochs - 1:
                    u_merge_attr = torch.cat((u_merge_attr, gcn_merge_output.detach()), 0)
                u_input = u_explicit_attr[start_index:end_index]
                self.gan_merge.forward_backward(u_input, gcn_merge_output,
                                                step=3, epoch=i, iter=iter)

        # opposite
        logging.info('Step 4: Opposite relation learning')
        for i in range(self.epochs):
            for iter in range(self.batch_num_v):
                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                if iter == self.batch_num_v - 1:
                    end_index = self.v_num
                v_adj_batch = self.v_adj[start_index:end_index]
                v_adj_tensor = sparse_mx_to_torch_sparse_tensor(v_adj_batch).to(device=self.device)

                gcn_opposite_output = self.gcn_opposite(u_merge_attr, v_adj_tensor)
                v_input = v_implicit_attr[start_index:end_index]
                self.gan_opposite.forward_backward(v_input, gcn_opposite_output,
                                                   step=4, epoch=i, iter=iter)
