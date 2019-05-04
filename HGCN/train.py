from __future__ import division
from __future__ import print_function

import logging
import os

import numpy as np
import torch

from gan.models import GAN
from pygcn.models import GCN


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


"""Single layer for adversarial loss"""


class AdversarialHGCNLayer(object):
    def __init__(self, bipartite_graph_data_loader, u_attr_dimensions, v_attr_dimensions,
                 dis_hidden_dim, epochs, learning_rate, weight_decay, dropout, device):
        logging.info('AdversarialHGCNLayer')
        self.gcn_explicit = GCN(v_attr_dimensions, u_attr_dimensions).to(device)
        self.gcn_implicit = GCN(u_attr_dimensions, v_attr_dimensions).to(device)
        self.gcn_merge = GCN(v_attr_dimensions, u_attr_dimensions).to(device)
        self.gcn_opposite = GCN(u_attr_dimensions, v_attr_dimensions).to(device)

        # outfeat=1: binary classification
        self.gan_explicit = GAN(self.gcn_explicit, u_attr_dimensions, dis_hidden_dim,
                                learning_rate, weight_decay, dropout, device, outfeat=1)
        self.gan_implicit = GAN(self.gcn_implicit, v_attr_dimensions, dis_hidden_dim,
                                learning_rate, weight_decay, dropout, device, outfeat=1)
        self.gan_merge = GAN(self.gcn_merge, u_attr_dimensions, dis_hidden_dim,
                             learning_rate, weight_decay, dropout, device, outfeat=1)
        self.gan_opposite = GAN(self.gcn_opposite, v_attr_dimensions, dis_hidden_dim,
                                learning_rate, weight_decay, dropout, device, outfeat=1)

        self.bipartite_graph_data_loader = bipartite_graph_data_loader
        self.batch_size = bipartite_graph_data_loader.batch_size
        self.device = device
        self.epochs = epochs

        logging.info('AdversarialHGCNLayer')
        self.batch_num_u = bipartite_graph_data_loader.get_batch_num_u()
        self.batch_num_v = bipartite_graph_data_loader.get_batch_num_v()
        self.u_attr = bipartite_graph_data_loader.get_u_attr_array()
        self.v_attr = bipartite_graph_data_loader.get_v_attr_array()
        self.u_adj = bipartite_graph_data_loader.get_u_adj()
        self.v_adj = bipartite_graph_data_loader.get_v_adj()
        self.u_num = len(self.u_attr)
        self.v_num = len(self.v_attr)
        logging.info('AdversarialHGCNLayer')

    def relation_learning(self):
        # explicit
        logging.info('Step 1: Explicit relation learning')
        u_explicit_attr = torch.FloatTensor([]).to(self.device)
        for i in range(self.epochs):
            for iter in range(self.batch_num_u):
                logging.info('load adj and attribute')
                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                if iter == self.batch_num_u - 1:
                    end_index = self.u_num
                u_attr_batch = self.u_attr[start_index:end_index]
                u_adj_batch = self.u_adj[start_index:end_index]

                u_attr_tensor = torch.as_tensor(u_attr_batch, dtype=torch.float, device=self.device)
                u_adj_tensor = sparse_mx_to_torch_sparse_tensor(u_adj_batch).to(device=self.device)
                logging.info('finished loading the tensor')
                # training
                gcn_explicit_output = self.gcn_explicit(torch.as_tensor(self.v_attr, device=self.device), u_adj_tensor)
                logging.info('finished GCN step')

                # record the last epoch output from gcn as new hidden representation
                if i == self.epochs - 1:
                    u_explicit_attr = torch.cat((u_explicit_attr, gcn_explicit_output.detach()), 0)
                self.gan_explicit.forward_backward(u_attr_tensor, gcn_explicit_output, step=1, epoch=i, iter=iter)
                logging.info('finished GAN step')
                # validation
                # if iter % VALIDATE_ITER == 0:
                #     gcn_explicit_output = self.gcn_explicit(self.v_attr, u_adj)
                #     self.gan_explicit.forward(u_input, gcn_explicit_output, iter)

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
                # if iter % VALIDATE_ITER == 0:
                #     gcn_implicit_output = self.gcn_implicit(self.u_attr, v_adj)
                #     self.gan_implicit.forward(v_input, gcn_implicit_output, iter)

        # merge
        logging.info('Step 3: Merge relation learning')
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
                # if iter % VALIDATE_ITER == 0:
                #     gcn_merge_output = self.gcn_merge(v_implicit_attr, u_adj)
                #     self.gan_merge.forward(u_explicit_attr, gcn_merge_output, iter)

        self.save_embedding_to_file(u_merge_attr.numpy(), None)

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
                # if iter % VALIDATE_ITER == 0:
                #     gcn_opposite_output = self.gcn_merge(u_merge_attr, v_adj)
                #     self.gan_opposite.forward(v_implicit_attr, gcn_opposite_output, iter)

    # format:
    # line1: number of the node, dimension of the embedding vector
    # line2: node_id, embedding vector
    # line3: ...
    # lineN: node_id, embedding vector
    def save_embedding_to_file(self, gcn_merge_output, node_id_list):
        gcn_merge_output = np.zeros((1000, 32))
        node_id_list = np.zeros((1000, 1))
        node_num = gcn_merge_output.shape[0]
        dimension_embedding = gcn_merge_output[1]
        output_folder = "./out"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        f = open(output_folder + '/hgcn.emb', 'w')
        f.write(node_num + " " + dimension_embedding + "\n")
        for n_idx in range(node_num):
            f.write(node_id_list[n_idx] + ' ')
            emb_vec = gcn_merge_output[n_idx]
            for d_idx in range(dimension_embedding):
                if d_idx != dimension_embedding - 1:
                    f.write(emb_vec[d_idx] + ' ')
                else:
                    f.write(emb_vec[d_idx])
        f.close()


"""Model selection / start training"""


class HeterogeneousGCN(object):
    def __init__(self, bipartite_graph_data_loader, args, device):
        self.epochs = args.epochs
        self.dis_hidden_dim = args.dis_hidden
        self.learning_rate = args.lr
        self.weight_decay = args.weight_decay
        self.dropout = args.dropout
        self.u_attr_dimensions = bipartite_graph_data_loader.get_u_attr_dimensions()
        self.v_attr_dimensions = bipartite_graph_data_loader.get_v_attr_dimensions()
        self.bipartite_graph_data_loader = bipartite_graph_data_loader
        self.device = device

    def adversarial_train(self):
        logging.info('adversarial_train')
        adversarial_hgcn = AdversarialHGCNLayer(self.bipartite_graph_data_loader,
                                                self.u_attr_dimensions,
                                                self.v_attr_dimensions,
                                                self.dis_hidden_dim,
                                                self.epochs,
                                                self.learning_rate,
                                                self.weight_decay,
                                                self.droput,
                                                self.device)
        logging.info('relation_learning')
        adversarial_hgcn.relation_learning()


if __name__ == '__main__':
    print("")
