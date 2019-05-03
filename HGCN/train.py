from __future__ import division
from __future__ import print_function

import logging
from time import sleep

import torch
from memory_profiler import profile

from data.utils import load_data
from gan.models import GAN
from pygcn.models import GCN
from utils import (EPOCHS)

import numpy as np

"""Single layer for adversarial loss"""


class AdversarialHGCNLayer(object):
    def __init__(self, bipartite_graph_data_loader, u_attr_dimensions, v_attr_dimensions, device):
        logging.info('AdversarialHGCNLayer')
        self.gcn_explicit = GCN(v_attr_dimensions, u_attr_dimensions).to(device)
        self.gcn_implicit = GCN(u_attr_dimensions, v_attr_dimensions).to(device)
        self.gcn_merge = GCN(v_attr_dimensions, u_attr_dimensions).to(device)
        self.gcn_opposite = GCN(u_attr_dimensions, v_attr_dimensions).to(device)
        self.gan_explicit = GAN(self.gcn_explicit, u_attr_dimensions, device, outfeat=1)
        self.gan_implicit = GAN(self.gcn_implicit, v_attr_dimensions, device, outfeat=1)
        self.gan_merge = GAN(self.gcn_merge, u_attr_dimensions, device, outfeat=1)
        self.gan_opposite = GAN(self.gcn_opposite, v_attr_dimensions, device, outfeat=1)

        self.bipartite_graph_data_loader = bipartite_graph_data_loader
        self.device = device
        self.batch_size = bipartite_graph_data_loader.batch_size

        logging.info('AdversarialHGCNLayer')
        self.batch_num_u = bipartite_graph_data_loader.get_batch_num_u()
        self.batch_num_v = bipartite_graph_data_loader.get_batch_num_v()
        self.u_attr = bipartite_graph_data_loader.get_u_attr_array()
        self.v_attr = bipartite_graph_data_loader.get_v_attr_array()
        self.u_adj = bipartite_graph_data_loader.get_u_adj()
        self.v_adj = bipartite_graph_data_loader.get_v_adj()
        self.u_num = len(self.u_attr)
        logging.info('AdversarialHGCNLayer')

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    @profile(precision=4, stream=open('memory_profiler.log', 'w+'))
    def relation_learning(self):
        # explicit
        logging.info('Step 1: Explicit relation learning')
        u_explicit_attr = torch.FloatTensor([]).to(self.device)
        for i in range(EPOCHS):
            for iter in range(self.batch_num_u):
                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                if iter == self.batch_num_u - 1:
                    end_index = self.u_num
                u_attr_batch = self.u_attr[start_index:end_index]
                u_adj_batch = self.u_adj[start_index:end_index]
                u_attr_tensor = torch.as_tensor(u_attr_batch, dtype=torch.float)

                # here the data type is not float, so TORCH will allocate new memory.
                u_adj_tensor = self.sparse_mx_to_torch_sparse_tensor(u_adj_batch)

                print(iter)
                # training
                gcn_explicit_output = self.gcn_explicit(torch.as_tensor(self.v_attr), u_adj_tensor)
                # record the last epoch output from gcn as new hidden representation
                if i == EPOCHS - 1:
                    u_explicit_attr = torch.cat((u_explicit_attr, gcn_explicit_output.detach()), 0)
                self.gan_explicit.forward_backward(u_attr_tensor, gcn_explicit_output, step=1, epoch=i, iter=iter)

                # validation
                # if iter % VALIDATE_ITER == 0:
                #     gcn_explicit_output = self.gcn_explicit(self.v_attr, u_adj)
                #     self.gan_explicit.forward(u_input, gcn_explicit_output, iter)
        sleep(100000)
        # implicit
        logging.info('Step 2: Implicit relation learning')
        v_implicit_attr = torch.FloatTensor([]).to(self.device)
        for i in range(EPOCHS):
            for iter in range(self.batch_num_v):
                # load batch data
                v_input, v_adj = self.bipartite_graph_data_loader.get_one_batch_group_v_with_adjacent(iter)
                v_input, v_adj = torch.FloatTensor(v_input).to(self.device), torch.FloatTensor(v_adj).to(self.device)
                gcn_implicit_output = self.gcn_implicit(u_explicit_attr, v_adj)
                if i == EPOCHS - 1:
                    v_implicit_attr = torch.cat((v_implicit_attr, gcn_implicit_output.detach()), 0)
                self.gan_implicit.forward_backward(v_input, gcn_implicit_output, step=2, epoch=i, iter=iter)
                # if iter % VALIDATE_ITER == 0:
                #     gcn_implicit_output = self.gcn_implicit(self.u_attr, v_adj)
                #     self.gan_implicit.forward(v_input, gcn_implicit_output, iter)

        # merge
        logging.info('Step 3: Merge relation learning')
        u_merge_attr = torch.FloatTensor([]).to(self.device)
        for i in range(EPOCHS):
            for iter in range(self.batch_num_u):
                _, u_adj = self.bipartite_graph_data_loader.get_one_batch_group_u_with_adjacent(iter)
                u_adj = torch.FloatTensor(u_adj).to(self.device)
                gcn_merge_output = self.gcn_merge(v_implicit_attr, u_adj)
                if i == EPOCHS - 1:
                    u_merge_attr = torch.cat((u_merge_attr, gcn_merge_output.detach()), 0)
                start_batch = iter * self.batch_size
                u_input = u_explicit_attr[start_batch:start_batch + self.batch_size]
                self.gan_merge.forward_backward(u_input, gcn_merge_output,
                                                step=3, epoch=i, iter=iter)
                # if iter % VALIDATE_ITER == 0:
                #     gcn_merge_output = self.gcn_merge(v_implicit_attr, u_adj)
                #     self.gan_merge.forward(u_explicit_attr, gcn_merge_output, iter)

        # opposite
        logging.info('Step 4: Opposite relation learning')
        for i in range(EPOCHS):
            for iter in range(self.batch_num_v):
                _, v_adj = self.bipartite_graph_data_loader.get_one_batch_group_v_with_adjacent(iter)
                v_adj = torch.FloatTensor(v_adj).to(self.device)
                gcn_opposite_output = self.gcn_opposite(u_merge_attr, v_adj)
                start_batch = iter * self.batch_size
                v_input = v_implicit_attr[start_batch:start_batch + self.batch_size]
                self.gan_opposite.forward_backward(v_input, gcn_opposite_output,
                                                   step=4, epoch=i, iter=iter)
                # if iter % VALIDATE_ITER == 0:
                #     gcn_opposite_output = self.gcn_merge(u_merge_attr, v_adj)
                #     self.gan_opposite.forward(v_implicit_attr, gcn_opposite_output, iter)


"""Single layer for decoder layer"""

# class DecoderGCNLayer(object):
#     def __init__(self, featuresU_dimensions, featuresV_dimensions, device, dimensions_output_from_gcn=2):
#         """For decoder layer, we can define any output dimension from GCN layer"""
#         dimensions_output_from_gcn = dimensions_output_from_gcn
#         self.gcn_explicit = GCN(featuresV_dimensions, dimensions_output_from_gcn).to(device)
#         self.gcn_implicit = GCN(featuresU_dimensions, dimensions_output_from_gcn).to(device)
#         self.gcn_merge = GCN(featuresV_dimensions, dimensions_output_from_gcn).to(device)
#         self.gcn_opposite = GCN(featuresU_dimensions, dimensions_output_from_gcn).to(device)
#         self.decoder_explicit = HGCNDecoder(self.gcn_explicit, dimensions_output_from_gcn, featuresU_dimensions, device)
#         self.decoder_implicit = HGCNDecoder(self.gcn_implicit, dimensions_output_from_gcn, featuresV_dimensions, device)
#         self.decoder_merge = HGCNDecoder(self.gcn_merge, dimensions_output_from_gcn, featuresU_dimensions, device)
#         self.decoder_opposite = HGCNDecoder(self.gcn_opposite, dimensions_output_from_gcn, featuresV_dimensions, device)
#
#     def relation_learning(self):
#         # explicit
#         logging.info('Step 1: Explicit relation learning')
#         last_epoch_explicit_output_from_decoder = []
#         for i in range(EPOCHS):  # separate the data into batches
#             for iter in range(Number of batches):
#                 # load batch data
#                 batch_index = BipartiteGraphDataLoader.get_batch_num()  # potential memory bug
#                 u_input, v_input, u_adj = BipartiteGraphDataLoader.get_one_batch_group_u_with_adjacent(batch_index)
#                 # training
#                 gcn_explicit_output = self.gcn_explicit(v_input, u_adj)
#                 self.decoder_explicit.forward_backward(u_input, gcn_explicit_output, step=1, epoch=i, iter=iter)
#                 if i == EPOCHS - 1:
#                     last_epoch_explicit_output_from_decoder.append(self.decoder_explicit.decoder_output.detach())
#                 # validation
#                 if iter % VALIDATE_ITER == 0:
#                     gcn_explicit_output = self.gcn_explicit(v_input, u_adj)
#                     self.decoder_explicit.forward(u_input, gcn_explicit_output, iter)
#
#         # implicit
#         logging.info('Step 2: Implicit relation learning')
#         last_epoch_implicit_output_from_decoder = []
#         for i in range(EPOCHS):
#             for iter in range(Number of batches):
#                 batch_index = BipartiteGraphDataLoader.get_batch_num()
#                 u_input, v_input, v_adj = BipartiteGraphDataLoader.get_one_batch_group_v_with_adjacent(batch_index)
#                 gcn_implicit_output = self.gcn_implicit(u_input, v_adj)
#                 self.decoder_implicit.forward_backward(v_input, gcn_implicit_output, step=2, epoch=i, iter=iter)
#                 if i == EPOCHS - 1:
#                     last_epoch_implicit_output_from_decoder.append(self.decoder_implicit.decoder_output.detach())
#                 if iter % VALIDATE_ITER == 0:
#                     gcn_implicit_output = self.gcn_implicit(u_input, v_adj)
#                     self.decoder_implicit.forward(v_input, gcn_implicit_output, iter)
#
#         # merge
#         logging.info('Step 3: Merge relation learning')
#         # the output is the output from decoder, since we can set different dimensions of the output from GCN
#         last_epoch_merge_output_from_decoder = []
#         for i in range(EPOCHS):
#             for iter in range(Number of batches):
#                 batch_index = BipartiteGraphDataLoader.get_batch_num()
#                 _, _, u_adj = BipartiteGraphDataLoader.get_one_batch_group_u_with_adjacent(batch_index)
#                 gcn_merge_output = self.gcn_merge(last_epoch_implicit_output_from_decoder[iter], u_adj)
#                 self.decoder_merge.forward_backward(last_epoch_explicit_output_from_decoder[iter],
#                                                     gcn_merge_output, step=3, epoch=i, iter=iter)
#                 if iter == EPOCHS - 1:
#                     last_epoch_merge_output_from_decoder.append(self.decoder_merge.decoder_output.detach())
#                 if iter % VALIDATE_ITER == 0:
#                     gcn_merge_output = self.gcn_merge(last_epoch_implicit_output_from_decoder[iter], u_adj)
#                     self.decoder_merge.forward(last_epoch_explicit_output_from_decoder[iter], gcn_merge_output, iter)
#
#         # opposite
#         logging.info('Step 4: Opposite relation learning')
#         for i in range(EPOCHS):
#             for iter in range(Number of batches):
#                 batch_index = BipartiteGraphDataLoader.get_batch_num()
#                 _, _, v_adj = BipartiteGraphDataLoader.get_one_batch_group_v_with_adjacent(batch_index)
#                 gcn_opposite_output = self.gcn_opposite(last_epoch_merge_output_from_decoder[iter], v_adj)
#                 self.decoder_opposite.forward_backward(last_epoch_implicit_output_from_decoder[iter],
#                                                        gcn_opposite_output, step=4, epoch=i, iter=iter)
#                 if iter % VALIDATE_ITER == 0:
#                     gcn_opposite_output = self.gcn_opposite(last_epoch_merge_output_from_decoder[iter], v_adj)
#                     self.decoder_opposite.forward(last_epoch_implicit_output_from_decoder[iter], gcn_opposite_output,
#                                                   iter)


"""Model selection / start training"""


class HeterogeneousGCN(object):
    def __init__(self, bipartite_graph_data_loader, device):
        # self.hidden_dimensions = hidden_dimensions
        # self.dropout = dropout
        self.u_attr_dimensions = bipartite_graph_data_loader.get_u_attr_dimensions()
        self.v_attr_dimensions = bipartite_graph_data_loader.get_v_attr_dimensions()
        self.bipartite_graph_data_loader = bipartite_graph_data_loader
        self.device = device

    # def decoder_train(self):
    #     decoder_hgcn = DecoderGCNLayer(self.u_attr_dimensions,
    #                                    self.v_attr_dimensions,
    #                                    self.device,
    #                                    dimensions_output_from_gcn=2)
    #     # training_dict = {'Uinput': Uinput, 'Vinput': Vinput, 'adjU': adjU, 'adjV': adjV}
    #     # val_dict = {'Uinput': Uinput, 'Vinput': Vinput, 'adjU': adjU, 'adjV': adjV}
    #     decoder_hgcn.relation_learning()

    def adversarial_train(self):
        logging.info('adversarial_train')
        adversarial_hgcn = AdversarialHGCNLayer(self.bipartite_graph_data_loader,
                                                self.u_attr_dimensions,
                                                self.v_attr_dimensions,
                                                self.device)
        logging.info('relation_learning')
        adversarial_hgcn.relation_learning()


if __name__ == '__main__':
    adjU, adjV, featuresU, featuresV = load_data()
    h_gcn = HeterogeneousGCN(adjU, adjV, featuresU, featuresV)
    # h_gcn.adversarial_train()
    h_gcn.decoder_train()
