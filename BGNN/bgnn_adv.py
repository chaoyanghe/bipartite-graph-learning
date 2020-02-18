from __future__ import division
from __future__ import print_function

import logging
import os

import numpy as np
import torch

from adversarial.models import AdversarialLearning
from gcn.models import GCN, twoLayersGCN


class BGNNAdversarial(object):
    def __init__(self, bipartite_graph_data_loader, args, device, layer_depth=3, rank=-1, dataset="cora"):
        self.rank = rank
        self.dataset = dataset
        self.epochs = args.epochs
        self.dis_hidden_dim = args.dis_hidden
        self.learning_rate = args.lr
        self.weight_decay = args.weight_decay
        self.dropout = args.dropout
        self.u_attr_dimensions = bipartite_graph_data_loader.get_u_attr_dimensions()
        self.v_attr_dimensions = bipartite_graph_data_loader.get_v_attr_dimensions()
        bipartite_graph_data_loader = bipartite_graph_data_loader
        self.device = device

        self.layer_depth = layer_depth
        self.bipartite_graph_data_loader = bipartite_graph_data_loader
        self.batch_size = bipartite_graph_data_loader.batch_size
        self.device = device

        self.batch_num_u = bipartite_graph_data_loader.get_batch_num_u()
        self.batch_num_v = bipartite_graph_data_loader.get_batch_num_v()
        self.u_attr = bipartite_graph_data_loader.get_u_attr_array()
        self.v_attr = bipartite_graph_data_loader.get_v_attr_array()
        self.u_adj = bipartite_graph_data_loader.get_u_adj()
        self.v_adj = bipartite_graph_data_loader.get_v_adj()
        self.u_num = len(self.u_attr)
        self.v_num = len(self.v_attr)
        self.f_loss = open("BGNN-Adv-loss.txt", "a")

        self.gcn_explicit = GCN(self.v_attr_dimensions, self.u_attr_dimensions).to(device)
        self.gcn_implicit = GCN(self.u_attr_dimensions, self.v_attr_dimensions).to(device)
        self.gcn_merge = GCN(self.v_attr_dimensions, self.u_attr_dimensions).to(device)

        self.learning_type = args.learning_type

    # initialize the layers, start with v as input
    def __layer_initialize(self):
        gcn_layers = []
        adversarial_layers = []
        for i in range(self.layer_depth):
            if i % 2 == 0:
                one_gcn_layer = GCN(self.v_attr_dimensions, self.u_attr_dimensions).to(self.device)
                gcn_layers.append(one_gcn_layer)
                adversarial_layers.append(
                    AdversarialLearning(one_gcn_layer, self.u_attr_dimensions, self.v_attr_dimensions, self.dis_hidden_dim, self.learning_rate,
                                        self.weight_decay, self.dropout, self.device, outfeat=1))
            else:
                one_gcn_layer = GCN(self.u_attr_dimensions, self.v_attr_dimensions).to(self.device)
                gcn_layers.append(one_gcn_layer)
                adversarial_layers.append(
                    AdversarialLearning(one_gcn_layer, self.v_attr_dimensions, self.u_attr_dimensions, self.dis_hidden_dim, self.learning_rate,
                                        self.weight_decay, self.dropout, self.device, outfeat=1))
        return gcn_layers, adversarial_layers

    # end to end learning
    def __layer__gcn(self, real_embedding, real_adj, fake_embedding, fake_adj, step):
        gcn = twoLayersGCN(self.v_attr_dimensions, self.u_attr_dimensions).to(self.device)
        adversarial = AdversarialLearning(gcn, self.u_attr_dimensions, self.v_attr_dimensions, self.dis_hidden_dim, self.learning_rate,
                                           self.weight_decay, self.dropout, self.device, outfeat=1)
        real_adj = self.__sparse_mx_to_torch_sparse_tensor(real_adj)
        fake_adj = self.__sparse_mx_to_torch_sparse_tensor(fake_adj)
        for i in range(self.epochs):
            gc3_output, gc4_output = gcn(real_embedding, real_adj, fake_embedding, fake_adj)

            lossD, lossG = adversarial.two_layers_forward_backward(real_embedding, fake_embedding, gc3_output, gc4_output, step=step, epoch=i, iter=iter)

        new_real_embedding, _ = gcn(real_embedding, real_adj, fake_embedding, fake_adj)
        return new_real_embedding.detach()

    # run the layer-wise inference
    def __layer_inference(self, gcn, adversarial, real_batch_num, real_num, real_embedding, real_adj, fake_embedding,
                          step):
        for i in range(self.epochs):
            for iter in range(real_batch_num):
                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                if iter == real_batch_num - 1:
                    end_index = real_num
                attr_batch = real_embedding[start_index:end_index]
                adj_batch_temp = real_adj[start_index:end_index]
                adj_batch = self.__sparse_mx_to_torch_sparse_tensor(adj_batch_temp).to(device=self.device)

                gcn_output = gcn(fake_embedding, adj_batch)
                lossD, lossG = adversarial.forward_backward(attr_batch, gcn_output, step=step, epoch=i, iter=iter)
                self.f_loss.write("%s %s\n" % (lossD, lossG))

        new_real_embedding = torch.FloatTensor([]).to(self.device)
        for iter in range(real_batch_num):
            start_index = self.batch_size * iter
            end_index = self.batch_size * (iter + 1)
            if iter == real_batch_num - 1:
                end_index = real_num
            adj_batch_temp = real_adj[start_index:end_index]
            adj_batch = self.__sparse_mx_to_torch_sparse_tensor(adj_batch_temp).to(device=self.device)
            gcn_output = gcn(torch.as_tensor(fake_embedding, device=self.device), adj_batch)
            new_real_embedding = torch.cat((new_real_embedding, gcn_output.detach()), 0)
        self.f_loss.write("###Depth finished!\n")

        return new_real_embedding

    def adversarial_learning(self):
        # default start with V as input
        gcn_layers, adversarial_layers = self.__layer_initialize()
        logging.info('adversarial_train')
        u_previous_embedding = torch.as_tensor(self.u_attr, dtype=torch.float, device=self.device)
        v_previous_embedding = torch.as_tensor(self.v_attr, dtype=torch.float, device=self.device)
        if self.learning_type == 'inference':
            for i in range(self.layer_depth):
                if i % 2 == 0:
                    u_previous_embedding = self.__layer_inference(gcn_layers[i], adversarial_layers[i],
                                                                  self.batch_num_u,
                                                                  self.u_num, u_previous_embedding, self.u_adj,
                                                                  v_previous_embedding, i)
                else:
                    v_previous_embedding = self.__layer_inference(gcn_layers[i], adversarial_layers[i],
                                                                  self.batch_num_v,
                                                                  self.v_num, v_previous_embedding, self.v_adj,
                                                                  u_previous_embedding, i)
        elif self.learning_type == 'end2end':
            u_previous_embedding = self.__layer__gcn(u_previous_embedding, self.u_adj, v_previous_embedding, self.v_adj, 0)

        self.__save_embedding_to_file(u_previous_embedding.numpy(), self.bipartite_graph_data_loader.get_u_list())

    def __sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def __save_embedding_to_file(self, gcn_merge_output, node_id_list):
        """ embedding file format:
            line1: number of the node, dimension of the embedding vector
            line2: node_id, embedding vector
            line3: ...
            lineN: node_id, embedding vector

        :param gcn_merge_output:
        :param node_id_list:
        :return:
        """
        logging.info("Start to save embedding file")
        # print(gcn_merge_output)
        node_num = gcn_merge_output.shape[0]
        logging.info("node_num = %s" % node_num)
        dimension_embedding = gcn_merge_output.shape[1]
        logging.info("dimension_embedding = %s" % dimension_embedding)
        output_folder = "./out/bgnn-adv/" + str(self.dataset)
        if self.rank != -1:
            output_folder = "/mnt/shared/home/bipartite-graph-learning/out/bgnn-adv/" + self.dataset + "/" + str(
                self.rank)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        f_emb = open(output_folder + '/bgnn.emb', 'w')
        f_node_list = open(output_folder + '/node_list', 'w')

        str_first_line = str(node_num) + " " + str(dimension_embedding) + "\n"
        f_emb.write(str_first_line)
        for n_idx in range(node_num):
            f_emb.write(str(node_id_list[n_idx]) + ' ')
            f_node_list.write(str(node_id_list[n_idx]))
            emb_vec = gcn_merge_output[n_idx]
            for d_idx in range(dimension_embedding):
                if d_idx != dimension_embedding - 1:
                    f_emb.write(str(emb_vec[d_idx]) + ' ')
                else:
                    f_emb.write(str(emb_vec[d_idx]))
            if n_idx != node_num - 1:
                f_emb.write('\n')
                f_node_list.write('\n')
        f_emb.close()
        f_node_list.close()
        logging.info("Saved embedding file")
