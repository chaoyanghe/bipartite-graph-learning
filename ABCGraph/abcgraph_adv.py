from __future__ import division
from __future__ import print_function

import logging
import os

import numpy as np
import torch

from adversarial.models import AdversarialLearning
from gcn.models import GCN

import wandb

class ABCGraphAdversarial(object):
    def __init__(self, bipartite_graph_data_loader, args, device, rank=-1, dataset="cora"):
        self.rank = rank
        self.dataset = dataset
        epochs = args.epochs
        dis_hidden_dim = args.dis_hidden
        learning_rate = args.lr
        weight_decay = args.weight_decay
        dropout = args.dropout
        u_attr_dimensions = bipartite_graph_data_loader.get_u_attr_dimensions()
        v_attr_dimensions = bipartite_graph_data_loader.get_v_attr_dimensions()
        bipartite_graph_data_loader = bipartite_graph_data_loader
        device = device

        logging.info('ABCGraphAdversarial')
        self.gcn_explicit = GCN(v_attr_dimensions, u_attr_dimensions).to(device)
        self.gcn_implicit = GCN(u_attr_dimensions, v_attr_dimensions).to(device)
        self.gcn_merge = GCN(v_attr_dimensions, u_attr_dimensions).to(device)
        self.gcn_opposite = GCN(u_attr_dimensions, v_attr_dimensions).to(device)

        self.adversarial_explicit = AdversarialLearning(self.gcn_explicit, u_attr_dimensions, dis_hidden_dim,
                                                        learning_rate, weight_decay, dropout, device, outfeat=1)
        self.adversarial_implicit = AdversarialLearning(self.gcn_implicit, v_attr_dimensions, dis_hidden_dim,
                                                        learning_rate, weight_decay, dropout, device, outfeat=1)
        self.adversarial_merge = AdversarialLearning(self.gcn_merge, u_attr_dimensions, dis_hidden_dim,
                                                     learning_rate, weight_decay, dropout, device, outfeat=1)
        self.adversarial_opposite = AdversarialLearning(self.gcn_opposite, v_attr_dimensions, dis_hidden_dim,
                                                        learning_rate, weight_decay, dropout, device, outfeat=1)

        self.bipartite_graph_data_loader = bipartite_graph_data_loader
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
        print("self.u_num = %d" % self.u_num)
        print("self.v_num = %d" % self.v_num)
        self.f_loss = open("ABCGraph-Adv-loss.txt", "a")

    def adversarial_learning(self):
        logging.info('adversarial_train')

        # explicit training
        logging.info('###Depth 1 starts!\n')
        for i in range(self.epochs):
            for iter in range(self.batch_num_u):
                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                if iter == self.batch_num_u - 1:
                    end_index = self.u_num
                u_attr_batch = self.u_attr[start_index:end_index]
                u_adj_batch = self.u_adj[start_index:end_index]

                # prepare data to the tensor
                u_attr_tensor = torch.as_tensor(u_attr_batch, dtype=torch.float, device=self.device)
                u_adj_tensor = self.__sparse_mx_to_torch_sparse_tensor(u_adj_batch).to(device=self.device)

                # training
                gcn_explicit_output = self.gcn_explicit(torch.as_tensor(self.v_attr, device=self.device), u_adj_tensor)
                lossD, lossG = self.adversarial_explicit.forward_backward(u_attr_tensor, gcn_explicit_output, step=1,
                                                                          epoch=i,
                                                                          iter=iter)
                self.f_loss.write("%s %s\n" % (lossD, lossG))
                wandb.log({"lossD": lossD, "epoch": i + self.epochs*0})
                wandb.log({"lossG": lossG, "epoch": i + self.epochs*0})

        self.f_loss.write("###Depth 1 finished!\n")
        # explicit inference
        u_explicit_attr = torch.FloatTensor([]).to(self.device)
        for iter in range(self.batch_num_u):
            start_index = self.batch_size * iter
            end_index = self.batch_size * (iter + 1)
            if iter == self.batch_num_u - 1:
                end_index = self.u_num
            u_adj_batch = self.u_adj[start_index:end_index]

            # prepare data to the tensor
            u_adj_tensor = self.__sparse_mx_to_torch_sparse_tensor(u_adj_batch).to(device=self.device)

            # inference
            gcn_explicit_output = self.gcn_explicit(torch.as_tensor(self.v_attr, device=self.device), u_adj_tensor)
            u_explicit_attr = torch.cat((u_explicit_attr, gcn_explicit_output.detach()), 0)

        # implicit training
        logging.info('###Depth 2 starts!\n')
        for i in range(self.epochs):
            for iter in range(self.batch_num_v):
                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                if iter == self.batch_num_v - 1:
                    end_index = self.v_num
                v_attr_batch = self.v_attr[start_index:end_index]
                v_adj_batch = self.v_adj[start_index:end_index]

                # prepare the data to the tensor
                v_attr_tensor = torch.as_tensor(v_attr_batch, dtype=torch.float, device=self.device)
                v_adj_tensor = self.__sparse_mx_to_torch_sparse_tensor(v_adj_batch).to(device=self.device)

                # training
                gcn_implicit_output = self.gcn_implicit(u_explicit_attr, v_adj_tensor)
                lossD, lossG = self.adversarial_implicit.forward_backward(v_attr_tensor, gcn_implicit_output, step=2,
                                                                          epoch=i,
                                                                          iter=iter)
                self.f_loss.write("%s %s\n" % (lossD, lossG))
                wandb.log({"lossD": lossD, "epoch": i + self.epochs*1})
                wandb.log({"lossG": lossG, "epoch": i + self.epochs*1})

        self.f_loss.write("###Depth 2 finished!\n")
        # implicit inference
        v_implicit_attr = torch.FloatTensor([]).to(self.device)
        for iter in range(self.batch_num_v):
            start_index = self.batch_size * iter
            end_index = self.batch_size * (iter + 1)
            if iter == self.batch_num_v - 1:
                end_index = self.v_num
            v_adj_batch = self.v_adj[start_index:end_index]

            # prepare the data to the tensor
            v_adj_tensor = self.__sparse_mx_to_torch_sparse_tensor(v_adj_batch).to(device=self.device)

            # inference
            gcn_implicit_output = self.gcn_implicit(u_explicit_attr, v_adj_tensor)
            v_implicit_attr = torch.cat((v_implicit_attr, gcn_implicit_output.detach()), 0)

        # merge
        logging.info('###Depth 3 starts!\n')
        for i in range(self.epochs):
            for iter in range(self.batch_num_u):
                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                if iter == self.batch_num_u - 1:
                    end_index = self.u_num
                u_adj_batch = self.u_adj[start_index:end_index]

                # prepare the data to the tensor
                u_adj_tensor = self.__sparse_mx_to_torch_sparse_tensor(u_adj_batch).to(device=self.device)

                # training
                gcn_merge_output = self.gcn_merge(v_implicit_attr, u_adj_tensor)
                u_input = u_explicit_attr[start_index:end_index]
                lossD, lossG = self.adversarial_merge.forward_backward(u_input, gcn_merge_output, step=3, epoch=i,
                                                                       iter=iter)
                self.f_loss.write("%s %s\n" % (lossD, lossG))
                wandb.log({"lossD": lossD, "epoch": i + self.epochs*2})
                wandb.log({"lossG": lossG, "epoch": i + self.epochs*2})

        u_merge_attr = torch.FloatTensor([]).to(self.device)
        for iter in range(self.batch_num_u):
            start_index = self.batch_size * iter
            end_index = self.batch_size * (iter + 1)
            if iter == self.batch_num_u - 1:
                end_index = self.u_num
            u_adj_batch = self.u_adj[start_index:end_index]

            # prepare the data to the tensor
            u_adj_tensor = self.__sparse_mx_to_torch_sparse_tensor(u_adj_batch).to(device=self.device)

            # inference
            gcn_merge_output = self.gcn_merge(v_implicit_attr, u_adj_tensor)
            u_merge_attr = torch.cat((u_merge_attr, gcn_merge_output.detach()), 0)

        self.f_loss.write("###Depth 3 finished!\n")
        print(u_merge_attr)
        self.__save_embedding_to_file(u_merge_attr.cpu().numpy(), self.bipartite_graph_data_loader.get_u_list())

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
        output_folder = "./out/abcgraph-adv/" + str(self.dataset)
        if self.rank != -1:
            output_folder = "/mnt/shared/home/bipartite-graph-learning/out/abcgraph-adv/" + self.dataset + "/" + str(
                self.rank)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        f_emb = open(output_folder + '/abcgraph.emb', 'w')
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
        wandb.run.summary["embedding_output_folder"] = str(output_folder)
        wandb.save(os.path.join(wandb.run.dir, output_folder + '/abcgraph.emb'))
        wandb.save(os.path.join(wandb.run.dir, output_folder + '/node_list'))
