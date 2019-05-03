from time import sleep

import numpy as np
import torch

from data.bipartite_graph_data_loader import BipartiteGraphDataLoader

if __name__ == '__main__':
    NODE_LIST_PATH = "./../data/Tencent-QQ/node_list"
    NODE_ATTR_PATH = "./../data/Tencent-QQ/node_attr"
    NODE_LABEL_PATH = "./../data/Tencent-QQ/node_true"

    EDGE_LIST_PATH = "./../data/Tencent-QQ/edgelist"

    GROUP_LIST_PATH = "./../data/Tencent-QQ/group_list"
    GROUP_ATTR_PATH = "./../data/Tencent-QQ/group_attr"
    bipartite_graph_data_loader = BipartiteGraphDataLoader(100, NODE_LIST_PATH, NODE_ATTR_PATH, NODE_LABEL_PATH,
                                                           EDGE_LIST_PATH,
                                                           GROUP_LIST_PATH, GROUP_ATTR_PATH)
    bipartite_graph_data_loader.load()

    batch_size = 100

    batch_num_u = bipartite_graph_data_loader.get_batch_num_u()
    u_attr = bipartite_graph_data_loader.get_u_attr_array()
    v_attr = bipartite_graph_data_loader.get_v_attr_array()
    u_adj = bipartite_graph_data_loader.get_u_adj()
    v_adj = bipartite_graph_data_loader.get_v_adj()
    u_num = len(u_attr)
    for batch_index in range(batch_num_u):
        start_index = batch_size * batch_index
        end_index = batch_size * (batch_index + 1)
        if batch_index == batch_num_u - 1:
            end_index = u_num
        u_attr_np = u_attr[start_index:end_index]
        u_adj_np = u_adj[start_index:end_index]
        u_attr_tensor = torch.as_tensor(u_attr_np, dtype=torch.float)
        u_adj_tensor = torch.as_tensor(u_adj_np, dtype=torch.float)
        print(u_attr_np[0])
        print(u_adj_np[0])
        del u_adj_tensor
        print(batch_index)

    sleep(10000)