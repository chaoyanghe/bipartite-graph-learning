#!/usr/bin/env bash

rm -rf ./bipartite-class_map.json
rm -rf ./bipartite-G.json
rm -rf ./bipartite-id_map.json
rm -rf ./bipartite-id_map_node.json
rm -rf ./bipartite_graph_data_loading.log
rm -rf ./bipartite-walks.txt
rm -rf ./bipartite-feats.npy

python graphsage_data_loader_cora.py

python random_walk.py --walk_len 5 --n_walks 50


