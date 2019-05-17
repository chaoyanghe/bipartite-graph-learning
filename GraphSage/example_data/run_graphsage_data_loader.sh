#!/usr/bin/env bash

rm -rf ./bipartite-class_map.json
rm -rf ./bipartite-G.json
rm -rf ./bipartite-id_map.json
rm -rf ./bipartite-id_map_node.json
rm -rf ./bipartite_graph_data_loading.log

python graphsage_data_loader.py


