#!/usr/bin/env bash

rm -rf ./bipartite-class_map.json
rm -rf ./bipartite-G.json
rm -rf ./bipartite-id_map.json
rm -rf ./bipartite-id_map_node.json

python graphsage_data_loader.py
