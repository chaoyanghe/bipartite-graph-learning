#!/usr/bin/env bash

rm -rf ./class_map.json
rm -rf ./G.json
rm -rf ./id_map.json

python graphsage_data_loader.py
