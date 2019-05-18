#!/usr/bin/env bash

rm -rf gae.log
rm -rf out/*

python3 ./gae/train.py \
--dataset cora \
--model gcn_vae \
--epochs 100


python3 ./gae/classification.py