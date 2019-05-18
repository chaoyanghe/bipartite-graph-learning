#!/usr/bin/env bash

rm -rf ./out
rm -rf HGCN.log_embedding

python3 ./HGCN/hgcn_main.py \
--dataset cora \
--model gan_gcn \
--gpu False \
--epochs 1 \
--batch_size 10 \
--lr 0.0003 \
--weight_decay 0.0005 \
--dropout 0.5 \
--gcn_output_dim 24

python3 ./HGCN/multi-classification.py \
--dataset cora

#python3 ./HGCN/hgcn_main.py \
#--dataset citeseer \
#--model gan_gcn \
#--gpu False \
#--epochs 3 \
#--batch_size 20 \
#--lr 0.001 \
#--weight_decay 0.0005 \
#--dropout 0.5 \
#--gcn_output_dim 24
#
#python3 ./HGCN/multi-classification.py \
#--dataset citeseer