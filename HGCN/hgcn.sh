#!/usr/bin/env bash

rm -rf HGCN.log

python3 hgcn_main.py \
--model gan_gcn \
--gpu False \
--batch_size 100 \
--epochs 1 \
--lr 0.001 \
--weight_decay 0.1 \
--dis_hidden 10 \
--dropout 0.5

python3 classification.py