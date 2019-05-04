#!/usr/bin/env bash

python3 hgcn_main.py \
--model gan_gcn \
--gpu False \
--epochs 1 \
--lr 0.001 \
--weight_decay 0.1 \
--dis_hidden 10 \
--dropout 0.5