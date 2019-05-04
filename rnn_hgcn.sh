#!/usr/bin/env bash

python3 ./HGCN/hgcn_main.py \
--model gan_gcn \
--gpu True \
--epochs ##epochs## \
--lr ##lr## \
--weight_decay ##weight_decay## \
--dis_hidden ##dis_hidden## \
--dropout ##dropout##
