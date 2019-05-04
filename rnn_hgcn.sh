#!/usr/bin/env bash

python3 ./HGCN/hgcn_main.py \
--model gan_gcn \
--gpu True \
--batch_size ##batch_size## \
--epochs ##epochs## \
--lr ##lr## \
--weight_decay ##weight_decay## \
--dis_hidden ##dis_hidden## \
--dropout ##dropout##

python3 ./HGCN/classification.py