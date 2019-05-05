#!/usr/bin/env bash

# for AutoML
#python3 ./HGCN/hgcn_main.py \
#--model gan_gcn \
#--gpu True \
#--batch_size ##batch_size## \
#--epochs ##epochs## \
#--lr ##lr## \
#--weight_decay ##weight_decay## \
#--dis_hidden ##dis_hidden## \
#--dropout ##dropout##
#
#python3 ./HGCN/classification.py


# for local
rm -rf ./out
python3 ./HGCN/hgcn_main.py \
--model gan_gcn \
--gpu False \
--batch_size 1000 \
--epochs 30 \
--lr 0.0003 \
--weight_decay 0.0005 \
--dis_hidden 32 \
--dropout 0.5

python3 ./HGCN/classification.py