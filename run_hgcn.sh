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
# s
rm -rf ./out
python3 ./HGCN/hgcn_main.py \
--model decoder_gcn \
--gpu False \
--batch_size 500 \
--epochs 1 \
--lr 0.0003 \
--weight_decay 0.0005 \
--dis_hidden 32 \
--dropout 0.5 \
--gcn_output_dim 10

python3 ./HGCN/classification.py