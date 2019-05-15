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
rm -rf HGCN.log_embedding

python3 ./HGCN/hgcn_main.py \
--model decoder_gcn \
--gpu False \
--epochs 1 \
--batch_size 512 \
--lr 0.0003 \
--weight_decay 0.0005 \
--dropout 0.5 \
--gcn_output_dim 24 \
--encoder_hidfeat 16 \
--decoder_hidfeat 8


python3 ./HGCN/classification.py