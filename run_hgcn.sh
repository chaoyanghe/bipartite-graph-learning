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
<<<<<<< HEAD
--epochs 1 \
--lr 0.0003 \
--weight_decay 0.0005 \
--dis_hidden 32 \
--dropout 0.5 \
--gcn_output_dim 10
=======
--epochs 3 \
--lr 0.0003 \
--weight_decay 0.001 \
--dis_hidden 16 \
--dropout 0.4
>>>>>>> 5f0a5da07742aab45882830270484d354387eb5e

python3 ./HGCN/classification.py