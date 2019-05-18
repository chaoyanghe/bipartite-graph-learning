#!/usr/bin/env bash


DATASET=$1
MODEL=$2

rm -rf ./out/hgcn/$DATASET

if [ "$DATASET" = "tencent" ]
then
    echo $DATASET

    python3 ./HGCN/hgcn_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu False \
    --epochs 1 \
    --batch_size 10 \
    --lr 0.0003 \
    --weight_decay 0.0005 \
    --dropout 0.5 \
    --gcn_output_dim 24

    python3 ./HGCN/binary_classification.py

elif [ "$DATASET" = "cora" ]
then
    echo $DATASET

    python3 ./HGCN/hgcn_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu False \
    --epochs 1 \
    --batch_size 10 \
    --lr 0.0003 \
    --weight_decay 0.0005 \
    --dropout 0.5 \
    --gcn_output_dim 24

    python3 ./HGCN/multi_classification.py --dataset $DATASET

elif [ "$DATASET" = "citeseer" ]
then
    echo $DATASET

    python3 ./HGCN/hgcn_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu False \
    --epochs 3 \
    --batch_size 20 \
    --lr 0.001 \
    --weight_decay 0.0005 \
    --dropout 0.5 \
    --gcn_output_dim 24

    python3 ./HGCN/multi_classification.py --dataset $DATASET
fi