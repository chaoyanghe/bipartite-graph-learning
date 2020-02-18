#!/usr/bin/env bash


DATASET=$1
MODEL="adv"


rm -rf ./out/abcgraph-$MODEL/$DATASET


if [ "$DATASET" = "tencent" ]
then
    echo $DATASET

    python3 ./BGNN/bgnn_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu 0 \
    --epochs 3 \
    --batch_size 500 \
    --lr 0.0003 \
    --weight_decay 0.0005 \
    --dropout 0.4 \
    --gcn_output_dim 16 \
    --layer_depth 3

    python3 ./BGNN/binary_classification.py --dataset $DATASET --model $MODEL

elif [ "$DATASET" = "cora" ]
then
    echo $DATASET

    python3 ./BGNN/bgnn_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu 0 \
    --epochs 2 \
    --batch_size 400 \
    --lr 0.0004 \
    --weight_decay 0.001 \
    --dropout 0.35 \
    --gcn_output_dim 24 \
    --layer_depth 3

    python3 ./BGNN/multi_classification.py --dataset $DATASET --model $MODEL

elif [ "$DATASET" = "citeseer" ]
then
    echo $DATASET

    python3 ./BGNN/bgnn_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu 0 \
    --epochs 4 \
    --batch_size 400 \
    --lr 0.0004 \
    --weight_decay 0.000800 \
    --dropout 0.400000 \
    --gcn_output_dim 16 \
    --layer_depth 1

    python3 ./BGNN/multi_classification.py --dataset $DATASET --model $MODEL
elif [ "$DATASET" = "pubmed" ]
then
    echo $DATASET

    python3 ./BGNN/bgnn_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu 0 \
    --epochs 3 \
    --batch_size 700 \
    --lr 0.0004 \
    --weight_decay 0.0005 \
    --dropout 0.35 \
    --gcn_output_dim 24 \
    --layer_depth 1

    python3 ./BGNN/multi_classification.py --dataset $DATASET --model $MODEL

fi