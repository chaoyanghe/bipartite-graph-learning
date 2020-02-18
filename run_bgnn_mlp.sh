#!/usr/bin/env bash


DATASET=$1
MODEL="mlp"


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
    --weight_decay 0.001 \
    --dropout 0.4 \
    --gcn_output_dim 24 \
    --encoder_hidfeat 16 \
    --decoder_hidfeat 8

    python3 ./BGNN/binary_classification.py --dataset $DATASET --model $MODEL

elif [ "$DATASET" = "cora" ]
then
    echo $DATASET

    python3 ./BGNN/bgnn_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu 0 \
    --epochs 3 \
    --batch_size 400 \
    --lr 0.0005 \
    --weight_decay 0.005 \
    --dropout 0.4 \
    --gcn_output_dim 16 \
    --encoder_hidfeat 24 \
    --decoder_hidfeat 8

    python3 ./BGNN/multi_classification.py --dataset $DATASET --model $MODEL

elif [ "$DATASET" = "citeseer" ]
then
    echo $DATASET

    python3 ./BGNN/bgnn_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu 0 \
    --epochs 3 \
    --batch_size 500 \
    --lr 0.0003 \
    --weight_decay 0.001 \
    --dropout 0.4 \
    --gcn_output_dim 24 \
    --encoder_hidfeat 16 \
    --decoder_hidfeat 8

    python3 ./BGNN/multi_classification.py --dataset $DATASET --model $MODEL
elif [ "$DATASET" = "pubmed" ]
then
    echo $DATASET

    python3 ./BGNN/bgnn_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu 0 \
    --epochs 10 \
    --batch_size 100 \
    --lr 0.0003 \
    --weight_decay 0.001 \
    --dropout 0.4 \
    --gcn_output_dim 24 \
    --encoder_hidfeat 16 \
    --decoder_hidfeat 8

    python3 ./BGNN/multi_classification.py --dataset $DATASET --model $MODEL
fi