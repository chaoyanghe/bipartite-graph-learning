#!/usr/bin/env bash


DATASET=$1
MODEL="vae"

echo ./out/hgcn-$MODEL/$DATASET

rm -rf ./out/hgcn-$MODEL/$DATASET

if [ "$DATASET" = "tencent" ]
then
    echo $DATASET

    python3 ./HGCN/hgcn_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu True \
    --epochs 3 \
    --batch_size 500 \
    --lr 0.0003 \
    --weight_decay 0.001 \
    --dropout 0.4 \
    --gcn_output_dim 24 \
    --encoder_hidfeat 16 \
    --decoder_hidfeat 8

    python3 ./HGCN/binary_classification.py --dataset $DATASET --model $MODEL

elif [ "$DATASET" = "cora" ]
then
    echo $DATASET

    python3 ./HGCN/hgcn_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu False \
    --epochs 3 \
    --batch_size 10 \
    --lr 0.003 \
    --weight_decay 0.001 \
    --dropout 0.4 \
    --gcn_output_dim 24 \
    --encoder_hidfeat 24 \
    --decoder_hidfeat 24

    python3 ./HGCN/multi_classification.py --dataset $DATASET --model $MODEL

elif [ "$DATASET" = "citeseer" ]
then
    echo $DATASET

    python3 ./HGCN/hgcn_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu False \
    --epochs 3 \
    --batch_size 500 \
    --lr 0.0003 \
    --weight_decay 0.001 \
    --dropout 0.4 \
    --gcn_output_dim 24 \
    --encoder_hidfeat 16 \
    --decoder_hidfeat 8

    python3 ./HGCN/multi_classification.py --dataset $DATASET --model $MODEL
fi