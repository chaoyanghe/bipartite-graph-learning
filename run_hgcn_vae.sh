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
    --gpu False \
    --epochs 1 \
    --batch_size 512 \
    --lr 0.0003 \
    --weight_decay 0.0005 \
    --dropout 0.5 \
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
    --epochs 1 \
    --batch_size 512 \
    --lr 0.0003 \
    --weight_decay 0.0005 \
    --dropout 0.5 \
    --gcn_output_dim 24 \
    --encoder_hidfeat 16 \
    --decoder_hidfeat 8

    python3 ./HGCN/multi_classification.py --dataset $DATASET --model $MODEL

elif [ "$DATASET" = "citeseer" ]
then
    echo $DATASET

    python3 ./HGCN/hgcn_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu False \
    --epochs 1 \
    --batch_size 512 \
    --lr 0.0003 \
    --weight_decay 0.0005 \
    --dropout 0.5 \
    --gcn_output_dim 24 \
    --encoder_hidfeat 16 \
    --decoder_hidfeat 8

    python3 ./HGCN/multi_classification.py --dataset $DATASET --model $MODEL
fi