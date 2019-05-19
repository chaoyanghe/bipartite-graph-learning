#!/usr/bin/env bash


DATASET=$1
MODEL="gan"

echo ./out/hgcn-$MODEL/$DATASET

rm -rf ./out/hgcn-$MODEL/$DATASET

if [ "$DATASET" = "tencent" ]
then
    echo $DATASET

    python3 ./HGCN/hgcn_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu True \
    --epochs 5 \
    --batch_size 5000 \
    --lr 0.003 \
    --weight_decay 0.001 \
    --dropout 0.5 \
    --gcn_output_dim 24

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
    --lr 0.0003 \
    --weight_decay 0.0005 \
    --dropout 0.5 \
    --gcn_output_dim 24

    python3 ./HGCN/multi_classification.py --dataset $DATASET --model $MODEL

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

    python3 ./HGCN/multi_classification.py --dataset $DATASET --model $MODEL
elif [ "$DATASET" = "pubmed" ]
then
    echo $DATASET

    python3 ./HGCN/hgcn_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu False \
    --epochs 3 \
    --batch_size 100 \
    --lr 0.0003 \
    --weight_decay 0.0005 \
    --dropout 0.5 \
    --gcn_output_dim 24

    python3 ./HGCN/multi_classification.py --dataset $DATASET --model $MODEL

fi