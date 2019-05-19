#!/usr/bin/env bash



DATASET=$1
rm -rf out/gae/$DATASET/*

if [ "$DATASET" = "tencent" ]
then
    echo $DATASET
    echo "OOM"

elif [ "$DATASET" = "cora" ]
then
    echo $DATASET

    python3 ./gae/train.py \
    --dataset cora \
    --model gcn_vae \
    --epochs 500 \
    --hidden1 64 \
    --hidden2 32 \
    --weight_decay 0.005 \
    --dropout 0.4 \
    --learning_rate 0.001

    python3 ./gae/classification.py --dataset $DATASET


elif [ "$DATASET" = "citeseer" ]
then
    echo $DATASET


    python3 ./gae/train.py \
    --dataset citeseer \
    --model gcn_vae \
    --epochs 500 \
    --hidden1 64 \
    --hidden2 32 \
    --weight_decay 0.005 \
    --dropout 0.4 \
    --learning_rate 0.0001
    python3 ./gae/classification.py --dataset $DATASET

elif [ "$DATASET" = "pubmed" ]
then
    echo $DATASET


    python3 ./gae/train.py \
    --dataset citeseer \
    --model gcn_vae \
    --epochs 500 \
    --hidden1 64 \
    --hidden2 32 \
    --weight_decay 0.005 \
    --dropout 0.4 \
    --learning_rate 0.0001
    python3 ./gae/classification.py --dataset $DATASET
fi


