#!/usr/bin/env bash


DATASET=$1
MODEL="adv"

echo ./out/abcgraph-$MODEL/$DATASET

rm -rf ./out/abcgraph-$MODEL/$DATASET

export WANDB_RESUME=allow
export WANDB_RUN_ID=$(python3 -c 'import wandb; wandb.util.generate_id();')
echo $WANDB_RUN_ID

if [ "$DATASET" = "tencent" ]
then
    echo $DATASET

    python3 ./ABCGraph/abcgraph_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu False \
    --epochs 3 \
    --batch_size 500 \
    --lr 0.0003 \
    --weight_decay 0.0005 \
    --dropout 0.4 \
    --gcn_output_dim 16

    python3 ./ABCGraph/binary_classification.py --dataset $DATASET --model $MODEL

elif [ "$DATASET" = "cora" ]
then
    echo $DATASET

    python3 ./ABCGraph/abcgraph_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu False \
    --epochs 2 \
    --batch_size 400 \
    --lr 0.0004 \
    --weight_decay 0.001 \
    --dropout 0.35 \
    --gcn_output_dim 24

    python3 ./ABCGraph/multi_classification.py --dataset $DATASET --model $MODEL

elif [ "$DATASET" = "citeseer" ]
then
    echo $DATASET

    python3 ./ABCGraph/abcgraph_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu False \
    --epochs 4 \
    --batch_size 400 \
    --lr 0.0004 \
    --weight_decay 0.000800 \
    --dropout 0.400000 \
    --gcn_output_dim 16

    python3 ./ABCGraph/multi_classification.py --dataset $DATASET --model $MODEL
elif [ "$DATASET" = "pubmed" ]
then
    echo $DATASET

    python3 ./ABCGraph/abcgraph_main.py \
    --dataset $DATASET \
    --model $MODEL \
    --gpu False \
    --epochs 20 \
    --batch_size 700 \
    --lr 0.0004 \
    --weight_decay 0.0005 \
    --dropout 0.35 \
    --gcn_output_dim 24

    python3 ./ABCGraph/multi_classification.py --dataset $DATASET --model $MODEL

fi