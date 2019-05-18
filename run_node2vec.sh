#!/usr/bin/env bash

DATASET=$1
source ./Node2Vec/set_dynamic_lib.sh
python3 ./Node2Vec/gen_emb.py --dataset $DATASET

if [ "$DATASET" = "tencent" ]
then
    echo $DATASET
    python3 ./Node2Vec/multi_classification.py --dataset $DATASET
elif [ "$DATASET" = "cora" ]
then
    echo $DATASET
    python3 ./Node2Vec/binary_classification.py --dataset $DATASET
elif [ "$DATASET" = "citeseer" ]
then
    echo $DATASET
    python3 ./Node2Vec/binary_classification.py
fi

