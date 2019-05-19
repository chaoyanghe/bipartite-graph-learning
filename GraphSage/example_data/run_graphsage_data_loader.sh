#!/usr/bin/env bash


if [ "$1" = "cora" ]
then
    echo $1
    rm -rf ./cora/*
    python3 graphsage_data_loader_cora.py
elif [ "$1" = "tencent" ]
then
    echo $1
    rm -rf ./tencent/*
    python3 graphsage_data_loader_tencent.py
elif [ "$1" = "citeseer" ]
then
    echo $1
    rm -rf ./citeseer/*
    python3 graphsage_data_loader_citeseer.py
fi


python3 random_walk.py \
--dataset $1 \
--walk_len 5 \
--n_walks 50


