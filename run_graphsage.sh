#!/usr/bin/env bash

pip install networkx==1.11

DATASET=$1
MODEL=$2
if [ -z "$2" ]
then
    echo "empty model name"
    MODEL="graphsage_mean"
fi

echo ./out/graphsage/$DATASET
rm -rf ./out/graphsage/$DATASET


## only execute once
cd ./GraphSage/example_data/
sh run_graphsage_data_loader.sh $DATASET
cd ../..

if [ "$DATASET" = "tencent" ]
then
   echo $DATASET
   cd ./GraphSage
   python3 -m graphsage.unsupervised_train \
   --dataset $DATASET \
   --train_prefix ./example_data/$DATASET/bipartite \
   --base_log_dir ./log_embedding \
   --model $MODEL \
   --max_total_steps 100000000 \
   --validate_iter 500000 \
   --epochs 1 \
   --learning_rate 0.0001 \
   --dropout 0.0 \
   --weight_decay 0.0005 \
   --samples_1 25 \
   --samples_2 10 \
   --dim_1 16 \
   --dim_2 8 \
   --neg_sample_size 20 \
   --batch_size 1024 \
   --save_embeddings True \
   --walk_len 5 \
   --n_walks 50 \
   --print_every 10

   cd ..
   python3 ./GraphSage/classification.py
elif [ "$DATASET" = "cora" ]
then
   echo $DATASET
   cd ./GraphSage
   python3 -m graphsage.unsupervised_train \
   --dataset $DATASET \
   --train_prefix ./example_data/$DATASET/bipartite \
   --base_log_dir ./log_embedding \
   --model $MODEL \
   --max_total_steps 100000000 \
   --validate_iter 500000 \
   --epochs 1 \
   --learning_rate 0.0001 \
   --dropout 0.0 \
   --weight_decay 0.0005 \
   --samples_1 25 \
   --samples_2 10 \
   --dim_1 16 \
   --dim_2 8 \
   --neg_sample_size 20 \
   --batch_size 1024 \
   --save_embeddings True \
   --walk_len 5 \
   --n_walks 50 \
   --print_every 10

   cd ..
   python3 ./GraphSage/multi_classification.py --dataset $DATASET
elif [ "$DATASET" = "citeseer" ]
then
   echo $DATASET
   cd ./GraphSage
   python3 -m graphsage.unsupervised_train \
   --dataset $DATASET \
   --train_prefix ./example_data/$DATASET/bipartite \
   --base_log_dir ./log_embedding \
   --model $MODEL \
   --max_total_steps 100000000 \
   --validate_iter 500000 \
   --epochs 1 \
   --learning_rate 0.0001 \
   --dropout 0.0 \
   --weight_decay 0.0005 \
   --samples_1 25 \
   --samples_2 10 \
   --dim_1 16 \
   --dim_2 8 \
   --neg_sample_size 20 \
   --batch_size 1024 \
   --save_embeddings True \
   --walk_len 5 \
   --n_walks 50 \
   --print_every 10

   cd ..
   python3 ./GraphSage/multi_classification.py --dataset $DATASET
elif [ "$DATASET" = "pubmed" ]
then
   echo $DATASET
   cd ./GraphSage
   python3 -m graphsage.unsupervised_train \
   --dataset $DATASET \
   --train_prefix ./example_data/$DATASET/bipartite \
   --base_log_dir ./log_embedding \
   --model $MODEL \
   --max_total_steps 100000000 \
   --validate_iter 500000 \
   --epochs 1 \
   --learning_rate 0.0001 \
   --dropout 0.0 \
   --weight_decay 0.0005 \
   --samples_1 25 \
   --samples_2 10 \
   --dim_1 16 \
   --dim_2 8 \
   --neg_sample_size 20 \
   --batch_size 1024 \
   --save_embeddings True \
   --walk_len 5 \
   --n_walks 50 \
   --print_every 10

   cd ..
   python3 ./GraphSage/multi_classification.py --dataset $DATASET
fi

pip install networkx==2.2