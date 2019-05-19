#!/usr/bin/env bash

rm -rf ./out/graphsage/*


pip3 install networkx==1.11


echo $1
DATASET=$1

if [ "$DATASET" = "" ]
then
   DATASET="cora"
   echo $DATASET
else
   echo $DATASET
fi


## only execute once
cd ./GraphSage/example_data/
sh run_graphsage_data_loader.sh $DATASET
cd ../..


cd ./GraphSage
python3 -m graphsage.unsupervised_train \
--dataset $DATASET \
--train_prefix ./example_data/$DATASET/bipartite \
--base_log_dir ./log_embedding \
--model gcn \
--max_total_steps 100000000 \
--validate_iter 500000 \
--epochs 1 \
--learning_rate 0.0001 \
--dropout 0.0 \
--weight_decay 0.0005 \
--samples_1 25 \
--samples_2 10 \
--dim_1 16 \
--dim_2 4 \
--neg_sample_size 20 \
--batch_size 512 \
--save_embeddings True \
--walk_len 5 \
--n_walks 50 \
--print_every 10

cd ..


python3 ./GraphSage/multi-classification.py --dataset $DATASET --model $MODEL


pip3 install networkx==2.2