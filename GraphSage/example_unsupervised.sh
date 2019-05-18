#!/usr/bin/env bash

rm -rf ./log_embedding/*
rm -rf ./out/*


python3 -m graphsage.unsupervised_train.py \
--train_prefix ./example_data//bipartite \
--model graphsage_mean \
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
--base_log_dir ./log_embedding \
--walk_len 5 \
--n_walks 50 \
--print_every 10

python3 ./multi-classification.py --dataset cora
