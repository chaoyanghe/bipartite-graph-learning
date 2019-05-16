#!/usr/bin/env bash

rm -rf ./log_embedding/*
rm -rf ./out/*


python -m graphsage.unsupervised_train \
--train_prefix ./example_data/bipartite \
--identity_dim 16 \
--model graphsage_mean \
--max_total_steps 100000000 \
--validate_iter 5 \
--epochs 10 \
--learning_rate 0.0001 \
--dropout 0.0 \
--weight_decay 0.5 \
--samples_1 25 \
--samples_2 10 \
--dim_1 24 \
--dim_2 4 \
--neg_sample_size 20 \
--batch_size 512 \
--save_embeddings True \
--base_log_dir ./log_embedding \
--walk_len 5 \
--n_walks 50

python ./classification.py
