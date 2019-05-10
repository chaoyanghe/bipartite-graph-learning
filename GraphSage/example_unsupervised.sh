#!/usr/bin/env bash

rm -rf ./log_embedding/unsup-example_data/

python -m graphsage.unsupervised_train --train_prefix ./example_data/bipartite --identity_dim 128 --model graphsage_mean \
--max_total_steps 1000 --validate_iter 10 --base_log_dir ./log_embedding
