#!/usr/bin/env bash

python -m graphsage.unsupervised_train --train_prefix ./example_data/bipartite --identity_dim 64 --model graphsage_mean \
--max_total_steps 1000 --validate_iter 10
