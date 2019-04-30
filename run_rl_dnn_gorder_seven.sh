#!/bin/bash
# to test runable of train_rl_gorder.py
# classification model steps = steps * n_iterations * n_trains

python ./gorder/rl/train_rl_gorder.py \
 --batch_size ##batch_size## \
 --steps ##steps## \
 --learning_rate ##learning_rate## \
 --w 5 \
 --n_hidden ##n_hidden## \
 --n_eval_data 1000 \
 --rl_learning_rate ##rl_learning_rate## \
 --n_iterations ##n_iterations## \
 --n_trains ##n_trains## \
 --tuning_rate ##tuning_rate## \
 --input_data_folder /opt/ml/disk/rl_data/fb \
 --model_dir /opt/ml/env/model_dir/rl_dnn_gorder_fb_fast 
