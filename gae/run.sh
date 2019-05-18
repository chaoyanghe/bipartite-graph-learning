#!/usr/bin/env bash

rm -rf gae.log

python train.py --model gcn_vae --dataset cora --epochs 100