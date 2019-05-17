#!/usr/bin/env bash

rm -rf gae.log

python train.py --model gcn_vae --epochs 3