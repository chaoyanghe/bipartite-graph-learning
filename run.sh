#!/usr/bin/env bash

# HGCN (GAN)
nohup sh run_hgcn_gan.sh tencent > log_run_hgcn_gan_tencent.txt 2>&1 &
nohup sh run_hgcn_gan.sh cora > log_run_hgcn_gan_cora.txt 2>&1 &
nohup sh run_hgcn_gan.sh citeseer > log_run_hgcn_gan_citeseer.txt 2>&1 &

# HGCN (VAE)
nohup sh run_hgcn_vae.sh tencent > log_run_hgcn_vae_tencent.txt 2>&1 &
nohup sh run_hgcn_vae.sh cora > log_run_hgcn_vae_cora.txt 2>&1 &
nohup sh run_hgcn_vae.sh citeseer > log_run_hgcn_vae_citeseer.txt 2>&1 &

# Node2Vec
nohup sh run_node2vec.sh tencent > log_run_node2vec_tencent.txt 2>&1 &
nohup sh run_node2vec.sh cora > log_run_node2vec_cora.txt 2>&1 &
nohup sh run_node2vec.sh citeseer > log_run_node2vec_citeseer.txt 2>&1 &

# GCN
nohup sh run_gcn.sh tencent > log_run_gcn_tencent.txt 2>&1 &
nohup sh run_gcn.sh cora > log_run_gcn_cora.txt 2>&1 &
nohup sh run_gcn.sh citeseer > log_run_gcn_citeseer.txt 2>&1 &

# GraphSAGE
nohup sh run_graphsage.sh tencent > log_run_graphsage_tencent.txt 2>&1 &
nohup sh run_graphsage.sh cora > log_run_graphsage_cora.txt 2>&1 &
nohup sh run_graphsage.sh citeseer > log_run_graphsage_citeseer.txt 2>&1 &

# GAE
nohup sh run_gae.sh cora > log_run_gae_cora.txt 2>&1 &
nohup sh run_gae.sh citeseer > log_run_gae_citeseer.txt 2>&1 &