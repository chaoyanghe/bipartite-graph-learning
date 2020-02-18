#!/usr/bin/env bash

# BGNN (Adv)
nohup sh run_bgnn_adv.sh tencent > log_run_abcgraph_adv_tencent.txt 2>&1 &
nohup sh run_bgnn_adv.sh cora > log_run_abcgraph_adv_cora.txt 2>&1 &
nohup sh run_bgnn_adv.sh citeseer > log_run_abcgraph_adv_citeseer.txt 2>&1 &
nohup sh run_bgnn_adv.sh pubmed > log_run_abcgraph_adv_pubmed.txt 2>&1 &

# BGNN (MLP)
nohup sh run_bgnn_mlp.sh tencent > log_run_abcgraph_mlp_tencent.txt 2>&1 &
nohup sh run_bgnn_mlp.sh cora > log_run_abcgraph_mlp_cora.txt 2>&1 &
nohup sh run_bgnn_mlp.sh citeseer > log_run_abcgraph_mlp_citeseer.txt 2>&1 &
nohup sh run_bgnn_mlp.sh pubmed > log_run_abcgraph_mlp_pubmed.txt 2>&1 &

# Node2Vec
nohup sh run_node2vec.sh tencent > log_run_node2vec_tencent.txt 2>&1 &
nohup sh run_node2vec.sh cora > log_run_node2vec_cora.txt 2>&1 &
nohup sh run_node2vec.sh citeseer > log_run_node2vec_citeseer.txt 2>&1 &
nohup sh run_node2vec.sh pubmed > log_run_node2vec_pubmed.txt 2>&1 &

# GCN
nohup sh run_gcn.sh tencent > log_run_gcn_tencent.txt 2>&1 &
nohup sh run_gcn.sh cora > log_run_gcn_cora.txt 2>&1 &
nohup sh run_gcn.sh citeseer > log_run_gcn_citeseer.txt 2>&1 &
nohup sh run_gcn.sh pubmed > log_run_gcn_pubmed.txt 2>&1 &

# GraphSAGE
nohup sh run_graphsage.sh tencent > log_run_graphsage_tencent.txt 2>&1 &
nohup sh run_graphsage.sh cora > log_run_graphsage_cora.txt 2>&1 &
nohup sh run_graphsage.sh citeseer > log_run_graphsage_citeseer.txt 2>&1 &
nohup sh run_graphsage.sh pubmed > log_run_graphsage_pubmed.txt 2>&1 &

# GAE
nohup sh run_gae.sh cora > log_run_gae_cora.txt 2>&1 &
nohup sh run_gae.sh citeseer > log_run_gae_citeseer.txt 2>&1 &
nohup sh run_gae.sh pubmed > log_run_gae_pubmed.txt 2>&1 &