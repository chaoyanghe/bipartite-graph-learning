# Adversarial Learning with Graph Convolutional Networks for Unsupervised Bipartite Graph Embedding


## Model Performance Comparision
|                | HGCN (GAN)                 | HGCN (VAE)                 | Node2Vec                    | GCN                         | GraphSAGE                   | GAE                         |
| :------------- | :----------:               | :----------:               | -----------:                | -----------:                | -----------:                | -----------:                |
| Metrics        | F1 / Accuracy              | F1 / Accuracy              | F1 / Accuracy               | F1 / Accuracy               | F1 / Accuracy               | F1 / Accuracy               |
| Tencent        | 0.541600 /                 | 0.50312128 /               | 0.657056                    | Not Finished                | Not Finished                | N/A (*)                     |
| Cora           | 0.813461 /                 | 0.302895 /                 | 0.763529                    | 0.740193                    | 0.686367                    | 0.731786                    |
| Citeseer       | 0.673574 /                 | 0.246943 /                 | 0.645093                    | 0.596460                    | 0.621766                    | 0.603029                    |
| PubMed         | 0.875158 / 0.874746        | 0.832246 / 0.831136        | 0.645093                    | 0.596460                    | 0.621766                    | 0.603029                    |



# Reproducibility


## Preparation
~~~
pip3 install -r requirements.txt
~~~

## Peproduciable Scripts Overview
|                | HGCN (GAN)                 | HGCN (VAE)                 | Node2Vec                    | GCN                         | GraphSAGE                   | GAE                         |
| :------------- | :----------:               | :----------:               | -----------:                | -----------:                | -----------:                | -----------:                |
| Platform       | MacOS/Linux                | MacOS/Linux                | Only Linux (*)              | MacOS/Linux                 | MacOS/Linux                 | MacOS/Linux                |
| Tencent        | sh run_hgcn_gan.sh tencent | sh run_hgcn_vae.sh tencent | sh run_node2vec.sh tencent  | sh run_gcn.sh tencent       | sh run_graphsage.sh tencent | N/A (*)                     |
| Cora           | sh run_hgcn_gan.sh cora    | sh run_hgcn_vae.sh cora    | sh run_node2vec.sh cora     | sh run_gcn.sh cora          | sh run_graphsage.sh cora    | sh run_gae.sh cora          |
| Citeseer       | sh run_hgcn_gan.sh citeseer| sh run_hgcn_vae.sh citeseer| sh run_node2vec.sh citeseer | sh run_gcn.sh citeseer      | sh run_graphsage.sh citeseer| sh run_gae.sh citeseer      |
| PubMed         | sh run_hgcn_gan.sh pubmed  | sh run_hgcn_vae.sh pubmed  | sh run_node2vec.sh pubmed   | sh run_gcn.sh pubmed        | sh run_graphsage.sh pubmed  | sh run_gae.sh pubmed        |

Only Linux (*): For the Node2Vec model, its binary file is only ELF 64-bit LSB executable, x86-64, for GNU/Linux.

N/A (*): For the GAE model, the code of the original GAE paper can not simply applied to the large-scale bipartite graph due to the memory constrain. 
To apply GAE to the large-scale graph data is another research topic, so we don't report the result in the "Tencent" dataset. From the other datasets, we can see that our model's performance is better than the GAE model. 

Background running: 
~~~
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

~~~

## HGCN
1. run HGCN model on the tencent dataset.
~~~
sh run_hgcn.sh tencent
~~~
It may take XX minutes to get the result.

2. run HGCN model on the cora dataset:
~~~
sh run_hgcn.sh cora
~~~
It may take XX minutes to get the result.

3. run HGCN model on the citeseer dataset:
~~~
sh run_hgcn.sh citeseer
~~~


## Node2Vec
####1. run Node2Vec model on the "tencent" dataset.
~~~
sh run_.sh tencent
~~~
It may take XX minutes to get the result.

####2. run GraphSAGE model on the "cora" dataset:
~~~
sh run_graphsage.sh cora
~~~
It may take XX minutes to get the result.

####3. run GraphSAGE model on the "citeseer" dataset:
~~~
sh run_graphsage.sh citeseer
~~~
It may take XX minutes to get the result.


## GCN
####1. run GraphSAGE model on the "tencent" dataset.
~~~
sh run_graphsage.sh tencent
~~~
It may take XX minutes to get the result.

####2. run GraphSAGE model on the "cora" dataset:
~~~
sh run_graphsage.sh cora
~~~
It may take XX minutes to get the result.

####3. run GraphSAGE model on the "citeseer" dataset:
~~~
sh run_graphsage.sh citeseer
~~~
It may take XX minutes to get the result.


## GraphSASE
####1. run GraphSAGE model on the "tencent" dataset.
~~~
sh run_graphsage.sh tencent
~~~
It may take XX minutes to get the result.

####2. run GraphSAGE model on the "cora" dataset:
~~~
sh run_graphsage.sh cora
~~~
It may take XX minutes to get the result.

####3. run GraphSAGE model on the "citeseer" dataset:
~~~
sh run_graphsage.sh citeseer
~~~
It may take XX minutes to get the result.



## one command to get all the result
~~~
sh run.sh
~~~

