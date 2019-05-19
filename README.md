# Adversarial Learning with Graph Convolutional Networks for Unsupervised Bipartite Graph Embedding


## Model Performance Comparision
|                | HGCN (GAN)                 | HGCN (VAE)                 | Node2Vec                    | GCN                         | GraphSAGE                   | GAE                         | Pure Node Attribute         |
| :------------- | :----------:               | :----------:               | -----------:                | -----------:                | -----------:                | -----------:                | -----------:                |
| Metrics        | F1 / Accuracy              | F1 / Accuracy              | F1 / Accuracy               | F1 / Accuracy               | F1 / Accuracy               | F1 / Accuracy               | F1 / Accuracy               |
| Tencent        | 0.625498 / 0.960591        | 0.503524 / 0.875297        | 0.591455 / 0.961049         | Not Finished                | Not Finished                | N/A (*)                     | 0.497346 / 0.881214         |
| Cora           | 0.801590 / 0.823129        | 0.215116 / 0.285714        | 0.756000 / 0.789116         | 0.762734 / 0.782313         | 0.800763 / 0.823129         | 0.754289 / 0.782313         | 0.758143 / 0.789116         |
| Citeseer       | 0.687996 / 0.756098        | 0.222869 / 0.349593        | 0.630637 / 0.723577         | 0.626836 / 0.715447         | 0.665325 / 0.747967         | 0.644748 / 0.731707         | 0.620755 / 0.707317         |
| PubMed         | 0.854867 / 0.851397       | 0.822525 / 0.815270        | 0.839086 / 0.833892         | 0.843036 / 0.838361         | 0.842696 / 0.838361         | 0.828317 / 0.823091         | 0.842813 / 0.838361         |



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
nohup sh run_hgcn_gan.sh pubmed > log_run_hgcn_gan_pubmed.txt 2>&1 &

# HGCN (VAE)
nohup sh run_hgcn_vae.sh tencent > log_run_hgcn_vae_tencent.txt 2>&1 &
nohup sh run_hgcn_vae.sh cora > log_run_hgcn_vae_cora.txt 2>&1 &
nohup sh run_hgcn_vae.sh citeseer > log_run_hgcn_vae_citeseer.txt 2>&1 &
nohup sh run_hgcn_vae.sh pubmed > log_run_hgcn_vae_pubmed.txt 2>&1 &

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


node2vec


