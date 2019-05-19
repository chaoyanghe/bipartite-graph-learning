# Adversarial Learning with Graph Convolutional Networks for Unsupervised Bipartite Graph Embedding


## Model Performance Comparision
|                | HGCN (GAN)                 | HGCN (VAE)                 | Node2Vec                    | GCN                         | GraphSAGE                   | GAE                         |
| :------------- | :----------:               | :----------:               | -----------:                | -----------:                | -----------:                | -----------:                |
| Metrics        | F1                         | F1                         | F1                          | F1                          | F1                          | F1                          |
| Tencent        |                            | sh run_hgcn.sh tencent vae | sh run_node2vec.sh tencent  | Not Finished                | Not Finished                | N/A (*)                     |
| Cora           | 0.813461                   | 0.302895                   | 0.745663                    | Not Finished                | 0.728645                    | 0.736460                    |
| Citeseer       | 0.675909                   | 0.207148                   | 0.645093                    | Not Finished                | 0.739309                    | 0.574944                    |



# Reproducibility


## Preparation
~~~
pip3 install -r requirements.txt
~~~

## Peproduciable Scripts Overview
|                | HGCN (GAN)                 | HGCN (VAE)                 | Node2Vec                    | GCN                         | GraphSAGE                   | GAE                         |
| :------------- | :----------:               | :----------:               | -----------:                | -----------:                | -----------:                | -----------:                |
| Platform       | MacOS/Linux                | MacOS/Linux                | Only Linux (*)              | MacOS/Linux                 | MacOS/Linux                 | MacOS/Linux                |
| Tencent        | sh run_hgcn_gan.sh tencent | sh run_hgcn_vae.sh tencent | sh run_node2vec.sh tencent  | Not Finished                | Not Finished                | N/A (*)                     |
| Cora           | sh run_hgcn_gan.sh cora    | sh run_hgcn_vae.sh cora    | sh run_node2vec.sh cora     | Not Finished                | sh run_graphsage.sh cora    | sh run_gae.sh cora          |
| Citeseer       | sh run_hgcn_gan.sh citeseer| sh run_hgcn_vae.sh citeseer| sh run_node2vec.sh citeseer | Not Finished                | sh run_graphsage.sh citeseer| sh run_gae.sh citeseer      |

Only Linux (*): For the Node2Vec model, its binary file is only ELF 64-bit LSB executable, x86-64, for GNU/Linux.

N/A (*): For the GAE model, the code of the original GAE paper can not simply applied to the large-scale bipartite graph due to the memory constrain. 
To apply GAE to the large-scale graph data is another research topic, so we don't report the result in the "Tencent" dataset. From the other datasets, we can see that our model's performance is better than the GAE model. 

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

