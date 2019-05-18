# Adversarial Learning with Graph Convolutional Networks for Unsupervised Bipartite Graph Embedding


# Reproducibility


## Preparation
~~~
pip3 install -r requirements.txt
~~~

## Peproduciable Scripts Overview
|                | HGCN (GAN)                 | HGCN (VAE)                 | Node2Vec                    | GCN                         | GraphSAGE                   | GAE                         |
| :------------- | :----------:               | :----------:               | -----------:                | -----------:                | -----------:                | -----------:                |
| Platform       | MacOS/Linux                | MacOS/Linux                | Only Linux (*)              | Not Finished                | Not Finished                | Not Finished                |
| Tencent        | sh run_hgcn.sh tencent gan | sh run_hgcn.sh tencent vae | sh run_node2vec.sh tencent  | Not Finished                | Not Finished                | Not Finished                |
| Cora           | sh run_hgcn.sh cora gan    | sh run_hgcn.sh cora vae    | sh run_node2vec.sh cora     | Not Finished                | Not Finished                | Not Finished                |
| Citeseer       | sh run_hgcn.sh citeseer gan| sh run_hgcn.sh citeseer vae| sh run_node2vec.sh citeseer | Not Finished                | Not Finished                | Not Finished                |

*: For the Node2Vec model, its binary file is only ELF 64-bit LSB executable, x86-64, for GNU/Linux.

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

