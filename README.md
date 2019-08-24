# Adversarial Representation Learning on Large-Scale Bipartite Graphs

# Reproducibility


## Preparation
~~~
pip3 install -r requirements.txt
pip install --upgrade wandb
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
~~~

## Peproduciable Scripts Overview
|                | ABCGraph (Adversarial)         | ABCGraph (MLP)                 | Node2Vec                    | GCN                         | GraphSAGE                   | GAE                         |
| :------------- | :----------:                   | :----------:                   | -----------:                | -----------:                | -----------:                | -----------:                |
| Platform       | MacOS/Linux                    | MacOS/Linux                    | Only Linux (*)              | MacOS/Linux                 | MacOS/Linux                 | MacOS/Linux                 |
| Tencent        | sh run_abcgraph_adv.sh tencent | sh run_abcgraph_mlp.sh tencent | sh run_node2vec.sh tencent  | sh run_gcn.sh tencent       | sh run_graphsage.sh tencent | N/A (*)                     |
| Cora           | sh run_abcgraph_adv.sh cora    | sh run_abcgraph_mlp.sh cora    | sh run_node2vec.sh cora     | sh run_gcn.sh cora          | sh run_graphsage.sh cora    | sh run_gae.sh cora          |
| Citeseer       | sh run_abcgraph_adv.sh citeseer| sh run_abcgraph_mlp.sh citeseer| sh run_node2vec.sh citeseer | sh run_gcn.sh citeseer      | sh run_graphsage.sh citeseer| sh run_gae.sh citeseer      |
| PubMed         | sh run_abcgraph_adv.sh pubmed  | sh run_abcgraph_mlp.sh pubmed  | sh run_node2vec.sh pubmed   | sh run_gcn.sh pubmed        | sh run_graphsage.sh pubmed  | sh run_gae.sh pubmed        |

Only Linux (*): For the Node2Vec model, its binary file is only ELF 64-bit LSB executable, x86-64, for GNU/Linux.

N/A (*): For the GAE model, the code of the original GAE paper can not simply applied to the large-scale bipartite graph due to the memory constrain. 
To apply GAE to the large-scale graph data is another research topic, so we don't report the result in the "Tencent" dataset. From the other datasets, we can see that our model's performance is better than the GAE model. 

Background running: 
~~~
# ABCGraph (Adversarial)
nohup sh run_abcgraph_adv.sh tencent > log_run_abcgraph_adv_tencent.txt 2>&1 &
nohup sh run_abcgraph_adv.sh cora > log_run_abcgraph_adv_cora.txt 2>&1 &
nohup sh run_abcgraph_adv.sh citeseer > log_run_abcgraph_adv_citeseer.txt 2>&1 &
nohup sh run_abcgraph_adv.sh pubmed > log_run_abcgraph_adv_pubmed.txt 2>&1 &

# ABCGraph (MLP)
nohup sh run_abcgraph_mlp.sh tencent > log_run_abcgraph_mlp_tencent.txt 2>&1 &
nohup sh run_abcgraph_mlp.sh cora > log_run_abcgraph_mlp_cora.txt 2>&1 &
nohup sh run_abcgraph_mlp.sh citeseer > log_run_abcgraph_mlp_citeseer.txt 2>&1 &
nohup sh run_abcgraph_mlp.sh pubmed > log_run_abcgraph_mlp_pubmed.txt 2>&1 &

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
nohup sh run_gae.sh tencent > log_run_gae_tencent.txt 2>&1 &
nohup sh run_gae.sh cora > log_run_gae_cora.txt 2>&1 &
nohup sh run_gae.sh citeseer > log_run_gae_citeseer.txt 2>&1 &
nohup sh run_gae.sh pubmed > log_run_gae_pubmed.txt 2>&1 &

# AS-GCN
cd ASGCN/data
python tencent_dataset_loader.py
nohup python run_pubbmed.py --dataset tencent > running_asgcn.txt
~~~