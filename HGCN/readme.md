#Reproducibility


The reported performance of our paper can reproduce using the following seeds and hyper-parameters.
To set seeds of our model, please jump to the initialization code of in the hgcn_main.py file, from where you can set the the PyTorch random seed. 

### For "Tencent" Data Set:


### For "Cora" Data Set:

Micro F1: 0.86394558

Macro F1: 0.83796866

Seed: 613965

Hyper-Parameters: --batch_size 400 --epochs 2 --lr 0.000400 --weight_decay 0.001000 --dis_hidden 24 --dropout 0.350000

### For "Citeseer" Data Set:

Micro F1: 0.77235772

Macro F1: 0.70304816

Seed: 896714

Parameters: --batch_size 400 --epochs 4 --lr 0.000400 --weight_decay 0.000800 --dis_hidden 16 --dropout 0.400000


### For "PubMed" Data Set: