#Reproducibility


The reported performance of our paper can reproduce using the following seeds and hyper-parameters.
To set seeds of our model, please jump to the initialization code of in the abcgraph_main.py file, from where you can set the the PyTorch random seed. 

## ABCGraph-Adversarial
#### For "Tencent" Data Set:
~~~
F1: 0.63928876

Seed: 197858

Parameters: --batch_size 500 --epochs 3 --lr 0.0003 --weight_decay 0.0005 --dis_hidden 16 --dropout 0.4
~~~

#### For "Cora" Data Set:

~~~
Micro F1: 0.86394558

Macro F1: 0.83796866

Seed: 613965

Parameters: --batch_size 400 --epochs 2 --lr 0.000400 --weight_decay 0.001000 --dis_hidden 24 --dropout 0.350000
~~~

#### For "Citeseer" Data Set:

~~~
Micro F1: 0.77235772

Macro F1: 0.70304816

Seed: 896714

Parameters: --batch_size 400 --epochs 4 --lr 0.000400 --weight_decay 0.000800 --dis_hidden 16 --dropout 0.400000
~~~


#### For "PubMed" Data Set:
~~~
Micro F1: 0.86219739

Macro F1: 0.86478358

Seed: 340324

Parameters: --batch_size 700 --epochs 3 --lr 0.000400 --weight_decay 0.000500 --dis_hidden 24 --dropout 0.350000
~~~

## ABCGraph-MLP

#### For "Cora" Data Set
~~~

Micro F1: 0.80952381

Macro F1: 0.78411502

Seed: 533871

Parameters: --epochs 5 --batch_size 128 --lr 0.001000 --weight_decay 0.000800 --dropout 0.200000 --gcn_output_dim 48  --encoder_hidfeat 16 --decoder_hidfeat 24

~~~

#### For "Citeseer" Data Set
~~~

Micro F1: 0.76422764

Macro F1: 0.67623898

Seed: 968358

Parameters: --epochs 3 --batch_size 64 --lr 0.001000 --weight_decay 0.005000 --dropout 0.200000 --gcn_output_dim 48  --encoder_hidfeat 16 --decoder_hidfeat 16
~~~

#### For "PubMed" Data Set
~~~

Micro F1: 0.84320298

Macro F1: 0.84727253

Seed: 604194

Parameters: --epochs 3 --batch_size 128 --lr 0.000100 --weight_decay 0.005000 --dropout 0.200000 --gcn_output_dim 48  --encoder_hidfeat 24 --decoder_hidfeat 16
~~~
