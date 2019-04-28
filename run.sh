#!/usr/bin/env bash

python ./main.py

model="node2vec"
#model="GCN"
#model="GraphSage"
#model="HGCN"

if [ $model == node2vec ]
then

    cd ./Node2Vec/
    sh ./run_node2vec.sh
    cd ..

elif [ $model == GCN ]
then

    echo "TODO"

elif [ $model == GraphSage ]
then

    cd ./GraphSage/
    sh ./example_supervised.sh
    cd ..

elif [ $model == HGCN ]

    cd ./HGCN
    sh ./hgcn.sh
    cd ..

then
    echo "Nothing to do"
fi
