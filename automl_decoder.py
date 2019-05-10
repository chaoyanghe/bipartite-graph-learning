import argparse
import logging
import os
import queue
import random
import socket
import sys

import psutil
import setproctitle
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == "__main__":
    setproctitle.setproctitle("HGCN:" + str(rank))

    logging.basicConfig(filename="./HGCN.log_embedding",
                        level=logging.INFO,
                        format=str(rank) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    hostname = socket.gethostname()
    logging.debug("#############process ID = " + str(rank) +
                  ", host name = " + hostname + "########" +
                  ", process ID = " + str(os.getpid()) +
                  ", process Name = " + str(psutil.Process(os.getpid())))

    # 972 parallel processes
    hpo_batch_size = [400, 500, 600]  # 3
    hpo_epochs = [2, 3, 4]  # 3
    hpo_lr = [0.0002, 0.0003]  # 2
    hpo_weight_decay = [0.001, 0.0005]  # 2
    hpo_dis_hidden = [16, 20, 24]  # 3
    hpo_dropout = [0.35, 0.4, 0.45]  # 3
    hpo_gcn_output = [16, 20, 24]  # 3

    paras = dict()
    cnt = 0

    for gcn_output_dim in hpo_gcn_output:
        for batch_size in hpo_batch_size:
            for epochs in hpo_epochs:
                for lr in hpo_lr:
                    for weight_decay in hpo_weight_decay:
                        for dis_hidden in hpo_dis_hidden:
                            for dropout in hpo_dropout:
                                paras[cnt] = (batch_size, epochs, lr, weight_decay, dis_hidden, dropout, gcn_output_dim)
                                cnt += 1

    (batch_size, epochs, lr, weight_decay, dis_hidden, dropout, gcn_output_dim) = paras[rank]

    print("start hgcn_cmd")
    hgcn_cmd = "/mnt/shared/etc/anaconda3/bin/python3 /mnt/shared/home/bipartite-graph-learning/HGCN/hgcn_main.py " \
               "--model decoder_gcn --gpu False --batch_size %d --epochs %d --lr %f --weight_decay %f --dis_hidden %d" \
               " --dropout %f --gcn_output_dim %d --rank %d" % (
                   batch_size,
                   epochs,
                   lr,
                   weight_decay,
                   dis_hidden,
                   dropout,
                   gcn_output_dim,
                   rank)
    os.system(hgcn_cmd)
    print("end hgcn_cmd")

    print("start lr_cmd")
    lr_cmd = "/mnt/shared/etc/anaconda3/bin/python3 /mnt/shared/home/bipartite-graph-learning/HGCN/classification.py " \
             "--rank %s" % rank
    os.system(lr_cmd)
    print("end lr_cmd")
