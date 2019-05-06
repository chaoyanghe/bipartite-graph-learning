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

    logging.basicConfig(filename="./HGCN.log",
                        level=logging.DEBUG,
                        format=str(rank) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    hostname = socket.gethostname()
    logging.debug("#############process ID = " + str(rank) +
                  ", host name = " + hostname + "########" +
                  ", process ID = " + str(os.getpid()) +
                  ", process Name = " + str(psutil.Process(os.getpid())))

    # 972 parallel processes
    hpo_batch_size = [500, 1000, 1500]  # 3
    hpo_epochs = [3, 5, 8, 10]  # 4
    hpo_lr = [0.0003, 0.0005, 0.001]  # 3
    hpo_weight_decay = [0.001, 0.0001, 0.00001]  # 3
    hpo_dis_hidden = [16, 24, 32]  # 3
    hpo_dropout = [0.4, 0.5, 0.6]  # 3

    hpo_cnt = 0
    paras = dict()
    for batch_size in hpo_batch_size:
        for epochs in hpo_epochs:
            for lr in hpo_lr:
                for weight_decay in hpo_weight_decay:
                    for dis_hidden in hpo_dis_hidden:
                        for dropout in hpo_dis_hidden:
                            paras[hpo_cnt] = (batch_size, epochs, lr, weight_decay, dis_hidden, dropout)
                            hpo_cnt += 1

    (batch_size, epochs, lr, weight_decay, dis_hidden, dropout) = paras[rank]

    hgcn_cmd = "/mnt/shared/etc/anaconda3/bin/python3 ./HGCN/hgcn_main.py --model gan_gcn --gpu False --batch_size %d --epochs %d --lr %f --weight_decay %f --dis_hidden %d --dropout %f --rank %d" % (
        batch_size,
        epochs,
        lr,
        weight_decay,
        dis_hidden,
        dropout,
        rank)
    os.system(hgcn_cmd)

    lr_cmd = "/mnt/shared/etc/anaconda3/bin/python3 ./HGCN/classification.py --rank %s" % rank
    os.system(lr_cmd)
