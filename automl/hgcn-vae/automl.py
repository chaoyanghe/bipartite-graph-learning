import logging
import os
import socket

import psutil
import setproctitle
from mpi4py import MPI

import conf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == "__main__":
    dataset_name = conf.dataset_name
    model_name = conf.model_name

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
    hpo_batch_size = [64, 128, 400]  # 3
    hpo_dropout = [0.0, 0.2, 0.4]  # 3
    hpo_epochs = [3, 5, 10]  # 3
    hpo_lr = [0.001, 0.0001]  # 2
    hpo_weight_decay = [0.05, 0.005, 0.0008]  # 3
    hpo_gcn_out_dim = [16, 32, 48]  # 3
    hpo_encode_hid_fea_dim = [16, 24]  # 2
    hpo_decoder_hid_fea_dim = [8]  # 1

    hpo_cnt = 0
    paras = dict()

    for batch_size in hpo_batch_size:
        for epochs in hpo_epochs:
            for lr in hpo_lr:
                for weight_decay in hpo_weight_decay:
                        for dropout in hpo_dropout:
                            for gcn_out_dim in hpo_gcn_out_dim:
                                for encode_hid_fea_dim in hpo_encode_hid_fea_dim:
                                    for decoder_hid_fea_dim in hpo_encode_hid_fea_dim:
                                        paras[hpo_cnt] = (batch_size, epochs, lr, weight_decay, dropout, gcn_out_dim,
                                                          encode_hid_fea_dim, decoder_hid_fea_dim)
                                        hpo_cnt += 1

    (batch_size, epochs, lr, weight_decay, dropout, gcn_out_dim, encode_hid_fea_dim, decoder_hid_fea_dim) = paras[rank]

    hgcn_cmd = "/mnt/shared/etc/anaconda3/bin/python3 /mnt/shared/home/bipartite-graph-learning/HGCN/hgcn_main.py --model %s --dataset %s --gpu False --epochs %d --batch_size %d --lr %f --weight_decay %f --dropout %f --gcn_output_dim %d  --encoder_hidfeat %d --decoder_hidfeat %d --rank %d" % (
        model_name,
        dataset_name,
        epochs,
        batch_size,
        lr,
        weight_decay,
        dropout,
        gcn_out_dim,
        encode_hid_fea_dim,
        decoder_hid_fea_dim,
        rank)
    os.system(hgcn_cmd)
    print("end hgcn_cmd")

    print("start lr_cmd")
    lr_cmd = "/mnt/shared/etc/anaconda3/bin/python3 /mnt/shared/home/bipartite-graph-learning/HGCN/binary_classification.py --dataset %s --model %s --rank %d" % (
        dataset_name,
        model_name,
        rank)
    if dataset_name != "tencent":
        lr_cmd = "/mnt/shared/etc/anaconda3/bin/python3 /mnt/shared/home/bipartite-graph-learning/HGCN/multi_classification.py --dataset %s --model %s --rank %d" % (
            dataset_name,
            model_name,
            rank)

    os.system(lr_cmd)
    print("end lr_cmd")
