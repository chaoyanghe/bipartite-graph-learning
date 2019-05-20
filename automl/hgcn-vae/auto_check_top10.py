import os
from time import sleep

import conf

dataset_name = conf.dataset_name
model_name = conf.model_name


def topten(auc_dict, paras):
    if len(auc_dict) < 10:
        return

    auc_dict_sorted = sorted(auc_dict.items(), key=lambda d: d[1], reverse=True)

    print("HGCN. The top 10 AUC will be:")
    cnt = 0
    for (index, auc) in auc_dict_sorted:
        (batch_size, epochs, lr, weight_decay, dropout, gcn_out_dim, encode_hid_fea_dim, decoder_hid_fea_dim) = paras[
            index]
        str = "--epochs %d --batch_size %d --lr %f --weight_decay %f --dropout %f --gcn_output_dim %d  --encoder_hidfeat %d --decoder_hidfeat %d --rank %d" % (
            dataset_name,
            epochs,
            batch_size,
            lr,
            weight_decay,
            dropout,
            gcn_out_dim,
            encode_hid_fea_dim,
            decoder_hid_fea_dim)
        print("index = %s, auc = %s. Parameters: %s" % (index, auc, str))
        cnt += 1
        if cnt > 10:
            break


if __name__ == "__main__":
    # 972 parallel processes
    hpo_batch_size = [64, 128, 400]  # 3
    hpo_dropout = [0.0, 0.2, 0.4]  # 3
    hpo_epochs = [3, 5, 10]  # 3
    hpo_lr = [0.001, 0.0001]  # 2
    hpo_weight_decay = [0.05, 0.005, 0.0008]  # 3
    hpo_gcn_out_dim = [16, 32, 48]  # 3
    hpo_encode_hid_fea_dim = [16, 24]  # 2
    hpo_decoder_hid_fea_dim = [8]  # 1

    bool_all_generated = False
    auc_dict = {}
    paras = dict()
    while not bool_all_generated:
        hpo_cnt = 0
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
                                        auc_file = "/mnt/shared/home/bipartite-graph-learning/out/hgcn-%s/%s/%d/hgcn.res.f1_precision_recall" % (
                                            model_name, dataset_name, hpo_cnt)
                                        # print(auc_file)
                                        if os.path.exists(auc_file):
                                            with open(auc_file, "r") as f:
                                                line_one = f.readline()
                                                print(line_one)
                                                auc_value = line_one.split(" ")[-1]
                                                str = "--epochs %d --batch_size %d --lr %f --weight_decay %f --dropout %f --gcn_output_dim %d  --encoder_hidfeat %d --decoder_hidfeat %d --rank %d" % (
                                                    dataset_name,
                                                    epochs,
                                                    batch_size,
                                                    lr,
                                                    weight_decay,
                                                    dropout,
                                                    gcn_out_dim,
                                                    encode_hid_fea_dim,
                                                    decoder_hid_fea_dim)
                                                print("auc = %s. Parameters: %s" % (auc_value, str))
                                                # read the AUC value
                                                auc_dict[hpo_cnt] = line_one
                                        hpo_cnt += 1
        if len(auc_dict) == 972:
            print("all genrated!")
            topten(auc_dict, paras)
            bool_all_generated = True
        else:
            print("generated len = %d" % len(auc_dict))
            topten(auc_dict, paras)
            auc_dict.clear()
            paras.clear()
            print("start next round checking")
            sleep(3)
