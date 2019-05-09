import os
from time import sleep


def topten(auc_dict, paras):
    if len(auc_dict) < 10:
        return

    auc_dict_sorted = sorted(auc_dict.items(), key=lambda d: d[1], reverse=True)

    print("HGCN. The top 10 AUC will be:")
    cnt = 0
    for (index, auc) in auc_dict_sorted:
        (batch_size, epochs, lr, weight_decay, dis_hidden, dropout, gcn_output_dim) = paras[index]
        str = "--batch_size %d --epochs %d --lr %f --weight_decay %f --dis_hidden %d --dropout %f --gcn_output_dim %d" % (
            batch_size,
            epochs,
            lr,
            weight_decay,
            dis_hidden,
            dropout,
            gcn_output_dim)
        print("index = %s, auc = %s. Parameters: %s" % (index, auc, str))
        cnt += 1
        if cnt > 10:
            break


if __name__ == "__main__":
    # 972 parallel processes
    hpo_batch_size = [400, 500, 600]  # 3
    hpo_epochs = [2, 3, 4]  # 3
    hpo_lr = [0.0002, 0.0003]  # 2
    hpo_weight_decay = [0.001, 0.0005]  # 2
    hpo_dis_hidden = [16, 20, 24]  # 3
    hpo_dropout = [0.35, 0.4, 0.45]  # 3
    hpo_gcn_output = [16, 20, 24]  # 3

    bool_all_generated = False
    auc_dict = {}
    paras = dict()
    while not bool_all_generated:
        hpo_cnt = 0
        for batch_size in hpo_batch_size:
            for epochs in hpo_epochs:
                for lr in hpo_lr:
                    for weight_decay in hpo_weight_decay:
                        for dis_hidden in hpo_dis_hidden:
                            for dropout in hpo_dropout:
                                for gcn_output_dim in hpo_gcn_output:
                                    paras[hpo_cnt] = (batch_size, epochs, lr, weight_decay, dis_hidden, dropout, gcn_output_dim)
                                    auc_file = "/mnt/shared/home/bipartite-graph-learning/out/%d/hgcn.res_auc" % hpo_cnt
                                    if os.path.exists(auc_file):
                                        with open(auc_file, "r") as f:
                                            first_line = f.readline()
                                            auc_value = first_line.split("_")
                                            str = "--batch_size %d --epochs %d --lr %f --weight_decay %f " \
                                                  "--dis_hidden %d --dropout %f --gcn_output_dim %d --rank %d" % (
                                                batch_size,
                                                epochs,
                                                lr,
                                                weight_decay,
                                                dis_hidden,
                                                dropout,
                                                gcn_output_dim,
                                                hpo_cnt)
                                            print("auc = %s. Parameters: %s" % (auc_value, str))
                                            # read the AUC value
                                            auc_dict[hpo_cnt] = float(auc_value[0])
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
