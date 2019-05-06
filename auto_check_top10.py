import os
from time import sleep


def topten(auc_dict, paras):
    if len(auc_dict) < 10:
        return
    auc_list_sorted = [v for v in sorted(auc_dict.values())]

    print("HGCN. The top 10 AUC will be:")
    for index in range(10):
        str = "--batch_size %d --epochs %d --lr %f --weight_decay %f --dis_hidden %d --dropout %f" % (
            batch_size,
            epochs,
            lr,
            weight_decay,
            dis_hidden,
            dropout)
        print(str)
        print("index = %d. auc = %f. Parameters: %s" % (index, auc_list_sorted[index], str))


if __name__ == "__main__":
    # 972 parallel processes
    hpo_batch_size = [500, 1000, 1500]  # 3
    hpo_epochs = [3, 5, 8, 10]  # 4
    hpo_lr = [0.0003, 0.0005, 0.001]  # 3
    hpo_weight_decay = [0.001, 0.0001, 0.00001]  # 3
    hpo_dis_hidden = [16, 24, 32]  # 3
    hpo_dropout = [0.4, 0.5, 0.6]  # 3

    bool_all_generated = False
    auc_dict = []
    paras = dict()
    while not bool_all_generated:
        hpo_cnt = 0
        for batch_size in hpo_batch_size:
            for epochs in hpo_epochs:
                for lr in hpo_lr:
                    for weight_decay in hpo_weight_decay:
                        for dis_hidden in hpo_dis_hidden:
                            for dropout in hpo_dropout:
                                paras[hpo_cnt] = (batch_size, epochs, lr, weight_decay, dis_hidden, dropout)
                                auc_file = "/mnt/shared/home/bipartite-graph-learning/out/%d/hgcn.res_auc" % hpo_cnt
                                if os.path.exists(auc_file):
                                    with open(auc_file, "r") as f:
                                        first_line = f.readline()
                                        auc_value = first_line.split("_")
                                        str = "--batch_size %d --epochs %d --lr %f --weight_decay %f --dis_hidden %d --dropout %f --rank %d" % (
                                            batch_size,
                                            epochs,
                                            lr,
                                            weight_decay,
                                            dis_hidden,
                                            dropout,
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
