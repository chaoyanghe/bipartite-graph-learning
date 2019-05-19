import os
from time import sleep


def topten(auc_dict, paras):
    if len(auc_dict) < 10:
        return

    auc_dict_sorted = sorted(auc_dict.items(), key=lambda d: d[1], reverse=True)

    print("HGCN. The top 10 AUC will be:")
    cnt = 0
    for (index, auc) in auc_dict_sorted:
        (batch_size, epochs, lr, weight_decay, dis_hidden, dropout) = paras[index]
        str = "--batch_size %d --epochs %d --lr %f --weight_decay %f --dis_hidden %d --dropout %f" % (
            batch_size,
            epochs,
            lr,
            weight_decay,
            dis_hidden,
            dropout)
        print("index = %s, auc = %s. Parameters: %s" % (index, auc, str))
        cnt += 1
        if cnt > 10:
            break


if __name__ == "__main__":
    # 972 parallel processes
    hpo_batch_size = [400, 500, 600, 700]  # 3
    hpo_epochs = [2, 3, 4]  # 4
    hpo_lr = [0.0002, 0.0003, 0.0004]  # 3
    hpo_weight_decay = [0.001, 0.0005, 0.0008]  # 3
    hpo_dis_hidden = [16, 20, 24]  # 3
    hpo_dropout = [0.35, 0.4, 0.45]  # 3

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
                                paras[hpo_cnt] = (batch_size, epochs, lr, weight_decay, dis_hidden, dropout)
                                auc_file = "/mnt/shared/home/bipartite-graph-learning/out/hgcn_gan/tencent/%d/hgcn.res.f1_precision_recall" % hpo_cnt
                                # print(auc_file)
                                if os.path.exists(auc_file):
                                    with open(auc_file, "r") as f:
                                        for l_idx in range(3):
                                            line_three = f.readline()
                                            print(line_three)
                                        auc_value = line_three.split(" ")[-1]
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
