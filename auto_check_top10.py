import os
from time import sleep

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
                                auc_file = "/mnt/shared/home/bipartite-graph-learning/" + str(hpo_cnt) + "/hgcn.res_auc"
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
                                        auc_dict.append(float(auc_value[0]))
                                hpo_cnt += 1
        if len(auc_dict) == 972:
            print("all genrated!")
            bool_all_generated = True
        else:
            auc_dict.clear()
            paras.clear()
            print("start next round checking")
            sleep(3)

    auc_dict = auc_dict.sort(reverse=True)

    print("HGCN. The top 10 AUC will be:")
    auc_top10 = auc_dict[0:9]
    for index in range(10):
        (batch_size, epochs, lr, weight_decay, dis_hidden, dropout) = paras[index]
        str = "--batch_size %d --epochs %d --lr %f --weight_decay %f --dis_hidden %d --dropout %f" % (
            batch_size,
            epochs,
            lr,
            weight_decay,
            dis_hidden,
            dropout)
        print("auc = %f. Parameters: %s" % (auc_top10[index], str))
