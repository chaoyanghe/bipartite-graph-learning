if __name__ == "__main__":
    # 972 parallel processes
    hpo_batch_size = [500, 1000, 1500]  # 3
    hpo_epochs = [3, 5, 8, 10]  # 4
    hpo_lr = [0.0003, 0.0005, 0.001]  # 3
    hpo_weight_decay = [0.001, 0.0001, 0.00001]  # 3
    hpo_dis_hidden = [16, 24, 32]  # 3
    hpo_dropout = [0.4, 0.5, 0.6]  # 3

    hpo_cnt = 0
    auc_dict = []
    paras = dict()
    for batch_size in hpo_batch_size:
        for epochs in hpo_epochs:
            for lr in hpo_lr:
                for weight_decay in hpo_weight_decay:
                    for dis_hidden in hpo_dis_hidden:
                        for dropout in hpo_dropout:
                            paras[hpo_cnt] = (batch_size, epochs, lr, weight_decay, dis_hidden, dropout)
                            result_path = "/mnt/shared/home/bipartite-graph-learning/" + str(hpo_cnt)
                            with open(result_path + "hgcn.res_auc", "r") as f:
                                first_line = f.readline()
                            auc_value = first_line.split("_")
                            # read the AUC value
                            auc_dict.append(float(auc_value[0]))
                            hpo_cnt += 1

    auc_dict = auc_dict.sort(reverse=True)

    print("HGCN. The top 10 AUC will be:")
    auc_top10 = auc_dict[0:9]
    for index in range(10):
        (batch_size, epochs, lr, weight_decay, dis_hidden, dropout) = paras[index]
        print("auc = %s. (batch_size, epochs, lr, weight_decay, dis_hidden, dropout) = (%s, %s, %s, %s, %s, %s)" % (
            str(auc_top10[index]), str(batch_size), str(epochs), str(lr), str(weight_decay), str(dis_hidden),
            str(dropout)))
