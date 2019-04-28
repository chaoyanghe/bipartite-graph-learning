import os
import csv

import numpy as np
import pandas as pd

from utils import (STEP1, STEP2, STEP3, STEP4)

# TRAINING_LOSS_PATH_1 = 'metrics/experiments_results/training_loss1.csv'
# TRAINING_LOSS_PATH_2 = 'metrics/experiments_results/training_loss2.csv'
# TRAINING_LOSS_PATH_3 = 'metrics/experiments_results/training_loss3.csv'
# TRAINING_LOSS_PATH_4 = 'metrics/experiments_results/training_loss4.csv'
#
# LOSS_DICT = {1: [TRAINING_LOSS_PATH_1, 'explicit_relation'],
#              2: [TRAINING_LOSS_PATH_2, 'implicit_relation'],
#              3: [TRAINING_LOSS_PATH_3, 'merge_relation'],
#              4: [TRAINING_LOSS_PATH_4, 'opposite_relation']}

EXPERIMENTS_PATH = 'metrics/experiments_results'
STEP_LOSS = {1: 'step1_loss',
             2: 'step2_loss',
             3: 'step3_loss',
             4: 'step4_loss'}


def create_metrics_files(csvname='training_loss.csv'):
    path = os.path.join(EXPERIMENTS_PATH, csvname)
    if not os.path.exists(path):
        with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            COLUMN_NAMES = ['epochs', 'step1_loss', 'step2_loss', 'step3_loss', 'step4_loss']
            writer.writerow(COLUMN_NAMES)
        csvfile.close()
    else:
        raise Exception('File already exists')


def print_metrics(step, epoch, loss, csvname='training_loss.csv'):
    """write the metrics to file"""
    path = os.join.path(EXPERIMENTS_PATH, csvname)
    metrics = pd.read_csv(path)
    # the first step to build the table
    if step == 1:
        metric = {'epochs': epoch, STEP_LOSS[step]: loss}
        metrics = metrics.append(metric, ignore_index=True)
    else:
        metrics.loc[int(epoch)-1, STEP_LOSS[step]] = loss  # epoch starts from 1
    metrics.to_csv(path, index=False)

# TODO: add the writing function for gan
def print_gan_metrics(step, epoch, loss):
    """write the GAN metrics to file"""
    pass