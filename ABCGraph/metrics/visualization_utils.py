from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import os
import pandas as pd

from utils import (EPOCHS, TRAINING_LOSS_PATH)
from metrics.writer import STEP_LOSS


def _set_plot_properties(properties):
    """Sets some plt properties."""
    if 'xlim' in properties:
        plt.xlim(properties['xlim'])
    if 'ylim' in properties:
        plt.ylim(properties['ylim'])
    if 'xlabel' in properties:
        plt.xlabel(properties['xlabel'])
    if 'ylabel' in properties:
        plt.ylabel(properties['ylabel'])

def load_written_data():
    """Loads the data from the given files."""
    if not os.path.exists(TRAINING_LOSS_PATH):
        training_loss = pd.read_csv(TRAINING_LOSS_PATH)
    else:
        raise Exception('File does not exist.')

    if training_loss is not None:
        training_loss.sort_values(by=EPOCHS, inplace=True)

    return training_loss

def plot_loss_vs_epochs(loss_file, figsize=(10, 8), title_fontsize=16, **kwargs):
    plt.figure(figsize=figsize)
    plt.title('Loss vs Epochs', fontsize=title_fontsize)
    plt.plot(loss_file[EPOCHS], loss_file[STEP_LOSS[1]], linestyle='-', label='explicit relation')
    plt.plot(loss_file[EPOCHS], loss_file[STEP_LOSS[2]], linestyle='-', label='implicit relation')
    plt.plot(loss_file[EPOCHS], loss_file[STEP_LOSS[3]], linestyle='-', label='merge relation')
    plt.plot(loss_file[EPOCHS], loss_file[STEP_LOSS[4]], linestyle='-', label='opposite relation')

    plt.legend(loc='upper left')
    plt.xlabel = 'Epochs'
    plt.ylable = 'Training Loss'
    _set_plot_properties(kwargs)
    plt.show()
    plt.close()

def load_data(file_path):
    if os.path.exists(file_path):
        file = pd.read_csv(file_path)
    else:
        raise Exception('No such file exists.')
    return file






