import numpy as np
import pickle


def load_data():
    extensions = ['x', 'tx', 'allx', 'graph']
    objects = []

    for extension in extensions:
        with open("data/ind.nell.0.001.{}".format(extension), 'rb') as input_file:
            objects.append(pickle.load(input_file, encoding='latin1'))


if __name__ == '__main__':
    load_data()