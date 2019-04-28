import numpy as np

REAL_LABEL = 1
FAKE_LABEL = 0
BATCH_SIZE = 1

def load_data():
    """
    Output:
    adjacent matrix, intput feature matrix, training dataset, validation dataset, testing dataset

    Dataset form [batch, features]
    """
    # TODO: Loading data, output: adjacent matrix
    adjU = [[1, 1],
            [1, 0],
            [1, 0]]
    adjV = [[1, 1, 1],
            [1, 0, 0]]
    featuresU = np.random.rand(12).reshape(3, 4)
    featuresV = np.random.rand(10).reshape(2, 5)

    return np.array(adjU), np.array(adjV), featuresU, featuresV
