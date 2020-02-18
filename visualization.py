from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

# figure form
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# default parameters
pca_dimensions = 50
tsne_dimensions = 2
num_clusters = {'cora': 7,
                'citeseer': 6,
                'pubmed': 3,
                'tencent': 2}


def load_emb_data(fname, ind=None):
    data = pd.read_csv(fname, delimiter=" ", skiprows=1, index_col=0, header=None)
    if ind is not None:
        data = data.reindex(ind, copy=False, fill_value=0)
    return data


def load_index_data(fname):
    ind = pd.read_csv(fname, delimiter='\t', index_col=0, header=None)
    return ind.index


def load_label_data(fname, dataset):
    if dataset != 'tencent':
        label = pd.read_csv(fname, delimiter='\t', index_col=0, header=None, names=['labels'])
    else:
        label = pd.read_csv(fname, delimiter='\t', names=['labels'])
    return label.labels.to_numpy()


def t_sne(data, _pca=True):
    X = data.to_numpy()
    if _pca and X.shape[1] > pca_dimensions:
        pca = PCA(n_components=pca_dimensions)
        X = pca.fit_transform(X)
    X_embedded = TSNE(n_components=tsne_dimensions).fit_transform(X)
    return X_embedded


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', required=False)
    parser.add_argument('--model', type=str, default='gae', required=False)
    args = parser.parse_args()

    input_file_path = './out/' + args.model + '/' + args.dataset
    if args.model not in ['abcgraph-adv', 'abcgraph-mlp']:
        embedding_path = input_file_path + '/' + args.model + '.emb'
    else:
        embedding_path = input_file_path + '/abcgraph' + '.emb'
    node_list_path = input_file_path + '/node_list'
    label_path = './data/' + args.dataset + '/node_true'

    ind = load_index_data(node_list_path)
    data = load_emb_data(embedding_path)
    labels = load_label_data(label_path, args.dataset)

    if args.dataset == 'tencent':
        data_true = data[labels]
        data_false = data[np.array(set([i for i in range(len(data))]) - set(labels))]
        data_false = data_false[np.random.choice(len(data_false), len(data_true), replace=False)]
        data = np.concatenate([data_true, data_false], axis=0)

    x = t_sne(data)
    if args.dataset != 'tencent':
        for i in range(num_clusters[args.dataset]):
            x_cat = x[np.where(labels == i)]
            plt.scatter(x_cat[:, 0], x_cat[:, 1])
        plt.legend(['labels: {}'.format(i) for i in range(len(set(labels)))])
        plt.title('{} -- {}'.format(args.model, args.dataset), fontsize=20, weight='bold')
        plt.show()
    else:
        plt.scatter(x[:len(labels), 0], x[len(labels):, 1])
        plt.show()

