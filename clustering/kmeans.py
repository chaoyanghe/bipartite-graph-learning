import numpy as np
import argparse
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import normalized_mutual_info_score

# defined parameters
# the number of clusters for set U
num_clusters = {'cora': 7,
                'citeseer': 6,
                'pubmed': 3,
                'tencent': 2}
pca_dimensions = 50
random_state = 20  # initialize kmeans seed


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


def clustering(data, label, clusters, _pca=True):
    X = data.to_numpy()
    y = label
    if _pca and X.shape[1] > pca_dimensions:
        pca = PCA(n_components=pca_dimensions)
        X = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
    kmeans = KMeans(n_clusters=clusters, random_state=random_state)
    kmeans.fit(X_train)
    y_pred = kmeans.predict(X_test)
    metric_score = normalized_mutual_info_score(y_test, y_pred)
    print(metric_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='citeseer', required=False)
    parser.add_argument('--model', type=str, default='abcgraph-adv', required=False)
    args = parser.parse_args()

    input_file_path = '../out/' + args.model + '/' + args.dataset
    if args.model not in ['abcgraph-adv', 'abcgraph-mlp']:
        embedding_path = input_file_path + '/' + args.model + '.emb'
    else:
        embedding_path = input_file_path + '/abcgraph.emb'
    node_list_path = input_file_path + '/node_list'
    label_path = '../data/' + args.dataset + '/node_true'

    ind = load_index_data(node_list_path)
    data = load_emb_data(embedding_path)
    label = load_label_data(label_path, args.dataset)

    clusters = num_clusters[args.dataset]

    clustering(data, label, clusters)
