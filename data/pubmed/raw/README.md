## Reference
- [PubMed dataset](http://rtw.ml.cmu.edu/rtw/)
- [Can't run on Nell dataset #14](https://github.com/tkipf/gcn/issues/14)
- [tkipf/gcn](https://github.com/tkipf/gcn.git)

## My work done
1. combine test and training sets
    - test set contains **1000** data points
    - training set contains **18717** data points
1. split each class (a total of 3) evenly
    1. about one half of the data is **node** or **U** while the other half is **group** or **V**
    1. take the original graph and keep only the edges that connects a vertex in **U** and a vertex in **V**
        - this means within **U** and **V**, all vertices are disconnected
    1. remove all disconnected (isolated) nodes
        - this leaves **13992** data points
1. `node_{attr, list, true}` file contains only the data in set **U**
1. `group_{attr, list}` file contains only the data in set **V**

## Dataset File Format TL;DR
- ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
- ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
- ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
- ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
- ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
- ind.dataset_str.ally => the one-hot labels for instances in ind.dataset_str.allx as numpy.ndarray object;
- ind.dataset_str.graph => a dict in the format `{index: [index_of_neighbor_nodes]}` as collections.defaultdict
        object;
- ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object

## Prepare the data

### Inductive learning

The input to the inductive model contains:
- `x`, the feature vectors of the labeled training instances,
- `y`, the one-hot labels of the labeled training instances,
- `allx`, the feature vectors of both labeled and unlabeled training instances (a superset of `x`),
- `graph`, a `dict` in the format `{index: [index_of_neighbor_nodes]}.`

Let n be the number of both labeled and unlabeled training instances. These n instances should be indexed from 0 to n - 1 in `graph` with the same order as in `allx`.

### Preprocessed datasets

Datasets for Citeseet, Cora, and Pubmed are available in the directory `data`, in a preprocessed format stored as numpy/scipy files.

The dataset for DIEL is available at http://www.cs.cmu.edu/~lbing/data/emnlp-15-diel/emnlp-15-diel.tar.gz. We also provide a much more succinct version of the dataset that only contains necessary files and some (not very well-organized) pre-processing code here at http://cs.cmu.edu/~zhiliny/data/diel_data.tar.gz.

The NELL dataset can be found here at http://www.cs.cmu.edu/~zhiliny/data/nell_data.tar.gz.

In addition to `x`, `y`, `allx`, and `graph` as described above, the preprocessed datasets also include:
- `tx`, the feature vectors of the test instances,
- `ty`, the one-hot labels of the test instances,
- `test.index`, the indices of test instances in `graph`, for the inductive setting,
- `ally`, the labels for instances in `allx`.

The indices of test instances in `graph` for the transductive setting are from `#x` to `#x + #tx - 1`, with the same order as in `tx`.

You can use `cPickle.load(open(filename))` to load the numpy/scipy objects `x`, `y`, `tx`, `ty`, `allx`, `ally`, and `graph`. `test.index` is stored as a text file.

## Hyper-parameter tuning

Refer to `test_ind.py` and `test_trans.py` for the definition of different hyper-parameters (passed as arguments). Hyper-parameters are tuned by randomly shuffle the training/test split (i.e., randomly shuffling the indices in `x`, `y`, `tx`, `ty`, and `graph`). For the DIEL dataset, we tune the hyper-parameters on one of the ten runs, and then keep the same hyper-parameters for all the ten runs.
