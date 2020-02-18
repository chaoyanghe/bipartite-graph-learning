import networkx as nx
import pandas as pd
import numpy as np

"""
    Generate the toy dataset 
"""

if __name__ == '__main__':
    # edge_list = np.array([[0, 2], [1, 2], [1, 3], [2, 5], [2, 6], [3, 5], [4, 5], [5, 6], [5, 7]])
    edge_list = np.array(
        [[0, 1], [0, 5], [5, 1], [5, 6], [1, 2], [1, 7], [6, 7], [2, 3], [2, 7], [3, 7], [7, 8], [3, 8], [3, 4],
         [4, 8]])

    features = np.array(
        [[1, 0, 2, 'Probabilistic_Methods'],
         [1, 2, 1, 'Probabilistic_Methods'],
         [0, 1, 2, 'Theory'],
         [2, 4, 1, 'Probabilistic_Methods'],
         [1, 9, 5, 'Theory'],
         [1, 4, 2, 'Theory'],
         [8, 6, 8, 'Case_Based'],
         [1, 9, 8, 'Rule_Learning'],
         [2, 2, 2, 'Case_Based']
         ]
    )
    graph = nx.from_edgelist(edge_list)
    nx.write_edgelist(graph, './raw/toy_dataset.cites', data=False, delimiter='\t')
    df = pd.DataFrame(features)
    df.to_csv('./raw/toy_dataset.content', sep='\t', header=False)
