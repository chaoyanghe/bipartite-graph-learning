import sys
import json
import random
from networkx.readwrite import json_graph

WALK_LEN = 5
N_WALKS = 50

def run_random_walks(G, nodes, walk_len, n_walks):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(n_walks):
            curr_node = node
            for j in range(walk_len):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node, curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs


if __name__ == "__main__":
    """ Run random walks """
    graph_file = './bipartite-G.json'
    out_file = './bipartite-walks.txt'
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes, WALK_LEN, N_WALKS)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
