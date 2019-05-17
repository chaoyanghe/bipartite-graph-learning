import re
import numpy as np
from collections import defaultdict
import os.path as path
from typing import List
import networkx as nx
from networkx.algorithms import bipartite
from copy import deepcopy

CONTENT_FILE_PATH = "./cora.content"
CITES_FILE_PATH = "./cora.cites"
OUT_PUT_DIR = "../"
U_FILE_PREFIX = "node"
V_FILE_PREFIX = "group"
CONTENT_LINE_RE = r'^(\S+)((?:\s\d){1433})\s' \
                  r'(Case_Based|Genetic_Algorithms|Neural_Networks|Probabilistic_Methods' \
                  r'|Reinforcement_Learning|Rule_Learning|Theory)$'
CITES_LINE_RE = r'^(\S+)\t(\S+)$'
NUM_FEATURE = 1433
NUM_PAPER = 2708
RAW_CLASS_TO_UV_MAP = {"Case_Based": 'u', "Genetic_Algorithms": 'u', "Neural_Networks": 'v',
                       "Probabilistic_Methods": 'u', "Reinforcement_Learning": 'u', "Rule_Learning": 'v', "Theory": 'v'}
RAW_CLASS_TO_CLASS_IN_U_DICT = {"Case_Based": 0, "Reinforcement_Learning": 1, "Probabilistic_Methods": 2,
                                "Genetic_Algorithms": 3}
V_FEATURE_CUT = 1000


def str_to_vec(bool_str):
    str_vec = bool_str.split()
    assert len(str_vec) == NUM_FEATURE
    return str_vec


def write_attr_and_list(output_dir, prefix, idx_to_feat_vec_dict: dict):
    with open(path.join(output_dir, prefix + "_attr"), 'w') as o_attr_f:
        for idx, feat_vec in idx_to_feat_vec_dict.items():
            o_attr_f.write("{}\t{}\n".format(str(idx), "\t".join(feat_vec)))

    with open(path.join(output_dir, prefix + "_list"), 'w') as o_list_f:
        for idx in idx_to_feat_vec_dict.keys():
            o_list_f.write("{}\n".format(str(idx)))


def remove_keys_from_dict(my_dict: dict, keys2remove):
    for k in keys2remove:
        my_dict.pop(k, None)
    return my_dict


if __name__ == '__main__':
    #####
    # Parse content file
    #####
    raw_class_number_stat = defaultdict(int)
    raw_paper_id_to_idx_dict = {}
    u_idx_to_feature_vec_dict = {}
    v_idx_to_feature_vec_dict = {}
    u_idx_set = set()
    v_idx_set = set()
    u_idx_to_class_dict = {}
    raw_paper_id_no_feat_set = set()  # papers doesnt appear in the content file are removed from the graph,
    #                                     as we dont know their classes

    with open(CONTENT_FILE_PATH, 'r') as in_f:
        content = in_f.readlines()

    for line in content:
        matches = re.search(CONTENT_LINE_RE, line)
        assert matches

        raw_paper_id = matches.group(1)
        feature_vector = str_to_vec(matches.group(2))
        raw_class = matches.group(3)

        # indexing raw paper ids
        idx = raw_paper_id_to_idx_dict.setdefault(raw_paper_id, len(raw_paper_id_to_idx_dict))

        # count number of raw groups
        raw_class_number_stat[matches.group(3)] += 1

        # record feature vector according to u/v
        if RAW_CLASS_TO_UV_MAP[raw_class] == 'u':
            u_idx_to_feature_vec_dict[idx] = feature_vector
            u_idx_set.add(idx)
            u_idx_to_class_dict[idx] = RAW_CLASS_TO_CLASS_IN_U_DICT[raw_class]
        else:
            v_idx_to_feature_vec_dict[idx] = feature_vector[:V_FEATURE_CUT]  # cut v feature vector to create difference
            v_idx_set.add(idx)

    assert len(u_idx_set & v_idx_set) == 0

    print(raw_class_number_stat)
    assert len(raw_paper_id_to_idx_dict) == NUM_PAPER

    #####
    # Parse cites file
    #####
    # sets to record which idx hasn't appear in the bipartite edges and needs to be purged
    u_idx_not_appear_set = deepcopy(u_idx_set)
    v_idx_not_appear_set = deepcopy(v_idx_set)
    bipartite_edges_set = set()  # TODO: confirm the graph is not directed

    with open(CITES_FILE_PATH, 'r') as in_f:
        content = in_f.readlines()

    for line in content:
        matches = re.search(CITES_LINE_RE, line)
        assert matches

        vertex0_idx = raw_paper_id_to_idx_dict.get(matches.group(1), None)
        vertex1_idx = raw_paper_id_to_idx_dict.get(matches.group(2), None)

        # remove vertexes that doesnt appear in the content file
        if vertex0_idx is None:
            raw_paper_id_no_feat_set.add(matches.group(1))
            continue
        if vertex1_idx is None:
            raw_paper_id_no_feat_set.add(matches.group(2))
            continue

        if (vertex0_idx in u_idx_set) and (vertex1_idx in v_idx_set):
            u_idx_not_appear_set.discard(vertex0_idx)
            v_idx_not_appear_set.discard(vertex1_idx)
            bipartite_edges_set.add((vertex0_idx, vertex1_idx))
        elif (vertex1_idx in u_idx_set) and (vertex0_idx in v_idx_set):
            u_idx_not_appear_set.discard(vertex1_idx)
            v_idx_not_appear_set.discard(vertex0_idx)
            bipartite_edges_set.add((vertex1_idx, vertex0_idx))

    print(raw_paper_id_no_feat_set)
    print(len(raw_paper_id_no_feat_set))

    # Purge the isolated vertex
    u_idx_set -= u_idx_not_appear_set
    v_idx_set -= v_idx_not_appear_set
    u_idx_to_feature_vec_dict = remove_keys_from_dict(u_idx_to_feature_vec_dict, u_idx_not_appear_set)
    v_idx_to_feature_vec_dict = remove_keys_from_dict(v_idx_to_feature_vec_dict, v_idx_not_appear_set)
    u_idx_to_class_dict = remove_keys_from_dict(u_idx_to_class_dict, u_idx_not_appear_set)

    print(len(u_idx_set) - len(v_idx_set))

    # Assert bipartite
    B = nx.Graph()
    B.add_nodes_from(u_idx_set, bipartite=0)
    B.add_nodes_from(v_idx_set, bipartite=1)
    B.add_edges_from(bipartite_edges_set)
    assert nx.is_bipartite(B)

    #####
    # Output
    #####
    write_attr_and_list(OUT_PUT_DIR, U_FILE_PREFIX, u_idx_to_feature_vec_dict)
    write_attr_and_list(OUT_PUT_DIR, V_FILE_PREFIX, v_idx_to_feature_vec_dict)

    with open(path.join(OUT_PUT_DIR, U_FILE_PREFIX + "_true"), 'w') as o_u_class_f:
        for u_idx, u_class in u_idx_to_class_dict.items():
            o_u_class_f.write("{}\t{}\n".format(str(u_idx), str(u_class)))

    with open(path.join(OUT_PUT_DIR, "edgelist"), 'w') as o_e_file:
        for u_idx, v_idx in bipartite_edges_set:
            o_e_file.write("{}\t{}\n".format(str(u_idx), str(v_idx)))

    ####
    # Final statistics
    ####
    print("---")
    print(len(u_idx_set))
    print(len(v_idx_set))
    print(len(u_idx_set) + len(v_idx_set))
    print(len(bipartite_edges_set))
