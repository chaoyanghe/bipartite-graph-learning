import re
import os.path as path
import networkx as nx
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
RAW_CLASS_SET = {"Case_Based", "Genetic_Algorithms", "Neural_Networks", "Probabilistic_Methods",
                 "Reinforcement_Learning", "Rule_Learning", "Theory"}
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
    idx_to_raw_class_dict = {}
    raw_paper_id_to_idx_dict = {}
    u_idx_to_feature_vec_dict = {}
    v_idx_to_feature_vec_dict = {}
    u_idx_set = set()
    v_idx_set = set()
    u_idx_to_class_dict = {}
    raw_paper_id_no_feat_set = set()  # papers doesnt appear in the content file are removed from the graph,
    #                                     as we dont know their classes
    idx_to_intra_raw_class_idx_dicts = {k: {} for k in RAW_CLASS_SET}
    raw_class_to_class_dict = {k: i for i, k in enumerate(RAW_CLASS_SET)}
    print("raw_class_to_class_dict:{}".format(raw_class_to_class_dict))

    with open(CONTENT_FILE_PATH, 'r') as in_f:
        content = in_f.readlines()

    for line in content:
        matches = re.search(CONTENT_LINE_RE, line)
        assert matches

        raw_paper_id = matches.group(1)
        raw_class = matches.group(3)

        # indexing raw paper ids
        idx = raw_paper_id_to_idx_dict.setdefault(raw_paper_id, len(raw_paper_id_to_idx_dict))
        idx_to_raw_class_dict[idx] = raw_class
        idx_to_intra_raw_class_idx_dicts[raw_class].setdefault(idx, len(idx_to_intra_raw_class_idx_dicts[raw_class]))

    print("Raw Class Counts: {}".format({rc: len(d) for rc, d in idx_to_intra_raw_class_idx_dicts.items()}))
    assert sum([len(d) for d in idx_to_intra_raw_class_idx_dicts.values()]) == NUM_PAPER

    print("idx_to_intra_raw_class_idx_dicts: {}".format(idx_to_intra_raw_class_idx_dicts))

    for line in content:
        matches = re.search(CONTENT_LINE_RE, line)
        assert matches

        raw_paper_id = matches.group(1)
        feature_vector = str_to_vec(matches.group(2))
        raw_class = matches.group(3)

        idx = raw_paper_id_to_idx_dict.get(raw_paper_id)

        # record feature vector according to u/v, half split each raw class to u and v
        if idx_to_intra_raw_class_idx_dicts[raw_class][idx] <= len(idx_to_intra_raw_class_idx_dicts[raw_class]) // 2:
            u_idx_to_feature_vec_dict[idx] = feature_vector
            u_idx_set.add(idx)
            u_idx_to_class_dict[idx] = raw_class_to_class_dict[raw_class]
        else:
            v_idx_to_feature_vec_dict[idx] = feature_vector[:V_FEATURE_CUT]  # cut v feature vector to create difference
            v_idx_set.add(idx)

    assert len(u_idx_set & v_idx_set) == 0
    assert ((len(u_idx_set) + len(v_idx_set)) == NUM_PAPER)

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

    num_edges_to_delete = 0
    for line in content:
        matches = re.search(CITES_LINE_RE, line)
        assert matches

        vertex0_idx = raw_paper_id_to_idx_dict.get(matches.group(1), None)
        vertex1_idx = raw_paper_id_to_idx_dict.get(matches.group(2), None)

        # record vertexes that doesnt appear in the content file
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
        else:
            if vertex0_idx in u_idx_set:
                vertex0_uv_group = 'u'
            elif vertex0_idx in v_idx_set:
                vertex0_uv_group = 'v'
            else:
                raise Exception

            if vertex1_idx in u_idx_set:
                vertex1_uv_group = 'u'
            elif vertex0_idx in v_idx_set:
                vertex1_uv_group = 'v'
            else:
                raise Exception

            num_edges_to_delete += 1
            print("{} Delete (raw id): {}->{}, (idx): {}->{}, (raw class): {}->{}, (u/v): {}->{}"
                  .format(num_edges_to_delete,
                          matches.group(1), matches.group(2),
                          vertex0_idx, vertex1_idx,
                          idx_to_raw_class_dict[vertex0_idx], idx_to_raw_class_dict[vertex1_idx],
                          vertex0_uv_group, vertex1_uv_group))

    print("raw_paper_id_no_feat_set: ({}) {}".format(len(raw_paper_id_no_feat_set), raw_paper_id_no_feat_set))

    # Purge the isolated vertex
    u_idx_set -= u_idx_not_appear_set
    v_idx_set -= v_idx_not_appear_set
    u_idx_to_feature_vec_dict = remove_keys_from_dict(u_idx_to_feature_vec_dict, u_idx_not_appear_set)
    v_idx_to_feature_vec_dict = remove_keys_from_dict(v_idx_to_feature_vec_dict, v_idx_not_appear_set)
    u_idx_to_class_dict = remove_keys_from_dict(u_idx_to_class_dict, u_idx_not_appear_set)

    print("|u|-|v|: {}".format(len(u_idx_set) - len(v_idx_set)))

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
    print("Nodes (Group U/V), Edges, Features (Group U/V), Classes (Group U/V)\n{}/{}, {}, {}/{}, {}/"
          .format(len(u_idx_set), len(v_idx_set), len(bipartite_edges_set), NUM_FEATURE, V_FEATURE_CUT,
                  len(RAW_CLASS_SET)))
