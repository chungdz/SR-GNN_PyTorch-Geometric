# encoding: utf-8
"""
1. 构造一个entity 的 vocab
2. 构造一个 new_id ==> Graph 的 dict
"""

import os
import json
import pickle
import argparse

import pandas as pd
import numpy as np
import networkx as nx

def build_entire_graph(train_data):
    edges = list()

    for sequences, y in zip(train_data[0], train_data[1]):
        cur_len = len(sequences)
        assert(cur_len > 0)
        if cur_len < 2:
            continue
        if cur_len == 2:
            edges.append([sequences[0], sequences[1], 1])
            edges.append([sequences[1], sequences[0], 1])
        else:
            for i in range(1, cur_len - 1):
                edges.append([sequences[i], sequences[i - 1], 1])
                edges.append([sequences[i - 1], sequences[i], 1])

    edge_df = pd.DataFrame(edges, columns=["from", "to", "weight"])
    edge_weights = edge_df.groupby(["from", "to"]).apply(lambda x: sum(x["weight"]))
    weighted_edges = edge_weights.to_frame().reset_index().values

    dg = nx.DiGraph()
    dg.add_weighted_edges_from(weighted_edges)

    return dg

def build_neighbors_dict(graph, max_neighbor_cnt=5):
    """
    For each entity, sampling max_neighbor_cnt neighbors in the entity graph
    Args:
        all_ents: all entities in all news
        graph: the entity graph
        max_neighbor_cnt:

    Returns:
        a dict mapping an ent to a set of its neighbors
    """
    ent_neighbors_dict = {}
    all_ents = graph.nodes()
    for ent in all_ents:
        # 按照共现次数降序选取邻居
        sorted_neighbors = sorted(dict(graph[ent]).items(), key=lambda item: -1 * item[1]['weight'])
        neighbors = [x[0] for x in sorted_neighbors][:max_neighbor_cnt]
        if len(neighbors) == 0:
            ent_neighbors_dict[ent] = []
        ent_neighbors_dict[ent] = list(set(neighbors))

    return ent_neighbors_dict


def main(cfg):

    train_data = pickle.load(open(os.path.join(cfg.dataset, 'raw/train.txt'), 'rb'))
    test_data = pickle.load(open(os.path.join(cfg.dataset, 'raw/test.txt'), 'rb'))
    item_dict = json.load(open(os.path.join(cfg.dataset, 'item_dict.json'), 'r', encoding='utf-8'))
    # Build News Graph
    print("Building Graph")
    graph = build_entire_graph((train_data[0] + test_data[0], train_data[1] + test_data[1]))
    print("Original Graph built from all items:", len(graph.nodes))

    # padding node
    assert(not graph.has_node(0))
    graph.add_node(0)
    # independent node
    for i in range(len(item_dict)):
        if not graph.has_node(i):
            graph.add_node(i)
            # graph.add_edge(0, i, weight=1)
            # graph.add_edge(i, 0, weight=1)

    graph_path = os.path.join(cfg.dataset, "graph.bin")
    pickle.dump(graph, open(graph_path, 'wb'))
    # build neighbor dict
    ndict = build_neighbors_dict(graph)
    neighbor_path = os.path.join(cfg.dataset, "neighbor.pkl")
    pickle.dump(ndict, open(neighbor_path, 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose/sample')
    opt = parser.parse_args()

    main(opt)
