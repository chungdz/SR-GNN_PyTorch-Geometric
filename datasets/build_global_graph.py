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
    edges = set()

    for sequences, y in zip(train_data[0], train_data[1]):
        cur_len = len(sequences)
        if cur_len < 2:
            continue
        if cur_len == 2:
            edges.add((sequences[0], sequences[1]))
            edges.add((sequences[1], sequences[0]))
        else:
            for i in range(1, cur_len - 1):
                edges.add((sequences[i], sequences[i - 1]))
                edges.add((sequences[i - 1], sequences[i]))

    dg = nx.DiGraph()
    dg.add_edges_from(list(edges))

    return dg


def main(cfg):

    train_data = pickle.load(open(os.path.join(cfg.dataset, 'raw/train.txt'), 'rb'))
    # Build News Graph
    print("Building Graph")
    graph = build_entire_graph(train_data)
    print("Original Graph built from all items:", len(graph.nodes))

    graph.add_node("<pad>")
    graph_path = os.path.join(cfg.dataset, "graph.bin")
    pickle.dump(graph, open(graph_path, 'wb'))

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose/sample')
    opt = parser.parse_args()

    main(opt)
