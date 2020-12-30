# -*- coding: utf-8 -*-
"""Script for building the training examples.

"""
import os
import json
import random
import pickle
import argparse
import multiprocessing as mp
from typing import List, Dict
from numpy.lib.function_base import append
import math
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import networkx as nx
from graph_datasets import MultiSessionsGraph
from torch_geometric.data import InMemoryDataset, Data

random.seed(7)

def build_subgraph(graph, hist_items, neighbor_items):
    
    all_items = hist_items | neighbor_items
    # nodes 按照 all_ents 的顺序返回和编码
    all_items = list(all_items)
    sub_graph = graph.subgraph(all_items)

    impression_ent_id_map = {ent: idx for idx, ent in enumerate(all_items)}
    sub_graph = nx.relabel_nodes(sub_graph, impression_ent_id_map)

    edges = list(sub_graph.edges)
    source_nodes, target_nodes = [], []
    for edge in edges:
        source_nodes.append(edge[0])
        target_nodes.append(edge[1])
    edge_index = [source_nodes, target_nodes]

    hist_mask = [1 if x in hist_items else 0 for x in all_items]

    return all_items, edge_index, hist_mask

def trim_seq(seq, length=30):
    slen = len(seq)

    if slen >= length:
        return seq[-length:]
    else:
        new_seq = []
        for t in range(length - slen):
            new_seq.append(0)
        return new_seq + seq

def build_examples(rank, args, session_seq, graph, neighbor_dict, output_path):
    
    random.seed(7)

    data_list = []
    no_neighbor = 0
    no_edge = 0
    for sequences, y in tqdm(zip(session_seq[0], session_seq[1])):
        item_set = set()
        cur_len = len(sequences)
        for i in range(cur_len):
            item = sequences[cur_len - i - 1]
            if graph.has_node(item):
                item_set.add(item)
            if len(item_set) >= args.node_num:
                break

        assert(len(item_set) > 0)
        neighbor_set = set()
        for item in item_set:
            for n in neighbor_dict[item]:
                neighbor_set.add(n)

        x, edge_index, hist_mask = build_subgraph(graph, item_set, neighbor_set)
        if len(neighbor_set) < 1:
            no_neighbor += 1
        if len(edge_index[0]) < 1:
            no_edge += 1

        x = torch.LongTensor(x).unsqueeze(1)
        y = torch.LongTensor([y])
        edge_index = torch.LongTensor(edge_index)
        curdata = Data(x=x, edge_index=edge_index, y=y)
        curdata.sequences = torch.LongTensor(trim_seq(sequences, length=args.seq_len)).unsqueeze(0)
        curdata.hist_mask = torch.LongTensor(hist_mask).unsqueeze(1)
        data_list.append(curdata)

    data, slices = MultiSessionsGraph.collate(data_list)
    torch.save((data, slices), output_path)
    # print(no_neighbor, no_edge)


def main(args):
    session_seq = pickle.load(open(os.path.join(args.dataset, 'raw', args.filetype + '.txt'), 'rb'))
    graph = pickle.load(open(os.path.join(args.dataset, 'graph.bin'), 'rb'))
    neighbor_dict = pickle.load(open(os.path.join(args.dataset, "neighbor.pkl"), 'rb'))

    subdf_len = math.ceil(len(session_seq[0]) / args.processes)
    cut_indices = [x * subdf_len for x in range(1, args.processes)]
    session_seqs = np.split(np.array(session_seq[0], dtype=object), cut_indices)
    session_ys = np.split(session_seq[1], cut_indices)
    
    processes = []
    for i in range(args.processes):
        output_path = os.path.join(args.dataset, "raw", args.filetype + "-{}.pt".format(i))
        assert(len(session_seqs[i]) == len(session_ys[i]))
        p = mp.Process(target=build_examples, args=(i, args, (session_seqs[i], session_ys[i]), graph, neighbor_dict, output_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--filetype", default="train", type=str,
                        help="train or test")
    parser.add_argument("--node_num", default=60, type=int,
                        help="subgraph max node number")
    parser.add_argument("--seq_len", default=30, type=int,
                        help="length of sequence")
    parser.add_argument("--processes", default=10, type=int,
                        help="Processes number")
    parser.add_argument("--dataset", default="diginetica", type=str,
                        help="dataset name")

    args = parser.parse_args()

    main(args)