# -*- coding: utf-8 -*-
"""
Created on 4/4/2019
@author: RuihongQiu
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv

class DynamicScore(nn.Module):
    def __init__(self, hidden_size, n_node, layer_num=3):
        super(DynamicScore, self).__init__()
        self.hidden_size = hidden_size
        self.state_lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.layer_num = layer_num
        self.residual_lstms = nn.ModuleList([nn.LSTM(self.hidden_size, self.hidden_size) for k in range(layer_num)])
        self.norm = nn.LayerNorm(self.hidden_size)
        self.loss = nn.MSELoss()
        # self.embedding = nn.Embedding(n_node, self.hidden_size)
        # self.embedding = embedding

    def forward(self, seq, y):
        seq_len = seq.size(1)
        # seq batch_size, seq_size, hidden_size
        seq = seq.permute(1, 0, 2)
        state_seq, (final_state, final_cell) = self.state_lstm(seq)

        cur_seq = state_seq
        cur_len = seq_len
        cur_state = final_state

        for l in range(self.layer_num):

            residual_list = []
            for i in range(1, cur_len):
                residual_list.append(cur_seq[i] - cur_seq[i - 1])
            residual = torch.stack(residual_list, dim=0)
            cur_seq, (final_residual, final_residual_cell) = self.residual_lstms[l](residual)
            cur_state += final_residual
            cur_len = cur_len - 1

        next_state = self.norm(cur_state).reshape(-1, self.hidden_size)

        # news loss
        y = y.permute(1, 0, 2)
        # y 1, batch_size, hidden_size
        real_state, (_h, _c) = self.state_lstm(y, (final_state, final_cell))
        real_state = real_state.reshape(-1, self.hidden_size)
        
        return next_state, self.loss(next_state, real_state)


class GNNModel(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """
    def __init__(self, hidden_size, n_node):
        super(GNNModel, self).__init__()
        self.hidden_size, self.n_node = hidden_size, n_node
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.dynamic = DynamicScore(self.hidden_size, self.n_node)
        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data):
        x, edge_index, batch, seq, y = data.x - 1, data.edge_index, data.batch, data.sequences, data.y

        seq = self.embedding(seq)
        y = self.embedding(y).unsqueeze(1)

        dynamic, MSELoss = self.dynamic(seq, y)

        score = torch.mm(dynamic, self.embedding.weight.transpose(1, 0))
        
        return score, MSELoss
