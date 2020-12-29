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
    def __init__(self, hidden_size, n_node):
        super(DynamicScore, self).__init__()
        self.hidden_size = hidden_size
        self.state_lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.residual_lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.residual2_lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.residual3_lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.loss = nn.MSELoss()
        # self.embedding = nn.Embedding(n_node, self.hidden_size)
        # self.embedding = embedding

    def forward(self, seq, y):
        seq_len = seq.size(1)
        # seq batch_size, seq_size, hidden_size
        seq = seq.permute(1, 0, 2)
        state_seq, (final_state, final_cell) = self.state_lstm(seq)

        residual_list = []
        for i in range(1, seq_len):
            residual_list.append(state_seq[i] - state_seq[i - 1])
        residual = torch.stack(residual_list, dim=0)
        residual_seq, (final_residual, final_residual_cell) = self.residual_lstm(residual)

        residual_list2 = []
        for i in range(1, seq_len - 1):
            residual_list2.append(residual_seq[i] - residual_seq[i - 1])
        residual2 = torch.stack(residual_list2, dim=0)
        residual2_seq, (final_residual2, final_residual_cell2) = self.residual2_lstm(residual2)

        residual_list3 = []
        for i in range(1, seq_len - 2):
            residual_list3.append(residual2_seq[i] - residual2_seq[i - 1])
        residual3 = torch.stack(residual_list3, dim=0)
        residual3_seq, (final_residual3, final_residual_cell3) = self.residual3_lstm(residual3)

        next_state = self.norm(final_state + final_residual + final_residual2 + final_residual3).reshape(-1, self.hidden_size)

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
