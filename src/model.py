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
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn import GlobalAttention

class GateLayer(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int) -> None:
        super(GateLayer, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.Tanh()
        )

        self.gate_layer = nn.Linear(hidden_size, 1)

    def forward(self, seqs):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :return: shape [batch_size, 1]
        """
        gates = self.gate_layer(self.h1(seqs))
        return gates

class GatNet(nn.Module):
    def __init__(self, node_feat_size, conv_hidden_size, dropout=0.2):
        super(GatNet, self).__init__()
        self.conv1 = GATConv(node_feat_size, conv_hidden_size // 2, dropout=dropout, heads=2)

        self.gate_layer = nn.Sequential(
            nn.Linear(2 * node_feat_size, node_feat_size, bias=False),
            nn.Sigmoid()
        )
        self.fuse_layer = nn.Sequential(
            nn.Linear(node_feat_size, node_feat_size, bias=False),
            nn.Tanh()
        )
        self.output_proj = nn.Linear(node_feat_size, node_feat_size)
        # self.layer_norm = nn.LayerNorm(node_feat_size)

    def forward(self, nodes, edge_index):
        """
        nodes: shape [*, node_feat_size]
        edge_index: shape [2, *]
        """
        x = self.conv1(nodes, edge_index)
        # x = self.output_proj(x)
        h = torch.cat([nodes, x], dim=-1)
        gate = self.gate_layer(h)
        output = gate * self.fuse_layer(x) + (1.0 - gate) * nodes
        # output = self.gate_layer(h) * self.fuse_layer(h)
        
        # output = self.layer_norm(x + nodes)
        return output

class MaskGlobalAttention(GlobalAttention):
    def forward(self, x, batch, mask, size=None):
        """
        x: shape [node_num, in_channel]
        batch: shape [node_num, ]
        mask: shape [node_num, *]

        return: shape [batch_size, *, out_channel]
        """
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        out_list = []
        mask_size = mask.size(1)
        gate = gate.squeeze(1)
        for i in range(mask_size):
            mask_i = mask[:, i]
            gate_mask = gate.masked_fill(mask_i == 0, -1e9)
            gate_mask = softmax(gate_mask.view(-1, 1), batch, num_nodes=size)
            out = scatter_add(gate_mask * x, batch, dim=0, dim_size=size)
            out_list.append(out.unsqueeze(1))

        return torch.cat(out_list, dim=1)

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
        self.graph_emb = nn.Embedding(self.n_node, self.hidden_size)
        self.dynamic = DynamicScore(self.hidden_size, self.n_node)
        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()

        self.gnn = GatNet(hidden_size, hidden_size)
        self.gnn2 = GatNet(hidden_size, hidden_size)
        self.gate_layer = GateLayer(hidden_size, int(hidden_size / 2))
        self.pool = MaskGlobalAttention(self.gate_layer)
        self.weight_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data):
        x, edge_index, batch, seq, y, hist_mask = data.x - 1, data.edge_index, data.batch, data.sequences, data.y, data.hist_mask

        seq = self.embedding(seq)
        y = self.embedding(y).unsqueeze(1)
        dynamic, MSELoss = self.dynamic(seq, y)

        nodes = self.graph_emb(x).squeeze()
        node_hiddens = self.gnn(nodes, edge_index)
        node_hiddens = self.gnn2(node_hiddens, edge_index)
        pooled_hiddens = self.pool(x=node_hiddens, batch=batch, mask=hist_mask).squeeze()

        user_weight = self.weight_layer(torch.cat([dynamic, pooled_hiddens], dim=-1))

        score_1 = torch.mm(dynamic, self.embedding.weight.transpose(1, 0))
        score_2 = torch.mm(pooled_hiddens, self.graph_emb.weight.transpose(1, 0))
        final_score = user_weight * score_1 + (1.0 - user_weight) * score_2
        
        return final_score, MSELoss
