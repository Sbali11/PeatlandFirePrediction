import argparse
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, FastRGCNConv, GatedGraphConv
#from torch_geometric.nn import GraphUNet
from torch_geometric.utils.loop import remove_self_loops, contains_self_loops
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling, GCNConv
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_geometric.utils.repeat import repeat
from torch_geometric.utils.dropout import dropout_adj

class GraphRUNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, act=F.relu, num_relations=4):
        super(GraphRUNet, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(FastRGCNConv(in_channels, channels, num_relations=num_relations))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(channels, channels))
        self.up_convs.append(GCNConv(channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_type, batch=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))
        x = self.down_convs[0](x, edge_index, edge_type)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_types = [edge_type]
        perms = []

        for i in range(1, self.depth + 1):
            #edge_index, edge_type = self.augment_adj(edge_index, edge_type,
            #                                           x.size(0))
            x, edge_index, edge_type, batch, perm, _ = self.pools[i - 1](
                    x, edge_index, edge_type, batch)
            x = self.down_convs[i](x, edge_index)
            x = self.act(x)
            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_types += [edge_type]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i
            res = xs[j]
            edge_index = edge_indices[j]
            edge_type = edge_types[j]
            perm = perms[j]
            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)
            x = self.up_convs[i](x, edge_index)
            x = self.act(x) if i < self.depth - 1 else x
        return x

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)

class Model(torch.nn.Module):
    def __init__(self, dataset, dmodel=16, out_channels=1):
        super(Model, self).__init__()
        num_nodes =  dataset.x.shape[0]
        num_relations = dataset.num_relations
        ft_shape = dataset.x.shape[1]
        out_days = dataset.out_days
        self.in_days = dataset.in_days
        num_output_nodes = dataset.num_output_nodes
        print("DMODEL", dmodel)
        self.lstm = nn.LSTM(ft_shape//self.in_days, ft_shape//self.in_days, batch_first=True)
        self.unet = GraphRUNet(ft_shape//self.in_days, dmodel, dmodel, depth=2, num_relations=num_relations)
        self.out = nn.Linear( dmodel, out_channels)
        self.out_channels = out_channels
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data):
        b = data.y.shape[0]
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_types
        x = x.view(x.shape[0], self.in_days, x.shape[1]//self.in_days)
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.unet(x, edge_index, edge_type)
        return self.out(self.relu(x))
