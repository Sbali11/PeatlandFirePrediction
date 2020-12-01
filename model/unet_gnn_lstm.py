
import argparse
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, FastRGCNConv, GatedGraphConv
from torch_geometric.nn import GraphUNet
from .unet import Model as Unet
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
from collections import OrderedDict
from .ConvBlock import *

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
        #self.down_convs.append(FastRGCNConv(in_channels, channels, num_relations=num_relations))

        self.down_convs.append(GCNConv(in_channels, channels))
        print("IN_CHANNELS", in_channels)
        #, num_relations=num_relations))
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
        print("XX", x.shape)
        x = self.down_convs[0](x, edge_index)
        #, edge_type)
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
        self.kh = 483
        self.kw = 910
        num_nodes =  dataset.x.shape[0]
        num_relations = dataset.num_relations
        ft_shape = dataset.x.shape[1]
        out_days = dataset.out_days
        self.in_days = dataset.in_days
        num_output_nodes = dataset.num_output_nodes
        self.gconv1 = GCNConv(ft_shape, ft_shape)
        #self.gconv1 = GCNConv(ft_shape, ft_shape)

        in_channels = 2*ft_shape
        features = dmodel
        self.encoder1 = Model._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = Model._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = Model._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = Model._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = Model._block(features * 8, features * 16, name="bottleneck")
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = Model._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = Model._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = Model._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = Model._block(features * 2, features, name="dec1")
        self.lstm = nn.LSTM(features, out_channels*2, batch_first=True)
        self.out = nn.Linear(out_channels*2, out_channels )
        self.out_days = out_days
        self.dmodel = dmodel
        torch.nn.init.xavier_uniform_(self.gconv1.weight)
        torch.nn.init.xavier_uniform_(self.encoder1[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder1[3].weight)
        torch.nn.init.xavier_uniform_(self.encoder2[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder2[3].weight)
        torch.nn.init.xavier_uniform_(self.encoder3[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder3[3].weight)
        self.out_channels = out_channels
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, data):
        b = data.y.shape[0]
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_types
        f = data.x
        x = self.gconv1(data.x, edge_index)
        first_size = x.shape[0]
        f = f.view(1, self.kh, self.kw, data.in_days, -1).permute(0, 3, 4, 1, 2)
        x = x.view(1, self.kh, self.kw, data.in_days, -1).permute(0, 3, 4, 1, 2)
        x = torch.cat([x, f], dim=1).view(self.in_days, -1, self.kh, self.kw)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck, output_size=enc4.size())
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4, output_size=enc3.size())
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3, output_size=enc2.size())
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2, output_size=enc1.size())
        dec1 = torch.cat((dec1, enc1), dim=1)
        b = 1
        h = self.kh
        w = self.kw
        seq_len = data.in_days
        dec1 = self.decoder1(dec1).view(b, seq_len, -1, h, w).permute(0, 3, 4, 1, 2)
        out, _ = self.lstm(dec1.reshape(b*h*w, seq_len, -1))
        out = self.out(out[:, -1]).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return out.unsqueeze(2)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            padding_mode="circular",
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            padding_mode="circular",
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )