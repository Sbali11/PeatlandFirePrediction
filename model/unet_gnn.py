
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
from .ConvBlock import *

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
        in_channels = 2*ft_shape
        features = dmodel
        self.encoder1 = Block(in_channels, features, b_type="start")
        self.encoder2 = Block(features, features * 2, b_type="down_sample")
        self.encoder3 = Block(features * 2, features * 4, b_type="down_sample")
        self.encoder4 = Block(features * 4, features * 8, b_type="down_sample")
        self.bottleneck = Block(features * 8, features * 16, b_type="bottleneck")
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = Block(features * 16, features * 8, b_type="up_sample")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = Block(features *  8, features * 4, b_type="up_sample")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = Block(features * 4, features * 2, b_type="up_sample")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = Block(features * 2, features, b_type="up_sample")
        self.out_days = out_days
        self.dmodel = dmodel
        torch.nn.init.xavier_uniform_(self.gconv1.weight)
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
        f = f.reshape(1, self.kh, self.kw, -1).permute(0, 3, 1, 2)
        x = x.reshape(1, self.kh, self.kw, -1).permute(0, 3, 1, 2)
        x = torch.cat([x, f], dim=1)
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
        dec1 = self.decoder1(dec1)
        out = dec1
        return out.unsqueeze(2)