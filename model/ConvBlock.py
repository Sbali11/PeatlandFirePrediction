# Single Block architecture is similar to https://github.com/esowc/wildfire-forecasting

import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dblock = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                out_channels=out_channels,
                                                kernel_size=3,
                                                padding=1,
                                                bias=True,
                                                padding_mode="circular"),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.ReLU(inplace=True))
        torch.nn.init.xavier_uniform_(self.dblock[0].weight)

    def forward(self, x):
        return self.dblock(x)


class Block(nn.Module):
    def __init__(self, in_channels, features, b_type='down_sample'):
        super().__init__()
        if b_type in ['down_sample', 'bottleneck']:
            self.block =  nn.Sequential(
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    ConvLayer(in_channels, features),
                                    ConvLayer(features, features))
        else:
            self.block =  nn.Sequential(ConvLayer(in_channels, features),
                                    ConvLayer(features, features))

    def forward(self, x):
        return self.block(x)
