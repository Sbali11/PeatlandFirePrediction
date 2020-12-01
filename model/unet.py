"""
Original U-Net model implementation.
"""

import torch
import torch.nn as nn
from .ConvBlock import *


class Model(nn.Module):
    def __init__(self, static_ft, temporal_ft, out_channels, dmodel, out_features):
        super().__init__()
        features = dmodel
        in_channels = temporal_ft + static_ft
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
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels*out_features, kernel_size=1)
        self.out_days = out_features

    def forward(self, peat_map, temporal_ft, static_ft, future_step, hidden_state=None):
        b, _, _, h, w = temporal_ft.size()
        t_ft = temporal_ft.reshape(b, -1, h, w)
        s_ft = static_ft.reshape(b, -1, h, w)
        x = torch.cat([s_ft, t_ft], dim=1)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)     
        print(x.size(), enc1.size())   
        bottleneck = self.bottleneck(enc4)
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
        dec1 = self.decoder1(dec1)
        out =  self.conv(dec1).view(b, -1, self.out_days, h, w)
        return out * peat_map
