import torch
import torch.nn as nn
from .ConvBlock import *

class Model(nn.Module):
    def __init__(self, static_ft, temporal_ft, out_channels, dmodel, out_features, kernel=2, stride=2, h=483, w=910, in_days=3):
        super().__init__()
        features = dmodel
        self.in_days = in_days
        in_channels = temporal_ft//in_days + static_ft
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

        self.out_days = out_features
        self.lstm = nn.LSTM(features, out_channels*2, batch_first=True)
        self.out = nn.Linear(out_channels*2, out_channels )
        self.dmodel = dmodel
        torch.nn.init.xavier_uniform_(self.out.weight)

    def forward(self, peat_map, temporal_ft, static_ft, future_step, hidden_state=None):
        b, t, seq_len, h, w = temporal_ft.size()
        _, k, _, _, _ = static_ft.size()
        t_ft = temporal_ft.permute(0, 2, 1, 3, 4).reshape(-1, t, h, w)
        s_ft = static_ft.repeat((1, 1, seq_len, 1, 1)).permute(0, 2, 1, 3, 4).reshape(-1, k, h,  w)
        x = torch.cat([s_ft, t_ft], dim=1) # (b*seq_len), num_ft, h, w 
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
        dec1 = self.decoder1(dec1).reshape(b, seq_len, -1, h, w).permute(0, 3, 4, 1, 2)
        out, _ = self.lstm(dec1.reshape(b*h*w, seq_len, -1))
        out = self.out(out[:, -1]).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return out.unsqueeze(2)