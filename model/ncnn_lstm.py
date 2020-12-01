import torch
import torch.nn as nn
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ConvBlock import * 

class Model(nn.Module):
    def __init__(self, static_ft, temporal_ft, out_channels, dmodel, out_features, h=483, w=910, in_days=3):
        super().__init__()
        features = dmodel
        self.in_days = in_days
        in_channels = temporal_ft//in_days + static_ft
        self.encoder1 = Block(in_channels, features, b_type="conv")
        self.encoder2 = Block(features, features,  b_type="conv")
        self.encoder3 = Block(features, features,  b_type="conv")
        self.encoder4 = Block(features, features,  b_type="conv")
        self.lstm = nn.LSTM(features, 2*out_channels, batch_first=True)
        self.out = nn.Linear(2*out_channels, out_channels)
        self.dmodel = dmodel
        torch.nn.init.xavier_uniform_(self.out.weight)

    def forward(self, peat_map, temporal_ft, static_ft, future_step, hidden_state=None):
        b, t, seq_len, h, w = temporal_ft.size()
        _, k, _, _, _ = static_ft.size()
        t_ft = temporal_ft.permute(0, 2, 1, 3, 4).reshape(-1, t, h, w)
        s_ft = static_ft.repeat((1, 1, seq_len, 1, 1)).permute(0, 2, 1, 3, 4).reshape(-1, k, h,  w)
        x = torch.cat([s_ft, t_ft], dim=1) # (b*seq_len), num_ft, h, w 
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))
        x = F.relu(self.encoder3(x))
        x = F.relu(self.encoder4(x))
        x = x.reshape(b, seq_len, -1, h, w).permute(0, 3, 4, 1, 2)
        x = x.reshape(b*h*w, seq_len, -1)
        out, _ = self.lstm(x)
        out = self.out(out)
        out = out[:, -1].reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return out.unsqueeze(2)