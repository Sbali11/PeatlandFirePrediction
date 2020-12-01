import torch
import torch.nn as nn
from collections import OrderedDict
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, static_ft, temporal_ft, out_channels, dmodel, out_features, h=483, w=910, in_days=3):
        super().__init__()
        features = dmodel
        self.in_days = in_days
        in_channels = temporal_ft//in_days + static_ft
        self.linear = nn.Linear(static_ft + temporal_ft, out_channels)
    
    def forward(self, peat_map, temporal_ft, static_ft, future_step, hidden_state=None):
        b, t, seq_len, h, w = temporal_ft.size()
        _, k, _, _, _ = static_ft.size()
        temporal_ft = temporal_ft.permute(0, 3, 4, 1, 2).reshape(-1, t * seq_len)
        static_ft = static_ft.permute(0, 3, 4, 1, 2).reshape(temporal_ft.shape[0], -1)
        x = torch.cat([static_ft, temporal_ft], dim=-1)
        num_ft = x.shape[1]
        x = self.linear(x)
        #print(x.shape)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return x.unsqueeze(2)
