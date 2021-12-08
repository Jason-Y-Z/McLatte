""" 
Representation learning models, adapted from 
https://github.com/vanderschaarlab/SyncTwin-NeurIPS-2021
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import math
import torch
import torch.nn as nn
from ._grud import GRUD
from .config import DEVICE


class LstmEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, treatment_dim, bidirectional=True, device=DEVICE):
        super(LstmEncoder, self).__init__()
        self.input_dim = input_dim
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=bidirectional).to(device)
        self.hidden_dim = hidden_dim if not bidirectional else hidden_dim * 2
        
        attn_v_init = torch.ones(self.hidden_dim).to(device)
        self.attn_v = nn.Parameter(attn_v_init)

        self.trt_linear = nn.Linear(self.hidden_dim + treatment_dim, self.hidden_dim).to(device)

    def forward(self, x, a, *_):
        h, _ = self.lstm(x)  # [T, batch_size, hidden_dim]  
        attn_score = torch.matmul(h, self.attn_v) / math.sqrt(self.hidden_dim)  # [T, batch_size]
        attn_weight = torch.softmax(attn_score, dim=0)  # [T, batch_size]
        attn_h = torch.sum(h * attn_weight.unsqueeze(-1), dim=0)  # [batch_size, hidden_dim]

        a_attn_h = torch.cat((attn_h, a), dim=1)  # [batch_size, hidden_dim + treatment_dim]
        C = self.trt_linear(a_attn_h)  # [batch_size, hidden_dim]
        return C


class GRUDEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, treatment_dim, device=DEVICE):
        super(GRUDEncoder, self).__init__()
        self.input_dim = input_dim
        self.device = device

        self.grud = GRUD(input_dim, hidden_dim, device=device).to(device)
        self.hidden_dim = hidden_dim

        attn_v_init = torch.ones(self.hidden_dim).to(device)
        self.attn_v = nn.Parameter(attn_v_init)
        
        self.trt_linear = nn.Linear(hidden_dim + treatment_dim, hidden_dim).to(device)

    def forward(self, x, a, t, mask):
        grud_in = self.grud.get_input_for_grud(t, x, mask)
        h = self.grud(grud_in).permute((1, 0, 2))  # [T, batch_size, hidden_dim] 

        attn_score = torch.matmul(h, self.attn_v) / math.sqrt(self.hidden_dim)  # [T, batch_size]
        attn_weight = torch.softmax(attn_score, dim=0)  # [T, batch_size]

        attn_h = torch.sum(h * attn_weight.unsqueeze(-1), dim=0)  # [batch_size, hidden_dim]
        a_attn_h = torch.cat((attn_h, a), dim=1)  # [batch_size, hidden_dim + treatment_dim]
        C = self.trt_linear(a_attn_h)  # [batch_size, hidden_dim]
        return C


class LstmDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_seq_len, device=DEVICE):
        super(LstmDecoder, self).__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.lstm = nn.LSTM(hidden_dim, hidden_dim).to(device)
        self.linear = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, C, *_):  # C: [batch_size, hidden_dim]
        # Run first timestep to get the first hidden vector
        out, hidden = self.lstm(C.unsqueeze(0)) 
        out = self.linear(out)  # [batch_size, output_dim]

        # Decode the covariates by running through the timesteps
        out_list = [out]
        for _ in range(self.max_seq_len - 1):
            out, hidden = self.lstm(C.unsqueeze(0), hidden) 
            out = self.linear(out)
            out_list.append(out)
        return torch.cat(out_list, dim=0)  # [T, batch_size, output_dim]


class GRUDDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_seq_len, device=DEVICE):
        super(GRUDDecoder, self).__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.grud = GRUD(hidden_dim, hidden_dim, device=device).to(device)
        self.linear = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, C, t, mask):  # C: [batch_size, hidden_dim]
        x = C[None, :, :].repeat(t.shape[0], 1, 1)
        t = t[:, :, 0:1].repeat(1, 1, x.shape[2])
        grud_in = self.grud.get_input_for_grud(t, x, mask)
        
        h = self.grud(grud_in).permute((1, 0, 2))  # [T, batch_size, hidden_dim]
        
        out = self.linear(h)  # [T, batch_size, output_dim]
        return out


class LinearDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_seq_len, device=DEVICE):
        super(LinearDecoder, self).__init__()
        assert output_dim == 1
        self.device = device
        self.max_seq_len = max_seq_len
        self.linear = nn.Linear(hidden_dim, max_seq_len).to(device)

    def forward(self, C, *_):  # C: [batch_size, hidden_dim]
        out = self.linear(C)  
        out = out.T.unsqueeze(-1)
        return out  # [T, batch_size, output_dim]
