""" 
Representation learning encoder models, adapted from 
https://github.com/vanderschaarlab/SyncTwin-NeurIPS-2021
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import math
import synctwin.grud as grud
import torch
import torch.nn as nn
from synctwin._config import DEVICE


class RegularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirectional=True, device=DEVICE):
        super(RegularEncoder, self).__init__()
        self.input_dim = input_dim
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=bidirectional).to(device)
        if bidirectional:
            self.hidden_dim = hidden_dim * 2
        else:
            self.hidden_dim = hidden_dim

        attn_v_init = torch.ones(self.hidden_dim).to(device)
        self.attn_v = nn.Parameter(attn_v_init)

    def forward(self, x, t, mask):
        # T, B, Dh
        h, _ = self.lstm(x)  # pylint: disable=not-callable

        # T, B
        attn_score = torch.matmul(h, self.attn_v) / math.sqrt(self.hidden_dim)
        attn_weight = torch.softmax(attn_score, dim=0)

        # B, Dh
        C = torch.sum(h * attn_weight.unsqueeze(-1), dim=0)
        return C


class GRUDEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, device=DEVICE):
        super(GRUDEncoder, self).__init__()
        self.input_dim = input_dim
        self.device = device

        self.grud = grud.GRUD(input_dim, hidden_dim, device=device).to(device)
        self.hidden_dim = hidden_dim

        attn_v_init = torch.ones(self.hidden_dim).to(device)
        self.attn_v = nn.Parameter(attn_v_init)

    def forward(self, x, t, mask):
        grud_in = self.grud.get_input_for_grud(t, x, mask)

        # T, B, Dh
        h = self.grud(grud_in).permute((1, 0, 2))  # pylint: disable=not-callable

        # T, B
        attn_score = torch.matmul(h, self.attn_v) / math.sqrt(self.hidden_dim)
        attn_weight = torch.softmax(attn_score, dim=0)

        # B, Dh
        C = torch.sum(h * attn_weight.unsqueeze(-1), dim=0)
        return C
    