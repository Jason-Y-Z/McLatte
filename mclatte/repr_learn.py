""" 
Representation learning models, adapted from 
https://github.com/vanderschaarlab/SyncTwin-NeurIPS-2021
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import math
import torch
import torch.nn as nn


class LstmEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, treatment_dim):
        super(LstmEncoder, self).__init__()
        self.input_dim = input_dim

        self._lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=False)
        self.hidden_dim = hidden_dim
        
        attn_v_init = torch.ones(self.hidden_dim)
        self._attn_v = nn.Parameter(attn_v_init)

        self._trt_linear = nn.Linear(self.hidden_dim + treatment_dim, self.hidden_dim)

    def forward(self, x, a, *_):
        h, _ = self._lstm(x)  # [T, batch_size, hidden_dim]  
        attn_score = torch.matmul(h, self._attn_v) / math.sqrt(self.hidden_dim)  # [T, batch_size]
        attn_weight = torch.softmax(attn_score, dim=0)  # [T, batch_size]
        attn_h = torch.sum(h * attn_weight.unsqueeze(-1), dim=0)  # [batch_size, hidden_dim]

        a_attn_h = torch.cat((attn_h, a), dim=1)  # [batch_size, hidden_dim + treatment_dim]
        C = self._trt_linear(a_attn_h)  # [batch_size, hidden_dim]
        return C


class LstmDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_seq_len):
        super(LstmDecoder, self).__init__()
        self.max_seq_len = max_seq_len
        self._lstm = nn.LSTM(hidden_dim, hidden_dim)
        self._linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, C, *_):  # C: [batch_size, hidden_dim]
        # Run first timestep to get the first hidden vector
        out, hidden = self._lstm(C.unsqueeze(0)) 
        out = self._linear(out)  # [batch_size, output_dim]

        # Decode the covariates by running through the timesteps
        out_list = [out]
        for _ in range(self.max_seq_len - 1):
            out, hidden = self._lstm(C.unsqueeze(0), hidden) 
            out = self._linear(out)
            out_list.append(out)
        return torch.cat(out_list, dim=0)  # [T, batch_size, output_dim]


ENCODERS = {
    'lstm': LstmEncoder,
}
DECODERS = {
    'lstm': LstmDecoder,
}
