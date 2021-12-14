""" 
Representation learning decoder models, adapted from 
https://github.com/vanderschaarlab/SyncTwin-NeurIPS-2021
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import synctwin.grud as grud
import torch
import torch.nn as nn


class RegularDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_seq_len):
        super(RegularDecoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, C, t, mask):
        # C: B, Dh
        out, hidden = self.lstm(C.unsqueeze(0))  # pylint: disable=not-callable
        out = self.lin(out)

        out_list = [out]
        # run the remaining iterations
        for t in range(self.max_seq_len - 1):
            out, hidden = self.lstm(
                C.unsqueeze(0), hidden
            )  # pylint: disable=not-callable
            out = self.lin(out)
            out_list.append(out)

        return torch.cat(out_list, dim=0)


class LSTMTimeDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_seq_len):
        super(LSTMTimeDecoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.time_lin = nn.Linear(1, hidden_dim)

    def forward(self, C, t, mask):
        t_delta = t[1:] - t[:-1]
        t_delta_mat = torch.cat((torch.zeros_like(t_delta[0:1, ...]), t_delta), dim=0)
        time_encoded = self.time_lin(t_delta_mat[:, :, 0:1])

        # C: B, Dh
        lstm_in = torch.cat((C.unsqueeze(0), time_encoded[0:1, ...]), dim=-1)
        out, hidden = self.lstm(lstm_in)  # pylint: disable=not-callable
        out = self.lin(out)

        out_list = [out]
        # run the remaining iterations
        for t in range(self.max_seq_len - 1):
            lstm_in = torch.cat(
                (C.unsqueeze(0), time_encoded[(t + 1) : (t + 2), ...]), dim=-1
            )
            out, hidden = self.lstm(lstm_in, hidden)  # pylint: disable=not-callable
            out = self.lin(out)
            out_list.append(out)

        return torch.cat(out_list, dim=0)


class GRUDDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_seq_len):
        super(GRUDDecoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.grud = grud.GRUD(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, C, t, mask):
        # C: B, Dh
        x = C[None, :, :].repeat(t.shape[0], 1, 1)
        mask = torch.ones_like(x)
        t = t[:, :, 0:1].repeat(1, 1, x.shape[2])

        grud_in = self.grud.get_input_for_grud(t, x, mask)

        # T, B, Dh
        h = self.grud(grud_in).permute((1, 0, 2))  # pylint: disable=not-callable
        out = self.lin(h)

        return out


class LinearDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_seq_len):
        super(LinearDecoder, self).__init__()
        assert output_dim == 1
        self.max_seq_len = max_seq_len
        self.lin = nn.Linear(hidden_dim, max_seq_len)

    def forward(self, C, t, mask):
        # C: B, Dh -> B, T
        out = self.lin(C)  # pylint: disable=not-callable
        out = out.T.unsqueeze(-1)

        return out
