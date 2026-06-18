#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from torch import nn
import torch
import torch.nn as nn
import torch.nn.init as init

# class SINN(nn.Module):
#     def __init__(self, observed_dimension, hidden_size, num_layers, output_size, dropout_p=0):
#         super().__init__()
#         self.lstm = nn.LSTM(
#             input_size=observed_dimension,
#             hidden_size=hidden_size,
#             num_layers=num_layers
#         )
#         self.readout = nn.Linear(
#             in_features=hidden_size,
#             out_features=output_size
#         )
#         self.dropout = nn.Dropout(p=dropout_p)

#         # Call custom initialization
#         self._init_weights()

#     def _init_weights(self):
#         # Initialize LSTM weights
#         for name, param in self.lstm.named_parameters():
#             if "weight_ih" in name:  # Input-hidden weights
#                 init.xavier_uniform_(param.data)
#             elif "weight_hh" in name:  # Hidden-hidden weights
#                 init.orthogonal_(param.data)
#             elif "bias" in name:
#                 param.data.fill_(0)
#                 # Optional: Set forget gate bias to 1 for better training stability
#                 n = param.size(0)
#                 param.data[n//4:n//2].fill_(1.0)

#         # Initialize Linear layer
#         init.xavier_uniform_(self.readout.weight)
#         if self.readout.bias is not None:
#             init.zeros_(self.readout.bias)

#     def forward(self, x, state=None):
#         x, (h, c) = self.lstm(self.dropout(x), state)
#         return self.readout(x), (h, c)


class SINN(nn.Module):
    def __init__(self, observed_dimension, hidden_size, num_layers, output_size, dropout_p=0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=observed_dimension,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.readout = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, state=None):
        x, (h, c) = self.lstm(self.dropout(x), state)
        return self.readout(x), (h, c)

# class SINN(nn.Module):
#     def __init__(self, observed_dimension, hidden_size, num_layers, output_size, architecture, dropout_p=0):
#         super().__init__()
#         self.architecture = architecture
#         if self.architecture == 'LSTM':
#             self.arch = nn.LSTM(
#                 input_size=observed_dimension,
#                 hidden_size=hidden_size,
#                 num_layers=num_layers
#             )
#         elif self.architecture == 'GRU':
#             self.arch = nn.GRU(
#                 input_size=observed_dimension,
#                 hidden_size=hidden_size,
#                 num_layers=num_layers
#             )
#         else:
#             raise RuntimeError(f'Unknown architecture {architecture}.')
#         self.readout = nn.Linear(
#             in_features=hidden_size,
#             out_features=output_size
#         )
#         self.dropout = nn.Dropout(p=dropout_p)

#     def forward(self, x, state=None):
#         if self.architecture == 'LSTM':
#             x, (h, c) = self.arch(self.dropout(x), state)
#             return self.readout(x), (h, c)
#         elif self.architecture == 'GRU':
#             x, (h) = self.arch(self.dropout(x), state)
#             return self.readout(x), (h)