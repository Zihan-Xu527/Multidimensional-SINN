#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn


class SINN(nn.Module):
    """
    Simple LSTM-based SINN generator.

    It maps an input noise sequence to a generated stochastic trajectory.

    Input:
        x.shape = (seq_len, batch_size, observed_dimension)

    Output:
        output.shape = (seq_len, batch_size, output_size)
    """

    def __init__(self, observed_dimension, hidden_size, num_layers, output_size, dropout_p=0):
        super().__init__()

        # LSTM backbone. The input dimension is the dimension of the random
        # input/noise at each time step.
        self.lstm = nn.LSTM(
            input_size=observed_dimension,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

        # Linear readout maps hidden states to the generated variables.
        self.readout = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )

        # Optional dropout applied to the input sequence.
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, state=None):
        # x.shape = (seq_len, batch_size, observed_dimension)
        # state is the optional initial LSTM state (h, c).
        x, (h, c) = self.lstm(self.dropout(x), state)

        # output.shape = (seq_len, batch_size, output_size)
        return self.readout(x), (h, c)