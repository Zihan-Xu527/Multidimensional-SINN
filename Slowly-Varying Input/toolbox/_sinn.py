#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init


class SINN(nn.Module):
    """
    LSTM-based trunk network used in SINN.

    In the slowly-varying example, this network takes the random input/noise
    sequence and produces time-dependent trunk features.

    Input shape:
        x.shape = (seq_len, batch_size, observed_dimension)

    Output shape:
        output.shape = (seq_len, batch_size, output_size)

    Example:
        observed_dimension = 2    # 2D white-noise input
        output_size = 20          # trunk feature dimension
    """

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
        # x.shape = (seq_len, batch_size, observed_dimension)
        # state is the optional initial LSTM state (h, c).
        x, (h, c) = self.lstm(self.dropout(x), state)

        # output.shape = (seq_len, batch_size, output_size)
        return self.readout(x), (h, c)


class BranchNet(nn.Module):
    """
    Branch network for the slowly-varying parameter.

    This network takes the slowly-varying parameter sequence, for example
    theta(t), and produces time-dependent branch features.

    Input shape:
        x.shape = (seq_len, batch_size, param_dim)

    Output shape:
        output.shape = (seq_len, batch_size, output_size)

    Example:
        param_dim = 1       # one slowly-varying parameter theta(t)
        output_size = 20    # same feature dimension as the trunk network
    """

    def __init__(self, param_dim, hidden_size, num_layers, output_size, dropout_p=0):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=param_dim,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.readout = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout_p)

        # Keep the custom initialization from the original slowly-varying code.
        self._init_weights()

    def _init_weights(self):
        # Initialize RNN weights.
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Initialize the linear readout layer.
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x, state=None):
        # x.shape = (seq_len, batch_size, param_dim)
        x, h = self.rnn(self.dropout(x), state)
        out = self.readout(x)
        return out, h


class SINNHybrid(nn.Module):
    """
    Hybrid trunk-branch SINN for slowly-varying stochastic processes.

    The trunk network processes the random input sequence.
    The branch network processes the slowly-varying parameter sequence.

    Their outputs are split into two halves and paired by inner products:

        first half  -> output component 1
        second half -> output component 2

    Therefore, the final output has shape:

        out.shape = (seq_len, batch_size, 2)

    In the adsorption-desorption example, the two output components are
    generated increments such as Delta theta_a and Delta theta_d.
    """

    def __init__(self, trunk_net, branch_net):
        super().__init__()
        self.trunk = trunk_net
        self.branch = branch_net

    def forward(self, noise_seq, param_seq):
        # noise_seq.shape = (seq_len, batch_size, noise_dim)
        # param_seq.shape = (seq_len, batch_size, param_dim)
        trunk_out, _ = self.trunk(noise_seq)
        branch_out, _ = self.branch(param_seq)

        D = trunk_out.size(-1)
        assert D % 2 == 0, "feature dimension must be even to split into two halves."
        assert branch_out.size(-1) == D, "trunk and branch feature dimensions must match."

        # Split feature dimension into two parts, one for each output channel.
        mid = D // 2
        t1, t2 = trunk_out[:, :, :mid], trunk_out[:, :, mid:]
        b1, b2 = branch_out[:, :, :mid], branch_out[:, :, mid:]

        # Inner products give scalar outputs for each component.
        inner1 = torch.sum(t1 * b1, dim=-1, keepdim=True)
        inner2 = torch.sum(t2 * b2, dim=-1, keepdim=True)

        # Final generated 2D stochastic process.
        out = torch.cat([inner1, inner2], dim=-1)
        return out
