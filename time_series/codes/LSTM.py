# To BE : TSlib Covention 팔로우하도록 재수정 필요

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMModel(nn.Module):
    """
    A simple LSTM model
    """
    def __init__(self, configs):
        super(LSTMModel, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_dim = configs.hidden_dim
        self.channels = configs.enc_in
        self.individual = configs.individual

        self.lstm = nn.LSTM(input_size=self.channels, hidden_size=self.hidden_dim, batch_first=True)
        
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.hidden_dim, self.pred_len))
        else:
            self.Linear = nn.Linear(self.hidden_dim, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)  # Initial hidden state
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)  # Initial cell state
        out, _ = self.lstm(x, (h0, c0))  # LSTM computation
        
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](out[:, -1, :])  # Using last LSTM output
            x = output
        else:
            x = self.Linear(out[:, -1, :]).unsqueeze(1).expand(-1, self.pred_len, -1)
        
        return x  # [Batch, Output length, Channel]