"""Sequence encoders used by BlendEmo."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, mask=None):
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class LightEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = AttentionPooling(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        x = self.net(x)
        x = self.pool(x, mask)
        return self.norm(x)


class SeqEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pool = AttentionPooling(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.pool(x, mask)
        return self.norm(x)
