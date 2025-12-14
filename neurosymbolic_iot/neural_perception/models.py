from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CasasGRUClassifier(nn.Module):
    """
    Token embedding + GRU over variable-length event sequences.
    Input: token_ids [B, T], time_deltas [B, T], lengths [B]
    """
    def __init__(self, vocab_size: int, num_classes: int, emb_dim: int = 64, hidden: int = 128, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=emb_dim + 1,  # + time delta feature
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, token_ids: torch.Tensor, time_deltas: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.emb(token_ids)  # [B, T, E]
        td = time_deltas.unsqueeze(-1)  # [B, T, 1]
        inp = torch.cat([x, td], dim=-1)  # [B, T, E+1]

        packed = pack_padded_sequence(inp, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)  # h: [L, B, H]
        last = h[-1]             # [B, H]
        return self.head(last)


class SphereLSTMClassifier(nn.Module):
    """
    LSTM over fixed-length acceleration sequences.
    Input: seq [B, T, F]
    """
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 128, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, (h, _) = self.lstm(seq)  # h: [L, B, H]
        last = h[-1]                  # [B, H]
        return self.head(last)
