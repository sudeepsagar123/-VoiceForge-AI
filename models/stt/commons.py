"""
Conformer-CTC Common Utilities
Shared layers for the Conformer-based STT model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: [B, T, D]
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding using sinusoidal embeddings."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, length):
        """Get positional encoding for given length."""
        return self.pe[:length].unsqueeze(0)


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)"""

    def forward(self, x):
        return x * torch.sigmoid(x)


def make_pad_mask(lengths, max_len=None):
    """Create padding mask.

    Args:
        lengths: [B] tensor of lengths
        max_len: int, maximum length

    Returns:
        mask: [B, max_len] boolean tensor (True = padded)
    """
    if max_len is None:
        max_len = lengths.max()
    indices = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return indices >= lengths.unsqueeze(1)
