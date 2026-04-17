"""
VITS2 Text Encoder
Transformer-based encoder with multi-head attention and feed-forward layers.
Outputs mean and log-variance for the VAE latent space.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tts.commons import LayerNorm


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with relative positional encoding."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: int = 4,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        # Relative position embedding
        if window_size is not None:
            n_heads_rel = 1
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: [B, C, T]
            attn_mask: [B, 1, T] or None
        """
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        x, _ = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s = key.size()
        t_t = query.size(2)

        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))

        if self.window_size is not None:
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)

        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )

        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        if pad_length > 0:
            padded = F.pad(relative_embeddings, [0, 0, pad_length, pad_length])
        else:
            padded = relative_embeddings
        return padded[:, slice_start:slice_end]

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, 1])
        x_flat = x.view(batch, heads, length * (2 * length))
        x_flat = F.pad(x_flat, [0, length - 1])
        x_final = x_flat.view(batch, heads, length + 1, 2 * length - 1)[
            :, :, :length, length - 1 :
        ]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, length - 1])
        x_flat = x.view(batch, heads, length ** 2 + length * (length - 1))
        x_flat = F.pad(x_flat, [length, 0])
        x_final = x_flat.view(batch, heads, length, 2 * length)[:, :, :, 1:]
        return x_final


class FFN(nn.Module):
    """Feed-Forward Network with Conv1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):
    """VITS2 Text Encoder.

    Processes text/phoneme IDs through transformer blocks and outputs
    statistics (mean, log-variance) for the variational posterior.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        out_channels: int = 192,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.emb = nn.Embedding(vocab_size, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.encoder = Encoder(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
        )

        # Project to mean and log-variance
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        """
        Args:
            x: [B, T_text] token IDs
            x_lengths: [B] lengths

        Returns:
            x: [B, hidden_channels, T_text] encoder output
            m: [B, out_channels, T_text] mean
            logs: [B, out_channels, T_text] log-variance
            x_mask: [B, 1, T_text] mask
        """
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T]

        x_mask = torch.unsqueeze(
            torch.arange(x.size(2), device=x.device) < x_lengths.unsqueeze(1), 1
        ).float()

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class Encoder(nn.Module):
    """Stack of transformer blocks."""

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_layers = n_layers

        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout)
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        return x * x_mask
