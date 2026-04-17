"""
Conformer Attention Module
Multi-Head Self-Attention with relative positional encoding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativeMultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with relative positional encoding.

    This is the attention mechanism used inside each Conformer block.
    Relative positions help the model understand the distance between
    audio frames without absolute position dependence.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_pos = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model)

        # Learnable biases for content and position
        self.u_bias = nn.Parameter(torch.zeros(num_heads, self.d_k))
        self.v_bias = nn.Parameter(torch.zeros(num_heads, self.d_k))

        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_pos.weight)
        nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, x, pos_enc, mask=None):
        """
        Args:
            x: [B, T, D] input features
            pos_enc: [1, T, D] relative positional encoding
            mask: [B, 1, T] padding mask (True = padded)

        Returns:
            out: [B, T, D] attended features
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        pos = self.W_pos(pos_enc).view(1, seq_len, self.num_heads, self.d_k)

        # Transpose for attention: [B, heads, T, d_k]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        pos = pos.permute(0, 2, 1, 3)  # [1, heads, T, d_k]

        # Content-based attention
        content_score = torch.matmul(
            (q + self.u_bias.unsqueeze(0).unsqueeze(2)),
            k.transpose(-2, -1)
        )

        # Position-based attention
        pos_score = torch.matmul(
            (q + self.v_bias.unsqueeze(0).unsqueeze(2)),
            pos.transpose(-2, -1)
        )
        pos_score = self._relative_shift(pos_score)

        # Combined scores
        scores = (content_score + pos_score) / math.sqrt(self.d_k)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attend to values
        context = torch.matmul(attn_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_out(context)

    def _relative_shift(self, pos_score):
        """Perform relative shift for position scores.

        Converts absolute position scores to proper relative position scores.
        """
        batch_size, num_heads, seq_len1, seq_len2 = pos_score.shape
        zeros = torch.zeros(
            (batch_size, num_heads, seq_len1, 1),
            device=pos_score.device, dtype=pos_score.dtype
        )
        padded = torch.cat([zeros, pos_score], dim=-1)
        padded = padded.view(batch_size, num_heads, seq_len2 + 1, seq_len1)
        pos_score = padded[:, :, 1:, :].view_as(pos_score)
        return pos_score
