"""
Conformer Encoder
Stack of Conformer blocks combining self-attention and convolution.
"""

import torch
import torch.nn as nn

from models.stt.attention import RelativeMultiHeadAttention
from models.stt.convolution import ConformerConvModule
from models.stt.commons import Swish, RelativePositionalEncoding


class FeedForwardModule(nn.Module):
    """Conformer Feed-Forward Module.

    Architecture: LayerNorm → Linear → Swish → Dropout → Linear → Dropout
    Used as the first and last sub-layer in each Conformer block (with ×0.5 residual).
    """

    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, T, D]
        Returns:
            out: [B, T, D]
        """
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class ConformerBlock(nn.Module):
    """Single Conformer Block.

    Architecture (Macaron-style):
        ½ × FFN → MHSA → ConvModule → ½ × FFN → LayerNorm

    This sandwich structure captures both global (attention) and
    local (convolution) patterns in speech, which is why Conformer
    outperforms pure Transformer for ASR.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_ff: int = 1024,
        num_heads: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Two feed-forward modules (Macaron-style, each ×0.5)
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)

        # Multi-Head Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = RelativeMultiHeadAttention(d_model, num_heads, dropout)
        self.self_attn_dropout = nn.Dropout(dropout)

        # Convolution Module
        self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout)

        # Final LayerNorm
        self.final_layer_norm = nn.LayerNorm(d_model)

        self.ff_scale = 0.5  # Macaron connection scale

    def forward(self, x, pos_enc, mask=None):
        """
        Args:
            x: [B, T, D] input features
            pos_enc: [1, T, D] relative positional encoding
            mask: [B, 1, T] padding mask (True = padded)

        Returns:
            out: [B, T, D]
        """
        # ½ × FFN (first)
        residual = x
        x = residual + self.ff_scale * self.ff1(x)

        # Multi-Head Self-Attention
        residual = x
        x_norm = self.self_attn_layer_norm(x)
        x_attn = self.self_attn(x_norm, pos_enc, mask=mask)
        x = residual + self.self_attn_dropout(x_attn)

        # Convolution Module
        residual = x
        x = residual + self.conv_module(x)

        # ½ × FFN (second)
        residual = x
        x = residual + self.ff_scale * self.ff2(x)

        # Final LayerNorm
        x = self.final_layer_norm(x)

        return x


class ConformerEncoder(nn.Module):
    """Stack of Conformer blocks forming the complete encoder.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension (typically 4× d_model)
        num_heads: Number of attention heads
        num_layers: Number of Conformer blocks
        conv_kernel_size: Kernel size for convolution module
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 256,
        d_ff: int = 1024,
        num_heads: int = 4,
        num_layers: int = 12,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.pos_enc = RelativePositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, x_lengths):
        """
        Args:
            x: [B, T, D] input features (from subsampling)
            x_lengths: [B] actual lengths

        Returns:
            out: [B, T, D] encoded features
            out_lengths: [B] lengths (unchanged)
        """
        # Create padding mask: True for padded positions
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= x_lengths.unsqueeze(1)
        mask = mask.unsqueeze(1)  # [B, 1, T]

        # Get positional encoding
        pos_enc = self.pos_enc(max_len)  # [1, T, D]

        for layer in self.layers:
            x = layer(x, pos_enc, mask=mask)

        return x, x_lengths
