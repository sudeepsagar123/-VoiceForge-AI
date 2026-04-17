"""
Conformer Convolution Module
Captures local acoustic patterns using depthwise-separable convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stt.commons import Swish


class ConformerConvModule(nn.Module):
    """Conformer Convolution Module.

    Architecture:
        LayerNorm → Pointwise Conv → GLU → Depthwise Conv
        → BatchNorm → Swish → Pointwise Conv → Dropout

    The depthwise convolution captures local patterns (like phonemes),
    while the pointwise convolutions mix features across channels.
    """

    def __init__(
        self,
        d_model: int = 256,
        kernel_size: int = 31,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        padding = (kernel_size - 1) // 2

        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise expansion (1×1 conv)
        self.pointwise_conv1 = nn.Conv1d(
            d_model, 2 * d_model, kernel_size=1, bias=bias
        )

        # Depthwise conv (groups=d_model means each channel is convolved independently)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=padding,
            bias=bias,
        )

        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()

        # Pointwise projection back (1×1 conv)
        self.pointwise_conv2 = nn.Conv1d(
            d_model, d_model, kernel_size=1, bias=bias
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, T, D]

        Returns:
            out: [B, T, D]
        """
        x = self.layer_norm(x)

        # [B, T, D] → [B, D, T] for conv
        x = x.transpose(1, 2)

        # Pointwise conv + GLU
        x = self.pointwise_conv1(x)  # [B, 2D, T]
        x = F.glu(x, dim=1)          # [B, D, T]

        # Depthwise conv
        x = self.depthwise_conv(x)   # [B, D, T]
        x = self.batch_norm(x)
        x = self.swish(x)

        # Pointwise projection
        x = self.pointwise_conv2(x)  # [B, D, T]
        x = self.dropout(x)

        # [B, D, T] → [B, T, D]
        return x.transpose(1, 2)
