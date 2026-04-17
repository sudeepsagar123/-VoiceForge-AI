"""
VITS2 Normalizing Flow
Transforms between simple latent distribution and complex speech distribution.
Uses affine coupling layers with WaveNet internals.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm

from models.tts.commons import WN, Flip, Log


class ResidualCouplingLayer(nn.Module):
    """Single affine coupling layer for the normalizing flow."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Args:
            x: [B, channels, T]
            x_mask: [B, 1, T]
            g: optional conditioning
            reverse: if True, run inverse flow

        Returns:
            x: transformed tensor
            logdet: log determinant of Jacobian
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, dim=1)

        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask

        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, dim=1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], dim=1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], dim=1)
            return x


class ResidualCouplingBlock(nn.Module):
    """Stack of residual coupling layers with flips."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.flows = nn.ModuleList()

        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Forward: z → x (latent → speech distribution)
        Reverse: x → z (speech distribution → latent)
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for flow in self.flows:
            if hasattr(flow, "remove_weight_norm"):
                flow.remove_weight_norm()
