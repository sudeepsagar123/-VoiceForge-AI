"""
VITS2 Common Utilities
Shared functions and layers used across the VITS2 model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm


def init_weights(m, mean=0.0, std=0.01):
    """Initialize weights for Conv layers."""
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    """Calculate padding for 'same' convolution."""
    return int((kernel_size * dilation - dilation) / 2)


def sequence_mask(length, max_length=None):
    """Create boolean mask from sequence lengths.

    Args:
        length: [B] tensor of lengths
        max_length: scalar, default max(length)

    Returns:
        mask: [B, max_length] boolean tensor
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    Generate alignment path from durations.

    Args:
        duration: [B, 1, T_text] tensor of durations (integer)
        mask: [B, 1, T_mel, T_text] tensor

    Returns:
        path: [B, 1, T_mel, T_text] one-hot alignment path
    """
    b, _, t_mel, t_text = mask.shape
    cum_duration = torch.cumsum(duration.squeeze(1), dim=1)  # [B, T_text]

    path = torch.zeros(b, t_mel, t_text, dtype=mask.dtype, device=mask.device)
    cum_duration_flat = cum_duration.reshape(b * t_text)

    # Create proper path
    for i in range(b):
        start = 0
        for j in range(t_text):
            end = int(cum_duration[i, j].item())
            if end > start and end <= t_mel:
                path[i, start:end, j] = 1.0
            start = end

    return path.unsqueeze(1)  # [B, 1, T_mel, T_text]


class LayerNorm(nn.Module):
    """Channel-wise Layer Normalization for 1D sequences."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        """
        Args:
            x: [B, C, T] tensor
        """
        x = x.transpose(1, -1)  # [B, T, C]
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)  # [B, C, T]


class WN(nn.Module):
    """WaveNet-style dilated convolution stack.

    Used in the posterior encoder and normalizing flow.
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        p_dropout: float = 0.0,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels > 0:
            self.cond_layer = weight_norm(
                nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            )

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = get_padding(kernel_size, dilation)
            self.in_layers.append(
                weight_norm(
                    nn.Conv1d(
                        hidden_channels,
                        2 * hidden_channels,
                        kernel_size,
                        dilation=dilation,
                        padding=padding,
                    )
                )
            )

            # Last layer outputs only residual (no skip)
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            self.res_skip_layers.append(
                weight_norm(nn.Conv1d(hidden_channels, res_skip_channels, 1))
            )

    def forward(self, x, x_mask, g=None):
        """
        Args:
            x: [B, hidden_channels, T]
            x_mask: [B, 1, T]
            g: [B, gin_channels, 1] optional conditioning
        """
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
                x_in = x_in + g_l

            # Gated activation: tanh(a) * sigmoid(b)
            acts = torch.tanh(x_in[:, : self.hidden_channels, :]) * torch.sigmoid(
                x_in[:, self.hidden_channels :, :]
            )
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts

        return output * x_mask

    def remove_weight_norm(self):
        for layer in self.in_layers:
            remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            remove_weight_norm(layer)
        if hasattr(self, "cond_layer"):
            remove_weight_norm(self.cond_layer)


class Log(nn.Module):
    def forward(self, x, x_mask, reverse=False):
        if not reverse:
            y = torch.log(torch.clamp(x, min=1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            return x, logdet
        else:
            return x
