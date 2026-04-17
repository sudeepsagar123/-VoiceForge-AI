"""
VITS2 Posterior Encoder
Encodes raw audio (linear spectrogram) into the latent space during training.
Uses WaveNet-style dilated convolutions.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm

from models.tts.commons import WN


class PosteriorEncoder(nn.Module):
    """Posterior Encoder for VITS2.

    Takes linear spectrogram as input and outputs latent representation z.
    This is only used during training (teacher-forced latent).
    """

    def __init__(
        self,
        in_channels: int = 513,       # n_fft // 2 + 1
        hidden_channels: int = 192,
        out_channels: int = 192,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 16,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        """
        Args:
            x: [B, in_channels, T] linear spectrogram
            x_lengths: [B] lengths
            g: [B, gin_channels, 1] optional speaker embedding

        Returns:
            z: [B, out_channels, T] sampled latent
            m: [B, out_channels, T] mean
            logs: [B, out_channels, T] log-variance
            x_mask: [B, 1, T] mask
        """
        x_mask = torch.unsqueeze(
            torch.arange(x.size(2), device=x.device) < x_lengths.unsqueeze(1), 1
        ).float()

        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        # Reparameterization trick
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()
