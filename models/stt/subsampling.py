"""
Conformer Subsampling Frontend
Reduces the temporal resolution of audio features using strided convolutions.
"""

import torch
import torch.nn as nn

from models.stt.commons import PositionalEncoding


class Conv2dSubsampling(nn.Module):
    """Convolutional subsampling frontend for Conformer.

    Two Conv2d layers with stride 2 → 4× temporal downsampling.
    This reduces sequence length significantly, making the
    transformer attention computationally feasible for long audio.

    Input:  [B, T, feature_dim]  (e.g., T=1600 frames for 16 sec audio)
    Output: [B, T/4, d_model]    (e.g., T/4=400 frames)
    """

    def __init__(
        self,
        input_dim: int = 83,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calculate the output feature dimension after Conv2d subsampling
        # After two stride-2 convolutions: feature_dim → ceil(feature_dim/2) → ceil(that/2)
        conv_out_dim = input_dim
        for _ in range(2):
            conv_out_dim = (conv_out_dim + 2 * 1 - 3) // 2 + 1  # (dim + 2*pad - kernel) // stride + 1

        self.linear = nn.Linear(d_model * conv_out_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

    def forward(self, x, x_lengths):
        """
        Args:
            x: [B, T, feature_dim] audio features
            x_lengths: [B] actual lengths

        Returns:
            out: [B, T/4, d_model] subsampled features
            out_lengths: [B] new lengths after subsampling
        """
        # Add channel dimension: [B, T, F] → [B, 1, T, F]
        x = x.unsqueeze(1)

        # Apply conv subsampling
        x = self.conv(x)  # [B, d_model, T/4, F']

        # Reshape: [B, d_model, T/4, F'] → [B, T/4, d_model * F']
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)

        # Linear projection: [B, T/4, d_model * F'] → [B, T/4, d_model]
        x = self.linear(x)

        # Add positional encoding
        x = self.pos_enc(x)

        # Update lengths (4× downsampling)
        out_lengths = ((x_lengths - 1) // 2 - 1) // 2 + 1
        out_lengths = out_lengths.clamp(min=1)

        return x, out_lengths
