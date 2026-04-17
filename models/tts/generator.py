"""
VITS2 HiFi-GAN Generator
Neural vocoder that converts latent representations directly to raw audio waveforms.
Uses transposed convolutions with Multi-Receptive Field Fusion (MRF).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from models.tts.commons import init_weights, get_padding

LRELU_SLOPE = 0.1


class ResBlock1(nn.Module):
    """Residual block with dilated convolutions (Type 1)."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilation:
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size,
                        dilation=d, padding=get_padding(kernel_size, d),
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size,
                        dilation=1, padding=get_padding(kernel_size, 1),
                    )
                )
            )

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class ResBlock2(nn.Module):
    """Residual block with dilated convolutions (Type 2, simpler)."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList()

        for d in dilation:
            self.convs.append(
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size,
                        dilation=d, padding=get_padding(kernel_size, d),
                    )
                )
            )
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for c in self.convs:
            remove_weight_norm(c)


class Generator(nn.Module):
    """HiFi-GAN Generator for VITS2.

    Converts latent z → raw audio waveform.
    Uses progressive upsampling with Multi-Receptive Field Fusion.
    """

    def __init__(
        self,
        initial_channel: int = 192,
        resblock_type: str = "1",
        resblock_kernel_sizes: list = [3, 7, 11],
        resblock_dilation_sizes: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates: list = [8, 8, 2, 2],
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: list = [16, 16, 4, 4],
        gin_channels: int = 0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Initial convolution
        self.conv_pre = weight_norm(
            nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        )

        # Choose resblock type
        ResBlock = ResBlock1 if resblock_type == "1" else ResBlock2

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2 ** i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        in_ch, out_ch, k,
                        stride=u, padding=(k - u) // 2,
                    )
                )
            )

        # MRF (Multi-Receptive Field Fusion) residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock(ch, k, d))

        # Final convolution
        self.conv_post = weight_norm(
            nn.Conv1d(ch, 1, 7, 1, padding=3)
        )

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        # Optional speaker conditioning
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        """
        Args:
            x: [B, initial_channel, T] latent representation
            g: [B, gin_channels, 1] optional speaker conditioning

        Returns:
            audio: [B, 1, T_audio] waveform
        """
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            # MRF: sum outputs from all resblocks at this level
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
