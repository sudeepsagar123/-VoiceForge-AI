"""
VITS2 Duration Predictor
Predicts per-phoneme durations using a stochastic approach.
Improved in VITS2 with adversarial training for more natural rhythm.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tts.commons import LayerNorm


class DurationPredictor(nn.Module):
    """Deterministic duration predictor with conv stack.

    Used as a simpler alternative to the stochastic version.
    """

    def __init__(
        self,
        in_channels: int = 192,
        filter_channels: int = 256,
        kernel_size: int = 3,
        p_dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels

        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        """
        Args:
            x: [B, in_channels, T_text]
            x_mask: [B, 1, T_text]

        Returns:
            log_duration: [B, 1, T_text]
        """
        x = self.conv_1(x * x_mask)
        x = F.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)

        x = self.conv_2(x * x_mask)
        x = F.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        x = self.proj(x * x_mask)
        return x * x_mask


class StochasticDurationPredictor(nn.Module):
    """VITS2-style Stochastic Duration Predictor.

    Uses flow-based modeling for more natural, varied duration predictions.
    Includes adversarial training (VITS2 improvement).
    """

    def __init__(
        self,
        in_channels: int = 192,
        filter_channels: int = 192,
        kernel_size: int = 3,
        p_dropout: float = 0.5,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.n_flows = n_flows

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)

        # Flow layers
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(ConvFlow(filter_channels, filter_channels, kernel_size, n_layers=3))
            self.flows.append(ElementwiseAffine(2))

        # Posterior (training) encoder
        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        for _ in range(4):
            self.post_flows.append(ConvFlow(filter_channels, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(ElementwiseAffine(2))

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        """
        Args:
            x: [B, in_channels, T_text] text encoder output
            x_mask: [B, 1, T_text]
            w: [B, 1, T_text] ground truth log-durations (training only)
            g: conditioning (speaker embedding)
            reverse: if True, sample durations (inference)

        Returns:
            If not reverse: log-likelihood of actual durations
            If reverse: predicted log-durations
        """
        x = x.detach()
        x = self.pre(x)
        if g is not None:
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            # Training: compute log-likelihood
            assert w is not None
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask

            e_q = torch.randn(w.size(0), 2, w.size(2), device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q

            logdet_tot_q = 0
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=x + h_w)
                logdet_tot_q += logdet_q

            z_u, z1 = torch.split(z_q, [1, 1], dim=1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask

            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )

            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q ** 2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0_cat = torch.cat([z0, z1], dim=1)
            for flow in self.flows:
                z0_cat, logdet = flow(z0_cat, x_mask, g=x, reverse=reverse)
                logdet_tot += logdet

            z_final = z0_cat
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + z_final ** 2) * x_mask, [1, 2])
                - logdet_tot
            )

            return nll + logq

        else:
            # Inference: sample durations
            flows = list(reversed(self.flows))
            z = torch.randn(x.size(0), 2, x.size(2), device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)

            z0, z1 = torch.split(z, [1, 1], dim=1)
            logw = z0
            return logw


class DDSConv(nn.Module):
    """Dilated and Depth-Separable Convolution."""

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        super().__init__()
        self.n_layers = n_layers
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        for i in range(n_layers):
            dilation = kernel_size ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(channels, channels, kernel_size, groups=channels,
                          dilation=dilation, padding=padding)
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class ConvFlow(nn.Module):
    """Convolutional coupling layer for the duration flow."""

    def __init__(self, in_channels, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0):
        super().__init__()
        self.half_channels = in_channels // 2
        self.num_bins = num_bins
        self.tail_bound = tail_bound

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, dim=1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, c, t, bins*3-1]

        unnorm_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels) if hasattr(self, 'filter_channels') else h[..., : self.num_bins]
        unnorm_heights = h[..., self.num_bins : 2 * self.num_bins]
        unnorm_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1, unnorm_widths, unnorm_heights, unnorm_derivatives,
            inverse=reverse, tails="linear", tail_bound=self.tail_bound,
        )

        x = torch.cat([x0, x1], dim=1) * x_mask
        logdet = torch.sum(logabsdet * x_mask.squeeze(1).unsqueeze(-1), [1, 2])

        if not reverse:
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Module):
    """Element-wise affine transform (learnable scale + shift per channel)."""

    def __init__(self, channels):
        super().__init__()
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


def piecewise_rational_quadratic_transform(
    inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
    inverse=False, tails=None, tail_bound=1.0, min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3,
):
    """Rational quadratic spline transform.

    Simplified implementation — uses linear tails outside [-tail_bound, tail_bound].
    """
    if tails == "linear":
        inside = (inputs >= -tail_bound) & (inputs <= tail_bound)

        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs)

        # Apply identity outside the bounds
        outputs[~inside] = inputs[~inside]
        logabsdet[~inside] = 0

        # Apply spline inside
        if inside.any():
            outputs[inside], logabsdet[inside] = rational_quadratic_spline(
                inputs[inside],
                unnormalized_widths[inside],
                unnormalized_heights[inside],
                unnormalized_derivatives[inside],
                inverse=inverse,
                left=-tail_bound, right=tail_bound,
                bottom=-tail_bound, top=tail_bound,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
            )
        return outputs, logabsdet
    else:
        return rational_quadratic_spline(
            inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=min_bin_width, min_bin_height=min_bin_height, min_derivative=min_derivative,
        )


def rational_quadratic_spline(
    inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
    inverse=False, left=0.0, right=1.0, bottom=0.0, top=1.0,
    min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3,
):
    """Core rational quadratic spline computation."""
    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    if inverse:
        bin_idx = (cumheights[..., :-1] <= inputs[..., None]).sum(dim=-1) - 1
    else:
        bin_idx = (cumwidths[..., :-1] <= inputs[..., None]).sum(dim=-1) - 1

    bin_idx = bin_idx.clamp(0, num_bins - 1)

    input_cumwidths = cumwidths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_bin_widths = widths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_cumheights = cumheights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_heights = heights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_delta = input_heights / input_bin_widths

    input_derivatives = derivatives[..., :-1].gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        discriminant = discriminant.clamp(min=0)

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet
