"""
VITS2 — Full Model
Variational Inference with Adversarial Learning for End-to-End Text-to-Speech (v2)

Combines:
  - Text Encoder (Transformer)
  - Posterior Encoder (WaveNet)
  - Normalizing Flow
  - Stochastic Duration Predictor
  - HiFi-GAN Generator
  - Monotonic Alignment Search (MAS)

References:
  - VITS: https://arxiv.org/abs/2106.06103
  - VITS2: https://arxiv.org/abs/2307.16430
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.tts.text_encoder import TextEncoder
from models.tts.posterior_encoder import PosteriorEncoder
from models.tts.flow import ResidualCouplingBlock
from models.tts.duration_predictor import StochasticDurationPredictor, DurationPredictor
from models.tts.generator import Generator
from models.tts.commons import sequence_mask, generate_path


class VITS2(nn.Module):
    """
    VITS2 End-to-End TTS Model.

    Training:   text + audio → losses (mel, kl, duration, adv, fm)
    Inference:  text → audio waveform
    """

    def __init__(
        self,
        # Text encoder
        vocab_size: int = 256,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_enc_layers: int = 6,
        enc_kernel_size: int = 3,
        enc_dropout: float = 0.1,
        # Posterior encoder
        spec_channels: int = 513,
        post_kernel_size: int = 5,
        post_dilation: int = 1,
        post_n_layers: int = 16,
        # Flow
        flow_hidden: int = 192,
        flow_kernel_size: int = 5,
        flow_dilation: int = 1,
        flow_n_layers: int = 4,
        flow_n_flows: int = 4,
        # Duration predictor
        dp_hidden: int = 192,
        dp_kernel_size: int = 3,
        dp_dropout: float = 0.5,
        dp_n_flows: int = 4,
        # HiFi-GAN Generator
        upsample_rates: list = None,
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: list = None,
        resblock_type: str = "1",
        resblock_kernel_sizes: list = None,
        resblock_dilation_sizes: list = None,
        # Audio
        segment_size: int = 8192,
        hop_length: int = 256,
        # Speaker conditioning
        n_speakers: int = 0,
        gin_channels: int = 0,
        # Training flags
        use_sdp: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.n_speakers = n_speakers
        self.use_sdp = use_sdp

        # Defaults
        upsample_rates = upsample_rates or [8, 8, 2, 2]
        upsample_kernel_sizes = upsample_kernel_sizes or [16, 16, 4, 4]
        resblock_kernel_sizes = resblock_kernel_sizes or [3, 7, 11]
        resblock_dilation_sizes = resblock_dilation_sizes or [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        # ── Sub-modules ──

        self.enc_p = TextEncoder(
            vocab_size=vocab_size,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_enc_layers,
            kernel_size=enc_kernel_size,
            p_dropout=enc_dropout,
            out_channels=hidden_channels,
        )

        self.enc_q = PosteriorEncoder(
            in_channels=spec_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=post_kernel_size,
            dilation_rate=post_dilation,
            n_layers=post_n_layers,
            gin_channels=gin_channels,
        )

        self.flow = ResidualCouplingBlock(
            channels=hidden_channels,
            hidden_channels=flow_hidden,
            kernel_size=flow_kernel_size,
            dilation_rate=flow_dilation,
            n_layers=flow_n_layers,
            n_flows=flow_n_flows,
            gin_channels=gin_channels,
        )

        if use_sdp:
            self.dp = StochasticDurationPredictor(
                in_channels=hidden_channels,
                filter_channels=dp_hidden,
                kernel_size=dp_kernel_size,
                p_dropout=dp_dropout,
                n_flows=dp_n_flows,
                gin_channels=gin_channels,
            )
        else:
            self.dp = DurationPredictor(
                in_channels=hidden_channels,
                filter_channels=256,
                kernel_size=dp_kernel_size,
                p_dropout=dp_dropout,
            )

        self.dec = Generator(
            initial_channel=hidden_channels,
            resblock_type=resblock_type,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

        # Speaker embedding
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        """
        Training forward pass.

        Args:
            x: [B, T_text] text token IDs
            x_lengths: [B] text lengths
            y: [B, spec_channels, T_spec] linear spectrogram
            y_lengths: [B] spectrogram lengths
            sid: [B] speaker IDs (optional)

        Returns:
            Dictionary with all outputs needed for loss computation
        """
        # Speaker conditioning
        g = None
        if self.n_speakers > 1:
            g = self.emb_g(sid).unsqueeze(-1)  # [B, gin, 1]

        # ── Text Encoder ──
        x_enc, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # ── Posterior Encoder (from audio) ──
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)

        # ── Normalizing Flow (posterior → prior) ──
        z_p = self.flow(z, y_mask, g=g)

        # ── Monotonic Alignment Search ──
        with torch.no_grad():
            # Compute alignment between text and audio
            s_p_sq_r = torch.exp(-2 * logs_p)  # [B, C, T_text]
            # neg_cent = -(z_p^2 * s_p_sq_r - 2 * z_p * m_p * s_p_sq_r + m_p^2 * s_p_sq_r)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, dim=1).unsqueeze(1)  # [B, 1, T_text]
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)  # [B, T_mel, T_text]
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))  # [B, T_mel, T_text]
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, dim=1).unsqueeze(1)  # [B, 1, T_text]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            # Apply masks
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_alignment_search(
                neg_cent.unsqueeze(1), attn_mask
            ).detach()

        # Duration = sum of alignment along mel axis
        w = attn.sum(2)  # [B, 1, T_text]

        # ── Duration Predictor Loss ──
        if self.use_sdp:
            l_length = self.dp(x_enc, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x_enc, x_mask)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)

        # ── Expand text features using alignment ──
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # ── Random segment for decoder ──
        z_slice, ids_slice = self._rand_slice_segments(
            z, y_lengths, self.segment_size // self.hop_length
        )

        # ── HiFi-GAN Decoder ──
        o = self.dec(z_slice, g=g)

        return {
            "output": o,
            "l_length": l_length,
            "attn": attn,
            "ids_slice": ids_slice,
            "x_mask": x_mask,
            "y_mask": y_mask,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "m_q": m_q,
            "logs_q": logs_q,
        }

    @torch.no_grad()
    def infer(self, x, x_lengths, sid=None, noise_scale=0.667, length_scale=1.0, noise_scale_w=0.8):
        """
        Inference: text → audio waveform.

        Args:
            x: [B, T_text] text token IDs
            x_lengths: [B] text lengths
            sid: [B] speaker IDs (optional)
            noise_scale: controls variance of latent z
            length_scale: controls speaking speed (< 1 = faster, > 1 = slower)
            noise_scale_w: controls variance of duration prediction

        Returns:
            o: [B, 1, T_audio] generated waveform
        """
        g = None
        if self.n_speakers > 1:
            g = self.emb_g(sid).unsqueeze(-1)

        # Text encoding
        x_enc, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # Duration prediction
        if self.use_sdp:
            logw = self.dp(x_enc, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x_enc, x_mask)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(
            sequence_mask(y_lengths, None), 1
        ).float()

        # Build alignment from predicted durations
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        # Expand text features
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # Sample latent z from prior
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # Inverse flow: prior → posterior
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        # Decode to waveform
        o = self.dec(z * y_mask, g=g)
        return o, attn, y_mask

    def _rand_slice_segments(self, x, x_lengths, segment_size):
        """Randomly slice a segment from each batch item."""
        b, d, t = x.size()
        ids_str_max = x_lengths - segment_size + 1
        ids_str_max = ids_str_max.clamp(min=0)
        ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)

        ret = torch.zeros(b, d, segment_size, device=x.device, dtype=x.dtype)
        for i in range(b):
            idx_start = ids_str[i]
            idx_end = idx_start + segment_size
            end = min(idx_end, t)
            length = end - idx_start
            ret[i, :, :length] = x[i, :, idx_start:end]

        return ret, ids_str

    def remove_weight_norm(self):
        """Remove weight normalization for faster inference."""
        self.enc_q.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.dec.remove_weight_norm()


def monotonic_alignment_search(neg_cent, attn_mask):
    """
    Monotonic Alignment Search (MAS) — Viterbi-style dynamic programming.

    Finds the most likely monotonic alignment between text and mel frames.

    Args:
        neg_cent: [B, 1, T_mel, T_text] log-probability of alignment
        attn_mask: [B, 1, T_mel, T_text] mask

    Returns:
        path: [B, 1, T_mel, T_text] optimal monotonic alignment
    """
    # Use numpy for DP (GPU tensors → CPU for this operation)
    neg_cent = neg_cent.squeeze(1).cpu().numpy()
    attn_mask = attn_mask.squeeze(1).cpu().numpy()
    device = torch.device("cpu")  # Will move back to original device

    b, t_mel, t_text = neg_cent.shape
    path = np.zeros_like(neg_cent)

    for batch in range(b):
        # Get actual lengths from mask
        mel_len = int(attn_mask[batch, :, 0].sum())
        text_len = int(attn_mask[batch, 0, :].sum())

        if mel_len == 0 or text_len == 0:
            continue

        # DP table
        Q = np.full((mel_len, text_len), -1e9)
        Q[0, 0] = neg_cent[batch, 0, 0]

        for i in range(1, mel_len):
            for j in range(min(i + 1, text_len)):
                if j == 0:
                    Q[i, j] = Q[i - 1, j] + neg_cent[batch, i, j]
                else:
                    Q[i, j] = max(Q[i - 1, j], Q[i - 1, j - 1]) + neg_cent[batch, i, j]

        # Backtrack
        j = text_len - 1
        for i in range(mel_len - 1, -1, -1):
            path[batch, i, j] = 1
            if j > 0 and (i == 0 or Q[i - 1, j - 1] >= Q[i - 1, j]):
                j -= 1

    return torch.from_numpy(path).unsqueeze(1).float()


# ── Model Test ──

if __name__ == "__main__":
    print("=" * 60)
    print("VITS2 Model — Architecture Test")
    print("=" * 60)

    model = VITS2(
        vocab_size=100,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_enc_layers=6,
        spec_channels=513,
        upsample_rates=[8, 8, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4],
        segment_size=8192,
        hop_length=256,
        use_sdp=False,  # Use deterministic DP for test
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size:           ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")

    # Test inference
    batch_size = 2
    text_len = 50
    x = torch.randint(0, 100, (batch_size, text_len))
    x_lengths = torch.LongTensor([text_len, text_len - 10])

    print("\n--- Inference Test ---")
    model.eval()
    with torch.no_grad():
        audio, attn, mask = model.infer(x, x_lengths)
    print(f"Input text shape:  {x.shape}")
    print(f"Output audio shape: {audio.shape}")
    print(f"Attention shape:    {attn.shape}")
    print(f"Audio duration:     ~{audio.shape[2] / 22050:.2f} seconds")

    print("\n✅ VITS2 model architecture test passed!")
