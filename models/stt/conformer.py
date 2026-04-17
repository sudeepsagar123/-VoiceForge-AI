"""
Conformer-CTC Full Model
Complete speech-to-text model with Conformer encoder and CTC decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stt.subsampling import Conv2dSubsampling
from models.stt.encoder import ConformerEncoder
from models.stt.commons import make_pad_mask


class ConformerCTC(nn.Module):
    """
    Conformer-CTC Speech-to-Text Model.

    Architecture:
        Audio Features → Conv2d Subsampling (4×) → Conformer Encoder (12 blocks) → CTC Decoder

    Uses CTC (Connectionist Temporal Classification) loss which:
    - Does NOT require pre-aligned labels
    - Handles variable-length input/output naturally
    - Enables streaming-compatible inference
    """

    def __init__(
        self,
        input_dim: int = 83,
        d_model: int = 256,
        d_ff: int = 1024,
        num_heads: int = 4,
        num_layers: int = 12,
        conv_kernel_size: int = 31,
        vocab_size: int = 5000,
        dropout: float = 0.1,
        blank_id: int = 0,
    ):
        super().__init__()
        self.blank_id = blank_id
        self.vocab_size = vocab_size

        # ── Subsampling Frontend ──
        self.subsampling = Conv2dSubsampling(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout,
        )

        # ── Conformer Encoder ──
        self.encoder = ConformerEncoder(
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            num_layers=num_layers,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        # ── CTC Decoder (simple linear projection) ──
        self.ctc_proj = nn.Linear(d_model, vocab_size)

        # ── SpecAugment (data augmentation during training) ──
        self.spec_augment = SpecAugment()

    def forward(self, features, feat_lengths, apply_augment=True):
        """
        Forward pass.

        Args:
            features: [B, T, D] audio features (log mel + pitch)
            feat_lengths: [B] actual frame counts
            apply_augment: whether to apply SpecAugment (training only)

        Returns:
            log_probs: [T/4, B, vocab_size] CTC log probabilities
            out_lengths: [B] output lengths after subsampling
        """
        # SpecAugment (training only)
        if self.training and apply_augment:
            features = self.spec_augment(features)

        # Subsampling (4× temporal reduction)
        x, out_lengths = self.subsampling(features, feat_lengths)

        # Conformer encoder
        x, out_lengths = self.encoder(x, out_lengths)

        # CTC projection
        logits = self.ctc_proj(x)  # [B, T/4, vocab_size]

        # CTC expects [T, B, C] format
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

        return log_probs, out_lengths

    @torch.no_grad()
    def decode(self, features, feat_lengths):
        """
        Greedy CTC decoding for inference.

        Args:
            features: [B, T, D] audio features
            feat_lengths: [B] actual lengths

        Returns:
            decoded: list of token ID lists (one per batch item)
        """
        log_probs, out_lengths = self.forward(features, feat_lengths, apply_augment=False)

        # log_probs: [T, B, V] → argmax per timestep
        predictions = torch.argmax(log_probs, dim=-1)  # [T, B]
        predictions = predictions.transpose(0, 1)  # [B, T]

        decoded = []
        for i in range(predictions.shape[0]):
            seq = predictions[i, : out_lengths[i]].tolist()
            # CTC collapse: remove blanks and merge repeated tokens
            collapsed = []
            prev = self.blank_id
            for token in seq:
                if token != self.blank_id and token != prev:
                    collapsed.append(token)
                prev = token
            decoded.append(collapsed)

        return decoded

    @torch.no_grad()
    def beam_decode(self, features, feat_lengths, beam_width=10):
        """
        CTC Beam Search decoding.

        More accurate than greedy but slower. Uses prefix beam search
        to find the most likely sequence.
        """
        log_probs, out_lengths = self.forward(features, feat_lengths, apply_augment=False)
        log_probs = log_probs.transpose(0, 1)  # [B, T, V]

        results = []
        for i in range(log_probs.shape[0]):
            seq_log_probs = log_probs[i, : out_lengths[i]]  # [T, V]

            # Simple prefix beam search
            beams = [([], 0.0)]  # (prefix, log_prob)

            for t in range(seq_log_probs.shape[0]):
                new_beams = {}
                top_k = torch.topk(seq_log_probs[t], min(beam_width, self.vocab_size))

                for prefix, score in beams:
                    for j in range(top_k.values.shape[0]):
                        token = top_k.indices[j].item()
                        token_score = top_k.values[j].item()

                        if token == self.blank_id:
                            key = tuple(prefix)
                        elif prefix and token == prefix[-1]:
                            key = tuple(prefix)
                        else:
                            key = tuple(prefix + [token])

                        new_score = score + token_score
                        if key not in new_beams or new_beams[key] < new_score:
                            new_beams[key] = new_score

                # Keep top-k beams
                beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)
                beams = [(list(k), v) for k, v in beams[:beam_width]]

            results.append(beams[0][0] if beams else [])

        return results


class SpecAugment(nn.Module):
    """SpecAugment data augmentation for speech recognition.

    Applies:
    1. Frequency masking — masks random frequency bands
    2. Time masking — masks random time steps

    This dramatically improves ASR robustness without any external data.
    """

    def __init__(
        self,
        freq_mask_count: int = 2,
        freq_mask_width: int = 27,
        time_mask_count: int = 10,
        time_mask_ratio: float = 0.05,
    ):
        super().__init__()
        self.freq_mask_count = freq_mask_count
        self.freq_mask_width = freq_mask_width
        self.time_mask_count = time_mask_count
        self.time_mask_ratio = time_mask_ratio

    def forward(self, x):
        """
        Args:
            x: [B, T, F] audio features

        Returns:
            x_augmented: [B, T, F] augmented features
        """
        x = x.clone()  # Don't modify the original
        batch_size, time_steps, freq_dim = x.shape

        # Frequency masking
        for _ in range(self.freq_mask_count):
            f = torch.randint(0, self.freq_mask_width, (1,)).item()
            f0 = torch.randint(0, max(1, freq_dim - f), (1,)).item()
            x[:, :, f0 : f0 + f] = 0.0

        # Time masking
        max_time_mask = int(time_steps * self.time_mask_ratio)
        for _ in range(self.time_mask_count):
            t = torch.randint(0, max(1, max_time_mask), (1,)).item()
            t0 = torch.randint(0, max(1, time_steps - t), (1,)).item()
            x[:, t0 : t0 + t, :] = 0.0

        return x


# ── Model Test ──

if __name__ == "__main__":
    print("=" * 60)
    print("Conformer-CTC Model — Architecture Test")
    print("=" * 60)

    model = ConformerCTC(
        input_dim=83,
        d_model=256,
        d_ff=1024,
        num_heads=4,
        num_layers=12,
        conv_kernel_size=31,
        vocab_size=5000,
        dropout=0.1,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size:           ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")

    # Test forward pass
    batch_size = 2
    time_steps = 400  # ~4 seconds of audio
    feature_dim = 83

    features = torch.randn(batch_size, time_steps, feature_dim)
    feat_lengths = torch.LongTensor([400, 350])

    print("\n--- Forward Pass Test ---")
    model.eval()
    with torch.no_grad():
        log_probs, out_lengths = model(features, feat_lengths, apply_augment=False)

    print(f"Input shape:   {features.shape} (batch={batch_size}, T={time_steps}, D={feature_dim})")
    print(f"Output shape:  {log_probs.shape} (T/4, batch, vocab)")
    print(f"Out lengths:   {out_lengths.tolist()}")

    # Test greedy decode
    print("\n--- Greedy Decode Test ---")
    decoded = model.decode(features, feat_lengths)
    for i, tokens in enumerate(decoded):
        print(f"  Sample {i}: {len(tokens)} tokens decoded")

    # Test beam decode
    print("\n--- Beam Decode Test ---")
    beam_decoded = model.beam_decode(features, feat_lengths, beam_width=5)
    for i, tokens in enumerate(beam_decoded):
        print(f"  Sample {i}: {len(tokens)} tokens decoded")

    print("\n✅ Conformer-CTC model architecture test passed!")
