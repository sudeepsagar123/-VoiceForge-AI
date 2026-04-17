"""
Training Loss Functions for TTS (VITS2) and STT (Conformer-CTC)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════
#  TTS Losses (VITS2)
# ═══════════════════════════════════════════

def generator_loss(disc_outputs):
    """Adversarial generator loss (LSGAN style).

    The generator wants the discriminator to think generated audio is real.
    """
    loss = 0
    for dg in disc_outputs:
        dg = dg.float()
        loss += torch.mean((1 - dg) ** 2)
    return loss


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """Adversarial discriminator loss (LSGAN style).

    The discriminator wants to correctly classify real vs generated.
    """
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
    return loss


def feature_matching_loss(fmap_real, fmap_generated):
    """Feature matching loss.

    Penalizes differences between intermediate discriminator features
    of real and generated audio. This stabilizes GAN training.
    """
    loss = 0
    for dr, dg in zip(fmap_real, fmap_generated):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """KL divergence loss between posterior and prior.

    Keeps the text encoder's predicted distribution close to the
    posterior encoder's distribution learned from real audio.
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    loss = kl / torch.sum(z_mask)
    return loss


def mel_loss(y_mel, y_hat_mel):
    """L1 mel spectrogram reconstruction loss."""
    return F.l1_loss(y_mel, y_hat_mel)


# ═══════════════════════════════════════════
#  STT Losses (Conformer-CTC)
# ═══════════════════════════════════════════

class CTCLoss(nn.Module):
    """CTC Loss for speech recognition.

    Connectionist Temporal Classification enables training ASR models
    without requiring pre-aligned audio-text pairs.
    """

    def __init__(self, blank_id: int = 0, reduction: str = "mean"):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction=reduction, zero_infinity=True)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Args:
            log_probs: [T, B, vocab_size] CTC log probabilities
            targets: [B, S] target token IDs
            input_lengths: [B] input sequence lengths (after subsampling)
            target_lengths: [B] target sequence lengths

        Returns:
            loss: scalar CTC loss
        """
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
