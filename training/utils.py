"""
Training Utilities
Checkpointing, logging, metrics, and learning rate scheduling.
"""

import os
import glob
import torch
import logging
from typing import Optional, Dict


def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Set up a logger with both file and console output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_checkpoint(
    filepath: str,
    model,
    optimizer,
    scheduler=None,
    epoch: int = 0,
    step: int = 0,
    best_loss: float = float("inf"),
    extra: dict = None,
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if extra is not None:
        state.update(extra)

    torch.save(state, filepath)


def load_checkpoint(
    filepath: str,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu",
):
    """Load model checkpoint.

    Returns:
        Dictionary with epoch, step, best_loss
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "best_loss": checkpoint.get("best_loss", float("inf")),
    }


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int = 5):
    """Remove old checkpoints, keeping only the newest `keep_last_n`."""
    pattern = os.path.join(checkpoint_dir, "epoch_*.pt")
    checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime)

    if len(checkpoints) > keep_last_n:
        for ckpt in checkpoints[:-keep_last_n]:
            os.remove(ckpt)


class NoamScheduler:
    """Noam learning rate scheduler (Transformer warmup + inverse sqrt decay).

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 10000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self):
        step = max(1, self._step)
        return (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * self.warmup_steps ** -1.5,
        )

    def state_dict(self):
        return {"step": self._step}

    def load_state_dict(self, state_dict):
        self._step = state_dict["step"]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) between reference and hypothesis.

    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=words in reference
    """
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()

    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        return 0.0 if m == 0 else 1.0

    # Dynamic programming
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,       # insertion
                    d[i - 1][j - 1] + 1,   # substitution
                )

    return d[n][m] / n
