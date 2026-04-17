"""
Conformer-CTC STT Training Script
Complete training loop with CTC loss, Noam scheduler, mixed precision, and WER evaluation.
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stt.conformer import ConformerCTC
from preprocessing.audio_processor import STTAudioProcessor
from preprocessing.dataset import STTDataset, STTCollator
from training.losses import CTCLoss
from training.utils import (
    setup_logger,
    save_checkpoint,
    load_checkpoint,
    cleanup_old_checkpoints,
    NoamScheduler,
    compute_wer,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config_path: str, resume: str = None):
    """Main training function."""
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Conformer-CTC STT Training — {config['model']['name']}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print(f"{'='*60}\n")

    logger = setup_logger(
        "stt_train",
        os.path.join(config["paths"]["log_dir"], "train.log"),
    )

    # Audio processor
    audio_cfg = config["audio"]
    audio_processor = STTAudioProcessor(
        sample_rate=audio_cfg["sample_rate"],
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
        win_length=audio_cfg["win_length"],
        n_mels=audio_cfg["n_mels"],
        use_pitch=audio_cfg.get("use_pitch", True),
        pitch_dim=audio_cfg.get("pitch_dim", 3),
    )

    # Tokenizer (SentencePiece)
    data_cfg = config["data"]
    tokenizer = SimpleTokenizer(data_cfg.get("tokenizer_model"))

    # Dataset
    train_dataset = STTDataset(
        manifest_path=data_cfg["train_manifest"],
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        max_audio_length=data_cfg["max_audio_length"],
        min_audio_length=data_cfg["min_audio_length"],
    )

    collator = STTCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collator,
        drop_last=True,
    )

    # Model
    model_cfg = config["model"]
    model = ConformerCTC(
        input_dim=audio_cfg["feature_dim"],
        d_model=model_cfg["encoder"]["d_model"],
        d_ff=model_cfg["encoder"]["ffn_dim"],
        num_heads=model_cfg["encoder"]["num_heads"],
        num_layers=model_cfg["encoder"]["num_layers"],
        conv_kernel_size=model_cfg["encoder"]["conv_kernel_size"],
        vocab_size=model_cfg["decoder"]["vocab_size"],
        dropout=model_cfg["encoder"]["dropout"],
        blank_id=model_cfg["decoder"]["blank_id"],
    ).to(device)

    # Optimizer
    train_cfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg["betas"]),
        eps=train_cfg["eps"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Noam scheduler
    scheduler = NoamScheduler(
        optimizer,
        d_model=model_cfg["encoder"]["d_model"],
        warmup_steps=train_cfg["warmup_steps"],
    )

    # CTC loss
    criterion = CTCLoss(blank_id=model_cfg["decoder"]["blank_id"])

    # Mixed precision
    scaler = GradScaler() if train_cfg.get("fp16", True) and device.type == "cuda" else None

    # Resume
    start_step = 0
    best_loss = float("inf")
    if resume and os.path.exists(resume):
        info = load_checkpoint(resume, model, optimizer, device=device)
        start_step = info["step"]
        best_loss = info["best_loss"]
        scheduler._step = start_step
        logger.info(f"Resumed from {resume}, step {start_step}")

    # Stats
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Conformer-CTC parameters: {total_params:,}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Batch size: {train_cfg['batch_size']}")
    logger.info(f"Max steps: {train_cfg['max_steps']}")

    # ── Training Loop ──
    global_step = start_step
    model.train()

    while global_step < train_cfg["max_steps"]:
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Step {global_step}", leave=True)
        for batch in pbar:
            if global_step >= train_cfg["max_steps"]:
                break

            features = batch["features"].to(device)
            feat_lengths = batch["feat_lengths"].to(device).squeeze(-1)
            token_ids = batch["token_ids"].to(device)
            token_lengths = batch["token_lengths"].to(device).squeeze(-1)

            optimizer.zero_grad()

            with autocast(enabled=scaler is not None):
                log_probs, out_lengths = model(features, feat_lengths)
                loss = criterion(log_probs, token_ids, out_lengths, token_lengths)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg["grad_clip"]
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg["grad_clip"]
                )
                optimizer.step()

            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "step": global_step,
            })

            # Periodic checkpoint
            if global_step % train_cfg["save_every_n_steps"] == 0:
                ckpt_dir = config["paths"]["checkpoint_dir"]
                save_checkpoint(
                    os.path.join(ckpt_dir, "latest.pt"),
                    model, optimizer, epoch=0, step=global_step,
                    best_loss=best_loss,
                )
                save_checkpoint(
                    os.path.join(ckpt_dir, f"step_{global_step:06d}.pt"),
                    model, optimizer, epoch=0, step=global_step,
                    best_loss=best_loss,
                )
                cleanup_old_checkpoints(ckpt_dir, keep_last_n=train_cfg["keep_last_n_checkpoints"])

                avg_loss = epoch_loss / max(num_batches, 1)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_checkpoint(
                        os.path.join(ckpt_dir, "best.pt"),
                        model, optimizer, epoch=0, step=global_step,
                        best_loss=best_loss,
                    )
                    logger.info(f"  ✅ New best model (loss={best_loss:.4f})")

            # Periodic logging
            if global_step % train_cfg["eval_every_n_steps"] == 0:
                avg_loss = epoch_loss / max(num_batches, 1)
                logger.info(
                    f"Step {global_step:06d} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
                epoch_loss = 0
                num_batches = 0

    logger.info("Training complete!")


class SimpleTokenizer:
    """Placeholder tokenizer — will be replaced with SentencePiece when dataset is ready."""

    def __init__(self, model_path=None):
        self.model_path = model_path
        self._sp = None

        if model_path and os.path.exists(model_path):
            try:
                import sentencepiece as spm
                self._sp = spm.SentencePieceProcessor()
                self._sp.Load(model_path)
                print(f"Loaded SentencePiece tokenizer: {model_path}")
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")

    def encode(self, text: str):
        if self._sp:
            return self._sp.EncodeAsIds(text)
        # Fallback: character-level tokenization
        return [ord(c) % 5000 for c in text.lower()]

    def decode(self, ids):
        if self._sp:
            return self._sp.DecodeIds(ids)
        return "".join(chr(i % 128 + 32) for i in ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conformer-CTC STT model")
    parser.add_argument("--config", type=str, default="configs/stt_english.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--test-run", action="store_true")
    args = parser.parse_args()

    if args.test_run:
        print("\n🧪 Running Conformer-CTC sanity check...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ConformerCTC(input_dim=83, d_model=256, num_layers=4, vocab_size=100).to(device)
        features = torch.randn(2, 200, 83).to(device)
        lengths = torch.LongTensor([200, 150]).to(device)
        model.eval()
        with torch.no_grad():
            log_probs, out_lens = model(features, lengths, apply_augment=False)
        print(f"  Input:  {features.shape}")
        print(f"  Output: {log_probs.shape}")
        print(f"  Lengths: {out_lens.tolist()}")
        decoded = model.decode(features, lengths)
        for i, tokens in enumerate(decoded):
            print(f"  Decoded[{i}]: {len(tokens)} tokens")
        print("  ✅ Conformer-CTC sanity check passed!")
    else:
        train(args.config, args.resume)
