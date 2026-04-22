"""
VITS2 TTS Training Script
Complete training loop with adversarial training, mixed precision, and checkpointing.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tts.vits2 import VITS2
from models.tts.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from preprocessing.audio_processor import AudioProcessor
from preprocessing.text_normalizer import TextNormalizer
from preprocessing.dataset import TTSDataset, TTSCollator, build_vocab
from training.losses import (
    generator_loss,
    discriminator_loss,
    feature_matching_loss,
    kl_loss,
    mel_loss,
)
from training.utils import (
    setup_logger,
    save_checkpoint,
    load_checkpoint,
    cleanup_old_checkpoints,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def slice_segments(x, ids_str, segment_size):
    """Slice random segments from waveform for discriminator training."""
    ret = torch.zeros(x.size(0), segment_size, device=x.device, dtype=x.dtype)
    for i in range(x.size(0)):
        idx_start = ids_str[i] * 256  # hop_length
        idx_end = idx_start + segment_size
        end = min(idx_end, x.size(1))
        length = end - idx_start
        if length > 0:
            ret[i, :length] = x[i, idx_start:end]
    return ret


def train(config_path: str, resume: str = None):
    """Main training function."""
    config = load_config(config_path)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"VITS2 TTS Training — {config['model']['name']}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'='*60}\n")

    # Logger
    logger = setup_logger(
        "tts_train",
        os.path.join(config["paths"]["log_dir"], "train.log"),
    )

    # Audio processor
    audio_cfg = config["audio"]
    audio_processor = AudioProcessor(
        sample_rate=audio_cfg["sample_rate"],
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
        win_length=audio_cfg["win_length"],
        n_mels=audio_cfg["n_mels"],
        mel_fmin=audio_cfg["mel_fmin"],
        mel_fmax=audio_cfg.get("mel_fmax"),
    )

    # Load or build vocabulary
    data_cfg = config["data"]
    vocab_path = os.path.join(os.path.dirname(data_cfg["train_filelist"]), "vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"Loaded vocabulary: {len(vocab)} tokens")
    else:
        print("Building vocabulary from training data...")
        texts = []
        with open(data_cfg["train_filelist"], "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    texts.append(parts[-1].lower())
        vocab = build_vocab(texts, vocab_path)

    # Datasets
    text_normalizer = TextNormalizer(language=data_cfg["language"])
    train_dataset = TTSDataset(
        filelist_path=data_cfg["train_filelist"],
        audio_processor=audio_processor,
        text_normalizer=text_normalizer,
        vocab=vocab,
        max_audio_length=data_cfg["max_audio_length"],
        max_text_length=data_cfg["max_text_length"],
        segment_size=audio_cfg["segment_size"],
    )

    collator = TTSCollator(segment_size=audio_cfg["segment_size"] // audio_cfg["hop_length"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,  # CRITICAL: Must be 0 to bypass Docker /dev/shm deadlocks
        pin_memory=False,
        collate_fn=collator,
        drop_last=True,
    )

    # Model
    model_cfg = config["model"]
    model = VITS2(
        vocab_size=len(vocab),
        hidden_channels=model_cfg["text_encoder"]["hidden_dim"],
        filter_channels=model_cfg["text_encoder"]["ffn_dim"],
        n_heads=model_cfg["text_encoder"]["num_heads"],
        n_enc_layers=model_cfg["text_encoder"]["num_layers"],
        spec_channels=audio_cfg["n_fft"] // 2 + 1,
        segment_size=audio_cfg["segment_size"],
        hop_length=audio_cfg["hop_length"],
        use_sdp=False,  # Start with deterministic DP
    ).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Discriminators
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    if torch.cuda.device_count() > 1:
        mpd = torch.nn.DataParallel(mpd)
        msd = torch.nn.DataParallel(msd)

    # Optimizers
    train_cfg = config["training"]
    optim_g = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg["betas"]),
        eps=train_cfg["eps"],
        weight_decay=train_cfg["weight_decay"],
    )
    optim_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg["betas"]),
        eps=train_cfg["eps"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=train_cfg["lr_decay"])
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=train_cfg["lr_decay"])

    # Mixed precision
    scaler = GradScaler() if train_cfg.get("fp16", True) and device.type == "cuda" else None

    # Resume from checkpoint
    start_epoch = 0
    best_loss = float("inf")
    if resume and os.path.exists(resume):
        info = load_checkpoint(resume, model, optim_g, scheduler_g, device)
        start_epoch = info["epoch"]
        best_loss = info["best_loss"]
        logger.info(f"Resumed from {resume}, epoch {start_epoch}")

    # Loss weights
    lw = train_cfg["loss_weights"]

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"VITS2 parameters: {total_params:,}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Batch size: {train_cfg['batch_size']}")
    logger.info(f"Starting training from epoch {start_epoch}")

    # ── Training Loop ──
    for epoch in range(start_epoch, train_cfg["epochs"]):
        model.train()
        mpd.train()
        msd.train()

        epoch_losses = {"total": 0, "mel": 0, "kl": 0, "dur": 0, "gen": 0, "fm": 0, "disc": 0}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:04d}", leave=True)
        for batch in pbar:
            # Move to device
            text_ids = batch["text_ids"].to(device)
            text_lengths = batch["text_lengths"].to(device).squeeze(-1)
            specs = batch["specs"].to(device)
            spec_lengths = batch["spec_lengths"].to(device).squeeze(-1)
            waveforms = batch["waveforms"].to(device)

            # ── Generator Forward ──
            with autocast(enabled=scaler is not None):
                outputs = model(text_ids, text_lengths, specs, spec_lengths)
            # Generate target mel (must run outside autocast due to FP16 STFT bugs in PyTorch 2.4+)
            with torch.cuda.amp.autocast(enabled=False):
                # Get real audio segments
                y = slice_segments(
                    waveforms, outputs["ids_slice"],
                    audio_cfg["segment_size"],
                ).unsqueeze(1)
                y_mel = audio_processor.get_mel_spectrogram(y.squeeze(1))

            # ── Generator Forward (Losses) ──
            with autocast(enabled=scaler is not None):
                y_hat = outputs["output"]  # Generated audio segment
                
                # Mel loss (prediction mel is still under autocast for speed)
                y_hat_mel = audio_processor.get_mel_spectrogram(y_hat.squeeze(1))
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * lw["mel"]

            # ── Discriminator Training ──
            optim_d.zero_grad()
            with autocast(enabled=scaler is not None):
                y_dp_r, y_dp_g, _, _ = mpd(y, y_hat.detach())
                y_ds_r, y_ds_g, _, _ = msd(y, y_hat.detach())
                loss_disc = discriminator_loss(y_dp_r, y_dp_g) + discriminator_loss(y_ds_r, y_ds_g)

            if scaler:
                scaler.scale(loss_disc).backward()
                scaler.step(optim_d)
            else:
                loss_disc.backward()
                optim_d.step()

            # ── Generator Training ──
            optim_g.zero_grad()
            with autocast(enabled=scaler is not None):
                y_dp_r, y_dp_g, fmap_p_r, fmap_p_g = mpd(y, y_hat)
                y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = msd(y, y_hat)

                loss_gen = generator_loss(y_dp_g) + generator_loss(y_ds_g)
                loss_fm = (
                    feature_matching_loss(fmap_p_r, fmap_p_g)
                    + feature_matching_loss(fmap_s_r, fmap_s_g)
                ) * lw["feature_matching"]

                loss_kl_val = kl_loss(
                    outputs["z_p"], outputs["logs_q"],
                    outputs["m_p"], outputs["logs_p"],
                    outputs["y_mask"],
                ) * lw["kl"]

                loss_dur = torch.mean(outputs["l_length"]) * lw["duration"]

                loss_total = loss_mel + loss_kl_val + loss_dur + loss_gen + loss_fm

            if scaler:
                scaler.scale(loss_total).backward()
                scaler.step(optim_g)
                scaler.update()
            else:
                loss_total.backward()
                optim_g.step()

            # Track losses
            epoch_losses["total"] += loss_total.item()
            epoch_losses["mel"] += loss_mel.item()
            epoch_losses["kl"] += loss_kl_val.item()
            epoch_losses["dur"] += loss_dur.item()
            epoch_losses["gen"] += loss_gen.item()
            epoch_losses["fm"] += loss_fm.item()
            epoch_losses["disc"] += loss_disc.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss_total.item():.3f}",
                "mel": f"{loss_mel.item():.3f}",
                "kl": f"{loss_kl_val.item():.3f}",
            })

        # End of epoch
        scheduler_g.step()
        scheduler_d.step()

        avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        logger.info(
            f"Epoch {epoch+1:04d} | "
            f"Loss: {avg_losses['total']:.4f} | "
            f"mel={avg_losses['mel']:.4f} "
            f"kl={avg_losses['kl']:.4f} "
            f"dur={avg_losses['dur']:.4f} "
            f"gen={avg_losses['gen']:.4f} "
            f"fm={avg_losses['fm']:.4f} "
            f"disc={avg_losses['disc']:.4f}"
        )

        # Save checkpoints
        ckpt_dir = config["paths"]["checkpoint_dir"]
        save_checkpoint(
            os.path.join(ckpt_dir, "latest.pt"),
            model.module if hasattr(model, "module") else model,
            optim_g, scheduler_g, epoch + 1,
            best_loss=best_loss,
        )

        if avg_losses["total"] < best_loss:
            best_loss = avg_losses["total"]
            save_checkpoint(
                os.path.join(ckpt_dir, "best.pt"),
                model.module if hasattr(model, "module") else model, 
                optim_g, scheduler_g, epoch + 1,
                best_loss=best_loss,
            )
            logger.info(f"  ✅ New best model saved (loss={best_loss:.4f})")

        if (epoch + 1) % train_cfg["save_every_n_epochs"] == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, f"epoch_{epoch+1:04d}.pt"),
                model.module if hasattr(model, "module") else model,
                optim_g, scheduler_g, epoch + 1,
                best_loss=best_loss,
            )
            cleanup_old_checkpoints(ckpt_dir, keep_last_n=train_cfg["keep_last_n_checkpoints"])

    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VITS2 TTS model")
    parser.add_argument("--config", type=str, default="configs/tts_english.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--test-run", action="store_true", help="Run a quick sanity check")
    args = parser.parse_args()

    if args.test_run:
        print("\n🧪 Running VITS2 sanity check...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VITS2(vocab_size=100, use_sdp=False).to(device)
        x = torch.randint(0, 100, (2, 30)).to(device)
        x_len = torch.LongTensor([30, 20]).to(device)
        model.eval()
        with torch.no_grad():
            audio, _, _ = model.infer(x, x_len)
        print(f"  Input: {x.shape}, Output: {audio.shape}")
        print(f"  Audio: ~{audio.shape[2]/22050:.2f}s at 22050 Hz")
        print("  ✅ VITS2 sanity check passed!")
    else:
        train(args.config, args.resume)
