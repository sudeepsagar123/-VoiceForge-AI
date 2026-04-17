"""
PyTorch Dataset classes for TTS and STT training.
"""

import os
import json
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Tuple

from preprocessing.audio_processor import AudioProcessor, STTAudioProcessor
from preprocessing.text_normalizer import TextNormalizer


# ─────────────────────────────────────────────
#  TTS Dataset (for VITS2)
# ─────────────────────────────────────────────

class TTSDataset(Dataset):
    """Dataset for VITS2 TTS training.

    Filelist format (one per line):
        audio_path|text
    """

    def __init__(
        self,
        filelist_path: str,
        audio_processor: AudioProcessor,
        text_normalizer: TextNormalizer,
        vocab: Dict[str, int],
        max_audio_length: float = 15.0,
        max_text_length: int = 300,
        segment_size: int = 8192,
        language: str = "en-in",
    ):
        self.audio_processor = audio_processor
        self.text_normalizer = text_normalizer
        self.vocab = vocab
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.segment_size = segment_size
        self.language = language

        # Load filelist
        self.data = []
        with open(filelist_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "|" in line:
                    parts = line.split("|")
                    audio_path = parts[0].strip()
                    text = parts[-1].strip()  # Use last column for normalized text
                    if text and os.path.exists(audio_path):
                        self.data.append((audio_path, text))

        print(f"[TTSDataset] Loaded {len(self.data)} samples from {filelist_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path, text = self.data[idx]

        # --- Text Processing ---
        text = self.text_normalizer.normalize(text)
        text_ids = self._text_to_ids(text)
        text_ids = torch.LongTensor(text_ids)
        text_lengths = torch.LongTensor([len(text_ids)])

        # --- Audio Processing ---
        waveform, sr = self.audio_processor.load_audio(audio_path)
        waveform = waveform.squeeze(0)  # [T]

        # Get linear spectrogram (for posterior encoder)
        spec = self.audio_processor.get_linear_spectrogram(waveform)  # [n_fft//2+1, T_spec]
        spec_lengths = torch.LongTensor([spec.shape[1]])

        return {
            "text_ids": text_ids,
            "text_lengths": text_lengths,
            "spec": spec,
            "spec_lengths": spec_lengths,
            "waveform": waveform,
        }

    def _text_to_ids(self, text: str) -> List[int]:
        """Convert text to integer IDs using vocabulary."""
        ids = []
        for char in text.lower():
            if char in self.vocab:
                ids.append(self.vocab[char])
        # Truncate if too long
        if len(ids) > self.max_text_length:
            ids = ids[: self.max_text_length]
        return ids


class TTSCollator:
    """Collate function for TTS batches with padding."""

    def __init__(self, segment_size: int = 8192):
        self.segment_size = segment_size

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Find max lengths
        max_text_len = max(item["text_ids"].shape[0] for item in batch)
        max_spec_len = max(item["spec"].shape[1] for item in batch)
        max_wav_len = max(item["waveform"].shape[0] for item in batch)

        batch_size = len(batch)
        n_freq = batch[0]["spec"].shape[0]

        # Allocate padded tensors
        text_ids = torch.zeros(batch_size, max_text_len, dtype=torch.long)
        text_lengths = torch.zeros(batch_size, dtype=torch.long)
        specs = torch.zeros(batch_size, n_freq, max_spec_len)
        spec_lengths = torch.zeros(batch_size, dtype=torch.long)
        waveforms = torch.zeros(batch_size, max_wav_len)

        for i, item in enumerate(batch):
            t_len = item["text_ids"].shape[0]
            s_len = item["spec"].shape[1]
            w_len = item["waveform"].shape[0]

            text_ids[i, :t_len] = item["text_ids"]
            text_lengths[i] = t_len
            specs[i, :, :s_len] = item["spec"]
            spec_lengths[i] = s_len
            waveforms[i, :w_len] = item["waveform"]

        return {
            "text_ids": text_ids,
            "text_lengths": text_lengths,
            "specs": specs,
            "spec_lengths": spec_lengths,
            "waveforms": waveforms,
        }


# ─────────────────────────────────────────────
#  STT Dataset (for Conformer-CTC)
# ─────────────────────────────────────────────

class STTDataset(Dataset):
    """Dataset for Conformer-CTC STT training.

    Manifest format (JSON lines):
        {"audio_filepath": "...", "text": "...", "duration": 5.2}
    """

    def __init__(
        self,
        manifest_path: str,
        audio_processor: STTAudioProcessor,
        tokenizer,
        max_audio_length: float = 30.0,
        min_audio_length: float = 0.5,
    ):
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length

        # Load manifest
        self.data = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                duration = entry.get("duration", 999)
                if self.min_audio_length <= duration <= self.max_audio_length:
                    self.data.append(entry)

        print(f"[STTDataset] Loaded {len(self.data)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.data[idx]
        audio_path = entry["audio_filepath"]
        text = entry["text"]

        # --- Audio Features ---
        waveform, sr = self.audio_processor.load_audio(audio_path)
        features = self.audio_processor.extract_features(waveform)  # [T, D]
        feat_length = torch.LongTensor([features.shape[0]])

        # --- Text Tokenization ---
        token_ids = self.tokenizer.encode(text)
        token_ids = torch.LongTensor(token_ids)
        token_length = torch.LongTensor([len(token_ids)])

        return {
            "features": features,
            "feat_length": feat_length,
            "token_ids": token_ids,
            "token_length": token_length,
        }


class STTCollator:
    """Collate function for STT batches with padding."""

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_feat_len = max(item["features"].shape[0] for item in batch)
        max_token_len = max(item["token_ids"].shape[0] for item in batch)
        batch_size = len(batch)
        feat_dim = batch[0]["features"].shape[1]

        features = torch.zeros(batch_size, max_feat_len, feat_dim)
        feat_lengths = torch.zeros(batch_size, dtype=torch.long)
        token_ids = torch.zeros(batch_size, max_token_len, dtype=torch.long)
        token_lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, item in enumerate(batch):
            f_len = item["features"].shape[0]
            t_len = item["token_ids"].shape[0]

            features[i, :f_len] = item["features"]
            feat_lengths[i] = f_len
            token_ids[i, :t_len] = item["token_ids"]
            token_lengths[i] = t_len

        return {
            "features": features,
            "feat_lengths": feat_lengths,
            "token_ids": token_ids,
            "token_lengths": token_lengths,
        }


def build_vocab(texts: List[str], save_path: str) -> Dict[str, int]:
    """Build character-level vocabulary from texts.

    Reserves:
        0 = <pad>
        1 = <unk>
        2 = <bos>
        3 = <eos>
    """
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    chars = set()
    for text in texts:
        chars.update(text.lower())

    for i, char in enumerate(sorted(chars), start=4):
        vocab[char] = i

    # Save vocabulary
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    print(f"[Vocab] Built vocabulary with {len(vocab)} tokens → {save_path}")
    return vocab
