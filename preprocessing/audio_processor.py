"""
Audio Processor for TTS and STT preprocessing.
Handles loading, resampling, mel extraction, and feature computation.
"""

import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple


class AudioProcessor:
    """Process audio for TTS (mel spectrograms) and STT (filterbank features)."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

        # Create mel filterbank
        self.mel_basis = {}
        self.hann_window = {}

    def load_audio(
        self, path: str, target_sr: Optional[int] = None
    ) -> Tuple[torch.Tensor, int]:
        """Load and optionally resample audio file.

        Returns:
            waveform: [1, T] tensor
            sample_rate: int
        """
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype='float32')
            waveform = torch.FloatTensor(data)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T  # soundfile returns [T, C], we need [C, T]
        except ImportError:
            waveform, sr = torchaudio.load(path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        target_sr = target_sr or self.sample_rate
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
            sr = target_sr

        return waveform, sr

    def get_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram from waveform.

        Args:
            waveform: [1, T] or [T] tensor

        Returns:
            mel: [n_mels, T_mel] tensor
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Get or create hann window
        device = waveform.device
        dtype = waveform.dtype
        key = f"{self.n_fft}_{device}_{dtype}"

        if key not in self.hann_window:
            self.hann_window[key] = torch.hann_window(
                self.win_length, device=device, dtype=dtype
            )

        # Pad waveform
        pad_amount = (self.n_fft - self.hop_length) // 2
        waveform = torch.nn.functional.pad(
            waveform, (pad_amount, pad_amount), mode="reflect"
        )

        # STFT
        spec = torch.stft(
            waveform.squeeze(0),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[key],
            center=False,
            return_complex=True,
        )
        spec = torch.abs(spec)  # magnitude

        # Mel filterbank
        mel_key = f"{self.n_fft}_{device}_{dtype}"
        if mel_key not in self.mel_basis:
            mel_fb = torchaudio.functional.melscale_fbanks(
                n_freqs=self.n_fft // 2 + 1,
                f_min=self.mel_fmin,
                f_max=self.mel_fmax or self.sample_rate / 2.0,
                n_mels=self.n_mels,
                sample_rate=self.sample_rate,
            ).to(device=device, dtype=dtype)
            self.mel_basis[mel_key] = mel_fb

        # Apply mel filterbank: [n_fft//2+1, T] @ [n_fft//2+1, n_mels] -> [n_mels, T]
        mel = torch.matmul(self.mel_basis[mel_key].T, spec)

        # Log mel
        mel = torch.clamp(mel, min=1e-5)
        mel = torch.log(mel)

        return mel

    def get_linear_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract linear spectrogram (for VITS2 posterior encoder).

        Returns:
            spec: [n_fft//2+1, T_spec] tensor
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        device = waveform.device
        dtype = waveform.dtype
        key = f"{self.n_fft}_{device}_{dtype}"

        if key not in self.hann_window:
            self.hann_window[key] = torch.hann_window(
                self.win_length, device=device, dtype=dtype
            )

        pad_amount = (self.n_fft - self.hop_length) // 2
        waveform = torch.nn.functional.pad(
            waveform, (pad_amount, pad_amount), mode="reflect"
        )

        spec = torch.stft(
            waveform.squeeze(0),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[key],
            center=False,
            return_complex=True,
        )
        spec = torch.abs(spec)
        return spec


class STTAudioProcessor(AudioProcessor):
    """Extended audio processor for STT with filterbank + pitch features."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,      # 10ms
        win_length: int = 400,       # 25ms
        n_mels: int = 80,
        use_pitch: bool = True,
        pitch_dim: int = 3,
    ):
        super().__init__(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
        )
        self.use_pitch = use_pitch
        self.pitch_dim = pitch_dim

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract log mel-filterbank + optional pitch features.

        Args:
            waveform: [1, T] tensor

        Returns:
            features: [T_frames, feature_dim] tensor
              feature_dim = n_mels (+ pitch_dim if use_pitch)
        """
        # Log mel-filterbank
        mel = self.get_mel_spectrogram(waveform)  # [n_mels, T]
        mel = mel.T  # [T, n_mels]

        if self.use_pitch:
            pitch_features = self._extract_pitch(waveform, mel.shape[0])
            features = torch.cat([mel, pitch_features], dim=-1)
        else:
            features = mel

        # Apply CMVN (Cepstral Mean Variance Normalization)
        features = self._apply_cmvn(features)

        return features

    def _extract_pitch(
        self, waveform: torch.Tensor, target_length: int
    ) -> torch.Tensor:
        """Extract pitch features (F0 + delta + delta-delta).

        Returns:
            pitch_features: [target_length, pitch_dim] tensor
        """
        try:
            # Detect pitch using torchaudio
            pitch = torchaudio.functional.detect_pitch_frequency(
                waveform,
                self.sample_rate,
                freq_low=60,
                freq_high=500,
            ).squeeze(0)

            # Resample pitch to match mel length
            if len(pitch) != target_length:
                pitch = torch.nn.functional.interpolate(
                    pitch.unsqueeze(0).unsqueeze(0),
                    size=target_length,
                    mode="linear",
                    align_corners=False,
                ).squeeze()

            # Compute delta and delta-delta
            delta = torch.zeros_like(pitch)
            delta[1:] = pitch[1:] - pitch[:-1]
            delta_delta = torch.zeros_like(delta)
            delta_delta[1:] = delta[1:] - delta[:-1]

            return torch.stack([pitch, delta, delta_delta], dim=-1)

        except Exception:
            # Fallback: return zeros if pitch extraction fails
            return torch.zeros(target_length, self.pitch_dim)

    def _apply_cmvn(self, features: torch.Tensor) -> torch.Tensor:
        """Apply utterance-level Cepstral Mean Variance Normalization."""
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-5)
        return (features - mean) / std
