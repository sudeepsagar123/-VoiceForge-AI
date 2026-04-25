"""
TTS Inference — Text to Speech
Handles long text by splitting into sentences and cross-fading audio segments.
"""

import os
import sys
import re
import argparse
import json
import torch
import torchaudio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tts.vits2 import VITS2
from preprocessing.text_normalizer import TextNormalizer


class TTSInference:
    """Text-to-Speech inference engine.

    Handles:
    - Long text (splits into sentences, cross-fades)
    - Speed control
    - Multiple voices (speaker IDs)
    """

    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        device: str = "auto",
        sample_rate: int = 22050,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.sample_rate = sample_rate

        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        print(f"Loaded vocab: {len(self.vocab)} tokens")

        # Load model
        self.model = VITS2(
            vocab_size=len(self.vocab),
            use_sdp=False,
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model.remove_weight_norm()
        print(f"Loaded model from {checkpoint_path}")

        self.normalizer = TextNormalizer(language="en-in")

    def synthesize(
        self,
        text: str,
        output_path: str = "output.wav",
        speed: float = 1.0,
        noise_scale: float = 0.667,
        max_chars: int = 200,
        crossfade_ms: int = 50,
    ) -> str:
        """Synthesize speech from text.

        Args:
            text: Input text (any length)
            output_path: Path to save WAV file
            speed: Speaking speed (< 1 = faster, > 1 = slower)
            noise_scale: Controls voice variation
            max_chars: Max characters per chunk
            crossfade_ms: Crossfade between chunks in ms

        Returns:
            output_path: Path to generated WAV file
        """
        # Normalize text
        text = self.normalizer.normalize(text)

        # Split into sentences
        sentences = self._split_sentences(text, max_chars)
        print(f"Processing {len(sentences)} segments...")

        # Generate audio for each sentence
        audio_segments = []
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            ids = self._text_to_ids(sentence)
            if len(ids) == 0:
                continue

            x = torch.LongTensor([ids]).to(self.device)
            x_lengths = torch.LongTensor([len(ids)]).to(self.device)

            with torch.no_grad():
                audio, _, _ = self.model.infer(
                    x, x_lengths,
                    noise_scale=noise_scale,
                    length_scale=speed,
                )

            audio_np = audio.squeeze().cpu().numpy()
            audio_segments.append(audio_np)

        if not audio_segments:
            print("Warning: No audio generated!")
            return output_path

        # Concatenate with crossfade
        crossfade_samples = int(crossfade_ms * self.sample_rate / 1000)
        full_audio = self._crossfade_segments(audio_segments, crossfade_samples)

        # Normalize volume
        max_val = np.abs(full_audio).max()
        if max_val > 0:
            full_audio = full_audio / max_val * 0.95

        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        try:
            import soundfile as sf
            sf.write(output_path, full_audio, self.sample_rate)
        except ImportError:
            torchaudio.save(
                output_path,
                torch.FloatTensor(full_audio).unsqueeze(0),
                self.sample_rate,
            )
        duration = len(full_audio) / self.sample_rate
        print(f"Saved: {output_path} ({duration:.1f}s)")
        return output_path

    def _text_to_ids(self, text: str):
        ids = []
        for char in text.lower():
            if char in self.vocab:
                ids.append(self.vocab[char])
            elif "<unk>" in self.vocab:
                ids.append(self.vocab["<unk>"])
        return ids

    def _split_sentences(self, text: str, max_chars: int = 200):
        """Split text into sentences respecting max length."""
        # Split on sentence boundaries
        raw = re.split(r"(?<=[.!?;])\s+", text)

        sentences = []
        for s in raw:
            s = s.strip()
            if not s:
                continue
            # Further split if still too long
            if len(s) > max_chars:
                # Split on commas or clause boundaries
                parts = re.split(r"(?<=,)\s+", s)
                current = ""
                for part in parts:
                    if len(current) + len(part) > max_chars:
                        if current:
                            sentences.append(current.strip())
                        current = part
                    else:
                        current += " " + part
                if current:
                    sentences.append(current.strip())
            else:
                sentences.append(s)

        return sentences

    def _crossfade_segments(self, segments, crossfade_samples):
        """Concatenate audio segments with crossfading."""
        if len(segments) == 1:
            return segments[0]

        total_length = sum(len(s) for s in segments)
        total_length -= crossfade_samples * (len(segments) - 1)
        result = np.zeros(max(total_length, 0))

        pos = 0
        for i, seg in enumerate(segments):
            if i == 0:
                result[pos : pos + len(seg)] = seg
                pos += len(seg) - crossfade_samples
            else:
                # Crossfade region
                fade_len = min(crossfade_samples, len(seg), len(result) - pos)
                if fade_len > 0:
                    fade_out = np.linspace(1, 0, fade_len)
                    fade_in = np.linspace(0, 1, fade_len)
                    result[pos : pos + fade_len] = (
                        result[pos : pos + fade_len] * fade_out + seg[:fade_len] * fade_in
                    )
                # Non-overlapping part
                remaining = len(seg) - fade_len
                if remaining > 0 and pos + fade_len + remaining <= len(result):
                    result[pos + fade_len : pos + fade_len + remaining] = seg[fade_len:]
                pos += len(seg) - crossfade_samples

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VITS2 TTS Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/tts_english/output.wav")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    engine = TTSInference(args.checkpoint, args.vocab)

    if args.text:
        engine.synthesize(args.text, args.output, speed=args.speed)
    else:
        # Interactive mode
        print("\n🎤 VITS2 TTS — Interactive Mode")
        print("Type text to synthesize, 'quit' to exit\n")
        count = 0
        while True:
            text = input("📝 Enter text: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            count += 1
            out = f"outputs/tts_english/output_{count:03d}.wav"
            engine.synthesize(text, out, speed=args.speed)
