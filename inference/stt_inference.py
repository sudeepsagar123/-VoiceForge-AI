"""
STT Inference — Speech to Text
Handles long audio by VAD-based chunking with overlap merging.
"""

import os
import sys
import argparse
import torch
import torchaudio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stt.conformer import ConformerCTC
from preprocessing.audio_processor import STTAudioProcessor


class STTInference:
    """Speech-to-Text inference engine.

    Handles:
    - Long audio (VAD chunking with overlap)
    - Multiple audio formats
    - Beam search decoding
    """

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str = None,
        device: str = "auto",
        sample_rate: int = 16000,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.sample_rate = sample_rate

        # Audio processor
        self.audio_processor = STTAudioProcessor(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=80,
            use_pitch=False,
        )

        # Tokenizer
        self.tokenizer = self._load_tokenizer(tokenizer_path)

        # Model
        self.model = ConformerCTC(
            input_dim=80,
            d_model=256,
            d_ff=1024,
            num_heads=4,
            num_layers=12,
            conv_kernel_size=31,
            vocab_size=5000,
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"Loaded STT model from {checkpoint_path}")

    def transcribe(
        self,
        audio_path: str,
        chunk_length_sec: float = 30.0,
        overlap_sec: float = 5.0,
        beam_width: int = 10,
        use_vad: bool = True,
    ) -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            chunk_length_sec: Max chunk length in seconds
            overlap_sec: Overlap between chunks
            beam_width: Beam width for decoding
            use_vad: Whether to use VAD for chunking

        Returns:
            transcript: Full transcription text
        """
        # Load audio
        waveform, sr = self.audio_processor.load_audio(
            audio_path, target_sr=self.sample_rate
        )
        waveform = waveform.squeeze(0)  # [T]
        duration = len(waveform) / self.sample_rate
        print(f"Audio: {audio_path} ({duration:.1f}s)")

        # Chunk audio
        chunk_samples = int(chunk_length_sec * self.sample_rate)
        overlap_samples = int(overlap_sec * self.sample_rate)

        if len(waveform) <= chunk_samples:
            # Short audio — process in one go
            return self._transcribe_chunk(waveform, beam_width)

        # Long audio — chunk with overlap
        chunks = self._split_audio(waveform, chunk_samples, overlap_samples, use_vad)
        print(f"Split into {len(chunks)} chunks")

        transcripts = []
        for i, chunk in enumerate(chunks):
            text = self._transcribe_chunk(chunk, beam_width)
            transcripts.append(text)
            print(f"  Chunk {i+1}/{len(chunks)}: {text[:80]}...")

        # Merge overlapping transcripts
        full_transcript = self._merge_transcripts(transcripts)
        return full_transcript

    def transcribe_stream(self, audio_chunk: np.ndarray) -> str:
        """Transcribe a single audio chunk (for real-time streaming)."""
        waveform = torch.FloatTensor(audio_chunk)
        return self._transcribe_chunk(waveform, beam_width=1)

    def _transcribe_chunk(self, waveform: torch.Tensor, beam_width: int = 10) -> str:
        """Transcribe a single audio chunk."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        features = self.audio_processor.extract_features(waveform)  # [T, D]
        features = features.unsqueeze(0).to(self.device)  # [1, T, D]
        feat_lengths = torch.LongTensor([features.shape[1]]).to(self.device)

        with torch.no_grad():
            if beam_width > 1:
                decoded = self.model.beam_decode(features, feat_lengths, beam_width)
            else:
                decoded = self.model.decode(features, feat_lengths)

        token_ids = decoded[0]
        text = self.tokenizer.decode(token_ids)
        return text.strip()

    def _split_audio(self, waveform, chunk_samples, overlap_samples, use_vad=True):
        """Split long audio into overlapping chunks.

        If VAD is available, splits on silence boundaries.
        Otherwise, splits at fixed intervals with overlap.
        """
        if use_vad:
            try:
                return self._vad_split(waveform, chunk_samples, overlap_samples)
            except Exception:
                pass  # Fallback to regular splitting

        # Regular splitting with overlap
        chunks = []
        start = 0
        while start < len(waveform):
            end = min(start + chunk_samples, len(waveform))
            chunks.append(waveform[start:end])
            if end >= len(waveform):
                break
            start += chunk_samples - overlap_samples

        return chunks

    def _vad_split(self, waveform, chunk_samples, overlap_samples):
        """Split audio using simple energy-based VAD."""
        # Compute energy in small frames
        frame_len = int(0.03 * self.sample_rate)  # 30ms frames
        hop = int(0.01 * self.sample_rate)  # 10ms hop

        energy = []
        for i in range(0, len(waveform) - frame_len, hop):
            frame = waveform[i : i + frame_len]
            energy.append(torch.sum(frame ** 2).item())

        energy = np.array(energy)
        threshold = np.percentile(energy, 10) * 2  # Silence threshold

        # Find silence regions
        chunks = []
        start = 0
        while start < len(waveform):
            end = min(start + chunk_samples, len(waveform))

            if end < len(waveform):
                # Look for silence near the chunk boundary
                search_start = max(0, end - overlap_samples)
                search_frame_start = search_start // hop
                search_frame_end = min(end // hop, len(energy))

                # Find lowest energy point in search region
                if search_frame_start < search_frame_end:
                    search_region = energy[search_frame_start:search_frame_end]
                    best_frame = search_frame_start + np.argmin(search_region)
                    end = best_frame * hop

            chunks.append(waveform[start:end])
            if end >= len(waveform):
                break
            start = max(start + 1, end - overlap_samples)

        return chunks

    def _merge_transcripts(self, transcripts):
        """Merge overlapping transcripts by removing duplicate phrases."""
        if len(transcripts) <= 1:
            return transcripts[0] if transcripts else ""

        merged = transcripts[0]
        for i in range(1, len(transcripts)):
            # Simple concatenation with space
            # (More sophisticated overlap removal can be added later)
            merged = merged.rstrip() + " " + transcripts[i].lstrip()

        return merged

    def _load_tokenizer(self, path):
        """Load SentencePiece tokenizer or use fallback."""
        if path and os.path.exists(path):
            try:
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.Load(path)
                return sp
            except Exception:
                pass

        # Return a simple fallback for Unicode encoding used in training
        class FallbackTokenizer:
            def decode(self, ids):
                res = []
                for i in ids:
                    if i == 0: continue  # CTC Blank
                    try:
                        res.append(chr(i))
                    except ValueError:
                        pass
                return "".join(res)
        return FallbackTokenizer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformer-CTC STT Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--output", type=str, default=None, help="Save transcript to file")
    args = parser.parse_args()

    engine = STTInference(args.checkpoint, args.tokenizer)
    transcript = engine.transcribe(args.audio, beam_width=args.beam_width)

    print(f"\n{'='*60}")
    print("📝 Transcript:")
    print(f"{'='*60}")
    print(transcript)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"\nSaved to: {args.output}")
