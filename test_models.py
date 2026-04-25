"""
VoiceForge AI — Lightweight Model Test
Tests one model at a time to avoid RAM issues.
Usage: python test_models.py tts   OR   python test_models.py stt
"""

import os
import sys
import torch
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_tts():
    print("\n" + "=" * 60)
    print("  TTS (VITS2) Model Test")
    print("=" * 60)

    checkpoint = "checkpoints/tts_kannada/best.pt"
    vocab_path = "vocab_tts.json"

    if not os.path.exists(checkpoint):
        print("[FAIL] TTS checkpoint not found!")
        return
    if not os.path.exists(vocab_path):
        print("[FAIL] vocab_tts.json not found!")
        return

    # Check metadata only (lightweight)
    print("  Loading checkpoint metadata...")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch = ckpt.get('epoch', 'N/A')
    best_loss = ckpt.get('best_loss', 'N/A')
    param_count = sum(p.numel() for p in ckpt['model_state_dict'].values())
    del ckpt
    gc.collect()

    print(f"  Epoch trained    : {epoch}")
    print(f"  Best loss        : {best_loss}")
    print(f"  Parameters       : {param_count:,}")

    # Grade
    if isinstance(best_loss, float):
        if best_loss < 30:
            grade = "A (Excellent)"
        elif best_loss < 45:
            grade = "B (Good)"
        elif best_loss < 60:
            grade = "C (Fair - needs more epochs)"
        else:
            grade = "D (Needs more training)"
        print(f"  Quality Grade    : {grade}")
        print(f"\n  Loss Breakdown:")
        print(f"    Current  : {best_loss:.2f}")
        print(f"    Target   : < 30.0 for good quality")
        print(f"    Progress : {max(0, (1 - best_loss/60) * 100):.1f}% towards target")

    # Try inference
    print("\n  Attempting audio generation...")
    try:
        from inference.tts_inference import TTSInference
        engine = TTSInference(checkpoint, vocab_path)
        
        os.makedirs("outputs/tts_test", exist_ok=True)
        
        tests = [
            ("namaskara", "outputs/tts_test/test_namaskara.wav"),
            ("kannada bhaashe", "outputs/tts_test/test_kannada.wav"),
        ]
        
        for text, out_path in tests:
            start = time.time()
            engine.synthesize(text, out_path, speed=1.0)
            elapsed = time.time() - start
            size = os.path.getsize(out_path) / 1024 if os.path.exists(out_path) else 0
            print(f"    '{text}' -> {size:.0f}KB, {elapsed:.1f}s [OK]")
        
        print("\n  [OK] TTS is working! Check outputs/tts_test/ for audio files.")
        del engine
        gc.collect()
    except Exception as e:
        print(f"  [FAIL] Inference error: {e}")


def test_stt():
    print("\n" + "=" * 60)
    print("  STT (Conformer-CTC) Model Test")
    print("=" * 60)

    checkpoint = "checkpoints/stt_kannada/best.pt"

    if not os.path.exists(checkpoint):
        print("[FAIL] STT checkpoint not found!")
        return

    # Check metadata
    print("  Loading checkpoint metadata...")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    step = ckpt.get('step', 'N/A')
    best_loss = ckpt.get('best_loss', 'N/A')
    param_count = sum(p.numel() for p in ckpt['model_state_dict'].values())
    del ckpt
    gc.collect()

    print(f"  Training step    : {step}")
    print(f"  Best CTC loss    : {best_loss}")
    print(f"  Parameters       : {param_count:,}")

    # Grade 
    if isinstance(best_loss, float):
        if best_loss < 1.0:
            grade = "A (Excellent)"
        elif best_loss < 2.0:
            grade = "B (Good)"
        elif best_loss < 3.5:
            grade = "C (Fair - basic recognition)"
        else:
            grade = "D (Needs more data/training)"
        print(f"  Quality Grade    : {grade}")
        print(f"\n  Loss Breakdown:")
        print(f"    Current  : {best_loss:.4f}")
        print(f"    Target   : < 1.0 for good accuracy")
        print(f"    Progress : {max(0, (1 - best_loss/5) * 100):.1f}% towards target")

    # Try transcription if test audio exists
    test_audio = "outputs/tts_test/test_namaskara.wav"
    if os.path.exists(test_audio):
        print(f"\n  Round-trip test (TTS->STT):")
        try:
            from inference.stt_inference import STTInference
            engine = STTInference(checkpoint, tokenizer_path=None)
            start = time.time()
            transcript = engine.transcribe(test_audio, beam_width=5)
            elapsed = time.time() - start
            print(f"    Input    : {test_audio}")
            print(f"    Output   : {transcript[:80]}")
            print(f"    Time     : {elapsed:.1f}s")
            del engine
            gc.collect()
        except Exception as e:
            print(f"    [FAIL] {e}")
    else:
        print(f"\n  [INFO] Run 'python test_models.py tts' first to generate test audio")


def main():
    print("\n" + "#" * 60)
    print("  VoiceForge AI - Model Accuracy Report")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode == "tts":
        test_tts()
    elif mode == "stt":
        test_stt()
    else:
        # Test one at a time with cleanup
        test_tts()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        test_stt()

    print("\n" + "#" * 60)
    print("  Test Complete!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
