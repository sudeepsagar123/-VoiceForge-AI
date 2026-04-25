"""
VoiceForge AI — Lightweight Test Interface
Tests TTS OR STT individually to avoid RAM issues.
Usage: python test_app.py tts   OR   python test_app.py stt
"""

import os
import sys
import torch
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_tts_test():
    """Interactive TTS test — type text, hear speech."""
    from inference.tts_inference import TTSInference
    
    print("\n" + "=" * 50)
    print("  TTS Test — Type Kannada/English text")
    print("  Type 'quit' to exit")
    print("=" * 50)
    
    engine = TTSInference("checkpoints/tts_kannada/best.pt", "vocab_tts.json")
    os.makedirs("outputs/tts_test", exist_ok=True)
    
    count = 0
    while True:
        text = input("\n  Enter text: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
            
        count += 1
        out_path = f"outputs/tts_test/output_{count:03d}.wav"
        try:
            engine.synthesize(text, out_path, speed=1.0)
            print(f"  Audio saved: {out_path}")
            print(f"  Playing audio...")
            # Auto-play on Windows
            os.system(f'start "" "{os.path.abspath(out_path)}"')
        except Exception as e:
            print(f"  Error: {e}")


def run_stt_test():
    """Interactive STT test — provide audio file path, get transcription."""
    from inference.stt_inference import STTInference
    
    print("\n" + "=" * 50)
    print("  STT Test — Provide audio file path")
    print("  Type 'quit' to exit")
    print("=" * 50)
    
    engine = STTInference("checkpoints/stt_kannada/best.pt", tokenizer_path=None)
    
    # Auto-test with TTS output if available
    test_files = [f for f in os.listdir("outputs/tts_test") if f.endswith(".wav")] if os.path.exists("outputs/tts_test") else []
    if test_files:
        print(f"\n  Found {len(test_files)} TTS-generated files for auto-test:")
        for f in test_files[:3]:
            path = os.path.join("outputs/tts_test", f)
            try:
                transcript = engine.transcribe(path, beam_width=5)
                print(f"    {f} -> '{transcript[:60]}'")
            except Exception as e:
                print(f"    {f} -> Error: {e}")
    
    while True:
        path = input("\n  Audio file path (or 'quit'): ").strip().strip('"')
        if path.lower() in ("quit", "exit", "q"):
            break
        if not os.path.exists(path):
            print(f"  File not found: {path}")
            continue
        try:
            transcript = engine.transcribe(path, beam_width=5)
            print(f"  Transcription: {transcript}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    
    if mode == "tts":
        run_tts_test()
    elif mode == "stt":
        run_stt_test()
    else:
        print("\nUsage:")
        print("  python test_app.py tts   - Test Text-to-Speech")
        print("  python test_app.py stt   - Test Speech-to-Text")
