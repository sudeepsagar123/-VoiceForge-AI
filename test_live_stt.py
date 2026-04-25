"""
VoiceForge AI — Live Microphone STT Test
Records from your microphone and transcribes using the Conformer-CTC model.
"""

import os
import sys
import wave
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def record_audio(filename="outputs/mic_recording.wav", duration=5, sample_rate=16000):
    """Record audio from microphone."""
    import pyaudio
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    p = pyaudio.PyAudio()
    
    print(f"\n  [MIC] Recording for {duration} seconds... SPEAK NOW!")
    print("  " + "=" * 40)
    
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=sample_rate, input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    for i in range(0, int(sample_rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
        # Progress bar
        progress = int((i / (sample_rate / CHUNK * duration)) * 40)
        print(f"\r  [{'#' * progress}{'.' * (40-progress)}] {i * CHUNK / sample_rate:.1f}s", end="")
    
    print(f"\r  [{'#' * 40}] {duration}s - DONE!")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"  Saved: {filename}")
    return filename


def main():
    print("\n" + "#" * 50)
    print("  VoiceForge AI — Live STT Test")
    print("#" * 50)
    
    # Load STT engine
    print("\n  Loading Conformer-CTC model...")
    from inference.stt_inference import STTInference
    engine = STTInference("checkpoints/stt_kannada/best.pt", tokenizer_path=None)
    print("  Model loaded!\n")
    
    while True:
        print("\n  Options:")
        print("    1. Record from microphone (5 seconds)")
        print("    2. Record from microphone (10 seconds)")
        print("    3. Transcribe an audio file")
        print("    4. Quit")
        
        choice = input("\n  Choose (1/2/3/4): ").strip()
        
        if choice == "1":
            audio_path = record_audio(duration=5)
        elif choice == "2":
            audio_path = record_audio(duration=10)
        elif choice == "3":
            audio_path = input("  Enter audio file path: ").strip().strip('"')
            if not os.path.exists(audio_path):
                print(f"  File not found: {audio_path}")
                continue
        elif choice in ("4", "q", "quit"):
            print("\n  Goodbye!")
            break
        else:
            print("  Invalid choice!")
            continue
        
        # Transcribe
        print("\n  Transcribing...")
        start = time.time()
        try:
            transcript = engine.transcribe(audio_path, beam_width=5)
            elapsed = time.time() - start
            print(f"\n  {'=' * 50}")
            print(f"  TRANSCRIPTION: {transcript}")
            print(f"  {'=' * 50}")
            print(f"  Time: {elapsed:.2f}s")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
