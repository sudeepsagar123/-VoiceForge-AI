import os
import argparse
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

def extract_hf_audio(hf_id, output_dir):
    print(f"📦 Extracting {hf_id} to raw wav files...")
    os.makedirs(output_dir, exist_ok=True)
    wavs_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    
    # Load dataset from cache
    dataset = load_dataset(hf_id, cache_dir=os.path.join(output_dir, "hf_cache"))
    
    # Check if it has a train split
    split = "train" if "train" in dataset else list(dataset.keys())[0]
    data = dataset[split]
    
    # Prepare filelist
    filelist_path = os.path.join(output_dir, "train.txt")
    
    audio_col = None
    text_col = None
    
    # Auto-detect column names
    for col in data.column_names:
        if col in ["audio", "speech", "wav"]:
            audio_col = col
        if col in ["text", "sentence", "transcript"]:
            text_col = col

    if not audio_col or not text_col:
        print(f"❌ Could not auto-detect audio/text columns. Found: {data.column_names}")
        return

    print(f"✅ Found audio column: '{audio_col}' | text column: '{text_col}'")
    
    with open(filelist_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(tqdm(data, desc="Saving wavs")):
            try:
                # Some datasets store relative path in audio dict, or raw array
                audio_data = item[audio_col]
                text = item[text_col].strip().replace("\n", " ")
                
                # Write audio array to wav
                wav_name = f"sample_{i:06d}.wav"
                wav_path = os.path.join(wavs_dir, wav_name)
                
                if "array" in audio_data and "sampling_rate" in audio_data:
                    sf.write(wav_path, audio_data["array"], audio_data["sampling_rate"])
                elif "path" in audio_data and os.path.exists(audio_data["path"]):
                    # Sometimes it's already a local path, just copy it
                    import shutil
                    shutil.copy(audio_data["path"], wav_path)
                else:
                    continue # Skip if unrecognized format
                
                # Write to train.txt
                f.write(f"data/kannada/tts/wavs/{wav_name}|{text}\n")
            except Exception as e:
                continue

    # Create val.txt (last 2%)
    with open(filelist_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    split_idx = int(len(lines) * 0.98)
    with open(filelist_path, "w", encoding="utf-8") as f:
        f.writelines(lines[:split_idx])
    with open(os.path.join(output_dir, "val.txt"), "w", encoding="utf-8") as f:
        f.writelines(lines[split_idx:])

    print(f"✅ Finished extracting! Created train.txt and val.txt with {len(lines)} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_id", required=True, help="Hugging Face Dataset ID")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    
    extract_hf_audio(args.hf_id, args.output)
