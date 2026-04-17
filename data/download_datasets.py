"""
Dataset Download & Preprocessing Script
Downloads and preprocesses datasets for English (Indian accent) and Kannada.
"""

import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
#  Dataset Information & Download Links
# ═══════════════════════════════════════════════════════════

DATASETS = {
    # ── English TTS ──
    
    # ── Kannada TTS ──
    "indictts_kannada": {
        "name": "IndicTTS Kannada (IIT Madras)",
        "type": "tts",
        "language": "kn",
        "source": "huggingface",
        "hf_id": "SPRINGLab/IndicTTS_Kannada",
        "hours": "7.35h",
        "description": "Studio-quality, 48kHz, 1M + 1F speakers",
    },
    "rasa_kannada": {
        "name": "ai4bharat/Rasa (Kannada subset)",
        "type": "tts",
        "language": "kn",
        "source": "huggingface",
        "hf_id": "ai4bharat/Rasa",
        "hours": "variable",
        "description": "Multi-style: neutral, command, conversational, narration",
    },
    # ── Kannada STT ──
    "openslr79_kannada": {
        "name": "OpenSLR SLR79 Kannada",
        "type": "stt",
        "language": "kn",
        "source": "url",
        "urls": [
            "https://openslr.trmal.net/resources/79/kn_in_female.zip",
            "https://openslr.trmal.net/resources/79/kn_in_male.zip",
        ],
        "hours": "~5h",
        "description": "Google multi-speaker, quality-checked, CC-BY-SA-4.0",
    },
}


def list_datasets():
    """Print all available datasets."""
    print("\n" + "=" * 70)
    print("📦 Available Datasets")
    print("=" * 70)

    for key, info in DATASETS.items():
        print(f"\n  📁 {key}")
        print(f"     Name: {info['name']}")
        print(f"     Type: {info['type'].upper()}")
        print(f"     Language: {info['language']}")
        print(f"     Size: {info['hours']}")
        print(f"     Desc: {info['description']}")

    print("\n" + "=" * 70)
    print("\nUsage:")
    print("  python data/download_datasets.py --dataset indictts_kannada --output data/kannada/tts")
    print("  python data/download_datasets.py --dataset indicvoices_r --output data/english/tts")


def download_hf_dataset(hf_id: str, output_dir: str, subset: str = None):
    """Download dataset from Hugging Face."""
    try:
        from datasets import load_dataset

        print(f"\n📥 Downloading {hf_id} from Hugging Face...")
        print(f"   Output: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        kwargs = {"cache_dir": os.path.join(output_dir, "hf_cache")}
        if subset:
            kwargs["name"] = subset

        dataset = load_dataset(hf_id, **kwargs)

        if isinstance(dataset, dict):
            print(f"✅ Downloaded! Splits: {list(dataset.keys())}")
            for split_name, split_data in dataset.items():
                print(f"   {split_name}: {len(split_data)} samples")
        else:
            print(f"✅ Downloaded! {len(dataset)} samples")

        return dataset

    except ImportError:
        print("❌ Error: 'datasets' package not installed.")
        print("   Run: pip install datasets")
        return None
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None


def download_url_dataset(urls: list, output_dir: str):
    """Download dataset files from URLs."""
    import urllib.request
    import zipfile

    os.makedirs(output_dir, exist_ok=True)

    for url in urls:
        filename = url.split("/")[-1]
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"  ⏩ Already exists: {filename}")
            continue

        print(f"  📥 Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"  ✅ Saved: {filepath}")

        # Extract if zip
        if filepath.endswith(".zip"):
            print(f"  📦 Extracting...")
            with zipfile.ZipFile(filepath, "r") as z:
                z.extractall(output_dir)
            print(f"  ✅ Extracted to {output_dir}")


def create_tts_filelist(data_dir: str, output_path: str):
    """Create TTS filelist (audio_path|text) from dataset directory.

    Scans for common formats: LJSpeech metadata, HF datasets, TSV files.
    """
    entries = []

    # Try LJSpeech-style metadata.csv
    metadata_path = os.path.join(data_dir, "metadata.csv")
    if os.path.exists(metadata_path):
        wavs_dir = os.path.join(data_dir, "wavs")
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    uid = parts[0]
                    text = parts[-1]
                    wav_path = os.path.join(wavs_dir, f"{uid}.wav")
                    if os.path.exists(wav_path):
                        entries.append(f"{wav_path}|{text}")

    # Try TSV format
    for tsv in Path(data_dir).glob("*.tsv"):
        with open(tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    audio_path = os.path.join(data_dir, parts[0])
                    text = parts[1]
                    if os.path.exists(audio_path):
                        entries.append(f"{audio_path}|{text}")

    if entries:
        # Split train/val (98/2)
        import random
        random.seed(42)
        random.shuffle(entries)
        split = int(len(entries) * 0.98)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        train_path = output_path.replace(".txt", "_train.txt") if "_train" not in output_path else output_path
        val_path = train_path.replace("train", "val")

        with open(train_path, "w", encoding="utf-8") as f:
            f.write("\n".join(entries[:split]))
        with open(val_path, "w", encoding="utf-8") as f:
            f.write("\n".join(entries[split:]))

        print(f"✅ Created filelists:")
        print(f"   Train: {train_path} ({split} samples)")
        print(f"   Val:   {val_path} ({len(entries) - split} samples)")
    else:
        print(f"⚠️  No data found in {data_dir}")


def create_stt_manifest(data_dir: str, output_path: str):
    """Create STT manifest (JSON lines) from dataset directory."""
    import torchaudio

    entries = []

    # Find all audio files
    audio_extensions = {".wav", ".flac", ".mp3", ".ogg"}
    for root, _, files in os.walk(data_dir):
        for f in files:
            if any(f.lower().endswith(ext) for ext in audio_extensions):
                audio_path = os.path.join(root, f)
                # Look for corresponding text file
                txt_path = os.path.splitext(audio_path)[0] + ".txt"
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as tf:
                        text = tf.read().strip()
                    try:
                        info = torchaudio.info(audio_path)
                        duration = info.num_frames / info.sample_rate
                        entries.append({
                            "audio_filepath": audio_path,
                            "text": text,
                            "duration": round(duration, 2),
                        })
                    except Exception:
                        pass

    if entries:
        import random
        random.seed(42)
        random.shuffle(entries)
        split = int(len(entries) * 0.95)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        train_path = output_path.replace(".json", "_train.json") if "_train" not in output_path else output_path
        val_path = train_path.replace("train", "val")

        for path, data in [(train_path, entries[:split]), (val_path, entries[split:])]:
            with open(path, "w", encoding="utf-8") as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"✅ Created manifests:")
        print(f"   Train: {train_path} ({split} samples)")
        print(f"   Val:   {val_path} ({len(entries) - split} samples)")
    else:
        print(f"⚠️  No paired audio/text data found in {data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare datasets")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset key to download")
    parser.add_argument("--output", type=str, default="data/", help="Output directory")
    parser.add_argument("--list", action="store_true", help="List all available datasets")
    parser.add_argument("--create-filelist", type=str, default=None, help="Create TTS filelist from directory")
    parser.add_argument("--create-manifest", type=str, default=None, help="Create STT manifest from directory")
    args = parser.parse_args()

    if args.list or args.dataset is None:
        list_datasets()
    elif args.create_filelist:
        create_tts_filelist(args.create_filelist, os.path.join(args.output, "train.txt"))
    elif args.create_manifest:
        create_stt_manifest(args.create_manifest, os.path.join(args.output, "train_manifest.json"))
    elif args.dataset in DATASETS:
        info = DATASETS[args.dataset]
        if info["source"] == "huggingface":
            download_hf_dataset(info["hf_id"], args.output, info.get("subset"))
        elif info["source"] == "url":
            download_url_dataset(info["urls"], args.output)
    else:
        print(f"❌ Unknown dataset: {args.dataset}")
        print(f"Available: {', '.join(DATASETS.keys())}")
