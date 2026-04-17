#!/bin/bash
set -e

echo "============================================================"
echo "🚀 VoiceForge AI - Kannada Models Setup & Training Script"
echo "============================================================"

# Check if NVIDIA GPU is active
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️ WARNING: nvidia-smi not found. Training will be slow on CPU."
fi

# 1. Install Dependencies
echo "------------------------------------------------------------"
echo "📦 Installing Dependencies..."
echo "------------------------------------------------------------"
pip install -r requirements.txt
pip install datasets sentencepiece  # specifically needed for downloads/tokenization

# 2. Download Kannada Datasets
echo "------------------------------------------------------------"
echo "📥 Downloading Kannada Datasets..."
echo "------------------------------------------------------------"
# TTS data format: text | wav
python data/download_datasets.py --dataset indictts_kannada --output data/kannada/tts
# Extract HF Parquet to WAVs and create train.txt / val.txt
python data/extract_hf.py --hf_id SPRINGLab/IndicTTS_Kannada --output data/kannada/tts

# STT data format: wav -> jsonl
python data/download_datasets.py --dataset openslr79_kannada --output data/kannada/stt
python data/download_datasets.py --create-manifest data/kannada/stt

# 3. Train Kannada TTS (VITS2)
echo "------------------------------------------------------------"
echo "🗣️ Starting Kannada TTS Training..."
echo "------------------------------------------------------------"
# Run for a background process (using nohup) or standard
echo "To run TTS Training:"
echo "nohup python training/train_tts.py --config configs/tts_kannada.yaml > logs/tts_kannada.out 2>&1 &"

# 4. Train Kannada STT (Conformer-CTC)
echo "------------------------------------------------------------"
echo "📝 Starting Kannada STT Training..."
echo "------------------------------------------------------------"
echo "To run STT Training:"
echo "nohup python training/train_stt.py --config configs/stt_kannada.yaml > logs/stt_kannada.out 2>&1 &"

echo "============================================================"
echo "✅ Setup script completed. Check the commands above to start training."
echo "Use 'tail -f logs/tts_kannada.out' to monitor progress."
