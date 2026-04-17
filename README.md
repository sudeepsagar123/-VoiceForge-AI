# 🎙️ VoiceForge AI — TTS & STT from Scratch

> **Text-to-Speech** (VITS2) and **Speech-to-Text** (Conformer-CTC) built from scratch  
> Supports **English (Indian accent)** and **Kannada (all dialects)**

---

## 🏗️ Architecture

| Component | Model | What it does |
|-----------|-------|-------------|
| **Text → Speech** | VITS2 | End-to-end: text directly → waveform (no separate vocoder) |
| **Speech → Text** | Conformer-CTC | Conv + Transformer hybrid with CTC decoding |
| **Long Text** | Sentence splitting + cross-fade | Handles paragraphs of any length |
| **Long Audio** | VAD chunking + overlap merge | Handles hours of audio |

## 📁 Project Structure

```
├── configs/              # YAML configs for all models
├── data/                 # Dataset download & storage
├── preprocessing/        # Text normalization, audio processing, datasets
├── models/
│   ├── tts/              # VITS2 (text encoder, flow, HiFi-GAN, discriminator)
│   └── stt/              # Conformer-CTC (attention, convolution, encoder)
├── training/             # Training loops, losses, utilities
├── inference/            # TTS & STT inference engines
└── requirements.txt
```

## 🚀 Quick Start

### 1. Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Test Architectures (CPU, no data needed)
```bash
# Test VITS2 TTS model
python -m models.tts.vits2

# Test Conformer-CTC STT model  
python -m models.stt.conformer

# Sanity check training scripts
python training/train_tts.py --test-run
python training/train_stt.py --test-run
```

### 3. Download Datasets
```bash
# List all available datasets
python data/download_datasets.py --list

# Download English TTS data
python data/download_datasets.py --dataset indicvoices_r --output data/english/tts

# Download Kannada TTS data
python data/download_datasets.py --dataset indictts_kannada --output data/kannada/tts
```

### 4. Train (when GPU is available)
```bash
# Train English TTS
python training/train_tts.py --config configs/tts_english.yaml

# Train English STT
python training/train_stt.py --config configs/stt_english.yaml

# Train Kannada TTS
python training/train_tts.py --config configs/tts_kannada.yaml

# Train Kannada STT
python training/train_stt.py --config configs/stt_kannada.yaml
```

### 5. Inference
```bash
# Text to Speech
python inference/tts_inference.py --checkpoint checkpoints/tts_english/best.pt \
    --vocab data/english/tts/vocab.json --text "Hello, how are you?"

# Speech to Text
python inference/stt_inference.py --checkpoint checkpoints/stt_english/best.pt \
    --audio recording.wav
```

## 📊 Datasets

### English (Indian Accent)
| Dataset | Size | Use |
|---------|------|-----|
| IndicVoices-R | ~175h | TTS primary |
| Svarah | 9.6h | TTS fine-tuning |
| NPTEL2020 | 15,700h | STT primary |
| Common Voice (en-IN) | 2000+h | STT supplement |

### Kannada (All Dialects)
| Dataset | Size | Use |
|---------|------|-----|
| SYSPIN (IISc) | 58h | TTS primary |
| IndicTTS | 7.35h | TTS secondary |
| IISc-MILE | 350h | STT primary |
| LDC-IL | 179h (4 dialects) | STT dialect coverage |
