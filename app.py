import os
import sys
import torch
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.tts_inference import TTSInference
from inference.stt_inference import STTInference

# Initialize global engines
tts_engine = None
stt_engine = None

def load_engines():
    global tts_engine, stt_engine
    
    tts_checkpoint = "checkpoints/tts_kannada/best.pt"
    tts_vocab = "vocab_tts.json"
    
    stt_checkpoint = "checkpoints/stt_kannada/best.pt"
    stt_tokenizer = "data/kannada/stt/tokenizer.model"

    # Only load TTS if checkpoint exists
    if os.path.exists(tts_checkpoint) and os.path.exists(tts_vocab):
        try:
            tts_engine = TTSInference(tts_checkpoint, tts_vocab)
            print("âœ… TTS Engine Loaded!")
        except Exception as e:
            print(f"âš ï¸ Failed to load TTS: {e}")
            
    # Only load STT if checkpoint exists
    if os.path.exists(stt_checkpoint):
        try:
            stt_engine = STTInference(stt_checkpoint, stt_tokenizer)
            print("âœ… STT Engine Loaded!")
        except Exception as e:
            print(f"âš ï¸ Failed to load STT: {e}")

def run_tts(text, speed):
    if tts_engine is None:
        return None, "âš ï¸ TTS Model not loaded! Did you put best.pt in checkpoints/tts_kannada/ ?"
    if not text.strip():
        return None, "Please enter some text."
        
    out_path = "outputs/tts_kannada/demo_output.wav"
    try:
        tts_engine.synthesize(text, out_path, speed=speed)
        return out_path, "âœ… Speech Generated Successfully!"
    except Exception as e:
        return None, f"âŒ Error: {str(e)}"

def run_stt(audio_filepath):
    if stt_engine is None:
        return "âš ï¸ STT Model not loaded! Did you put best.pt in checkpoints/stt_kannada/ ?"
    if not audio_filepath:
        return "Please provide an audio file."
        
    try:
        transcript = stt_engine.transcribe(audio_filepath, beam_width=10)
        return transcript
    except Exception as e:
        return f"âŒ Error: {str(e)}"

# Load the models right away
load_engines()

# Create Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple"), title="VoiceForge AI") as app:
    gr.Markdown("# ðŸŽ™ï¸ VoiceForge AI - Kannada Models")
    gr.Markdown("Text-to-Speech (VITS2) and Speech-to-Text (Conformer-CTC) built from scratch.")
    
    with gr.Tab("ðŸ“ Text to Speech"):
        with gr.Row():
            with gr.Column():
                tts_input = gr.Textbox(lines=5, label="Input text (Kannada/English)")
                tts_speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speech Speed (<1 is faster)")
                tts_btn = gr.Button("Generate Speech", variant="primary")
            with gr.Column():
                tts_audio = gr.Audio(label="Generated Audio", type="filepath")
                tts_status = gr.Markdown()
                
        tts_btn.click(fn=run_tts, inputs=[tts_input, tts_speed], outputs=[tts_audio, tts_status])
        
    with gr.Tab("ðŸ—£ï¸ Speech to Text"):
        with gr.Row():
            with gr.Column():
                stt_audio = gr.Audio(label="Upload or Record Audio", type="filepath")
                stt_btn = gr.Button("Transcribe", variant="primary")
            with gr.Column():
                stt_text = gr.Textbox(lines=5, label="Transcription Output")
                
        stt_btn.click(fn=run_stt, inputs=[stt_audio], outputs=[stt_text])

if __name__ == "__main__":
    os.makedirs("outputs/tts_kannada", exist_ok=True)
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
