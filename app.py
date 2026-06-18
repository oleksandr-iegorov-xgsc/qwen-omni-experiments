import gradio as gr
import soundfile as sf
import torch
import time
import os
import json
import re
import shutil
import argparse
from datetime import datetime

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from transformers import BitsAndBytesConfig

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

MODEL_CHOICES = {
    "Qwen2.5-Omni-3B": "Qwen/Qwen2.5-Omni-3B",
    "Qwen2.5-Omni-7B": "Qwen/Qwen2.5-Omni-7B",
}

HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history")
HISTORY_JSON = os.path.join(HISTORY_DIR, "history.json")
os.makedirs(HISTORY_DIR, exist_ok=True)

# Default system-prompt file, assumed to sit alongside app.py (override with
# --system-prompt-file). Run the app from its own directory so this resolves.
DEFAULT_SYSTEM_PROMPT_FILE = "prompt_qwen2.5_default.txt"

# Regex patterns for chatty trailing phrases the model likes to append.
# Each pattern matches from the start of a sentence to the end of the string.
_CHATTY_TAIL_PATTERNS = re.compile(
    r'(?:^|\.\s+|\!\s+|\?\s+)'           # sentence boundary before the filler
    r'('
    r'(?:If you (?:have|need|want|would like).*)'
    r'|(?:Feel free to .*)'
    r'|(?:(?:Please )?[Ll]et me know .*)'
    r'|(?:Don\'t hesitate to .*)'
    r'|(?:I\'m here (?:to help|if you).*)'
    r'|(?:Is there anything else .*)'
    r'|(?:(?:I )?[Hh]ope (?:this|that) helps.*)'
    r'|(?:I\'d be happy to .*)'
    r')$',
    re.DOTALL,
)


def strip_chatty_tail(text: str) -> str:
    """Remove trailing 'feel free to ask' style filler sentences."""
    text = text.strip()
    # Try stripping up to 2 trailing filler sentences (they can chain)
    for _ in range(2):
        m = _CHATTY_TAIL_PATTERNS.search(text)
        if m:
            text = text[:m.start(1)].rstrip().rstrip('.').rstrip() + '.'
        else:
            break
    return text.strip()


# Global state for loaded model/processor
loaded_model = {"name": None, "model": None, "processor": None}

# Initial system prompt text, populated at startup from --system-prompt-file
SYSTEM_PROMPT = ""


def load_system_prompt(path):
    """Read the initial system prompt used to seed every conversation."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"System prompt file not found: {path}\n"
            f"Create it or pass a valid path via --system-prompt-file."
        )
    with open(path, "r") as f:
        prompt = f.read().strip()
    if not prompt:
        raise ValueError(f"System prompt file is empty: {path}")
    return prompt


def load_history():
    if os.path.exists(HISTORY_JSON):
        with open(HISTORY_JSON, "r") as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_JSON, "w") as f:
        json.dump(history, f, indent=2)


def load_model_if_needed(model_choice):
    model_name = MODEL_CHOICES[model_choice]
    if loaded_model["name"] == model_name:
        return

    if loaded_model["model"] is not None:
        del loaded_model["model"]
        del loaded_model["processor"]
        torch.cuda.empty_cache()

    # To run 4-bit quantized (lower VRAM, slightly lower quality) uncomment this
    # block together with the `quantization_config=bnb_config` line below.
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    # )

    # torch_dtype is REQUIRED here. bfloat16 is the model's native dtype (~21GB).
    # Without it the model loads in float32 (~42GB), overflows the GPU, and
    # device_map="auto" offloads the talker to CPU/meta -- which then crashes
    # mid-generate with "Tensor.item() cannot be called on meta tensors".
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config,
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    loaded_model["name"] = model_name
    loaded_model["model"] = model
    loaded_model["processor"] = processor


def run_inference(model_choice, input_file, mic_audio, max_new_tokens):
    # Prefer mic recording if provided, otherwise use uploaded file
    if mic_audio is not None:
        audio_data, sr = sf.read(mic_audio)
        # Resample to 16kHz to match model expectations and avoid expensive librosa resampling
        target_sr = 16000
        max_samples = target_sr * 10  # 10-second cap at target sample rate
        if sr != target_sr:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
        sf.write(mic_audio, audio_data, target_sr)
        input_file = mic_audio
    if input_file is None:
        raise gr.Error("Please upload a file or record from your microphone.")

    timings = {}

    # Stage 1: Model loading
    t0 = time.time()
    load_model_if_needed(model_choice)
    timings["1_model_load"] = time.time() - t0

    model = loaded_model["model"]
    processor = loaded_model["processor"]

    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": input_file},
            ],
        },
    ]

    USE_AUDIO_IN_VIDEO = True

    # Stage 2: Audio extraction (librosa.load / process_mm_info)
    t0 = time.time()
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    timings["2_audio_extraction"] = time.time() - t0

    audio_dur = f"{len(audios[0])/16000:.2f}s" if audios else "N/A"

    # Stage 3: Feature extraction (WhisperFeatureExtractor + tokenization)
    t0 = time.time()
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    timings["3_feature_extraction"] = time.time() - t0

    input_tokens = inputs["input_ids"].shape[-1]

    # Stage 4: Model generation (thinker + talker)
    t0 = time.time()
    text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO,
                                     thinker_max_new_tokens=int(max_new_tokens))
    timings["4_model_generate"] = time.time() - t0

    # Stage 5: Decoding + saving
    t0 = time.time()
    decoded_text = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response_text = decoded_text[0] if decoded_text else ""
    # response_text = strip_chatty_tail(response_text)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_{timestamp}.wav"
    output_path = os.path.join(HISTORY_DIR, output_filename)
    sf.write(
        output_path,
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )

    input_basename = os.path.basename(input_file)
    input_hist_name = f"input_{timestamp}_{input_basename}"
    input_hist_path = os.path.join(HISTORY_DIR, input_hist_name)
    shutil.copy2(input_file, input_hist_path)
    timings["5_decode_and_save"] = time.time() - t0

    total = sum(timings.values())
    output_audio_dur = audio.reshape(-1).shape[0] / 24000
    output_text_len = len(response_text.split())

    # Print timing breakdown to terminal
    print(f"\n{'='*50}")
    print(f"TIMING BREAKDOWN ({model_choice})")
    print(f"{'='*50}")
    for stage, t in timings.items():
        pct = (t / total * 100) if total > 0 else 0
        print(f"  {stage:30s} {t:7.2f}s  ({pct:5.1f}%)")
    print(f"  {'TOTAL':30s} {total:7.2f}s")
    print(f"  Input audio: {audio_dur} | Input tokens: {input_tokens}")
    print(f"  Output audio: {output_audio_dur:.2f}s | Output text: {output_text_len} words")
    print(f"{'='*50}\n")

    # Save to history
    history = load_history()
    history.insert(0, {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_choice,
        "input_file": input_hist_name,
        "output_file": output_filename,
        "text_response": response_text,
        "inference_time": round(total, 2),
        "timings": {k: round(v, 2) for k, v in timings.items()},
    })
    save_history(history)

    # Build status with timing breakdown
    timing_lines = " | ".join(f"{k.split('_',1)[1]}: {v:.2f}s" for k, v in timings.items())
    status = (f"Total: {total:.2f}s using {model_choice}\n"
              f"{timing_lines}\n"
              f"Input: {audio_dur} audio, {input_tokens} tokens | "
              f"Output: {output_audio_dur:.1f}s audio, {output_text_len} words")
    return status, response_text, output_path, render_history()


def render_history():
    history = load_history()
    if not history:
        return "<p style='color:#888;'>No inference history yet.</p>"

    rows = []
    for i, entry in enumerate(history):
        input_path = os.path.join(HISTORY_DIR, entry["input_file"])
        output_path = os.path.join(HISTORY_DIR, entry["output_file"])

        # Build clickable file links using Gradio's file serving
        input_link = f'<a href="/file={input_path}" target="_blank">{entry["input_file"]}</a>' if os.path.exists(input_path) else entry["input_file"]
        output_link = f'<a href="/file={output_path}" target="_blank">{entry["output_file"]}</a>' if os.path.exists(output_path) else entry["output_file"]

        text_preview = entry.get("text_response", "")
        if len(text_preview) > 120:
            text_preview = text_preview[:120] + "..."

        rows.append(f"""
        <tr>
            <td>{entry['timestamp']}</td>
            <td><b>{entry['model']}</b></td>
            <td>{input_link}</td>
            <td>{output_link}</td>
            <td>{text_preview}</td>
            <td>{entry['inference_time']}s</td>
        </tr>""")

    table = f"""
    <table style="width:100%; border-collapse:collapse; font-size:14px;">
        <thead>
            <tr style="border-bottom:2px solid #444;">
                <th style="text-align:left; padding:8px;">Time</th>
                <th style="text-align:left; padding:8px;">Model</th>
                <th style="text-align:left; padding:8px;">Input</th>
                <th style="text-align:left; padding:8px;">Output</th>
                <th style="text-align:left; padding:8px;">Text Response</th>
                <th style="text-align:left; padding:8px;">Inference</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    """
    return table


def load_history_entry(evt: gr.SelectData):
    """When a history row is clicked, load that entry's output audio."""
    history = load_history()
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row_idx < 0 or row_idx >= len(history):
        return gr.skip(), gr.skip(), gr.skip()
    entry = history[row_idx]
    output_path = os.path.join(HISTORY_DIR, entry["output_file"])
    audio_val = output_path if os.path.exists(output_path) else None
    return entry.get("text_response", ""), audio_val, f"Loaded history entry from {entry['timestamp']}"


with gr.Blocks(title="Qwen2.5-Omni") as demo:
    gr.Markdown("# Qwen2.5-Omni Inference")

    with gr.Tabs():
        with gr.Tab("Inference"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Radio(
                        choices=list(MODEL_CHOICES.keys()),
                        value="Qwen2.5-Omni-7B",
                        label="Model",
                    )
                    input_file = gr.File(
                        label="Upload Audio/Video",
                        file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a",
                                    ".mp4", ".mov", ".avi", ".mkv", ".webm"],
                    )
                    gr.Markdown("**— or —**")
                    mic_audio = gr.Audio(
                        label="Record from Microphone (max 10s)",
                        sources=["microphone"],
                        type="filepath",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=10, maximum=200, value=20, step=5,
                        label="Response Length (thinker tokens)",
                    )
                    run_btn = gr.Button("Run Inference", variant="primary", size="lg")

                with gr.Column(scale=1):
                    status_box = gr.Textbox(label="Status", interactive=False)
                    response_text = gr.Textbox(
                        label="Text Response", lines=6, interactive=False
                    )
                    output_audio = gr.Audio(
                        label="Output Audio", type="filepath", interactive=False
                    )

        with gr.Tab("History"):
            refresh_btn = gr.Button("Refresh History", size="sm")
            history_html = gr.HTML(value=render_history())
            refresh_btn.click(fn=render_history, outputs=history_html)

    run_btn.click(
        fn=run_inference,
        inputs=[model_choice, input_file, mic_audio, max_tokens_slider],
        outputs=[status_box, response_text, output_audio, history_html],
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-Omni speech-to-speech Gradio app"
    )
    parser.add_argument(
        "--system-prompt-file",
        default=DEFAULT_SYSTEM_PROMPT_FILE,
        help="Path to a text file holding the initial system prompt "
             "(default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    SYSTEM_PROMPT = load_system_prompt(args.system_prompt_file)
    print(f"Loaded system prompt from {args.system_prompt_file} "
          f"({len(SYSTEM_PROMPT)} chars).")

    # Preload default model at startup so first inference is fast
    print("Preloading default model (Qwen2.5-Omni-7B)...")
    load_model_if_needed("Qwen2.5-Omni-7B")
    print("Model ready.")

    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[HISTORY_DIR])
