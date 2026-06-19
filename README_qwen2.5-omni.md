# Qwen2.5-Omni — Speech-to-Speech

Gradio app (`app.py`) for Qwen2.5-Omni. Audio/video in → text + speech out, with a
per-stage timing breakdown and run history.

- **App:** `app.py` · **Port:** 7860 · **Env:** `.venv` (transformers 4.52.3, torch 2.10.0, gradio 6.9.0)
- **Models:** `Qwen/Qwen2.5-Omni-3B`, `Qwen/Qwen2.5-Omni-7B` (default **7B**, switchable in the UI)
- **Classes:** `Qwen2_5OmniForConditionalGeneration` + `Qwen2_5OmniProcessor`
- **History:** `history/` (JSON + 24 kHz output `.wav`s)

## Setup

Uses [uv](https://docs.astral.sh/uv/) for the environment and dependencies:

```bash
git clone https://github.com/oleksandr-iegorov-xgsc/qwen-omni-experiments.git
cd qwen-omni-experiments
uv venv .venv
uv pip install -r requirements_qwen2.5-omni.txt
```

(An existing `.venv` is already provisioned, so this is informational.)

## Run

```bash
# default system prompt (prompt_qwen2.5_default.txt)
uv run app.py

# alternate persona prompt
uv run app.py --system-prompt-file prompt_qwen2.5_tier1.txt
```

Open http://localhost:7860. The 7B model preloads at startup (~21 GB bf16), so the first
inference is fast.

**UI:** pick the model, **upload** audio/video *or* **record** from the mic (capped at
10 s, resampled to 16 kHz), set response length (thinker tokens, 10–200, default 20), then
Run. The **History** tab lists past runs with playable input/output links.

Accepted uploads: `.wav .mp3 .flac .ogg .m4a .mp4 .mov .avi .mkv .webm`.

## Hardware / loading notes

- The model loads with **`torch_dtype=torch.bfloat16`** and `device_map="auto"`.
  **This dtype is required.** Without it the model loads in float32 (~42 GB), overflows a
  32 GB card, and `device_map="auto"` offloads the Talker to CPU/meta — which then crashes
  mid-generate with `RuntimeError: Tensor.item() cannot be called on meta tensors`.
- **4-bit (optional):** uncomment the `BitsAndBytesConfig` block and the
  `quantization_config=bnb_config` line in `load_model_if_needed()` for lower VRAM at
  slightly lower quality. (`app_4_bit.py` is a standalone 4-bit variant.)

## System prompt

`--system-prompt-file PATH` seeds every conversation (default `prompt_qwen2.5_default.txt`).
The file is read once at startup and must exist and be non-empty. Two prompts ship:

- **`prompt_qwen2.5_default.txt`** — the canonical "You are Qwen…" prompt.
  **Recommended:** Qwen2.5-Omni's speech generation was trained against it, so it gives the
  most reliable audio output.
- **`prompt_qwen2.5_tier1.txt`** — a custom Tier-1 support-agent persona (with a subtle
  curling/hockey easter egg). Fun for review, but straying from the canonical prompt can
  degrade speech output — listen for it.

> Changing the prompt requires a restart to take effect (it is loaded at launch, not per request).
