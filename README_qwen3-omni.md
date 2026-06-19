# Qwen3-Omni — Speech-to-Speech

> 🚧 **Status: work in progress — not functional end-to-end yet.**
> Everything below describes the *intended* design and the wiring that exists so far.
> The model does not yet run cleanly start-to-finish; treat the run steps as the target,
> not a working recipe. Qwen2.5-Omni (`app.py`) is the working path today.

Gradio app (`app_qwen3_omni.py`) over a standalone load/inference engine
(`qwen3_omni_engine.py`). Audio/video in → text + optional speech out, with a timing
breakdown and run history.

- **App:** `app_qwen3_omni.py` · **Engine:** `qwen3_omni_engine.py` · **Port:** 7861
- **Env:** `.venv3` (transformers **5.12.1** — ships the `Qwen3OmniMoe*` classes that 4.52.x lacks)
- **Model:** `cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit` (default; MoE, 30B total / ~3B active).
  Override with the `QWEN3_OMNI_MODEL` env var.
- **Classes:** `Qwen3OmniMoeForConditionalGeneration` + `Qwen3OmniMoeProcessor`
- **History:** `history_qwen3/` (JSON + 24 kHz output `.wav`s)

## Setup

A provisioned env already exists at `.venv3`. It needs **transformers ≥ 5** (with
`Qwen3OmniMoe*`), plus `accelerate`, `bitsandbytes`, `qwen-omni-utils`, `gradio`,
`librosa`, and `soundfile`. No frozen requirements file ships for this env yet — to create
one:

```bash
.venv3/bin/pip freeze > requirements_qwen3-omni.txt
```

## Run

```bash
export HF_TOKEN=hf_xxx                              # read from env; never hardcoded
export PYTORCH_ALLOC_CONF=expandable_segments:True
.venv3/bin/python app_qwen3_omni.py
```

Open http://localhost:7861. The model preloads at startup.

**UI:** upload/record audio (mic capped at 10 s), pick a **speaker voice**
(Chelsie / Ethan / Aiden), set response length (thinker tokens, default 50), and toggle
**Generate speech output**.

- **Text-only mode:** unchecking the speech toggle disables the Talker and frees ~10 GB
  VRAM — useful when you only need the transcript/answer.

## Hardware / loading notes (single 32 GB GPU)

- Use the **Instruct** checkpoint. The *Thinking* checkpoint is Thinker-only and
  **cannot emit audio**.
- AWQ weights are ~10–11 GB; the Talker adds ~10 GB → ~25–27 GB. Fits 32 GB with little
  slack — keep batch size 1 and the context short.
- The engine tries a **fallback ladder** and prints which strategy loaded:
  - **AWQ:** GPU-only pinned to `cuda:0` (primary) → `device_map="auto"`.
  - **Non-AWQ:** 4-bit (bnb) pinned to cuda:0 → 4-bit GPU+CPU offload → bf16 GPU+CPU
    offload (offload scratch in `offload/`).
- Pinning to `cuda:0` with `low_cpu_mem_usage=False` deliberately avoids accelerate's
  meta-device path (the same `Tensor.item() cannot be called on meta tensors` failure class
  seen with Qwen2.5).
- **Don't** use the staged vLLM-Omni multi-engine path on one GPU — separate CUDA contexts
  blow up VRAM. This engine runs **in-process** by design.

## System prompt

Set in `app_qwen3_omni.py` as a constant
(`"You are Qwen-Omni, a smart voice assistant…"`) and passed to
`engine.generate(system_prompt=…)`. Edit the constant to change it; the engine falls back
to a built-in default if none is supplied.

## HF token

The engine reads `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` from the environment and normalizes
both — it never hardcodes a token. Public weights work without one; gated repos will error
clearly if it's missing.
