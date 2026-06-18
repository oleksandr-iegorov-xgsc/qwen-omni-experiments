# Omni Speech-to-Speech Bench

A small Gradio workbench for **reviewing and comparing two speech-to-speech multimodal
models** side by side: **Qwen2.5-Omni** and **Qwen3-Omni**. Speak (or upload audio/video)
→ the model replies with text **and** synthesized speech. Every run prints a per-stage
timing breakdown and is saved to a browsable history.

The two models have different dependencies (notably different `transformers` versions),
so each runs from its **own virtual environment and app**, and has its own README:

| Model | App | Per-model README | venv | Port |
|---|---|---|---|---|
| Qwen2.5-Omni (3B / 7B) | `app.py` | [README_qwen2.5-omni.md](README_qwen2.5-omni.md) | `.venv` | 7860 |
| Qwen3-Omni-30B-A3B 🚧 | `app_qwen3_omni.py` | [README_qwen3-omni.md](README_qwen3-omni.md) | `.venv3` | 7861 |

> 🚧 **Qwen3-Omni is a work in progress — it does not run end-to-end yet.**
> Qwen2.5-Omni is the working path today; the Qwen3-Omni docs describe the intended design.

Both are intended to run at once (different ports), but together they will not fit a single
32 GB GPU — review them one at a time unless you have the VRAM.

## Requirements (shared)

- Linux + NVIDIA GPU with ~32 GB VRAM (developed on an RTX 5090).
- CUDA-enabled PyTorch (torch 2.10 / cu128 in `.venv`).
- A Hugging Face token in the environment (recommended; required for any gated weights).
  Never hardcode it in source:
  ```bash
  export HF_TOKEN=hf_xxx
  ```
- `PYTORCH_ALLOC_CONF=expandable_segments:True` helps on tight VRAM. The apps set it,
  but exporting it before launch is more reliable (the allocator may read it at import).

## Quick start

```bash
# Qwen2.5-Omni  ->  http://localhost:7860  (recommended: run with uv)
uv run app.py

# Qwen3-Omni    ->  http://localhost:7861   (🚧 work in progress, not functional yet)
export HF_TOKEN=hf_xxx
.venv3/bin/python app_qwen3_omni.py
```

See each per-model README for setup, options, and gotchas.

## How they compare

| | Qwen2.5-Omni | Qwen3-Omni |
|---|---|---|
| Architecture | 3B / 7B dense | 30B-A3B MoE (~3B active) |
| Default precision | bf16 (native, ~21 GB for 7B) | AWQ 4-bit (~10–11 GB) |
| Speech out (Talker) | always on | toggle (off saves ~10 GB) |
| Voices | model default | Chelsie / Ethan / Aiden |
| System prompt | file via `--system-prompt-file` | in-app constant |

## Repo layout

- `app.py` — Qwen2.5-Omni Gradio app (port 7860)
- `prompt_qwen2.5_default.txt` — canonical Qwen system prompt (recommended for reliable speech)
- `prompt_qwen2.5_tier1.txt` — alternate "Tier-1 support agent" persona prompt
- `requirements_qwen2.5-omni.txt` — frozen deps for the `.venv` (Qwen2.5) environment
