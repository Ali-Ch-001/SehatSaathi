---
title: SehatSaathi · صحت ساتھی
emoji: 🩺
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: true
license: apache-2.0
short_description: Offline AI health assistant for rural Pakistan (Urdu)
suggested_hardware: cpu-basic
models:
  - Ali-001-ch/sehatsaathi-gemma4-e4b
  - Ali-001-ch/sehatsaathi-gemma4-e4b-lora
  - Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF
tags:
  - gemma
  - gemma-4
  - unsloth
  - urdu
  - medical
  - pakistan
  - multilingual
  - llama-cpp
  - gguf
  - ollama
---

# 🩺 SehatSaathi · صحت ساتھی

### Multilingual, multimodal AI health assistant for rural Pakistan

SehatSaathi is **Gemma 4 E4B fine-tuned with Unsloth** to act as a culturally-grounded, multilingual community-health triage assistant for Pakistan and similar low-resource settings.

> **This Space runs the same Q4_K_M GGUF that runs offline on a $200 Android phone.** What you experience here is exactly what a Lady Health Worker would see in a flood-zone clinic with no internet — that's the whole point.

This Space lets judges and users **try the model live** with two input modes:

- 🗣️ **Text** — ask in Urdu, Roman Urdu, or English (or code-mixed)
- 🎤 **Voice** — speak in Urdu / English (auto-transcribed via Whisper)
- 📷 **Image** — uploaded but flagged: vision needs the full multimodal model locally (`Ali-001-ch/sehatsaathi-gemma4-e4b`), not the GGUF.

Submission to **The Gemma 4 Good Hackathon** by [Ali Ch](https://www.ali-ch.dev).

## What's special

- Pakistani disease patterns: dengue, XDR-typhoid, hepatitis, malaria, TB, snake bite, heat stroke, postpartum hemorrhage
- Replies in the user's language, never English-only
- Always lists red-flag symptoms for hospital referral
- Refuses to prescribe antibiotics or controlled drugs
- **Same artifact runs offline on a phone** via the GGUF Q4_K_M build (~5.5 GB)

## Project links

| Resource | Link |
|---|---|
| 💻 Source code | [github.com/Ali-Ch-001/SehatSaathi](https://github.com/Ali-Ch-001/SehatSaathi) |
| 🎥 Demo video  | [youtu.be/-nGJp-r9FRU](https://youtu.be/-nGJp-r9FRU) |
| 📒 Kaggle training notebook | [kaggle.com/code/alich2/sehatsaathi-gemma4-finetune](https://www.kaggle.com/code/alich2/sehatsaathi-gemma4-finetune) |
| 🤗 GGUF (this Space uses) | [`Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF`](https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF) — 8B, Text Generation |
| 🤗 Merged 16-bit | [`Ali-001-ch/sehatsaathi-gemma4-e4b`](https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b) — Image-Text-to-Text |
| 🤗 LoRA adapter | [`Ali-001-ch/sehatsaathi-gemma4-e4b-lora`](https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b-lora) |

## Run offline on your machine

```bash
ollama pull hf.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF:Q4_K_M
ollama run  hf.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF:Q4_K_M
```

## ⚕️ Disclaimer

SehatSaathi is **NOT** a substitute for a qualified doctor. In a real medical emergency call **1122 (Pakistan Rescue)** or your local emergency number immediately. Use this tool as triage support only.

License: Apache-2.0 · base model under Gemma Terms of Use.
