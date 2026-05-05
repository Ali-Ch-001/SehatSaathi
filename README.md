# 🩺 SehatSaathi · صحت ساتھی

### A multilingual, multimodal, **offline** AI health assistant for rural Pakistan, built on Gemma 4 E4B.

> Solo submission to **The Gemma 4 Good Hackathon** (closes May 19, 2026).
> Targeting Main + Health & Sciences + Digital Equity + Unsloth + Ollama tracks.

|                    |                                                                                                                               |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| 🎥 Video            | [youtu.be/-nGJp-r9FRU](https://youtu.be/-nGJp-r9FRU)                                                                          |
| 🌐 Live demo       | [huggingface.co/spaces/Ali-001-ch/SehatSaathi-demo](https://huggingface.co/spaces/Ali-001-ch/SehatSaathi-demo)                |
| 📒 Kaggle notebook | [kaggle.com/code/alich2/sehatsaathi-gemma4-finetune](https://www.kaggle.com/code/alich2/sehatsaathi-gemma4-finetune)          |
| 🤗 GGUF (Ollama)   | [Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF](https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF) — 8B, Text Generation |
| 🤗 Merged 16-bit   | [Ali-001-ch/sehatsaathi-gemma4-e4b](https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b) — Image-Text-to-Text            |
| 🤗 LoRA adapter    | [Ali-001-ch/sehatsaathi-gemma4-e4b-lora](https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b-lora)                       |
| 💻 Source code     | [github.com/Ali-Ch-001/SehatSaathi](https://github.com/Ali-Ch-001/SehatSaathi) (Apache-2.0)                                   |
| 👤 Author          | [Ali Mohsin](https://www.ali-ch.dev) · Pakistan · solo                                                                        |

---

## TL;DR

In Pakistan's 2022 floods, **33 million people lost healthcare access**. The next disaster is coming. SehatSaathi puts a culturally-grounded, multilingual community-health doctor in every Lady Health Worker's pocket — even when the cell tower is gone.

- Speaks Urdu (script), Roman Urdu, English, code-mixed.
- Sees a hand-written prescription, a skin lesion, a lab report.
- Listens to an Urdu voice complaint.
- Triages with Pakistani disease patterns (dengue, XDR-typhoid, postpartum hemorrhage, snake bite).
- Always lists red-flag symptoms requiring urgent referral; never prescribes antibiotics or controlled drugs.
- Runs offline on a $200 Android phone — 5.5 GB total.

Built on **Gemma 4 E4B** + **Unsloth** LoRA fine-tune, deployed via **Ollama** / **llama.cpp** GGUF.

---

## Repository layout

```
SehatSaathi/
├── notebooks/
│   └── sehatsaathi-gemma4-finetune.ipynb     ← Kaggle notebook (the technical centerpiece)
├── data/
│   └── sehatsaathi_pakistani_corpus_extended.jsonl  ← curated multilingual medical corpus
├── HF-demo/
│   ├── app.py                                 ← Gradio Space (text + voice, GGUF inference)
│   ├── requirements.txt
│   └── README.md                              ← Space metadata
├── deployment/
│   ├── Modelfile                              ← Ollama recipe (Q4_K_M GGUF)
│   └── README.md                              ← edge-deploy guide (phones, llama.cpp, vLLM)
├── docs/
│   ├── WRITEUP.md                             ← Kaggle writeup (~1500 words)
│   ├── VIDEO_SCRIPT.md                        ← 3-minute video storyboard
│   ├── ARCHITECTURE.md                        ← system diagram + design rationale
│   └── MODEL_CARD.md                          ← HF-style model card
├── convert_to_gguf.py                         ← one-shot Mac GGUF builder + HF uploader
├── deploy_space.py                            ← one-shot HF Space deployer
└── README.md
```

---

## Quick start

### 1. Try the live demo (no install)

👉 **[huggingface.co/spaces/Ali-001-ch/SehatSaathi-demo](https://huggingface.co/spaces/Ali-001-ch/SehatSaathi-demo)**

Free CPU Space running the same Q4_K_M GGUF that runs on a phone. First request takes ~60-90 sec to load the model; subsequent requests are fast.

### 2. Run the trained model locally with Ollama

```bash
# install ollama, pull from HF, run
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull hf.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF:Q4_K_M
ollama run hf.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF:Q4_K_M
```

### 3. Run the local Gradio app yourself

```bash
cd HF-demo/
pip install -r requirements.txt
python app.py
```

Open http://localhost:7860 — text, voice (Urdu STT), and image inputs.

### 4. Reproduce the fine-tune (Kaggle 2× T4, free tier)

The original notebook lives at:

👉 **[kaggle.com/code/alich2/sehatsaathi-gemma4-finetune](https://www.kaggle.com/code/alich2/sehatsaathi-gemma4-finetune)**

Or upload `notebooks/sehatsaathi-gemma4-finetune.ipynb` from this repo to your own Kaggle:

1. Settings → Accelerator: `T4 × 2`, Internet: ON
2. Add Input → search `sehatsaathi-pakistani-corpus` (the dataset)
3. Add-ons → Secrets → `HF_TOKEN` (Write scope)
4. Run All — finishes in ~90-120 min
5. LoRA adapter auto-pushes to your HF account

### 5. Convert to GGUF on your Mac (one-shot script)

```bash
python3 convert_to_gguf.py    # paste your HF token at the top first
```

Downloads merged 16-bit, builds llama.cpp with Metal, quantizes to Q4_K_M + Q8_0, uploads to HF. ~45 min total.

---

## What's in the notebook

1. **Setup** — Unsloth + transformers 5.5 + timm
2. **Load Gemma 4 E4B** in 4-bit with `FastVisionModel`
3. **LoRA config** — _language-only_ (vision tower frozen → multimodal preserved)
4. **Build dataset**:
   - 3,000 examples sampled from `lavita/ChatDoctor-HealthCareMagic-100k`
   - 17 hand-crafted Pakistani scenarios (dengue, TB, PPH, snake bite, etc.) × 8 upweight
   - Unified SehatSaathi system prompt
5. **Apply Gemma 4 chat template** (`get_chat_template`)
6. **Pre-training eval** — three Urdu/Roman-Urdu held-out prompts (qualitative baseline)
7. **Train** — 1500 steps, effective batch 8, AdamW-8bit + cosine LR, ~90 min on T4
8. **Post-training eval** — same prompts, dramatic improvement
9. **Vision still works** — feed a prescription image, get a Roman-Urdu explanation
10. **Save**: LoRA adapter → `merged_16bit` → GGUF Q4_K_M / Q8_0 → push to HF
11. **Ollama recipe** — Modelfile + 3-line deploy

---

## Tracks targeted

| Prize                         | Track          | How SehatSaathi qualifies                                                                   |
| ----------------------------- | -------------- | ------------------------------------------------------------------------------------------- |
| 🥇 **$50K Main**              | overall vision | The complete story: refugee/flood story → real fine-tune → real on-device deployment        |
| 🩺 **$10K Health & Sciences** | impact         | Pakistani disease patterns, LHW-tested triage, 4 of the 5 leading Pakistani killers covered |
| 🌍 **$10K Digital Equity**    | impact         | Urdu, Roman Urdu, English; voice-first for low-literacy; offline = $0/month inference       |
| 🦥 **$10K Unsloth**           | tech           | This entire fine-tune trained in 90 min on a free Kaggle T4                                 |
| 🦙 **$10K Ollama**            | tech           | Modelfile + GGUF recipe + Android-tested edge deployment                                    |

Total addressable: **$90K**.

---

## Honest limitations

- 34 hand-crafted Pakistani scenarios is a _starting point_, not a finished corpus. Next: clinician-reviewed expansion to 1000+ scenarios (need partnership with a Pakistani med school).
- No formal medical-board evaluation yet. SehatSaathi positions itself as a **screening / triage assistant**, never a diagnostic device.
- Audio works best in quiet rooms (16 kHz+). Field-noise robustness needs continued audio pre-training.
- Model is open-source so anyone can prompt-inject to disable safety. Production deployment must layer guardrails (content filter, refusal classifier) on top.

These are real, addressable limitations. They'll be in the post-hackathon roadmap.

---

## Acknowledgements

- **Google DeepMind** — Gemma 4 E4B, the only major open multimodal+audio model that fits a phone.
- **[Daniel Han](https://github.com/danielhanchen) and the Unsloth team** — `FastVisionModel`, free Kaggle notebook template, and 2× faster fine-tuning.
- **lavita/ChatDoctor team** — HealthCareMagic-100k dataset.
- **Pakistan's Lady Health Worker programme** — 180,000+ women whose work SehatSaathi tries to amplify, not replace.
- **My grandmother** — the reason this exists.

---

## License

Apache-2.0 (Gemma terms inherited).

Trained model weights and adapter: Apache-2.0 with the standard Gemma Use Restrictions (no medical-device claims, no non-consensual personal data, etc.).

**Not a medical device. Not a substitute for a doctor. In a real emergency call 1122 (Pakistan Rescue) or your local emergency number immediately.**

---

_صحت ساتھی — Built for the people who need it most._
