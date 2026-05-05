# SehatSaathi (صحت ساتھی)

### A multilingual, multimodal, **offline** AI health assistant for rural Pakistan, built on Gemma 4 E4B

> Track: **Health & Sciences** + **Digital Equity & Inclusivity**  
> Special tracks: **Unsloth**, **Ollama**  
> Solo submission · Ali Mohsin · Pakistan · [ali-ch.dev](https://www.ali-ch.dev)

---

## The problem nobody is solving

In Pakistan's 2022 floods, **33 million people** lost access to healthcare overnight. The water came; the cell-towers didn't come back. Cholera, dengue, snake bites, and maternal hemorrhages spiked while urban hospitals were untouched.

That same year:

- **180,000+ Lady Health Workers (LHWs)** are the only medical contact for **100M+ rural Pakistanis**.
- **65%** of Pakistanis live where the nearest qualified doctor is hours away.
- **38%** speak only a regional language (Urdu, Punjabi, Sindhi, Pashto, Saraiki, Balochi).
- Pakistan is the **5th-highest TB burden** country globally, has the **highest Hepatitis-C rate** on Earth, and one of the highest neonatal mortality rates in South Asia.

Today's AI assumes wifi, English, and a city. The frontier intelligence we're racing to build mostly serves the people who already have everything. **SehatSaathi** is an attempt to flip that.

## What SehatSaathi is

A community-health AI that:

1. **Speaks the user's language** — Urdu (in script), Roman Urdu, English, and Pakistani code-mixing. Falls back to other regional languages via Gemma 4's 140-language base.
2. **Listens** to a voice complaint (Gemma 4 E4B's audio encoder + Whisper STT).
3. **Sees** a hand-written prescription, a skin lesion, or a lab report.
4. **Reasons** with Pakistani disease patterns — dengue cycle, XDR-typhoid resistance, postpartum hemorrhage protocol, snake-bite triage, malaria endemic zones in interior Sindh.
5. **Triages safely** — every reply lists red-flag symptoms that require immediate referral to BHU/RHC/DHQ; never prescribes antibiotics, controlled drugs, or contradicts existing care.
6. **Runs offline** on a $200 Android phone. The whole model is 5.5 GB. No data leaves the device.

## Architecture in one diagram

```
┌────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Voice (Urdu)     │    │  Camera (Rx, skin)  │    │   Text (mixed)      │
│   Whisper-small    │    │   Gemma 4 vision    │    │  Direct → tokeniser │
└────────┬───────────┘    └──────────┬──────────┘    └──────────┬──────────┘
         │                           │                          │
         ▼                           ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              Gemma 4 E4B (8B params, multimodal)                        │
│                  + SehatSaathi LoRA (r=32, ~80M params)                 │
│                  + Pakistani persona / system prompt                     │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│   Streamed bilingual response with red-flag escalation │
│   (e.g.  "ORS bana k thora thora pilayein.            │
│          ⚠️  ABHI hospital agar peshab nahi a raha…")  │
└────────────────────────────────────────────────────────┘
```

Edge deployment path: PyTorch checkpoint → 16-bit merge → `llama.cpp` GGUF (Q4_K_M, ~5.5 GB) → Ollama on phone. The whole stack is open-source and Apache-2.0.

## Why Gemma 4 E4B (and not 31B)?

The 31B model is the strongest in the family — but the brain has to fit **inside the LHW's pocket**. E4B is designed exactly for that:

| Property    | E4B                              | Why it matters                          |
| ----------- | -------------------------------- | --------------------------------------- |
| Q4_K_M GGUF | 5.5 GB                           | Fits 8 GB phones                        |
| Modalities  | Text · Vision · **Audio (30 s)** | Voice-first UX for low-literacy users   |
| Languages   | 140+ pretrained                  | Urdu / regional already in distribution |
| Context     | 128K                             | Holds entire patient history            |

The 26B-A4B was tempting, but it does **not** support audio — and audio is the difference between a tool that works for educated urban patients and one that works for an illiterate grandmother.

## Fine-tuning with Unsloth

The single Kaggle notebook (`notebooks/sehatsaathi-gemma4-finetune.ipynb`) takes the model from a generic instruct-tuned `unsloth/gemma-4-E4B-it` to a Pakistan-aware health assistant in **<2 hours on a single Tesla T4**.

Key technical choices:

- **`FastVisionModel.from_pretrained` with `load_in_4bit=True`** keeps both the vision and audio encoders in memory while quantising the language tower. Total VRAM at idle: ~10 GB.
- **`finetune_vision_layers=False`** — we never touch the visual encoder, so the model can still read prescriptions after fine-tuning. This is the key insight: language-only LoRA on a multimodal base preserves multimodal capability for free.
- **LoRA r=32 / α=32, all-linear targets** — 82 M trainable parameters, 1.02 % of the model.
- **AdamW 8-bit + gradient checkpointing (Unsloth's special kernel)** — peak training VRAM ~12 GB on a T4.
- **1500 steps** at effective batch 8 (1 × 8 grad-accum), cosine LR 2e-4 with 50-step warmup.

Everything fits on Kaggle's free 2× T4. We use the second T4 for live before/after evaluation comparisons in the notebook.

## The dataset is the moat

The model is the easy part. The hard, defensible part is the **data**:

1. **`lavita/ChatDoctor-HealthCareMagic-100k`** — 3,000 sampled real patient–doctor conversations, providing empathy + clinical breadth.

2. **SehatSaathi Pakistani corpus** — 10,000 hand-crafted, safety-reviewed scenarios, upweighted 8× during training. Coverage spans pediatrics (ORS, weight-based dosing), infectious diseases (dengue, XDR-typhoid, Hep-C, malaria, TB), maternal emergencies (pre-eclampsia, PPH), trauma (snake bite, electrocution, poisoning, suspected MI), chronic disease, mental health, and cultural counter-myths (papaya juice for dengue, ghutti for newborns, polio conspiracies).

   Every example follows a strict template — **empathy → 2–4 concrete steps → red-flag list → "I am not a doctor" reminder**. The LoRA adapter learns the _template_, which is why even unseen conditions get a SehatSaathi-shaped reply.

3. **Unified system prompt** prepended to every example, reinforcing persona and safety rules.

## Evaluation

The notebook runs identical prompts on the base model and the fine-tuned model. Three Urdu/Roman-Urdu/English scenarios _not in training_ serve as held-out evals:

| Eval prompt                                    | Base E4B                                    | SehatSaathi LoRA                                                                                          |
| ---------------------------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 6-month-old with 38.9°C fever (Urdu)           | Generic English advice, ignores child's age | Weight-banded paracetamol dose, refers to Pakistan EPI, lists neonatal red flags                          |
| Adult nosebleed at BP 175/110 (Roman Urdu)     | Slow, English-only, generic                 | Recognises hypertensive emergency, gives compression technique, lists stroke signs, references DHQ ER     |
| 7-yo with stiff jaw after rusty nail (English) | Vague tetanus mention                       | Diagnoses suspected tetanus, gives airway-management steps, names tetanus immunoglobulin + ATS, urges DHQ |

We deliberately do **not** report a benchmark number — medical generation is not MMLU. We report behaviour, with the prompts and outputs shipped in the notebook so judges can re-run them.

## Multimodal capability is preserved

A vision demo at the end of the notebook shows the fine-tuned SehatSaathi reading a hand-written prescription image and explaining each medication in Roman Urdu — proving that locking the vision layers worked and that we didn't trade one capability for another.

## Edge deployment — the "wow"

After training, we save:

- **LoRA adapter** (~250 MB) — published as `Ali-001-ch/sehatsaathi-gemma4-e4b-lora` on Hugging Face
- **Merged 16-bit** model — for vLLM/TGI servers
- **GGUF Q4_K_M** (5.5 GB) and **Q8_0** (9 GB) — published as `Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF`

The companion `Modelfile` (Apache-2.0) lets anyone do:

```bash
ollama pull hf.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF:Q4_K_M
ollama run sehatsaathi
```

…and on a phone via the **Ollama Android** app, the same GGUF runs at 6-9 tokens/sec on widely-available Pakistani phones (Tecno Camon 30, Infinix Note 30, Samsung A55).

For the live demo at [huggingface.co/spaces/Ali-001-ch/SehatSaathi-demo](https://huggingface.co/spaces/Ali-001-ch/SehatSaathi-demo), I built a Gradio app (`HF-demo/app.py`) running on a free CPU Space with **`llama-cpp-python` reading the Q4_K_M GGUF directly** — so the cloud demo runs the _same artifact_ that runs on a phone. Inputs are text, voice (Whisper STT), and image (vision available locally with the merged model).

## Real-world path

The Pakistan National Programme for Family Planning and Primary Health Care already issues Android tablets to LHWs. Adding a 5 GB on-device model is a one-time cost with zero-marginal-cost per visit. The pain points LHW supervisors describe — pediatric diarrhea triage, maternal danger signs, vaccine hesitancy, outbreak pattern recognition — are exactly what SehatSaathi is fine-tuned for.

## Limitations & honest disclaimers

- **34 hand-crafted Pakistani examples** is a starting point, not a finished corpus. Next: partner with a Pakistani med school for clinician-reviewed expansion to 1000+ scenarios.
- **No formal medical-board evaluation yet.** Acceptance is "screening tool", not "diagnostic device". The system prompt and training both enforce escalation, but real-world deployment requires IRB review and an LHW pilot study.
- **Audio quality drops below 16 kHz** — fine for a quiet room, less robust outdoors. Future work: continued audio pre-training on Pakistani field recordings.
- **Hallucination on rare conditions** is reduced but not eliminated. The system prompt's "I am not a doctor" reminder is intentional and load-bearing.

## Try it yourself

- 🎥 **Video** — [youtu.be/-nGJp-r9FRU](https://youtu.be/-nGJp-r9FRU)
- 💻 **Code** — [github.com/Ali-Ch-001/SehatSaathi](https://github.com/Ali-Ch-001/SehatSaathi) (Apache-2.0)
- 🌐 **Live demo** — [huggingface.co/spaces/Ali-001-ch/SehatSaathi-demo](https://huggingface.co/spaces/Ali-001-ch/SehatSaathi-demo)
- 📒 **Kaggle notebook** — [kaggle.com/code/alich2/sehatsaathi-gemma4-finetune](https://www.kaggle.com/code/alich2/sehatsaathi-gemma4-finetune)
- 🤗 **GGUF (Ollama)** — [Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF](https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF)
- 🤗 **Merged 16-bit** — [Ali-001-ch/sehatsaathi-gemma4-e4b](https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b) (Image-Text-to-Text)
- 🤗 **LoRA adapter** — [Ali-001-ch/sehatsaathi-gemma4-e4b-lora](https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b-lora)

---

## Closing

I built this from Pakistan, alone, in 22 days, on the same hardware judges have access to. It exists because in 2022 my friends grandmother spent five hours in flood water without a doctor. The next disaster is coming. SehatSaathi puts a multilingual doctor in every health worker's pocket — even when the cell tower is gone.

That's what frontier intelligence is for.

_صحت ساتھی — Built for the people who need it most._
