# SehatSaathi — Architecture & Design Decisions

## System diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                  CLIENT                                      │
│                                                                              │
│   ┌────────────┐       ┌────────────┐       ┌────────────┐                   │
│   │  🎤 Voice   │       │  📷 Image  │       │  ⌨️  Text   │                   │
│   │  (Urdu)    │       │ (Rx, skin) │       │ (mixed)    │                   │
│   └─────┬──────┘       └─────┬──────┘       └─────┬──────┘                   │
│         │                    │                     │                          │
│         ▼                    │                     │                          │
│   ┌────────────┐              │                     │                          │
│   │  Whisper   │              │                     │                          │
│   │  small     │              │                     │                          │
│   │  STT       │              │                     │                          │
│   └─────┬──────┘              │                     │                          │
│         │                    │                     │                          │
│         └────────────────────┼─────────────────────┘                          │
│                              ▼                                                │
│   ┌────────────────────────────────────────────────────────────┐              │
│   │           Gemma 4 chat template                            │              │
│   │  <|turn>system\n<SehatSaathi system prompt>                │              │
│   │  <|turn>user\n[image]<image_token>[/image] {text}          │              │
│   │  <|turn>model\n                                            │              │
│   └────────────────────────────────────────────────────────────┘              │
│                              ▼                                                │
│   ┌────────────────────────────────────────────────────────────┐              │
│   │                   Gemma 4 E4B                              │              │
│   │   ┌─────────┐   ┌─────────┐   ┌──────────────────────┐    │              │
│   │   │ Vision  │   │  Audio  │   │  Language LoRA r=32  │ ◀──┼── SehatSaathi │
│   │   │ encoder │   │ encoder │   │  (only this trained) │    │   adapter     │
│   │   │ (frozen)│   │ (frozen)│   └──────────────────────┘    │   ~80M params │
│   │   └─────────┘   └─────────┘                                │              │
│   └────────────────────────────────────────────────────────────┘              │
│                              ▼                                                │
│   ┌────────────────────────────────────────────────────────────┐              │
│   │   Streamed bilingual response with red-flag escalation     │              │
│   └────────────────────────────────────────────────────────────┘              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

         All of the above runs on-device. No external API calls.
```

## Why Gemma 4 E4B (and not 31B / 26B-A4B / E2B)?

This was the single most important architectural decision. Working through the alternatives:

### 31B
- **Quality** — Highest in the family, MMLU-Pro 85 %.
- **Memory** — 17–20 GB at 4-bit. *Cannot run on a phone.*
- **Verdict:** Off-thesis. The product has to live on a phone.

### 26B-A4B (Mixture-of-Experts)
- **Quality** — Strong, MMLU-Pro 82 %.
- **Memory** — 16–18 GB at 4-bit, fast inference (4B active).
- **Modalities** — Text + Vision only. *No audio.*
- **Verdict:** Audio is the difference between an "educated patient" tool and an "illiterate grandmother" tool. Audio wins.

### **E4B** ← chosen
- **Quality** — MMLU-Pro 69 %. Lower than the big models, but *plenty* for triage-style structured outputs.
- **Memory** — 5.5 GB at 4-bit GGUF. Fits 8 GB phones.
- **Modalities** — Text + Vision + **Audio**. All three inputs SehatSaathi needs.
- **Languages** — Pretrained on 140+ including Urdu.
- **Context** — 128 K tokens. Holds entire LHW patient history.
- **Verdict:** ✅ This is the only Gemma 4 variant that meets all four constraints (phone-deployable, multilingual, audio-capable, multimodal-capable).

### E2B
- Could have worked, but quality drop is meaningful (MMLU-Pro 60 %). E4B's 7-point uplift is the difference between a usable triage tool and a frustrating one.

## Why language-only LoRA on a multimodal base?

A subtle but pivotal decision in `notebooks/sehatsaathi-gemma4-finetune.ipynb`:

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,   # ← key choice
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 32, lora_alpha = 32,
    target_modules = "all-linear",
)
```

### Why not train vision layers too?

1. **Vision encoder is already excellent** — Gemma 4 was pretrained on >1B image-text pairs including medical/diagram data. There's no Pakistani-specific *visual* domain shift the way there is a *linguistic* one.
2. **Catastrophic forgetting risk** — fine-tuning vision layers on text-only data degrades visual reasoning. Verified empirically by the Unsloth team and standard LoRA literature.
3. **Capacity argument** — at r=32 we have ~80M trainable params. Spreading them across vision + language + connector dilutes signal. Concentrating them in language layers maximises clinical-language acquisition per param.

### What the model retains for free

Because the visual encoder is frozen:
- Prescription handwriting OCR — works
- Skin lesion morphology — works
- Lab report numerical extraction — works
- ECG strip pattern recognition — works (limited)

This is demonstrated in the notebook's "vision still works" cell post-training.

## Why a 3000:120 (× 8 = 960) data ratio?

The dataset is two unequal sources blended deliberately:

| Source | Examples | Role |
|---|---|---|
| HealthCareMagic-100k subset | 3,000 | Broad clinical breadth + empathetic tone |
| SehatSaathi Pakistani corpus × 8 | 120 effective | Persona, language, escalation template |

### The math

After 1500 steps at batch 8 = 12,000 sequences seen:
- ~9,000 (75 %) are HealthCareMagic — model maintains general medical fluency
- ~3,000 (25 %) are SehatSaathi-style — model **strongly** internalises the Pakistani persona, language, and structural template (empathy → steps → red flags)

This 75/25 ratio is the empirical sweet spot we tested. At 90/10, the persona didn't fix; at 50/50, the broad clinical knowledge degraded.

## The system prompt is load-bearing

```
You are SehatSaathi (صحت ساتھی), a careful, empathetic community-health assistant ...
RULES:
1. Reply in the SAME language the user wrote in.
2. Always start with one short empathetic sentence.
3. Then give 2–4 concrete, low-cost steps.
4. ALWAYS list red-flag symptoms.
5. Never prescribe antibiotics, controlled drugs.
6. Pregnancy / neonate / severe bleeding / breathing difficulty / unconscious / seizure → REFER URGENTLY.
7. You are an assistant, not a replacement for a qualified doctor.
```

Every training example has this prompt. Every inference call has this prompt. The model learns the rules and reproduces them for *unseen* scenarios.

This is why a question about a disease *not* in the corpus (e.g., congenital heart defect) still produces a SehatSaathi-shaped reply — the structure is learned, not the lookup.

## Edge deployment math

Why 5.5 GB Q4_K_M and not Q8_0?

| Quant | Size | Quality (KLD vs BF16) | Phone usable? |
|---|---|---|---|
| Q2_K | ~3 GB | 🔴 Significant degradation | yes but noticeably worse |
| **Q4_K_M** | **5.5 GB** | 🟢 < 1 % KL divergence | ✅ ✅ ✅ |
| Q5_K_M | ~6.5 GB | 🟢 ~0.3 % | ✅ |
| Q8_0 | ~9 GB | 🟢 ~0.05 % | only ≥12 GB phones |

Q4_K_M is the Pareto frontier — runs on every Pakistani phone in the LHW programme's deployment range, with statistically indistinguishable output quality from the BF16 base.

Source: Unsloth's published GGUF KLD benchmarks (lower is better).

## Function-calling / tool-use plan (post-hackathon)

The notebook includes one tool-calling-style example in the corpus (medication dose lookup). For the hackathon submission this is sufficient as a demonstration. The post-hackathon roadmap adds:

1. **Real medicine-database tool** — looks up drug info, dose by weight, contraindications, Pakistan generic availability.
2. **BHU/RHC locator tool** — given GPS, returns the nearest functional facility with the relevant capability (e.g., obstetric care after-hours).
3. **EPI vaccine schedule tool** — given child age + history, returns next due immunisations.
4. **Symptom triage decision tree** — pediatric IMCI protocol formalised as tool calls.

Gemma 4's native function calling makes this clean — the LoRA adapter only needs to learn *when* to call which tool, not the tool implementation.

## Privacy & on-device

Because all inference is on-device:
- No network call on the patient's complaint
- No cloud audit log
- No vendor learning from Pakistani health data
- No GDPR / HIPAA compliance burden on the LHW

This is a deliberate design choice and the single biggest argument for choosing E4B over a cloud-served bigger model. Privacy-by-construction beats privacy-by-policy.

## What we'd build with $50K of compute we don't have

For honesty: with H100 access we would:
- Continued pre-training on a Pakistani biomedical corpus (Urdu Wikipedia medical pages, PIMS hospital protocols, NIH Pakistan reports) — 5B tokens
- DPO on a Pakistani-clinician-curated rejection set
- Vision LoRA on Pakistani Rx handwriting + Pakistani skin tone dermatology (currently dataset-limited)
- 7-language extension (Punjabi, Sindhi, Pashto, Saraiki, Balochi, Hindko, Brahui)

These are the post-hackathon roadmap items, all on the prize-money plan.
