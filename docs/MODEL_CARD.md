---
license: apache-2.0
library_name: peft
base_model: unsloth/gemma-4-E4B-it
tags:
  - gemma
  - gemma-4
  - unsloth
  - lora
  - peft
  - medical
  - healthcare
  - urdu
  - multilingual
  - pakistan
  - text-generation-inference
language:
  - ur
  - en
  - pa
  - sd
  - ps
pipeline_tag: image-text-to-text
datasets:
  - lavita/ChatDoctor-HealthCareMagic-100k
  - Ali-001-ch/sehatsaathi-pakistani-corpus
---

# SehatSaathi · صحت ساتھی · Gemma 4 E4B LoRA

A culturally-grounded, multilingual community-health AI for rural Pakistan, built on Gemma 4 E4B and fine-tuned with Unsloth.

> "From the floods in Sindh to the mountains of Gilgit, every Lady Health Worker should have a doctor in her pocket — even when the cell tower is gone."

## TL;DR

- **Base:** [`unsloth/gemma-4-E4B-it`](https://huggingface.co/unsloth/gemma-4-E4B-it) — 8B params, multimodal (text + vision + audio)
- **Adapter:** LoRA r=32 / α=32, language-only (vision + audio towers untouched)
- **Trained on:** 3,000 patient–doctor conversations (English) + 120 hand-crafted Pakistani scenarios (Urdu / Roman Urdu / English / code-mixed)
- **Hardware:** Single Tesla T4, ~90 minutes
- **Use case:** Triage assistant for community health workers, rural patients, parents
- **Phone-deployable:** companion GGUF [`Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF`](https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF) is 5.5 GB Q4_K_M

## What it does well

- ✅ Replies in user's language (Urdu / Roman Urdu / English / code-mixed)
- ✅ Pakistani disease patterns (dengue, XDR-typhoid, hepatitis, malaria, TB, snake bite, heat stroke)
- ✅ Maternal & neonatal red flags
- ✅ Empathy-first response template
- ✅ Always lists when to escalate to BHU / RHC / DHQ
- ✅ Refuses to prescribe antibiotics or controlled drugs
- ✅ Multimodal preserved: vision works for prescriptions, lab reports, skin

## What it does NOT do

- ❌ Diagnose definitively
- ❌ Replace a qualified doctor
- ❌ Handle non-Pakistani drug brand names well (use generic names)
- ❌ Process audio reliably below 16 kHz / in noisy field conditions
- ❌ Resist adversarial prompt injection that tries to bypass safety

## Quick start

### Live demo (no install)

👉 **[huggingface.co/spaces/Ali-001-ch/SehatSaathi-demo](https://huggingface.co/spaces/Ali-001-ch/SehatSaathi-demo)** — runs the same Q4_K_M GGUF that runs on a phone.

### Ollama (recommended, runs offline on your laptop)

```bash
ollama pull hf.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF:Q4_K_M
ollama run hf.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF:Q4_K_M
```

### Python (transformers + PEFT)

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import torch

processor = AutoProcessor.from_pretrained("unsloth/gemma-4-E4B-it")
base = AutoModelForImageTextToText.from_pretrained(
    "unsloth/gemma-4-E4B-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "Ali-001-ch/sehatsaathi-gemma4-e4b-lora").merge_and_unload()

SYSTEM = "You are SehatSaathi (صحت ساتھی), a careful, empathetic community-health assistant for Pakistan. Reply in the user's language, give 2-4 concrete steps, list red flags, never prescribe antibiotics."

messages = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
    {"role": "user",   "content": [{"type": "text", "text": "میرے بچے کو دو دن سے دست لگ رہے ہیں، کیا کروں؟"}]},
]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=400, temperature=1.0, top_p=0.95, top_k=64, do_sample=True)
print(processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

### Unsloth (4-bit, fastest local)

```python
from unsloth import FastVisionModel
model, processor = FastVisionModel.from_pretrained(
    "Ali-001-ch/sehatsaathi-gemma4-e4b-lora",
    load_in_4bit = True,
)
FastVisionModel.for_inference(model)
```

## Recommended inference parameters

These match Google's official Gemma 4 defaults:

| Param | Value |
|---|---|
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | 64 |
| repetition_penalty | 1.0 |
| max_new_tokens | 400-600 |

Lower temperature (0.5–0.7) gives more conservative, doctor-like responses but loses some empathy. Tune to taste.

## Training details

| Hyperparameter | Value |
|---|---|
| LoRA rank | 32 |
| LoRA alpha | 32 |
| LoRA dropout | 0 |
| Target modules | all-linear (language layers only) |
| Vision/audio towers | frozen |
| Sequence length | 2048 |
| Batch size (effective) | 8 (1 × 8 grad-accum) |
| Optimiser | AdamW 8-bit |
| LR / scheduler | 2e-4 / cosine, 50 warmup |
| Training steps | 1500 |
| Wall-clock | ~90 min on Tesla T4 |
| Trainable params | 82M / 8.07B = 1.02% |
| Peak VRAM | ~12 GB |

## Evaluation

We do not report a benchmark number — medical generation is qualitative, and benchmarks like MMLU don't capture cultural fluency or escalation behaviour. The companion notebook ([`Ali-Ch-001/SehatSaathi`](https://github.com/Ali-Ch-001/SehatSaathi)) contains 3 held-out Urdu / Roman-Urdu / English prompts and shows the dramatic before/after for each.

For a representative sample, see the [Kaggle notebook](https://www.kaggle.com/code/alich2/sehatsaathi-gemma4-finetune) and the live [HF Space demo](https://huggingface.co/spaces/Ali-001-ch/SehatSaathi-demo).

## Bias, limitations, safety

- The Pakistani corpus is intentionally curated to counter common dangerous folk myths (papaya juice for dengue platelets, ghutti for newborns, polio drops conspiracies). Model may push back on those even when user is sincere.
- The model is fluent in *Pakistani* Urdu and Roman Urdu specifically. Indian Hindi/Urdu users may notice register differences.
- The 34 hand-crafted Pakistani examples are a starting point; an outbreak of a regional disease not in the corpus (e.g., Crimean-Congo Hemorrhagic Fever) will be handled less specifically than dengue.
- Model is open-source, and a determined attacker can prompt-inject around the safety rules. Production deployment should layer guardrails (refusal classifier, content filter) on top.
- Gender/cultural bias review: spot-checks show the model treats male and female complaints with equivalent care and uses gender-neutral language for unspecified patients. No formal disparity audit yet.

## Citation

```bibtex
@misc{sehatsaathi2026,
  author       = {Ali Ch},
  title        = {{SehatSaathi}: A Multilingual Community-Health AI for Rural Pakistan},
  year         = {2026},
  howpublished = {Hugging Face},
  url          = {https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b-lora},
  note         = {Submission to The Gemma 4 Good Hackathon}
}
```

## Acknowledgements

- Google DeepMind — Gemma 4 family
- Unsloth team — `FastVisionModel` and free Kaggle notebooks
- lavita team — HealthCareMagic-100k dataset
- Pakistan's Lady Health Workers — for inspiration

## Disclaimer

**This model is not a medical device. It is not a substitute for professional medical advice, diagnosis, or treatment. In case of medical emergency, call 1122 (Pakistan Rescue) or your local emergency number immediately.**

License: Apache-2.0, with Gemma Terms of Use applying to the base model weights.
