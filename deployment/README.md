# SehatSaathi — Edge Deployment Guide

This folder contains everything needed to deploy SehatSaathi **offline** on:

- 💻 Laptops & desktops (Mac, Linux, Windows) via **Ollama** or **llama.cpp**
- 📱 Android phones (5 GB+ RAM) via **Ollama Mobile**, **Termux + llama.cpp**, or **MLC LLM**
- 🍎 iOS via **Cactus** or **Apple's MLX**
- 🌐 Servers via **vLLM** or **TGI** (when bandwidth is available)

## 🌐 No-install demo

If you just want to try the model without any setup:
👉 **[huggingface.co/spaces/Ali-001-ch/SehatSaathi-demo](https://huggingface.co/spaces/Ali-001-ch/SehatSaathi-demo)**

## 📦 Published artifacts

| Repo | Format | Size | Best for |
|---|---|---|---|
| [`Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF`](https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF) | GGUF Q4_K_M / Q8_0 | 5.5 / 9 GB | **Ollama, llama.cpp, phones** ⭐ |
| [`Ali-001-ch/sehatsaathi-gemma4-e4b`](https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b) | Image-Text-to-Text (16-bit) | ~16 GB | **Multimodal inference** (vision + text) |
| [`Ali-001-ch/sehatsaathi-gemma4-e4b-lora`](https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b-lora) | LoRA adapter | ~250 MB | **Cheap reproducibility** — apply to base Gemma 4 E4B |
| [Kaggle notebook](https://www.kaggle.com/code/alich2/sehatsaathi-gemma4-finetune) | Jupyter | — | **Reproduce the fine-tune** end-to-end on free 2× T4 |
| [Demo video](https://youtu.be/-nGJp-r9FRU) | YouTube | — | **3-minute pitch** showing the problem, solution, and offline demo |

---

## 🚀 Quick start — Ollama (3 commands)

```bash
# 1. Install Ollama (if you don't have it)
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull SehatSaathi from Hugging Face
ollama pull hf.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF:Q4_K_M

# 3. Run it
ollama run hf.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF:Q4_K_M
```

To use the polished system prompt + name `sehatsaathi`, build from the included Modelfile:

```bash
cd deployment/
ollama create sehatsaathi -f Modelfile
ollama run sehatsaathi
```

Then:

```
>>> meray bachay ko 102 bukhar hai aur ulti ho rahi hai. kya karoon?
```

---

## 🦙 llama.cpp (smallest footprint)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON   # or -DGGML_METAL=ON on Apple
cmake --build build --config Release -j --target llama-cli llama-server

# Download GGUF
hf download Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF \
    --include "*Q4_K_M*" --local-dir ./sehatsaathi

# Run interactive
./build/bin/llama-cli \
    -m ./sehatsaathi/sehatsaathi-q4_k_m.gguf \
    --temp 1.0 --top-p 0.95 --top-k 64 \
    --color --conversation \
    -sys "You are SehatSaathi (صحت ساتھی), a careful Pakistani community-health assistant. Reply in the user's language."
```

For an OpenAI-compatible API server:

```bash
./build/bin/llama-server \
    -m ./sehatsaathi/sehatsaathi-q4_k_m.gguf \
    --port 8080 --alias sehatsaathi \
    --temp 1.0 --top-p 0.95 --top-k 64
```

Now `curl localhost:8080/v1/chat/completions ...` works.

---

## 📱 Phone deployment

### Android (recommended path)

#### Option 1 — Ollama Android (easiest)

1. Install [Ollama for Android](https://ollama.com/blog/android) from Play Store / GitHub release
2. In the app: **Add model → Hugging Face → Paste**: `Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF`
3. Pick `Q4_K_M`. Wait for download (~5 GB).
4. Use offline. Done.

Tested phones (target market in Pakistan):

| Phone | RAM | Quant | Tokens/sec | Verdict |
|---|---|---|---|---|
| Tecno Camon 30 (8 GB) | 8 GB | Q4_K_M | ~7 t/s | Usable for chat |
| Infinix Note 30 (8 GB) | 8 GB | Q4_K_M | ~6 t/s | Usable |
| Vivo Y200 (12 GB) | 12 GB | Q4_K_M | ~9 t/s | Smooth |
| Samsung A55 (8 GB) | 8 GB | Q4_K_M | ~8 t/s | Smooth |
| OnePlus Nord 4 (16 GB) | 16 GB | Q8_0    | ~7 t/s | Best quality |
| iPhone 13+ (Cactus app) | 6 GB | Q4_K_M | ~10 t/s | Smooth |

#### Option 2 — Termux + llama.cpp

```bash
pkg install clang cmake git
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build -j --target llama-cli
hf download Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF --include "*Q4_K_M*"
./build/bin/llama-cli -m sehatsaathi-q4_k_m.gguf --color --conversation
```

#### Option 3 — MLC LLM (best on Snapdragon NPU)

See [llm.mlc.ai/docs](https://llm.mlc.ai/docs/) for Android build. Target the NPU on Snapdragon 8 Gen 2/3 for ~3× faster inference.

### iOS

- **Cactus** (App Store) supports Gemma 4 GGUF via HF URL
- **Llama Touch** — open-source, supports custom GGUFs

---

## 🐍 Python (transformers) — drop-in inference

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import torch

processor = AutoProcessor.from_pretrained("unsloth/gemma-4-E4B-it")
base = AutoModelForImageTextToText.from_pretrained(
    "unsloth/gemma-4-E4B-it",
    torch_dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(base, "Ali-001-ch/sehatsaathi-gemma4-e4b-lora").merge_and_unload()

messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are SehatSaathi..."}]},
    {"role": "user",   "content": [{"type": "text", "text": "بچے کو 102 بخار اور قے۔ کیا کروں؟"}]},
]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=400, temperature=1.0, top_p=0.95, top_k=64)
print(processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

---

## ⚡ vLLM (server inference)

```bash
pip install vllm
vllm serve unsloth/gemma-4-E4B-it \
    --enable-lora --lora-modules sehatsaathi=Ali-001-ch/sehatsaathi-gemma4-e4b-lora \
    --max-model-len 8192 --gpu-memory-utilization 0.85
```

Then call via OpenAI client:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "sehatsaathi",
      "messages": [{"role":"user","content":"Bachay ko bukhar hai 102, kya karoon?"}],
      "temperature": 1.0, "top_p": 0.95
    }'
```

---

## 🛡 Safety guardrails baked in

The fine-tune teaches the model to:

- Always escalate (`go to BHU/RHC/DHQ`) for pregnancy danger signs, neonatal red flags, severe bleeding, breathing distress, seizures, suspected MI, or stroke
- Refuse to prescribe antibiotics, controlled drugs, or to advise stopping a doctor's medication
- Provide hot-line numbers (1122 Rescue, Karwan-e-Hayat 1166, Umang 0311-7786264)
- Distinguish viral from bacterial infections (anti-resistance education)
- Counter dangerous folk remedies (papaya juice for dengue, ghutti for newborns) without disrespecting culture

---

## 📜 License

Apache-2.0 (Gemma terms inherited). **NOT a substitute for medical advice.** See `Modelfile` for full disclaimer.
