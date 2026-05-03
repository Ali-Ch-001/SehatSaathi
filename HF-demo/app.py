"""
SehatSaathi (صحت ساتھی) — Live Demo via Ollama
═══════════════════════════════════════════════════════════════════
Multilingual AI health assistant for rural Pakistan.
Built on Gemma 4 E4B, fine-tuned with Unsloth.

Backend: Ollama subprocess (pre-built binary, no compilation)
Model:   Pulled live from Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF (Q4_K_M)

This is the SAME GGUF that runs offline on a $200 Android phone — judges
can `ollama run` the same model on their laptop in 3 commands.

🤗 Spaces:    https://huggingface.co/spaces/Ali-001-ch/SehatSaathi-demo
🤗 GGUF:      https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF
🤗 Merged:    https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b
🤗 LoRA:      https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b-lora

Author: Ali Mohsin · The Gemma 4 Good Hackathon · 2026
"""

from __future__ import annotations

# ───────────────────────────────────────────────────────────────────────────
# Monkey-patch gradio_client.utils to handle bool JSON schemas.
# Bug: https://github.com/gradio-app/gradio/issues/8956
# ───────────────────────────────────────────────────────────────────────────
import gradio_client.utils as _gcu

_orig_get_type = _gcu.get_type
def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)
_gcu.get_type = _safe_get_type

_orig_jstpt = _gcu._json_schema_to_python_type
def _safe_jstpt(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    try:
        return _orig_jstpt(schema, defs)
    except (TypeError, KeyError, AttributeError):
        return "Any"
_gcu._json_schema_to_python_type = _safe_jstpt

import os
import json
import time
import threading
from typing import Iterator, Optional

import gradio as gr
import requests

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

OLLAMA_HOST   = os.environ.get("OLLAMA_HOST",   "127.0.0.1:11434")
OLLAMA_URL    = f"http://{OLLAMA_HOST}"
MODEL_TAG     = os.environ.get("MODEL_TAG",     "hf.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF:Q4_K_M")
# whisper-small has poor Urdu accuracy (~30% WER). Medium is much better (~15%) at the cost of ~3× slower CPU inference.
WHISPER_ID    = os.environ.get("WHISPER_ID",    "openai/whisper-medium")
HF_TOKEN      = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

# Language hints for Whisper. Whisper accepts ISO codes OR English names —
# we use English names because they're more reliable across transformers versions.
WHISPER_LANG_OPTIONS = [
    ("Auto-detect",  None),
    ("Urdu (اردو)",  "urdu"),
    ("English",      "english"),
    ("Hindi",        "hindi"),
    ("Punjabi",      "punjabi"),
    ("Pashto",       "pushto"),   # Whisper uses "pushto" not "pashto"
]

SYSTEM_PROMPT = (
    "You are SehatSaathi (صحت ساتھی), a careful, empathetic community-health assistant "
    "designed for Pakistan and similar low-resource settings. You help Lady Health Workers, "
    "rural patients, and parents understand symptoms and decide what to do next.\n\n"
    "RULES:\n"
    "1. Reply in the SAME language the user wrote in: Urdu in Urdu script, Roman Urdu in Roman Urdu, "
    "English in English. If they code-mix, you may code-mix.\n"
    "2. Always start with one short empathetic sentence.\n"
    "3. Then give 2-4 concrete, low-cost steps (ORS, paracetamol dose by weight, when to keep at home).\n"
    "4. ALWAYS list red-flag symptoms that require going to a hospital or BHU/RHC immediately.\n"
    "5. Never prescribe antibiotics, controlled drugs, or tell anyone to stop a doctor's medication.\n"
    "6. If pregnancy, child <2 months, severe bleeding, breathing difficulty, unconscious, "
    "or seizure → REFER URGENTLY.\n"
    "7. You are an assistant, not a replacement for a qualified doctor."
)

EXAMPLE_PROMPTS = [
    "میرے 2 سال کے بچے کو دو دن سے دست لگ رہے ہیں اور وہ بہت کمزور ہو گیا ہے۔ کیا کروں؟",
    "meray 28 saal k cousin ko 4 din se 103 fever, sar dard, body pain. Lahore mein hain. Dengue hai?",
    "My father has been coughing over a month with blood in sputum, weight loss, night sweats. Could it be TB?",
    "Karachi mein loadshedding hai. 70 saal k waalid garmi mein behosh ho gaye, skin garam aur khushk. Foran kya karoon?",
    "A 50 year old man, sudden chest pain, sweating, pain to left arm, started 30 min ago. We are in Multan.",
    "I have fever and headache",
]


# ──────────────────────────────────────────────────────────────────────
# Ollama lifecycle
# ──────────────────────────────────────────────────────────────────────

_model_pulled = False
_pull_lock    = threading.Lock()
_pull_status  = "idle"


OLLAMA_LOG_PATH = "/tmp/ollama.log"


def _tail_ollama_log(n: int = 40) -> str:
    """Return last n lines of /tmp/ollama.log for surfacing in UI on errors."""
    try:
        with open(OLLAMA_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:]) if lines else "(log empty)"
    except FileNotFoundError:
        return "(no ollama log found at /tmp/ollama.log)"
    except Exception as e:
        return f"(could not read log: {e})"


def _wait_for_ollama(timeout: int = 30) -> bool:
    """Block until ollama serve is responsive."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def _ensure_model_pulled() -> str:
    """Pull GGUF from HF on first invocation. Returns status string."""
    global _model_pulled, _pull_status

    if _model_pulled:
        return "ready"

    with _pull_lock:
        if _model_pulled:
            return "ready"

        if not _wait_for_ollama():
            _pull_status = "error: ollama daemon not responding"
            return _pull_status

        # Check if already pulled
        try:
            tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
            for m in tags.get("models", []):
                if MODEL_TAG.split(":")[0] in m.get("name", ""):
                    print(f"✓ Model already in Ollama cache: {m['name']}")
                    _model_pulled = True
                    _pull_status = "ready"
                    return _pull_status
        except Exception as e:
            print(f"  warning: tag check failed: {e}")

        # Pull it
        print(f"⏬  Pulling {MODEL_TAG} from HuggingFace via Ollama...")
        _pull_status = "downloading"
        try:
            with requests.post(
                f"{OLLAMA_URL}/api/pull",
                json={"model": MODEL_TAG, "stream": True},
                stream=True,
                timeout=900,
            ) as r:
                r.raise_for_status()
                last_pct = -1
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        evt = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    status = evt.get("status", "")
                    if "total" in evt and "completed" in evt and evt["total"]:
                        pct = int(100 * evt["completed"] / evt["total"])
                        if pct != last_pct and pct % 5 == 0:
                            print(f"   {status}: {pct}%")
                            last_pct = pct
                    elif status:
                        print(f"   {status}")

            print(f"✓  Model pulled into Ollama: {MODEL_TAG}")
            _model_pulled = True
            _pull_status = "ready"
        except Exception as e:
            _pull_status = f"error pulling: {type(e).__name__}: {e}"
            print(f"❌ {_pull_status}")
            raise
    return _pull_status


def _preload_ollama_async() -> None:
    """Ensure model is pulled and 'warmed up' in Ollama memory at startup."""
    def _worker():
        try:
            # 1. Ensure the GGUF is downloaded
            status = _ensure_model_pulled()
            if status == "ready":
                # 2. Send a dummy request to force Ollama to load the model into RAM/VRAM
                print(f"🔥 Warming up {MODEL_TAG} in Ollama memory...")
                import requests
                requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": MODEL_TAG,
                        "prompt": "hi",
                        "options": {"num_predict": 1},
                        "stream": False
                    },
                    timeout=120
                )
                print(f"✓ {MODEL_TAG} is warm and ready.")
        except Exception as e:
            print(f"[ollama-preload] Failed to warm up model: {e}")

    t = threading.Thread(target=_worker, daemon=True, name="ollama-preloader")
    t.start()
    print(f"[ollama-preload] Started background pull/warmup in thread {t.name}")


# ──────────────────────────────────────────────────────────────────────
# Whisper for STT (CPU)
# ──────────────────────────────────────────────────────────────────────

_whisper = None


_whisper_lock = threading.Lock()


def _load_whisper():

    global _whisper
    if _whisper is not None:
        return _whisper
    with _whisper_lock:
        if _whisper is not None:
            return _whisper
        print(f"⏬  Loading {WHISPER_ID} for STT (this can take 60-120s on first run)...", flush=True)
        from transformers import pipeline
        try:
            _whisper = pipeline(
                "automatic-speech-recognition",
                model          = WHISPER_ID,
                device         = -1, # CPU
                token          = HF_TOKEN,
                chunk_length_s = 30,
            )
            print("✓  Whisper loaded — voice transcription ready.", flush=True)
        except Exception as e:
            print(f"❌  Whisper load failed: {e}", flush=True)
            raise
    return _whisper


def _preload_whisper_async() -> None:
    """Kick off Whisper download/load in the background while the UI is starting.
    By the time the user records their first clip, the model is warm."""
    def _worker():
        try:
            _load_whisper()
        except Exception as e:
            print(f"[whisper] background preload failed (will retry on demand): {e}", flush=True)
    t = threading.Thread(target=_worker, daemon=True, name="whisper-preloader")
    t.start()
    print(f"[whisper] background preload started in thread {t.name}", flush=True)


def transcribe(audio_path: Optional[str], language: Optional[str] = None) -> str:
    
    if not audio_path:
        return ""

    # Fix: Wait up to 2s for Gradio to finish writing the audio file to disk.
    # This prevents the 'first click fails' bug where the file is 0 bytes initially.
    import os
    for _ in range(4):
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 100:
            break
        time.sleep(0.5)

    try:
        pipe = _load_whisper()
    except Exception as e:
        return f"[Speech model failed to load: {e}]"

    def _run(lang: Optional[str]) -> str:
        gen_kwargs = {"task": "transcribe"}
        if lang:
            gen_kwargs["language"] = lang
        print(f"[whisper] transcribing audio={audio_path}  lang={lang!r}", flush=True)
        out = pipe(audio_path, generate_kwargs=gen_kwargs)
        text = (out.get("text") or "").strip()
        print(f"[whisper] result ({len(text)} chars): {text[:200]!r}", flush=True)
        return text

    try:
        text = _run(language)
        # Retry with auto-detect if forced language failed or returned common noise
        if language and (not text or text.lower() in ("you", "thank you.", ".", "bye.")):
            print(f"[whisper] forced-{language} suspicious; retrying auto-detect", flush=True)
            text = _run(None)
        return text
    except Exception as e:
        print(f"[whisper] ERROR: {e}", flush=True)
        return ""


# ──────────────────────────────────────────────────────────────────────
# Inference via Ollama HTTP API
# ──────────────────────────────────────────────────────────────────────


def generate_stream(
    user_text: str,
    history: list,
    system_prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
) -> Iterator[str]:
    _ensure_model_pulled()

    messages = [{"role": "system", "content": system_prompt}]
    for past_user, past_asst in history:
        if past_user:
            messages.append({"role": "user", "content": past_user})
        if past_asst:
            messages.append({"role": "assistant", "content": past_asst})
    messages.append({"role": "user", "content": user_text})

    accumulated = ""
    try:
        with requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model":    MODEL_TAG,
                "messages": messages,
                "stream":   True,
                "options": {
                    "temperature":    temperature,
                    "top_p":          top_p,
                    "top_k":          top_k,
                    "num_predict":    max_new_tokens,
                },
            },
            stream  = True,
            timeout = 600,
        ) as r:
            if r.status_code != 200:
                # Capture body + tail of ollama log for diagnostics
                body = ""
                try:
                    body = r.text[:1500]
                except Exception:
                    pass
                log_tail = _tail_ollama_log(40)
                accumulated += (
                    f"\n\n⚠️ Inference error: HTTP {r.status_code}\n"
                    f"```\n{body}\n```\n\n"
                    f"**Ollama log (last 40 lines):**\n```\n{log_tail}\n```"
                )
                yield accumulated
                return

            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    evt = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if "error" in evt:
                    accumulated += f"\n\n⚠️ Ollama error: {evt['error']}"
                    yield accumulated
                    return
                delta = evt.get("message", {}).get("content", "")
                if delta:
                    accumulated += delta
                    yield accumulated
                if evt.get("done"):
                    return
    except Exception as e:
        log_tail = _tail_ollama_log(40)
        accumulated += (
            f"\n\n⚠️ Inference error: {type(e).__name__}: {e}\n\n"
            f"**Ollama log (last 40 lines):**\n```\n{log_tail}\n```"
        )
        yield accumulated


# ──────────────────────────────────────────────────────────────────────
# Gradio handlers
# ──────────────────────────────────────────────────────────────────────


def respond(
    user_text: str,
    audio_path: Optional[str],
    chat_history: list,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    voice_lang: Optional[str] = None,
):
    # "auto" sentinel from the dropdown means "let Whisper detect"
    lang_hint = None if (voice_lang in (None, "", "auto")) else voice_lang

    transcribed_via_voice = False

    if audio_path and not (user_text and user_text.strip()):
        # If Whisper hasn't finished preloading yet, show warming message
        if _whisper is None:
            warming = chat_history + [(
                "🎤  *(processing voice…)*",
                "_Warming up the speech model — this takes ~60-90s on first request. "
                "Subsequent voice messages will be near-instant._",
            )]
            yield warming, "", None

        user_text = transcribe(audio_path, language=lang_hint)
        transcribed_via_voice = True

    if not (user_text and user_text.strip()):
        if transcribed_via_voice:
            chat_history = chat_history + [(
                "🎤  *(no speech detected)*",
                "I couldn't hear anything in that audio. Try recording again "
                "clearly or type your question.",
            )]
            yield chat_history, "", None
            return
        yield chat_history, "", None
        return

    # If we transcribed from voice, surface the transcription in the chat so the
    # user can see what Whisper heard (and notice if it heard the wrong thing).
    if transcribed_via_voice:
        lang_label = lang_hint or "auto-detect"
        user_visible = f"🎤  *(transcribed · {lang_label})*  →  {user_text}"
    else:
        user_visible = user_text

    chat_history = chat_history + [(user_visible, "_Loading model on first request… (~60-90s)_")]
    yield chat_history, "", None

    try:
        for partial in generate_stream(
            user_text     = user_text,
            history       = chat_history[:-1],
            system_prompt = system_prompt,
            max_new_tokens = int(max_new_tokens),
            temperature    = float(temperature),
        ):
            chat_history[-1] = (user_visible, partial)
            yield chat_history, "", None
    except Exception as e:
        chat_history[-1] = (user_visible, f"⚠️ Error: {e}")
        yield chat_history, "", None


def clear_chat():
    return [], "", None


# ──────────────────────────────────────────────────────────────────────
# UI (dark theme)
# ──────────────────────────────────────────────────────────────────────

CSS = """
:root, .dark {
    --primary:    #4ade80;
    --primary-2:  #22c55e;
    --accent:     #fbbf24;
    --danger:     #f87171;
    --bg:         #0b1220;
    --surface:    #111827;
    --surface-2:  #1f2937;
    --border:     #334155;
    --text:       #e5e7eb;
    --text-dim:   #9ca3af;
}
html, body, .gradio-container, .gradio-container * { color-scheme: dark !important; }
.gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    min-height: 100vh;
}
.gradio-container .prose, .gradio-container p,
.gradio-container li, .gradio-container span,
.gradio-container label, .gradio-container .label-wrap {
    color: var(--text) !important;
}
#header h1 {
    color: var(--primary);
    margin: 0;
    font-size: 2.4rem;
    letter-spacing: -0.5px;
    text-shadow: 0 0 24px rgba(74, 222, 128, 0.25);
}
#header h2 { color: var(--primary-2); margin: 0; font-weight: 500; font-size: 1.15rem; }
#header h3 { color: var(--text-dim); font-weight: 400; margin-top: 0.2rem; font-size: 0.95rem; }
#tagline   { color: var(--text-dim); max-width: 760px; margin-top: 0.6rem; line-height: 1.6; }
#tagline b { color: var(--text); }
#disclaimer {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.08), rgba(251, 191, 36, 0.04));
    border-left: 4px solid var(--accent);
    border: 1px solid rgba(251, 191, 36, 0.25);
    padding: 12px 16px;
    border-radius: 8px;
    margin: 12px 0;
    font-size: 0.9rem;
    color: #fde68a;
}
#disclaimer b { color: #fbbf24; }
.message.bot, .message.user { font-size: 1.02rem !important; }
.gradio-container .chatbot, .gradio-container [class*='chatbot'] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px;
}
.gradio-container .message.user, .gradio-container [data-testid='user'] {
    background: rgba(34, 197, 94, 0.10) !important;
    border: 1px solid rgba(34, 197, 94, 0.25) !important;
    color: var(--text) !important;
}
.gradio-container .message.bot, .gradio-container [data-testid='bot'] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}
.gradio-container textarea, .gradio-container input[type='text'],
.gradio-container .form, .gradio-container .gr-input,
.gradio-container .block, .gradio-container .panel {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
.gradio-container textarea:focus, .gradio-container input[type='text']:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(74, 222, 128, 0.15) !important;
}
.gradio-container button.primary, .gradio-container button[variant='primary'] {
    background: linear-gradient(135deg, var(--primary-2), #16a34a) !important;
    color: #0b1220 !important;
    font-weight: 600;
    border: none !important;
}
.gradio-container button.primary:hover {
    box-shadow: 0 0 20px rgba(74, 222, 128, 0.35);
    filter: brightness(1.05);
}
.gradio-container button.secondary, .gradio-container button:not(.primary) {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
.gradio-container .examples-table, .gradio-container .accordion {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
.gradio-container .accordion summary { color: var(--text) !important; }
.gradio-container input[type='range'] { accent-color: var(--primary); }
.gradio-container .markdown a { color: var(--primary); }
.gradio-container code {
    background: rgba(74, 222, 128, 0.12) !important;
    color: var(--primary) !important;
    padding: 2px 6px;
    border-radius: 4px;
}
.gradio-container pre {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}
.gradio-container .image-container, .gradio-container .audio-container {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px;
}
footer { color: var(--text-dim) !important; }
"""

DARK_JS = """
() => {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.replace(url.href);
        return;
    }
    document.documentElement.classList.add('dark');
    document.body.classList.add('dark');
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title    = "SehatSaathi · صحت ساتھی",
        css      = CSS,
        js       = DARK_JS,
        theme    = gr.themes.Base(
            primary_hue   = "emerald",
            secondary_hue = "emerald",
            neutral_hue   = "slate",
            font          = [gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ).set(
            body_background_fill          = "#0b1220",
            body_background_fill_dark     = "#0b1220",
            body_text_color               = "#e5e7eb",
            body_text_color_dark          = "#e5e7eb",
            background_fill_primary       = "#111827",
            background_fill_primary_dark  = "#111827",
            background_fill_secondary     = "#1f2937",
            background_fill_secondary_dark= "#1f2937",
            border_color_primary          = "#334155",
            border_color_primary_dark     = "#334155",
            block_background_fill         = "#111827",
            block_background_fill_dark    = "#111827",
            block_border_color            = "#334155",
            block_border_color_dark       = "#334155",
            block_label_background_fill   = "#1f2937",
            block_label_background_fill_dark = "#1f2937",
            block_label_text_color        = "#e5e7eb",
            block_label_text_color_dark   = "#e5e7eb",
            block_title_text_color        = "#4ade80",
            block_title_text_color_dark   = "#4ade80",
            input_background_fill         = "#0f172a",
            input_background_fill_dark    = "#0f172a",
            input_border_color            = "#334155",
            input_border_color_dark       = "#334155",
            button_primary_background_fill         = "linear-gradient(135deg, #22c55e, #16a34a)",
            button_primary_background_fill_dark    = "linear-gradient(135deg, #22c55e, #16a34a)",
            button_primary_text_color              = "#0b1220",
            button_primary_text_color_dark         = "#0b1220",
            button_secondary_background_fill       = "#1f2937",
            button_secondary_background_fill_dark  = "#1f2937",
            button_secondary_text_color            = "#e5e7eb",
            button_secondary_text_color_dark       = "#e5e7eb",
            color_accent_soft             = "rgba(74, 222, 128, 0.15)",
            color_accent_soft_dark        = "rgba(74, 222, 128, 0.15)",
        ),
        analytics_enabled = False,
    ) as demo:
        with gr.Column(elem_id="header"):
            gr.HTML(
                """
                <h1>🩺 SehatSaathi <span style="font-family:'Noto Naskh Arabic',serif;">· صحت ساتھی</span></h1>
                <h2>Your community-health AI for rural Pakistan</h2>
                <h3>Built on Gemma 4 E4B · fine-tuned with Unsloth · runs offline on a phone via GGUF</h3>
                <p id="tagline">
                    Speaks <b>Urdu</b>, <b>Roman Urdu</b>, and <b>English</b>. Gives Pakistan-specific
                    medical guidance with red-flag escalation. <b>This demo runs the same Q4_K_M GGUF
                    that runs offline on a phone</b> via Ollama — same artifact, same speed.
                </p>
                """
            )
            gr.HTML(
                """
                <div id="disclaimer">
                    ⚕️ <b>Important:</b> SehatSaathi is a screening + triage assistant. It is <b>not</b>
                    a replacement for a qualified doctor. In a real medical emergency, call
                    <b>1122 (Pakistan Rescue)</b> or go to the nearest hospital immediately.
                </div>
                """
            )

        chat_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label             = "SehatSaathi",
                    height            = 520,
                    show_copy_button  = True,
                    type              = "tuples",
                )

                with gr.Row():
                    text_input = gr.Textbox(
                        placeholder = "Ask in Urdu, Roman Urdu, or English… ('Bachay ko bukhar hai 102 — kya karoon?')",
                        scale       = 4,
                        show_label  = False,
                        container   = False,
                        lines       = 2,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    audio_input = gr.Audio(
                        label   = "🎤 Speak in your language (Urdu / English)",
                        sources = ["microphone", "upload"],
                        type    = "filepath",
                        scale   = 3,
                    )
                    voice_lang_dd = gr.Dropdown(
                        label   = "🌐 Voice language",
                        choices = [(label, code or "auto") for label, code in WHISPER_LANG_OPTIONS],
                        value   = "urdu",   # default to Urdu since most demo users will be in Pakistan
                        info    = "Force the transcription language — auto-detect mis-identifies short Urdu clips.",
                        interactive = True,
                        scale   = 1,
                    )

                with gr.Row():
                    clear_btn = gr.Button("🗑️  New conversation")

                gr.Examples(
                    examples         = [[p] for p in EXAMPLE_PROMPTS],
                    inputs           = [text_input],
                    label            = "✨  Try these scenarios",
                    examples_per_page = 6,
                )

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️  Settings")
                system_box = gr.Textbox(
                    value      = SYSTEM_PROMPT,
                    lines      = 12,
                    show_label = False,
                    info       = "System prompt — edit live to test variations",
                )
                with gr.Accordion("Generation parameters", open=False):
                    max_tokens_slider  = gr.Slider(64, 1024, value=512, step=32, label="Max new tokens")
                    temperature_slider = gr.Slider(0.1, 1.5, value=1.0, step=0.05, label="Temperature")

                gr.Markdown(f"""
### 🔌  Backend
**Ollama** subprocess · CPU · Q4_K_M GGUF
~5–10 tok/s · same artifact judges can run on their laptop

### 📦  Model
[`{MODEL_TAG.replace('hf.co/', '')}`](https://huggingface.co/{MODEL_TAG.replace('hf.co/', '').split(':')[0]})

### 🌍  Languages
Urdu · Roman Urdu · English · Punjabi · Sindhi · Pashto (best-effort)

### 🧠  Capabilities here
• Multilingual medical Q&A (Urdu / Roman Urdu / English)
• Voice input (Whisper STT — pick the language for best accuracy)
• Pakistan-specific triage with red-flag escalation

### 🖼️  Want vision + image input?
Run the full multimodal model locally:
[`Ali-001-ch/sehatsaathi-gemma4-e4b`](https://huggingface.co/Ali-001-ch/sehatsaathi-gemma4-e4b)
(needs a GPU; this CPU Space only does text + voice)

### 🔋  Run on your laptop / phone

```bash
ollama run hf.co/Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF:Q4_K_M
```

### 📜  License
Apache-2.0 · Gemma Terms · NOT a medical device.

---
Built by [Ali Mohsin](https://www.ali-ch.dev) for the **Gemma 4 Good Hackathon**.
""")

        # ── Event wiring ────────────────────────────────────────────────
        send_btn.click(
            respond,
            inputs  = [text_input, audio_input, chat_state, system_box,
                       max_tokens_slider, temperature_slider, voice_lang_dd],
            outputs = [chatbot, text_input, audio_input],
        ).then(lambda h: h, inputs=chatbot, outputs=chat_state)

        text_input.submit(
            respond,
            inputs  = [text_input, audio_input, chat_state, system_box,
                       max_tokens_slider, temperature_slider, voice_lang_dd],
            outputs = [chatbot, text_input, audio_input],
        ).then(lambda h: h, inputs=chatbot, outputs=chat_state)

        clear_btn.click(clear_chat, outputs=[chatbot, text_input, audio_input])\
                 .then(lambda: [], outputs=chat_state)

    return demo


# ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Pre-warm both models in the background. 
    # Whisper (~1.5GB) and Gemma (~5.5GB) will load while the UI starts up.
    _preload_whisper_async()
    _preload_ollama_async()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1, max_size=20)
    demo.launch(
        server_name = os.environ.get("HOST", "0.0.0.0"),
        server_port = int(os.environ.get("PORT", "7860")),
        share       = os.environ.get("SHARE", "false").lower() == "true",
        show_api    = False,
    )
