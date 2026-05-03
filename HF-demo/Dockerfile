# SehatSaathi · صحت ساتھی   HF Space (Docker SDK)
# Base image: official `ollama/ollama` — Ollama is already installed.
# We layer Python + Gradio on top.
# ─────────────────────────────────────────────────────────────────────────────

FROM ollama/ollama:latest

# Override the upstream image's ENTRYPOINT (it auto-starts `ollama serve`)
ENTRYPOINT []

# ── System tools + Python 3.10 ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        curl ca-certificates ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── HF Spaces requires UID 1000 ─────────────────────────────────────────────
RUN if id -u 1000 >/dev/null 2>&1; then \
        groupmod -n user $(id -gn 1000); \
        usermod -l user $(id -un 1000); \
        usermod -d /home/user -m user; \
    else \
        useradd -m -u 1000 user; \
    fi
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    OLLAMA_MODELS=/home/user/.ollama/models \
    OLLAMA_HOST=127.0.0.1:11434 \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PORT=7860 \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# ── Install Python deps (cached unless requirements.txt changes) ─────────────
COPY --chown=user:user requirements.txt .
RUN pip install --user --no-cache-dir --break-system-packages --upgrade pip setuptools wheel \
    && pip install --user --no-cache-dir --break-system-packages -r requirements.txt

# ── App code ────────────────────────────────────────────────────────────────
COPY --chown=user:user app.py .

# ── Pre-create writable dirs ────────────────────────────────────────────────
RUN mkdir -p /home/user/.ollama/models /home/user/.cache/huggingface

EXPOSE 7860

# ── Start ollama serve in background, then the Gradio app ───────────────────
CMD ["bash", "-c", "ollama serve > /tmp/ollama.log 2>&1 & echo 'ollama PID '$!; sleep 4; exec python3 app.py"]
