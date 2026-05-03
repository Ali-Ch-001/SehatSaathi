#!/usr/bin/env python3
"""
Deploy SehatSaathi demo to Hugging Face Space
─────────────────────────────────────────────────────────────
Pushes HF-demo/app.py, HF-demo/requirements.txt, and HF-demo/README.md to:

    https://huggingface.co/spaces/Ali-001-ch/SehatSaathi-demo

Setup once:
    1. Paste your HF token below (or export HF_TOKEN env var)
    2. Run:  python3 deploy_space.py

The script will:
    • Create the Space if it doesn't exist (Gradio SDK, ZeroGPU hardware)
    • Upload the three required files
    • Print the live URL

You can re-run this anytime to push updates.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ╔══════════════════════════════════════════════════════════════╗
# ║ CONFIG — paste your HF token (or set HF_TOKEN env var)       ║
# ╚══════════════════════════════════════════════════════════════╝

HF_TOKEN  = os.environ.get("HF_TOKEN") or ""

HF_USER   = "Ali-001-ch"
SPACE_ID  = f"{HF_USER}/SehatSaathi-demo"

# Hardware tiers (set what you can afford):
#   "cpu-basic"  — FREE, 16 GB RAM, 2 vCPU. Works with the Q4_K_M GGUF (~5-10 tok/s).
#   "cpu-upgrade"— $0.03/hr, 32 GB RAM, 8 vCPU. Faster (~15-25 tok/s).
#   "t4-small"   — $0.40/hr, Nvidia T4 16 GB. Fastest, lets you swap in the merged model.
#   "zero-a10g"  — Free with HF Pro ($9/mo). H200 per-request, fastest possible.
#
# You can also leave it unset (None) and pick manually in the Space settings UI.
HARDWARE  = os.environ.get("SPACE_HARDWARE", "cpu-basic")

# Files to upload (relative to repo root)
DEMO_DIR = Path(__file__).parent / "HF-demo"
FILES_TO_UPLOAD = {
    DEMO_DIR / "Dockerfile":       "Dockerfile",
    DEMO_DIR / "app.py":           "app.py",
    DEMO_DIR / "requirements.txt": "requirements.txt",
    DEMO_DIR / "README.md":        "README.md",
}

# ─────────────────────────────────────────────────────────────


def main() -> None:
    # Sanity check on the token
    if HF_TOKEN.startswith("hf_PASTE") or len(HF_TOKEN) < 30:
        print("✗  Paste your HF token at the top of this file (or set HF_TOKEN env var).")
        print("   Get one from: https://huggingface.co/settings/tokens (Write scope)")
        sys.exit(1)

    # Ensure huggingface_hub is installed
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Installing huggingface_hub ...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub[cli]"])
        from huggingface_hub import HfApi, create_repo

    api = HfApi(token=HF_TOKEN)

    # 1. Create Space if needed (Docker SDK).
    # Note: if Space already exists with sdk=gradio, it will switch to docker
    # automatically when we push a new README.md with sdk: docker.
    print(f"📦  Ensuring Space exists: {SPACE_ID}")
    try:
        create_repo(
            repo_id     = SPACE_ID,
            token       = HF_TOKEN,
            repo_type   = "space",
            space_sdk   = "docker",
            exist_ok    = True,
            private     = False,
        )
        print("    ✓ Space ready (Docker SDK)")
    except Exception as e:
        print(f"    ✗ create_repo failed: {e}")
        sys.exit(1)

    # 2. Set hardware tier
    print(f"⚙️  Requesting hardware: {HARDWARE}")
    try:
        api.request_space_hardware(repo_id=SPACE_ID, hardware=HARDWARE, token=HF_TOKEN)
        print(f"    ✓ Hardware set to {HARDWARE}")
    except Exception as e:
        msg = str(e)
        if "Subscribe to PRO" in msg or "ZeroGPU" in msg:
            print(f"    ⚠ {HARDWARE} requires HF Pro ($9/mo). Falling back to cpu-basic.")
            try:
                api.request_space_hardware(repo_id=SPACE_ID, hardware="cpu-basic", token=HF_TOKEN)
                print("    ✓ Hardware set to cpu-basic (FREE)")
            except Exception as e2:
                print(f"    ⚠ Could not set hardware ({e2}); using Space's default.")
        else:
            print(f"    ⚠ Could not set hardware ({e}); using Space's default.")

    # 3. Upload the demo files
    print(f"⬆️  Uploading {len(FILES_TO_UPLOAD)} files to {SPACE_ID} ...")
    for local, remote in FILES_TO_UPLOAD.items():
        if not local.exists():
            print(f"    ✗ Missing local file: {local}")
            sys.exit(1)
        size_kb = local.stat().st_size / 1024
        print(f"    • {local.name} ({size_kb:.1f} KB) → {remote}")
        api.upload_file(
            path_or_fileobj = str(local),
            path_in_repo    = remote,
            repo_id         = SPACE_ID,
            repo_type       = "space",
            token           = HF_TOKEN,
            commit_message  = "Update SehatSaathi demo",
        )

    print()
    print("═══════════════════════════════════════════════════════════════════")
    print("✅  DONE — your live demo will be available at:")
    print()
    print(f"    https://huggingface.co/spaces/{SPACE_ID}")
    print()
    print("    The Space takes ~3-5 minutes to build on first deploy.")
    print("    Watch the build at the URL above; once ready, click 'App'.")
    print()
    print("    First request may take 30-60s while Gemma 4 loads on ZeroGPU.")
    print("═══════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
