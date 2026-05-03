#!/usr/bin/env python3
"""
SehatSaathi · One-shot GGUF Conversion & Upload Script
═══════════════════════════════════════════════════════════════════════

Takes your fine-tuned merged Gemma 4 E4B model on HuggingFace and turns it into:

    Ali-001-ch/sehatsaathi-gemma4-e4b-GGUF  (Q4_K_M + Q8_0)

Just paste your token below and run:

    python3 convert_to_gguf.py

What this does (in order):

    1.  Verify/install python deps (huggingface_hub, hf_transfer)
    2.  Clone + build llama.cpp with Metal (Mac) or CUDA (Linux)
    3.  Download the merged-16bit model from HF (~16 GB)
    4.  Convert HF → F16 GGUF (~16 GB)
    5.  Quantize to Q4_K_M (~5.5 GB) and Q8_0 (~9 GB)
    6.  Create the destination GGUF repo on HF
    7.  Upload both quantizations
    8.  Test that Q4_K_M loads successfully
    9.  Clean up intermediate files (optional)

Disk needed:    ~40 GB free
Time on M1/M2:  ~30-60 min total
Time on Linux:  ~20-45 min total
Resumable:      yes — re-run if it crashes mid-way
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path
from textwrap import dedent

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIG — PASTE YOUR HF TOKEN HERE                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

HF_TOKEN = ""   # ← from https://huggingface.co/settings/tokens

HF_USER  = "Ali-001-ch"
SRC_REPO = f"{HF_USER}/sehatsaathi-gemma4-e4b"        # your merged 16-bit
DST_REPO = f"{HF_USER}/sehatsaathi-gemma4-e4b-GGUF"   # gguf output (will be created)

QUANTS = ["Q4_K_M", "Q8_0"]   # which quantizations to produce + upload

WORKSPACE = Path.home() / "sehatsaathi-gguf-workspace"

CLEANUP_AFTER_SUCCESS = True   # delete merged + f16 after upload to free disk
KEEP_QUANTIZED_LOCALLY = True  # keep the q4/q8 files for local Ollama use

# ╔══════════════════════════════════════════════════════════════════════╗

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_RED    = "\033[91m"
C_BOLD   = "\033[1m"
C_END    = "\033[0m"


def step(n: int, total: int, msg: str) -> None:
    bar = "═" * 70
    print(f"\n{C_BOLD}{C_GREEN}{bar}\n  Step {n}/{total} — {msg}\n{bar}{C_END}")


def warn(msg: str) -> None:
    print(f"{C_YELLOW}⚠  {msg}{C_END}")


def err(msg: str) -> None:
    print(f"{C_RED}✗  {msg}{C_END}")


def ok(msg: str) -> None:
    print(f"{C_GREEN}✓  {msg}{C_END}")


def run(cmd, cwd: Path | None = None, check: bool = True,
        env: dict | None = None, stream: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, optionally streaming output live."""
    if isinstance(cmd, str):
        printable = cmd
        shell = True
    else:
        printable = " ".join(str(c) for c in cmd)
        shell = False
    print(f"{C_BOLD}$ {printable}{C_END}")
    if stream:
        return subprocess.run(cmd, cwd=cwd, check=check, env=env, shell=shell)
    return subprocess.run(cmd, cwd=cwd, check=check, env=env, shell=shell,
                          capture_output=True, text=True)


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def assert_token() -> None:
    if HF_TOKEN.startswith("hf_PASTE") or len(HF_TOKEN) < 30:
        err("Edit this file and replace HF_TOKEN with your real token.")
        err("Get one from: https://huggingface.co/settings/tokens (Write scope)")
        sys.exit(1)
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


# ──────────────────────────────────────────────────────────────────────────
# Steps
# ──────────────────────────────────────────────────────────────────────────


def step_1_install_python_deps() -> None:
    step(1, 9, "Installing Python deps (huggingface_hub, hf_transfer)")
    run([sys.executable, "-m", "pip", "install", "-q", "-U",
         "huggingface_hub[cli]>=0.24.0", "hf_transfer", "tqdm"])
    # Conversion script deps
    run([sys.executable, "-m", "pip", "install", "-q",
         "torch", "numpy", "sentencepiece", "protobuf", "transformers", "safetensors"])
    ok("Python deps OK")


def step_2_setup_workspace() -> None:
    step(2, 9, f"Setting up workspace at {WORKSPACE}")
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    free = disk_free_gb(WORKSPACE)
    print(f"  Free disk in {WORKSPACE.parent}: {free:.1f} GB")
    if free < 40:
        warn(f"Only {free:.1f} GB free — you need ~40 GB. Free up space and re-run.")
        if free < 25:
            err("Refusing to start with <25 GB free.")
            sys.exit(1)
    ok("Workspace ready")


def step_3_clone_and_build_llamacpp() -> None:
    step(3, 9, "Clone + build llama.cpp")

    repo = WORKSPACE / "llama.cpp"

    if not repo.exists():
        run(["git", "clone", "--depth", "1",
             "https://github.com/ggml-org/llama.cpp", str(repo)])
    else:
        print(f"  llama.cpp already cloned at {repo}, pulling latest...")
        try:
            run(["git", "-C", str(repo), "pull", "--ff-only"], check=False)
        except Exception:
            warn("git pull failed — using existing checkout")

    quant_bin = repo / "build" / "bin" / "llama-quantize"
    cli_bin   = repo / "build" / "bin" / "llama-cli"

    if quant_bin.exists() and cli_bin.exists():
        ok("llama.cpp already built")
        return

    # Determine GPU backend
    is_mac = platform.system() == "Darwin"
    cmake_flags = ["-DBUILD_SHARED_LIBS=OFF"]
    if is_mac:
        cmake_flags += ["-DGGML_METAL=ON"]
        print("  Building with Metal (Apple GPU) support")
    elif shutil.which("nvcc"):
        cmake_flags += ["-DGGML_CUDA=ON"]
        print("  Building with CUDA support")
    else:
        cmake_flags += ["-DGGML_CUDA=OFF"]
        print("  Building CPU-only")

    if not shutil.which("cmake"):
        err("cmake not found.")
        if is_mac:
            err("Install with: brew install cmake")
        else:
            err("Install with: sudo apt-get install cmake build-essential")
        sys.exit(1)

    run(["cmake", "-B", str(repo / "build"), str(repo)] + cmake_flags)
    run(["cmake", "--build", str(repo / "build"),
         "--config", "Release", "-j", "--clean-first",
         "--target", "llama-quantize", "llama-cli"])

    if not quant_bin.exists():
        err(f"Build succeeded but {quant_bin} not found — bad llama.cpp checkout.")
        sys.exit(1)
    ok(f"Built llama.cpp at {repo / 'build'}")

    # Install python deps for the convert script
    req_file = repo / "requirements.txt"
    if req_file.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)])


def step_4_download_merged_model() -> Path:
    step(4, 9, f"Downloading merged 16-bit model from {SRC_REPO}")
    target = WORKSPACE / "merged-16bit"
    if (target / "config.json").exists():
        # Quick sanity: do we have any safetensors / model files?
        weights = list(target.glob("*.safetensors")) + list(target.glob("model*.bin"))
        if weights:
            print(f"  Found existing model at {target}, skipping download.")
            ok(f"Reusing {len(weights)} weight files at {target}")
            return target

    target.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=SRC_REPO,
        local_dir=str(target),
        max_workers=8,
        token=HF_TOKEN,
    )
    ok(f"Downloaded {SRC_REPO} → {target}")
    return target


def step_5_convert_to_f16_gguf(model_dir: Path) -> Path:
    step(5, 9, "Converting HF → F16 GGUF")
    out = WORKSPACE / "sehatsaathi-f16.gguf"
    if out.exists() and out.stat().st_size > 1_000_000_000:
        print(f"  F16 GGUF already exists at {out}, skipping.")
        ok(f"Reusing {out}")
        return out

    convert_script = WORKSPACE / "llama.cpp" / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        err(f"Conversion script not found at {convert_script}")
        sys.exit(1)

    run([
        sys.executable, str(convert_script),
        str(model_dir),
        "--outfile", str(out),
        "--outtype", "f16",
    ])

    if not out.exists():
        err("F16 conversion failed — output file missing.")
        sys.exit(1)
    ok(f"F16 GGUF saved → {out} ({out.stat().st_size / 1e9:.2f} GB)")
    return out


def step_6_quantize(f16_path: Path) -> dict[str, Path]:
    step(6, 9, f"Quantizing to {' + '.join(QUANTS)}")
    quant_bin = WORKSPACE / "llama.cpp" / "build" / "bin" / "llama-quantize"
    out_paths: dict[str, Path] = {}
    for q in QUANTS:
        target = WORKSPACE / f"sehatsaathi-{q.lower()}.gguf"
        if target.exists() and target.stat().st_size > 1_000_000_000:
            print(f"  {q} already exists at {target}, skipping.")
            out_paths[q] = target
            continue
        run([str(quant_bin), str(f16_path), str(target), q])
        if not target.exists():
            err(f"{q} quantization produced no output file")
            sys.exit(1)
        out_paths[q] = target
        ok(f"{q} → {target} ({target.stat().st_size / 1e9:.2f} GB)")
    return out_paths


def step_7_create_repo() -> None:
    step(7, 9, f"Creating destination repo {DST_REPO}")
    from huggingface_hub import create_repo, HfApi
    api = HfApi(token=HF_TOKEN)
    create_repo(
        repo_id=DST_REPO,
        token=HF_TOKEN,
        repo_type="model",
        exist_ok=True,
        private=False,
    )
    ok(f"Repo ready: https://huggingface.co/{DST_REPO}")

    # Upload a README
    readme = dedent(f"""\
        ---
        license: apache-2.0
        base_model: unsloth/gemma-4-E4B-it
        tags:
          - gemma
          - gemma-4
          - unsloth
          - gguf
          - medical
          - urdu
          - pakistan
          - llama-cpp
          - ollama
        language:
          - ur
          - en
        pipeline_tag: text-generation
        ---

        # SehatSaathi · صحت ساتھی · GGUF (Q4_K_M + Q8_0)

        Quantized GGUF builds of the SehatSaathi LoRA fine-tune of Gemma 4 E4B,
        for offline phone & laptop deployment via **Ollama** or **llama.cpp**.

        - Source merged model: [`{SRC_REPO}`](https://huggingface.co/{SRC_REPO})
        - LoRA adapter only: [`{HF_USER}/sehatsaathi-gemma4-e4b-lora`](https://huggingface.co/{HF_USER}/sehatsaathi-gemma4-e4b-lora)
        - Base: [`unsloth/gemma-4-E4B-it`](https://huggingface.co/unsloth/gemma-4-E4B-it)

        ## Files

        | File | Quant | Size | Use case |
        |---|---|---|---|
        | `sehatsaathi-q4_k_m.gguf` | Q4_K_M | ~5.5 GB | Phones, edge devices |
        | `sehatsaathi-q8_0.gguf`   | Q8_0   | ~9 GB   | Laptops, near-lossless |

        ## Quick start with Ollama

        ```bash
        ollama pull hf.co/{DST_REPO}:Q4_K_M
        ollama run  hf.co/{DST_REPO}:Q4_K_M
        ```

        ## Quick start with llama.cpp

        ```bash
        ./llama-cli -m sehatsaathi-q4_k_m.gguf \\
            --temp 1.0 --top-p 0.95 --top-k 64 --color --conversation
        ```

        ## Disclaimer

        SehatSaathi is a medical screening / triage assistant, NOT a substitute
        for a qualified doctor. In an emergency call **1122 (Pakistan Rescue)**.
        License: Apache-2.0 + Gemma Terms of Use.
        """)
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=DST_REPO,
        token=HF_TOKEN,
    )
    ok("README uploaded")


def step_8_upload(out_paths: dict[str, Path]) -> None:
    step(8, 9, "Uploading GGUFs to HuggingFace")
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    for q, path in out_paths.items():
        size_gb = path.stat().st_size / 1e9
        print(f"  Uploading {path.name} ({size_gb:.2f} GB) — this can take 5-15 min on slow links...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=DST_REPO,
            token=HF_TOKEN,
        )
        ok(f"Uploaded {path.name}")


def step_9_test_and_cleanup(out_paths: dict[str, Path]) -> None:
    step(9, 9, "Smoke-testing Q4_K_M with a simple inference")
    cli_bin = WORKSPACE / "llama.cpp" / "build" / "bin" / "llama-cli"
    q4 = out_paths.get("Q4_K_M")
    if q4 and cli_bin.exists():
        prompt = "Bachay ko 102 bukhar hai aur ulti ho rahi hai. Kya karoon?"
        try:
            run([
                str(cli_bin),
                "-m", str(q4),
                "-p", prompt,
                "-n", "120",          # 120 tokens
                "--temp", "1.0", "--top-p", "0.95", "--top-k", "64",
                "--no-display-prompt",
            ])
            ok("Inference OK — model is functional")
        except Exception as e:
            warn(f"Inference test failed (model still uploaded): {e}")

    if CLEANUP_AFTER_SUCCESS:
        merged = WORKSPACE / "merged-16bit"
        f16    = WORKSPACE / "sehatsaathi-f16.gguf"
        if merged.exists():
            print(f"  Removing {merged} (~16 GB) ...")
            shutil.rmtree(merged, ignore_errors=True)
        if f16.exists():
            print(f"  Removing {f16} (~16 GB) ...")
            f16.unlink(missing_ok=True)
        if not KEEP_QUANTIZED_LOCALLY:
            for path in out_paths.values():
                path.unlink(missing_ok=True)
        ok("Cleanup done")
    else:
        print(f"  CLEANUP_AFTER_SUCCESS=False, leaving files at {WORKSPACE}")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"{C_BOLD}{C_GREEN}")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  SehatSaathi · صحت ساتھی   GGUF Conversion + HF Upload          ║")
    print("║  Source : " + SRC_REPO.ljust(56) + "║")
    print("║  Target : " + DST_REPO.ljust(56) + "║")
    print("║  Quants : " + (", ".join(QUANTS)).ljust(56) + "║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"{C_END}")

    assert_token()

    try:
        step_1_install_python_deps()
        step_2_setup_workspace()
        step_3_clone_and_build_llamacpp()
        merged_dir = step_4_download_merged_model()
        f16_path   = step_5_convert_to_f16_gguf(merged_dir)
        out_paths  = step_6_quantize(f16_path)
        step_7_create_repo()
        step_8_upload(out_paths)
        step_9_test_and_cleanup(out_paths)
    except subprocess.CalledProcessError as e:
        err(f"Command failed (exit {e.returncode}): {e}")
        print()
        print("This script is resumable — fix the issue and re-run.")
        print("Common fixes:")
        print("  • Free more disk space (need ~40 GB)")
        print("  • Re-login: huggingface-cli login")
        print("  • On Mac, install build tools: brew install cmake")
        print("  • On Linux: sudo apt-get install build-essential cmake")
        sys.exit(1)
    except KeyboardInterrupt:
        warn("Interrupted by user. Re-run to continue from where it stopped.")
        sys.exit(130)

    print(f"\n{C_BOLD}{C_GREEN}")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  ✅  ALL DONE                                                    ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Your GGUFs are live at:                                          ║")
    print(f"║  https://huggingface.co/{DST_REPO}".ljust(67) + "║")
    print("║                                                                  ║")
    print("║  Try locally:                                                    ║")
    print(f"║    ollama pull hf.co/{DST_REPO}:Q4_K_M".ljust(67) + "║")
    print(f"║    ollama run  hf.co/{DST_REPO}:Q4_K_M".ljust(67) + "║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"{C_END}\n")


if __name__ == "__main__":
    main()
