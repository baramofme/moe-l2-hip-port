#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# moe-l2 one-click installer
# Run large MoE models on consumer NVIDIA GPUs (8GB VRAM).
#
#   curl -fsSL https://raw.githubusercontent.com/yalun753/moe-l2/main/scripts/install.sh | bash
#
# Steps:
#   1. check system (Linux x86_64)
#   2. check NVIDIA GPU + driver
#   3. check Python >= 3.10
#   4. install moe-l2 (PyPI) + download prebuilt GPU binaries
#   5. optional: download demo model (~11.5GB)
#   6. run `moe-l2 doctor` self-check
# ─────────────────────────────────────────────────────────────
set -euo pipefail

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; NC=$'\033[0m'
info()  { echo -e "${GREEN}[moe-l2]${NC} $*"; }
warn()  { echo -e "${YELLOW}[moe-l2]${NC} $*"; }
fail()  { echo -e "${RED}[moe-l2]${NC} $*" >&2; exit 1; }

info "moe-l2 one-click installer — large MoE on consumer GPUs"
info "--------------------------------------------------------"

# ── 1. system ──
info "Checking system..."
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) info "  OK: Linux x86_64" ;;
    Linux-aarch64) fail "ARM Linux not supported (no CUDA on this platform)" ;;
    *) fail "Unsupported platform: $(uname -s)-$(uname -m) (need Linux x86_64)" ;;
esac

# ── 2. NVIDIA GPU + driver ──
info "Checking NVIDIA GPU..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found. Install NVIDIA driver first, e.g.:
    sudo apt install nvidia-driver-535   # Ubuntu
    sudo dnf install akmod-nvidia        # Fedora
    (then reboot and re-run this script)"
fi
GPU_LINE="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
if [ -z "$GPU_LINE" ]; then
    fail "nvidia-smi found but no GPU reported — check driver."
fi
info "  OK: $GPU_LINE"

# ── 3. Python ──
info "Checking Python..."
if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 not found. Install Python >= 3.10, e.g.: sudo apt install python3 python3-pip python3-venv"
fi
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_OK="$(python3 -c 'import sys; print(1 if sys.version_info >= (3,10) else 0)')"
if [ "$PY_OK" != "1" ]; then
    fail "Python $PY_VER found — need >= 3.10."
fi
info "  OK: Python $PY_VER"

# ── 4. install moe-l2 + binaries ──
info "Installing moe-l2 from PyPI..."
if command -v uv >/dev/null 2>&1; then
    info "  using uv (fast)"
    uv tool install --python "$(command -v python3)" moe-l2 || uv pip install --python "$(command -v python3)" moe-l2
elif command -v pip3 >/dev/null 2>&1; then
    info "  using pip3"
    pip3 install --user moe-l2
else
    fail "Neither uv nor pip3 available. Install python3-pip (or uv)."
fi

if ! command -v moe-l2 >/dev/null 2>&1; then
    # pip --user installs to ~/.local/bin
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v moe-l2 >/dev/null 2>&1 || fail "moe-l2 not found on PATH after install."

info "Downloading prebuilt GPU binaries (llama-server + CUDA .so)..."
moe-l2 download-bins

# ── 5. optional: demo model ──
MODEL_DIR="${MODEL_DIR:-$HOME/.moe-l2/models}"
echo
read -r -p "[moe-l2] Download demo model Qwen3.6-35B-A3B (~11.5GB) now? [y/N] " ANSWER
if [[ "$ANSWER" =~ ^[Yy]$ ]]; then
    info "Downloading qwen3.6-35b → $MODEL_DIR (resumable, Ctrl+C to pause)..."
    moe-l2 model download --model qwen3.6-35b --dir "$MODEL_DIR"
else
    warn "Skipped model download. Later run:  moe-l2 model download --model qwen3.6-35b"
fi

# ── 6. doctor ──
echo
info "Running environment self-check..."
moe-l2 doctor || warn "Some checks failed — see messages above (usually just: download model)."

echo
info "Install complete!"
info "  Start server:   moe-l2 start --model $MODEL_DIR/Qwen3.6-35B-A3B-UD-IQ2_M.gguf --gpu"
info "  Or list models: moe-l2 model list"
