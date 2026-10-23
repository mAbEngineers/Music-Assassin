#!/usr/bin/env bash
# Music Assassin v0.4.4 — Setup Script
# Installs all Python deps + downloads ONNX model files
# Usage: bash setup_v044.sh [path/to/venv]
#
# Example:
#   bash setup_v044.sh /home/yoake/Documents/venvs/assassin_venv_v0.4.2

set -e

VENV="${1:-/home/yoake/Documents/venvs/assassin_venv_v0.4.2}"
PIP="$VENV/bin/pip"
REPO="$(cd "$(dirname "$0")" && pwd)"
MODELS="$REPO/models"

echo "========================================"
echo "  Music Assassin v0.4.4 Setup"
echo "  venv: $VENV"
echo "  repo: $REPO"
echo "========================================"

# ── 1. Rust (required for deepfilternet) ─────────────────────────────────────
if ! command -v rustc &>/dev/null; then
    echo ""
    echo "[1/4] Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "[1/4] Rust already installed: $(rustc --version)"
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# ── 2. Python packages ────────────────────────────────────────────────────────
echo ""
echo "[2/4] Installing Python packages..."
"$PIP" install --upgrade pip -q

# Core separation + ML
"$PIP" install \
    "demucs>=4.0.0" \
    "audio-separator>=0.17" \
    "torch>=2.0.0" \
    "torchaudio>=2.0.0" \
    "numpy>=2.0.0" \
    "scipy>=1.5.0" \
    "soundfile>=0.12.0" \
    "ffmpeg-python>=0.2.0" \
    "librosa>=0.8.0" \
    "psutil>=5.9.0" \
    -q

# ONNX inference (openvino + pyloudnorm: fast engine, fast_separation.py)
"$PIP" install \
    "onnxruntime>=1.17.0" \
    "sherpa-onnx>=1.13.0" \
    "openvino>=2024.0" \
    "pyloudnorm>=0.1.0" \
    -q

# Speech enhancement (Rust compile — takes 3-5 min on first install)
echo "  Installing deepfilternet (compiles Rust extension — may take 3-5 min)..."
"$PIP" install "deepfilternet>=0.5.6" -q

# DeepFilterNet torchaudio shim (torchaudio 2.2+ removed torchaudio.backend.common)
TORCHAUDIO_DIR="$VENV/lib/python3.12/site-packages/torchaudio"
if [ -d "$TORCHAUDIO_DIR" ] && [ ! -f "$TORCHAUDIO_DIR/backend/common.py" ]; then
    echo "  Applying torchaudio.backend.common shim..."
    mkdir -p "$TORCHAUDIO_DIR/backend"
    touch "$TORCHAUDIO_DIR/backend/__init__.py"
    cat > "$TORCHAUDIO_DIR/backend/common.py" << 'SHIM'
"""Compatibility shim: torchaudio.backend.common removed in torchaudio 2.2."""
from dataclasses import dataclass

@dataclass
class AudioMetaData:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str
SHIM
    echo "  Shim written to $TORCHAUDIO_DIR/backend/common.py"
fi

# Spleeter + tensorflow (optional — requires pandas, may conflict with numpy 2.x)
echo ""
read -p "  Install Spleeter + TensorFlow? (large ~500 MB, optional) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    "$PIP" install "tensorflow>=2.16.0" "pandas>=1.3.0" "spleeter>=2.4.0" -q
    echo "  Spleeter installed."
else
    echo "  Spleeter skipped."
fi

# OpenVINO (optional, accelerates ONNX inference via AVX2/AVX-512)
echo ""
read -p "  Install onnxruntime-openvino? (~180 MB, accelerates ONNX enhancers) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    "$PIP" install onnxruntime-openvino -q
    echo "  onnxruntime-openvino installed."
else
    echo "  OpenVINO skipped."
fi

# ── 3. ONNX model files ────────────────────────────────────────────────────────
# Organized by source library: models/<source>/...
echo ""
echo "[3/4] Downloading ONNX model files to $MODELS..."
SHERPA_DIR="$MODELS/sherpa_onnx"
SPEECHDENOISER_DIR="$MODELS/speechdenoiser"
mkdir -p "$SHERPA_DIR" "$SPEECHDENOISER_DIR"

BASE_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models"

download_model() {
    local name="$1" url="$2" dest="$3"
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "  $name: already present ($(du -sh "$dest" | cut -f1))"
    else
        echo "  Downloading $name..."
        wget -q --show-progress "$url" -O "$dest"
        echo "  $name: done ($(du -sh "$dest" | cut -f1))"
    fi
}

download_model "DPDFNet baseline (8 MB)" \
    "$BASE_URL/dpdfnet_baseline.onnx" \
    "$SHERPA_DIR/dpdfnet_baseline.onnx"

download_model "GTCRN simple (524 KB)" \
    "$BASE_URL/gtcrn_simple.onnx" \
    "$SHERPA_DIR/gtcrn_simple.onnx"

download_model "SpeechDenoiser 48k (16 MB)" \
    "https://github.com/yuyun2000/SpeechDenoiser/raw/main/48k/denoiser_model.onnx" \
    "$SPEECHDENOISER_DIR/speechdenoiser.onnx"

download_model "DPDFNet2 48kHz HR (11 MB)" \
    "$BASE_URL/dpdfnet2_48khz_hr.onnx" \
    "$SHERPA_DIR/dpdfnet2_48khz_hr.onnx"

# sherpa-onnx source separation models (see docs/REALTIME_FILTER_RESEARCH.md §9)
SEP_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models"

download_model "sherpa UVR Voc_FT (64 MB)" \
    "$SEP_URL/UVR-MDX-NET-Voc_FT.onnx" \
    "$SHERPA_DIR/sherpa_uvr_voc_ft.onnx"

if [ ! -d "$SHERPA_DIR/sherpa-onnx-spleeter-2stems-int8" ]; then
    echo "  Downloading sherpa Spleeter 2-stem int8 (52 MB)..."
    wget -q --show-progress "$SEP_URL/sherpa-onnx-spleeter-2stems-int8.tar.bz2" \
        -O "$SHERPA_DIR/spleeter2_int8.tar.bz2"
    tar xjf "$SHERPA_DIR/spleeter2_int8.tar.bz2" -C "$SHERPA_DIR"
    rm "$SHERPA_DIR/spleeter2_int8.tar.bz2"
    echo "  sherpa Spleeter 2-stem int8: done"
else
    echo "  sherpa Spleeter 2-stem int8: already present"
fi

# ── 4. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Verifying installation..."
"$VENV/bin/python" - << 'PYCHECK'
import importlib.util, os

checks = {
    "torch":            "DEMUCS / VAD",
    "demucs":           "DEMUCS separation",
    "audio_separator":  "MDX-Net separation",
    "df":               "DeepFilterNet enhancer",
    "sherpa_onnx":      "DPDFNet / GTCRN enhancers",
    "onnxruntime":      "ONNX Runtime (MDX-Net + SpeechDenoiser)",
    "openvino":         "OpenVINO (fast engine: Voc_FT BF16, MRX)",
    "scipy":            "DSP filters",
    "soundfile":        "WAV I/O",
}

models = {
    "sherpa_onnx/dpdfnet_baseline.onnx":   "DPDFNet",
    "sherpa_onnx/gtcrn_simple.onnx":       "GTCRN",
    "speechdenoiser/speechdenoiser.onnx":  "SpeechDenoiser",
    "sherpa_onnx/dpdfnet2_48khz_hr.onnx":  "DPDFNet2 48kHz HR",
    "sherpa_onnx/sherpa_uvr_voc_ft.onnx":  "sherpa UVR Voc_FT (separation)",
}

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

print("\n  Packages:")
for pkg, desc in checks.items():
    ok = importlib.util.find_spec(pkg) is not None
    print(f"    {'OK  ' if ok else 'MISS'} {pkg:<20} {desc}")

print("\n  Model files:")
for fname, desc in models.items():
    path = os.path.join(MODELS_DIR, fname)
    exists = os.path.isfile(path)
    size   = f"{os.path.getsize(path)//1024} KB" if exists else "missing"
    print(f"    {'OK  ' if exists else 'MISS'} {fname:<30} {desc}  ({size})")

# torchaudio shim
try:
    from torchaudio.backend.common import AudioMetaData
    print("\n  torchaudio shim: OK")
except ImportError:
    print("\n  torchaudio shim: MISSING (deepfilternet will fail)")
PYCHECK

echo ""
echo "========================================"
echo "  Setup complete."
echo "  Run tests: $VENV/bin/python test_processors.py"
echo "========================================"
