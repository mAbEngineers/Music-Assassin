# Music-Assassin Build System

## Build Scripts and PyInstaller Configurations

---

## What's Here

```
build/
├── build.sh / build.bat              (universal build scripts — macOS/Linux/Windows)
├── setup.sh / setup.bat              (one-time environment setup)
├── MusicAssassin_v0.0.1.spec
├── MusicAssassin_v0.0.2.spec
├── MusicAssassin_v0.1.0.spec
├── MusicAssassin_v0.2.0.spec
├── MusicAssassin_v0.3.3.spec
├── MusicAssassin_v0.4.0.spec         (Demucs only)
├── MusicAssassin_v0.4.1.spec         (Demucs + Spleeter)
├── MusicAssassin_v0.4.4.spec         (fast engine + all separators — current)
├── MusicAssassin_v0.5.0.spec
├── runtime_hook_v0.4.0.py
├── runtime_hook_v0.4.1.py
├── runtime_hook_v0.4.4.py            (sets MA_BUNDLED_MODEL_DIR)
├── runtime_hook_v0.5.0.py
├── requirements_packaging.txt
└── output_v*/                        (build outputs, gitignored)
```

---

## Quick Start

### Linux / macOS
```bash
chmod +x build/build.sh
./build/build.sh 0.4.4
```

### Windows
```batch
build\build.bat 0.4.4
```

### With a specific venv
```bash
VENV_PATH=$HOME/.venvs/music-assassin ./build/build.sh 0.4.4
```

---

## Available Versions

| Version | Command | Separators | Python | Status |
|---------|---------|------------|--------|--------|
| 0.0.1 | `./build.sh 0.0.1` | MoviePy + Spleeter | 3.8+ | Legacy |
| 0.0.2 | `./build.sh 0.0.2` | MoviePy + Spleeter | 3.8+ | Legacy |
| 0.1.0 | `./build.sh 0.1.0` | FFmpeg + Spleeter | 3.8+ | Legacy |
| 0.2.0 | `./build.sh 0.2.0` | FFmpeg + Spleeter | 3.8+ | Legacy |
| 0.3.3 | `./build.sh 0.3.3` | Spleeter 4-stems + DSP | 3.8+ | Stable |
| **0.4.0** | `./build.sh 0.4.0` | **Demucs only** | 3.8+ | Stable |
| **0.4.1** | `./build.sh 0.4.1` | **Demucs + Spleeter** | 3.8–3.11 | Stable |
| 0.4.2 | `./build.sh 0.4.2` | Demucs + Spleeter + MDX-Net | 3.9–3.11 | Stable |
| **0.4.4** | `./build.sh 0.4.4` | **Fast Voc_FT engine (OpenVINO) + Demucs + Spleeter + MDX-Net; keep effects** | 3.9–3.12 | **Current** |
| 0.5.0 | `./build.sh 0.5.0` | Demucs only (ONNX streaming) | 3.9+ | Experimental |

---

## Environment Setup

### v0.4.4 (Current) — Python 3.9–3.12

```bash
bash setup_v044.sh $HOME/.venvs/music-assassin   # installs openvino, pyloudnorm and the rest
source $HOME/.venvs/music-assassin/bin/activate
uv pip install -r build/requirements_packaging.txt

# Required before every build: download the three fast-engine models into the cache the spec bundles from
python -c "import fast_separation as f; [print(f.model_path(k)) for k in f._MODELS]"
```

### v0.4.2 — requires Python 3.11

`audio-separator` requires Python 3.9+, Spleeter requires Python < 3.12.
Python 3.11 is the compatible intersection.

```bash
# Run the setup script for guided instructions:
./build/setup.sh

# Or manually:
uv python install 3.11
uv venv $HOME/Documents/venvs/assassin_venv_v0.4.2 --python 3.11
source $HOME/Documents/venvs/assassin_venv_v0.4.2/bin/activate
uv pip install audio-separator onnxruntime soundfile   # MDX-Net
uv pip install demucs torch torchaudio                 # Demucs
uv pip install spleeter==2.4.2                         # Spleeter (pins TF 2.12.1)
uv pip install -r requirements/requirements_packaging.txt
```

### v0.4.1 — Python 3.8–3.11

```bash
source venv/bin/activate
pip install -r requirements/requirements_pinned.txt
pip install -r requirements/requirements_packaging.txt
```

### v0.4.0 — Python 3.8+

```bash
pip install demucs torch torchaudio pyinstaller
```

---

## Spec File Notes

### v0.4.0 — Demucs only
- Includes: PyTorch, torchaudio, Demucs
- Size: ~1.5 GB
- No TensorFlow / Spleeter

### v0.4.1 — Demucs + Spleeter
- Includes: PyTorch, Demucs, Spleeter, TensorFlow 2.12
- Size: ~2–2.5 GB

### v0.4.4 — fast engine + all separators (Current)
- The spec builds from `versions/MusicAssassinGUI_v0.4.4.py`, **not** the root copy. Keep the two identical.
- Includes: everything in v0.4.2, plus DeepFilterNet, Silero VAD, OpenVINO and `fast_separation.py`.
- **OpenVINO** has no PyInstaller hook, so the spec collects it itself:
  - It keeps `libs/` together, because libopenvino looks for its plugins next to itself.
  - It drops the GPU/NPU plugins, the GPU tuning `cache.json`, and the TensorFlow/PyTorch/Paddle/JAX frontends, about 75 MB in total.
- **Models are bundled:** Voc_FT, MRX and CDX23 (151 MB) are copied from the build machine's model cache (`~/.cache/music-assassin/models`, or `MA_MODEL_DIR`) into `ma_models/`.
  - The runtime hook points `MA_BUNDLED_MODEL_DIR` there, so the app works offline from the first run.
  - The spec warns, and the bundle downloads on first use, if the cache wasn't warmed.
- **Size:** about 890 MB on Linux (2026-10-23). The v0.4.4 spec also trims bundle bloat; see commit `e9327e3`.
- **Check after building:** the app log should say `Fast engine: True (openvino)`. If it says `onnxruntime`, OpenVINO wasn't collected and separation will be roughly 2× slower.
- **Windows CI** (`.github/workflows/build-windows.yml`) installs openvino and pyloudnorm, warms the models, and fails if the engine isn't on OpenVINO. It has not been run since these changes.

### v0.4.2 — Demucs + Spleeter + MDX-Net
- Includes: PyTorch, Demucs, Spleeter, TensorFlow 2.12, onnxruntime, audio-separator
- Size: ~2.5–3 GB
- Note: MDX-Net model weights (~66–130 MB) download on first run, not bundled

### v0.3.3 — Spleeter DSP
- Includes: Spleeter, TensorFlow, SciPy
- Size: ~1.5 GB

---

## Version Comparison

| Feature | v0.4.0 | v0.4.1 | v0.4.2 |
|---------|--------|--------|--------|
| Demucs (htdemucs / ft / mdx_extra) | ✅ | ✅ | ✅ |
| Spleeter 4-stems | ❌ | ✅ | ✅ |
| MDX-Net vocals-only (UVR) | ❌ | ❌ | ✅ |
| Pre-chunked separation (60s, bounded RAM) | — | ✅ | ✅ |
| INT8 quantisation (htdemucs) | ❌ | ✅ | ✅ |
| Session model cache | ❌ | ✅ | ✅ |
| 3 DSP modes | ❌ | ✅ | ✅ |
| Python requirement | 3.8+ | 3.8–3.11 | 3.9–3.11 |

---

## Build Output Structure

```
build/output_v0.4.2/
├── dist/
│   ├── MusicAssassin_v0.4.2          (executable — Linux/macOS)
│   ├── MusicAssassin_v0.4.2.exe      (executable — Windows)
│   └── _internal/                    (bundled deps — do not delete)
│       ├── demucs/
│       ├── spleeter/
│       ├── torch/
│       ├── audio_separator/
│       └── ...
└── build/                            (PyInstaller temp files)
```

---

## Build Times

| Step | Time |
|------|------|
| Install dependencies | 5–15 min |
| PyInstaller analysis | 5 min |
| Bundle creation | 15–25 min |
| **Total** | **25–45 min** |

---

## Troubleshooting

### Build script not executable
```bash
chmod +x build/build.sh build/setup.sh
```

### "FFmpeg not found"
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg
# macOS
brew install ffmpeg
```

### audio-separator / onnxruntime missing
```bash
uv pip install audio-separator onnxruntime soundfile
```

### Out of disk space
Need 5 GB+ free. v0.4.2 bundle is ~2.5–3 GB.

### Build hangs > 2 hours
```bash
Ctrl+C
rm -rf build/output_v0.4.2
./build/build.sh 0.4.2
```

---

## Distribution

```bash
# Tar archive (Linux/macOS)
tar -czf Music-Assassin-v0.4.2.tar.gz build/output_v0.4.2/dist/

# Zip (Windows PowerShell)
Compress-Archive -Path build\output_v0.4.2\dist -DestinationPath Music-Assassin-v0.4.2.zip
```

---

*Music-Assassin Build System — v0.4.2 current*
