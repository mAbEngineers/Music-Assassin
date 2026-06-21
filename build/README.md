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
├── MusicAssassin_v0.4.1.spec         (Demucs only)
├── MusicAssassin_v0.4.2.spec         (Demucs + Spleeter)
├── MusicAssassin_v0.5.0.spec
├── runtime_hook_v0.4.1.py
├── runtime_hook_v0.4.2.py
├── runtime_hook_v0.5.0.py
├── requirements_packaging.txt
└── output_v*/                        (build outputs, gitignored)
```

---

## Quick Start

### Linux / macOS
```bash
chmod +x build/build.sh
./build/build.sh 0.4.3
```

### Windows
```batch
build\build.bat 0.4.3
```

### With a specific venv
```bash
VENV_PATH=$HOME/.venvs/music-assassin ./build/build.sh 0.4.3
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
| **0.4.1** | `./build.sh 0.4.1` | **Demucs only** | 3.8+ | Stable |
| **0.4.2** | `./build.sh 0.4.2` | **Demucs + Spleeter** | 3.8–3.11 | Stable |
| **0.4.3** | `./build.sh 0.4.3` | **Demucs + Spleeter + MDX-Net** | 3.9–3.11 | **Current** |
| 0.5.0 | `./build.sh 0.5.0` | Demucs only (ONNX streaming) | 3.9+ | Experimental |

---

## Environment Setup

### v0.4.3 (Current) — requires Python 3.11

`audio-separator` requires Python 3.9+, Spleeter requires Python < 3.12.
Python 3.11 is the compatible intersection.

```bash
# Run the setup script for guided instructions:
./build/setup.sh

# Or manually:
uv python install 3.11
uv venv $HOME/.venvs/music-assassin --python 3.11
source $HOME/.venvs/music-assassin/bin/activate
pip install audio-separator onnxruntime soundfile   # MDX-Net
pip install demucs torch torchaudio                 # Demucs
pip install spleeter==2.4.2 tensorflow==2.14.0      # Spleeter
pip install -r requirements/requirements_packaging.txt  # PyInstaller etc.
```

### v0.4.2 — Python 3.8–3.11

```bash
source venv/bin/activate
pip install -r requirements/requirements_pinned.txt
pip install -r requirements/requirements_packaging.txt
```

### v0.4.1 — Python 3.8+

```bash
pip install demucs torch torchaudio pyinstaller
```

---

## Spec File Notes

### v0.4.1 — Demucs only
- Includes: PyTorch, torchaudio, Demucs
- Size: ~1.5 GB
- No TensorFlow / Spleeter

### v0.4.2 — Demucs + Spleeter
- Includes: PyTorch, Demucs, Spleeter, TensorFlow
- Size: ~2–2.5 GB

### v0.4.3 — Demucs + Spleeter + MDX-Net (Current)
- Includes: PyTorch, Demucs, Spleeter, TensorFlow, onnxruntime, audio-separator
- Size: ~2.5–3 GB
- Note: MDX-Net model weights (~66–130 MB) download on first run, not bundled

### v0.3.3 — Spleeter DSP
- Includes: Spleeter, TensorFlow, SciPy
- Size: ~1.5 GB

---

## Version Comparison

| Feature | v0.4.1 | v0.4.2 | v0.4.3 |
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
build/output_v0.4.3/
├── dist/
│   ├── MusicAssassin_v0.4.3          (executable — Linux/macOS)
│   ├── MusicAssassin_v0.4.3.exe      (executable — Windows)
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
pip install audio-separator onnxruntime soundfile
```

### Out of disk space
Need 5 GB+ free. v0.4.3 bundle is ~2.5–3 GB.

### Build hangs > 2 hours
```bash
Ctrl+C
rm -rf build/output_v0.4.3
./build/build.sh 0.4.3
```

---

## Distribution

```bash
# Tar archive (Linux/macOS)
tar -czf Music-Assassin-v0.4.3.tar.gz build/output_v0.4.3/dist/

# Zip (Windows PowerShell)
Compress-Archive -Path build\output_v0.4.3\dist -DestinationPath Music-Assassin-v0.4.3.zip
```

---

*Music-Assassin Build System — v0.4.3 current*
