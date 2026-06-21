# Requirements Files Guide - Music-Assassin v0.4.1

This directory contains multiple requirements files optimized for different use cases and setups.

## Quick Reference

| File | Use Case | Size | Speed | Audio Quality |
|------|----------|------|-------|---------------|
| `requirements.txt` | **Default (Recommended)** | Medium | Fast | Excellent |
| `requirements_latest.txt` | Latest versions of all deps | Medium | Fast | Excellent |
| `requirements_pinned.txt` | Production/Reproducible | Medium | Fast | Excellent |
| `requirements_demucs_only.txt` | DEMUCS neural separation | ~1.5GB | Slower | Best |
| `requirements_mdxnet_only.txt` | **MDX-Net vocals-only (v0.4.4)** | ~66–130MB/model | Fast on CPU | Best vocals |
| `requirements_spleeter_only.txt` | Lightweight Spleeter only | ~500MB | Faster | Good |
| `requirements_by_version.txt` | Historical version info | Documentation | - | - |

---

## Installation Instructions by Use Case

### 🎯 **Default Setup (Recommended)**
Best balance of quality, speed, and compatibility.

```bash
pip install -r requirements.txt
```

Or with GPU support:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

### 🧠 **DEMUCS-Only (Best Quality)**
If you want the highest quality vocal extraction using neural networks. Takes ~1.5GB storage but produces the most natural-sounding results.

```bash
pip install -r requirements_demucs_only.txt
```

**Pros:**
- Neural network-based separation
- Most natural vocal quality
- No robotic artifacts

**Cons:**
- Larger download (~1.5GB)
- Slower processing
- Requires more RAM

---

### 🎤 **MDX-Net-Only (Best Vocals, Light & Fast) — v0.4.4**
The recommended path for **vocals-only** output. MDX-Net (UVR) models run on
onnxruntime — cleaner vocal isolation than Spleeter and a step up from Demucs for
vocals, while staying fast on CPU with small per-model downloads (~66–130 MB).

```bash
pip install -r requirements_mdxnet_only.txt
```

Models (selectable in the GUI dropdown, auto-downloaded on first use):
- `UVR-MDX-NET-Voc_FT.onnx` — fast, ~66 MB (default)
- `Kim_Vocal_2.onnx` — best CPU quality, ~130 MB

**Pros:** vocals-focused, fast on CPU, tiny downloads, no TensorFlow.
**Cons:** vocals-only (no other stems); needs onnxruntime.
List all models: `audio-separator --list_models`.

---

### ⚡ **Spleeter-Only (Lightweight & Fast)**
For minimal setup with quick processing. Spleeter uses 4-stem separation (vocals, drums, bass, other).

```bash
pip install -r requirements_spleeter_only.txt
```

**Pros:**
- Lightweight (~500MB)
- Fast processing
- Good vocal extraction
- Well-established

**Cons:**
- Slight robotic elements
- Heavier instruments sometimes remain

---

### 📌 **Production Setup (Reproducible)**
Use pinned versions for CI/CD pipelines, Docker containers, or when you need exact reproducibility.

```bash
pip install -r requirements_pinned.txt
```

**Use when:**
- Setting up in production
- Creating Docker images
- Need consistent results across machines
- Running automated tests

---

### 🆕 **Latest Versions**
Uses the absolute latest stable versions of all dependencies.

```bash
pip install -r requirements_latest.txt
```

**Use when:**
- You want the newest features
- You don't need backward compatibility

---

## Version History Overview

For historical context:

- **v0.0.1-0.0.2**: MoviePy era (slow video handling)
- **v0.1.0-0.1.2**: FFmpeg transition (much faster)
- **v0.2.0-0.2.2**: Cleanup phase (better memory usage)
- **v0.3.0-0.3.1**: Stream handling improvements
- **v0.3.2**: Chunked processing, Spleeter 4-stems
- **v0.3.3**: Harmonic rejection filters, DSP pipeline
- **v0.3.4**: YouTube support (experimental)
- **v0.4.0**: DEMUCS introduction
- **v0.4.1**: DEMUCS/Spleeter hybrid **(Current)**

See `requirements_by_version.txt` for specific dependencies per version.

---

## GPU Acceleration

### For NVIDIA GPUs (CUDA)

```bash
# Install PyTorch with CUDA 11.8 support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install rest of requirements
pip install -r requirements.txt
```

**Supported CUDA versions:** 11.7, 11.8, 12.1

### For AMD GPUs (ROCm)

```bash
# Install PyTorch with ROCm support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Then install rest of requirements
pip install -r requirements.txt
```

### For Apple Silicon (MPS)

```bash
pip install torch torchaudio
pip install -r requirements.txt
```

---

## Optional Dependencies

### YouTube Support
Uncomment in any requirements file or install separately:

```bash
pip install yt-dlp>=2024.1.0
```

### Development & Testing
```bash
pip install pytest pytest-cov
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
Install PyTorch separately:
```bash
pip install torch torchaudio
```

### "ModuleNotFoundError: No module named 'tensorflow'"
Already included in `spleeter` dependency, but if needed:
```bash
pip install tensorflow>=2.11.0
```

### Memory issues?
Use `requirements_spleeter_only.txt` - it requires less RAM.

### Slow processing?
- Use `requirements_spleeter_only.txt` for faster 4-stem separation
- Install GPU support for DEMUCS
- Reduce audio sample rate in preprocessing

### YouTube URLs not working?
Install YouTube support:
```bash
pip install yt-dlp>=2024.1.0
```

---

## Version Compatibility

- **Python:** 3.8 - 3.11 (tested on 3.10)
- **OS:** Windows, macOS, Linux
- **FFmpeg:** 4.0+ required (system package, not pip)

Verify FFmpeg installation:
```bash
ffmpeg -version
```

Install if missing:
- **Windows:** `choco install ffmpeg` or download from ffmpeg.org
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg` (Ubuntu/Debian)

---

## Comparison Matrix

### Audio Quality (vocals-only)
1. **MDX-Net (UVR)** (best vocals): Voc_FT / Kim_Vocal_2 — cleanest vocal isolation, light
2. **DEMUCS** (excellent): Neural network separation, natural vocals (general 4-stem)
3. **Spleeter 4-stem** (good): Well-established, consistent results
4. **Spleeter 2-stem** (basic): Original spleeter format

### Processing Speed
1. **Spleeter 2-stem** (fastest): ~2-3 min per 10 min audio
2. **Spleeter 4-stem** (fast): ~3-5 min per 10 min audio  
3. **DEMUCS** (slower): ~5-10 min per 10 min audio (without GPU)

### Disk/RAM Requirements
1. **Spleeter only** (minimal): ~500MB | 2GB RAM
2. **Both/Default** (medium): ~2GB | 4GB RAM
3. **DEMUCS only** (large): ~1.5GB | 6GB RAM

---

## Default Installation Recommendation

For most users:
```bash
# 1. Install PyTorch with GPU if available
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Install Music-Assassin dependencies
pip install -r requirements.txt

# 3. (Optional) YouTube support
pip install yt-dlp

# 4. Run
python MusicAssassinGUI_v0.4.1.py
```

This gives you both DEMUCS and Spleeter with automatic fallback support.

---

For more information, see [README.md](README.md) or visit the [GitHub repository](https://github.com/yoake/Music-Assassin).
