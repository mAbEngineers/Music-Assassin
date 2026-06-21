# Music-Assassin
Removes music from long audio and video files using AI source separation.

## Download / Run

- **Latest**: v0.4.2 (Demucs + Spleeter + MDX-Net) — requires Python 3.9+
- **Stable**: v0.4.1 (Demucs + Spleeter) — requires Python 3.11 (Spleeter < 3.12)
- **Legacy**: v0.4.0 (Demucs only, early prototype)

## Latest Version: v0.4.2 — Demucs + Spleeter + MDX-Net

**New in v0.4.2**:
- **MDX-Net vocal models** via `audio-separator` (UVR-MDX-NET-Voc_FT · Kim_Vocal_2)
  - Vocals-only specialists: cleaner separation than general 4-stem models
  - Runs on onnxruntime (CPU) — fast, ~66–130 MB vs 600 MB for mdx_extra
  - Auto-downloads weights on first use
- **All models now pre-chunk** the waveform (60s + 2s overlap) before processing
  - Prevents OOM on long files for all models including htdemucs_ft / mdx_extra (4-model bags)
  - Spleeter no longer loads the whole file into TensorFlow at once

**Inherited from v0.4.1**:
- Demucs + Spleeter dual-separator with model cache and INT8 quantisation
- 3 processing modes: Normal / Balanced / Aggressive
- Advanced pre/post-filtering (noise gate, harmonic rejection, de-esser, formant boost)

### Features

| Feature | Status |
|---|---|
| MDX-Net vocals-only separation (UVR) | v0.4.2+ |
| DEMUCS neural source separation | v0.4.1+ |
| Spleeter 4-stems separation | v0.4.2+ |
| Pre-chunked processing (60s, bounded RAM) | v0.4.2+ |
| INT8 quantisation (htdemucs, 2-3x faster) | v0.4.2+ |
| Session model cache (no reload per file) | v0.4.2+ |
| 3 processing modes (Normal/Balanced/Aggressive) | v0.3.2+ |
| Advanced pre-filtering (gate + high-pass) | v0.4.2+ |
| Advanced post-filtering (harmonic rejection) | v0.3.2+ |
| Audio formats (MP3, WAV, FLAC, M4A, AAC, OGG, OPUS, WMA) | All versions |
| Video formats (MP4, MKV, AVI, MOV, WEBM, WMV, FLV) | v0.1.0+ |
| Subtitle preservation | v0.3.0+ |
| Video stream passthrough (no re-encoding) | v0.3.0+ |
| Batch file processing with queue | All versions |

## Installation & Usage

### v0.4.2 (Demucs + Spleeter + MDX-Net) — Python 3.11 required

Spleeter requires Python < 3.12, and audio-separator requires Python 3.9+, so Python 3.11 is the sweet spot.

```bash
git clone https://github.com/yoake/Music-Assassin.git
cd Music-Assassin

# Create Python 3.11 venv (uv recommended)
uv python install 3.11
uv venv $HOME/.venvs/music-assassin --python 3.11
source $HOME/.venvs/music-assassin/bin/activate

# Install all separators (use uv pip — faster and respects the active venv)
uv pip install audio-separator onnxruntime soundfile   # MDX-Net
uv pip install demucs torch torchaudio                 # Demucs
uv pip install spleeter==2.4.2                         # Spleeter (pins TF 2.12.1)

python MusicAssassinGUI_v0.4.2.py
```

### v0.4.1 (Demucs + Spleeter) — Python 3.8–3.11

```bash
source venv/bin/activate   # Python 3.8–3.11
pip install -r requirements/requirements_pinned.txt
python MusicAssassinGUI_v0.4.1.py
```

### v0.4.0 (Demucs only) — Python 3.8+

```bash
pip install demucs torch torchaudio
python MusicAssassinGUI_v0.4.0.py
```

### Basic Usage

1. Launch the application
2. Select separator from the dropdown (MDX-Net recommended for clean vocals)
3. Click **Browse Files** to add audio/video files
4. Select processing mode (Balanced recommended)
5. Click **Start Processing**
6. Output saved as `{filename}-vocals.{extension}` in the same folder

**Model guide**:
- **MDX-Net Voc_FT** — fastest, ~66 MB, excellent vocals-only (best default)
- **MDX-Net Kim_v2** — slightly higher quality, ~130 MB, still fast on CPU
- **DEMUCS htdemucs** — fast, natural-sounding, general 4-stem
- **DEMUCS htdemucs_ft / mdx_extra** — higher quality, more RAM, slower
- **Spleeter** — legacy fallback, fastest to load, lower quality

---

## Version History

### v0.4.2 — Demucs + Spleeter + MDX-Net (Current)
- MDX-Net vocal specialist models (UVR) via audio-separator + onnxruntime
- All three separators chunked: 60s + 2s overlap, overlap-add
- Chunked progress bar with per-chunk ETA
- Requires Python 3.9+, best on Python 3.11

### v0.4.1 — Demucs + Spleeter (chunked)
- **FIX**: Demucs pre-chunks waveform before tensor creation — htdemucs_ft / mdx_extra (4-model bags) no longer OOM on long files
- **FIX**: Spleeter pre-chunks waveform — TF tensor expansion stays bounded
- INT8 dynamic quantisation (htdemucs, ~2-3x CPU speedup)
- Session model cache — weights loaded once per session
- Global post-filter normalisation (no more static at end)
- Tail fade-out on Demucs output

### v0.4.0 — Demucs only (early prototype)
- DEMUCS v4 async chunked processing (30s chunks)
- Basic vocal enhancement chain (de-esser, compressor, saturation, normalise)
- Simple GUI with mode selection (Premium / Fast / UltraLow RAM)

### v0.3.4 — YouTube Edition (experimental)
- Direct YouTube URL download + process via yt-dlp
- All features of v0.3.3

### v0.3.3 — Enhanced DSP Edition
- Harmonic rejection for piano/violin/guitar (75% residue reduction)
- Notch filters: instrument fundamentals 82–440 Hz
- Chunked post-processing: 30s chunks, 60% lower peak memory
- Modern dark UI, 3 processing modes

### v0.3.2
- Spleeter 4-stems, 20s chunked processing, 3 modes
- Real-time progress bar

### v0.3.0 – v0.3.1
- FFprobe stream detection, multi-audio-track support, queue management

### v0.2.x
- Spleeter 2-stem, spawn fix, reduced imports, audio output fix

### v0.1.x
- FFmpeg pipeline (replaced MoviePy), progress bar, sample padding

### v0.0.1 – v0.0.2
- Initial MoviePy + Spleeter proof of concept, progress logging

---

## Requirements

| Version | Python | Key deps |
|---|---|---|
| v0.4.2 | 3.9–3.11 | audio-separator, onnxruntime, demucs, torch, spleeter==2.4.2, tensorflow==2.12 |
| v0.4.1 | 3.8–3.11 | demucs, torch, spleeter==2.4.2, tensorflow==2.12 |
| v0.4.0 | 3.8+ | demucs, torch, torchaudio |
| v0.3.x | 3.8+ | spleeter==2.3.0, scipy, librosa |

All versions require **FFmpeg** (system package, not pip).

See `requirements/` for pinned and per-version files.

---

## Contributing

Feedback and contributions welcome. Planned improvements:
- GPU acceleration toggle
- Optional vocal-denoise stage (DeepFilterNet)
- SCNet / RoFormer models for higher-quality separation (GPU tier)
- OOPS centre-channel extraction for stereo fast-path

---

## Acknowledgments

- **Demucs** — Meta Research
- **Spleeter** — Deezer Research
- **audio-separator / UVR** — NomadKaraoke / Anjok07
- **FFmpeg** — Video/audio processing
- **SciPy** — Digital signal processing

---

*Music Assassin v0.4.2 — Demucs + Spleeter + MDX-Net with chunked processing and advanced DSP*
