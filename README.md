# Music Assassin

Music Assassin removes background music from long audio and video files — for
example, stripping the background music out of an anime episode or a movie so
only dialogue and sound effects remain. It works entirely offline, on CPU, using
open-source AI models. No cloud services, no paid APIs.

This README explains what the project does, which version to use, and exactly
what changed release-by-release — from the first proof-of-concept script to the
current AI-powered pipeline.

---

## Known Limitation — Speech Enhancers Do Not Remove Music

Confirmed by diagnostics in this repo (see `docs/COMPREHENSIVE_METHOD_COMPARISON.md`
for the full model-by-model table, or `docs/MODEL_BENCHMARKS_v0.4.4.md` and
`docs/REALTIME_FILTER_RESEARCH.md` for the underlying methodology): every
speech-enhancement model tested —
DeepFilterNet, DPDFNet (all variants), GTCRN, SpeechDenoiser, DTLN, MetricGAN+ —
attenuates noise superbly (−29 to −63 dB) but passes prominent or vocal-heavy
music almost untouched (−0.4 to −1.7 dB). They're trained on speech-vs-noise
corpora, so they classify music as speech and leave it alone.

The app's plumbing (separation → DSP → enhancer → VAD) is fully working — the
enhancer stage just isn't, and can't be, the music remover. Actual music removal
requires a source-separation model (Demucs/MDX-Net/UVR/Spleeter class). **A
real-time-capable separator is the key open problem for this repo**: current
best measured option is a chunked `sherpa-onnx` Spleeter-2stem-int8 loop (RTF
0.067 on a 4-core CPU — see the research doc), with HS-TasNet (no public
weights yet) and RT-STT (no code released yet) as the architectures to watch.

---

## Which Version Should I Use?

| Situation | Use |
|---|---|
| I just want to run the app | **v0.4.4** — most complete pipeline, actively maintained |
| I need it to work on an older Python (3.8–3.11) | v0.4.1 or v0.4.0 |
| I'm curious about experimental next-gen models | v0.4.2-RT (research branch, not for daily use) |
| I care about minimal startup RAM over features | v0.5.1 (parked side-experiment, see note below) |

**Latest / Recommended: v0.4.4** — Demucs + Spleeter + MDX-Net separation, with
a VAD silence gate and optional DeepFilterNet cleanup on top. Requires Python
3.9–3.12 (3.12 tested).

> **A note on the version numbers.** Development branched at v0.4.2 into two
> separate experiments that were never merged back together: the **v0.4.x
> line** kept all three separators (Demucs/Spleeter/MDX-Net) and layered on
> VAD gating + DeepFilterNet, ending at **v0.4.4**, which is the actively
> maintained version today. The **v0.5.x line** was an earlier, parallel
> rewrite that dropped Spleeter entirely to minimize startup RAM, aimed at a
> future real-time/streaming use case. It predates v0.4.4's fixes and features
> despite the higher version number, and isn't being developed further — it's
> kept in the repo as a reference architecture (see `docs/LIVE_APP_PLAN.md`
> for where that real-time work is headed next).

---

## Version History

Grouped by era, in plain language. "+" = added, "FIX" = bug fix.

### Era 1 — Proof of Concept (v0.0.1 – v0.0.2)
- **v0.0.1**: First working prototype. MoviePy + Spleeter, minimal GUI, no logging.
- **v0.0.2**: + On-screen log window so you can see what the app is doing while it runs.

### Era 2 — FFmpeg Rewrite (v0.1.0 – v0.1.2)
- **v0.1.0**: Replaced MoviePy with direct FFmpeg calls (MoviePy was slow and fragile) — faster and more reliable. + Video file support.
- **v0.1.1**: FIX bug where the end of the processed audio got cut short. FIX app crashing on close.
- **v0.1.2**: Switched to streaming audio through FFmpeg via pipes instead of writing temp files — uses much less memory. + Real progress bar.

### Era 3 — Stability & Batch Queueing (v0.2.0 – v0.2.2)
- **v0.2.0**: FIX a bug that spawned stray background processes (Spleeter thread-pool leak).
- **v0.2.1**: + Proper file queue — batches of files now process in order without hanging.
- **v0.2.2**: Reduced unnecessary imports (faster startup). FIX an audio-output glitch.

### Era 4 — Video & Subtitles (v0.3.0 – v0.3.1)
- **v0.3.0**: Major rewrite. + Subtitle track preservation. + Video-stream passthrough (video is copied, not re-encoded, so quality/speed aren't wasted on the picture). Internal code reorganized.
- **v0.3.1**: + Editable file list — remove queued files before processing starts.

### Era 5 — Verified Checkpoint & Aggressive Variant (v0.3.2 – v0.3.2.2)
- **v0.3.2**: "Verified working" stability checkpoint — confirmed 180 MB peak RAM with natural-sounding vocals.
- **v0.3.2.2**: Side branch. + Aggressive multi-stage music suppression mode that tries to remove more music while preserving sound effects (SFX mode).

### Era 6 — Enhanced DSP Edition (v0.3.3 – v0.3.4)
- **v0.3.3**: Big signal-processing upgrade. + Harmonic rejection (cuts residual piano/violin/guitar bleed by ~75%). + Notch filters targeting instrument frequency ranges (82–440 Hz). + Chunked post-processing (~60% lower peak memory). + 3 processing modes: Normal / Balanced / Aggressive. + Modern dark UI. + Real-time resource monitor.
- **v0.3.4**: + Direct YouTube URL support — paste a link and the app downloads and processes it automatically (via yt-dlp).

### Era 7 — AI Source Separation, current line (v0.4.0 – v0.4.4)
This is where the app switched from pure DSP/Spleeter to real neural
source-separation models, and is the actively developed line today.

- **v0.4.0** *(early prototype)*: + Demucs neural separation (Meta Research) — first version to use a true AI source-separation model instead of Spleeter/DSP alone. Async 30-second chunked processing. Simple mode selection (Premium / Fast / UltraLow RAM).
- **v0.4.1**: + Spleeter added back as a second selectable engine alongside Demucs. FIX both engines now pre-chunk long files (60s + 2s overlap) before feeding the model — previously, long files could crash with out-of-memory errors. + Session model cache (weights load once, not per file). + INT8 quantization for Demucs (~2–3× faster on CPU). FIX static noise at the end of files (normalization was happening per-chunk instead of once across the whole output). + Fade-out on the last half-second to remove Demucs tail artifacts.
- **v0.4.2**: + MDX-Net added as a third engine, via `audio-separator` (UVR-MDX-NET-Voc_FT and Kim_Vocal_2 models) — a vocals-only specialist, smaller (66–130 MB) and cleaner than the general-purpose 4-stem models. Requires Python 3.9+.
- **v0.4.2-RT** *(research/testing branch, not a numbered release for daily use)*: A side branch for evaluating next-generation models before they're promoted into the main line. + MDX23C models (higher SDR than Voc_FT, but require torch and run slower on CPU). + Demucs extras: 6-stem separation (adds piano/guitar), Hybrid Demucs v3, quantized variants. + Optional test-time augmentation (runs the model twice and averages — small quality boost, 2× runtime cost). + Infrastructure stubs for HS-TasNet and RT-STT, two real-time architectures with no public trained weights yet — ready to activate the moment weights become available. + Experimental HPSS pre-filter and SpeechBrain MetricGAN+ post-filter, mirroring a DSP→AI→DSP pipeline design.
- **v0.4.4** *(current / recommended)*: Builds on v0.4.2's three-engine pipeline and adds two new cleanup stages. + **VAD gate** (Silero VAD) — silences the gaps between spoken/sung lines where leftover music was bleeding through. + Optional **DeepFilterNet** pass — a learned speech-cleanup model that removes residual noise/artifacts left after separation. FIX global (not per-chunk) normalization, removing end-of-file static for good. FIX last-chunk fade-to-zero padding, removing a boundary artifact at chunk edges.

### Era 8 — CPU-Optimized Streaming Rewrite, parked experiment (v0.5.0 – v0.5.1)
A separate architectural branch, not a continuation of v0.4.4 — see the
numbering note above. Explores what the app would look like stripped down for
a future always-on/real-time use case.

- **v0.5.0**: Demucs-only (dropped Spleeter/TensorFlow entirely — saves ~1.5 GB of RAM at startup). + Unified model wrapper supporting both PyTorch and ONNX Runtime backends. + Outer 60-second chunked processing so peak RAM never holds a full song's separated output at once. + INT8 quantization, session model cache, determinate progress bar with ETA. Target: under 500 MB RAM for a 3-minute song.
- **v0.5.1**: Same feature set as v0.5.0, with the same two audio bugs independently found and fixed here: FIX static at the end of files (switched from per-chunk to global normalization). FIX abrupt cutoff on the final chunk (now fades to silence instead of hard-padding with zeros).

---

## Installation & Usage

### v0.4.4 (recommended) — Python 3.9–3.12

```bash
git clone https://github.com/yoake/Music-Assassin.git
cd Music-Assassin

# Create a venv (uv recommended)
uv python install 3.12
uv venv $HOME/.venvs/music-assassin --python 3.12
source $HOME/.venvs/music-assassin/bin/activate

# Installs everything, including model downloads
bash setup_v044.sh $HOME/.venvs/music-assassin

python MusicAssassinGUI_v0.4.4.py
```

DeepFilterNet needs a Rust toolchain at install time:
`curl https://sh.rustup.rs | sh -s -- -y`

### v0.4.2 (Demucs + Spleeter + MDX-Net, no VAD/DeepFilterNet) — Python 3.9–3.11

Spleeter requires Python < 3.12, so Python 3.11 is the sweet spot if you want
all three engines without the newer cleanup stages.

```bash
uv python install 3.11
uv venv $HOME/.venvs/music-assassin --python 3.11
source $HOME/.venvs/music-assassin/bin/activate

uv pip install audio-separator onnxruntime soundfile   # MDX-Net
uv pip install demucs torch torchaudio                 # Demucs
uv pip install spleeter==2.4.2                         # Spleeter (pins TF 2.12.1)

python versions/MusicAssassinGUI_v0.4.2.py
```

### v0.4.1 (Demucs + Spleeter) — Python 3.8–3.11

```bash
source venv/bin/activate
pip install -r requirements/requirements_pinned.txt
python versions/MusicAssassinGUI_v0.4.1.py
```

### v0.4.0 (Demucs only) — Python 3.8+

```bash
pip install demucs torch torchaudio
python versions/MusicAssassinGUI_v0.4.0.py
```

### v0.5.1 (parked streaming experiment, Demucs only) — Python 3.9+

```bash
pip install demucs torch torchaudio onnxruntime
python versions/MusicAssassinGUI_v0.5.1.py
```

> All non-current versions live in `versions/`. Only the actively-developed
> release (`MusicAssassinGUI_v0.4.4.py`) and the research/testing branch
> (`MusicAssassinGUI_v0.4.2_RT.py`) stay in the repo root.

### Basic Usage (all versions)

1. Launch the application.
2. Select a separator engine from the dropdown (MDX-Net recommended for clean vocals).
3. Click **Browse Files** to add audio/video files.
4. Select a processing mode (Balanced recommended).
5. Click **Start Processing**.
6. Output is saved as `{filename}-vocals.{extension}` in the same folder.

**Model guide**:
- **MDX-Net Voc_FT** — fastest, ~66 MB, excellent vocals-only (best default)
- **MDX-Net Kim_v2** — slightly higher quality, ~130 MB, still fast on CPU
- **DEMUCS htdemucs** — fast, natural-sounding, general 4-stem
- **DEMUCS htdemucs_ft / mdx_extra** — higher quality, more RAM, slower
- **Spleeter** — legacy fallback, fastest to load, lower quality

---

## Features by Version

| Feature | Introduced |
|---|---|
| VAD silence gate (Silero) | v0.4.4 |
| DeepFilterNet post-separation cleanup (optional) | v0.4.4 |
| MDX-Net vocals-only separation (UVR) | v0.4.2+ |
| DEMUCS neural source separation | v0.4.0+ |
| Spleeter 4-stem separation | v0.4.1+ (v0.2.x+ as standalone) |
| Pre-chunked processing (bounded RAM on long files) | v0.4.1+ |
| INT8 quantization (Demucs, 2–3× faster) | v0.4.1+ |
| Session model cache (no reload per file) | v0.4.1+ |
| Global (whole-file) audio normalization, no end-of-file static | v0.4.1+ |
| 3 processing modes (Normal / Balanced / Aggressive) | v0.3.3+ |
| Harmonic rejection / notch filtering | v0.3.3+ |
| Direct YouTube URL processing | v0.3.4 |
| Subtitle preservation | v0.3.0+ |
| Video stream passthrough (no re-encoding) | v0.3.0+ |
| Batch file processing with queue | v0.2.1+ |
| Video format support (MP4, MKV, AVI, MOV, WEBM, WMV, FLV) | v0.1.0+ |
| Audio format support (MP3, WAV, FLAC, M4A, AAC, OGG, OPUS, WMA) | All versions |

---

## Research & Roadmap Docs

This repo doubles as a research lab for the underlying models — the docs below
are the detailed record behind the summaries in this README:

- **`docs/COMPREHENSIVE_METHOD_COMPARISON.md`** — master lookup table of every model/technique tried or researched (separators, enhancers, VAD, DSP, acceleration).
- **`docs/MODEL_BENCHMARKS_v0.4.4.md`** — the v0.4.4 benchmark pass: speed/RAM measurements on real content.
- **`docs/REALTIME_FILTER_RESEARCH.md`** — architecture research for turning this into a live, system-wide music filter (latency budgets, OS integration).
- **`docs/REALTIME_MUSIC_REMOVAL_BUILD_GUIDE.md`** — per-OS technical build guide for a real-time desktop/mobile version.
- **`docs/LIVE_APP_PLAN.md`** — the plan for splitting this into two repos: this one stays a research lab, a new `music-assassin-live` repo becomes a "volume booster"-style always-on app built on whichever model wins here.
- **`docs/SEPARATION_TECHNIQUES_RESEARCH.md`** — background research on source-separation techniques.

---

## Requirements

| Version | Python | Key deps |
|---|---|---|
| v0.4.4 | 3.9–3.12 (3.12 tested) | demucs, audio-separator, spleeter, torch, tensorflow, onnxruntime, sherpa-onnx, deepfilternet |
| v0.4.2-RT | 3.9–3.11 | as v0.4.2, plus speechbrain (optional, ~300 MB) |
| v0.4.2 | 3.9–3.11 | audio-separator, onnxruntime, demucs, torch, spleeter==2.4.2, tensorflow==2.12 |
| v0.4.1 | 3.8–3.11 | demucs, torch, spleeter==2.4.2, tensorflow==2.12 |
| v0.4.0 | 3.8+ | demucs, torch, torchaudio |
| v0.5.0 / v0.5.1 | 3.9+ | demucs, torch, torchaudio, onnxruntime (no Spleeter/TensorFlow) |
| v0.3.x | 3.8+ | spleeter==2.3.0, scipy, librosa |

All versions require **FFmpeg** (system package, not pip). DeepFilterNet
(v0.4.4) additionally requires a Rust toolchain at install time.

See `requirements.txt` (current, v0.4.4) and `requirements/` (per-version and
per-engine files, plus `REQUIREMENTS_GUIDE.md`) for exact pinned versions.

---

## Contributing

Feedback and contributions welcome. Planned improvements:
- Merge the v0.5.x streaming architecture forward into the v0.4.4 line (or supersede it once the live-app split happens — see `docs/LIVE_APP_PLAN.md`)
- GPU acceleration toggle
- SCNet / RoFormer models for higher-quality separation (GPU tier)
- Mid/side center-channel extraction as a pre-filter (validated in research, not yet wired into the app)
- Real-time-capable separator (HS-TasNet / RT-STT) once public weights or code exist

---

## Acknowledgments

- **Demucs** — Meta Research
- **Spleeter** — Deezer Research
- **audio-separator / UVR** — NomadKaraoke / Anjok07
- **DeepFilterNet** — Rikorose
- **Silero VAD** — Silero Team
- **sherpa-onnx** — k2-fsa
- **FFmpeg** — Video/audio processing
- **SciPy** — Digital signal processing

---

*Music Assassin v0.4.4 — Demucs + Spleeter + MDX-Net separation, with VAD gating and DeepFilterNet cleanup on top*
