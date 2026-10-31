# Changelog

Full release-by-release history, split out of `README.md` to keep that file
short. Grouped by era, in plain language. "+" = added, "FIX" = bug fix.

## Era 1 — Proof of Concept (v0.0.1 – v0.0.2)
- **v0.0.1**: First working prototype. MoviePy + Spleeter, minimal GUI, no logging.
- **v0.0.2**: + On-screen log window so you can see what the app is doing while it runs.

## Era 2 — FFmpeg Rewrite (v0.1.0 – v0.1.2)
- **v0.1.0**: Replaced MoviePy with direct FFmpeg calls (MoviePy was slow and fragile) — faster and more reliable. + Video file support.
- **v0.1.1**: FIX bug where the end of the processed audio got cut short. FIX app crashing on close.
- **v0.1.2**: Switched to streaming audio through FFmpeg via pipes instead of writing temp files — uses much less memory. + Real progress bar.

## Era 3 — Stability & Batch Queueing (v0.2.0 – v0.2.2)
- **v0.2.0**: FIX a bug that spawned stray background processes (Spleeter thread-pool leak).
- **v0.2.1**: + Proper file queue — batches of files now process in order without hanging.
- **v0.2.2**: Reduced unnecessary imports (faster startup). FIX an audio-output glitch.

## Era 4 — Video & Subtitles (v0.3.0 – v0.3.1)
- **v0.3.0**: Major rewrite. + Subtitle track preservation. + Video-stream passthrough (video is copied, not re-encoded, so quality/speed aren't wasted on the picture). Internal code reorganized.
- **v0.3.1**: + Editable file list — remove queued files before processing starts.

## Era 5 — Verified Checkpoint & Aggressive Variant (v0.3.2 – v0.3.2.2)
- **v0.3.2**: "Verified working" stability checkpoint — confirmed 180 MB peak RAM with natural-sounding vocals.
- **v0.3.2.2**: Side branch. + Aggressive multi-stage music suppression mode that tries to remove more music while preserving sound effects (SFX mode).

## Era 6 — Enhanced DSP Edition (v0.3.3 – v0.3.4)
- **v0.3.3**: Big signal-processing upgrade. + Harmonic rejection (cuts residual piano/violin/guitar bleed by ~75%). + Notch filters targeting instrument frequency ranges (82–440 Hz). + Chunked post-processing (~60% lower peak memory). + 3 processing modes: Normal / Balanced / Aggressive. + Modern dark UI. + Real-time resource monitor.
- **v0.3.4**: + Direct YouTube URL support — paste a link and the app downloads and processes it automatically (via yt-dlp).

## Era 7 — AI Source Separation, current line (v0.4.0 – v0.4.5)
This is where the app switched from pure DSP/Spleeter to real neural
source-separation models, and is the actively developed line today.

- **v0.4.0** *(early prototype)*: + Demucs neural separation (Meta Research) — first version to use a true AI source-separation model instead of Spleeter/DSP alone. Async 30-second chunked processing. Simple mode selection (Premium / Fast / UltraLow RAM).
- **v0.4.1**: + Spleeter added back as a second selectable engine alongside Demucs. FIX both engines now pre-chunk long files (60s + 2s overlap) before feeding the model — previously, long files could crash with out-of-memory errors. + Session model cache (weights load once, not per file). + INT8 quantization for Demucs (~2–3× faster on CPU). FIX static noise at the end of files (normalization was happening per-chunk instead of once across the whole output). + Fade-out on the last half-second to remove Demucs tail artifacts.
- **v0.4.2**: + MDX-Net added as a third engine, via `audio-separator` (UVR-MDX-NET-Voc_FT and Kim_Vocal_2 models) — a vocals-only specialist, smaller (66–130 MB) and cleaner than the general-purpose 4-stem models. Requires Python 3.9+.
- **v0.4.2-RT** *(research/testing branch, not a numbered release for daily use)*: A side branch for evaluating next-generation models before they're promoted into the main line. + MDX23C models (higher SDR than Voc_FT, but require torch and run slower on CPU). + Demucs extras: 6-stem separation (adds piano/guitar), Hybrid Demucs v3, quantized variants. + Optional test-time augmentation (runs the model twice and averages — small quality boost, 2× runtime cost). + Infrastructure stubs for HS-TasNet and RT-STT, two real-time architectures with no public trained weights yet — ready to activate the moment weights become available. + Experimental HPSS pre-filter and SpeechBrain MetricGAN+ post-filter, mirroring a DSP→AI→DSP pipeline design.
- **v0.4.4**: Builds on v0.4.2's three-engine pipeline and adds two new cleanup stages. + **VAD gate** (Silero VAD) — silences the gaps between spoken/sung lines where leftover music was bleeding through. + Optional **DeepFilterNet** pass — a learned speech-cleanup model that removes residual noise/artifacts left after separation. FIX global (not per-chunk) normalization, removing end-of-file static for good. FIX last-chunk fade-to-zero padding, removing a boundary artifact at chunk edges.
- **v0.4.5** *(2026-10, current release)*:
  - + **Fast separation engine** (`fast_separation.py`): Voc_FT on OpenVINO, BF16 where the CPU supports it, with onnxruntime as fallback. A 10-min episode takes about 58 s end to end on a 16-core CPU, down from about 225 s for separation alone. It is the new default.
  - + **Keep sound effects** (Off / Fast MRX / Quality CDX23), with an effects level control. Effects are separated in parallel and added back after the VAD gate.
  - + "MRX dialogue" fastest separator option.
  - FIX the window showed MDX-Net while the app actually ran DEMUCS at startup.
  - FIX the one-time Silero VAD load (9–22 s) now happens in the background.
  - Packaged builds bundle the fast-engine models (Voc_FT, MRX, CDX23) and OpenVINO. **This is the version published as the current GitHub release.**
  - *Since the release (on `main`, not yet in a build):* + New app icon, a sliced treble clef, now also shown in the window, the Linux taskbar and on the Windows `.exe`. Packaging excludes unused GPU code (`triton`, torch test binaries), about 270 MB smaller. The −40 dB pre-gate is off: it did nothing at normal levels and badly distorted dialogue in quiet files (A2 audit, `docs/A2_AUDIT_RESULTS.md`). + **Stereo output**: stereo and surround sources now come out in stereo instead of mono, and kept sound effects keep their stereo image (A3, `docs/A3_STEREO_RESULTS.md`). A Stereo output checkbox turns it off; when there isn't the memory for stereo the app processes in mono and says so next to the file. The DEMUCS, MDX-Net and Spleeter separators are stereo too. FIX Spleeter never produced output (it returns stereo, which the app flattened to twice the length). FIX MDX-Net failed on every file after the first in a session (the cached model kept writing to the first file's temp folder).

## Era 8 — CPU-Optimized Streaming Rewrite, parked experiment (v0.5.0 – v0.5.1)
A separate architectural branch, not a continuation of v0.4.x — see the
numbering note in the README. Explores what the app would look like stripped
down for a future always-on/real-time use case.

- **v0.5.0**: Demucs-only (dropped Spleeter/TensorFlow entirely — saves ~1.5 GB of RAM at startup). + Unified model wrapper supporting both PyTorch and ONNX Runtime backends. + Outer 60-second chunked processing so peak RAM never holds a full song's separated output at once. + INT8 quantization, session model cache, determinate progress bar with ETA. Target: under 500 MB RAM for a 3-minute song.
- **v0.5.1**: Same feature set as v0.5.0, with the same two audio bugs independently found and fixed here: FIX static at the end of files (switched from per-chunk to global normalization). FIX abrupt cutoff on the final chunk (now fades to silence instead of hard-padding with zeros).

---

## Features by version

| Feature | Introduced |
|---|---|
| Fast separation engine (OpenVINO, ~58 s per 10 min end to end) | v0.4.5 |
| Keep sound effects (MRX / CDX23) with level control | v0.4.5 |
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

## Requirements by version

| Version | Python | Key deps |
|---|---|---|
| v0.4.5 | 3.9–3.12 (3.12 tested) | openvino, onnxruntime, pyloudnorm, demucs, audio-separator, spleeter, torch, tensorflow, sherpa-onnx, deepfilternet |
| v0.4.4 | 3.9–3.12 (3.12 tested) | demucs, audio-separator, spleeter, torch, tensorflow, sherpa-onnx, deepfilternet (no fast engine) |
| v0.4.2-RT | 3.9–3.11 | as v0.4.2, plus speechbrain (optional, ~300 MB) |
| v0.4.2 | 3.9–3.11 | audio-separator, onnxruntime, demucs, torch, spleeter==2.4.2, tensorflow==2.12 |
| v0.4.1 | 3.8–3.11 | demucs, torch, spleeter==2.4.2, tensorflow==2.12 |
| v0.4.0 | 3.8+ | demucs, torch, torchaudio |
| v0.5.0 / v0.5.1 | 3.9+ | demucs, torch, torchaudio, onnxruntime (no Spleeter/TensorFlow) |
| v0.3.x | 3.8+ | spleeter==2.3.0, scipy, librosa |

All versions require **FFmpeg** (system package, not pip). DeepFilterNet
(v0.4.4+) additionally requires a Rust toolchain at install time. See
`requirements.txt` and `requirements/REQUIREMENTS_GUIDE.md` for exact pinned
versions.

---

## A note on the version numbers

Development branched at v0.4.2 into two separate experiments that were never
merged back together: the **v0.4.x line** kept all three separators
(Demucs/Spleeter/MDX-Net) and layered on VAD gating + DeepFilterNet + the
fast engine, running through **v0.4.5**. The **v0.5.x line** was an earlier,
parallel rewrite that dropped Spleeter entirely to minimize startup RAM,
aimed at a future real-time/streaming use case. It predates v0.4.4/v0.4.5's
fixes and features despite the higher version number, and isn't being
developed further — it's kept in the repo as a reference architecture (see
`docs/LIVE_APP_PLAN.md` for where that real-time work is headed next).
