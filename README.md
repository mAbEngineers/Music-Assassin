<div align="center">

# 🎵 Music Assassin

**Strip background music out of long audio and video — keep the dialogue, keep the SFX.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-informational)
![Offline](https://img.shields.io/badge/runs-100%25%20offline%2C%20CPU--only-success)
![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)

</div>

Point it at an anime episode, a movie, or any audio/video file with background
music over the dialogue, and it removes the music — using open-source AI
separation models, entirely offline. No cloud services, no paid APIs, no GPU
required.

---

## ✨ Highlights

- ⚡ **Fast** — the OpenVINO-accelerated engine processes a 10-minute episode in about **58 seconds** on a 16-core CPU (down from ~225 s through the older pipeline).
- 🎚️ **Keep the sound effects** — an optional pass extracts footsteps, impacts, and ambience separately and mixes them back in after the music is gone.
- 🧠 **A real model library, not one model** — Demucs, MDX-Net/UVR, Spleeter, and two dialogue/music/effects separators (MRX, CDX23) are all selectable, with VAD gating and a learned cleanup pass on top. See the [full lineup](#-model-library) below.
- 📦 **Fully offline, CPU-only** — every model runs locally once downloaded; nothing leaves your machine.
- 🎬 **Video-aware** — subtitles are kept, and the video stream passes through untouched instead of being re-encoded.
- 📂 **Batch + YouTube** — queue a folder of files, or just paste a YouTube URL.

---

## 🧩 Model library

Music Assassin isn't built on one separator — it ships a selectable library of
them, so you can trade speed for quality (or vice versa) per file.

| Model | Role | Size | Notes |
|---|---|---|---|
| **Voc_FT · fast engine** (OpenVINO) | Default separator | 66 MB | Same weights as MDX-Net Voc_FT, ~7–12× faster via OpenVINO BF16/onnxruntime |
| **MRX dialogue** | Fastest separator | 31 MB | Dialogue/music/effects in one pass, int8 ONNX, MIT-licensed |
| **CDX23** | Quality effects separator | ~80 MB | Best sound-effects retention of anything tested; used for "Keep effects: Quality" |
| **MDX-Net Voc_FT / Kim_Vocal_2** | Vocals-only specialists | 66–130 MB | Via `audio-separator`; the pre-fast-engine default |
| **DEMUCS** htdemucs / htdemucs_ft / mdx_extra | General 4-stem separation | 80 MB–320 MB | Meta Research; natural-sounding, INT8-quantized option |
| **Spleeter** | Legacy 4-stem fallback | ~150 MB | Fastest to load, lowest quality on anime-style vocals |
| **Silero VAD** | Silence gate | ~1 MB | Silences gaps between lines where leftover music bleeds through |
| **DeepFilterNet** (optional) | Post-separation cleanup | small | Learned speech enhancer for residual noise/artifacts |

The repo also doubles as a research lab: **20+ separation and speech-enhancement
models** have been benchmarked here, even where they didn't make the cut for
the app. See
[`docs/COMPREHENSIVE_METHOD_COMPARISON.md`](docs/COMPREHENSIVE_METHOD_COMPARISON.md)
for the full table.

> **Know the limits.** Speech-enhancement models (DeepFilterNet, GTCRN, DTLN,
> MetricGAN+, …) clean up noise but **do not remove music** — they're trained
> on speech-vs-noise, so they treat music as speech. Only the separators in
> the table above (Demucs/MDX-Net/MRX/CDX23/Spleeter class) actually remove
> it. Details in [`docs/COMPREHENSIVE_METHOD_COMPARISON.md`](docs/COMPREHENSIVE_METHOD_COMPARISON.md).

---

## 📥 Download

Prebuilt, ready to run — no Python install needed.

| Platform | File | Size |
|---|---|---|
| **Windows** 10/11 (x64) | [`MusicAssassin_v0.4.4.exe`](https://github.com/mAbEngineers/Music-Assassin/releases/download/v0.4.4/MusicAssassin_v0.4.4.exe) | ~394 MB |
| **Ubuntu / Debian** (x64) | [`MusicAssassin_v0.4.4.deb`](https://github.com/mAbEngineers/Music-Assassin/releases/download/v0.4.4/MusicAssassin_v0.4.4.deb) | ~610 MB |
| **Linux** (x64, portable) | [`MusicAssassin_v0.4.4`](https://github.com/mAbEngineers/Music-Assassin/releases/download/v0.4.4/MusicAssassin_v0.4.4) | ~610 MB |

- **Windows** — run the `.exe` directly, no install.
- **Debian/Ubuntu** — `sudo dpkg -i MusicAssassin_v0.4.4.deb`, then run `music-assassin`.
- **Linux portable** — `chmod +x MusicAssassin_v0.4.4 && ./MusicAssassin_v0.4.4`.

Demucs (`htdemucs`) and Silero VAD are bundled; other models download once on
first use. [All releases →](https://github.com/mAbEngineers/Music-Assassin/releases)

> 🚧 **v0.4.5 — the fast engine + keep-sound-effects build — is being
> packaged now** and isn't on the releases page yet. Want it today? Run from
> source (below) on the `feat/fast-engine-keep-effects` branch.

---

## 🚀 Run from source

```bash
git clone https://github.com/mAbEngineers/Music-Assassin.git
cd Music-Assassin

uv python install 3.12
uv venv $HOME/.venvs/music-assassin --python 3.12
source $HOME/.venvs/music-assassin/bin/activate

bash setup_v044.sh $HOME/.venvs/music-assassin   # installs deps + downloads models

python MusicAssassinGUI_v0.4.5.py                # current development build
```

DeepFilterNet needs a Rust toolchain: `curl https://sh.rustup.rs | sh -s -- -y`

Older Python, or want the three-separator build without the newer cleanup
stages? See the full per-version install matrix in
[`CHANGELOG.md`](CHANGELOG.md). Older GUIs live in [`versions/`](versions/).

### Basic usage

1. Launch the app and pick a separator from the dropdown (the fast engine is the default).
2. Optionally turn on **Keep sound effects**.
3. **Browse Files** to add audio/video, or paste a YouTube URL.
4. Pick a processing mode (Balanced recommended) and hit **Start Processing**.
5. Output lands as `{filename}-vocals.{extension}` next to the source file.

---

## ⚙️ Requirements

- **Python 3.9–3.12** (3.12 tested) to run from source.
- **FFmpeg** — system package, not pip (`apt install ffmpeg` / `brew install ffmpeg`).
- **Rust toolchain** — only if you want DeepFilterNet cleanup.
- CPU-only; no GPU/CUDA needed anywhere in the pipeline.

Exact per-version Python/dependency matrix: [`CHANGELOG.md`](CHANGELOG.md) and
[`requirements/REQUIREMENTS_GUIDE.md`](requirements/REQUIREMENTS_GUIDE.md).

---

## 📚 Research & roadmap

This repo is also the research lab behind the model choices above:

- [`HANDOVER.md`](HANDOVER.md) — current branch state, how to run/verify it, open issues.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — direction, candidates, open decisions.
- [`docs/COMPREHENSIVE_METHOD_COMPARISON.md`](docs/COMPREHENSIVE_METHOD_COMPARISON.md) — every model/technique tried, in one table.
- [`docs/OFFLINE_SPEED_RESEARCH.md`](docs/OFFLINE_SPEED_RESEARCH.md) — how the pipeline got under a minute per 10 minutes of audio.
- [`docs/LIVE_APP_PLAN.md`](docs/LIVE_APP_PLAN.md) — why a sibling repo, [`music-assassin-live`](https://github.com/mAbEngineers/music-assassin-live), exists for the real-time/streaming version.

---

## 🤝 Contributing

Feedback and PRs welcome. On the near-term list:
- GPU acceleration toggle
- Stereo output (currently mono)
- SCNet / RoFormer models for higher-quality separation
- A real-time-capable separator for the live app

---

## 🙏 Acknowledgments

**Demucs** (Meta Research) · **Spleeter** (Deezer Research) ·
**audio-separator / UVR** (NomadKaraoke / Anjok07) ·
**DeepFilterNet** (Rikorose) · **Silero VAD** (Silero Team) ·
**sherpa-onnx** (k2-fsa) · **OpenVINO** (Intel) ·
**MRX** (Mitsubishi Electric Research Labs) ·
**CDX23** (ZFTurbo / Sound Demixing Challenge 2023) ·
**FFmpeg** · **SciPy**

---

<div align="center">

MIT licensed — see [LICENSE](LICENSE).

*Fast Voc_FT engine · Demucs · Spleeter · MDX-Net · VAD gating · optional DeepFilterNet cleanup · optional sound-effect keeping*

</div>
