# Handover — Music-Assassin (offline app + research lab)

**Updated 2026-10-23.** Covers the fast-engine and "keep sound effects" work, the packaging changes, and what comes next.

Related handovers:
- **Live app, sound effects:** `docs/LIVE_SFX_HANDOVER.md`. A copy also sits in `music-assassin-live/docs/`.
- **Live app, general:** the live repo's own `HANDOVER.md` and `docs/ROADMAP.md`.

---

## 1. State of the tree

| | |
|---|---|
| **Branch** | `feat/fast-engine-keep-effects`, pushed to origin. **Not merged.** Merging is the user's call. PRs #2–#5 are also unmerged. |
| **Ahead of `main` (`e9327e3`)** | Earlier fixes already on the branch: `9d44d4e`, `e050183`/`34f1330`, `1aa2829` (htdemucs segment fix, packaging defects, crash logging). This round: `71a12d1` (fast engine + keep sound effects), `01f7d58` (packaging), `dc05589` (live SFX handover; `dialog.md` gitignored), `85059cd` (quality harness A1), and the commit holding this file plus the doc updates. |
| **Concurrency** | Several Claude sessions edit this gvfs-mounted tree at once. Stage only the exact paths you mean to commit (never `git add -A`), and push promptly: this disk is the only other copy. |
| **Live repo** | `music-assassin-live` has another session's uncommitted work. I added only `docs/LIVE_SFX_HANDOVER.md` there, also uncommitted. |

---

## 2. What changed this round

### 2.1 Fast separation engine — `fast_separation.py`

Runs UVR-MDX-NET-Voc_FT on **OpenVINO**: BF16 compute where the CPU supports it (AVX512_BF16 / AMX), FP32 otherwise, with onnxruntime as the fallback.

- **Threads:** set explicitly, to the number of physical cores.
- **Processing:** one pass over the whole file, with flat memory use.
- **Matches audio-separator** at 48 dB SI-SDR, with one deliberate fix. audio-separator scaled each 60 s chunk by its own peak, which made dialogue in quiet scenes quieter. The engine uses one global scale instead.
- **Also contains:**
  - `separate_dme()`: dialogue / music / effects separation with MRX (int8 ONNX) or CDX23 (torch HTDemucs).
  - `clean_effects()`: music over-subtraction.
  - Model download with size and sha256 checks.

### 2.2 App changes — `MusicAssassinGUI_v0.4.4.py`

- **Separator choices:** "Voc_FT · fast engine (Best)" is the default; also "… low overlap (Faster)" and "MRX dialogue (Fastest · lower quality)". The old audio-separator, Demucs and Spleeter entries remain.
- **Keep sound effects:** Off / Fast (MRX) / Quality (CDX23).
  - **How it works:** a parallel path extracts effects from the un-gated input and adds them back *after* the VAD gate, scaled by the post-process normalisation gain (`_last_post_gain`). A block limiter (`mix_effects`) then keeps peaks ≤ 0.98.
- **Effects level:** Quieter / Normal / Louder / Much louder, with a per-model default (`FX_LEVEL_DEFAULT`): MRX +3 dB (the user's choice by ear), CDX23 0 dB.
- **Fix:** the startup default ran DEMUCS while the window showed MDX-Net Voc_FT. `setFrames()` now calls `set_separator()` once.
- **Silero VAD preloads in a background thread** (`preload_silero_vad()`). Its first `torch.hub` load cost 9–22 s.
- **Per-stage `[time]` lines** in the log, and a fast-engine/OpenVINO line at startup.

### 2.3 Packaging — `build/MusicAssassin_v0.4.4.spec`, `build/runtime_hook_v0.4.4.py`, CI

- **The spec builds from `versions/MusicAssassinGUI_v0.4.4.py`, not the root file.** It had been stale; it's now synced. **Keep the two copies identical.**
- **OpenVINO is collected by hand** (PyInstaller has no hook for it). `libs/` is kept together because libopenvino finds its plugins next to itself. The GPU/NPU plugins, GPU `cache.json` and the non-ONNX frontends (~75 MB) are dropped.
- **Voc_FT, MRX and CDX23 (151 MB) are bundled** from the build machine's `~/.cache/music-assassin/models`. The runtime hook sets `MA_BUNDLED_MODEL_DIR`.
  - **Warm the cache before building:** `python -c "import fast_separation as f; [f.model_path(k) for k in f._MODELS]"`
- **Windows CI** now installs openvino and pyloudnorm, warms the models, and **fails if the engine isn't on OpenVINO**. The engine's fallback to onnxruntime is silent. **Not run yet.**
- **Verified on Linux:**
  - **Size:** the build is 890 MB (was 640 MB; the limit is 2 GB).
  - **Startup:** the built app logs `Fast engine: True (openvino)`.
  - **Engine in a bundle:** a test bundle with the same rules ran all three models from the bundled folder, on OpenVINO BF16, matching the un-frozen run at >129 dB SI-SDR.

### 2.4 Docs

| Doc | What it holds |
|---|---|
| `docs/OFFLINE_SPEED_RESEARCH.md` | All speed measurements, the full-pipeline per-stage tables, and the keep-effects leakage results |
| `docs/SIMILAR_PROJECTS_RESEARCH.md` | Survey of similar projects, effects-keeping designs, live-latency findings, UX ideas |
| `docs/LIVE_SFX_HANDOVER.md` | What the offline effects work means for the live app |

---

## 3. Numbers to know (measured; 10-min episode; Ryzen 9 9950X3D, 16 threads, idle)

| Setting | Total s per 10 min |
|---|---|
| Old app: audio-separator Voc_FT, separation only | 220–228 |
| **Best: fast engine + DeepFilterNet + VAD gate** (default) | **58** |
| Best + keep effects, MRX | 85 |
| Best + keep effects, CDX23 | 95 |
| Fastest: MRX for dialogue and effects, one pass | 55 |

- **Per stage (Best):** separation 39.5 s (2.8 s of it model load), DeepFilterNet 12.3, VAD gate 4.6, post-process 1.4.
- **4-core machines** without BF16 will be much slower. Estimates are in `OFFLINE_SPEED_RESEARCH.md`.
- **Effects leakage** (ground-truth test; lower is better): MRX −16.4 dB raw, −21.3 dB with over-subtraction (now its default); CDX23 −24.9 dB raw. A Wiener re-filter did nothing and was removed.
- **By ear (user):** the new outputs are "much better". Even the MRX-only setting keeps effects, slightly muted.
  - **Listening files:** `test_outputs/fast_engine_listening/` (git-ignored, with a README).

---

## 4. How to run and check things

- **Environment:** `bash setup_v044.sh <venv>`. It now installs openvino and pyloudnorm.
  - **Manual equivalent:** CPU torch/torchaudio/torchvision from the PyTorch CPU index, then `openvino onnxruntime numpy<2.5 scipy soundfile ffmpeg-python psutil pyloudnorm demucs audioread`, plus deepfilternet with `--no-deps`.
  - The torchaudio `backend/common.py` shim is still required for DeepFilterNet; `setup_v044.sh` writes it.
- **Headless full pipeline:** load the module with `importlib.util.spec_from_file_location` (the filename has dots). Set `app.root` to an object with no-op `update_idletasks()`/`update()`, set the globals (`separator_choice`, `fast_overlap`, `enhancer_choice`, `vad_enabled`, `fx_choice`, `fx_level_db`), then call `app._process_single_track(src, app.ffmpeg_input(src), 0, dst, False, [{"index": 0}], 0, duration)`.
  - Per-stage times are printed as `[time]` lines.
  - `quality_harness/pipeline.py` does this more cleanly, with each stage toggled through `StageConfig`.
- **Engine overrides:**

  | Variable | Effect |
  |---|---|
  | `MA_ENGINE_THREADS` | Thread count |
  | `MA_ENGINE_PRECISION=f32` | Force FP32 (OpenVINO issue #38256 reports bad BF16 matrix-multiply results on some Ryzen 9000 chips; re-check after OpenVINO upgrades) |
  | `MA_MODEL_DIR` | Model download cache location |
  | `MA_BUNDLED_MODEL_DIR` | Read-only bundled models (set by the runtime hook) |

- **Build:** `build/build.sh 0.4.4` writes to `build/output_v0.4.4/`. Warm the model cache first.

---

## 5. Gotchas that cost time this round

- **`git commit -F -` fails on this gvfs/sftp mount** ("could not read commit message"). On 2026-10-24, `-F <file>` with a file under `/tmp` failed the same way, even though the file existed. What works: `git commit -m "$(cat <file>)"`.
- **`git push` can hang on this mount.** One timed out at 5 min; the retry took 21 s. Wrap it in `timeout` and retry.
- **The repo path contains a colon** (`sftp:host=…`), so `PYTHONPATH=<repo>` silently splits it. Add the path with `sys.path.insert` instead.
- **Session scratch space under `/tmp` is wiped on reboot.** One reboot lost the listening files and the effects test scripts. Keep anything worth keeping in `test_outputs/` (git-ignored).
- **Check `uptime` before timing anything.** Other jobs on this machine pushed load to 20–47 and inflated timings by 25–60 %. The 58 s figure came from an idle run.
- **Run heavy jobs under `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 nice -n 10`** so a runaway can't take the desktop down.

---

## 6. Known issues, not yet fixed

1. **Output is mono.** Extraction and export both use `ac=1`, so every processed video loses its stereo image. Voc_FT and CDX23 are stereo models; the engine currently feeds them duplicated mono.
2. **Suspected distortion from the −40 dB per-sample pre-gate** in `audio_prefilter`. It zeroes every sample with |x| < 0.01, which includes every zero-crossing, so I expect crossover distortion on quiet speech. **Not yet measured.** The effects path already skips it (`gate=False`).
3. **Normal mode applies a 10 kHz low-pass.** The live repo found vocal damage concentrated in the top octave; worth measuring here.
4. **MRX effects leak audible music** (−21 dB after cleanup). The UI label says "some music bleed".
5. **Windows CI is unrun** since the OpenVINO and model changes.
6. **Leftover from the old pipeline:** `separate_with_audio_separator` is still there with its per-chunk peak scaling. It only runs if a user picks the old "MDX-Net" entries.

---

## 7. Next steps (agreed direction, 2026-10-23)

**Track A: offline app (this repo).** Recommended order:

| # | Item | Status |
|---|---|---|
| A1 | Quality harness: ground-truth corpus, scored metrics, before/after reports | **Committed** (`85059cd`, `quality_harness/`). Includes a rebuild of the lost effects-leak test as "Tier 2". The audits need a Python environment with onnxruntime/OpenVINO and the Voc_FT model; see `quality_harness/README.md`. |
| A2 | Audits with A1: pre-gate (§6.2), Normal-mode low-pass (§6.3), whether DeepFilter and the VAD gate help, overlap 0.25 vs 0.1 | Wired into the harness as `--audit a2` |
| A3 | Stereo output (§6.1) | Not started. Watch DeepFilterNet's cost, which doubles per channel and may push Best just past 60 s. |
| A4 | "Music level" control: add removed music back at −20 / −12 dB instead of deleting it | Harness has `--audit a4` |
| A5 | Remember settings between sessions; a 20 s preview / A-B before export | Not started |
| A6 | Release: run Windows CI, Linux build, merge, tag | Not started. **Merge and push need the user's go-ahead.** |

**Track B: live app (`music-assassin-live`).**
- **B1: report latency to players** (PipeWire monitor-port problem). This unlocks lip-sync and a "movie mode".
- **B2: movie mode** with a real separator.
- **B3: music-presence gate.**
- **B4: consolidate the docs**, whose ROADMAP and HANDOVER are now 2,000+ lines.

Details are in `docs/LIVE_SFX_HANDOVER.md` §4 and `docs/SIMILAR_PROJECTS_RESEARCH.md` §4.

**Track C: research.** Distil a small causal "music vs everything else" model, with CDX23 as teacher. Plan in `docs/SIMILAR_PROJECTS_RESEARCH.md` §3.4.

### Decisions waiting on the user

1. **Focus:** offline first, live first, or both in parallel.
2. **Live movie mode:** is a 0.5–1.5 s delay acceptable once video players are told about it? (Live ROADMAP Q3.)
3. **Merge** `feat/fast-engine-keep-effects` and PRs #2–#5, and whether the next release is v0.4.5.
