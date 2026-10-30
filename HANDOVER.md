# Handover — Music-Assassin (offline app + research lab)

**Updated 2026-10-27.** Covers the fast-engine and "keep sound effects" work, the packaging changes, the v0.4.5 release, and what comes next.

Related handovers:
- **Live app, sound effects:** `docs/LIVE_SFX_HANDOVER.md`. A copy also sits in `music-assassin-live/docs/`.
- **Live app, general:** the live repo's own `HANDOVER.md` and `docs/ROADMAP.md`.

---

## 1. State of the tree

| | |
|---|---|
| **Branch** | `main`. The fast-engine / keep-effects work is **merged and shipped** as **v0.4.5** (PR #8, merge commit `41149c3`; the v0.4.5 tag points at it). `main` is clean and had not moved since July before this; it has now. |
| **Release** | **v0.4.5 is published** (not a draft), target `main`, 3 assets: Linux portable (~1.09 GB), `.deb` (~1.09 GB), Windows `.exe` (~589 MB). Windows CI is green on the OpenVINO + model changes. README download links point at `releases/download/v0.4.5`. |
| **App file** | The live GUI is now `MusicAssassinGUI_v0.4.5.py` (root + byte-identical `versions/` copy). v0.4.4 and earlier remain in `versions/`. |
| **Concurrency** | Several Claude sessions edit this gvfs-mounted tree at once. Stage only the exact paths you mean to commit (never `git add -A`), and push promptly: this disk is the only other copy. README and CHANGELOG are owned by another session — don't re-apply edits there. |
| **Live repo** | `music-assassin-live` has another session's uncommitted work. The offline side added only `docs/LIVE_SFX_HANDOVER.md` there. |

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

### 2.2 App changes — `MusicAssassinGUI_v0.4.5.py`

- **Separator choices:** "Voc_FT · fast engine (Best)" is the default; also "… low overlap (Faster)" and "MRX dialogue (Fastest · lower quality)". The old audio-separator, Demucs and Spleeter entries remain.
- **Keep sound effects:** Off / Fast (MRX) / Quality (CDX23).
  - **How it works:** a parallel path extracts effects from the un-gated input and adds them back *after* the VAD gate, scaled by the post-process normalisation gain (`_last_post_gain`). A block limiter (`mix_effects`) then keeps peaks ≤ 0.98.
- **Effects level:** Quieter / Normal / Louder / Much louder, with a per-model default (`FX_LEVEL_DEFAULT`): MRX +3 dB (the user's choice by ear), CDX23 0 dB.
- **Fix:** the startup default ran DEMUCS while the window showed MDX-Net Voc_FT. `setFrames()` now calls `set_separator()` once.
- **Silero VAD preloads in a background thread** (`preload_silero_vad()`). Its first `torch.hub` load cost 9–22 s.
- **Per-stage `[time]` lines** in the log, and a fast-engine/OpenVINO line at startup.

### 2.3 Packaging — `build/MusicAssassin_v0.4.5.spec`, `build/runtime_hook_v0.4.5.py`, CI

- **The spec builds from `versions/MusicAssassinGUI_v0.4.5.py`, not the root file.** It had been stale; it's now synced. **Keep the two copies identical.**
- **OpenVINO is collected by hand** (PyInstaller has no hook for it). `libs/` is kept together because libopenvino finds its plugins next to itself. The GPU/NPU plugins, GPU `cache.json` and the non-ONNX frontends (~75 MB) are dropped.
- **Voc_FT, MRX and CDX23 (151 MB) are bundled** from the build machine's `~/.cache/music-assassin/models`. The runtime hook sets `MA_BUNDLED_MODEL_DIR`.
  - **Warm the cache before building:** `python -c "import fast_separation as f; [f.model_path(k) for k in f._MODELS]"`
- **Windows CI** installs openvino and pyloudnorm, warms the models, and **fails if the engine isn't on OpenVINO**. The engine's fallback to onnxruntime is silent. **Green for v0.4.5** after a fix: the "Verify the fast engine runs on OpenVINO" step uses `shell: python`, which runs a temp script whose own directory (not the repo root) is on `sys.path`, so `import fast_separation` failed; fixed by setting `env: PYTHONPATH: ${{ github.workspace }}` on that step.
- **Verified on Linux:**
  - **Size:** the shipped bundle is ~1.09 GB (limit 2 GB). The gap from the earlier 890 MB build is `triton` (~240 MB, a stray install in the build venv; CPU torch doesn't need it) plus torch's test binaries (~29 MB). The spec excludes both since 2026-10-30; not yet confirmed with a build.
  - **Startup:** the built app logs `Fast engine: True (openvino)`.
  - **Engine in a bundle:** a test bundle with the same rules ran all three models from the bundled folder, on OpenVINO BF16, matching the un-frozen run at >129 dB SI-SDR.
- **App icon (2026-10-30, after v0.4.5).** A dark ink treble clef on an ivory plate, cut once by a thin red line. The piece above the cut lifts away and fades, so the slice still shows at 32 px. The user chose it after several rounds, rejecting a target/scope, blood, two cuts and the live app's waveform as too busy or not distinct enough.
  - It deliberately differs from the live app's indigo waveform icon. The taskbar fix below is copied from the live repo's `docs/ICON_HANDOVER.md`.
  - `build/draw_icon.py` (Pillow) regenerates `build/icon.png`, `build/icon.ico` and `assets/icon*.png`. It needs the Noto Music font (SIL OFL) for the clef glyph; pass `--font` if it isn't in a standard path. The generated files are committed, so builds don't need the font.
    - `build/icon.png` is the 1024 px master; build.sh renders the `.deb`'s hicolor sizes from it.
    - `build/icon.ico` is the Windows exe icon; the spec sets it with `icon=`.
    - `assets/icon*.png` are the Tk window icons; the spec bundles them.
  - **Taskbar fix:** the app creates its window as `Tk(className=WM_CLASS)` with `WM_CLASS = "music-assassin"`. The `.desktop` file has a matching `StartupWMClass=music-assassin`. Both must stay identical, or GNOME shows a generic taskbar icon.
  - Checked with xprop on a real Tk window: `WM_CLASS = "music-assassin", "Music-assassin"`, with four `_NET_WM_ICON` sizes. Not yet checked in an installed `.deb` or the Windows exe.
- **Build off the mount.** PyInstaller's PKG assembly hits `OSError: [Errno 95] Operation not supported` writing to the gvfs/sftp mount. `build/build.sh` honours `BUILD_OUT` for the output dir: `BUILD_OUT=$HOME/ma_build VENV_PATH=~/.venvs/music-assassin bash build/build.sh 0.4.5`.

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

- **Build:** `BUILD_OUT=$HOME/ma_build VENV_PATH=~/.venvs/music-assassin bash build/build.sh 0.4.5` writes to `$HOME/ma_build/output_v0.4.5/`. Warm the model cache first. `BUILD_OUT` is required on this mount (Errno 95 otherwise — see §5).

---

## 5. Gotchas that cost time this round

- **`git commit -F -` fails on this gvfs/sftp mount** ("could not read commit message"). On 2026-10-24, `-F <file>` with a file under `/tmp` failed the same way, even though the file existed. What works: `git commit -m "$(cat <file>)"`; if even `-m` trips the COMMIT_EDITMSG read, fall back to `git commit-tree` plumbing (write a tree, commit-tree with `-p HEAD`, then `git update-ref HEAD <sha>`).
- **PyInstaller cannot package onto this mount — `OSError: [Errno 95] Operation not supported`** during PKG assembly. Build to local disk with `BUILD_OUT=$HOME/ma_build` (see §2.3 / §4).
- **The build venv needs `numpy==2.4.6`.** `numpy 1.26.4` is too old for current audio-separator/scipy (they need ≥2). This is the version Windows CI pins. Torch must be the `+cpu` wheels; `audioread` is a real dep (both were already correct in the working venv — don't assume they're missing from a failing build).
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
5. **Bundle is ~1.09 GB.** Cause found and fixed in the spec (see §2.3); needs a build to confirm ~870 MB. **Note:** `~/.venvs/music-assassin` currently has numpy 1.26.4, which mismatches its SciPy 1.18; rebuild needs numpy 2.x (see §5).
6. **Leftover from the old pipeline:** `separate_with_audio_separator` is still there with its per-chunk peak scaling. It only runs if a user picks the old "MDX-Net" entries.

---

## 7. Next steps (v0.4.5 is shipped; agreed direction, 2026-10-27)

The shipping blocker is cleared — v0.4.5 is out. The offline roadmap (Track A)
below is grouped by the axis each item moves: **quality**, **speed**, or
**user-friendliness**. A1 (the quality harness) is the tool the quality and
speed audits run through; it's already committed (`85059cd`, `quality_harness/`,
`--audit a2` / `--audit a4`). The audits need a Python env with
onnxruntime/OpenVINO and the Voc_FT model; `docs/A2_AUDIT_RESULTS.md` notes the
overlay-venv workaround for the shared venv's numpy/SciPy mismatch.

### Quality (how clean the output is)

| # | Item | Status / note |
|---|---|---|
| A3 | **Stereo output** (§6.1). The single biggest listener-visible win — every processed video is mono today, and Voc_FT/CDX23 are stereo models being fed duplicated mono. | Not started. Watch DeepFilterNet's cost (doubles per channel — a speed catch, see below). |
| A2-q | **Audit the quality-shaping stages:** the −40 dB pre-gate (§6.2), the Normal-mode 10 kHz low-pass (§6.3), DeepFilterNet, the VAD gate. | **Done 2026-10-30**, `docs/A2_AUDIT_RESULTS.md`. Pre-gate removed (destroyed dialogue in quiet files, bought nothing at normal level). DeepFilterNet helps (+1.2 to +2.2 dB contrast, no dialogue damage). VAD a small help. Low-pass and Balanced EQ are listening-session calls, defaults unchanged. |
| A4 | **"Music level" control** — add the removed music back at −20 / −12 dB instead of deleting it. A gentle mixback hides separation artefacts and sounds more natural than a hard cut. | Harness has `--audit a4` to find the level. UI is the user-friendliness half (A4-ui). |

### Speed (how fast, how big)

| # | Item | Status / note |
|---|---|---|
| A2-s | **Overlap 0.25 vs 0.1.** 0.1 is faster; measure whether it costs quality. If not, it's a free speed default. | **Measured 2026-10-30:** ~20% faster and neutral on real dialogue, but worse on the SFX-leak clip. Kept 0.25; recheck on a longer dense-music clip. |
| — | **Shrink the bundle** (~1.09 GB → ~870 MB; §6.5). | Cause found (triton + torch tests); spec fixed 2026-10-30. Confirm with the next build. |
| — | **Stereo's speed catch:** DeepFilterNet runs per channel, so stereo roughly doubles its ~12 s and can push "Best" past the 60 s budget for a 10-min clip. Decide: per-channel, or mono-side DeepFilter + stereo re-inject. | Tied to A3. |

### User-friendliness (how pleasant to use)

| # | Item | Status / note |
|---|---|---|
| A4-ui | **Expose the music-level control in the UI** once A4 finds the levels — a simple "remove / turn down a little / turn down a lot" choice. | Follows A4. |
| A5 | **Remember settings between sessions**, and a **20 s preview / A-B before committing a full export** — so a user isn't waiting a minute to find out a setting was wrong. | Not started. |

### Track B: live app (`music-assassin-live`)

- **B1: report latency to players** (PipeWire monitor-port problem). Unlocks lip-sync.
- **B3: music-presence gate.**
- **B4: consolidate the docs** (ROADMAP + HANDOVER are now 2,000+ lines).
- **B2 (movie mode) is dropped** — the user rejected the 0.5–1.5 s delay. Live music removal now rides entirely on Track C.

Details: `docs/LIVE_SFX_HANDOVER.md` §4 and `docs/SIMILAR_PROJECTS_RESEARCH.md` §4.

### Track C: research (now the only route to live music removal)

Distil a small **causal** "music vs everything else" model, CDX23 as teacher.
Elevated from a side-quest to the critical path for the live app, since movie
mode is off the table. Plan in `docs/SIMILAR_PROJECTS_RESEARCH.md` §3.4.

### Decisions already made (for the record)

- **Focus:** both offline and live, in parallel.
- **Live movie mode:** rejected — the 0.5–1.5 s delay is not acceptable; Track C is the live path instead.
- **Release:** v0.4.5, full version bump, merged and shipped (PR #8).

### Still open

- Which axis to pick up first — quality (A3 stereo is the headline), speed, or user-friendliness. The user framed the next round around these three; no ordering has been chosen yet.
