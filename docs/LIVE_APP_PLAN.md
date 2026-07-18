# Music Assassin Live — App Packaging & Repo Split Plan

> **Product goal:** a "volume booster"-style app. One toggle. When ON, all device
> output audio is intercepted, music is stripped, and the result plays through the
> speaker — no manual routing, no device pickers, no per-app setup.
>
> Companions: [REALTIME_FILTER_RESEARCH.md](REALTIME_FILTER_RESEARCH.md) (why/what),
> [REALTIME_MUSIC_REMOVAL_BUILD_GUIDE.md](REALTIME_MUSIC_REMOVAL_BUILD_GUIDE.md) (how, per-OS).
> This doc adds the two things they don't cover: **repo strategy** and the
> **auto-insertion layer** that makes it behave like a volume booster.

---

## 1. Repo Decision: split into two repos

**Recommendation: new repo for the app** (working name: `music-assassin-live`),
keep `Music-Assassin` as the research lab.

| | Music-Assassin (this repo) | music-assassin-live (new) |
|---|---|---|
| Purpose | Test new filtering methods: quality, CPU, RAM, speed | Shippable system-wide filter app |
| Deps | torch, demucs, speechbrain, librosa, audio-separator (~GBs) | sounddevice, numpy, sherpa-onnx/onnxruntime (~100 MB, **no torch**) |
| Change rhythm | Experiment churn, versioned scripts, logs, test outputs | Releases, tags, changelog, user-facing issues |
| Code shape | File-based pipelines, benchmark harness | Audio routing, streaming engine, UI, packaging recipes |

Why split rather than a subfolder:

1. **Packaging size and reproducibility.** PyInstaller/AppImage builds pull in the
   whole environment. Built from a clean repo with the minimal dep set, the app is
   ~100–150 MB; built from this repo's environment it drags torch (~800 MB) for
   nothing. A separate repo makes "the app never imports torch" enforceable.
2. **History and releases.** This repo's history is experiments (that's good — it's
   a lab notebook). An app needs `v1.0.1`-style tags a user can bisect. Mixing them
   means every model experiment pollutes app history and vice versa.
3. **Platform code accretes.** The app will grow PipeWire session logic, later a
   native plugin (C++/Rust), Windows APO work, packaging CI. None of it belongs in
   a model-research repo.
4. **The boundary is naturally thin** — a small streaming-processor interface plus
   exported model files (Section 3). Thin, stable interface = correct repo seam.

What this repo keeps doing (per its mission): trial new separation/enhancement
methods, benchmark them, and **export winners** across the boundary. `test_processors.py`
grows into the benchmark harness that produces the model cards below.

Model weights are committed to **neither** repo — published as GitHub Release
assets (or HuggingFace) with their model card; the app downloads on first run to
`~/.local/share/music-assassin/models/`.

---

## 2. The App, Layered

```
┌─ Layer 4: Shell ──────────────────────────────────────────────┐
│  Toggle window / tray icon · model picker · CPU & latency     │
│  meter · autostart · first-run model download                 │
├─ Layer 3: Processor plugins (contract with research repo) ────┤
│  GTCRN · DPDFNet · SpeechDenoiser · Passthrough               │
├─ Layer 2: Streaming engine (portable) ────────────────────────┤
│  duplex stream · ring buffers · worker thread · stateful      │
│  48k-stereo ⇄ 16k-mono resampling · pass-through fallback     │
│  · click-free enable/bypass crossfade                         │
├─ Layer 1: Interception / routing (per-OS) ────────────────────┤
│  Linux: PipeWire null-sink + default-sink swap                │
│  Windows (later): Equalizer APO / native APO — see §6         │
└───────────────────────────────────────────────────────────────┘
```

The layering is the point: Layers 1–2–4 don't care what model is inside Layer 3.
Research repo improves Layer 3 forever; the app never changes for a better model.

---

## 3. The Contract Between the Repos

### 3.1 StreamProcessor interface (lives in both repos, ~20 lines)

```python
class StreamProcessor(Protocol):
    name: str
    sample_rate: int        # model-native rate, e.g. 16000
    block_size: int         # model-native block in samples at sample_rate
    def reset(self) -> None: ...                    # clear hidden state
    def process(self, x: np.ndarray) -> np.ndarray  # float32 mono block in/out
```

### 3.2 Model card (`model_card.json`, produced by this repo's benchmark harness)

```json
{
  "name": "gtcrn_simple",
  "file": "gtcrn_simple.onnx",
  "runtime": "sherpa-onnx",
  "sample_rate": 16000,
  "block_size": 256,
  "license": "MIT",
  "benchmarks": { "ms_per_block": 3.1, "ram_mb": 40, "rtf": 0.19 },
  "quality": { "test_set": "assassin-eval-v1", "notes": "voice-only output" }
}
```

Promotion pipeline: candidate model → benchmarked here (speed/RAM/quality on a
fixed eval set) → export ONNX + card → GitHub Release → app lists it in the model
picker. No app code changes.

**Open-source gate:** the card's `license` field is mandatory; anything not
OSI-approved never gets promoted.

---

## 4. Layer 1 — Linux Routing (validated on this machine, 2026-07-02)

This machine: PipeWire 1.0.5 + WirePlumber, **no `pactl`** — use native
`pw-cli`/`wpctl` only. Null-sink create/destroy tested working.

### Enable sequence

```bash
# 1. Snapshot current default sink (persist to state file for crash recovery)
wpctl inspect @DEFAULT_AUDIO_SINK@   # record node.name + id

# 2. Create the trap sink (validated):
pw-cli create-node adapter '{ factory.name=support.null-audio-sink
    node.name=MusicAssassin node.description="Music Assassin"
    media.class=Audio/Sink object.linger=true audio.position=[FL FR] }'
#    node.description matters — without it the sink shows as "(null)" in mixers.

# 3. Make it the default → WirePlumber migrates every default-following stream
wpctl set-default <new-node-id>
```

4. Open one duplex stream: **capture** `MusicAssassin` monitor → **play** to the
   real hardware sink. With PortAudio/sounddevice (no pactl), select endpoints via
   the pulse compatibility layer: set `PULSE_SOURCE=MusicAssassin.monitor` and
   `PULSE_SINK=<real-sink-node.name>` in the process env, open device `"pulse"`.
   (pipewire-pulse serves the protocol; verified running here.)

### Disable / quit sequence

```bash
wpctl set-default <saved-id>     # restore
pw-cli destroy <trap-sink-id>    # WirePlumber migrates streams back
```

### Robustness requirements (not optional — hit two of these during validation)

- **Crash recovery:** previous default sink is written to a state file *before*
  swapping. On every startup: destroy any stale `MusicAssassin` nodes, restore
  default if the state file says we died mid-session.
- **Device hotplug:** the Bluetooth headset (soundcore Q20i) auto-suspends and
  vanishes from the sink list; on reconnect WirePlumber may re-default to it,
  bypassing us. Subscribe to default-sink changes (`pw-mon` or poll) — if default
  changed away from our trap sink while enabled, re-assert it and re-target the
  engine's output to the new real sink.
- **Volume keys:** with the trap sink as default, volume keys scale the *captured*
  signal (this is exactly how volume boosters behave — acceptable v1). Option
  later: pin trap sink at 100% and forward volume changes to the hardware sink.
- **Sink with no active profile:** when BT disconnects here, *zero* hardware sinks
  can exist momentarily. Engine must idle gracefully (keep capturing, discard
  output) instead of crashing.

---

## 5. Layer 2 — Streaming Engine

- Duplex sounddevice callback, `blocksize=1024` @ 48 kHz (~21 ms), stereo.
- Callback thread only moves blocks to/from lock-free ring buffers; inference on a
  worker thread. If the worker falls behind → emit dry signal (pass-through),
  never silence, never block the callback.
- **Stateful resampling** both directions (48 kHz stereo downmix → 16 kHz mono →
  model → 16 kHz → 48 kHz dual-mono). Streaming-safe resampler: `soxr` (LGPL) or
  `samplerate`/libsamplerate (BSD-2) — both have push/streaming APIs; naive
  per-block `scipy.signal.resample` causes boundary clicks.
- 10–20 ms crossfade when toggling process/bypass to avoid clicks.
- Latency budget v1 (Python): 21 ms in-buffer + ~1 ms resample + 2–4 ms GTCRN +
  21 ms out-buffer ≈ **45–70 ms**. Fine for music/video watching; borderline for
  lip-sync-critical use — that's what the Phase-3 native path fixes.

CPU fit (this machine, from measured model metrics): GTCRN ~3 ms/block ≈ a few %
of one core, ~40 MB RAM. DPDFNet ~8 ms/block is the "better quality" step-up and
still fits.

---

## 6. Platform Reality Check

| Platform | Route | Status |
|---|---|---|
| Linux | PipeWire trap-sink (above) | **Validated here; Phase 1** |
| Linux native | LADSPA/filter-chain plugin, C++/Rust + onnxruntime C API | Phase 3. Proven pattern: DeepFilterNet ships an official `deep-filter-ladspa` running realtime on CPU under PipeWire — same architecture, open source, use as reference. Also becomes loadable in EasyEffects for free. |
| Windows | ⚠️ VB-Cable (the usual advice, incl. our build guide) is **closed-source freeware** — conflicts with the project's open-source-only rule. Open-source route: Equalizer APO (GPL) hosting our effect, or a native APO — both need the Phase-3 C++ port first. | Phase 4, after native port |
| Android | "Volume booster" apps attach *built-in* effects (LoudnessEnhancer) to global audio session 0; **custom** effects on the output mix require vendor/ROM integration or root. No consumer path for system-wide filtering — in-app player only (build guide §10–11). | Not planned |
| iOS | System audio capture impossible. | Not planned |

---

## 7. Honest Quality Caveat

GTCRN / DPDFNet / SpeechDenoiser are **speech-enhancement** models: the output is
"voice only" — music is removed, but so are non-speech SFX (footsteps, explosions).
True realtime *music-vs-SFX* separation still waits on HS-TasNet weights or an
RT-STT release (tracked in research). This is fine for the product because of the
layering: ship the voice-only experience now, drop in a music-only-removal
processor the day one exists, zero app changes.

---

## 8. Phases

**Phase 0 — this repo (days):**
- Extract `StreamProcessor` interface; wrap the three existing ONNX models
  (already in `models/`) as streaming processors (sherpa-onnx `OnlineStream` for
  GTCRN/DPDFNet; plain ort + hidden-state loop for SpeechDenoiser).
- Extend `test_processors.py` into the benchmark harness that emits model cards.

**Phase 1 — new repo, Linux MVP (1–2 weeks):**
- Layer 1 routing module (the validated sequence + crash recovery + hotplug watch).
- Layer 2 engine. Deps: `sounddevice numpy sherpa-onnx soxr` (venv note:
  `assassin_venv_v0.4.2` already has sherpa-onnx 1.13.3 + ort 1.27; add
  sounddevice + soxr for prototyping before the new repo gets its own env).
- Minimal shell: one window — big toggle, model dropdown, latency/CPU readout.

**Phase 2 — hardening + packaging (1 week):**
- Tray icon, autostart (`.desktop` + optional systemd user service), config file,
  first-run model download, bypass A/B button.
- PyInstaller onefile → AppImage. Models not bundled (downloaded), keeps the
  binary ~120 MB.

**Phase 3 — native audio path (3–4 weeks, optional but the real product):**
- Port Layer 2+3 to a LADSPA/PipeWire filter-chain plugin (Rust or C++,
  onnxruntime C API), following the deep-filter-ladspa pattern. Python app shrinks
  to UI + session management. Latency → 10–20 ms, CPU drops, no Python runtime in
  the audio path. This artifact is also the entry ticket to Windows (APO) later.

---

*Created 2026-07-02 · supersedes §9/§13 of the build guide for packaging strategy*
