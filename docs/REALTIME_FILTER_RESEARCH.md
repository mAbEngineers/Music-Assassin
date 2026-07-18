# Real-Time Music Filtering — Architecture Research

How to make Music Assassin work as a live, system-wide music removal filter — covering latency constraints, deployment architectures, OS-level integration strategies, and the relevant models and tools.

---

## Table of Contents

1. [The Core Problem: Latency Budget](#1-the-core-problem-latency-budget)
2. [Real-Time Capable Separation Models](#2-real-time-capable-separation-models)
3. [System-Wide Audio Architectures](#3-system-wide-audio-architectures)
   - Windows — APO (Audio Processing Object)
   - Windows — Voicemeeter + Virtual Cable
   - Linux — PipeWire filter-chain
   - Linux — PulseAudio null sinks
   - Cross-platform — VST/VST3/CLAP plugin
4. [VLC Integration — Is It Worth It?](#4-vlc-integration--is-it-worth-it)
5. [How Volume Boosters / Device-Wide Apps Work](#5-how-volume-boosters--device-wide-apps-work)
6. [Streaming Architecture Design](#6-streaming-architecture-design)
7. [Existing Real-Time Implementations to Study](#7-existing-real-time-implementations-to-study)
8. [Implementation Options for Music Assassin](#8-implementation-options-for-music-assassin)
9. [Comparison Table](#9-comparison-table)
10. [Recommended Path](#10-recommended-path)

---

## 1. The Core Problem: Latency Budget

Real-time audio has hard latency requirements. Above a certain threshold, the human brain perceives the delay as unacceptable:

| Use case | Max acceptable latency |
|---|---|
| Live monitoring (singing into a mic) | < 10 ms |
| Live performance / broadcast | < 25 ms |
| Video playback sync (AV sync tolerance) | < 45 ms |
| Casual listening / podcast playback | < 100–200 ms |
| Background audio (call centers, hearing aids) | < 50 ms |

**Most source separation models are not real-time by default.** Demucs needs a minimum chunk of several seconds to work. The key question for real-time is: *can the model process one audio buffer before the next one arrives?*

### Why standard Demucs fails at real-time

- Demucs requires a minimum chunk size (~3–5 seconds) due to its deep encoder stack. It cannot process a 1024-sample block.
- At 44100 Hz, 5 s = 220,500 samples → ~113 ms to just fill the buffer before processing even starts.
- Processing time on CPU adds another 300–1000 ms per chunk.
- Total: **1–2 seconds of latency minimum** on a standard CPU, even with chunked streaming.

MDX-Net (ONNX) is faster but still adds 200–500 ms on CPU for typical chunk sizes.

**For video playback** (not live voice): 200–500 ms is tolerable if the video player is told to delay the video by the same amount. VLC and mpv support `--audio-desync` / `--av-offset`. This is the pragmatic near-term approach.

---

## 2. Real-Time Capable Separation Models

These are specifically designed or measured for low-latency streaming inference.

### HS-TasNet — Hybrid Spectrogram TasNet

**Paper:** [arxiv 2402.17701](https://arxiv.org/abs/2402.17701) — L-Acoustics, February 2024

**Architecture:**
- Hybrid: combines a waveform-domain branch (good for drums/bass transients) with a spectrogram-domain branch (good for vocals/harmonic content), mirroring HTDemucs but designed for streaming.
- Inspired by Conv-TasNet and HTDemucs; designed from the ground up for causal/streaming operation.
- Uses causal convolutions and a small look-ahead window instead of a large global context.

**Measured latency:** **23 ms** at 44100 Hz  
**SDR on MUSDB:** 4.65 (baseline) → 5.55 with more training data  
**Comparison:** HTDemucs achieves ~8.5 SDR but at 1–2 second latency. HS-TasNet trades ~3–4 dB SDR for 50–100× latency reduction.

**GitHub:**
- [temismink/HS-Tasnet](https://github.com/temismink/HS-Tasnet) — reference implementation
- [lucidrains/HS-TasNet](https://github.com/lucidrains/HS-TasNet) — clean PyTorch port

**Use case:** Live broadcast filtering, hearing aids, live venue remixing.

---

### RT-STT — Real-Time Single-Path TFC-TDF UNet

**Paper:** [arxiv 2511.13146](https://arxiv.org/abs/2511.13146) — November 2025

**Architecture:**
- Based on DTTNet (Dual-Path TFC-TDF UNet), modified into a single-path design.
- Adds a **feature fusion via channel expansion** technique to recover quality lost by removing the dual-path.
- Applies **quantization** (INT8) to further reduce inference time — same trick Music Assassin already uses for HTDemucs.
- Specifically benchmarked for real-time feasibility (real-time factor < 1.0).

**Performance:** Fewer parameters and shorter inference time than state-of-the-art while maintaining competitive SDR.

**Key insight from the paper:** Single-path modeling outperforms dual-path modeling in real-time scenarios because it eliminates the sequential inter/intra dependency that blocks parallelism.

**Use case:** Practical device deployment (mobile, embedded, low-power hardware).

---

### Conv-TasNet (the ancestor)

The foundational causal separation model. Fully convolutional, causal (no future context), can run at <10 ms latency. SDR is lower than modern models (~6 vs ~8.5 for HTDemucs) but it is the architecture that HS-TasNet and RT-STT evolve from. Pre-trained models available via the `asteroid` library.

---

### BSRNN Streaming Variants

Band-Split RNN (not RoFormer) can be made causal by replacing the bidirectional LSTM with unidirectional LSTM plus state caching between chunks. Each chunk only looks at past context. Loss in SDR vs. bidirectional is ~0.5–1.0 dB.

This is how hearing aid processors implement BSRNN-based speech enhancement — chunked unidirectional processing with state carried forward.

---

### VOXLESS (Commercial, Closed Source)

**By AudioSourceRE.** A DAW VST/AU/AAX plugin that does real-time vocal extraction in your DAW session with no pre-processing. Fully on-device, no cloud. Works with Cubase, Logic, Pro Tools, Reaper, FL Studio, Ableton.

**Key trade-off acknowledged by devs:** "trade-off between being able to do it in real-time and getting super clean results." Lower quality than offline Demucs, but true zero-latency (buffer-size dependent) operation.

**Architecture:** Proprietary, not based on open-source models. CPU-intensive; recommends maximum buffer size to avoid glitches.

**Why this matters:** Proof that real-time on-device vocal extraction in a plugin format is commercially viable and accepted by professionals. The category exists and users will pay for it.

---

## 3. System-Wide Audio Architectures

The question is: how does a volume booster or noise canceller work at the device level — affecting ALL audio on the computer regardless of which app is playing it?

---

### Windows — APO (Audio Processing Object)

**What it is:**
APOs are COM objects that Windows loads into the audio engine (audiodg.exe) as plugins. They process audio *before* it reaches the speakers, system-wide, transparently to all applications.

**How it works:**
1. App (Chrome, VLC, Spotify) sends audio to Windows Audio Session API (WASAPI).
2. Windows Audio Engine (audiodg.exe) mixes all app streams.
3. Your APO sits in the processing chain inside audiodg.exe, receives the mixed PCM, processes it, outputs to soundcard driver.
4. Result: ALL app audio goes through your filter automatically.

**Key tools using this:**
- **Equalizer APO** — open-source EQ/filter APO. Installs as a Windows audio enhancement plugin. Supports LADSPA/VST chains via configuration.
- **Peace EQ** — GUI for Equalizer APO.
- **Dolby Atmos / DTS Sound Unbound / Realtek enhancements** — all APOs.

**Limitations:**
- Does NOT work with apps using ASIO or WASAPI Exclusive Mode (they bypass the engine entirely). Spotify, most media players use WASAPI shared mode ✓. DAWs using ASIO bypass it ✗.
- Requires admin/driver installation.
- Must be written in C++ (COM object). Cannot be a Python script. But can call an external process or load a DLL.

**For Music Assassin:**
Could implement a thin APO wrapper in C++ that calls a named pipe/socket connected to the Python ML backend. Or implement the separation in C++ with ONNX Runtime (which already has a C++ API).

---

### Windows — Voicemeeter + Virtual Cable

**What it is:**
A more accessible approach to system-wide processing without writing a driver.

**How it works:**
1. Install **VB-Cable** (free virtual audio device) — creates a virtual speaker and virtual microphone.
2. Set VB-Cable Input as Windows default playback device → all app audio goes to VB-Cable.
3. In **Voicemeeter** (free virtual mixer), VB-Cable output appears as an input channel.
4. Load a VST plugin (like VOXLESS) in Voicemeeter's effects chain → processes the audio.
5. Route Voicemeeter output to real speakers.

**Result:** All audio from all apps passes through the VST plugin transparently.

**Diagram:**
```
Any app (Chrome/VLC/Spotify)
    ↓ (default playback = VB-Cable)
VB-Cable (virtual driver)
    ↓
Voicemeeter (virtual mixer)
    ↓ [VST effect slot → Music Assassin RT plugin]
Real speaker output
```

**Latency:** Voicemeeter adds ~10–30 ms buffer overhead. Combined with a real-time ML model, total ~40–100 ms. Acceptable for video watching.

**No driver writing required.** Music Assassin would need to ship as a VST3 plugin (or use an intermediate like a Voicemeeter VBAN/MIDI bridge).

**Alternative without Voicemeeter:** Use only VB-Cable + a WASAPI loopback reader in Python:
```python
# Python reads from VB-Cable monitor (WASAPI loopback),
# processes with ML, writes to real output device
import sounddevice as sd
# Read from VB-Cable output, process chunk, write to speakers
```

---

### Linux — PipeWire filter-chain

**What it is:**
PipeWire (replaces PulseAudio + JACK on modern Linux) has a built-in `filter-chain` module that creates a processing node in the audio graph. Supports LADSPA, LV2, and — crucially — **ONNX models directly**.

**From PipeWire docs:**
> "There is an optional ONNX filter available (when compiled with libonnxruntime) that can be selected with the `onnx` type."

**How it works:**
1. Create a PipeWire filter-chain node that wraps an ONNX model.
2. Connect it between the system sink (speakers) and applications in the PipeWire graph.
3. PipeWire routes all audio through it in real-time with RT process callbacks.

**Example filter-chain config** (conceptual):
```json
{
  "filter.graph": {
    "nodes": [
      {
        "type": "onnx",
        "name": "music_removal",
        "label": "/path/to/separation_model.onnx",
        "control": { "stem": "vocals" }
      }
    ]
  }
}
```

**This is the cleanest Linux architecture.** No Python overhead in the audio path — the ONNX model runs directly in PipeWire's C++ RT thread. Python is only used for model conversion and configuration.

**Latency:** PipeWire quantum (default 1024 samples @ 48kHz = ~21 ms) + model inference time. If the model inference completes within one quantum, total latency = 2× quantum = ~42 ms.

**Current limitation:** The separation models (Demucs, MDX-Net) are too slow to run within one PipeWire quantum on CPU. HS-TasNet or RT-STT would fit if compiled/quantized properly.

---

### Linux — PulseAudio null sinks (legacy approach)

On older systems without PipeWire:
1. Create a null sink: `pactl load-module module-null-sink sink_name=processed`
2. Set it as default output → apps stream to it.
3. A Python script reads from the monitor source of the null sink, processes it, writes to the real sink.

Workable but adds ~50–150 ms latency due to double-buffering through PulseAudio. PipeWire is strictly better.

---

### Cross-platform — VST3 / CLAP Plugin

**What it is:**
Music Assassin ships as a VST3 or CLAP plugin (`.vst3` / `.clap` file). Users load it in:
- Any DAW (Reaper, Ableton, FL Studio, Cubase, Logic) for offline file processing and real-time monitoring.
- Voicemeeter on Windows (for system-wide operation without a DAW).
- Carla plugin host on Linux (for system-wide operation).
- OBS Studio (via VST filter) for streaming.

**What writing a VST3 requires:**
- C++ using the Steinberg VST3 SDK, or JUCE framework (C++, abstracts VST3/AU/CLAP/AAX).
- The audio processing callback must run in a real-time thread — no Python GIL, no memory allocation.
- The ML model must be embedded as ONNX and called via ONNX Runtime C++ API.

**Python-to-VST bridge:** Not practical for production — Python has GIL issues and is too slow for RT audio callbacks. However, a **hybrid approach** works: C++ plugin shell + Python inference server:
```
VST3 callback (C++)
    ↓ shared memory / named pipe (< 1 ms IPC)
Python ML backend (running separately, GPU inference)
    ↑ returns processed buffer
VST3 output
```
This adds ~5–15 ms IPC overhead but allows keeping the ML logic in Python.

**JUCE framework** is the standard way to build cross-platform audio plugins and can wrap an ONNX Runtime call. JUCE is free for open-source projects.

---

## 4. VLC Integration — Is It Worth It?

**Short answer: No, VLC is not the right integration point.**

**Why:**
- VLC does not support VST plugins. This was requested by the community and rejected due to licensing complexity. No plans to implement.
- VLC has its own audio filter chain (`.dll`/`.so` modules in C), but writing a VLC audio filter module requires C, matching VLC's internal API, and recompiling.
- VLC is just one app — system-wide integration covers VLC and everything else simultaneously.

**What VLC does support:**
- `--audio-filter` modules (C, no Python, no easy ML).
- Audio output to a virtual device (VB-Cable / PipeWire node), so the virtual device approach covers VLC without touching VLC itself.
- `--av-offset` to compensate for processing delay (crucial for video sync if using offline chunked mode).

**Better approach for VLC users:**
Route VLC output through VB-Cable → Voicemeeter/WASAPI reader → ML processing → real speakers.  
VLC doesn't need modification. Processing is invisible to it.

**One scenario where VLC makes sense:** If you want Music Assassin to work *only* when VLC is open and *only* on what VLC is playing — then VLC's `transcode` module can call an external script. Still requires buffering and adds latency. Not worth the complexity.

---

## 5. How Volume Boosters / Device-Wide Apps Work

Examples: Dolby Atmos app, Boom3D, FxSound, DFX Audio Enhancer, Equalizer APO.

They all use one of two mechanisms:

### Mechanism 1 — Windows APO (Audio Processing Object)

Install a COM DLL that Windows loads into `audiodg.exe`. Processes all WASAPI shared-mode audio before it reaches the hardware driver. This is what Equalizer APO does — and what Dolby Atmos for Headphones uses.

**For Music Assassin:** Would need a C++ APO wrapper that calls the ONNX model. This is the most transparent (zero user configuration, works with all apps) but the hardest to build.

### Mechanism 2 — Virtual Audio Device Driver

Install a virtual soundcard driver (like VB-Cable does). Set it as the Windows/system default audio output. A user-space app reads from the virtual device, processes, writes to the real hardware device.

**For Music Assassin:** Ship a Python/Rust app that:
1. Creates (or relies on VB-Cable) a virtual input device.
2. Reads PCM from it.
3. Runs the ML model.
4. Outputs to the real speakers.

Users set "Music Assassin Virtual Speaker" as their default playback device once. Done. All apps go through it.

**Writing a Windows virtual audio driver** from scratch requires WDM kernel driver development (C/C++, Windows Driver Kit). Very complex. Better to distribute VB-Cable alongside Music Assassin and use it as the virtual device layer — VB-Cable is freeware.

### Mechanism 3 — WASAPI Loopback Capture (simpler, Windows only)

WASAPI has a "loopback" capture mode that lets you record what's currently playing on the speakers **without** a virtual device. Python's `sounddevice` and `pyaudiowpatch` support WASAPI loopback:

```python
import pyaudiowpatch as pyaudio
# Opens loopback capture from the default output device
# Processes audio, writes to output
```

**Limitation:** Introduces a feedback loop if you write back to the same default output. Must write to a different device or use a null sink approach.

### Mechanism 4 — PipeWire graph (Linux)

On Linux, PipeWire's graph is fully programmable. You can insert a processing node anywhere in the graph without a kernel driver:

```bash
pw-loopback --capture-props='node.name=Music_Assassin_in' \
            --playback-props='node.name=Music_Assassin_out'
```

Then connect it in the graph with `pw-link`. No driver writing, no root, just graph topology.

---

## 6. Streaming Architecture Design

The key design for any real-time pipeline is the **ring buffer + overlap-add** pattern:

```
Input stream (44100 Hz, stereo)
    ↓
Ring buffer (2× chunk size)
    ↓ [chunk ready trigger]
Chunk extraction with overlap (e.g. 50% hop)
    ↓
Normalization + windowing (Hann window)
    ↓
ML model inference (ONNX / Torch)
    ↓
Output chunk (vocals stem only)
    ↓
Overlap-add reconstruction
    ↓
Output ring buffer
    ↓
Playback
```

### Key parameters

| Parameter | Trade-off |
|---|---|
| **Chunk size** | Larger → better quality (more context), more latency |
| **Hop size** | 50% overlap → smooth transitions, 2× compute; 75% → 4× compute but smoother |
| **Overlap-add window** | Hann window eliminates blocking artifacts at chunk boundaries |
| **Model quantization** | INT8 → 2–4× speedup, ~0.3 dB SDR loss |
| **Thread separation** | Inference thread + audio I/O thread must not block each other |

### Threading model

```
Audio I/O thread (RT priority)
    Writes input to ring buffer
    Reads from output ring buffer → soundcard

ML inference thread (background, non-RT)
    Reads from input ring buffer when chunk ready
    Runs model
    Writes to output ring buffer

Ring buffer must be lock-free (atomic operations, no mutex in RT thread)
```

Music Assassin already does chunked processing. The upgrade for real-time is:
1. Replace file I/O with sounddevice stream callbacks.
2. Use a lock-free ring buffer (e.g. `python-sounddevice` with `sd.RawInputStream`).
3. Keep inference on a separate thread.
4. Feed output to `sd.RawOutputStream` with matching latency offset.

### Minimum viable real-time in Python (proof-of-concept)

```python
import sounddevice as sd
import numpy as np
from collections import deque
import threading

CHUNK_SAMPLES = 4096  # ~93 ms @ 44100 Hz
HOP = CHUNK_SAMPLES // 2
OVERLAP_BUFFER = np.zeros((CHUNK_SAMPLES, 2))

input_buf = deque(maxlen=CHUNK_SAMPLES * 4)
output_buf = deque(maxlen=CHUNK_SAMPLES * 4)

def audio_callback(indata, outdata, frames, time, status):
    input_buf.extend(indata)
    if len(output_buf) >= frames:
        outdata[:] = np.array([output_buf.popleft() for _ in range(frames)])
    else:
        outdata.fill(0)  # underrun — output silence

def inference_loop(model):
    while True:
        if len(input_buf) >= CHUNK_SAMPLES:
            chunk = np.array([input_buf.popleft() for _ in range(CHUNK_SAMPLES)])
            # Run model, get vocals stem
            vocals = model.separate(chunk)
            output_buf.extend(vocals)

# Start inference thread + stream
threading.Thread(target=inference_loop, args=(model,), daemon=True).start()
with sd.Stream(callback=audio_callback, latency='low', channels=2, samplerate=44100):
    sd.sleep(int(1e9))
```

The model must be fast enough that `inference_loop` empties and refills `output_buf` before the audio callback runs out of data. With a fast enough model (HS-TasNet or RT-STT at INT8), this is achievable on modern CPUs.

---

## 7. Existing Real-Time Implementations to Study

| Project | What it does | Why it's relevant |
|---|---|---|
| [Vocal Gate (free, 2026)](https://bedroomproducersblog.com/2026/04/02/vocal-gate-dan-k/) | AI noise gate VST3 plugin (speech isolation, removes mic artifacts) | Free, open-source, shows how to ship AI as a VST3 on all platforms |
| [VOXLESS (commercial)](https://www.audiosourcere.com/products/voxless-real-time-vocal-extraction-plugin-for-daws) | Real-time vocal extraction VST/AU/AAX | Proof the category exists; study their buffer size recommendations |
| [Equalizer APO](https://sourceforge.net/projects/equalizerapo/) | Windows APO EQ that supports VST chains | APO integration path for Music Assassin on Windows |
| [Voicemeeter (VB-Audio)](https://vb-audio.com/Voicemeeter/) | Virtual audio mixer with VST slots | Enables system-wide processing via virtual routing |
| [VB-Cable](https://vb-audio.com/Cable/) | Free virtual audio device (Windows) | The virtual cable layer for Voicemeeter approach |
| [PipeWire filter-chain](https://docs.pipewire.org/page_module_filter_chain.html) | Linux audio graph processing with ONNX support | Native ONNX in audio path on Linux — the ideal Linux deployment |
| [BSRNN hearing aid implementations](https://arxiv.org/pdf/2402.17701) | Causal BSRNN for real-time speech enhancement | Architecture pattern for causal streaming |
| [asteroid library](https://github.com/asteroid-team/asteroid) | Python: Conv-TasNet and real-time-capable models | Provides pre-trained causal separation models |
| [HS-TasNet reference impl](https://github.com/temismink/HS-Tasnet) | 23 ms latency hybrid model | The model to use if building true real-time |
| [Koala Noise Suppressor (Picovoice)](https://picovoice.ai/blog/voice-isolator/) | Cross-platform RT voice isolation (Python/mobile/web) | Shows real-time voice isolation from noise works on CPU at scale |

---

## 8. Implementation Options for Music Assassin

Ordered from least to most effort, least to most capability:

---

### Option A — AV Offset Mode (current + tweak, near-zero effort)

**What it is:** Don't make it real-time at all. Add a "live mode" that processes a short look-ahead buffer and tells the user to add an AV offset in their player.

**How:** Process audio in 2–3 second chunks with 50% overlap. Output the cleaned audio to a virtual device. User sets VLC/mpv's `--av-offset` to match the processing delay.

**Latency:** 2–5 seconds (fine for solo audio listening, not for video).

**Effort:** Days. Just stream from sounddevice instead of file.

**Limitations:** Not truly real-time. Requires the user to configure the player offset. Breaks lip sync.

---

### Option B — Python Streaming Daemon + Virtual Device (medium effort)

**What it is:** Music Assassin runs as a background daemon. Reads from VB-Cable (Windows) or PipeWire virtual sink (Linux). Processes with a fast model. Outputs to real speakers.

**Architecture:**
```
All apps → VB-Cable / PW virtual sink
              ↓
    Music Assassin daemon (Python)
         sounddevice InputStream
              ↓
    MDX-Net ONNX (fast, ~200 ms latency on CPU)
    or RT-STT quantized (if available)
              ↓
    sounddevice OutputStream → real speakers
```

**Latency:** 200–500 ms with MDX-Net (acceptable for music listening, too much for live voice).

**User setup (Windows):**
1. Install VB-Cable (free download).
2. Set VB-Cable as default Windows playback.
3. Run Music Assassin daemon.
Done. Works with every app.

**User setup (Linux):**
```bash
# Create virtual sink
pactl load-module module-null-sink sink_name=ma_sink
# or with PipeWire:
pw-cli create-node adapter ...
# Run Music Assassin daemon
python music_assassin_daemon.py --input ma_sink.monitor --output hw:0
```

**Effort:** 1–2 weeks for a working prototype.

---

### Option C — PipeWire ONNX filter-chain (Linux, low-level, cleanest)

**What it is:** Convert the separation model to ONNX, deploy directly inside PipeWire's filter-chain module. No Python in the audio path. Lowest possible latency on Linux.

**Steps:**
1. Export HTDemucs/MDX-Net/RT-STT to ONNX (already possible via `torch.onnx.export`).
2. Write a PipeWire filter-chain config that loads the ONNX model.
3. PipeWire handles all routing, scheduling, and real-time safety.

**Latency:** 1–2× PipeWire quantum (21–42 ms @ 48kHz with 1024-sample quantum) — only if the model is fast enough. With INT8 quantized RT-STT, achievable. With Demucs, not yet.

**Effort:** 2–4 weeks (model export + PipeWire config + testing).

**Limitation:** Linux-only. Requires PipeWire (modern Linux). Model must fit in one quantum for true low latency.

---

### Option D — VST3 Plugin via JUCE (cross-platform, hardest, most distributable)

**What it is:** Music Assassin ships as a VST3/CLAP plugin. Works in any DAW, Voicemeeter, OBS, Carla. Users install it once and use it anywhere.

**Architecture:**
- JUCE project in C++.
- Audio processing callback calls ONNX Runtime C++ API.
- Model weights embedded in plugin bundle or downloaded on first run.
- GUI in JUCE (can be minimal — just a bypass toggle and suppression strength slider).

**For system-wide on Windows:**
Users load it in Voicemeeter's effects chain with VB-Cable routing. One-time setup.

**For system-wide on Linux:**
Users load it in Carla (plugin host) connected to PipeWire graph.

**For DAW users:**
Just insert on a track.

**Latency:** Buffer-size dependent (typically 256–512 samples = 5–12 ms @ 44100 Hz, plus model inference time). With a fast model, total 20–50 ms.

**Effort:** 4–8 weeks minimum (C++ + JUCE + ONNX integration). Significant but gives a professional, distributable product.

---

### Option E — Equalizer APO APO Module (Windows-only, deeply integrated)

**What it is:** Implement Music Assassin as a Windows APO (Audio Processing Object) registered as a system enhancement. Works like Dolby Atmos — shows up in Windows Sound settings as an enhancement, always active when enabled.

**How APO works:**
- COM DLL implementing `IAudioProcessingObject` interface.
- Registered in Windows registry under the audio device's effect chain.
- Runs inside `audiodg.exe` at system level.
- No user has to set a default device — it's always there.

**Effort:** 6–12 weeks. Requires C++, COM, Windows Driver Kit, signing the DLL. Very complex.

**Alternative:** Wrap inside Equalizer APO's plugin framework (Equalizer APO supports convolver and VST chains). Less integration but much simpler.

---

## 9. Comparison Table

| Option | Latency | Platforms | User Effort | Dev Effort | Works with All Apps | Requires Driver |
|---|---|---|---|---|---|---|
| A — AV offset mode | 2–5 s | All | Medium (set offset) | Low | Yes (with player config) | No |
| B — Python daemon + VB-Cable | 200–500 ms | Win/Linux | Low (set default device) | Medium | Yes | No (VB-Cable) |
| C — PipeWire ONNX filter | 21–42 ms | Linux only | Low (config file) | Medium | Yes | No |
| D — VST3 plugin (JUCE) | 20–50 ms | All | Low (install plugin) | High | Via Voicemeeter/Carla | No |
| E — Windows APO | 5–15 ms | Windows only | Zero | Very High | Yes | Yes (DLL signing) |
| VLC plugin | n/a | VLC only | Low | High | No | No |

---

## 10. Recommended Path

### For Music Assassin specifically, the pragmatic progression:

**Phase 1 — Python streaming daemon (Option B)**
- Use `sounddevice` for real-time I/O.
- Use VB-Cable on Windows, PipeWire null sink on Linux as the virtual device.
- Run MDX23C or MDX-Net ONNX for inference (~200–400 ms latency).
- Ship as a tray app (system tray icon, enable/disable, model selector).
- This covers the primary use case: removing background music from videos while watching, podcasts, video calls.
- **Effort: 2–4 weeks from current codebase.**

**Phase 2 — Reduce latency with RT-STT / HS-TasNet**
- Replace offline models with streaming-designed models.
- Target 50–100 ms total latency.
- Still delivered as Python daemon, no driver work.
- **Effort: 2–3 weeks additional.**

**Phase 3 — VST3 plugin (JUCE + ONNX)**
- For DAW users and pro integration.
- JUCE wraps the same ONNX model.
- Users can load it in Voicemeeter for system-wide use.
- **Effort: 4–8 weeks, requires C++ work.**

**Phase 4 — PipeWire native (Linux)**
- When RT-STT or HS-TasNet ONNX export is clean, submit as a PipeWire filter-chain config.
- Zero-Python audio path, proper RT scheduling.
- **Effort: 1–2 weeks once Phase 2 is done.**

### VLC specifically
Do not integrate with VLC directly. Instead, in the Music Assassin tray app, add a "VLC mode" button that:
1. Activates the virtual device routing.
2. Reads VLC's current AV offset via VLC's HTTP interface (VLC has a REST API at `localhost:8080`).
3. Sets the offset automatically to compensate for processing latency.

This gives a one-click VLC experience without touching VLC's internals.

---

## Key Libraries

| Library | Purpose | Platform |
|---|---|---|
| `sounddevice` | Python audio I/O (PortAudio wrapper), streams, callbacks | All |
| `pyaudiowpatch` | WASAPI loopback capture on Windows | Windows |
| `pipewire-python` | PipeWire graph control from Python | Linux |
| `onnxruntime` | Fast ONNX inference (CPU/GPU), already used | All |
| JUCE | C++ framework for VST3/AU/CLAP/AAX plugins | All |
| ONNX Runtime C++ | Same model, embedded in C++ VST | All |
| VB-Cable (VB-Audio) | Free virtual audio device, no coding needed | Windows |
| Equalizer APO | APO with VST/LADSPA chain support | Windows |
| Voicemeeter | Virtual mixer that hosts VST effects system-wide | Windows |
| Carla | Plugin host for system-wide VST on Linux | Linux |
| asteroid | Python: Conv-TasNet, causal models for real-time | All |

---

## 11. Update 2026-07-04 — New Options Found & Measured

> For a single master table with every method from this section (and rounds
> 4–5 below) in one consistent column set — size, time/RAM per minute,
> pre/post-processing, RT-suitability, SFX handling, best-use pattern — see
> `docs/COMPREHENSIVE_METHOD_COMPARISON.md`. This section remains the detailed
> narrative/methodology record.

Research pass over what appeared since the sections above were written, plus live
measurements on the dev machine (4-core CPU, 20 s dialogue clip @ 44.1 kHz mono,
same clip and machine as `docs/MODEL_BENCHMARKS_v0.4.4.md`).

> **⚠ Key finding (2026-07, confirmed in live-app measurements): speech-enhancement
> models do not remove music.** Every enhancer in this document — DeepFilterNet,
> DPDFNet (all variants), GTCRN, SpeechDenoiser, DTLN, MetricGAN+ — attenuates
> noise at −29 to −63 dB but passes prominent or vocal-heavy music nearly
> untouched (−0.4 to −1.7 dB): they classify music as speech. Enhancers below are
> evaluated strictly as *post-separation cleanup* for quiet residuals. True music
> removal requires a source-separation model, and a **realtime-capable separator
> is the key open problem of this repo** — current candidates: chunked
> sherpa-onnx Spleeter int8 loop (RTF 0.067, measured below), training HS-TasNet
> on a free GPU, or RT-STT if code ever ships.

### sherpa-onnx now does source separation — measured, and it changes the picture

sherpa-onnx (≥ the 1.13.3 already installed) gained an `OfflineSourceSeparation`
API running **Spleeter 2-stem** and **UVR/MDX models** (converted from the same
TRvlvr repo the audio-separator models come from) through its lean C++ runtime.
Measured on this machine:

| Config | Time / 60 s audio | RTF | Peak RSS | vs. current stack |
|---|---|---|---|---|
| **Spleeter 2-stem int8, 1 thread** | **4.0 s** | **0.067** | **~600 MB** | ~34× faster, ~5× less RAM than audio-separator MDX |
| Spleeter 2-stem int8, 4 threads | 3.9 s | 0.064 | ~615 MB | threads don't matter — already I/O-bound |
| sherpa UVR Voc_FT, 4 threads | 84.4 s | 1.41 | ~4.3 GB | ~1.6× faster than audio-separator (134.7 s), slightly more RAM |
| sherpa UVR Voc_FT, 1 thread | 143.4 s | 2.39 | ~4.3 GB | published RTF 0.59-0.73 is from a much faster CPU |

Notes:
- **Spleeter int8 is the first separator measured on this machine that is
  genuinely real-time capable** (15× headroom at 1 thread). Quality is the open
  question — Spleeter 2-stem is architecturally weaker than MDX — but with the
  DF→VAD chain after it, listening comparison files exist:
  `test_outputs/bench_dialogue07m00_sherpa_spleeter_i8_full.wav` vs
  `bench_dialogue07m00_sherpa_uvr_vocft_full.wav` vs the earlier
  `bench_dialogue07m00_audio_separator_UVR-MDX-NET-Voc_FT.wav`.
- The sherpa Spleeter STFT implementation **requires stereo input** `(2, N)`
  (crashes on mono with "Invalid channel 1. Max 1"); duplicate the mono channel.
- The UVR path is the same Voc_FT weights as the current default separator, so
  its output quality is identical by construction — the 1.6× speedup is pure
  runtime efficiency. RAM is not better, though.
- Models: `models/sherpa_uvr_voc_ft.onnx` (64 MB),
  `models/sherpa-onnx-spleeter-2stems-int8/` (2 × 26 MB), downloaded from the
  [source-separation-models release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/source-separation-models).

### DPDFNet variants beyond baseline

The [speech-enhancement-models release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models)
now has, beyond the `dpdfnet_baseline` (8.5 MB, 0.36 G MACs) already in `models/`:

| Model | Params | MACs | Size | Billed as |
|---|---|---|---|---|
| dpdfnet2 | 2.49M | 1.35G | 9.9 MB | real-time / embedded |
| dpdfnet4 | 2.84M | 2.36G | 11.2 MB | balanced |
| dpdfnet8 | 3.54M | 4.37G | 14.1 MB | best enhancement quality |
| **dpdfnet2_48khz_hr** | 2.58M | 2.42G | 10.3 MB | **48 kHz high-resolution** |

`dpdfnet2_48khz_hr` matters most: the current dpdfnet path resamples
44.1k→16k→44.1k, discarding everything above 8 kHz — exactly the high-frequency
content of the high-pitched voices this project fights for. Untested by ear here.

### Other candidates worth knowing about (not yet tried)

- **DeepFilterNet official LADSPA plugin** (`deep-filter-ladspa`, Rust, in the
  [DeepFilterNet repo](https://github.com/Rikorose/DeepFilterNet/tree/main/ladspa)) —
  real-time DeepFilterNet3 inside a PipeWire `filter-chain`, no Python. Directly
  compatible with the routing already validated in `docs/LIVE_APP_PLAN.md`.
- **DTLN** ([breizhn/DTLN](https://github.com/breizhn/DTLN), MIT) — ~1M-param
  dual-LSTM enhancer shipping as two tiny stateful-streaming ONNX models, 16 kHz.
- **RNNoise** ([xiph/rnnoise](https://github.com/xiph/rnnoise), BSD) + the
  [werman LADSPA build](https://github.com/werman/noise-suppression-for-voice) —
  nearly-free 48 kHz real-time denoise; quality below DeepFilterNet.
- **HS-TasNet status (re-checked 2026-07)**: still no public weights.
  [lucidrains' MIT implementation](https://github.com/lucidrains/HS-TasNet) is
  mature (training code, real-time sounddevice example) — training on MusDB via a
  free Colab/Kaggle GPU is the realistic route to weights. RT-STT
  ([arxiv 2511.13146](https://arxiv.org/abs/2511.13146)) still has no code release.

### Free DSP wins currently unused

- **Stereo mid/side pre-filter** — the pipeline currently downmixes to mono
  (`-ac 1`) before anything runs, discarding the strongest cheap discriminator in
  anime mixes: dialogue is center-panned, music/SFX are stereo-wide. Keeping
  `(L+R)/2` and attenuating per-bin by side-channel energy is causal, ~free on
  CPU, and would strip wide-panned music *before* any model.
- **AVX512_VNNI** — this CPU has it (visible in TF startup logs). It accelerates
  int8 inference; int8 models (Spleeter above, quantized UVR) get it for free in
  onnxruntime, and the OpenVINO EP (optional line in requirements.txt) may add more.

### Measured 2026-07-04 (round 3): enhancer alternatives + mid/side

All enhancers tested in the same pipeline slot (sherpa-UVR vocals → DSP →
*enhancer* → VAD) on the 20 s dialogue clip; listening files in `test_outputs/`
as `bench_dialogue07m00_sherpa_uvr_<tag>_full.wav`.

| Enhancer | Time / 60 s | RAM delta | Bandwidth | Notes |
|---|---|---|---|---|
| DeepFilterNet3 (`atten_lim_db=20`) | 16.2 s | ~585 MB | 48 kHz | current default, quality reference |
| **dpdfnet2_48khz_hr** | 16.9 s | **~40 MB** | **48 kHz** | same speed as DF at 1/14th the RAM; quality by ear pending |
| **MetricGAN+** (SpeechBrain — the v0.4.2 method) | **9.8 s** | ~30 MB (after ~500 MB SpeechBrain+torch load) | 16 kHz only | faster than DF, but everything above 8 kHz is discarded by the 16 k resample — a real cost for high-pitched voices |

On the SpeechBrain "AI filter + DSP filter" approach in general: that is exactly
this app's current architecture (DSP pipeline + neural enhancer in series); the
only question is which neural filter fills the slot. MetricGAN+ is competitive on
speed but its 16 kHz ceiling makes it strictly worse than DeepFilterNet or
dpdfnet2_48khz_hr for this content unless CPU is the binding constraint.
SpeechBrain's separation models (SepFormer family) were not tested: they are
non-causal, heavy on CPU, and 8/16 kHz — worse than the sherpa options on every
axis that matters here.

**Mid/side center extraction, measured + tuned** (OP clip, music-only gap
4.48–7.81 s vs vocal segment 0.42–2.85 s, 200–4000 Hz band energy):

| Variant | Voice/music ratio | Vocal loss |
|---|---|---|
| plain mono downmix | 9.2× | — |
| center extract, gain `g = M²/(M²+S²)` | 12.0× | −0.00 dB |
| `g²` | 13.9× | −0.01 dB |
| `g⁴` | **16.8×** | **−0.01 dB** |

~1.8 dB of free music suppression at `g⁴` with unmeasurable vocal damage, at
4.4 s per audio-minute (unoptimized scipy STFT; trivially cheaper in C or with
smaller FFTs). Caveats: single-clip metric; aggressive exponents can introduce
"musical noise" artifacts that a gap-energy number won't show — listening files
`bench_op10s_midside_center{,_p2,_p4}.wav` vs `bench_op10s_mono_ref.wav` are the
real test. Worth wiring in as an optional stereo pre-filter stage before the
separator, and as a near-free first stage for the live app.

**Still untested:** DTLN (streaming ONNX enhancer — needs custom two-model state
wiring); dpdfnet8 (best-quality 16 k variant); sherpa Spleeter quality on singing
content (only dialogue clip compared so far); int8-quantized UVR variants.
→ All tested in round 4 below (no int8 UVR exists in the sherpa release; skipped).

### Measured 2026-07-04 (round 4): parallel architectures + remaining models

**Music-subtraction architecture** — instead of extracting vocals, predict the
*music* stem and compute `vocals = mix − predicted_music`. Because instrumental
models (UVR-MDX-NET-Inst family) are trained to recognize music specifically,
SFX is not part of what they subtract — so dialogue **and SFX** survive. This is
the model-based answer to "the AI removes SFX that isn't music":

- Tested `UVR-MDX-NET-Inst_HQ_3` (63 MB, via sherpa-onnx). On the dialogue clip
  its music prediction is near-silent (rms 0.008 — correct: barely any music
  there) and the subtraction output carries the rest (rms 0.097).
- Cost: 108–120 s per audio-minute (RTF ~1.9), ~4.3 GB — same class as Voc_FT,
  slightly slower.
- Listening: `test_outputs/r4_dialogue_vocals_by_subtraction.wav` (dialogue,
  should retain SFX) and `r4_op_vocals_by_subtraction.wav` (OP control — music
  removal quality on singing content).

**Parallel DSP residual add-back** (SFX recovery) — the complementary approach:
`residual = mix − vocals_stem` contains everything the separator removed
(music + SFX); a DSP mask (HPSS median-filter transient test × spectral-flatness
gate) re-adds only SFX-like bins. Add-back happens *after* the VAD gate so gated
SFX isn't lost. Mixture-consistent (adds back untouched original signal — no
processing artifacts possible, only music leakage). Measured leakage on the OP
clip (pure music — any add-back is failure):

| Mask variant | OP music leakage |
|---|---|
| v1 (soft percussive × flatness σ@0.30) | 20.5% |
| v2-strict (hard P>2H test × flatness σ@0.40) | 11.6% |
| v3-very-strict (P>3H × flatness σ@0.50) | 8.3% |

DSP features plateau around ~8–10% leakage: drums are transient and noisy, just
like SFX — no frame-level DSP statistic fully separates them. Conclusion: the
music-subtraction architecture above is the better route to SFX preservation;
the DSP add-back is only viable with a semantic classifier (e.g. YAMNet, 3.7M
params, AudioSet classes, CPU-real-time) deciding music-vs-SFX per frame instead
of flatness. Files: `r4_sfx_addback_v2_dialogue.wav` (recovered SFX track),
`sfx_addback_dialogue_*.wav` from v1.

**Remaining models, measured** (same slot/clip as round 3):

| Model | Time / 60 s | RAM delta | Notes |
|---|---|---|---|
| dpdfnet8 (16 kHz, "best quality") | 29.9 s | ~64 MB | ~2× slower than DeepFilterNet *and* bandwidth-capped at 8 kHz — dominated by dpdfnet2_48khz_hr on every axis; not recommended |
| **DTLN (streaming, 16 kHz)** | **4.6 s** | **~1 MB** | genuinely stateful streaming (32 ms frames), two ONNX models totalling 3.8 MB, ran in pure Python at 13× real-time — the cheapest enhancer measured; 16 kHz ceiling is its only real cost. Listening: `r4_dialogue_uvr_dtln.wav` |
| sherpa Spleeter int8 on singing (OP) | (speed as before) | — | quality-by-ear file: `r4_op_spleeter_vocals.wav` vs `r4_op_vocft_vocals.wav` (Voc_FT reference on same clip) |

int8-quantized UVR models do not exist in the sherpa release (fp32 only) —
quantizing one ourselves via onnxruntime dynamic quantization remains an open
option if UVR speed matters more than the Spleeter path.
→ Tested in round 5 below — result: don't.

### Measured 2026-07-07 (round 5): YAMNet classifier, self-quantization, alternate subtraction models, OpenVINO, new content type

**YAMNet semantic classifier for SFX add-back — the fix for round 4's DSP
leakage floor.** Round 4 found pure-DSP masks (transient × flatness) plateau at
8.3% music leakage on the OP control clip. Swapped the DSP mask for a real
classifier: [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
(3.7M params, 521 AudioSet classes, weights at
`https://storage.googleapis.com/audioset/yamnet.h5`, 15 MB) run per 0.48 s frame
on the separation residual; a frame is kept for add-back if its top-scoring
class isn't music/song/singing and it isn't silence.

| Approach | OP-control leakage | Dialogue residual kept |
|---|---|---|
| DSP v3-very-strict (round 4) | 8.3% | 5.3% |
| **YAMNet classifier** | **0.5%** | **94.1%** |

An order of magnitude tighter rejection of music, while keeping far more of the
real residual content. Caveat: YAMNet's absolute confidence scores on this
input are low (0.001–0.09 range) because a separator *residual* — a
difference/artifact signal — is out-of-distribution from the natural audio
YAMNet was trained on; "top class" can still be a meaningful signal even when
no class scores high confidence, but this deserves a by-ear check, not just the
energy metric, before trusting it as strongly as the OP-control number suggests.
Practical note: the original 2019 `yamnet.py` doesn't run as-is on TF 2.21 /
Keras 3 — it builds a Functional-model graph by calling raw `tf.*` ops (STFT,
padding) directly on a symbolic `Input()` tensor, which Keras 3 forbids, and
uses `/` in layer names, which Keras 3 also forbids. Fix: run feature extraction
(STFT/mel, plain TF ops, no Keras dependency) eagerly outside any graph, and
build only the CNN classifier head as a Functional model; `load_weights()` on
the original `.h5` still works because it matches by layer order, not name, so
renaming the `/`-containing names to `_` doesn't break checkpoint loading.
Cost: ~0.5 s inference per 20 s clip, single digit MB RAM beyond the TF import
itself. Files: `r5_yamnet_op_addback_LEAKAGE.wav`, `r5_yamnet_dialogue_addback_sfx.wav`.

**Self-quantizing UVR models to int8 — don't.** Tried `quantize_dynamic()`
(onnxruntime's generic dynamic quantization) on Inst_HQ_3, expecting a speed/RAM
win the way htdemucs's INT8 quantization gives 2-3×. Result was the opposite:

| | fp32 (round 4 baseline) | int8 (self-quantized) |
|---|---|---|
| File size | 63.7 MB | 17.9 MB |
| Time / 60 s audio | 108 s | **399.8 s (3.7× slower)** |
| Peak RSS | ~4.3 GB | ~4.1 GB (no real reduction) |
| Quality vs fp32 | — | 99.28% correlation, but 11.97% RMS difference |

Generic dynamic quantization only pays off when a model is Linear/MatMul-bound
(htdemucs's quantization specifically targets Linear layers). This UVR model's
cost is dominated by Conv2D/STFT-shaped ops and large intermediate activation
buffers — quantizing the weights doesn't touch that, and dynamic int8 adds
runtime quantize/dequantize overhead around each op, making it slower with a
real quality cost and no RAM benefit. Not worth pursuing further without
static/calibrated quantization or an architecture-aware quantization recipe.

**Alternate subtraction-architecture predictors + a new content type.** All
prior rounds only used the 00:10 (OP, singing) and 07:00 (dialogue) clips.
Added a 16:00 clip (new territory — this episode's action/dramatic content,
confirmed different: music_rms 0.07-0.12 here vs 0.008 on the quiet dialogue
clip) and compared three UVR "predict-the-non-vocal-stem" models on it:

| Model | Time / 60 s | Peak RSS | music_rms | subtraction_rms | Verdict |
|---|---|---|---|---|---|
| UVR-MDX-NET-Inst_HQ_3 (round 4 baseline) | 82.1 s | 4288 MB | 0.0716 | 0.1074 | reference |
| **UVR_MDXNET_KARA_2** | **50.2 s (1.6× faster)** | **2824 MB (34% less)** | 0.0692 | 0.1067 | near-identical output to Inst_HQ_3 at meaningfully lower cost — best subtraction candidate found so far |
| UVR-MDX-NET_Crowd_HQ_1 | 64.6 s | 3117 MB | 0.1239 | 0.0228 | poor fit — over-subtracts (its "non-vocal" class is trained on crowd/audience noise, not anime background music, so it likely mistakes dialogue for crowd noise and strips too much) |

KARA_2 (a karaoke-instrumental-prediction model) landing this close to Inst_HQ_3
in output energy, at 1.6× the speed and a third less RAM, makes it the new
default candidate for the subtraction architecture — pending a by-ear check.
Files: `r5_action16m00_{insthq3,kara2,crowdhq1}_{music_pred,subtraction}.wav`,
raw clip at `r5_action16m00_raw.wav`.

**OpenVINO execution provider — modest, real, but not for the sherpa-onnx path.**
`onnxruntime-openvino` was never actually benchmarked despite being an optional
line in `requirements.txt`. Installed it in an isolated throwaway venv (kept out
of the main dev venv — sherpa-onnx ships its own statically-built onnxruntime,
so installing `onnxruntime-openvino` there wouldn't even take effect, and risked
conflicting with the working `onnxruntime` install used by sherpa-onnx and
audio-separator). Ran the same UVR model's raw ONNX graph (fixed-shape
spectrogram input, bypassing both wrapper libraries) under both providers:

| Provider | Avg inference time |
|---|---|
| CPUExecutionProvider | 3466.8 ms |
| OpenVINOExecutionProvider | **2908.6 ms (~16% faster)** |

Real but modest — `get_providers()` showed OpenVINO falls back to
`CPUExecutionProvider` for ops it doesn't support, capping the win. Only
reachable today via the `audio-separator` path (plain onnxruntime, respects
externally-installed providers); sherpa-onnx would need to be rebuilt from
source with OpenVINO EP compiled in to benefit — not attempted.

---

## References

- [HS-TasNet paper — arxiv 2402.17701](https://arxiv.org/abs/2402.17701)
- [HS-TasNet GitHub (lucidrains)](https://github.com/lucidrains/HS-TasNet)
- [HS-TasNet GitHub (original)](https://github.com/temismink/HS-Tasnet)
- [RT-STT paper — arxiv 2511.13146](https://arxiv.org/abs/2511.13146)
- [Towards Practical Real-Time MSS — arxiv](https://arxiv.org/pdf/2511.13146)
- [PipeWire filter-chain docs](https://docs.pipewire.org/page_module_filter_chain.html)
- [PipeWire DSP filter tutorial](https://docs.pipewire.org/devel/page_tutorial7.html)
- [PipeWire ArchWiki](https://wiki.archlinux.org/title/PipeWire)
- [Equalizer APO (SourceForge)](https://sourceforge.net/projects/equalizerapo/)
- [Voicemeeter (VB-Audio)](https://vb-audio.com/Voicemeeter/)
- [VB-Cable (VB-Audio)](https://vb-audio.com/Cable/)
- [VOXLESS real-time VST](https://www.audiosourcere.com/products/voxless-real-time-vocal-extraction-plugin-for-daws)
- [Vocal Gate free VST3 plugin](https://bedroomproducersblog.com/2026/04/02/vocal-gate-dan-k/)
- [Demucs streaming issue #515](https://github.com/facebookresearch/demucs/issues/515)
- [Picovoice voice isolator guide](https://picovoice.ai/blog/voice-isolator/)
- [Virtual audio cables blog](https://blog.habets.se/2021/12/Virtual-audio-cables.html)
- [PipeWire virtual sink creation](https://craigwilson.blog/post/2024/2024-12-05-virtualaudiosplit/)
- [Moises Stems VST plugin](https://moises.ai/features/stems-vst-plugin/)
- [L-Acoustics HS-TasNet technical paper (PDF)](https://www.l-acoustics.com/wp-content/uploads/2024/04/real_time_demixer_2024_04_19.pdf)
