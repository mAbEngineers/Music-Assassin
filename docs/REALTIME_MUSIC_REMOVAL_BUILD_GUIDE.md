# Real-Time Music Removal App — Technical Build Guide
## Desktop (Linux / Windows) + Mobile

> **Goal:** Capture the system's audio output live, strip background music in real-time
> using on-device AI, and play back vocals + SFX with minimal latency — no cloud,
> no GPU required.

---

## Table of Contents
1. [Core Architecture](#1-core-architecture)
2. [Audio Capture Layer — Linux](#2-audio-capture-layer--linux)
3. [Audio Capture Layer — Windows](#3-audio-capture-layer--windows)
4. [AI Separation Models (ONNX / CPU)](#4-ai-separation-models-onnx--cpu)
5. [Inference Runtime Options](#5-inference-runtime-options)
6. [Latency Budget](#6-latency-budget)
7. [Threading Architecture](#7-threading-architecture)
8. [Complete Dependency Lists](#8-complete-dependency-lists)
9. [Distribution / Packaging](#9-distribution--packaging)
10. [Mobile — Platform Constraints](#10-mobile--platform-constraints)
11. [Mobile — What Is Actually Possible](#11-mobile--what-is-actually-possible)
12. [Mobile — Tech Stack Options](#12-mobile--tech-stack-options)
13. [Recommended Build Path](#13-recommended-build-path)

---

## 1. Core Architecture

Real-time music removal is a **loopback pipeline**:

```
┌─────────────────────────────────────────────────────────────────┐
│                       Operating System                          │
│                                                                 │
│  App (Spotify, VLC, browser)                                    │
│       │                                                         │
│       ▼                                                         │
│  Virtual / Monitor Audio Device  ◄── your app reads from here  │
│       │                                                         │
│       ▼                                                         │
│  Your App: AI Processing Engine                                 │
│       │  block in (e.g. 1024 samples) → model → block out       │
│       ▼                                                         │
│  Physical Speaker / Headphone Output                            │
└─────────────────────────────────────────────────────────────────┘
```

The key constraint is **latency**: the AI model must finish processing one audio
block before the next one arrives. At 44 100 Hz with a 1024-sample block, you have
**23 ms** per block. On a typical modern CPU this is achievable with the right model.

---

## 2. Audio Capture Layer — Linux

Linux has three audio subsystems in common use. Ubuntu 22.04+ ships PipeWire by
default; older Ubuntu uses PulseAudio. Both are supported.

### 2.1 PipeWire (Ubuntu 22.04+, recommended)

PipeWire is a graph-based audio engine. Every app, device, and virtual node is a
**node** in the graph. You connect nodes with `pw-link` or `wpctl`.

**Step 1 — Identify your sink monitor:**
```bash
pactl list short sources | grep monitor
# Output e.g.:  "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
```

**Step 2 — In Python, open that monitor as your input device:**
```python
import sounddevice as sd

# List devices — find the ".monitor" entry
print(sd.query_devices())

# Open a stream reading from the monitor (loopback of system output)
with sd.Stream(device=('alsa_output...monitor', 'default'),
               samplerate=44100, blocksize=1024, channels=2,
               callback=your_callback):
    ...
```

**Step 3 — Route processed audio back to speakers:**
- Create a null sink as a staging device, or write directly to `default` output.
- For professional routing, use `pw-loopback` or load a loopback module:
```bash
pactl load-module module-loopback latency_msec=20
```

**Dependencies:**
```
pipewire pipewire-pulse pipewire-jack  (usually pre-installed on Ubuntu 22.04+)
python3-sounddevice   OR   pip install sounddevice
```

---

### 2.2 PulseAudio (Ubuntu 20.04 and older)

PulseAudio exposes a **monitor source** for every sink automatically.

```bash
# List monitor sources
pactl list short sources | grep monitor
# e.g. "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"

# In Python — use that string as your input device name in sounddevice
```

**Create a loopback to send processed audio back:**
```bash
pactl load-module module-null-sink sink_name=music_assassin_out
pactl load-module module-loopback source=music_assassin_out.monitor
```

---

### 2.3 JACK (professional / ultra-low-latency)

JACK gives the lowest achievable latency on Linux (< 5 ms with a real-time kernel).
Required for professional production use but adds setup complexity.

```bash
# Install
sudo apt install jackd2 qjackctl

# Start JACK server (example: 64-sample buffer = ~1.5 ms at 44100 Hz)
jackd -d alsa -r 44100 -p 64 -n 2

# In Python
pip install JACK-Client
```

```python
import jack
client = jack.Client('MusicAssassin')
inport  = client.inports.register('input_1')
outport = client.outports.register('output_1')

@client.set_process_callback
def process(frames):
    audio = inport.get_array()
    processed = your_model_inference(audio)
    outport.get_array()[:] = processed

with client:
    jack.attach('system:capture_1', 'MusicAssassin:input_1')
    ...
```

---

### 2.4 ALSA Loopback (kernel-level, no PulseAudio/PipeWire)

For headless / embedded deployments:
```bash
sudo modprobe snd-aloop           # Creates hw:Loopback,0 and hw:Loopback,1
# Route system audio into Loopback device 0, read from Loopback device 1
```

---

### 2.5 Linux Summary Table

| Method | Latency | Ease | Notes |
|---|---|---|---|
| PipeWire monitor | 20–50 ms | Easy | Default on Ubuntu 22.04+ |
| PulseAudio monitor | 30–80 ms | Easy | Older Ubuntu |
| JACK | 1–10 ms | Complex | Real-time kernel recommended |
| ALSA loopback | 5–20 ms | Medium | Kernel module, headless-friendly |

---

## 3. Audio Capture Layer — Windows

Windows audio is managed by the **Windows Audio Session API (WASAPI)**. No virtual
driver is needed for capture — WASAPI has a built-in **loopback mode** that taps the
output stream directly.

### 3.1 WASAPI Loopback (built-in, no driver needed)

```python
import sounddevice as sd

# Query devices to find your output device's loopback entry
devices = sd.query_devices()
# Look for your speaker/headphone device — sounddevice surfaces it automatically
# as a valid input when using WASAPI

# Open loopback stream
with sd.InputStream(device='Speakers (Realtek Audio)',   # name varies
                    samplerate=44100, blocksize=1024,
                    channels=2, extra_settings=sd.WasapiSettings(exclusive=False)):
    ...
```

**Key points:**
- Works on Windows 8.1+ with no extra software.
- Runs in **shared mode** (slightly higher latency) or exclusive mode (lower latency,
  locks the device — not recommended for live monitoring).
- sounddevice wraps WASAPI automatically when on Windows.

---

### 3.2 VB-Audio Virtual Cable (free, for routing)

VB-Cable creates a virtual input/output pair. Set it as the **default Windows
playback device**, then read from it as your capture source. This gives cleaner
routing than WASAPI loopback because other apps don't hear the processing artefacts.

**Install:** https://vb-audio.com/Cable/
**No code changes** — just select `CABLE Output` as your input device in sounddevice.

---

### 3.3 ASIO (ultra-low-latency, optional)

ASIO bypasses the Windows audio stack entirely. Achieves 2–10 ms latency.
Requires either a professional audio interface with native ASIO drivers, or
**ASIO4ALL** (free) for consumer hardware.

```bash
pip install sounddevice   # ASIO requires PyAudio build with ASIO support
```

Not required for music removal at normal monitoring latency (20–50 ms is inaudible
for this use case).

---

### 3.4 Windows Summary Table

| Method | Latency | Ease | Notes |
|---|---|---|---|
| WASAPI loopback | 20–60 ms | Very easy | No driver, built-in |
| VB-Cable + WASAPI | 20–40 ms | Easy | Cleaner routing, free |
| ASIO4ALL | 5–15 ms | Medium | Lowest latency on consumer hardware |
| ASIO (pro interface) | 2–5 ms | Easy (if you have one) | Professional hardware |

---

## 4. AI Separation Models (ONNX / CPU)

All models below run entirely on CPU with no GPU. They are sorted from
**fastest/lightest** to **slowest/heaviest**.

### 4.1 Streaming / Real-Time Models (stateful, block-by-block)

These maintain hidden state between blocks — designed for live processing.

#### GTCRN (Galvanized Tri-Scale CRN)
- **Architecture:** Sub-band complex recurrent network
- **Inference time:** ~2–4 ms per 1024-sample block on a modern CPU
- **Model size:** ~2 MB
- **Quality:** Good speech isolation; light music suppression
- **Format:** ONNX, via Sherpa-ONNX
- **Install:** `pip install sherpa-onnx`
- **Download:** Sherpa-ONNX model releases on GitHub
- **Best for:** Low-end CPUs, real-time monitoring

#### DPDFNet (Dual-Path DeepFilterNet)
- **Architecture:** DeepFilterNet2 + DPRNN blocks
- **Inference time:** ~5–10 ms per block
- **Model size:** ~4 MB
- **Quality:** Excellent cross-band temporal modelling; treats musical transients
  as noise and attenuates them well
- **Format:** ONNX, via Sherpa-ONNX
- **Install:** `pip install sherpa-onnx`
- **Best for:** Best ONNX streaming option for music removal

#### SpeechDenoiser ONNX
- **Architecture:** End-to-end RNN with complex IRM masking
- **Inference time:** ~3–6 ms per block
- **Model size:** ~3 MB
- **Quality:** Good; zero-masks harmonic music frequencies in the spectral domain
- **Format:** ONNX, plain onnxruntime (no extra package needed)
- **Download:** github.com/yuyun2000/SpeechDenoiser
- **Best for:** No-extra-dependency deployment

---

### 4.2 Offline / Batch Models (process whole file or long chunks)

These do not maintain hidden state and process chunks of several seconds.

#### DeepFilterNet (torch)
- **Architecture:** Deep filtering on sub-band ERB-scale spectrogram
- **Inference time:** Real-time factor ~0.3× on CPU (3 min file → ~1 min processing)
- **Model size:** ~8 MB weights
- **Quality:** Best overall for speech enhancement / residual music removal
- **Format:** PyTorch (torch-based, not pure ONNX)
- **Install:** `pip install deepfilternet`
- **Best for:** Post-processing after DEMUCS/MDX-Net separation

#### MDX-Net (UVR-MDX-NET / Kim_Vocal)
- **Architecture:** U-Net on STFT magnitude spectrogram
- **Inference time:** ~0.5–1× real-time on CPU
- **Model size:** 66 MB (Voc_FT) / 130 MB (Kim_Vocal_2)
- **Quality:** Excellent vocals-only extraction (music removal specialist)
- **Format:** ONNX via audio-separator
- **Install:** `pip install audio-separator onnxruntime`
- **Best for:** Primary music-removal separator (currently in Music Assassin)

#### DEMUCS htdemucs
- **Architecture:** Hybrid Transformer U-Net, 4-stem
- **Inference time:** ~2–4× real-time on CPU (can be reduced with INT8 quant)
- **Model size:** ~320 MB
- **Quality:** Best overall separation; handles complex mixes well
- **Format:** PyTorch; ONNX export possible via torch.onnx.export
- **Install:** `pip install demucs`
- **Best for:** Highest quality when speed is not critical

---

### 4.3 Model Comparison Table

| Model | Type | ONNX | Size | CPU Real-time? | Music Removal Quality |
|---|---|---|---|---|---|
| GTCRN | Streaming | ✓ Native | 2 MB | Yes (<5 ms/block) | Good |
| DPDFNet | Streaming | ✓ Native | 4 MB | Yes (~8 ms/block) | Very good |
| SpeechDenoiser | Streaming | ✓ Native | 3 MB | Yes (<6 ms/block) | Good |
| DeepFilterNet | Offline | ✗ (torch) | 8 MB | ~3× slower | Excellent |
| MDX-Net Voc_FT | Offline | ✓ via audio-sep | 66 MB | ~0.8× | Excellent |
| DEMUCS htdemucs | Offline | Optional export | 320 MB | ~3× slower | Best |

---

## 5. Inference Runtime Options

### 5.1 onnxruntime (baseline, always works)
```bash
pip install onnxruntime
```
Uses `CPUExecutionProvider` — works on any x86_64 / ARM CPU.

### 5.2 onnxruntime-openvino (recommended for speed)
```bash
pip install onnxruntime-openvino
```
Drop-in replacement. Automatically uses AVX2/AVX-512 on Intel and AMD CPUs,
ARM Neon on ARM. ~1.5–2.5× faster than plain CPUExecutionProvider.
Adds ~100–150 MB to distribution size.

```python
import onnxruntime as ort

providers = [
    ('OpenVINOExecutionProvider', {
        'device_type': 'CPU_FP32',
        'num_of_threads': os.cpu_count(),
        'cache_dir': '~/.cache/music-assassin/ov_cache',
    }),
    'CPUExecutionProvider',   # fallback
]
session = ort.InferenceSession(model_path, providers=providers)
print("Active provider:", session.get_providers()[0])
```

### 5.3 Sherpa-ONNX (for streaming models)
```bash
pip install sherpa-onnx
```
Wraps onnxruntime with model-specific streaming APIs (hidden state management,
correct chunking, sample rate handling). Required for DPDFNet and GTCRN.

```python
import sherpa_onnx

enhancer = sherpa_onnx.OfflineStream()   # or OnlineStream for true real-time
config   = sherpa_onnx.SpeechEnhancementConfig(
    model=sherpa_onnx.SpeechEnhancementModelConfig(
        gtcrn='path/to/gtcrn.onnx'
    )
)
```

---

## 6. Latency Budget

Target for live audio monitoring: **< 50 ms total** (imperceptible to most listeners).
Target for gaming / video call: **< 20 ms** (requires JACK or ASIO).

```
┌─────────────────────────────────────────────────────┐
│  Total latency budget: ~30–50 ms (comfortable)      │
│                                                     │
│  Audio I/O buffer (1024 samples @ 44100 Hz)  ~23 ms │
│  Model inference per block                    ~5 ms  │
│  OS audio stack overhead                      ~5 ms  │
│  Python callback overhead                     ~2 ms  │
│                                              ──────  │
│  Total                                       ~35 ms  │
└─────────────────────────────────────────────────────┘
```

**Block size guide:**

| Block size | Latency @44100 Hz | Max model time | Risk |
|---|---|---|---|
| 256 samples | 5.8 ms | ~3 ms | High CPU starvation risk |
| 512 samples | 11.6 ms | ~7 ms | Tight on low-end CPUs |
| 1024 samples | 23.2 ms | ~15 ms | **Recommended for music removal** |
| 2048 samples | 46.4 ms | ~35 ms | Safe on all CPUs; slight lag |

---

## 7. Threading Architecture

Never run the AI model on the audio callback thread. The callback must return within
one block period or you get a dropout (crackle/pop).

```
┌────────────────────┐     queue.Queue      ┌──────────────────────┐
│  sounddevice       │  ─── raw blocks ───► │  AI Inference Thread  │
│  Callback Thread   │                      │  (ONNX / Sherpa)     │
│  (OS-managed,      │  ◄── processed ───   │  Runs at normal       │
│   real-time prio)  │     blocks           │  thread priority      │
└────────────────────┘                      └──────────────────────┘
         │                                            │
         ▼                                            ▼
   outdata[:] = ...                         hidden_states updated

┌────────────────────────────────────────────────────────────────┐
│  Main Thread: Tkinter / PyQt6 GUI                              │
│  Reads shared state (gain, model choice, enable/disable)       │
│  Never touches audio buffers directly                          │
└────────────────────────────────────────────────────────────────┘
```

**Python implementation skeleton:**
```python
import sounddevice as sd
import numpy as np
import queue
import threading
import onnxruntime as ort

RAW_Q       = queue.Queue(maxsize=4)   # raw blocks from callback
PROC_Q      = queue.Queue(maxsize=4)   # processed blocks back to callback
BLOCK_SIZE  = 1024
SAMPLE_RATE = 44100
hidden_state = np.zeros((1, 2, 64, 128), dtype=np.float32)   # model-specific

session = ort.InferenceSession('gtcrn.onnx', providers=['CPUExecutionProvider'])

def inference_thread():
    global hidden_state
    while True:
        block = RAW_Q.get()
        inp = np.expand_dims(block[:, 0], axis=0).astype(np.float32)
        out, hidden_state = session.run(
            ['output_audio', 'states_out'],
            {'input_audio': inp, 'states_in': hidden_state}
        )
        PROC_Q.put(out.squeeze(0))

def audio_callback(indata, outdata, frames, time, status):
    if not RAW_Q.full():
        RAW_Q.put_nowait(indata.copy())
    try:
        processed = PROC_Q.get_nowait()
        outdata[:, 0] = processed
        outdata[:, 1] = processed
    except queue.Empty:
        outdata[:] = indata   # pass-through if model hasn't caught up

threading.Thread(target=inference_thread, daemon=True).start()

with sd.Stream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
               channels=2, callback=audio_callback):
    input("Press Enter to stop...")
```

---

## 8. Complete Dependency Lists

### 8.1 Linux (Ubuntu 22.04+)

**System packages:**
```bash
sudo apt update
sudo apt install -y \
    python3 python3-pip python3-venv \
    pipewire pipewire-pulse pipewire-jack \
    libportaudio2 libsndfile1 \
    ffmpeg
```

**Python packages:**
```bash
pip install \
    sounddevice \
    numpy scipy \
    onnxruntime \
    sherpa-onnx \
    deepfilternet \
    audio-separator \
    torch torchaudio \
    demucs \
    tkinter       # built into Python on Ubuntu (python3-tk)
```

**Optional (speed):**
```bash
pip install onnxruntime-openvino   # AVX2/AVX-512 acceleration
```

**No virtual audio driver needed on Linux** — use PulseAudio/PipeWire monitor source.

---

### 8.2 Windows 10/11

**Python:** 3.10 or 3.11 (from python.org — include pip, add to PATH)

**Virtual audio driver (choose one):**
- **VB-Audio Virtual Cable** (free) — https://vb-audio.com/Cable/
- **VB-Audio Hi-Fi Cable** (free) — higher quality, 192 kHz support
- **ASIO4ALL** (optional, for lowest latency) — https://asio4all.com

**Python packages:**
```bash
pip install sounddevice numpy scipy onnxruntime sherpa-onnx deepfilternet audio-separator torch torchaudio demucs
```

**Optional:**
```bash
pip install onnxruntime-openvino
```

**Windows-specific notes:**
- sounddevice uses WASAPI by default — no extra config needed.
- Set VB-Cable Input as your default Windows Playback device.
- Read from "CABLE Output" as your sounddevice input.
- Write processed audio to your real speakers/headphones as output.

---

### 8.3 Dependency Size Reference

| Package | Installed Size | Required For |
|---|---|---|
| sounddevice + portaudio | ~5 MB | Audio I/O on all platforms |
| numpy + scipy | ~60 MB | DSP filters, array math |
| onnxruntime | ~20 MB | ONNX model inference |
| onnxruntime-openvino | ~180 MB | Accelerated ONNX inference |
| sherpa-onnx + models | ~15 MB | DPDFNet, GTCRN streaming |
| deepfilternet | ~30 MB | DeepFilter speech enhancement |
| audio-separator | ~10 MB | MDX-Net UVR models |
| torch + torchaudio | ~800 MB | DEMUCS, Silero VAD |
| demucs | ~350 MB weights | DEMUCS separation |

**Minimal real-time stack (no torch):**
`sounddevice + numpy + onnxruntime + sherpa-onnx` → ~100 MB total

---

## 9. Distribution / Packaging

### 9.1 Linux (.deb / AppImage / Flatpak)

**PyInstaller (simplest):**
```bash
pip install pyinstaller
pyinstaller --onefile --add-data "models/:models/" main.py
# Produces: dist/main  (self-contained binary)
```

**AppImage (runs on any Linux distro):**
```bash
pip install briefcase
briefcase create linux appimage
briefcase build linux appimage
```

**Notes:**
- Bundle the ONNX model files inside the dist folder.
- onnxruntime-openvino bundles its own runtime — include the `.so` files.
- PipeWire/PulseAudio are system-level — do not bundle them.
- The app must detect which audio subsystem is available at runtime.

### 9.2 Windows (.exe)

```bash
pyinstaller --onefile --noconsole \
  --add-data "models;models" \
  --hidden-import sounddevice \
  --hidden-import onnxruntime \
  main.py
```

**Notes:**
- WASAPI loopback works without any bundled driver.
- If bundling VB-Cable: include the VB-Cable installer and launch it via subprocess
  during first-run setup.
- Sign the `.exe` to avoid Windows Defender false positives on ONNX DLLs.

---

## 10. Mobile — Platform Constraints

### 10.1 iOS

| Constraint | Detail |
|---|---|
| **No system audio capture** | iOS does not allow any app to capture audio playing from other apps. AVAudioSession isolation is enforced at the OS level. |
| **Microphone only** | You can capture and process the device microphone in real time. |
| **Own-app audio** | You can process audio that your own app is playing (e.g., an in-app music player). |
| **No VST/AU plugins** | Plugin loading is not available on iOS. All DSP must be embedded. |
| **Background audio** | Requires `Audio` background mode in Info.plist; strict CPU limits apply. |
| **Model size** | App Store limit is 4 GB; practical limit for download is ~200 MB. CoreML is preferred over ONNX for on-device inference. |

**Workaround:** The only way to remove music from arbitrary audio on iOS is to:
1. Accept an audio **file** import (via Files app / share sheet), process it offline,
   and save the result — not real-time.
2. Or connect an external audio interface and process its input.

### 10.2 Android

| Constraint | Detail |
|---|---|
| **AudioPlaybackCapture API** | Available since Android 10 (API 29). Allows your app to capture audio from *other* apps — **but only if those apps set `allowAudioPlaybackCapture = true`**. Most music and streaming apps explicitly block this. |
| **Microphone** | Always available for real-time mic processing. |
| **Latency** | AAudio (API 26+) achieves ~10–20 ms on supported hardware. OpenSL ES (older) is ~30–50 ms. |
| **Model runtime** | ONNX Runtime for Android, TensorFlow Lite, or PyTorch Mobile. |
| **Background processing** | Android aggressively kills background processes. Use a foreground `Service` with a persistent notification. |
| **Root-only loopback** | For apps that block AudioPlaybackCapture, full system capture requires root access (`su` + `adb shell`). Not viable for consumer apps. |

---

## 11. Mobile — What Is Actually Possible

| Scenario | iOS | Android (10+) |
|---|---|---|
| Remove music from file imported by user | ✓ | ✓ |
| Remove music from microphone input live | ✓ | ✓ |
| Remove music from in-app media player | ✓ | ✓ |
| Remove music from Spotify / YouTube live | ✗ (impossible) | ✗ (apps block capture) |
| Remove music from system output globally | ✗ (impossible) | ✗ (requires root) |
| Use as Bluetooth/wired audio pass-through | ✗ | Partial (mic only) |

**Practical conclusion for mobile:**
The viable product is either:
- An **in-app media player** — user imports files, plays them through your app,
  music is removed live inside the app's audio graph.
- A **microphone processor** — target musicians, podcasters, streamers; capture mic,
  enhance speech, output to headphones.
- A **remote controller** for the desktop app — mobile UI sends settings over Wi-Fi;
  the desktop machine does the actual processing.

---

## 12. Mobile — Tech Stack Options

### 12.1 Flutter + C++ Audio Plugin (cross-platform)

```
Flutter UI (Dart)
    │
    ▼ platform channel
Native Audio Plugin (C++)
    │── Android: Oboe (AAudio backend) + ONNX Runtime Android
    └── iOS:     AVAudioEngine + CoreML (or ONNX Runtime iOS)
```

**Pros:** Single UI codebase, best performance.
**Cons:** Requires writing C++ audio plugin; complex build setup.

**Key packages:**
```yaml
# pubspec.yaml
dependencies:
  flutter:
  just_audio: ^0.9.0        # audio playback
  record: ^5.0.0            # microphone capture
  flutter_sound: ^9.0.0     # lower-level audio I/O
```

### 12.2 React Native + react-native-audio-api

Similar architecture to Flutter; JS UI, native audio processing in C++/Swift/Kotlin.

### 12.3 Kivy (Python, for quick prototyping only)

```bash
pip install kivy buildozer   # Android
pip install kivy             # iOS via kivy-ios
```

Kivy can run Python code on Android/iOS, but audio I/O is limited and latency is
too high for real-time processing. Use only for file-based offline processing.

### 12.4 Android-Specific Stack

```
Kotlin / Java UI (Activity + ViewModel)
    │
    ▼ JNI bridge
Oboe C++ library (low-latency AAudio)
    │
    ▼
ONNX Runtime for Android (C++ API)
    │
    ▼
Model: GTCRN or DPDFNet .onnx
```

**Setup:**
```gradle
// build.gradle
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.17.0'
    implementation 'com.google.oboe:oboe:1.8.0'
}
```

### 12.5 iOS-Specific Stack

```
SwiftUI
    │
    ▼
AVAudioEngine (tap on input node)
    │
    ▼
CoreML (preferred) or ONNX Runtime for iOS
    │
    ▼
Model: converted to .mlmodel via coremltools, or GTCRN .onnx
```

**Convert ONNX → CoreML:**
```bash
pip install coremltools
python3 -c "
import coremltools as ct
model = ct.converters.onnx.convert(model='gtcrn.onnx')
model.save('gtcrn.mlpackage')
"
```

---

## 13. Recommended Build Path

### Phase 1 — Desktop MVP (2–4 weeks)

1. **Linux:** Use PipeWire/PulseAudio monitor source. No driver needed.
   **Windows:** Install VB-Cable. Set as default playback device.

2. **Model:** Start with **GTCRN via Sherpa-ONNX** (2 MB, <5 ms/block, no torch).

3. **Runtime:** `onnxruntime` with CPUExecutionProvider. Add OpenVINO later.

4. **UI:** Tkinter (already in Music Assassin) — add device selector dropdowns,
   enable/disable toggle, and level meter.

5. **Threading:** Queue-based inference thread (see Section 7).

6. **Packaging:** PyInstaller. Bundle GTCRN .onnx inside dist.

### Phase 2 — Desktop Full (2–3 weeks additional)

- Add model selector: GTCRN / DPDFNet / SpeechDenoiser / DeepFilterNet.
- Add OpenVINO provider toggle.
- Add virtual device auto-setup helper (detect PipeWire vs PulseAudio on Linux;
  check for VB-Cable on Windows, prompt to install if missing).
- Add waveform / spectrum visualiser.

### Phase 3 — Mobile (4–8 weeks)

- Android: Kotlin + Oboe + ONNX Runtime for Android. Build in-app media player
  that applies GTCRN to the playback stream live.
- iOS: SwiftUI + AVAudioEngine + CoreML (converted GTCRN).
- Remote controller mode: connect to Phase 1 desktop app over local Wi-Fi socket.

---

## Quick Reference — Minimum Viable Package List

```bash
# Minimum real-time desktop stack (no torch, no GPU)
pip install sounddevice numpy scipy onnxruntime sherpa-onnx

# Add offline file separation (Music Assassin style)
pip install audio-separator deepfilternet

# Full stack including DEMUCS + VAD
pip install torch torchaudio demucs
```

```bash
# Speed boost (optional, adds ~180 MB)
pip install onnxruntime-openvino
```

```bash
# Android ONNX Runtime (add to build.gradle, not pip)
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.17.0'

# iOS ONNX Runtime (add to Podfile)
pod 'onnxruntime-objc', '~> 1.17.0'
```

---

*Last updated: 2026-06-27 | Music Assassin project*
