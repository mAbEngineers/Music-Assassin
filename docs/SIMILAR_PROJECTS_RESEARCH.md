# Similar Projects & Improvement Research

**Features, sound-effects preservation, and real-time quality.** Researched 2026-10-18.

This document covers three things:

- **Similar projects.** What open-source projects, Android apps, browser extensions and commercial products in the same space do.
- **Ideas to borrow.** What each of the three repos can take from them: this research lab, `music-assassin-live`, and `music-assassin-mobile`.
- **Three questions:**
  1. How to make the apps easier for non-technical users.
  2. How to keep sound effects (SFX) and ambience while removing only the music.
  3. How to raise real-time quality.

It builds on, and does not repeat:

- `REALTIME_FILTER_RESEARCH.md`
- `SEPARATION_TECHNIQUES_RESEARCH.md`
- `COMPREHENSIVE_METHOD_COMPARISON.md`
- `music-assassin-live/docs/ROADMAP.md`

Models already covered there are not re-described. Those include HS-TasNet, RT-STT, Band-SCNet, sherpa-onnx, DeepFilterNet, DTLN, GTCRN, DPDFNet, UVR/MDX, MRX, and Bandit.

**How claims were checked.** Every source is marked with one of these:

| Mark | Meaning |
|---|---|
| ✔ | I opened the source in this session and confirmed the claim. |
| ● | A research agent reported it with a cited source, usually code, a repo, a store page or a paper. |
| ? | Not verified. Treat it as a lead. |

---

## 0. The findings that matter

1. **There are now direct competitors in the same "halal / no-music" niche.**
   - Poise Voice Isolator (✔) is system-wide on Linux and Windows, MIT-licensed, and was updated yesterday.
   - NoMusic (●) is a real-time Chrome/Edge extension.
   - HaramMute (●) is server-side and paid. It has 50K+ Android installs and 30K Chrome users.

   Every one of these either uses a speech enhancer, so it loses SFX the same way we do, or processes offline on a server. The only product that removes music *in real time while keeping dialogue and SFX* is AudioShake's commercial SDK (✔, closed source).

2. **OSI-licensed weights that keep dialogue and SFX now exist.** These make an offline "keep effects" mode possible today, without the YAMNet/TensorFlow add-back:
   - ZFTurbo's CDX23 dialog/effect/music model: MIT, weights in a GitHub release (✔).
   - TIGER-DnR int8: Apache-2.0, 7.3 MB (✔).
   - MRX: MIT.

3. **The live app hides its own latency from video players. It also hides the Bluetooth headset's.**
   - PipeWire leaves monitor ports out of latency propagation. So nothing that happens after the trap sink's monitor reaches the apps playing into it (●, PipeWire source plus a measurement on this machine).
   - Video players therefore see about 0 ms, when the real figure on the soundcore Bluetooth headset is about 230 ms. This is a likely contributor to the lip-sync reports in the live repo's HANDOVER.
   - Fixing it gains more than lip-sync. Once latency is reported correctly, the app can *declare* 300–700 ms of processing delay and players will delay the picture to match. That makes a "movie mode" possible with far better non-causal models (§4).

4. **The industry recipe is to turn music down only when it is present, rather than delete it everywhere.**
   - TVs and streaming products cut the background by a *finite* amount. Amazon Dialogue Boost goes down to about −12 dB.
   - They act only when needed (Sonos, Roku Auto, Fraunhofer Dialog+).
   - A music-presence gate keeps SFX untouched in every music-free scene. This is the cheapest SFX win available for the live app (§3.3).

5. **Small real-time cinematic separators are proven elsewhere; we need our own weights.**
   - Fraunhofer Dialog+ uses about 370k parameters (●).
   - Amazon Dialogue Boost runs a distilled sub-band model on a 4-core Cortex-A55 Fire TV stick (●).
   - LiteCASS (✔ paper, Sept 2026) has 1.06M parameters, splits dialogue/music/SFX, and reports CPU RTF about 0.012. No code or weights are published.

   The realistic path is to distil a small causal "music vs everything else" model from the offline models in point 2 (§3.4).

6. **Android has a route that needs no running Shizuku.** RootlessJamesDSP (●, source read):
   1. Captures audio with AudioPlaybackCapture.
   2. Mutes each original by attaching a −200 dB DynamicsProcessing effect to that app's audio session.
   3. Needs only a **one-time** `DUMP` + `PROJECT_MEDIA` grant, which survives reboots.

   This sits alongside the mobile repo's current AudioPolicy + UID-2000 design (§6).

7. **What non-technical users complain about in this niche:**
   - paywalls and minute limits
   - processing that stalls
   - A/V drift

   Free, on-device and reliable is the differentiator (§5).

8. **Risk: Moises patent US12399678B1** (✔ claims read).
   - Claim 1 covers a UI control that activates a "system-level stem separation mode". The mode diverts system output audio away from the playback devices, separates segments of it into stems, and sends a stem to the speakers.
   - Claim 5 adds that the audio is the mix of all applications.

   This is close to `music-assassin-live`. Get a legal opinion before any commercial distribution (§7).

---

## 1. Ranked plan

Effort: S = days, M = 1–3 weeks, L = a month or more.

| # | What | Repo | Effort | Why |
|---|---|---|---|---|
| 1 | **Report latency correctly.** Replace the null-sink trap with a `module-loopback` front end (PipeWire 1.4+), or a filter-chain smart filter, that declares `ProcessLatency`. On PipeWire 1.0.5 use a `pw_filter` with `PW_FILTER_FLAG_CUSTOM_LATENCY`. | live | S–M | Fixes lip-sync on Bluetooth today, and enables #5. §4 |
| 2 | **Music-presence "Auto" gate**: `out = p·processed + (1−p)·dry` with constant delay, where `p` comes from CED-tiny or EfficientAT mn04 | live, mobile | S | Music-free scenes stay untouched, so SFX are kept. §3.3 |
| 3 | **Built 2026-10-23** (MRX + CDX23 in the app; see `HANDOVER.md`). **"Keep effects" offline mode** using CDX23 (quality, about 1–1.5 min per 10 min) or MRX int8 (fast, under 20 s per 10 min). Output = dialogue + effects (or mix − music). Benchmark against Voc_FT + YAMNet and KARA_2 subtraction. The TIGER int8 export measured too slow (see `OFFLINE_SPEED_RESEARCH.md`). | research → offline app | S–M | First OSI path that keeps SFX without TensorFlow. §3.1 |
| 4 | **Turn down, don't delete**: relabel the mix slider as **Music level** with named steps and a finite default (about −15 dB) | live | S | Industry default; hides artefacts; keeps some SFX. §3.3, §5.1 |
| 5 | **Dropped 2026-10:** the user rejected a 0.5–1.5 s delay even if players are told about it; live music removal now depends on a causal model (row 3.4 / Track C). Kept for the record. **Movie mode**: declared D ≈ 300–700 ms, chunked MRX (measured RTF 0.03 offline), stride 0.25–0.5 s. Keep the current 50 ms path as "Live / games". | live | M | Offline-class quality with lip-sync intact. §4 |
| 6 | **Real-time residual add-back**: a classifier on `mix − enhanced`, adding back the parts that are not music | live | M | Real-time version of the offline YAMNet result (94 % SFX kept). §3.3 |
| 7 | **Distil our own causal "music vs rest" model**: retrain GTCRN or DPDFNet with target = dialogue + SFX and noise = music, using pseudo-labels from #3's models | research | M–L | The only path to SFX-keeping removal at 50 ms. §3.4 |
| 8 | **Plain-language UX**: Music control with Auto and "Hear original", presets, per-app rules, `doctor`, Arabic UI | live, mobile | S–M | §5 |
| 9 | **Offline editability**: Dialogue / Effects / Music sliders with preview, a leftover stem, time-range keep/remove, all audio tracks | offline app | M | §5.2 |
| 10 | **Android one-time-grant route**: embedded wireless-ADB pairing, then `DUMP` + `PROJECT_MEDIA` grants, then capture plus a per-session −200 dB mute | mobile | M | No live Shizuku or shell process. §6 |
| 11 | **Legal check of US12399678B1** before any paid or store release | all | — | §7 |

---

## 2. Similar projects

### 2.1 Direct competitors: music removal, mostly for the halal / no-music audience

#### Desktop and system-wide

**Poise Voice Isolator** ✔ — [github.com/chabandou/poise-voice-isolator](https://github.com/chabandou/poise-voice-isolator)
- **License:** MIT. **Real-time:** yes.
- **What it is:** system-wide on Linux (its own PipeWire node) and Windows (VB-Cable). Uses DeepFilterNet3 by default; GTCRN and RNNoise are optional.
- **Worth borrowing:**
  - It reports a fixed delay to PipeWire so video players compensate.
  - A missed deadline plays the frame unprocessed, controlled by `POISE_PW_MARGIN_MS` = 21 ms.
  - `--doctor`, `--reset-audio` and `--fix-execstack` self-repair options.
  - A `latmeter` tool that measures the real A/V offset.
  - ONNX graph rewrites: DF3 about 15 % faster, GTCRN about 2.2× faster, by the author's numbers.
  - An Android build exists, Samsung-only ([poise-android](https://github.com/chabandou/poise-android) ●, routing method ?).

**ELUATE** ● — [github.com/anaxoniclabs/ELUATE](https://github.com/anaxoniclabs/ELUATE)
- **License:** code MIT; Bandit v2 weights CC-BY-SA. **Real-time:** no.
- **What it is:** splits speech, music and SFX, drops the music, and copies the video stream through bit-identical.
- **Worth borrowing:**
  - A ring-buffer streaming separator that keeps memory flat: 19 GB instead of 41 GB on an 84-minute file.
  - Disk-space and duration prechecks.

**Haram Music Remover** (hijam-git) ● — [github.com/hijam-git/haram-music-remover](https://github.com/hijam-git/haram-music-remover)
- **License:** none. **Real-time:** progressive.
- **What it is:** a desktop helper plus a Chrome toggle inside YouTube.
- **Worth borrowing:**
  - Processing starts at the seek position.
  - Progress is shown on the seek bar.
  - Ships as deb, AppImage, Windows and macOS builds with auto-update.

**bgm-remover** ● — [github.com/lijianying13/bgm-remover](https://github.com/lijianying13/bgm-remover)
- **License:** none. **Real-time:** no, but has a "watch while removing" mode.
- **Worth borrowing:**
  - A **music volume slider from 0 to 100 %**.
  - In watch mode, video frames are clocked to the processed audio.

**Twitch-Utils `mute`** ● — [github.com/TheDrHax/Twitch-Utils](https://github.com/TheDrHax/Twitch-Utils)
- **License:** GPL-3.0. **Real-time:** no.
- **Worth borrowing:** removal limited to given time ranges (`5:00~8:00 …`), which is the seed of a timeline editor.

**video-music-remover** ● — [github.com/asaah18/video-music-remover](https://github.com/asaah18/video-music-remover)
- **License:** MIT. **Real-time:** no.
- **Worth borrowing:** it processes **every audio track** in multi-language files.

**Islamic Audio Player** ● — [github.com/d3afult/Audio-Player](https://github.com/d3afult/Audio-Player)
- **License:** AGPL-3.0.
- **Worth borrowing:** YAMNet (TFLite) detects instruments and auto-skips those tracks. This is a music detector used as a trigger.

**fast-music-remover** ● — [github.com/omeryusufyagci/fast-music-remover](https://github.com/omeryusufyagci/fast-music-remover)
- **License:** MIT. 730★, but stale since January 2025. DeepFilterNet in Docker.
- Little to borrow.

> Avoid `Toinetteindecent428/youtube-music-remover`. It ships a prebuilt ZIP hidden in `icons/`, has 0 stars, and pads its topics with keywords (●).

#### Browser extensions

**NoMusic** ● — [repo](https://github.com/mohamedsadiqaguenaou/NoMusic), [Chrome Web Store](https://chromewebstore.google.com/detail/fepfnjijcojaoihcekfhonohejglmdbc)
- **License:** ISC. **Users:** 1,000, rated 4.8. **Real-time:** yes.
- **How it works:** tabCapture → offscreen document → ONNX Runtime Web running FastEnhancer in a worker, fed by a SharedArrayBuffer ring buffer → an RNNoise AudioWorklet.
- **Worth borrowing:** the architecture is ready to reuse for a browser version. It also offers an English/Arabic UI and Standard/Lite quality levels.

**YouTube Music Remover** (Amro-AL-Ghamdi) ● — [github.com/Amro-AL-Ghamdi/musicremover](https://github.com/Amro-AL-Ghamdi/musicremover)
- **License:** MIT code. **Real-time:** near real-time.
- **How it works:** it uses audio **YouTube has already buffered** as lookahead. That audio goes to a local server running Voc_FT, plus a DnR model for SFX. Playback is sample-scheduled, with a resync when drift exceeds 80 ms.
- **Worth borrowing:**
  - "Bleed suppression" Off/Normal/Strong, a post-mask built from the model's own music estimate.
  - A seek-bar strip showing processed / in-progress / queued.
  - A choice of what plays while waiting: pause, muted, or original.
  - A cache of the last 10 videos.
  - Its DnR model is CC-BY-NC.

**HaramMute** ● — [Chrome Web Store](https://chromewebstore.google.com/detail/bbkbpldbnkoncockinoapcmbiijejgpn), Play `com.harammute.mobile`
- **Users:** 30K on Chrome (rated 4.2); 50K+ on Android (rated 4.79). **Real-time:** no.
- **What it is:** server-side separation with a clean track synced to the video. Free tier is about 60 minutes a week, then a subscription.
- **What reviews say:**
  - Praised: "only getting rid of music, keeps voices and sound effects".
  - Complaints: the free limit is too low; "stuck at 5 %"; "YouTube is blocking requests"; A/V sync.
  - Requests: download and share; keep listening on another tab; Linux support.

**Music Remover** (M. Jaber) ● — [Chrome Web Store](https://chromewebstore.google.com/detail/music-remover/aoigalcpojgijmaclbmicnhejndafkbj)
- **Users:** about 1,000, rated 4.9. **Real-time:** no.
- **Worth borrowing:** a **shared cache of already-processed videos**, so later viewers get them instantly.

**NoNoiseYT** ● — [Chrome Web Store](https://chromewebstore.google.com/detail/nonoiseyt/afjmmmkjfhofbchaccefljbjgldbkbjk)
- **Users:** 6. **Real-time:** yes.
- **Worth borrowing:** a **spectral music detector gates the enhancer**, so audio passes through untouched when there is no music. It also has a strength slider.

#### Android

**Murem** ● — [repo](https://github.com/Ignema/youtube-music-remover), on F-Droid
- **License:** MIT, 75★. **Real-time:** no.
- **What it is:** python-audio-separator running in Termux on the phone, or on a self-hosted PC.
- **Worth borrowing:**
  - Share → remove music.
  - mDNS discovery of the PC.
  - A job queue that survives crashes.
  - A home-screen widget.
- **Halal | Music Remover Offline** (AppRiser, ●) is a similar offline 2-stem app with ads and 100+ installs.

### 2.2 Commercial reference products

| Product | What it separates | Real-time? | Controls given to users | Lesson |
|---|---|---|---|---|
| **AudioShake Music Removal SDK** ✔ (Dec 2025) | Music vs dialogue / SFX / ambience | Yes. iOS, macOS, Android, Windows, Linux | API only | Our exact target exists as a closed commercial product. Use it as a quality benchmark or licensing option. The "9 MB, <50 ms" figures are ? |
| **Samsung AI Sound Controller Pro** (2026 TVs) ● | Speech / music / effects | Yes, on the TV's NPU | Per-category levels, by menu or by voice command | Three faders are now a mass-market consumer feature |
| **Amazon Dialogue Boost** (Fire TV, 2025) ● | Dialogue vs the rest | Yes, on a 4× Cortex-A55 | Five named levels | Background cut of at most about 12 dB plus a 1–4 kHz lift. Sub-band model, distilled, pseudo-labelled on real TV audio ([Amazon Science](https://www.amazon.science/blog/dialogue-boost-how-amazon-is-using-ai-to-enhance-tv-and-movie-dialogue)) |
| **Fraunhofer MPEG-H Dialog+** ● | Dialogue vs background | Runs on files | Original / enhanced / more enhanced | About 370k parameters, fully convolutional STFT. 83 % of viewers liked having the switch, but only 46 % preferred the enhanced version ([paper](https://arxiv.org/pdf/2112.09494)). Singing detection added in 2025 |
| **Sonos Arc Ultra Speech Enhancement** ● | Speech, centre channel only | Yes | Low / Medium / High / Max | Engages "only when needed"; works on the centre channel only |
| **Sony Voice Zoom 3** ● | Voice vs the rest | Yes | One signed slider, from below 0 up to +6 | One slider, neutral default |
| **Roku Speech Clarity** ● | Dialogue vs background | Yes | Off / Low / High / **Auto** | Loudness-neutral |
| **DaVinci Resolve Dialogue Separator** ● | Voice / background / ambience | Yes (Fairlight FX) | Three sliders, each with a mute | The editing UI to copy for the offline app |
| **Samsung Galaxy Audio Eraser** ● | Voices, music, wind, nature, crowd, noise | Offline | Auto, one slider per category, "Hear original" | Phone-native A/B and category sliders |
| **Pixel Audio Magic Eraser** ● | Speech, music, noise, wind (crowd per some sources ?) | Offline | Only categories it detects appear, as chips | Show only what is present |
| **VirtualDJ Stems 2.0** ● | Up to 7 stems | GPU real-time; fast mode on CPU | Pads plus "Stems on EQ" knobs | Tiered models; light-model fallback when not prepared |
| **djay Neural Mix** ● | Drums / harmonics / vocals | Apple Neural Engine (AudioShake engine) | Neural Mix EQ mode; echo tail on mute | Map stems onto familiar knobs; never cut abruptly |
| **Waves Clarity Vx** ● / **Audionamix IDC** ● | Voice vs noise / background | Yes, about 43 ms | Background ±24 dB, Speech ±24 dB, Strength | "Strength" trades artefacts against reduction |

### 2.3 Open-source infrastructure worth reusing

**HushMic** ✔ — [github.com/Fovty/HushMic](https://github.com/Fovty/HushMic)
- **License:** MIT / Apache-2.0.
- **What it is:** writes a PipeWire `module-filter-chain` config with a DPDFNet **LADSPA** plugin.
  - PipeWire owns real-time scheduling, so no setcap is needed.
  - A watchdog recreates the device after a PipeWire restart.
  - Quitting restores the previous default.
  - Latency is reported on PipeWire 1.6+.
  - A native AVX2 engine uses about half the CPU of ONNX Runtime.
- **UX to borrow:** a tray with suppress / bypass / mute, and a **"Test my mic" window** that records 10 s and plays it raw and cleaned.

**BigLinux noise reduction** ● — [App](https://github.com/biglinux/biglinux-noise-reduction-pipewire), [native engines](https://github.com/biglinux/speech-enhancement-native)
- **License:** GPL-3.0 (app); Apache-2.0 (engines).
- **What it is:** already has a system-output filter, built on **WirePlumber smart filters** (`filter.smart=true`).
  - Rust int8 LADSPA engines for DFN3, DPDFNet-48k-HR, GTCRN and Silero.
  - Each plugin has a `latency` port that filter-chain adds to node latency.
- **UX to borrow:** simple and advanced views, and a master bypass that keeps the individual settings.

**EasyEffects** ● — [github.com/wwmm/easyeffects](https://github.com/wwmm/easyeffects)
- **License:** GPL-3.0.
- **Worth borrowing:**
  - Presets that load automatically per output device.
  - App exclusion matched by **app name plus media name**.
  - Its "Voice Suppressor Inverted Mode" uses L/R correlation, local kurtosis and instantaneous frequency. Those cues could strengthen our mid/side pre-filter.

**JDSP4Linux** ● — [github.com/Audio4Linux/JDSP4Linux](https://github.com/Audio4Linux/JDSP4Linux)
- **License:** GPL-3.0.
- **Worth borrowing:** per-device preset rules, a D-Bus service plus CLI, and per-app exclusion.

**StemgenRT 0.7.0** ● — [github.com/sweetspotsoundsystem/stemgen-rt](https://github.com/sweetspotsoundsystem/stemgen-rt)
- **License:** MIT; weights license not stated.
- **What it is:** a stateful ONNX model with 13 state tensors, about 5.8 ms latency, run single-threaded on ONNX Runtime. Outputs drums, bass and vocals.
- **Caveats:** no SDR is published, and it would drop SFX.
- **Worth borrowing:** a reference pattern for a real-time-safe streaming ONNX graph.

**mss_streaming** ● — [code](https://github.com/dylansechet/mss_streaming); paper ✔ [arXiv 2609.35397](https://arxiv.org/abs/2609.35397)
- **License:** MIT.
- **What it is:** streams off-the-shelf HT-Demucs and SCNet with no retraining.
- **Results:** below 93 ms of lookahead, quality falls steeply. At 93 ms the loss is under 1 dB versus offline (●).
- **Caveat:** the abstract (✔) says streamed models are much less compute-efficient, and only HT-Demucs runs faster than real time, on a GPU.
- **Worth borrowing:** guidance for sizing a lookahead budget, not a CPU solution.

**PipeWire filter-graph ONNX plugin** ● (PipeWire 1.6+)
- **What it is:** runs ONNX in the graph data thread. The shipped example is Silero VAD.
- **Limits:** no built-in STFT and no latency reporting. It only fits tiny models.

**anira** ● (Apache-2.0) / **RTNeural** ● (BSD-3)
- Real-time-safe inference libraries, useful if the live engine becomes native or a plugin.

**BSRoformer.cpp / MSST-GGML** ● (MIT)
- ggml ports of RoFormer separators that run on CPU, Vulkan or CUDA.
- A route to iGPU inference in movie mode (?).

**FastEnhancer** ● — [github.com/aask1357/fastenhancer](https://github.com/aask1357/fastenhancer)
- **License:** MIT code; weights license ?.
- **What it is:** streaming speech enhancer at 16 or 48 kHz, ONNX in releases. Sizes from 28K to 1.1M parameters.
- **Speed:** the T size has RTF 0.012 at 48 kHz.
- **Worth borrowing:** a candidate for the live app's model picker.

**Demucs-GUI** ● — [github.com/CarlGao4/Demucs-Gui](https://github.com/CarlGao4/Demucs-Gui)
- **License:** GPL-3.0.
- **Worth borrowing:**
  - A per-stem mixer from −500 % to +500 %.
  - "Minus stems".
  - An **"all left" residual stem**.
  - Presets.

**Intel OpenVINO Audacity plugins** ●
- **License:** GPL-3.0.
- **What it is:** Demucs and DeepFilterNet inside Audacity.
- **Worth borrowing:** a possible distribution channel for timeline-style editing.

---

## 3. Keeping sound effects and ambience

### 3.1 Separators that output dialogue, music and effects (DnR / cinematic)

| Model | License / weights | Size | Real-time on CPU? | Quality evidence | Use |
|---|---|---|---|---|---|
| **ZFTurbo MVSEP-CDX23** ✔ ([repo](https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing)) | **MIT**. Weights are in GitHub release `v.1.0.0`: 3 × 54 MB HTDemucs `.th` (✔ via the API) | about 54 MB per checkpoint | No (HTDemucs class) | DnR SDR (✔ README): dialogue 14.18 / effect 7.92 / music 6.75 single model; 14.68 / 8.48 / 7.30 for the 3-model ensemble. **The music stem can contain vocals**, which suits anime OP/ED | Offline "keep effects" mode, a quality teacher for distillation, the high-quality tier |
| **TIGER-DnR int8 ONNX** ✔ ([HF](https://huggingface.co/audiojs/tiger-dnr)) | **Apache-2.0** | 7.3 MB; 3 × 1.4M-parameter networks | **No. Measured RTF 1.3–3.9 on 16 threads** (`OFFLINE_SPEED_RESEARCH.md`). The export has 38,817 small nodes (unfused normalisation), so op overhead dominates. A re-export might fix this (unmeasured) | Median SNR (✔): dialogue 12.66 / music 10.23 / effects 8.03, vs MRX at 10.92 / 5.17 / 5.72 | Quality reference until a faster export exists. A LiteRT GPU port reports RTF about 0.37 on a Pixel 8a (●) |
| **MRX** ● ([int8 ONNX](https://huggingface.co/audiojs/mrx), [MERL](https://github.com/merlresearch/cocktail-fork-separation)) | **MIT** | 31 MB int8, about 30M parameters | **Measured: 16–18 s per 10 min (RTF 0.03) even on 4 threads.** BLSTM and non-causal | Weaker than TIGER on DnR. Has a loudness-adapted checkpoint for real films | **Fast "keep effects" offline mode**; chunked movie-mode candidate (§4) |
| **BandIt Plus DnR** ● (ZFTurbo release v.1.0.3) | Trainer MIT; **weights license ?** | about 37M parameters | No, RTF about 1.34 | DnR: speech 15.64 / music 9.18 / effects 9.69 | Offline teacher only |
| **Bandit v2 (DnR v3)** ● | Code Apache; **weights CC BY-SA** (not OSI) | 7 × 447 MB | No | MVSep leaderboard: music 9.06 / SFX 10.82 / speech 12.29 | Research only. **DnR v3 removed vocals from music**, so singing likely lands in "dialogue", which is wrong for OP/ED songs (?) |
| **LiteCASS** ✔ ([arXiv 2609.23453](https://arxiv.org/abs/2609.23453)) | **No code or weights** | 1.06M parameters, 0.72 GMAC/s | Throughput yes (RTF about 0.012 per the paper, ●). Causality not stated | Best average SI-SDR in its comparison; beats BandIt on SFX (●) | Architecture template for our own student model (§3.4) |
| **SAM Audio** ● (Meta) | Custom "SAM License", **not OSI** | about 0.5–3B parameters (?) | No, GPU | Returns `target` + `residual`, so prompting "music" and keeping the residual gives exactly "original minus music" | Quality ceiling and pseudo-label teacher, once the license is read |
| **CLAPSep** ● | MIT code; weights ? | — | No | Supports **negative queries** ("keep speech and effects, not music") | Research reference |

**Two dataset notes:**

- **DnR-nonverbal** ([arXiv 2506.02499](https://arxiv.org/abs/2506.02499) ●) puts laughter and screams into the speech stem. Existing models wrongly treat these as SFX, which matters for anime.
- **DnR v3** ● is CC BY-SA 4.0 and covers 30+ languages.

### 3.2 Music-presence detectors (for gating)

All of these are OSI-licensed and can be exported to ONNX.

| Detector | Size | Notes |
|---|---|---|
| **CED-tiny** ● ([HF](https://huggingface.co/mispeech/ced-tiny)) | 5.5M parameters, 16 kHz | Weights Apache-2.0, but the code repo sidebar says GPL-3.0; resolve that before shipping. Has `export_onnx.py`; sherpa-onnx ships it on Android |
| **EfficientAT mn04 / mn05** ● ([repo](https://github.com/fschmid56/EfficientAT)) | about 1M parameters, 0.11 GMACs | MIT. The mel front-end must be exported separately. Best fit for Android |
| **sherpa-onnx zipformer audio tagging** ● | 26 MB int8 | License ?. Easy, because sherpa-onnx is already in use |
| **inaSpeechSegmenter** ● | — | MIT, but TensorFlow and offline; reference only |

**Do not use Essentia models.** They are CC BY-NC, which fails the OSI gate.

**Caveat:** AI-OpenBMAT ([arXiv 2602.06823](https://arxiv.org/abs/2602.06823) ●) reports detector F1 **below 60 % for quiet background music**. Use a *soft* gate with hysteresis, and fall back to the separator's own music-energy ratio, rather than a hard switch.

### 3.3 Pipeline designs, cheapest first

**A. Music-presence gate (live and mobile, about 1 % of one core extra, estimated)**

`out = p·processed + (1−p)·dry`

- **Where `p` comes from:** a detector on 1 s windows with a 250 ms hop.
- **How fast it switches:** attack about 150 ms, release 1–2 s.
- **Delay rule:** dry and processed must share one constant delay, or lip-sync jumps.

What it buys: every music-free scene is passed through bit-exact, so all SFX and ambience survive. What it costs: about 0.3–0.5 s of music leaks at each music onset. NoNoiseYT and Islamic Audio Player use the same idea, and Sonos and Roku Auto do the commercial equivalent.

**B. Turn down, don't delete (live; nearly free)**

The existing dry/wet slider is already the industry's remix control. With wet = the enhancer's voice estimate `V`:

`w·V + (1−w)·mix = V + (1−w)·(mix − V)`

So `(1−w)` *is* the background gain. What changes:
- Label the slider as a **Music level** in named steps.
- Default it to a finite cut of about −12 to −20 dB, rather than full removal.

This preserves part of the SFX inside `mix − V` and hides artefacts. Amazon and Fraunhofer ship exactly this. It also links directly to the `floor_db` work in `music-assassin-live` ROADMAP B4.

**C. Real-time residual add-back (live, M)**

1. Compute the residual `R = mix − V`.
2. Classify R on 0.5 s windows with CED-tiny or mn04.
3. Add back the windows that are not music.

This is the streaming version of the offline YAMNet experiment (0.5 % leakage, 94 % SFX kept). Continuous ambience will come back well, but short SFX hits arrive about 0.5 s late or are lost.

**D. Subtractive cinematic output (offline now; movie mode later)**

`out = mix − music_est`

- `music_est` comes from CDX23, MRX, or an average of either with Inst_HQ_3. TIGER's music graph is better quality but its current export is too slow.
- Set the estimate to zero wherever the detector says p(music) < 0.2.

Mixture-consistent by construction: anything that is not music is kept. Optional extras:
- "Bleed suppression" Off/Normal/Strong as in the Amro extension: a post-mask from the music estimate.
- A Wiener refinement `G = |R|² / (|R|² + |M̂|²)`.

**Measured 2026-10-22** (details in `OFFLINE_SPEED_RESEARCH.md`, "Keep sound effects"):
- The app now runs a parallel effects path: dialogue from Voc_FT, plus an effects stem from MRX or CDX23 added back after the VAD gate.
- On a ground-truth mix (real episode effects inserted into a music-only clip), music leaks into the effects stem at −16 dB with MRX and −25 dB with CDX23.
- A three-stem Wiener re-filter changed leakage by under 0.5 dB. A music post-mask (spectral over-subtraction, β = 1) bought MRX 5 dB for about 3 dB of effects, so it is on for MRX only.

**E. Singing.** Add an explicit user choice: **"Singing counts as music"** (on for anime OP/ED) or off (musicals). Fraunhofer added singing detection in 2025 because this is a known failure. Check that any model trained on DnR v3 doesn't move singing into dialogue.

**Not recommended:**
- **Band-limited pass-through.** Skipping processing below 80 Hz or above 8 kHz lets bass, kick drums and cymbals through.
- **Transient / HPSS pass-through.** The repo already measured it at only 5 % kept.

### 3.4 Training our own small "music vs everything else" model

**Target and loss**
- Two stems: music (including singing), and rest = dialogue + SFX + ambience.
- Predict the music; compute `rest = mix − music`.
- Loss: SI-SDR plus multi-resolution STFT, weighted towards the rest stem, plus a penalty on rest energy during music frames.

**Data**
- DnR v2/v3 dialogue + effects, plus DnR-nonverbal, as "rest".
- FMA, MUSDB or MoisesDB *full mixes* as music, so that songs with vocals count as music.
- Pseudo-labels from §3.1's teachers run over about 20 hours of the team's own anime and movie audio.
- MUSDB is non-commercial, which may restrict the resulting weights.

**Architecture, cheapest first**
1. **Retrain the shipped enhancer** (DPDFNet/DFN3 or GTCRN; training code is open) with clean = rest and noise = music. Same runtime, same 20 ms blocks, zero app changes. Risk: speech-tuned ERB resolution may limit how well it separates SFX from music.
2. **HS-TasNet** via lucidrains' trainer (23 ms).
3. **A causal LiteCASS-style subband U-Net**, or the 131K-parameter causal TFC-TDF design ([arXiv 2609.12201](https://arxiv.org/abs/2609.12201) ●, code not yet released). The 131K design is the right size for Android.

**Lessons from other teams**
- **Train with continuous context.** A model trained block-wise collapsed to silence within 2 s when run frame by frame (●, 2609.12201).
- **The cost of causality is about 0.7 dB.** Cadenza CAD2 causal vs non-causal ConvTasNet scored 6.2 vs 6.9 dB (●).

**Compute (estimates)**

| Model | Free Kaggle/Colab T4 time |
|---|---|
| GTCRN-size | about 15–30 h |
| About 1M-parameter U-Net | 40–80 h |
| DFN3 | 2–4 days |

All fit inside Kaggle's 30 GPU-hours a week using checkpoints.

---

## 4. Real-time: getting more processing time through reported latency

**The finding.** All of this is ● from source plus measurement by the research agent. The source references are in its notes. The test was run on this machine with PipeWire 1.0.5. No test nodes were left behind (✔ checked).

**How latency is meant to reach video players:**
1. PipeWire carries latency through port `Latency` params. A node's `ProcessLatency` is added as latency passes through it ([latency.dox](https://gitlab.freedesktop.org/pipewire/pipewire/-/blob/master/doc/dox/internals/latency.dox)).
2. `pw_stream_get_time_n()` includes the result.
3. pipewire-pulse reports it to PulseAudio clients as sink latency.
4. mpv, VLC, Chromium (`pa_stream_get_latency`) and Firefox (cubeb-pulse) all read it to schedule video.

**Why it fails today.** **Monitor ports are excluded from latency aggregation** (`spa/plugins/audioconvert/audioconvert.c`). So with the trap-sink + monitor design, none of the following reaches the apps:
- the engine's ~50 ms
- the real Bluetooth sink's 180 ms + 1 quantum

**What already exists for declaring latency:**
- `module-loopback` accepts `ProcessLatency` / `latencyOffsetNsec` and adds them to what it forwards to its sink side (PipeWire **1.4.0+**). The commit message names "loopbacks as front-end nodes to underlying sinks" for PulseAudio apps, which is exactly our case.
  - **Caveat:** `target.delay.sec` delays audio *without* reporting it.
- Filter-chain supports `ProcessLatency` from PipeWire 1.6.x.
- The builtin `delay` filter has a separate reported `latency` value.
- Issue [#4678](https://gitlab.freedesktop.org/pipewire/pipewire/-/issues/4678) ("How to manually specify process latency in order for applications to compensate") was closed in May 2025.

**Measured on 1.0.5.** An inline `pw_filter` declaring 300 ms and 1000 ms raised the upstream port latency to exactly 480 ms and 1180 ms. PulseAudio-client latency rose to match, but the reading was noisy because the test relinked mid-stream.

Two catches on 1.0.5:
- A sink-only `pw_filter` needs `PW_FILTER_FLAG_CUSTOM_LATENCY` plus a manual Latency param.
- WirePlumber did **not** auto-route PulseAudio streams to `pw_filter` sinks. Loopback and filter-chain nodes are routed.

**Design implied for the live app (outline; needs a spike):**
1. **Front end.** A `module-loopback` whose capture side is the default `Audio/Sink`, feeding an internal null sink that the engine captures. Alternatively, a filter-chain smart filter (WirePlumber 0.5+).
2. **Declare the total delay.** Set `ProcessLatency` = engine delay D + output stream + real sink latency. Re-declare it whenever the output device changes (Bluetooth ↔ speakers).
3. **Two modes:**
   - **Live / games:** today's ~50 ms.
   - **Movie:** D ≈ 300–700 ms, chunked separation with 2–3 s windows and a 0.25–0.5 s stride, read ≥ 1 model window before the right edge, crossfaded.
4. **Keep D constant.** Hold the same delay whether the music gate is open or closed.
5. **Communication apps bypass.** Route `media.role=Communication` (Meet, Discord) and a user per-app list straight to the real sink, so calls never get D.
6. **Measure the result.** Check with mpv, Firefox and Chromium using a latency meter like Poise's `latmeter`. That **players actually re-time video** is the one thing not yet confirmed end to end.

**Version constraint.** Ubuntu 24.04 ships PipeWire 1.0.5, while the loopback route needs 1.4+. On 1.0.5 the fallback is a small native `pw_stream` / `pw_filter` loopback that declares custom latency.

**Other platforms:**
- **Windows:** VB-CABLE gives no way to declare latency. A SysVAD-derived driver that reports a lagging presentation position would work, but needs kernel-driver signing (●, inferred).
- **Android:** remote submix presentation position follows the capturer's reads with a 4096-frame pipe. At most about 85 ms can be hidden (●).

**Cheap runtime wins seen elsewhere:**
- ONNX graph rewrites (Poise).
- Native int8 SIMD engines at about ½ the ORT CPU (HushMic, BigLinux).
- ORT `session.intra_op.allow_spinning=0`, so the CPU idles when nothing is happening.
- Skip inference on silence.

---

## 5. Features and editability for non-technical users

### 5.1 Live app (Linux, later Windows)

What it already has: on/off, model picker, dry/wet mix, mid/side, hold-to-compare, meters, status messages, tray planned (C5).

Additions:

- **One "Music" control in plain words.**
  - Named steps: *Original → Softer → Quiet → Off*.
  - An **Auto** setting that acts only when music is detected (§3.3 A). Show a small "music detected" indicator so users can see it working.
  - The pattern copies Sony, Fire TV, Sonos and Roku.
- **Presets in the audience's language:**
  - *Strict (music fully off)*
  - *Lectures / speech*
  - *Cartoons (keep effects)*
  - *Nasheed / voices only*
  - *Movies (lip-sync mode)*

  Behind each preset sits a model, gate, floor and mode choice.
- **Advanced panel** with **Dialogue / Effects / Music faders and mutes**, once a 3-stem path exists. Same pattern as Resolve, Samsung 2026 TVs and Galaxy Audio Eraser.
- **A/B:**
  - Keep the hold-to-compare button.
  - Add a **"Test" window** that records 10 s and plays it back both ways (HushMic).
  - Fraunhofer's trial shows the switch matters more than the default.
- **Rules:**
  - Per-app exclude list, matched by app plus media name. Examples: a Quran recitation tab, calls.
  - Auto-enable when chosen apps play.
  - Presets per output device (EasyEffects, JDSP, Wavelet).
- **Loudness-neutral output.** Removing music should not make everything feel quieter (Roku, MPEG-H).
- **Smooth transitions.** Gain ramps of 50–200 ms whenever the mode or the gate changes (djay's echo tail is the extreme version).
- **Self-repair:** a `doctor` command and "Reset audio" in the menu (Poise, BigLinux).
- **Honest messages,** e.g. *"Songs with lots of singing may not be fully removed"*. YouTube's Erase Song does the same.
- **Arabic UI** as a first-class locale. NoMusic and islamic-toolkit ship English and Arabic, and many store reviews are in Arabic.

### 5.2 Offline app

- **Per-stem sliders with preview before export:** Dialogue / Effects / Music, each with mute. Separate once, then toggle stems instantly during preview (AudioXtractor ●).
- **"Leftover" residual stem** = original minus all stems, so nothing gets lost (Demucs-GUI's "all left").
- **Time-range keep/remove.** Start from Twitch-Utils' range syntax and grow it into a simple timeline with "remove music here" regions.
- **Process every audio track** in multi-language containers.
- **Queue that survives crashes; history; share and export** (Murem). HaramMute users explicitly ask for download and share.
- **Quality tier visible:** *Fast* (MRX int8) / *Keep effects* (CDX23) / *Best* (Voc_FT on OpenVINO BF16). Timings are in `OFFLINE_SPEED_RESEARCH.md`.

### 5.3 Mobile

- **Controls outside the app:** a Quick Settings tile, a persistent notification with on/off plus strength, and a home-screen widget (Mutify, Echo, Murem).
- **Same Music control and presets as desktop**, plus a **kid mode**: PIN-locked always-on, strict preset locked, start at boot. No competitor has this (idea, not sourced).
- **App list:** show apps that block capture greyed out with the reason, instead of failing silently (RootlessJamesDSP).
- **"Share to Music Assassin" offline mode** for files and links, using TIGER LiteRT on the GPU. This covers apps that cannot be captured.
- **Stay free and on-device.** Paywalls and minute limits are the most common complaint across every competitor.

### 5.4 Browser companion (new, optional)

**Option 1: process ahead of playback.** Use YouTube's already-buffered audio as lookahead (Amro design). This gives offline-model quality, and keeps SFX with a cinematic model, at zero perceived delay. Add:
- the seek-bar progress strip
- the "while not ready: pause / muted / original" choice
- a local cache

**Option 2: real-time fallback on any site.** Use NoMusic's tabCapture → worker → ONNX-web stack.

---

## 6. Android: reaching other apps' audio

**RootlessJamesDSP recipe** (● source read; GPL-3.0, so copy the *technique*, not the code):

**1. Capture.**
- `AudioPlaybackCaptureConfiguration` with USAGE_MEDIA, GAME and UNKNOWN.
- Exclude its own UID, user-blocked apps and apps that disallow capture.
- Output through an `AudioTrack` with `setAllowedCapturePolicy(ALLOW_CAPTURE_BY_NONE)`, so it never re-captures itself.

**2. Mute the originals.**
- **Find sessions:** parse session IDs from a `dumpsys media.audio_policy` dump (needs `DUMP`), and re-poll on `AudioPlaybackCallback`.
- **Mute:** attach a `DynamicsProcessing(priority = Int.MAX_VALUE, session)` effect with −200 dB input gain.
- **Fallback:** the hidden "Volume" effect.
- **Recovery:** re-enable the effect if another app takes control of it.
- **Why the capture still hears audio:** the captured copy branches off before the session effect chain. This is inferred from the fact that it works.

**3. One-time setup** over ADB or Shizuku. It survives reboots, so nothing privileged has to keep running:
- `pm grant <pkg> android.permission.DUMP`
- `appops set <pkg> PROJECT_MEDIA allow` (removes the consent dialog; also needed for lock-screen capture on Android 15+)
- optionally `appops set <pkg> SYSTEM_ALERT_WINDOW allow` for autostart.

**4. Limits.**
- Apps that opt out of capture are not processed. Spotify and Chrome are reported.
- It conflicts with other DynamicsProcessing apps such as Wavelet.
- It adds latency.

**What this means for `music-assassin-mobile`.** The repo already self-hosts ADB and gets UID 2000 (commit `60774d7`). Two routes follow:

| | Current design | One-time-grant alternative |
|---|---|---|
| How | AudioPolicy loopback running inside a UID-2000 process | Use that same bootstrap once, to run the two grants above; afterwards the plain app does capture + per-session mute |
| Strength | Captures apps that opt out of playback capture | Survives reboot without re-pairing wireless debugging |
| Weakness | The shell process must be re-established after a reboot | Apps that opt out are not processed |

The capture-opt-out list is therefore the deciding measurement. It is worth one spike to see which of the target apps (YouTube, browsers, VLC, Netflix) allow playback capture.

**Can a single APK do it with zero setup?**
- **Capture:** yes, with a consent dialog every time.
- **Silencing:** not reliably, because session IDs need `DUMP`.

Two source-reading hypotheses from the research agent are **untested on a device** and high-risk. Do not plan around them; at most, run a one-day spike on an Android 16 Pixel:
1. Policy-mix capture is unity-gain, so setting media volume to 0 would not silence the capture.
2. Android 13+ appears to allow registering a LOOP_BACK-only media mix without `MODIFY_AUDIO_ROUTING`. That would need a hidden-API bypass, which Google Play disallows.

**Other notes:**
- An unconfirmed report says Google may restrict loopback ADB after CVE-2026-0073 (?).
- HaramBlur (500K+ installs) shows this audience will accept heavy permissions such as Accessibility when the purpose is explained.

---

## 7. Risks

### Patent: Moises Systems US12399678B1

- **Status:** granted 2025-08-26, priority 2024-02-28 (✔ claims read). [Google Patents](https://patents.google.com/patent/US12399678B1/en).
- **Claim 1:** a system that presents *"a user interface frontend that displays a control for activating a system-level stem separation mode"*. Once activated, it:
  1. stops sending system output audio to the playback devices,
  2. separates segments of that audio into stems with a selected separation module,
  3. sends at least one stem to the playback devices.
- **Dependent claims:**
  - Claim 5: the audio is the mix of all sound-producing applications.
  - Claim 11: segments are processed while earlier ones play.

`music-assassin-live` (toggle → trap sink diverts all output → model → speaker) and the Android sink both look close to this. This is not a legal opinion. Get one before any commercial or store distribution. Prior-art dates for this project's own public releases are worth recording.

### Licenses that fail the OSI gate or are unclear

| Item | License | Status |
|---|---|---|
| Bandit v2 weights | CC BY-SA | Fails the OSI gate |
| DnR v3 dataset | CC BY-SA | Fails the OSI gate |
| Amro extension's DnR model | CC-BY-NC | Fails the OSI gate |
| Essentia models | CC BY-NC | Fails the OSI gate |
| MUSDB | Non-commercial | Fails the OSI gate |
| SAM Audio | Custom license | Fails the OSI gate |
| BandIt Plus DnR weights | Not stated | Unclear |
| StemgenRT weights | Not stated | Unclear |
| FastEnhancer weights | Not stated | Unclear |
| AVCASS | Not stated | Unclear |
| CED-tiny | Code GPL vs weights Apache | Unclear |
| ZFTurbo CDX23 | MIT (✔) | Still check whether DnR's own terms carry over to the weights |

### Competition

- **AudioShake** sells the target capability as an SDK.
- **HaramMute** owns the paid, server-side niche.
- **Poise** targets the same audience with the same architecture, is open source, and is active.

---

## 8. Verification status and open questions

**Verified in this session (✔):**
- CDX23 is MIT, its weights are in the GitHub release, and its SDR table.
- TIGER-DnR int8: license, size and SNR table.
- Poise features and routing.
- HushMic features.
- LiteCASS: paper exists, parameter count.
- AudioShake's real-time music-removal SDK announcement (Dec 2025).
- The Sechet paper's abstract and its compute caveat.
- Moises patent claims 1, 5 and 11.
- No leftover PipeWire test nodes.
- CPU speed of Voc_FT, MRX, CDX23, TIGER and Spleeter, measured on this machine (see `OFFLINE_SPEED_RESEARCH.md`).

**Open; each is worth a cheap check:**
1. Do mpv, Firefox and Chromium actually re-time video when a loopback declares 500 ms? (§4 spike.)
2. Do TIGER and CDX23 keep anime SFX and treat OP/ED singing as music on the team's own eval set? (Experiment #3 in §1.)
3. How accurate is CED-tiny or mn04 at detecting music on quiet background scores?
4. Which target Android apps opt out of playback capture?
5. Whether the LiteCASS model is causal, and whether code will be released.
6. The license questions in §7.

**Not verified at all (?):**
- AudioShake's 9 MB / <50 ms figures.
- Store install counts and review quotes. These come from the agent's Play/Chrome scrape; the Play page did not render for my check.
- poise-android's routing method.
- All runtime estimates marked "about" without a source.
