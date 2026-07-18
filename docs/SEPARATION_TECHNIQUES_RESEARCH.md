# Music Removal & Source Separation — Techniques, Tools & Research

Compiled research covering classical DSP, deep learning architectures, open-source projects, commercial tools, and applicability to Music Assassin.
**Open-source only** — all models and tools referenced here must be freely available (MIT/Apache/CC). Commercial-only tools are noted as reference-only.

---

## Table of Contents

1. [Classical / Traditional DSP](#1-classical--traditional-dsp)
   - Spectral Subtraction
   - Wiener Filtering
   - HPSS (Harmonic-Percussive Source Separation)
   - NMF (Non-Negative Matrix Factorization)
   - Adaptive Filters (LMS / RLS / FIR / IIR)
   - Notch Filters & Hum Removal
2. [Deep Learning Architectures](#2-deep-learning-architectures)
   - UNet / Spectrogram-domain UNets
   - LSTM / Open-Unmix
   - Demucs / HTDemucs (Waveform + Spectrogram Hybrid)
   - MDX-Net / MDX23C
   - BS-RoFormer (Band-Split RoPE Transformer)
   - MelBand-RoFormer
   - SCNet (Sparse Compression Network)
   - Conditioned UNet (text/label-guided)
   - Generative / Flow-based Separation
   - HS-TasNet (Real-Time, 23 ms)
   - RT-STT (Real-Time Single-Path, Nov 2025)
   - TISDiSS (Scalable Inference Framework)
   - Band-SCNet (Causal Lightweight, 2025)
3. [Open-Source Projects & Repos](#3-open-source-projects--repos)
4. [Inference Optimisation Techniques](#4-inference-optimisation-techniques)
   - Test-Time Augmentation (TTA)
   - Chunking & Overlap Strategies
   - Quantisation (INT8 / FP16)
   - Model Ensembling
   - two-stems mode (Demucs)
5. [Commercial AI Tools (reference only)](#5-commercial-ai-tools-reference-only)
   - LALAL.AI
   - remove.music
   - AudioCleaner.ai
   - AudioShake
   - Moises.ai
   - DaVinci Resolve (Fairlight AI)
5. [Frontier Research (2025)](#5-frontier-research-2025)
   - SAM Audio (Meta)
   - Mel-RoFormer + Melody Transcription
   - Ensemble / Hierarchical Stem Approaches
   - Knowledge-Driven / Cinematic Separation
6. [Technique Comparison Table](#6-technique-comparison-table)
7. [Applicability to Music Assassin](#7-applicability-to-music-assassin)
8. [Priority Upgrade Roadmap](#8-priority-upgrade-roadmap)

---

## 1. Classical / Traditional DSP

### Spectral Subtraction

**What it is:**
Estimate the average spectrum of the unwanted signal (music) from a quiet reference segment, then subtract that estimate from the mixed signal's power spectrum frame-by-frame in the STFT domain.

**How it works:**
1. Compute STFT of the input.
2. Estimate noise/music PSD from a reference window (silence or music-only segment).
3. Subtract estimated spectrum: `|Speech|² = |Mix|² − |Music|²`.
4. Apply half-wave rectification (floor at zero) to avoid negative power.
5. Reconstruct waveform via iSTFT using original phase.

**Strengths:**
- Zero dependency — pure numpy/scipy.
- Near-instant (no model loading).
- Useful as a residual cleanup step after neural separation.

**Weaknesses:**
- "Musical noise": subtraction artifacts create tonal chirping in background.
- Assumes music spectrum is stationary — fails with dynamic music.
- Requires a clean reference window with no speech.

**Use case for Music Assassin:**
Light post-processing pass after neural separation to suppress residual music bleed. Also useful in a "fast mode" when a music-only reference clip is provided by the user.

---

### Wiener Filtering (Generalized)

**What it is:**
An optimal linear filter that minimizes mean squared error between the estimated and true source. At each time-frequency bin, applies a soft mask: `W(t,f) = P_speech(t,f) / (P_speech(t,f) + P_music(t,f))`.

**How it works:**
All modern neural separators ultimately output something equivalent to a Wiener mask — the DNN estimates `P_speech` and `P_music` separately, then blends them via this formula. Even Demucs applies a Wiener-like recombination step.

**Strengths:**
- Theoretically optimal under Gaussian signal assumptions.
- Produces soft masks rather than binary cuts — fewer artifacts.
- Well-understood mathematically.

**Weaknesses:**
- Classical Wiener requires known signal statistics (not available in practice).
- Learned Wiener (neural) requires training data.

**Use case for Music Assassin:**
Understanding this helps tune post-processing: the "harmonic rejection" filter in the current pipeline is essentially a hand-crafted Wiener mask in a specific frequency band. Could be made adaptive.

---

### HPSS — Harmonic-Percussive Source Separation

**What it is:**
Exploits two structural properties of spectrograms:
- **Harmonic content** (pitched instruments, chords, melody) — stable horizontally (same frequencies persist over time).
- **Percussive content** (drums, transients) — broadband vertically (spike across all frequencies at onset).

**How it works:**
1. Compute power spectrogram.
2. Apply horizontal median filter → enhanced harmonic layer.
3. Apply vertical median filter → enhanced percussive layer.
4. Compute soft masks from the ratio: `H_mask(t,f) = H(t,f) / (H(t,f) + P(t,f))`.
5. Multiply masks by original spectrogram, reconstruct.

**Available in:** `librosa.effects.hpss()` — 2 lines of code.

**Strengths:**
- No model, no GPU, no download — instant.
- Naturally isolates melodic music from transient speech (speech has many transients but is not harmonically stable in the way music is).
- Can separate drums from melody as a standalone stage.

**Weaknesses:**
- Cannot cleanly separate vocals from instruments (both are harmonic).
- Median filter kernel size is a blunt tuning parameter.
- Breaks down with highly rhythmic speech or sung vocals.

**Use case for Music Assassin:**
Pre-processing stage before neural separation. Run HPSS, extract the harmonic layer, send only that to Demucs/MDX-Net. Reduces the input dimensionality the neural model must handle and can improve separation of melodic music beds under dialogue.

---

### NMF — Non-Negative Matrix Factorization

**What it is:**
Factorizes the magnitude spectrogram `V ≈ W × H` where:
- `W` = basis matrix (spectral templates — what each source "sounds like").
- `H` = activation matrix (when each basis is active over time).

**How it works:**
Initialize `W` and `H` randomly, then iteratively update using multiplicative update rules to minimize reconstruction error. In supervised NMF, you provide pre-learned `W` from isolated instrument recordings.

**Strengths:**
- Fully **unsupervised** — no training data needed if you provide templates.
- Interpretable: each column of `W` is a learned timbre.
- Lightweight: runs in real-time on CPU.
- Great for isolating specific instruments if you have a clean reference recording of that instrument.

**Weaknesses:**
- Struggles with polyphonic, complex mixes.
- Results are sensitive to initialization and number of components.
- Outperformed by deep learning on standard benchmarks by large margins.

**Use case for Music Assassin:**
"Reference-guided removal" — if a user can provide a 5-second clip of the background music alone, NMF can learn its spectral template and subtract it, without any neural model. Good fallback for cases where the music is a known loop or jingle. `sklearn.decomposition.NMF` is the simplest implementation.

---

### Adaptive Filters (LMS / RLS)

**What it is:**
Used in classic Adaptive Noise Cancellation (ANC). Requires two inputs:
- **Primary**: speech + music mix.
- **Reference**: music alone (from a second microphone or a known source).

The filter adapts its coefficients to make its output match the music in the primary channel, then subtracts it.

**Algorithms:**
- **LMS** (Least Mean Squares): simplest, slow convergence, stable.
- **NLMS** (Normalized LMS): faster, less sensitive to step size.
- **RLS** (Recursive Least Squares): fastest convergence, higher compute.

**Strengths:**
- Real-time capable (runs sample-by-sample).
- Handles non-stationary signals if convergence rate is tuned.
- No training data needed — adapts on-the-fly.

**Weaknesses:**
- Requires a **reference signal** — not available in single-track scenarios.
- Assumes music in the primary channel is a filtered version of the reference.
- IIR adaptive filters suffer from stability issues.

**Use case for Music Assassin:**
Applicable only if the user provides a separate music-only track (e.g., the original soundtrack file). In that case, LMS adaptive subtraction is faster and cleaner than neural separation for known-music removal. Could add as a "reference mode."

---

### Notch Filters & Hum Removal

**What it is:**
Narrow-band IIR filters that null specific frequencies. Common targets:
- 50 Hz / 60 Hz mains hum + harmonics (100/120/200/240 Hz).
- Fixed-pitch musical drone or tone.

**Implementation:** `scipy.signal.iirnotch(freq, Q, fs)` — one line per notch.

**Strengths:**
- Zero latency (can run in real-time).
- Zero artifacts outside the notch band.
- Handles deterministic tonal interference perfectly.

**Weaknesses:**
- Only works for known, fixed-frequency interference.
- Useless for broadband or melodically varying music.

**Use case for Music Assassin:**
Already partially implemented (hum remover in FairlightFX-inspired post-filter). Could be extended to auto-detect dominant music tones in quiet speech gaps using FFT peak-picking, then notch those specific frequencies.

---

## 2. Deep Learning Architectures

### UNet / Spectrogram-Domain UNets

**What it is:**
Encoder-decoder network with skip connections, operating on STFT magnitude spectrograms. The encoder compresses the spectrogram, the decoder reconstructs a mask per stem, skip connections preserve fine detail.

**Key variants:**
- `UNet` (original, 2D conv on spectrogram).
- `TFC-TDF UNet` — adds Time-Frequency Convolution + Time-Distributed Fully connected. ICASSP 2024.
- `Conditioned UNet` — same architecture but conditioned on text/label embedding to select which stem to separate.

**VR Architecture in UVR** uses a UNet variant trained specifically on vocal removal tasks.

**Strengths:**
- Well-studied, stable training.
- Efficient per-stem when fine-tuned.
- VR Architecture models have been trained on enormous community-contributed datasets via UVR.

**Weaknesses:**
- Fixed receptive field — can miss long-term temporal dependencies.
- Separate model per stem (4 models for 4 stems, unless multi-output head).

**Use case for Music Assassin:**
You already use VR Architecture models via `audio-separator`. The UVR model zoo has hundreds of fine-tuned variants covering: vocal focus, de-echo, dereverb, backing vocals, crowd noise. These are worth exploring selectively.

---

### LSTM / Open-Unmix (UMX)

**What it is:**
3-layer bidirectional LSTM operating on STFT magnitude, one network trained per stem. The academic reference baseline from SiSEC 2018.

**Repo:** [sigsep/open-unmix-pytorch](https://github.com/sigsep/open-unmix-pytorch)

**Strengths:**
- Well-documented, stable, easy to understand.
- Lightweight relative to transformer-based models.
- Good for understanding mask-based separation conceptually.

**Weaknesses:**
- Significantly outperformed by modern architectures on all benchmarks.
- LSTM sequential computation is slow at long sequence lengths.

**Use case for Music Assassin:**
Educational / baseline reference only. Not recommended for production. The jump from UMX to HTDemucs or BS-RoFormer is substantial.

---

### Demucs / HTDemucs (Hybrid Transformer Demucs)

**What it is:**
Meta's hybrid architecture combining waveform-domain (time-domain) U-Net with a spectrogram-domain branch. The two branches share information via cross-attention at multiple scales. HTDemucs adds Transformer encoder layers to the bottleneck.

**Models available (full list from Demucs repo):**
| Model | Description | Notes |
|---|---|---|
| `htdemucs` | 4-stem, balanced quality/speed | Default |
| `htdemucs_ft` | Fine-tuned per stem | Best Demucs quality |
| `htdemucs_6s` | 6-stem (adds guitar + piano) | Experimental |
| `hdemucs_mmi` | Hybrid Demucs v3 retrained on MusDB HQ | v3 baseline |
| `mdx` | MDX21 challenge winner (MusDB HQ only) | Competition model |
| `mdx_extra` | MDX track B + extra training data | Higher SDR |
| `mdx_q` | Quantized MDX (~300 MB) | Smaller, slight quality loss |
| `mdx_extra_q` | Quantized MDX Extra (~300 MB) | Best quality + quantized |

**Key Demucs inference flags:**
- `--two-stems=vocals` — only compute vocals vs. everything else (~2× faster, avoids computing drums/bass/other stems we don't need)
- `--shifts N` — random-shift TTA (GPU only); same polarity-flip idea but time-domain
- `--segment N` — reduce segment size to lower peak RAM (≥10 s recommended)
- `--overlap 0.25` — default 25% overlap between chunks; increase for smoother joins
- `--mp3` + `--mp3-bitrate 320` — output MP3 directly

**Strengths:**
- Produces naturally sounding output with minimal metallic artifacts.
- Waveform domain avoids STFT phase issues.
- Open weights, well-maintained by Meta.

**Weaknesses:**
- Large memory footprint (~600MB+ weights for mdx_extra).
- Slower than MDX-Net on CPU.
- Outperformed on vocals specifically by BS-RoFormer.

**Current status in Music Assassin:** Primary model. You use `htdemucs`, `htdemucs_ft`, `mdx_extra` via chunked processing.

---

### MDX-Net

**What it is:**
TFC-TDF UNet operating in the STFT domain. Developed by Kuielab & Woosung Choi, won the MDX21 competition. Uses Time-Frequency Channel (TFC) convolutions combined with Time-Distributed Fully-connected (TDF) layers.

**Models relevant to Music Assassin (via audio-separator):**
| Model | Size | Format | Specialty | Status |
|---|---|---|---|---|
| `UVR-MDX-NET-Voc_FT.onnx` | 66 MB | ONNX | Vocals, fast CPU | ✓ in use |
| `Kim_Vocal_2.onnx` | 130 MB | ONNX | Vocals, more robust | ✓ in use |
| `MDX23C-8KFFT-InstVoc_HQ.ckpt` | 448 MB | torch | Best vocals, MDXC arch | ✓ added v0.4.2-RT |
| `MDX23C-8KFFT-InstVoc_HQ_2.ckpt` | 448 MB | torch | Improved variant | ✓ added v0.4.2-RT |
| `UVR-MDX-NET-Inst_HQ_4.onnx` | 66 MB | ONNX | Instrumental stem | candidate |

**MDX23C architecture (MDXC):**
- Based on KUIELab TFC-TDF v3 with MDXC improvements.
- Outputs both vocals and instrumental stems simultaneously.
- Uses `.ckpt` format (torch), NOT onnxruntime — ~2–4× slower than ONNX MDX-Net on CPU.
- Weights hosted publicly on HuggingFace: `Blane187/all_public_uvr_models`.
- Auto-downloaded by audio-separator on first use (~448 MB per model).
- SDR ~10 on vocals vs ~8 for MDX-Net Voc_FT.

**Strengths:**
- ONNX variants: lightweight (66–130 MB), fast via onnxruntime, CPU-friendly.
- MDX23C: highest open-source vocal SDR of any model usable via audio-separator.

**Weaknesses:**
- MDX23C `.ckpt`: requires torch (heavy), ~2–4× slower than ONNX on CPU.
- Fixed input length per chunk (less flexible than waveform models).

**Current status in Music Assassin:** ONNX models in use since v0.4.2. MDX23C added in v0.4.2-RT testing branch.

---

### BS-RoFormer — Band-Split RoPE Transformer

**What it is:**
ByteDance / ICASSP 2024. Current **state-of-the-art** for vocal separation. Splits the complex spectrogram into logarithmically-spaced frequency subbands, then applies hierarchical Transformers:
1. **Inner-band Transformer** — models temporal patterns within each subband.
2. **Inter-band Transformer** — models relationships across subbands.

Uses **Rotary Position Embedding (RoPE)** for better positional generalization.

**Won:** Sound Demixing Challenge SDX23 (first place, music separation track).

**Repos:**
- [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) — clean PyTorch implementation.
- [openmirlab/bs-roformer-infer](https://github.com/openmirlab/bs-roformer-infer) — inference-only, lightweight API with auto checkpoint management.

**Performance:** Best-in-class on vocals, competitive on all 4 stems.

**Strengths:**
- SOTA vocal separation quality.
- Handles complex polyphonic mixes better than any UNet variant.
- Band-split structure aligns with perceptual frequency resolution of hearing.

**Weaknesses:**
- Large model (3+ GB for best weights).
- Slower inference than MDX-Net.
- GPU preferred for reasonable speed.

**Use case for Music Assassin:**
Add as a premium "Ultra" separation mode. Available via `audio-separator` with ZFTurbo's pretrained weights on HuggingFace.

---

### MelBand-RoFormer

**What it is:**
Variant of BS-RoFormer using **Mel-scaled frequency bands** instead of linear bands. Mel scale better matches human hearing perception and how instruments are distributed spectrally.

**Paper:** [arxiv 2310.01809](https://arxiv.org/pdf/2310.01809)

**ZFTurbo pretrained weights:** `mel-roformer-zfturbo-vocals-v1` on HuggingFace. Achieves **8.94 SDR** with optimal settings.

**Strengths:**
- Slightly better perceptual quality than BS-RoFormer due to Mel alignment.
- Same architecture benefits as BS-RoFormer.
- ZFTurbo weights available, plug-in via `audio-separator`.

**Weaknesses:**
- Similar compute requirements to BS-RoFormer.
- Not yet as widely tested as BS-RoFormer in production.

**Use case for Music Assassin:**
Alternative to BS-RoFormer; potentially better on vocals specifically due to Mel perceptual alignment. Worth A/B testing on your target use case (dialogue under music beds).

---

### SCNet — Sparse Compression Network

**What it is:**
ICASSP 2024. Combines band-splitting (from BS-RoFormer) with **Conformer blocks** (Convolution + Attention hybrid). Designed for efficiency: one model handles all 4 stems simultaneously.

**Key design choices:**
- Sparse compression reduces redundant representations.
- Conformer block (convolutional module + self-attention) captures both local and global dependencies.
- Single model output head per stem — no separate models needed.

**Performance:** SCNet XL IHF achieves **10.08 SDR** (same as MDX23C top variant) at fewer parameters than BS-RoFormer.

**Strengths:**
- One model for all 4 stems — simpler deployment.
- Competitive quality at lower parameter count.
- Faster inference than BS-RoFormer.

**Weaknesses:**
- Less vocal-specialized than BS-RoFormer.
- Newer, less community testing.

**Use case for Music Assassin:**
Best choice for a single high-quality general-purpose model that handles all stems without needing to load 4 separate models. Would simplify the architecture significantly.

---

### Conditioned UNet (Text/Label-Guided Separation)

**What it is:**
Standard UNet architecture extended with a conditioning mechanism — a text embedding or one-hot stem label is injected at each decoder layer, telling the network which stem to extract.

**Paper:** [arxiv 2512.15532 — Dec 2024](https://arxiv.org/pdf/2512.15532)

**Strengths:**
- One model handles arbitrary stems without retraining.
- Users can specify "remove only the piano" or "isolate kick drum."
- Natural language prompting is the direction SAM Audio takes this further.

**Weaknesses:**
- Still requires training on labeled multi-stem data.
- Text conditioning quality depends on how well text embeddings align with audio features.

**Use case for Music Assassin:**
Future direction — a "remove [drums/bass/guitar/piano]" per-instrument toggle in the GUI. Currently achievable with 4-stem model outputs + per-stem mute, but a conditioned model is cleaner.

---

### Generative / Flow-Based Source Separation

**What it is:**
Instead of discriminative mask estimation, trains a **normalizing flow** (generative) model for each source. At inference, uses likelihood-based objectives to find the combination of source samples that best explains the mixture.

**Repo:** [gzhu06/GenerativeSourceSeparation](https://github.com/gzhu06/GenerativeSourceSeparation)

**Strengths:**
- Can hallucinate plausible source content even when a stem is nearly absent.
- Theoretically more principled (proper probabilistic framework).
- Can generate multiple plausible separations (sampling).

**Weaknesses:**
- Very slow at inference (iterative optimization per mixture).
- Not yet competitive with discriminative models on benchmark metrics.
- Complex to train and deploy.

**Use case for Music Assassin:**
Research interest only at this stage. The direction is promising for cases where music completely dominates (speech nearly inaudible) — a discriminative model returns noise, but a generative model can reconstruct plausible speech.

---

### HS-TasNet — Hybrid Spectrogram TasNet (Real-Time, 23 ms)

**Paper:** [arxiv 2402.17701](https://arxiv.org/abs/2402.17701) — L-Acoustics, Feb 2024
**Repos:** [lucidrains/HS-TasNet](https://github.com/lucidrains/HS-TasNet) (MIT) · [temismink/HS-Tasnet](https://github.com/temismink/HS-Tasnet)
**Install:** `pip install HS-TasNet`

**What it is:**
Designed from the ground up for real-time/streaming inference. Hybrid waveform + spectrogram branches (like HTDemucs) but with causal convolutions and a small look-ahead window instead of global context.

**Key specs:**
- **Latency: 23 ms** at 44100 Hz — suitable for live broadcast and hearing aids.
- SDR: 4.65 on MusDB (vs 8.5 for HTDemucs) — the quality/latency tradeoff.
- Has a built-in `model.sounddevice_stream(duration_seconds=N)` method for real-time demo.
- Checkpoint loading via `HSTasNet.init_and_load_from('./path/to.ckpt')`.

**Status (June 2026):** No publicly available pretrained weights. Must train on MUSDB18.
Training: `python train.py --dataset_path /path/to/musdb18hq`

**Similar projects using same paradigm:**
- Conv-TasNet (ancestor) — fully causal, <10 ms, lower quality, available via `asteroid` library
- BSRNN (hearing aid variants) — causal unidirectional LSTM, chunks with state caching

**Use case for Music Assassin:**
Infrastructure added in v0.4.2-RT. Activate by placing a trained `.pt` checkpoint in `pretrained_models/`. This is the path to true real-time daemon mode.

---

### RT-STT — Real-Time Single-Path TFC-TDF UNet (Nov 2025)

**Paper:** [arxiv 2511.13146](https://arxiv.org/abs/2511.13146) — Nov 2025
**Status:** No public GitHub or weights as of June 2026.

**What it is:**
Evolves the Dual-Path TFC-TDF UNet (DTTNet, ICASSP 2024) into a **single-path** design optimised for real-time feasibility:
- Single-path modeling outperforms dual-path in RT scenarios by enabling parallelism.
- **Channel expansion fusion** technique recovers quality lost from removing the dual-path.
- Applies INT8 **quantisation** by default (same trick Music Assassin already uses for HTDemucs).
- Benchmarked with real-time factor < 1.0 — processes audio faster than it plays.

**Related open project:**
[junyuchen-cjy/DTTNet-Pytorch](https://github.com/junyuchen-cjy/DTTNet-Pytorch) — the ICASSP 2024 dual-path precursor, 10.12 dB cSDR on vocals, 86% fewer parameters than BSRNN. Public weights available.

**Use case for Music Assassin:**
Placeholder in v0.4.2-RT. When a public implementation appears, this becomes the preferred RT model (better than HS-TasNet at comparable latency, plus native INT8).

---

### TISDiSS — Training-Time and Inference-Time Scalable Source Separation

**Paper:** [arxiv 2509.15666](https://arxiv.org/pdf/2509.15666)

**What it is:**
A framework for flexible speed–quality trade-offs at inference time without retraining:
- **Unified early-split multi-loss supervision** — trains multiple output heads at different compute budgets.
- **Shared-parameter design** — one model, multiple inference depths.
- **Dynamic inference repetitions** — caller decides how many passes to run vs. how much time is available.

**Relevance:**
Directly applicable to Music Assassin: same model could run in "fast" mode (1 pass, lower quality) for live filtering or "quality" mode (N passes) for offline processing. The current Normal/Balanced/Aggressive mode distinction could map onto this.

---

### Band-SCNet — Causal Lightweight (2025)

**Paper:** [INTERSPEECH 2025 — yang25d](https://www.isca-archive.org/interspeech_2025/yang25d_interspeech.pdf)

**What it is:**
A **causal** (streaming-capable), lightweight variant of SCNet using band-splitting. Designed for high-performance real-time separation at low computational cost.

**Key specs:**
- Causal design — no future context, suitable for streaming.
- Lightweight — fewer parameters than SCNet standard.
- High performance relative to compute budget.

**Relationship to SCNet:**
SCNet uses band-splitting + Conformer blocks. Band-SCNet makes the Conformer causal and trims parameters. Represents the direction of real-time + quality convergence.

**Use case for Music Assassin:**
When public weights appear (expected alongside INTERSPEECH 2025 materials), this could replace HS-TasNet as the RT model — better quality at similar latency.

---

## 3. Open-Source Projects & Repos

| Project | Architecture | Stems | License | Notable Feature |
|---|---|---|---|---|
| [Demucs (Meta)](https://github.com/facebookresearch/demucs) | Hybrid waveform+spectrogram UNet + Transformer | 4 + 6-stem | MIT | Best general-purpose; `--two-stems` mode; 8 models |
| [UVR — Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) | MDX-Net, VR-Arch, Demucs | Vocals + instruments | MIT | Largest community model zoo; hundreds of fine-tuned variants |
| [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | Wraps UVR + Demucs + MDXC | All | MIT | Python API Music Assassin uses; MDX23C + overlap/chunk params |
| [HS-TasNet (lucidrains)](https://github.com/lucidrains/HS-TasNet) | Hybrid TasNet (causal) | 4 | MIT | **23 ms latency**; sounddevice_stream built-in; pip installable |
| [HS-TasNet (reference)](https://github.com/temismink/HS-Tasnet) | Same | 4 | MIT | Original ICASSP 2024 implementation |
| [DTTNet (dual-path TFC-TDF)](https://github.com/junyuchen-cjy/DTTNet-Pytorch) | Dual-Path TFC-TDF UNet | 4 | MIT | ICASSP 2024; 10.12 dB vocals; 86% fewer params than BSRNN; public weights |
| [BS-RoFormer (lucidrains)](https://github.com/lucidrains/BS-RoFormer) | Band-Split RoPE Transformer | 4 | MIT | SOTA vocals; SDX23 winner |
| [bs-roformer-infer](https://github.com/openmirlab/bs-roformer-infer) | BS-RoFormer | 4 | MIT | Inference-only, auto checkpoint download |
| [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) | BS-RoFormer, MelBand-RoFormer, SCNet, MDX23C | 4 | Apache 2 | Best open pretrained weights; TTA; ONNX/TensorRT export |
| [MVSEP-MDX23](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model) | MDX23C | 4 | Apache 2 | Contest-winning model; `--overlap_large`/`--overlap_small`; `--cpu` flag |
| [Open-Unmix (UMX)](https://github.com/sigsep/open-unmix-pytorch) | BiLSTM | 4 | MIT | Academic reference; torch 2.0 updated; well-documented |
| [Spleeter (Deezer)](https://github.com/deezer/spleeter) | UNet (TensorFlow) | 2 / 4 / 5 stems | MIT | Established; easy API; outclassed by newer models |
| [asteroid](https://github.com/asteroid-team/asteroid) | Conv-TasNet + many others | Speech/music | MIT | PyPI-installable; includes causal Conv-TasNet for RT |
| [Generative Source Separation](https://github.com/gzhu06/GenerativeSourceSeparation) | Normalizing Flows | Music stems | MIT | Flow-based; handles near-inaudible speech case |
| [librosa HPSS](https://librosa.org/doc/main/auto_examples/plot_hprss.html) | DSP (median filter) | Harmonic + Percussive | ISC | Zero-dependency fast pre-separation; `librosa.effects.hpss()` |
| [AudioSeparation (set-soft)](https://github.com/set-soft/AudioSeparation) | Wraps audio-separator | All | MIT | CLI + ComfyUI interface; useful reference for UI integration |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) | SepFormer, DPRNN, Conv-TasNet, MetricGAN+, MVDR | Speech | Apache 2 | DSP+DL hybrid pipeline; 100+ HF models; phase-bypass architecture |
| [SepReformer](https://github.com/dmlguq456/SepReformer) | Asymmetric Enc-Dec + Global/Local Transformer | Speech | MIT | NeurIPS 2024; better compute-quality ratio than SepFormer |
| [Bandit/Banquet (CASS)](https://github.com/kwatcharasupat/source-separation-landing) | Band-Split + FiLM conditioning | 4 cinematic stems | Non-commercial | Dialogue + singing vocals + music + SFX separation |

**Key open-source weight repositories (HuggingFace):**
| Weights | Source | Format | SDR |
|---|---|---|---|
| `MDX23C-8KFFT-InstVoc_HQ.ckpt` | [Blane187/all_public_uvr_models](https://huggingface.co/Blane187/all_public_uvr_models) | torch | ~10 |
| `MDX23C-8KFFT-InstVoc_HQ_2.ckpt` | Same repo | torch | ~10 |
| `mel-roformer-zfturbo-vocals-v1` | [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md) | torch | 8.94 |
| `BS-RoFormer` weights | [ZFTurbo pretrained models](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md) | torch | SOTA |
| `SCNet-XL-IHF` | Same | torch | 10.08 |

---

## 4. Inference Optimisation Techniques

These apply across model types and are independent of model selection. All are open-source/free.

### Test-Time Augmentation (TTA) — Polarity Flip

**What it is:** Run the model twice — once on normal audio, once on polarity-inverted audio. Average the results:
```
result = (model(x) - model(-x)) / 2
```
A well-trained model satisfies `model(-x) ≈ -model(x)`. Averaging cancels asymmetric biases.

**Effect:** +0.1–0.3 dB SDR. **Cost:** 2× inference time.
**Works with:** Any separator (Demucs, MDX-Net, MDX23C, HS-TasNet, audio-separator).
**Added to Music Assassin:** v0.4.2-RT (TTA checkbox).

**ZFTurbo's extended TTA** (from `inference.py`):
- Polarity flip (as above).
- Channel swap (L↔R) for stereo.
- Both combined (4 passes total).
- `--use_tta` flag in ZFTurbo's inference.py.

---

### Chunking & Overlap Strategies

**The core pattern:**
```
input → [chunk 0] [chunk 1] [chunk 2] ...
                 ↑overlap↑ ↑overlap↑
output ← crossfade at chunk boundaries (linear fade in/out)
```

**Key parameters from production repos:**

| Param | Demucs | audio-separator MDXC | Music Assassin |
|---|---|---|---|
| Chunk size | `--segment N` (≥10 s) | `--chunk_duration` (minutes) | 60 s hardcoded |
| Overlap | `--overlap 0.25` | `--overlap_large 0.6` | 2 s crossfade |
| Crossfade | Linear fade in/out | Simple concatenation | Linear fade |

**MDXC-specific (audio-separator):**
- `overlap` parameter: 2–50 (higher = better quality, slower). Default 8.
- `segment_size`: 256 default; larger uses more RAM but may improve quality.
- `batch_size`: 1 default; larger uses more RAM, slightly faster.

**ZFTurbo's `bigshifts_wrapper`:**
- Shifts audio by random amounts before chunking, averages the shifted outputs.
- Eliminates chunk-boundary artifacts more effectively than crossfading.
- Available in `ZFTurbo/Music-Source-Separation-Training` inference.py.

---

### Quantisation (INT8 / FP16)

| Method | Speedup | Quality loss | Works on |
|---|---|---|---|
| INT8 dynamic (torch) | 2–3× | ~0.1–0.3 dB | CPU; htdemucs Linear layers |
| FP16 (torch half) | ~2× | negligible | GPU only |
| ONNX INT8 | 2–4× | ~0.1–0.2 dB | CPU; MDX-Net ONNX models |
| TensorRT FP16 | 25–30× | negligible | NVIDIA GPU only |

**Already in Music Assassin:** INT8 dynamic for htdemucs (checkbox in UI).
**MDX-Net ONNX** already runs in FP32 via onnxruntime — can be quantised to INT8 with onnxruntime's quantization API for further speedup.

---

### Model Ensembling

Average the stem outputs from two or more models:
```python
vocals = (model_a.separate(audio) + model_b.separate(audio)) / 2
```
**Effect:** +0.5–1.0 dB SDR with no retraining.
**Best pairs:** HTDemucs_ft + MDX23C (different architectures, different error patterns).
**Cost:** 2× inference time (can run models in parallel on multi-core CPU with threading).
**Documented in:** [ZFTurbo ensemble.md](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/ensemble.md)

---

### two-stems Mode (Demucs)

Using `--two-stems=vocals` (CLI) or `Separator(two_stems='vocals')` (Python API) tells Demucs to only compute the vocals + everything-else split instead of all 4 stems. Since each stem head runs independently in HTDemucs:
- **~2× faster** — skips drums, bass, other heads.
- **Same vocals quality** as full 4-stem run.
- Available via `demucs.api.Separator` class.

Music Assassin currently calls `apply_model` at the low level and manually extracts the vocals index. Switching to `demucs.api.Separator(two_stems='vocals')` would give this speedup automatically.

---

### Input Normalisation + Denormalisation

From ZFTurbo's `inference.py`:
```python
if config.inference["normalize"]:
    mix, norm_params = normalize_audio(mix)
    # ... run model ...
    estimates = denormalize_audio(estimates, norm_params)
```
Normalising to a consistent RMS before inference prevents the model from dealing with extreme loud/quiet inputs, which can cause clipping or silence in the output. Music Assassin currently normalises in `audio_prefilter()` — the same principle.

---

## 5. Commercial AI Tools (reference only)

### LALAL.AI

**Models:** Phoenix → Cassiopeia → Orion → Andromeda (newest).

**Phoenix architecture highlights:**
- Trained on ~2 TB of audio data (2× more data than predecessor).
- Borrowed activation functions from computer vision, adapted for audio.
- Advanced normalization methods to reduce artifacts.
- 2× faster than predecessor; better backing vocal handling.

**Stems available:** Vocals, accompaniment, drums, bass, electric guitar, acoustic guitar, piano, synthesizer, strings, wind instruments, backing vocals.

**Differentiator:** Per-instrument stem removal at a granularity beyond standard 4-stem models. Also provides a **Voice Cleaner** (noise + reverb removal independent of stem separation).

**Weakness:** Closed source, paid service, no local deployment option.

**Relevant to Music Assassin:** Their per-instrument granularity and combined voice cleaning pipeline are the two biggest gaps Music Assassin currently has.

---

### remove.music

**Approach:** Music-specific model trained for a **single task**: remove the full instrumental bed (drums + bass + melody + score + ambient) while preserving dialogue, breath, and natural reverb.

**Key design decision:** Single-purpose model instead of general stem separator. This focus produces cleaner dialogue extraction than a general 4-stem split-and-subtract approach.

**Technical note:** Uses PANNs (Pre-trained Audio Neural Networks) to detect music presence probability before applying separation — avoids running expensive models on segments that are already clean speech.

**Relevant to Music Assassin:** 
- PANNs-based music detection gate — only run the separator on segments where music is actually detected. Saves compute and avoids re-processing clean speech.
- The concept of a music-detection pre-screen is directly implementable via `panns-inference` (Python package).

---

### AudioCleaner.ai

**Models:** V1 (reliable, broad compatibility) + V2 alpha (larger dataset, optimized for complex environments, more natural speech output).

**Stem-level removals offered:** Vocal removal, drum removal, guitar removal, bass removal, piano removal, full instrumental extraction, background music suppression.

**Additional tools:** Breath removal, silence removal, echo/reverb removal, wind noise, static/buzzing — all as discrete post-separation steps.

**Video support:** Accepts MP4/MKV/MOV up to 10 GB; handles extraction and remux internally.

**Relevant to Music Assassin:**
- Two-model tier framing maps to current MDX-Net (V1) vs. BS-RoFormer (V2).
- Per-stem removal toggles from 4-stem output.
- Video file input/output.
- Discrete post-processing steps (reverb, breath, silence) as optional checkboxes.

---

### AudioShake

**Target audience:** Film, TV, music licensing post-production.

**Capabilities:**
- Stem splitting with explicit instrument removal.
- Noise suppression pipeline combined with stem separation in one pass.
- API-first design for integration into professional workflows.

**Differentiator:** Combines noise suppression + stem separation in a unified pipeline rather than running them sequentially. This avoids artifacts that accumulate when you chain two separate models.

**Relevant to Music Assassin:** Combined noise+stem pipeline concept — rather than applying pre-filter → separator → post-filter as three stages, a unified model could handle all three simultaneously.

---

### Moises.ai

**Capabilities:** Stem splitting (4–8 stems), BPM + key detection, real-time capo, chord detection.

**Music detection approach:** Uses PANNs (Pre-trained Audio Neural Networks) to estimate the probability of music presence at each time frame before applying the separation model. Only processes frames above the music-presence threshold.

**Relevant to Music Assassin:** VAD (Voice Activity Detection) + music detection gating before running the heavy model. Already mentioned under remove.music — Moises confirms PANNs as the go-to tool for this.

---

### DaVinci Resolve Studio (Fairlight AI)

**Audio AI features:**
1. **Voice Isolation** — Fairlight AI effect that removes all non-speech content (music + noise) from a dialogue track in a single pass. Different from stem separation: it's tuned for "clean speech" as the output, not for musicological decomposition.
2. **Music Remixer** — Rebalances stems in the mix. Enables partial suppression (reduce music by 70%) rather than binary removal.
3. **Auto-ducking** — Detects dialogue presence via signal level; automatically reduces music track gain during speech segments. Not AI per se, but dialogue-aware music suppression.
4. **FairlightFX — Noise Reduction** — "Learn" mode: selects a noise-only segment, learns its spectrum, applies Wiener-like subtraction. Essentially spectral subtraction with a learned reference.
5. **FairlightFX — Hum Remover** — Targeted notch filter for 50/60 Hz hum + harmonics.
6. **FairlightFX — De-esser** — Sibilance suppressor (frequency-aware dynamic compression on high frequencies).
7. **IntelliTrack AI** — Uses video object tracking to automate panning on audio tracks. Not directly applicable, but represents the "multimodal" direction where video context informs audio processing.

**Relevant to Music Assassin:**
- Voice Isolation paradigm: instead of exposing stems, offer a single "isolate dialogue" button that combines separation + denoise in one user action.
- Music Remixer concept: suppression strength slider (invert instrumental stem, mix back at fractional gain).
- Auto-ducking: VAD-gated processing where music removal is only applied to segments containing speech.

---

## 5. Frontier Research (2025)

### SAM Audio — Meta AI (December 2025)

**Paper:** [arxiv 2512.18099](https://arxiv.org/abs/2512.18099)

**What it is:** A foundation model for **universal audio separation**. Analogous to SAM (Segment Anything Model) in computer vision. One model handles speech, music, general sounds.

**Architecture:** Diffusion Transformer trained with **flow matching** on large-scale audio data.

**Prompting modalities:**
1. **Text prompt** — "separate the background music" / "isolate the drums."
2. **Visual mask prompt** — select a region of a spectrogram visually.
3. **Temporal span prompt** — "give me the source active from 2.3s to 5.1s."

**Performance:** State-of-the-art across speech, music, musical instrument, and in-the-wild audio separation benchmarks. Outperforms both general-purpose models (prior foundation models) and specialized systems (Demucs, UVR).

**Current availability:** Weights public. Not yet integrated into `audio-separator`. Integration expected within 6–12 months given the community's track record with new models.

**Relevant to Music Assassin:**
This is the single biggest incoming upgrade. When integrated into `audio-separator`, text-prompted separation would replace the current hard-coded stem logic. The GUI's separator selector could become a text field: "remove background music while keeping dialogue."

---

### Mel-RoFormer + Vocal Melody Transcription

**Paper:** [arxiv 2409.04702](https://arxiv.org/pdf/2409.04702)

**What it is:** Extension of MelBand-RoFormer that simultaneously separates vocals AND outputs MIDI-level vocal melody transcription. Separation quality improves because the model is forced to understand the musical structure of the vocal.

**Relevant to Music Assassin:** Side benefit — if integrated, Music Assassin could optionally output a melody/pitch track alongside the cleaned audio. Useful for karaoke, transcription, or pitch correction workflows.

---

### Ensemble / Hierarchical Stem Approaches

**Paper:** [arxiv 2410.20773](https://arxiv.org/html/2410.20773v1)

**What it is:** Instead of one model for all stems, uses a hierarchy:
1. First pass: coarse separation (music vs. dialogue).
2. Second pass: refine each coarse output with a stem-specific model.

Also covers **model ensembling** — average predictions from multiple architectures (BS-RoFormer + SCNet + HTDemucs) to get better results than any single model.

**Relevant to Music Assassin:** Model ensembling is directly implementable — run two models and average their outputs. Can improve SDR by 0.5–1.0 dB without any retraining. The chunked processing pipeline already parallelizes well for this.

---

### Knowledge-Driven / Cinematic Audio Separation

**Paper:** [arxiv 2602.21476](https://arxiv.org/pdf/2602.21476)

**What it is:** Uses domain knowledge (music theory, score structure, instrument physics) to guide separation rather than relying purely on learned embeddings. Targets film/cinematic audio where the source material is professionally mixed.

**Relevant to Music Assassin:** Cinematic audio (films, YouTube videos, podcasts with music beds) is the primary use case. This approach is more robust when music is mixed at consistent levels following professional loudness standards (which most content is).

---

## 5b. SpeechBrain Toolkit — Hybrid DSP + DL Pipeline

**Repo:** [github.com/speechbrain/speechbrain](https://github.com/speechbrain/speechbrain) (Apache 2.0)
**Docs:** [speechbrain.readthedocs.io](https://speechbrain.readthedocs.io/en/latest/)
**Install:** `pip install speechbrain`
**HuggingFace:** [huggingface.co/speechbrain](https://huggingface.co/speechbrain) — 100+ pretrained models

### The Pipeline the User Remembers: Phase-Bypass DSP+AI Chain

SpeechBrain's source separation and enhancement pipeline is the clearest example of **DSP filters feeding into AI models with a feedforward of the original audio**. The architecture:

```
Input Mixture
      │
      ├──────────────────────────────────── PHASE BYPASS ──────────────┐
      │                                                                  │
      ▼                                                                  │
[DSP: STFT]  → Magnitude   →  [Neural Mask Network]  →  Masked Mag     │
              → Phase (original) ──────────────────────────────────────►│
                                                                         │
                                                               Masked Mag × exp(j·phase)
                                                                         │
                                                                    [DSP: ISTFT]
                                                                         │
                                                                  Output Source
```

In code (from SpeechBrain tutorial):
```python
# Encoder: STFT — DSP stage 1
y = torch.stft(inp, n_fft=self.fft_size, return_complex=True)
mag   = y.abs()    # magnitude only → goes to neural net
phase = y.angle()  # phase → BYPASSED (original audio feeds forward)

# Neural stage: LSTM/SepFormer produces a mask per source
mask = self.masking(mag)       # values 0-1 per frequency bin

# Decoder: original phase rejoins — DSP stage 2
sourcehat_dft = (mask * mag) * torch.exp(1j * phase)   # bypass reintegration
sourcehat     = torch.istft(sourcehat_dft, n_fft=self.fft_size)
```

**This is the feedforward the user remembered:** The original audio's phase information passes *through* the DSP (STFT) into the reconstruction, completely bypassing the neural network. The AI only modifies amplitude (via mask), the original time structure (phase) is preserved. This is why spectral masking produces natural-sounding output — it's not guessing the phase.

The SpeechBrain tutorial explicitly states: *"masks are basically band stop filters which aims to remove the interferences from the other source."* The neural mask IS a learned, adaptive DSP filter.

---

### Beamforming + Neural: Multi-Stage DSP → AI → DSP

For multi-microphone input, SpeechBrain implements a more explicit DSP → AI → DSP chain:

```
Microphone Array
      │
[DSP: STFT per mic]
      │
[DSP: Beamformer (MVDR/GEV/Delay-Sum)]  ← "without introducing nonlinearities"
   Generates frequency-wise weights W(k) × STFT of each mic
      │
[Neural: Post-filter / Mask Network]    ← cleans up beamformer residuals
      │
[DSP: ISTFT]
      │
Clean Speech
```

This is a fully **differentiable hybrid graph** — the DSP stages and neural stages are jointly trained. From the SpeechBrain paper: *"We implement delay-and-sum, MVDR, and GeV beamforming. Beamforming generates frequency-wise coefficients that multiply the STFT of each microphone and add the products to produce enhanced speech."*

Note: For single-channel audio (Music Assassin's use case), beamforming doesn't apply. But the same principle applies in the post-filter stage after stem separation.

---

### Two Enhancement Approaches in SpeechBrain

**1. Spectral Masking** (multiplicative, preserves phase):
- Noisy spectrogram → neural mask (0–1) → masked magnitude + original phase → ISTFT
- Preserves time structure; can produce artifacts if mask is imprecise
- Models: MetricGAN+, SEGAN, spectral masking baseline

**2. Spectral Mapping** (additive, learns clean spectrogram directly):
- Noisy spectrogram → neural network → predicted clean spectrogram
- More powerful but needs its own phase estimation
- Models: MetricGAN-U (unsupervised variant)

Music Assassin's current post-processing (formant enhancement, de-esser) is analogous to SpeechBrain's spectral masking — the original phase is implicitly preserved because the overlay-add crossfade and filtering happen in the time domain.

---

### Key SpeechBrain Models and Usage

All models use `from_hparams` for instant HuggingFace loading:

```python
from speechbrain.inference.separation  import SepformerSeparation
from speechbrain.inference.enhancement import SpectralMaskEnhancement

# Speech separation (speaker unmixing)
sep_model = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-whamr",
    savedir="pretrained_models/sepformer-whamr"
)
est_sources = sep_model.separate_file(path="dialogue_with_music.wav")
# Returns: Tensor [1, samples, 2] — two separated sources

# Speech enhancement post-processor (cleans vocals after separation)
enhancer = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank"
)
enhanced = enhancer.enhance_file("extracted_vocals.wav")
```

**Pretrained models (HuggingFace):**
| Model ID | Dataset | SI-SNRi | Input SR | Notes |
|---|---|---|---|---|
| `speechbrain/sepformer-wsj02mix` | WSJ0-2Mix | 22.4 dB | 8 kHz | 2-speaker separation |
| `speechbrain/sepformer-wsj03mix` | WSJ0-3Mix | 19.8 dB | 8 kHz | 3-speaker separation |
| `speechbrain/sepformer-libri2mix` | LibriMix 2spk | ~20 dB | 8 kHz | Robust to reading style |
| `speechbrain/sepformer-libri3mix` | LibriMix 3spk | ~19 dB | 8 kHz | 3-speaker |
| `speechbrain/resepformer-wsj02mix` | WSJ0-2Mix | 18.6 dB | 8 kHz | RE-SepFormer (efficient) |
| `speechbrain/sepformer-wham` | WHAM! | ~20 dB | 8 kHz | + environmental noise |
| `speechbrain/sepformer-whamr` | WHAMR! | 13.7 dB | 8 kHz | + noise + reverberation |
| `speechbrain/metricgan-plus-voicebank` | Voicebank-DEMAND | PESQ best | 16 kHz | Enhancement (not separation) |

---

### SepFormer Architecture — How the Encoder + Masknet + Decoder Works

The SepFormer uses a **learnable convolutional frontend** (not STFT) combined with dual-path Transformers in the masking network:

```
Waveform → [Conv1d Encoder]  → Chunks  → [Intra-chunk Transformer]
                                                 ↓
                                         [Inter-chunk Transformer]  ← Dual-path
                                                 ↓
                                         [Mask × Chunk Features]
                                                 ↓
                                         [ConvTranspose1d Decoder]
                                                 ↓
                                         Separated sources
```

Key: both Transformers use **multi-head self-attention with position encoding** — no recurrence. "Attention is all you need in speech separation."

Performance: 22.4 dB SI-SNRi (WSJ0-2Mix, dynamic mixing) — state-of-the-art at release (2020), still competitive.

---

### SepReformer — Asymmetric Encoder-Decoder (NeurIPS 2024)

**Paper:** [Separate and Reconstruct: Asymmetric Encoder-Decoder for Speech Separation](https://arxiv.org/abs/2406.05516)
**Repo:** [github.com/dmlguq456/SepReformer](https://github.com/dmlguq456/SepReformer) (MIT)

Evolutionary step from SepFormer:
- **Asymmetric** design: heavier encoder than decoder — most computation at encoding, reconstruction is lightweight.
- **Global + Local Transformer units** at each scale — captures both long-range dependencies (music beat structure, phrase level) and local patterns (note attack, phoneme timing) simultaneously.
- **More compute-efficient** than SepFormer at same quality level (better SI-SNR vs MACs ratio).
- Pretrained weights: `SepReformer-B` for WSJ0-2MIX (included in repo via Git LFS).

```bash
python run.py --model SepReformer_Base_WSJ0 --engine-mode infer_sample \
              --sample-file ./sample_wav/sample_WSJ.wav
```

---

### Cinematic Audio — Bandit/Banquet (ICASSP 2025 area)

**Paper:** [Facing the Music: Singing Voice Separation in Cinematic Audio](https://arxiv.org/abs/2408.03588)
**Code:** github.com/kwatcharasupat/source-separation-landing

Most relevant to Music Assassin (films, YouTube, TV content). 4-stem system:
- Dialogue · Instrumental music · **Singing vocals** · Effects (SFX)

Key insight: **singing voice is contextual** — the same vocal can belong to either the music stem or dialogue stem depending on story context. Standard 4-stem models get this wrong.

**Banquet** (the winner): Uses a FiLM (Feature-wise Linear Modulation) layer to condition a shared decoder per stem. Single shared decoder outperforms per-stem decoders (Bandit) because of better feature independence at the bottleneck.

⚠️ Weights carry non-commercial license (MUSDB18-HQ training data restriction). For Music Assassin (open-source), can be used for research/personal use, not redistribution.

---

### What SpeechBrain Is NOT Good For (Music Assassin Context)

| Limitation | Details |
|---|---|
| **8 kHz input** | All separation models expect 8 kHz. Music is 44.1 kHz. Must downsample → process → upsample. Quality loss. |
| **Speech-specific training** | Models trained on WSJ0/LibriMix speech, not music-over-dialogue content. Different spectral statistics. |
| **No music separation models** | No HTDemucs/MDX-Net equivalent in SpeechBrain's pretrained zoo. |
| **Slower than ONNX** | All SpeechBrain models require torch. MetricGAN+ (~16 kHz) is feasible on CPU; SepFormer at 8kHz is fast. |

---

### How SpeechBrain Applies to Music Assassin (Practical)

**Tier A — Use directly as post-processor (low effort, open-source, MIT):**
```python
# After Demucs/MDX-Net extracts vocals at 44100 Hz:
from speechbrain.inference.enhancement import SpectralMaskEnhancement
import torchaudio, torch

enhancer = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank"
)
# Resample 44.1k → 16k for MetricGAN+
vocals_16k = torchaudio.functional.resample(vocals_tensor, 44100, 16000)
clean_16k  = enhancer.enhance_batch(vocals_16k.unsqueeze(0), lengths=torch.tensor([1.0]))
# Resample back 16k → 44.1k
vocals_out = torchaudio.functional.resample(clean_16k.squeeze(0), 16000, 44100)
```
This adds a MetricGAN+ denoising pass over the extracted vocals, removing residual music bleed. Perceptually noticeable improvement especially for voices over loud music beds.

**Tier B — Use phase-bypass insight to understand Music Assassin's current pipeline:**
Music Assassin's existing `audio_prefilter()` → separation model → `advanced_postprocess_chunked()` IS implementing the same paradigm SpeechBrain formalises:
1. `audio_prefilter()` = DSP stage 1 (gate + high-pass = removes known-bad frequency ranges)
2. Separation model = neural mask stage (MDX-Net ONNX or HTDemucs)
3. Post-processing (de-esser, formant enhancement) = DSP stage 2

The models internally do the phase bypass automatically (MDX-Net: STFT-based masking, original phase kept; Demucs: waveform domain, no STFT phase needed at all).

**Tier C — Fine-tune SepFormer on MUSDB18HQ for music-aware separation (high effort):**
SpeechBrain supports fine-tuning via `from_hparams` + standard SpeechBrain training recipes. Fine-tuning `sepformer-whamr` on MUSDB18HQ would produce an 8kHz speech-from-music separator. Not recommended currently (CPU training is slow, and MDX23C already covers this at 44.1 kHz).

---

## 6. Technique Comparison Table

| Technique | Quality | Speed (CPU) | Memory | GPU needed | Open Source | Real-Time | Best For |
|---|---|---|---|---|---|---|---|
| Spectral Subtraction | Low | Fastest | Minimal | No | Yes | Yes | Stationary noise reference |
| Wiener Filter (classical) | Low-Med | Fast | Minimal | No | Yes | Yes | Known signal statistics |
| HPSS (librosa) | Low-Med | Fast | Minimal | No | Yes | Yes | Harmonic/percussive pre-separation |
| NMF | Med | Med | Low | No | Yes | No | Unsupervised, known instrument |
| LMS Adaptive Filter | Med | Fast | Minimal | No | Yes | Yes | Reference signal available |
| Notch Filter | High (narrow) | Fastest | Minimal | No | Yes | Yes | Fixed-frequency hum/tone |
| Spleeter (UNet/TF) | Med | Med | 400 MB | Optional | Yes (MIT) | No | Legacy baseline |
| Open-Unmix (LSTM) | Med | Med | 300 MB | Optional | Yes (MIT) | No | Academic reference |
| MDX-Net Voc_FT (ONNX) | High | Fast | 66 MB | No | Yes | No | CPU-friendly vocals |
| MDX-Net Kim_v2 (ONNX) | High | Fast | 130 MB | No | Yes | No | More robust CPU vocals |
| MDX23C InstVoc HQ (.ckpt) | Very High | Med-Slow | 448 MB | Optional | Yes | No | Best open-source vocals SDR |
| DTTNet (dual-path TFC-TDF) | Very High | Med | ~200 MB | Optional | Yes (MIT) | No | 86% fewer params than BSRNN |
| HTDemucs | High | Med | 600 MB | Recommended | Yes (MIT) | No | Natural-sounding 4-stem |
| HTDemucs_ft | Very High | Slow | 600 MB | Recommended | Yes (MIT) | No | Best Demucs quality |
| htdemucs_6s | High | Slow | 600 MB | Recommended | Yes (MIT) | No | 6-stem (guitar+piano) |
| mdx_q / mdx_extra_q | High | Med | ~300 MB | Optional | Yes (MIT) | No | Quantized Demucs; faster CPU |
| SCNet | Very High | Med | 300 MB | Recommended | Yes | No | Single model, all stems |
| BS-RoFormer | SOTA (vocals) | Slow | 3+ GB | Yes | Yes (MIT) | No | Best vocal separation |
| MelBand-RoFormer | SOTA (vocals) | Slow | 3+ GB | Yes | Yes (MIT) | No | Best perceptual vocal separation |
| HS-TasNet | Low-Med | Fast | ~100 MB | No | Yes (MIT) | **Yes (23 ms)** | Live/RT; needs training |
| RT-STT | Med-High | Fast | TBD | No | Pending | **Yes** | RT + INT8; awaiting release |
| Band-SCNet | High | Fast | TBD | No | Pending | **Yes** | RT causal, lightweight |
| SepFormer (SpeechBrain) | High (speech) | Med | ~150 MB | Optional | Yes (Apache) | No | 22.4 dB SI-SNR; 8kHz only |
| MetricGAN+ (SpeechBrain) | High (enhance) | Fast | ~50 MB | No | Yes (Apache) | No | Post-process vocals; 16kHz |
| SepReformer (NeurIPS 24) | High (speech) | Med | ~120 MB | Optional | Yes (MIT) | No | More efficient than SepFormer |
| Banquet (CASS) | High (cinematic) | Slow | TBD | Yes | Partial | No | 4-stem: dialogue+singing+music+SFX |
| SAM Audio (Meta) | SOTA (all) | TBD | TBD | Yes | Yes | No | Universal; text-prompted |

---

## 7. Applicability to Music Assassin

### Currently implemented (v0.4.3)
- HTDemucs, HTDemucs_ft, MDX_extra (Demucs)
- Spleeter (4-stem)
- UVR-MDX-NET-Voc_FT + Kim_Vocal_2 (MDX-Net ONNX via audio-separator)
- Pre-filtering: noise gate, high-pass
- Post-filtering: de-esser, harmonic rejection, formant enhancement, global normalisation
- Chunked processing (60 s chunks, 2 s crossfade) for low memory

### Added in v0.4.2-RT testing branch
- MDX23C-8KFFT-InstVoc_HQ.ckpt + HQ2 (via audio-separator)
- All Demucs variants: htdemucs_6s, hdemucs_mmi, mdx_q, mdx_extra_q
- HS-TasNet infrastructure (stub — needs weights)
- RT-STT placeholder
- TTA polarity-flip checkbox (+0.1–0.3 dB, 2× slower)

### Gap analysis

| Gap | Current | Open-Source Alternative |
|---|---|---|
| Best vocal SDR | Kim_Vocal_2 (~8.x SDR) | MDX23C HQ (~10 SDR) — added v0.4.2-RT |
| Per-instrument toggles | Vocals only | 4-stem output → per-stem mute sliders |
| Suppression strength | Binary (on/off) | Strength slider: `output = vocals + (1-α)×instrumental` |
| Music detection gating | None (runs on all segments) | PANNs pre-screen (`panns-inference` package) |
| Harmonic pre-filtering | High-pass + noise gate | `librosa.effects.hpss()` pre-pass |
| Reference-guided removal | Not available | NMF with user-provided music clip |
| Video input/output | Audio only | FFmpeg mux/demux wrapper |
| Echo/reverb post-processing | Not available | `pedalboard` reverb suppression or DeepFilterNet |
| Model ensembling | Single model | Average HTDemucs_ft + MDX23C outputs |
| two-stems speedup | Full 4-stem run | `demucs.api.Separator(two_stems='vocals')` — ~2× faster |
| ZFTurbo bigshifts TTA | Per-file crossfade | Import `bigshifts_wrapper` from ZFTurbo inference.py |
| Real-time filtering | Not available | HS-TasNet (needs weights) / Band-SCNet (pending) |
| Post-separation vocal cleanup | None | MetricGAN+ (SpeechBrain) as final pass — removes residual music bleed |
| Singing voice vs dialogue | Not distinguished | Banquet model — 4-stem with singing voice stem |
| Text-prompted separation | Not available | SAM Audio (pending audio-separator integration) |

---

## 8. Priority Upgrade Roadmap

### Tier 1 — Low effort, high impact (drop-in additions)

1. **MDX23C weights** via `audio-separator` — same integration path as current MDX-Net models. `~10 SDR` vs current `~8.x`. Just add a new entry to `model_choice`.

2. **HPSS pre-processing** — `librosa.effects.hpss(y)` before sending audio to the separator. Two lines of code. Helps on melodic music beds.

3. **Suppression strength slider** — scale the inverted instrumental stem before mixing back with vocals. `output = vocals + (1 - strength) * instrumental`. Lets users dial in partial music removal.

4. **Per-stem output toggles** — Demucs and Spleeter already output 4 stems. Expose checkboxes (drums / bass / other / vocals) to control which stems get subtracted. The stems are already computed.

### Tier 2 — Medium effort, significant impact

5. **MelBand-RoFormer or BS-RoFormer** as selectable model — available via `audio-separator` with ZFTurbo HuggingFace weights. Add to separator selector as "RoFormer (SOTA)." Largest quality jump possible.

6. **Video input/output** — FFmpeg wrapper: `ffmpeg_input(video).audio` → process → remux with `ffmpeg.concat`. Covers the main real-world use case (removing music from vlogs, interviews, documentary footage).

7. **PANNs music detection gate** — `panns-inference` Python package. Run a lightweight music presence classifier per chunk; skip the heavy model on speech-only chunks. Saves significant compute on long recordings.

8. **Model ensembling** — run two models (e.g., MDX23C + HTDemucs_ft), average their stem outputs. Straightforward numpy average. No retraining; ~0.5–1 dB SDR improvement.

### Tier 3 — Higher effort, future-looking

9. **NMF reference-guided mode** — accept an optional music reference clip from the user; use supervised NMF to subtract it. Useful for known background tracks (restaurant/venue music, YouTube-licensed jingles).

10. **SAM Audio integration** — monitor `audio-separator` for SAM Audio support; when available, add text-prompt separation mode. This is the long-term replacement for the model selector.

11. **Combined voice isolation pipeline** — inspired by Resolve's Voice Isolation: chain HPSS → BS-RoFormer (vocals) → de-reverb → de-noise into a single "Clean Dialogue" preset that hides all the complexity behind one button.

---

## References & Links

- [Meta Demucs](https://github.com/facebookresearch/demucs)
- [UVR — Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-separator (nomadkaraoke)](https://github.com/nomadkaraoke/python-audio-separator)
- [BS-RoFormer — lucidrains](https://github.com/lucidrains/BS-RoFormer)
- [BS-RoFormer paper — arxiv 2309.02612](https://arxiv.org/abs/2309.02612)
- [bs-roformer-infer — openmirlab](https://github.com/openmirlab/bs-roformer-infer)
- [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
- [ZFTurbo pretrained models index](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md)
- [MelBand-RoFormer paper — arxiv 2310.01809](https://arxiv.org/pdf/2310.01809)
- [Mel-RoFormer + melody — arxiv 2409.04702](https://arxiv.org/pdf/2409.04702)
- [SCNet paper](https://www.researchgate.net/publication/379816658_SCNet_Sparse_Compression_Network_for_Music_Source_Separation)
- [SAM Audio — Meta AI](https://ai.meta.com/research/publications/sam-audio-segment-anything-in-audio/)
- [SAM Audio — arxiv 2512.18099](https://arxiv.org/abs/2512.18099)
- [Open-Unmix (UMX)](https://github.com/sigsep/open-unmix-pytorch)
- [Spleeter (Deezer)](https://github.com/deezer/spleeter)
- [Generative Source Separation](https://github.com/gzhu06/GenerativeSourceSeparation)
- [HPSS — librosa](https://librosa.org/doc/main/auto_examples/plot_hprss.html)
- [HPSS — AudioLabs Erlangen](https://audiolabs-erlangen.de/resources/MIR/FMP/C8/C8S1_HPS.html)
- [LALAL.AI Phoenix blog](https://www.lalal.ai/blog/phoenix-neural-network-vocal-separation/)
- [remove.music](https://remove.music/)
- [AudioCleaner.ai](https://audiocleaner.ai/)
- [Ensemble approach paper — arxiv 2410.20773](https://arxiv.org/html/2410.20773v1)
- [Cinematic separation paper — arxiv 2602.21476](https://arxiv.org/pdf/2602.21476)
- [Open-Source Separation Projects map](https://source-separation.github.io/tutorial/intro/open_src_projects.html)
- [Conditioned UNet — arxiv 2512.15532](https://arxiv.org/pdf/2512.15532)
