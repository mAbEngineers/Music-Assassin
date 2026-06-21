# Music Removal & Source Separation — Techniques, Tools & Research

Compiled research covering classical DSP, deep learning architectures, open-source projects, commercial tools, and applicability to Music Assassin.

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
   - MDX-Net
   - BS-RoFormer (Band-Split RoPE Transformer)
   - MelBand-RoFormer
   - SCNet (Sparse Compression Network)
   - Conditioned UNet (text/label-guided)
   - Generative / Flow-based Separation
3. [Open-Source Projects & Repos](#3-open-source-projects--repos)
4. [Commercial AI Tools](#4-commercial-ai-tools)
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

**Models available:**
| Model | Description | Notes |
|---|---|---|
| `htdemucs` | 4-stem, balanced quality/speed | Default |
| `htdemucs_ft` | Fine-tuned per stem | Best Demucs quality |
| `mdx_extra` | MDX competition variant | Higher SDR, more VRAM |
| `htdemucs_6s` | 6-stem (adds guitar + piano) | Experimental |

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
| Model | Size | Specialty |
|---|---|---|
| `UVR-MDX-NET-Voc_FT` | ~66 MB | Vocals, fast CPU | ← currently used |
| `Kim_Vocal_2` | ~130 MB | Vocals, more robust | ← currently used |
| `MDX23C` (ZFTurbo) | ~200 MB | All 4 stems, **highest SDR** | not yet used |
| `UVR-MDX-NET-Inst_HQ_4` | ~66 MB | Instrumental | not yet used |

**Strengths:**
- Lightweight (66–200 MB vs 600 MB for mdx_extra).
- Runs fast on CPU via ONNX runtime.
- MDX23C achieves **10.08 SDR** on MUSDB — best open-weights model available.

**Weaknesses:**
- Fixed input length per chunk (less flexible than waveform models).
- ONNX format limits customization.

**Current status in Music Assassin:** You use `UVR-MDX-NET-Voc_FT` and `Kim_Vocal_2`. MDX23C is the obvious next addition.

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

## 3. Open-Source Projects & Repos

| Project | Architecture | Stems | Notable Feature |
|---|---|---|---|
| [Demucs (Meta)](https://github.com/facebookresearch/demucs) | Hybrid waveform+spectrogram UNet + Transformer | 4 (+ 6-stem variant) | Best general-purpose open-weights model |
| [UVR — Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) | MDX-Net, VR-Arch, Demucs wrappers | Vocals + instruments | Largest community model zoo; hundreds of fine-tuned variants |
| [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | Wraps UVR models + Demucs | All | Python API for UVR models — what Music Assassin uses |
| [BS-RoFormer (lucidrains)](https://github.com/lucidrains/BS-RoFormer) | Band-Split RoPE Transformer | 4 | SOTA architecture; clean PyTorch |
| [bs-roformer-infer](https://github.com/openmirlab/bs-roformer-infer) | BS-RoFormer | 4 | Inference-only, auto checkpoint download |
| [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) | BS-RoFormer, MelBand-RoFormer, SCNet, Demucs, MDX23C | 4 | Unified training + best pretrained weights; HuggingFace hosted |
| [Open-Unmix (UMX)](https://github.com/sigsep/open-unmix-pytorch) | BiLSTM | 4 | Academic reference baseline; well-documented |
| [Spleeter (Deezer)](https://github.com/deezer/spleeter) | UNet (TensorFlow) | 2 / 4 / 5 stems | Established; easy API; outclassed by newer models |
| [Generative Source Separation](https://github.com/gzhu06/GenerativeSourceSeparation) | Normalizing Flows | Music stems | Flow-based generative approach |
| [librosa HPSS](https://librosa.org/doc/main/auto_examples/plot_hprss.html) | DSP (median filter) | Harmonic + Percussive | Zero-dependency fast pre-separation |

---

## 4. Commercial AI Tools

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

## 6. Technique Comparison Table

| Technique | Quality | Speed | Memory | Requires GPU | Requires Training Data | Handles Real-Time | Best For |
|---|---|---|---|---|---|---|---|
| Spectral Subtraction | Low | Fastest | Minimal | No | No | Yes | Stationary noise reference |
| Wiener Filter (classical) | Low-Med | Fast | Minimal | No | No | Yes | Known signal statistics |
| HPSS | Low-Med | Fast | Minimal | No | No | Yes | Harmonic/percussive pre-separation |
| NMF | Med | Med | Low | No | Optional (templates) | No | Unsupervised, known instrument |
| LMS Adaptive Filter | Med | Fast | Minimal | No | No | Yes | Reference signal available |
| Notch Filter | High (narrow) | Fastest | Minimal | No | No | Yes | Fixed-frequency hum/tone |
| Spleeter (UNet) | Med | Med | 400 MB | Optional | Yes | No | Legacy baseline |
| Open-Unmix (LSTM) | Med | Med | 300 MB | Optional | Yes | No | Academic reference |
| MDX-Net (Voc_FT) | High | Fast | 66 MB | No (ONNX) | Yes | No | CPU-friendly vocals |
| MDX23C (ZFTurbo) | Very High | Med | 200 MB | Optional | Yes | No | Best CPU-friendly all-stems |
| HTDemucs | High | Med | 600 MB | Recommended | Yes | No | Natural-sounding output |
| HTDemucs_ft | Very High | Slow | 600 MB | Recommended | Yes | No | Best Demucs quality |
| SCNet | Very High | Med | 300 MB | Recommended | Yes | No | Single model, all stems |
| BS-RoFormer | SOTA (vocals) | Slow | 3+ GB | Yes | Yes | No | Best vocal separation |
| MelBand-RoFormer | SOTA (vocals) | Slow | 3+ GB | Yes | Yes | No | Best perceptual vocal separation |
| SAM Audio (Meta) | SOTA (all) | TBD | TBD | Yes | Yes | No | Universal; text-prompted |

---

## 7. Applicability to Music Assassin

### Currently implemented
- HTDemucs, HTDemucs_ft, MDX_extra (Demucs)
- Spleeter (4-stem)
- UVR-MDX-NET-Voc_FT + Kim_Vocal_2 (MDX-Net via audio-separator)
- Pre-filtering: noise gate, high-pass
- Post-filtering: de-esser, harmonic rejection, formant enhancement
- Chunked processing for low memory

### Gap analysis

| Gap | Current | Better Alternative |
|---|---|---|
| Best vocal SDR | Kim_Vocal_2 (~8.x SDR) | MDX23C or MelBand-RoFormer (~10 SDR) |
| Per-instrument toggles | Vocals only | 4-stem output → per-stem mute sliders |
| Suppression strength | Binary (on/off) | Strength slider via inverted instrumental mix-back |
| Music detection gating | None (runs on all segments) | PANNs pre-screen — only process music-present frames |
| Harmonic pre-filtering | High-pass + noise gate | HPSS pre-pass separates harmonic layer first |
| Reference-guided removal | Not available | NMF with user-provided music clip |
| Video input/output | Audio only | FFmpeg mux/demux wrapper |
| Echo/reverb post-processing | Not available | librosa/pedalboard reverb suppression step |
| Model ensembling | Single model | Average BS-RoFormer + SCNet outputs |
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
