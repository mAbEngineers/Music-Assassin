# Music Assassin — Comprehensive Method Comparison

Single master reference for **every model, technique, and architecture tried or
researched** across this project's music-removal/real-time work, in one
consistent column set. Supersedes the fragmented round-by-round tables in
`docs/REALTIME_FILTER_RESEARCH.md` (§9, rounds 3–5) and
`docs/MODEL_BENCHMARKS_v0.4.4.md` as the single point of comparison — those
docs remain the narrative/methodology record; this one is the lookup table.

All measurements taken on the developer's machine: 4-core CPU, 11 GB RAM
(typically 4-6 GB free under normal desktop load), no GPU. Test content: a
Japanese anime episode, three clips — 00:10 (OP theme, singing + heavy music),
07:00 (dialogue + light SFX, near-silent music), 16:00 (action/dramatic, new
content type added in round 5). "Time / 1 min audio" and "RAM" are measured
unless marked *(est.)*; where a number includes one-time model load, that's
noted — a persistent process amortizes load cost across many minutes, so
this is a worst-case figure for a single short clip.

**The one finding that reframes this whole table:** every enhancer here (rows
13–20) removes noise but **not music** — they attenuate noise −29 to −63 dB
while passing prominent music at only −0.4 to −1.7 dB. Only rows 1–12
(separators / subtraction models) can actually remove music. See
`README.md` § Known Limitation.

---

## 1. Separators (full source-separation models)

| Model | Size | Time / 1 min audio | RAM / 1 min audio | Pre-processing | Post-processing | CPU-only? | RT-suited? | SFX/noise handling | Best-use pattern | Quality (by-ear) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **DEMUCS htdemucs** (INT8) | ~80 MB quantized | **10.5 s (RTF 0.177)** ⚠️ see note | **~1.24 GB** | mono→stereo duplicate | none | Yes | ⚠️ Partial — offline chunked, no streaming API, but comfortably faster than real time | Preserves SFX per earlier-session ear test ("didn't cut background SFX") | vocal-extraction (native 4-stem `vocals` output) | Rated good historically. A −60.2 dB music-suppression / −0.02 dB vocal-retention pass was measured, but against demucs-derived ground truth — circular, do not quote (see `ROADMAP.md` §3) | **CORRECTED 2026-09-08 — the original figures were an app bug, not a model property.** `separate_with_demucs` passed `segment=CHUNK_SEC` (60) to `apply_model`, forcing a 60 s transformer context in place of htdemucs's native 7.8 s `training_length`; attention cost is quadratic in context length. Same-machine A/B on one 60 s chunk: **14,929 MB then a crash** with the bug, **1,237 MB and success** without it. The "12-day-stale ~1.2 GB estimate" this table dismissed as 5-6× too low was correct all along. ⚠️ Speed measured on a 32-core box with torch limited to 4 threads — re-measure on the 4-core reference machine before comparing this row's timing against the others. |
| DEMUCS htdemucs_ft *(est.)* | ~80 MB ×4 quantized | *(est.)* ~4× htdemucs | *(est.)* similar peak per sub-model (sequential), ~4× time | same as htdemucs | none | Yes | *(est.)* ⚠️ ~RTF 0.7, borderline | *(est.)* similar to htdemucs | vocal-extraction | Not tested | **The reason for not testing this is void as of 2026-09-08** — it was "too risky after the htdemucs OOM finding", and that finding was the `segment=60` bug (row 1). It is a bag of 4 HTDemucs sub-models run sequentially, so it hit the identical crash. Now cheap to measure; worth doing. |
| DEMUCS mdx_extra *(est.)* | similar | *(est.)* ~4× htdemucs | *(est.)* same as htdemucs_ft | same | none | Yes, impractical | ❌ No | Unverified | vocal-extraction | Labeled "Best" in app UI, unverified | Not live-tested, same reasoning as htdemucs_ft. |
| **Voc_FT · fast engine** (`fast_separation.py`, OpenVINO) | 66 MB ONNX (same weights as the audio-separator row below) | **3.2–4.4 s** (16 threads, BF16, RTF 0.05–0.07) | **2.9 GB** | one global peak-based scale (not per-chunk — see Notes on the row below) | none | Yes | ✅ Yes — well under real time | Vocals-only, same as the audio-separator row — doesn't keep SFX itself; see the "Keep sound effects" rows in §5 | vocal-extraction | **48 dB SI-SDR vs audio-separator's output** — same weights, different runtime; effectively identical | **Shipped 2026-10-23, now the app default.** Same ONNX weights as `UVR-MDX-NET-Voc_FT` below, run through OpenVINO (BF16 where the CPU supports it, FP32 otherwise) instead of plain onnxruntime — 7–12× faster at 16 threads. One deliberate behaviour change: a single whole-file peak scale, where the audio-separator wrapper scaled each 60 s chunk by its own peak and made quiet scenes quieter. Measured on a 16-core Zen 5; re-measure on the 4-core reference before comparing this row's speed to others (same caveat as the htdemucs row above). Source: `docs/OFFLINE_SPEED_RESEARCH.md`. |
| **MRX** (`kj7/mrx`, int8 ONNX) | 31 MB | **1.6–1.8 s** (4/16 threads, RTF 0.03) | **2.4 GB** | −27 LUFS loudness normalization | music over-subtraction (default on, β=1) | Yes | ✅ Yes | **Trained for this directly** — outputs dialogue/music/effects in one pass, DnR-native | dialogue+effects extraction, or as the effects half of "Keep sound effects" | Leaks music at −16.4 dB raw, −21.3 dB after over-subtraction (its default); effects kept at −1.3 dB SI-SDR | **Shipped 2026-10-23** as the "MRX dialogue (Fastest · lower quality)" separator choice and as "Keep sound effects: Fast". MIT-licensed. Full leak/keep numbers: `docs/LIVE_SFX_HANDOVER.md` §2.2; this is the R2 candidate from `ROADMAP.md`, shipped ahead of the DnR scoring that section called for. |
| **CDX23** (ZFTurbo `v.1.0.0`, torch HTDemucs) | ~80 MB | **5.9 s** (16 threads, RTF 0.10) | **1.4 GB** | −27 LUFS loudness normalization | none (over-subtraction defaults off — raw leaks less than MRX already) | Yes | ⚠️ Partial — well under real time per-call, but non-causal (needs the whole file, unlike MRX's streaming-friendly shape) | Outputs dialogue/music/effects; **its "speech" stem excludes singing**, so it is not a substitute for Voc_FT on sung content | the effects half of "Keep sound effects: Quality" — not used for dialogue in this app | Leaks music at −24.9 dB (about 8.5 dB less than MRX) and keeps effects at **+3.6 dB SI-SDR** — the best of any candidate in this table on both axes | **Shipped 2026-10-23** as "Keep sound effects: Quality". Licence of the released weights (vs the DnR training data's own terms) is unchecked — flagged, not cleared, in `docs/LIVE_SFX_HANDOVER.md` §4.3. |
| **MDX-Net UVR-MDX-NET-Voc_FT** (audio-separator/onnxruntime) | 66 MB ONNX | **134.7 s** (RTF 0.45×) | **~3.3 GB** | mono/stereo handling, 60s+2s-overlap chunking | none | Yes | ⚠️ Partial — chunked ~200-400ms windows, not a streaming API | **Removes SFX along with music** — it's a vocals-only specialist, strips everything non-vocal by design | vocal-extraction | Good, current app default | Only separator that reliably completes on this machine's typical free RAM. |
| MDX-Net Kim_Vocal_2 (audio-separator) | 130 MB ONNX | Not tested | Not tested | same as Voc_FT | none | Yes | Same caveat as Voc_FT | Not evaluated | vocal-extraction | Not evaluated | Untested speed/RAM trade-off from the larger model. |
| Spleeter (TF, 4-stems) | ~150 MB | Not tested live | TF import alone: 500 MB–1.5 GB before any separation | none beyond standard | none | Yes | ❌ No | Not evaluated | vocal-extraction (of 4 stems) | Historically weaker for anime vocals (general-purpose model) | TF import is now *lazy* (v0.4.4 fix) — only paid if Spleeter is actually selected. |
| **sherpa-onnx UVR Voc_FT** | 64 MB ONNX | **84.4 s** @ 4 threads (RTF 1.41); 143.4 s @ 1 thread | ~4.3 GB | same weights as audio-separator's Voc_FT | none | Yes | ❌ No — still slower than real-time | Same as audio-separator Voc_FT (identical weights) | vocal-extraction | Identical output to audio-separator Voc_FT (same weights) | 1.6× faster than audio-separator's wrapper for the *same* weights — pure runtime efficiency gain. |
| **sherpa-onnx Spleeter 2-stem int8** | 2× 26 MB | **4.0 s (RTF 0.067 — 15× real-time)** | **~600 MB** | **requires stereo input `(2,N)` — crashes on mono, must duplicate channel** | none | Yes | ✅ **Yes** — first genuinely RT-capable separator found | Not fully assessed — architecturally weaker 2-stem model, "accompaniment" stem mixes music+SFX | vocal-extraction (2-stem: vocals/accompaniment) | Not fully evaluated by ear on singing content vs Voc_FT (files saved, comparison pending) | Threads don't matter (1 thread ≈ 4 threads — I/O bound, not compute bound). |
| **sherpa-onnx UVR Inst_HQ_3** (subtraction) | 63 MB fp32 | ~82-120 s across clips (RTF ~1.9-2) | ~4.3 GB | none beyond stereo duplicate | **subtract from mix**: `vocals+SFX = mix − predicted_music` | Yes | ❌ No | **Preserves SFX by design** — predicts *only* music, so anything else (dialogue, SFX) survives subtraction | **subtraction** — predict music stem, subtract from mix | Correctly near-silent (rms 0.008) on dialogue clip with little music; plausible on action clip (music_rms 0.072) | This is the direct answer to "detect music and subtract it" — validated working. |
| **sherpa-onnx UVR_MDXNET_KARA_2** (subtraction) | 51 MB | **50.2 s (1.6× faster than Inst_HQ_3)** | **~2.8 GB (34% less than Inst_HQ_3)** | same as Inst_HQ_3 | same subtraction pattern | Yes | ❌ No, but closest of the subtraction models to practical | Same principle as Inst_HQ_3 — preserves SFX | **subtraction** — same as Inst_HQ_3, cheaper | Near-identical output energy to Inst_HQ_3 (music_rms 0.069 vs 0.072, sub_rms 0.107 vs 0.107) on the same clip — by-ear check still pending | **Current best subtraction-architecture candidate.** Karaoke-instrumental model repurposed for this use. |
| sherpa-onnx UVR-MDX-NET_Crowd_HQ_1 (subtraction, failed) | 57 MB | 64.6 s | ~3.1 GB | same as Inst_HQ_3 | same pattern | Yes | ❌ No | **Over-subtracts** — likely misclassifies dialogue as crowd noise | **Not recommended** | music_rms abnormally high (0.124), sub_rms abnormally low (0.023) vs the other two — signature of over-subtraction | Trained for crowd/audience noise removal, not anime background music. Negative result, documented to avoid re-testing. |
| Self-quantized int8 Inst_HQ_3 | 17.9 MB (from 63.7 MB fp32) | **399.8 s — 3.7× SLOWER than fp32** | ~4.1 GB (no real reduction) | same as Inst_HQ_3 | same | Yes | ❌ No, worse than fp32 | n/a | **Don't** | 99.28% correlation but 11.97% RMS difference vs fp32 — real quality loss | **Negative result.** Generic dynamic quantization only helps Linear/MatMul-bound models (like htdemucs's targeted INT8); this model is Conv/STFT-bound, so quantizing backfires: slower, no RAM win, worse quality. |

---

## 2. Enhancers (post-separation cleanup — none of these remove music)

| Model | Size | Time / 1 min audio | RAM delta | Pre-processing | Post-processing | CPU-only? | RT-suited? | Removes music? | Best-use pattern | Quality (by-ear) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **DeepFilterNet3** (`atten_lim_db=20`) | moderate (torch checkpoint) | **16.2 s** (incl. 1-5s one-time load) | **~585 MB** | resample → 48 kHz | resample back | Yes | ⚠️ Borderline — causal/frame-based architecture by design, batch-mode measurement here | **No** — cleans noise (−29 to −63 dB), passes music (−0.4 to −1.7 dB) | enhancer only, post-separation | **Best measured** — user-confirmed reduced guitar/piano bleed to "a soft tap on higher notes"; also required upstream of VAD (builds stable noise profile) | **Current default enhancer.** Official Rust LADSPA plugin exists for true real-time PipeWire use (not tested here). |
| DPDFNet baseline (sherpa-onnx) | 8.5 MB | 8.1 s | ~18 MB | resample 44.1k→16k | resample 16k→44.1k | Yes | Yes, tiny footprint | No (enhancer) | enhancer only | Not evaluated by ear | 16 kHz round-trip discards everything above 8 kHz — a real cost for this project's high-pitched-voice problem. |
| **DPDFNet2_48khz_hr** | 10.3 MB | **16.9 s — same speed as DeepFilterNet** | **~40 MB — 1/14th DeepFilterNet's RAM** | none needed (native 48 kHz) | none | Yes | Yes | No (enhancer) | enhancer only | By-ear comparison pending; full bandwidth preserved unlike baseline dpdfnet | **Sleeper hit** — matches DF speed at a fraction of the RAM, no bandwidth loss. Candidate to promote to default pending listening test. |
| dpdfnet8 (16 kHz "best quality") | 14.1 MB | 29.9 s — ~2× slower than DF | ~64 MB | resample 44.1k→16k | resample back | Yes | Yes | No (enhancer) | **Not recommended** | Dominated by dpdfnet2_48khz_hr on every axis | Bandwidth-capped at 8 kHz *and* slower — no reason to use over dpdfnet2_48khz_hr. |
| **GTCRN** | **524 KB — tiniest tested** | **3.83 s — fastest enhancer** | **~0.6 MB — smallest footprint** | resample 44.1k→16k | resample back | Yes | Yes, fastest+lightest | No (enhancer) | enhancer only, tightest RAM/CPU budgets | User directly reported: "still has some music like notes of guitar/piano" — more residue than DeepFilterNet | Confirmed quality/speed tradeoff — cheapest option, worst residual cleanup of the enhancers tested. |
| **SpeechDenoiser** (custom ONNX, 48 kHz native) | 16 MB | 15.0 s | ~7.6 MB | none (native 48 kHz) | none | Yes | **Yes — genuinely frame-based** (480-sample hops, explicit state tensor carried between frames) | No (enhancer) | enhancer only | Not evaluated by ear | Closest thing to a real streaming API already implemented in this app besides DTLN. |
| MetricGAN+ (SpeechBrain, 16 kHz) | moderate (~500 MB SpeechBrain+torch load) | 9.8 s — faster than DF | ~30 MB after one-time load | resample 44.1k→16k | resample back | Yes | Not specifically assessed (non-causal typically) | No (enhancer) | enhancer only | 16 kHz ceiling discards >8 kHz — a real cost here | This is the "SpeechBrain AI filter" method — same category as DeepFilterNet, and strictly worse for this content due to bandwidth cap. Was the v0.4.2 app's original enhancer. |
| **DTLN** (breizhn, streaming, 16 kHz) | **3.8 MB total (2 models)** | **4.6 s — 13× real-time, cheapest enhancer measured** | **~1 MB — smallest RAM of all enhancers** | resample 44.1k→16k | resample back | Yes | **Yes — genuinely stateful streaming** (32 ms frames, hidden state carried between calls) | No (enhancer) | enhancer only | Not evaluated by ear (energy-check only) | **Strongest RT building block among enhancers** — cheapest + most genuinely streaming. Needed custom two-model state wiring (no official streaming wrapper in this stack). MIT licensed. |

---

## 3. VAD / Gating

| Model | Size | Time / 1 min audio | RAM delta | Pre-processing | Post-processing | CPU-only? | RT-suited? | Removes music? | Best-use pattern | Quality | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Silero VAD + energy hybrid** (this app's tuned v10 logic) | Silero ~1 MB | 4.15 s | ~7 MB | requires enhancer output first (see Notes) | none | Yes | ⚠️ Partial — Silero itself streams natively (32ms frames), but this app's energy-anchoring math (median-relative threshold, ±1.5s expansion) is whole-clip/non-causal | No — gates non-vocal *gaps*, doesn't remove music playing simultaneously with vocals | gate silence/instrumental-only sections between speech | Extensively tuned this session — v10 retains high-pitched voices while rejecting SFX/music false-positives via a mean-confidence filter | **Must run AFTER the enhancer, not before** — a real pipeline-order bug was found and fixed this session (DeepFilterNet needs a continuous signal to build its noise model; gating first fragments it). |

---

## 4. DSP Pre-/Post-processing (no model, pure signal processing)

| Technique | Time / 1 min audio | RAM | Pre-req | Post-req | CPU-only? | RT-suited? | Removes music? | Best-use pattern | Quality | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| DSP pipeline (`advanced_postprocess_chunked`, balanced) | **0.20 s — negligible** | ~11 MB | none | none | Yes | Yes, negligible cost | No — mild EQ/filter polish only | always-on pre/post polish, not a music remover | Mild general cleanup | Safe to always leave on. |
| **Stereo mid/side center extraction** *(prototype)* | 4.4 s (unoptimized scipy STFT) | not separately measured (light) | **requires stereo source** — lost once downmixed to mono (current app downmixes immediately) | none | Yes | Yes — causal, cheap | **Suppresses wide-panned content** (typically music/ambience), preserves center-panned (dialogue) | **pre-filter before the separator** | Voice/music ratio improved 9.2×→16.8× (gain exponent `g⁴`) with ~0.01 dB measured vocal loss (within noise floor) — ~1.8 dB free suppression | **Currently unused** by the app — a real, free win being left on the table. Caveat: aggressive exponents can cause "musical noise" artifacts an energy metric won't show — needs a by-ear check (`bench_op10s_midside_center*.wav` vs `bench_op10s_mono_ref.wav`). |
| Spectral transient gate *(abandoned)* | Not benchmarked | — | — | — | Yes | n/a | **Ineffective** | **Not recommended** | Guitar/piano residue in this content is sustained, not transient — the gate's background estimator adapts to it and never fires | Still present as a UI toggle; candidate for removal. |

---

## 5. SFX Recovery / Residual Add-back (parallel to a vocal-extraction separator)

Applies to `residual = mix − vocals_stem` — everything a vocal-extraction
separator removed (music **and** SFX). Only relevant when using a
vocal-extraction model (Voc_FT, Spleeter) rather than the subtraction
architecture (§1 rows 9–10), which preserves SFX by construction and doesn't
need this step.

| Method | Time | RAM | Pre-req | Post-req | CPU-only? | RT-suited? | Music leakage (OP control) | Residual kept (dialogue) | Best-use pattern | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| DSP mask (HPSS transient × spectral-flatness gate) | fast, sub-second on test clips | light (pure DSP) | needs the residual (mix − vocals_stem) | add back **after** VAD gate, so gated SFX isn't lost twice | Yes | Yes (cheap), but quality is the ceiling, not speed | 8.3% (v3-very-strict, the best DSP variant) | 5.3% | Shows the ceiling of pure-DSP discrimination | Mixture-consistent (only adds back untouched original signal — no processing artifacts possible, only leakage risk). Superseded by the YAMNet approach below. |
| **YAMNet semantic classifier** | ~1.5s per 20s clip inference (+ TF import cost, amortized in a persistent process) | TF import: 500 MB–1.5 GB (model itself is 15 MB, tiny) | needs the residual; requires TF ≥2.16/Keras 3 compat patch (see `REALTIME_FILTER_RESEARCH.md`) | add back after VAD gate | Yes | Classifier inference itself is fast/frame-based (0.48s hop); heavy TF runtime dependency is the real cost | **0.5% — order of magnitude tighter than DSP** | **94.1% — far more than DSP's 5.3%** | **Recommended** — pairs with any vocal-extraction separator to recover SFX without the subtraction architecture's speed/RAM cost | Absolute confidence scores on residual audio are low (0.001–0.09 range) since a residual is out-of-distribution from YAMNet's natural-audio training data — "top class" is still meaningful, but by-ear verification is recommended before fully trusting the leakage number. |
| **Music over-subtraction on a second separator's own stem** (`fast_separation.clean_effects`) — **shipped 2026-10-23** | MRX ≈ 1.6–1.8s, CDX23 ≈ 5.9s per 10 min — the cost is a second separator call, not the subtraction itself | adds the second model's own footprint (MRX 2.4 GB, CDX23 1.4 GB) on top of the dialogue path | runs on an **un-gated** copy of the input — the −40 dB pre-gate erases ambience the effects stem needs | add back after VAD gate, level-matched to the dialogue path's post-process gain, through a block limiter | Yes | Yes, at MRX's cost; CDX23 is non-causal, needs the whole file | **MRX −21.3 dB, CDX23 −24.9 dB** — best of any row in this table; the model is trained to output an effects stem rather than classifying a vocal-extractor's leftovers | **MRX −1.3 dB, CDX23 +3.6 dB SI-SDR** | **In production now**, as "Keep sound effects: Fast/Quality" | Different mechanism from the two rows above: those classify the *residual* of a vocals-only separator; this runs a *second, DnR-native* separator in parallel and subtracts its own music estimate from its own effects estimate. A Wiener re-filter using the model's own three stems was also tried and moved leakage by under 0.5 dB — not worth the cost. Full method: `docs/LIVE_SFX_HANDOVER.md` §2.2–§2.3. |

---

## 6. Acceleration

| Method | Speed impact | Applies to | CPU-only? | Notes |
|---|---|---|---|---|
| **OpenVINO execution provider** (`onnxruntime-openvino`) | **~16% faster** (2909ms vs 3467ms, raw UVR model inference) | Only the `audio-separator` path (plain onnxruntime) | Yes | sherpa-onnx bundles its own statically-built onnxruntime and won't pick up an externally-installed OpenVINO provider without a from-source rebuild. Partial op support causes fallback to `CPUExecutionProvider` for unsupported ops, capping the win. Tested in an isolated throwaway venv to avoid destabilizing the working dependency set. **Superseded 2026-10-22:** this plugin runs FP32 on CPU. Native OpenVINO with BF16 compute (`fast_separation.py`) is about 5× faster than audio-separator on a 16-core Zen 5 CPU, and is now the app's default (see `OFFLINE_SPEED_RESEARCH.md`). |
| Self-quantization (int8, dynamic) | **Negative** — see §1, "Self-quantized int8 Inst_HQ_3" | Conv/STFT-bound models | Yes | Only pays off for Linear/MatMul-bound models (htdemucs's targeted quantization). Don't apply blindly to separator models. |

---

## 7. Researched, Not Live-Tested (no public weights / no code release / needs a build step)

| Method | Status | Claimed performance | Best-use pattern | Notes |
|---|---|---|---|---|
| **HS-TasNet** | No public trained weights as of this session | 23ms frame latency, 4ms inference on 4-core CPU, 4.65–5.55 SDR on MusDB (paper claims) | vocal-extraction (or joint stems) at true real-time, if trained | MIT-licensed training code exists ([lucidrains/HS-TasNet](https://github.com/lucidrains/HS-TasNet)) — training on MusDB via a free Colab/Kaggle GPU is the realistic path to weights. **This is the ideal target architecture**, blocked purely on lack of trained weights. |
| RT-STT (successor to HS-TasNet) | No code/weights released (paper: Nov 2025) | Claims fewer params + faster inference than HS-TasNet | — | Nothing actionable yet; watch for a code release. |
| DeepFilterNet official LADSPA plugin (Rust) | Not tested this session | Same DeepFilterNet3 quality, true real-time | System-wide real-time enhancer in a PipeWire `filter-chain` | Directly compatible with routing already validated in `docs/LIVE_APP_PLAN.md`. Zero-Python path for the enhancer already validated as best quality here. |
| RNNoise + werman LADSPA build | Not tested | Sub-10ms, nearly free | Real-time denoise, quality below DeepFilterNet | BSD licensed. |

---

## 8. Known Candidates Not Yet Benchmarked

Unlike §7 (blocked on missing weights/code), everything below is **available
right now** — either already partially wired into the codebase, or installable
through a library this project already depends on — but has no row above
because it's never actually been run on this machine. Full background for each
is in `docs/SEPARATION_TECHNIQUES_RESEARCH.md`; this is just the tracking list
so they don't get lost.

| Candidate | Availability | Why it's not measured yet | Notes |
|---|---|---|---|
| **MDX23C-8KFFT-InstVoc_HQ(.ckpt)** | Already in code (`MusicAssassinGUI_v0.4.2_RT.py`), auto-downloads via `audio-separator` | Added for the RT testing branch but never run through a benchmark pass | Requires torch (not ONNX) — expect 2-4× slower than MDX-Net Voc_FT on CPU per the techniques doc; ~448 MB download |
| **`2_HP-UVR.pth`** (VR architecture) | Loadable via `audio-separator` (already a dependency), not yet downloaded/used | Never researched or tested — an older architecture that got skipped over in favor of MDX-Net/Demucs | **Fastest separator found in any source so far.** Per the `audio-separator` maintainer's own comparison (same 3-min song, Apple M3 Max CPU): 19s for `2_HP-UVR.pth` vs 36s for `UVR-MDX-NET-Inst_HQ_4.onnx`, 109s for BS-RoFormer, 157s for MDX23C — roughly 2–8× faster than everything else in §1 at "good enough" quality. Numbers aren't from this machine, but the *ranking* is a strong signal. Highest-priority thing to benchmark on this list if the goal is speed. |
| **`UVR-MDX-NET-Inst_HQ_4.onnx`** | Loadable via `audio-separator`, already flagged as a candidate in the techniques doc but never detailed | Never benchmarked | Maintainer's own sweet-spot pick for instrumental/vocals when Roformer-class runtime isn't affordable — ~2× Voc_FT's cost for a real quality step up, well short of Roformer's 5-8×. |
| **`kuielab_a_vocals.onnx`** | Loadable via `audio-separator` | Never researched | Smaller/older MDX-family vocals-focused model, maintainer-recommended alongside Kim_Vocal_2; likely close to Voc_FT's speed class but unverified. |
| **`mel_band_roformer_bleed_suppressor_v1.ckpt`** | Loadable via `audio-separator` | Never researched | Purpose-built to suppress residual bleed-through in an *already-separated* vocals stem — directly targets this project's core problem (README "Known Limitation"). Roformer-class model, so not fast as a full separator, but may be cheap as a short post-pass on a small vocals-only file rather than the full mix. Worth testing as a post-processing stage, not a replacement separator. |
| **UVR De-Echo/De-Reverb models** — refined list from the maintainer's own recommendations: `UVR-DeEcho-DeReverb.pth`, `UVR-De-Echo-Normal.pth`, `UVR-De-Echo-Aggressive.pth`, `Reverb_HQ_By_FoxJoy.onnx`, `deverb_bs_roformer_8_384dim_10depth.ckpt`, `dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt` (highest published SDR of the group, 19.17) | Available today via `audio-separator` (already a dependency) — confirmed present in its current model list | Not researched in either doc until now; no code path calls them | Addresses a real, currently-untouched gap: dialogue reverb. Candidate post-processing stage after vocal extraction. |
| **Apollo** (JusperLee, band-sequence audio restoration) | Public weights/code, HuggingFace + GitHub | Not previously researched | Repairs artifacts from lossy separation/compression. **License: CC-BY-SA-4.0 — not OSI-approved.** Fine for use in this research repo, but fails the open-source gate in `docs/LIVE_APP_PLAN.md` §3 for promotion to the live app. |
| **BS-RoFormer / MelBand-RoFormer** | Public weights via `audio-separator` + ZFTurbo HuggingFace repo | Researched extensively (§2 of the techniques doc) but never live-tested | **OOM risk, but the stated precedent is void** — 3+ GB weights and GPU-recommended still argue for caution, but the supporting claim ("htdemucs already OOM'd at 5-6 GB") was an app bug, corrected in §1 row 1 on 2026-09-08. Still use the systemd-run cgroup memory cap before attempting; just don't treat the htdemucs precedent as evidence about this model. |
| **MERL `cocktail-fork-separation` (MRX model)** — **promoted 2026-10-23, see §1 and §5** | 4 pretrained checkpoints (git LFS) at [merlresearch/cocktail-fork-separation](https://github.com/merlresearch/cocktail-fork-separation), MIT licensed | Shipped as an int8 ONNX model, not the git-LFS/PyTorch-Lightning checkpoints this row describes — the feasibility questions below were bypassed, not answered | **Best architectural fit found for "keep dialogue + SFX, remove music"** — unlike every model above, it's trained directly on the **DnR (Divide and Remaster) dataset**: Dialogue / Music / Effects, not a repurposed 4-stem music-separation model. MIT license (no non-commercial restriction, unlike Bandit/Banquet below). CPU/real-time feasibility undocumented — repo's examples assume GPU (PyTorch Lightning); treat as offline/quality-first until benchmarked. Highest-priority item on this whole list to actually try. |
| **Bandit / Banquet (CASS, DnR-trained)** | Public code at [kwatcharasupat/source-separation-landing](https://github.com/kwatcharasupat/source-separation-landing) | Researched in the techniques doc (§2) but never live-tested | Same DnR lineage as MRX above (dialogue/singing-vocals/music/SFX), reportedly the best-quality fit for cinematic content. **License: non-commercial** (MUSDB18-HQ training data restriction) — usable in this research repo, blocked from the live app per the open-source gate. Attention-based, no documented streaming/causal path — not real-time. |

Source for the `2_HP-UVR`/`Inst_HQ_4`/Roformer timing comparison and the karaoke/vocals/de-reverb model recommendations above: [`audio-separator` maintainer's model-selection guide](https://github.com/nomadkaraoke/python-audio-separator/discussions/133) (GitHub Discussion #133).

---

## Bottom line

> **Update 2026-10-24 — the question below was answered by shipping, not by
> the A/B this section called for.** The app's default is now **Voc_FT · fast
> engine** (§1), not htdemucs and not the original MDX-Net wrapper: OpenVINO
> BF16 gives Voc_FT's established quality at RTF 0.05–0.07, faster than
> htdemucs's RTF 0.177 and in comparable memory (2.9 GB vs 1.24 GB). **htdemucs
> vs the fast engine, by ear, is still an open listening test** — the
> comparison that ran was fast-engine-vs-its-own-old-self (48 dB SI-SDR,
> confirming the optimization didn't change the output), not fast-engine-vs-
> htdemucs. `ROADMAP.md` R7.2 tracks this.
>
> **SFX preservation (next paragraph) is answered, concretely:** path 2's
> *mechanism* — a second model's own stem, over-subtracted and added back —
> shipped (§5), but using MRX/CDX23 rather than a vocal-extractor-plus-YAMNet.
> It leaks less (−21 to −25 dB vs path 2's 0.5%/94.1% figures, which aren't
> directly comparable — different leak metric, different corpus) and needs no
> TensorFlow. Path 1 (KARA_2 subtraction) and the original YAMNet row remain
> unshipped alternatives, not disproven ones.

**For file processing today:** MDX-Net Voc_FT (either wrapper) → DSP → DeepFilterNet
(or dpdfnet2_48khz_hr, pending ear test) → hybrid VAD gate is the best *validated*
combo (§1, §2, §3) — but **htdemucs is now the candidate to beat.** With the
`segment=60` bug fixed (§1 row 1, corrected 2026-09-08) it runs at ~1.24 GB and
RTF 0.177 against Voc_FT's ~3.3 GB and RTF 0.45: roughly 2.5× faster in a third
of the memory. It has not yet been A/B'd by ear against Voc_FT on this content,
and that comparison is now the highest-value listening test outstanding.

**For SFX preservation specifically**, two independent paths now exist and
should be evaluated against each other by ear:
1. **Subtraction architecture** (§1): KARA_2 predicts music only, subtract from
   mix. Preserves SFX by construction, but costs ~50s/audio-minute and ~2.8GB.
2. **Extraction + classifier add-back** (§5): any fast vocal-extraction
   separator (e.g. sherpa Spleeter int8 at RTF 0.067) + YAMNet residual
   add-back. Much cheaper on the separator side, adds TF's RAM/import cost for
   the classifier.

**For real-time**, no separator clears the bar yet except sherpa Spleeter int8
(RTF 0.067) — and its separation quality on singing content is still unverified
by ear (§1). Among enhancers, DTLN and SpeechDenoiser are the only genuinely
stateful-streaming options (§2). The stereo mid/side pre-filter (§4) is a free,
untapped win sitting upstream of all of this.
