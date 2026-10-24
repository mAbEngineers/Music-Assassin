# Music Assassin v0.4.4 — Model & Pipeline Benchmarks

> **For a single master table covering every method tried across this entire
> project** (including later rounds 3–5: KARA_2, YAMNet, DTLN, MetricGAN+,
> dpdfnet variants, subtraction architecture, mid/side pre-filter, OpenVINO),
> see `docs/COMPREHENSIVE_METHOD_COMPARISON.md`. This document covers only the
> initial v0.4.4 benchmark pass (Tables 1–2 below) plus the round 3–5 addenda.
>
> **Speed numbers here predate the fast engine (2026-10).** For current timings (Voc_FT on OpenVINO, about 58 s per 10 min end to end, plus keep-effects costs), see `docs/OFFLINE_SPEED_RESEARCH.md`.

Empirically measured on **2026-07-04**, on the developer's actual machine: 4-core CPU,
11 GB total RAM (typically 4-6 GB available under normal desktop load — VSCode +
browser). All separator/enhancer code paths run on **CPU only** (no CUDA GPU present).
Test clip: 20 s of Japanese dialogue with background SFX, extracted from a real
anime episode (`07:00`–`07:20`), sample rate 44.1 kHz mono.

Where a model could not be safely completed on this machine (see the htdemucs
incident below), that is reported as a finding in its own right — RAM headroom is
exactly the axis this document exists to compare.

**Methodology note on timing:** all "time per 60 s of audio" figures are extrapolated
linearly from the 20 s clip (`measured_time × 3`) and, for enhancers, include
one-time model load (checkpoint read + graph init), not just steady-state inference.
Model load is disk-cache-dependent (1–5 s) and is a fixed cost paid once per
process — a long-running batch job or a live-filter process amortizes it across
many minutes of audio, so per-file numbers here are a **worst case**, not the
steady-state throughput a persistent process would see.

---

## Table 1 — Separator / Music-Removal Models

| Model | Library / format | Model size | Peak RAM (measured) | Time / 60 s audio | Real-time factor | Vocal isolation quality | RT-streaming suitable? |
|---|---|---|---|---|---|---|---|
| **MDX-Net UVR-MDX-NET-Voc_FT** | audio-separator / onnxruntime (ONNX) | 66 MB | **~3.3 GB** (measured, stable) | **134.7 s** (measured) | 0.45× (slower than real-time) | Good — recommended default; kept SFX/dialogue intact in prior testing | ⚠️ Partial — chunked (~200-400 ms internal windows), not a true streaming API, but the ONNX graph itself is small/fast enough to be re-engineered for it |
| MDX-Net Kim_Vocal_2 | audio-separator / onnxruntime (ONNX) | 130 MB | Not tested | Not tested | — | Not evaluated this session | Same caveat as Voc_FT — untested RAM/speed trade-off from the larger model |
| **DEMUCS htdemucs (INT8)** | torch (single transformer model) | ~80 MB quantized | **~1.24 GB** *(corrected 2026-09-08 — was recorded as ">6-7 GB, OOM-killed")* | **10.5 s / 60 s chunk (RTF 0.177)** ⚠️ measured at 4 threads on a 32-core box, not the 4-core reference | N/A | Previously rated good by ear (didn't cut background SFX) | ⚠️ Partial — offline chunked, no streaming API, but comfortably faster than real time. The old "memory can't be reduced" claim was backwards: the app was *forcing* a 60 s context via `segment=CHUNK_SEC`, far **above** the model's native 7.8 s `training_length`. Removing that kwarg drops peak RSS ~12×. |
| DEMUCS htdemucs_ft | torch (BagOfModels ×4 sub-models, run sequentially) | ~80 MB ×4 quantized | Not tested live — the stated reason ("too risky after the htdemucs incident") is void as of 2026-09-08, that incident being the `segment=60` bug; — estimated similar peak *per sub-model* as htdemucs (sub-models run one at a time, not stacked), so roughly the same peak, ~4× the wall-clock time | Estimated ~4× htdemucs | N/A | Estimated marginal quality gain over htdemucs (ensemble average) | ❌ No |
| DEMUCS mdx_extra | torch (BagOfModels ×4) | similar | Same estimate as htdemucs_ft | Estimated ~4× htdemucs | — | Labeled "Best" in the app's own UI; unverified this session | ❌ No |
| Spleeter (4stems) | TensorFlow | ~150 MB | Not tested live — TF import alone adds 500 MB–1.5 GB before any separation happens (confirmed empirically earlier this session; this cost has now been made *lazy* — see Fixes below) | Not tested | — | Historically weaker for anime-style vocals (general-purpose 4-stem model, not tuned for the vocal/instrumental split this app needs) | ❌ No — heavyweight runtime, no streaming design |
| **sherpa-onnx Spleeter 2-stem int8** *(added 2026-07-04)* | sherpa-onnx C++ runtime | 2 × 26 MB | **~600 MB** (measured) | **4.0 s** (measured, RTF 0.067 — 1 thread suffices; 4 threads no faster) | **15× faster than real-time** | Architecturally below MDX — listening files in `test_outputs/` (`bench_dialogue07m00_sherpa_spleeter_i8_*.wav`) for by-ear judgment | ✅ **Yes in practice** — offline API, but at RTF 0.067 a 1-2 s chunked live loop keeps up easily; first separator measured on this machine that can |
| **sherpa-onnx UVR Voc_FT** *(added 2026-07-04)* | sherpa-onnx C++ runtime | 64 MB | ~4.3 GB (measured) | 84.4 s @ 4 threads (measured, RTF 1.41); 143.4 s @ 1 thread | 0.71× | Identical weights to audio-separator's Voc_FT — same output quality, 1.6× faster from runtime efficiency alone | ❌ No — still slower than real-time on this CPU |

**Read on RAM peaks — CORRECTED 2026-09-08.** The paragraph below originally
concluded that MDX-Net was "the *sole* practical, safe separator on this class of
machine" and that htdemucs's real peak was "5-6× higher" than the ~1.2 GB estimate
in earlier metrics.

**That conclusion was inverted.** The original text had already identified the
decisive variable — it noted the old estimate "appears to have never been measured
against this exact `apply_model` call shape (`segment=60, overlap=0.25`,
single-shot)" — but read it as the *estimate* being wrong rather than the *call
shape*. It was the call shape. Passing `segment=60` overrides htdemucs's native
7.8 s `training_length` with a 60 s context, and transformer attention is quadratic
in context length.

Measured same-machine A/B on one 60 s chunk, after removing the kwarg:

| | `segment=60` (as shipped) | no `segment` (fixed) |
|---|---|---|
| result | RuntimeError in `htdemucs.forward` | OK |
| peak RSS | 14,929 MB | **1,237 MB** |
| wall time | 28.3 s, then crashed | **10.5 s** (RTF 0.177) |

The ~1.2 GB estimate was accurate. htdemucs is practical on this class of machine,
and at RTF 0.177 it is roughly 2.5× faster than MDX-Net Voc_FT (RTF 0.45), the
app's current default. Speed figure measured at 4 threads on a 32-core box —
re-measure on the 4-core reference before ranking. See `ROADMAP.md` §1.4.

---

## Table 2 — Pre-/Post-processing Techniques

| Technique | Time / 60 s audio | RAM delta | Quality effect | RT-streaming suitable? | Notes |
|---|---|---|---|---|---|
| DSP pipeline (`advanced_postprocess_chunked`, balanced mode) | **0.20 s** | ~11 MB | Mild general cleanup (EQ/filtering); does not remove structural music, just polish | ✅ Yes — negligible cost | Always cheap regardless of mode; safe to leave on |
| **DeepFilterNet3** (enhancer, `atten_lim_db=20`) | **16.2 s** (incl. ~1-5 s one-time model load) | ~585 MB (model + torch runtime) | **Best measured** — user-confirmed: reduced guitar/piano bleed-through to "a soft tap on higher notes." Also a hard prerequisite for the VAD gate below (builds the stable noise profile VAD's energy fallback needs) | ⚠️ Borderline — architecture is frame-based/causal by design (built for real-time use elsewhere), but this cold-start number is dominated by one-time model load; steady-state per-frame cost is far lower | **Recommended default enhancer** |
| DPDFNet (sherpa-onnx) | 8.1 s | ~18 MB | Not evaluated by ear this session | ✅ Yes — tiny ONNX footprint | Untested quality, cheap to try |
| GTCRN (sherpa-onnx) | **3.8 s** (fastest enhancer tested) | ~0.6 MB (smallest footprint tested) | Leaves more residue — user reported lingering "guitar/piano notes" | ✅ Yes — fastest + lightest of everything tested | Good fallback for the tightest RAM/CPU budgets, at a quality cost |
| SpeechDenoiser (custom ONNX, 48 kHz) | 15.0 s | ~7.6 MB | Not evaluated by ear this session | ✅ Yes — genuinely frame-based (480-sample hops, explicit `states` tensor carried between frames) | Its I/O shape is the closest thing in this codebase to an actual streaming API — worth prioritizing if/when the live-app work needs a drop-in enhancer |
| **VAD gate** (hybrid Silero + energy fallback, this session's tuned v10 logic) | 4.2 s | ~7 MB | Gates non-vocal gaps/instrumental-only sections between speech; tuned this session to stop cutting high-pitched voices *and* stop letting music-only SFX segments through | ⚠️ Partial — Silero itself streams natively (32 ms frames), but this app's energy-anchoring math (median-relative threshold, ±1.5 s expansion window) is whole-clip, non-causal | **Must run after the enhancer, not before** — this was a real bug fixed this session (see Fixes) |
| Spectral transient gate | Not benchmarked (abandoned) | — | Ineffective: guitar residue in this content is sustained, not transient, so the gate's background estimator adapts to it and never fires | N/A | Still present as a UI toggle; not recommended |

---

## Fixes made this session (context for the numbers above)

1. **VAD/enhancer pipeline order bug** — the shipped app ran the VAD gate *before*
   the enhancer. This is the wrong order: DeepFilterNet needs a continuous signal to
   build a stable noise model, and gating first fragments it, which was causing
   Japanese singing and high-pitched voices to get cut. Fixed to
   `separate → DSP → enhancer → VAD` in `_process_single_track`.
2. **VAD energy fallback false-positives** — the hybrid VAD (added to catch
   high-pitched voices Silero misses) was anchoring its energy threshold to the
   *whole clip's* median energy, which let DF-attenuated music residue clear the
   bar. Fixed by anchoring the threshold to the median energy of
   Silero-detected voiced frames specifically, restricting the fallback to a
   ±1.5 s window around real Silero detections, and adding a mean-VAD-confidence
   filter that drops Silero segments whose average speech-probability score is
   itself low (catches SFX/music segments Silero briefly misfires on).
3. **Eager TensorFlow import** — `_check_spleeter()` used to `import tensorflow`
   at module load time just to check `tf.estimator` compatibility, costing every
   single app launch 500 MB–1.5 GB of RAM even for users who never touch Spleeter.
   Now it's a `find_spec`-only presence check; the real compatibility check moved
   into `separate_with_spleeter()`, running only if Spleeter is actually selected.

---

## What this means for "best default" and for real-time

**Best validated combo for file processing today:** MDX-Net (UVR-MDX-NET-Voc_FT)
→ DSP pipeline → DeepFilterNet (`atten_lim_db=20`) → hybrid VAD gate. It's the only
separator that completes reliably in the RAM this machine actually has free, and
every other stage in that chain is cheap (≤ 600 MB, ≤ 17 s per minute of audio).
This is also, empirically, the exact combination arrived at through this session's
iterative testing — the data confirms rather than contradicts that path.

**Addendum 2026-07-04:** the same Voc_FT weights now run 1.6× faster through
sherpa-onnx's C++ source-separation runtime (84 s vs 135 s per audio-minute,
identical output), and **sherpa's Spleeter 2-stem int8 is 34× faster than the
current default at ~600 MB RAM** — the first genuinely real-time-capable
separator measured on this machine. If its by-ear quality holds up behind the
DF→VAD chain, it changes the default recommendation for the live app. See
section 9 of `docs/REALTIME_FILTER_RESEARCH.md` for details and listening files.

**None of the separators are real-time-capable as-is.** All three architectures
(DEMUCS, MDX-Net's onnxruntime graph, Spleeter) process a fixed-size window
non-causally; there's no streaming/causal mode to fall back to. This matches the
conclusion already on file in `docs/REALTIME_FILTER_RESEARCH.md` and the
project's model-metrics memory: genuine real-time vocal *separation* needs an
architecture designed for it (e.g. HS-TasNet), which doesn't have public weights
yet. Of the pieces benchmarked here, **SpeechDenoiser**'s frame-by-frame API
(explicit hidden state carried between 480-sample hops) is the closest thing to a
real streaming building block already in this codebase, and DeepFilterNet's
architecture is designed to be causal even though today's usage is batch-mode —
both are better starting points than any separator for the live-filtering effort
described in `docs/LIVE_APP_PLAN.md`.

---

## Appendix — why htdemucs's memory can't just be tuned down

`apply_model()` accepts a `segment` parameter that's meant to sub-chunk long audio
internally with overlap-add. Passing a shorter value (tried: 10 s) to reduce peak
memory crashes with:

```
RuntimeError: shape '[1, 4, -1, 343980]' is invalid for input of size 3528000
```

`343980` samples ≈ 7.8 s at 44.1 kHz — htdemucs's transformer branch has a fixed
`training_length` baked into its reshape logic, and `apply_model` only handles this
correctly when it's driving that segmentation itself with the model's own default,
not when an external caller specifies an arbitrary value. So the memory cost of a
single htdemucs pass is not caller-tunable without patching demucs internals.
