# Offline Speed: Getting Under 1 Minute per 10 Minutes of Audio

**Researched and measured 2026-10-18.** Target: separation real-time factor (RTF) of 0.1 or lower, i.e. at most 60 s of processing per 10 min of audio.

Related doc: `SIMILAR_PROJECTS_RESEARCH.md` covers the models referred to here.

**How these numbers were measured**

- **Machine:** AMD Ryzen 9 9950X3D (Zen 5, 16 cores / 32 threads, with AVX-512 extensions VNNI and BF16), 30 GB RAM, Linux.
- **Background load:** about 6–8 cores were busy with unrelated jobs throughout. Repeat runs at 16 threads varied by up to 35 %. Idle-machine numbers will be equal or better.
- **Process limits:** every run was capped at 5 GB of memory and ran at `nice 10`.
- **What "s / 10 min" means:** steady-state time for 10 × 60 s app chunks. Model load time is listed separately. The headline row was also confirmed on a real 600 s file.
- **Quality column:** SI-SDR (a signal-similarity score in dB) of the vocals against today's output from the app.
  - **About 50 dB or more:** effectively the same output.
  - **About 30 dB:** small differences, mostly at window seams.
  - **About 14 dB:** clearly different.
  - **Other models:** this is only an agreement score, not a quality score.
- **Versions:** onnxruntime 1.31.0, openvino 2026.4.1, nncf 3.4.0, audio-separator 0.47.0, sherpa-onnx 1.13.8, torch 2.14.1, demucs 4.1.0.
- **Scripts and raw results:** in the session scratchpad (`speed/scripts`, `speed/results/table.md`). They are not committed.

---

## Answer

**On this machine:** yes, with no change in output. The app today needs **220–228 s per 10 min** (RTF 0.38). The same Voc_FT model, run through **OpenVINO with BF16 compute** and an explicit thread count, needs:

| Run | Time per 10 min | Quality vs today |
|---|---|---|
| Real 10-min file, 16 threads, today's overlap 0.25 | **41.7 s** + 3.2 s load | 50.6 dB (same output) |
| Same, with overlap 0 | **33.4 s** | 29.5 dB (seam differences) |

**On a 4-core laptop:** borderline at best.

- **Measured on 4 threads of this machine:**
  - OpenVINO BF16, overlap 0: 54–61 s.
  - OpenVINO BF16, overlap 0.25: 74–77 s.
  - Today's app: 157 s.
- **Real laptop cores are slower** (estimated 1.3–2×).
- **BF16 needs CPU support:**
  - AMD Zen 4 / Zen 5 laptops have it: about **75–120 s** per 10 min with Voc_FT.
  - Intel client chips don't, so they fall back to FP32: about **200–400 s** (from an AVX2-only emulation run at 217 s).
- **Laptops therefore need either a "Fast" model or a GPU** to get under 60 s.

**The whole pipeline (idle machine, measured 2026-10-23; see "Full pipeline in the app" below):**

| Setting | Total per 10 min |
|---|---|
| Default chain: fast engine + DeepFilterNet + VAD gate | **58 s**, under the target |
| + "Keep sound effects", Fast (MRX) | 85 s |
| + "Keep sound effects", Quality (CDX23) | 95 s |
| Fastest: MRX for both dialogue and effects | **55 s** |

---

## Measured results (Voc_FT unless noted)

| Configuration | Threads | s / 10 min | RTF | Peak RAM | vs today (dB) |
|---|---|---|---|---|---|
| **Today: audio-separator, app settings** (overlap 0.25, batch 1, segment 256, denoise off) | default | **220–228** | 0.38 | 3.6 GB | ref |
| same, threads set explicitly | 4 / 8 / 16 | 157 / 118 / 115 | 0.26 / 0.20 / 0.19 | 3.6 GB | identical |
| same, overlap 0.1 / 0 | 8 | 97 / 88 | 0.16 / 0.15 | 3.6 GB | 29.7 / 29.8 |
| same, denoise on | 8 | 231 | 0.39 | 3.6 GB | 39.4 (2× slower) |
| same, segment 512 / 128 | 8 | 271 / 213 | — | 2.7 / 1.5 GB | ≈30 (switches to the slower onnx2torch path) |
| Own MDX loop, onnxruntime FP32 | 4 / 8 / 16 | 142 / ~105 / ~104 | 0.24 / 0.17 / 0.17 | 3.25 GB | 61.0 |
| sherpa-onnx UVR Voc_FT (10 s pieces) | 4 / 16 | 138 / 99 | 0.23 / 0.16 | 4.9 GB | — (20 s and 60 s pieces hit the memory cap) |
| OpenVINO FP32 | 4 / 8 / 16 | 132 / 78 / 55 | 0.22 / 0.13 / 0.09 | 3.0 GB | 61.0 |
| **OpenVINO BF16** | 4 / 8 / 16 | **74–77 / 44 / 32–44** | 0.12 / 0.07 / 0.05–0.07 | 2.9 GB | **50.6** |
| **OpenVINO BF16, overlap 0.1 / 0** | 16 | **26 / 23.5** | 0.04 | 2.9 GB | 29.6 / 29.7 |
| OpenVINO BF16, overlap 0 | 4 | 54 (61 on the real 600 s file) | 0.09–0.10 | 2.9 GB | 29.7 |
| INT8: OpenVINO NNCF / onnxruntime static QDQ | 16 | 21 / 76 | 0.035 / 0.13 | 1.2–1.7 GB | **14.0 / 14.2** — too lossy |
| VR 2_HP-UVR (audio-separator) | 8 | 194 | 0.32 | 2.3 GB | different model |
| **MRX int8** (dialogue / music / effects, MIT) | 4 / 16 | **16 / 18** | 0.03 | 2.4 GB | different model; keeps SFX |
| Spleeter 2-stem int8 (sherpa-onnx) | 4 / 16 | **11 / 12** | 0.02 | 2.2 GB | different model; lower quality |
| CDX23 HTDemucs, 1 checkpoint, shifts 0, overlap 0.1 | 4 / 16 | 92 / 59 | 0.15 / 0.10 | 1.4 GB | different model; its "speech" stem excludes singing |
| TIGER-DnR int8 ONNX | 16 | 2360 (4 s hop) / 787 (no overlap) | 3.9 / 1.3 | 2.8 GB | — far too slow (see below) |

**Didn't help:**

| Setting | Result |
|---|---|
| `batch_size` in audio-separator | No effect: each chunk is a batch of one |
| True batch 4 in OpenVINO | Slower |
| 32 threads (SMT) | No gain over 16 |
| 2–4 parallel worker processes | No gain; two BF16 workers don't fit in 5 GB |
| onnxruntime spinning off / denormals-as-zero | No gain |
| The iGPU (2-CU RDNA2) | No runtime can use it, and it is weaker than the CPU anyway |

---

## Speed levers, ranked by measured effect

1. **Set the thread count explicitly. 2× faster, output identical, trivial change.**
   - **Today:** with onnxruntime's default, threads are pinned to cores, and any other load on the machine stalls them (220 → 115 s).
   - **Fix:** audio-separator has no setting for this, so it has to be patched or bypassed (see 4).
2. **OpenVINO runtime with BF16 compute. 1.9× faster at 4 threads, 2.3–3× at 16, output effectively identical (50.6 dB).**
   - **CPU support needed:** AVX512_BF16 (AMD Zen 4/5) or AMX (recent Intel Xeon). Fall back to OpenVINO FP32 elsewhere; that alone is 1.9× faster at 16 threads and about 1.07× at 4.
   - **Known bug:** OpenVINO issue [#38256](https://github.com/openvinotoolkit/openvino/issues/38256) reports wrong BF16 matrix-multiply results on a Ryzen 9950X. For Voc_FT the 50.6 dB agreement shows this model is not affected. Re-check that number for any new model or OpenVINO version.
   - **Why the earlier OpenVINO test showed only +16 %:** that used the onnxruntime OpenVINO plugin, which runs FP32 on CPU. Use OpenVINO directly, or set `INFERENCE_PRECISION_HINT=bf16`.
3. **Window overlap 0.25 → 0.1 or 0. 1.25–1.4× faster.**
   - **Cost:** about 30 dB difference from today's output, mostly at window seams.
   - **Published comparisons:** 0.02–0.2 dB vocal SDR for overlap changes of this size (deton24's separation guide).
   - **Before changing the default:** check by ear. Offer it as the "Fast" setting either way.
4. **Replace audio-separator with a small MDX loop. 1.1–1.15× faster, output identical (61 dB), less RAM.**
   - **Why it's faster:** audio-separator always runs a second STFT/iSTFT pass for the unused stem, and writes each chunk out to WAV and back.
   - **What it also enables:** levers 1 and 2. The benchmark's `bench_mdx.py` is a working starting point.
5. **"Fast" mode models for machines without BF16 or a GPU:**
   - **MRX int8:** 16–18 s per 10 min even on 4 threads. MIT-licensed. Splits dialogue / music / effects, so it keeps SFX. Its quality on anime is unmeasured.
   - **Spleeter int8:** 11–12 s, lower quality.
   - **CDX23 single checkpoint:** 59–92 s; best for SFX but borderline on speed.
6. **Skip music-free stretches** with a music detector. Not measured. How much time it saves depends on the share of each episode that has no music, which hasn't been measured on real content.
7. **GPU** (sourced estimates, not measured here):
   - **NVIDIA:** onnxruntime with CUDA / TensorRT, about 10–20 s per 10 min.
   - **AMD / Intel on Windows:** DirectML is supported in audio-separator for MDX models. Community reports are mixed.

**Do not use:**
- **INT8.** Every variant tried was 14 dB away from today's output, even with the sensitive layers kept in float. It would need accuracy-aware quantization.
- **audio-separator's `denoise`.** Runs the model twice.
- **Segment sizes other than 256.** These switch audio-separator to a slower path.

---

## Why TIGER-DnR is slow

TIGER's int8 ONNX export has **38,817 graph nodes**. The biggest groups are:

| Node type | Count |
|---|---|
| ReduceMean | 8,388 |
| Add | 5,556 |
| Mul | 4,734 |
| Sub | 3,861 |
| Div | 2,895 |
| Conv | 2,619 |
| Sqrt | 2,265 |

These are normalisation layers broken up into elementwise operations. The slowness comes from the overhead of running that many small operations, not from the model's size (1.4M parameters per network). So the "about 0.1–0.5 RTF" estimates in the earlier research were wrong.

A re-export with fused normalisation (ONNX `LayerNormalization` / `GroupNormalization`), or running it in OpenVINO, might change this. That is unmeasured. Until then, use **MRX** as the fast "keep effects" option and **CDX23** as the quality one.

---

## Full pipeline in the app (measured 2026-10-22)

The fast engine is `fast_separation.py`, wired into `MusicAssassinGUI_v0.4.4.py`. Each stage now prints `[time]` lines to the log.

**How this was run**
- **Harness:** a headless run of the app's own `_process_single_track()` with a dummy Tk root, on the same real 10-min episode file.
- **Settings:** processing mode Balanced, enhancer DeepFilter, VAD gate on.
- **Machine load:** two sessions.
  - **2026-10-23, idle machine (load under 1 at start).** These are the reference numbers.
  - **2026-10-22, busy machine.** Unrelated jobs held the load average at 20–28 of 32 threads.

**Idle machine (2026-10-23):**

| Stage (s per 10 min) | Best, no effects | Best + effects (MRX) | Best + effects (CDX23) | Fastest: MRX dialogue + MRX effects |
|---|---|---|---|---|
| Separation | 39.5 (incl. 2.8 s model load) | 35.6 | 35.5 | 36.8 |
| Effects stem | — | 30.5 | 46.5 | (same pass) |
| Post-process (Balanced) | 1.4 | 1.3 | 1.4 | 1.3 |
| DeepFilterNet | 12.3 | 11.8 | 7.5 | 12.1 |
| VAD gate | 4.6 | 4.5 | 3.8 | 4.3 |
| Effects mix + export | 0.2 | 0.3 | 0.4 | 0.3 |
| **Total** | **58** | **85** | **95** | **55** |

**Busy machine (2026-10-22), for comparison:**

| Stage (s per 10 min) | Best, no effects | Best + effects (MRX) | Best + effects (CDX23) | Fastest: MRX dialogue + MRX effects |
|---|---|---|---|---|
| Separation | 50.3 (Voc_FT, OpenVINO BF16) | 49.9 | 57.1 | 48.7 (MRX, one pass for both stems) |
| Effects stem | — | 38.6 | 79.8 | (same pass) |
| Post-process (Balanced) | 1.1 | 2.2 | 1.1 | 1.6 |
| DeepFilterNet | 15.9 | 18.2 | 9.9 | 15.2 |
| VAD gate | 4.8 | 5.3 | 4.3 | 5.6 |
| Effects mix + export | 0.3 | 0.6 | 0.4 | 0.4 |
| **Total** | **73** | **116** | **153** | **72** |

Other runs in the same session:
- **Separation only** (no enhancer, no VAD gate): 77 s at load 43.
- **Voc_FT at overlap 0.1 + MRX effects:** 105 s. Separation took 39 s, against 60 s at overlap 0.25.

**Findings**
- **DeepFilterNet costs 10–18 s per 10 min,** less than the 27 s estimate.
- **The VAD gate costs about 5 s per 10 min.** Its first use in a session also costs a one-time 9–22 s to load Silero through `torch.hub`. That load is mostly Python imports, so the app now starts it in a background thread at the start of each track (`preload_silero_vad()`), where it overlaps with separation.
- **MRX took 39–51 s on the busy machine and 30 s idle, against 16–18 s measured earlier.** The model itself took 17–22 s per 10 min. The rest is torch STFT/iSTFT work, which suffers most from the contention: two identical runs on a 60 s clip took 3.0 s and 5.2 s. The app now synthesises only the stems it needs.
- **OpenVINO vs onnxruntime for MRX:** 1.7 vs 2.2 s of model time per 60 s.

## Keep sound effects (measured 2026-10-22)

**How it works.** This runs in parallel with the de-musicing:
- **Dialogue** comes from the selected separator, as before.
- **The effects stem** is extracted from the original audio by MRX or CDX23. Its input skips the −40 dB per-sample noise gate, which would erase ambience.
- **Level matching:** the effects are scaled by the gain that post-processing applied to the dialogue, so they keep their original level relative to it.
- **Mixing:** they are added after the VAD gate, so sounds between lines survive. A 512-sample block limiter then keeps peaks at or below 0.98 without ducking the whole file.
- **When the separator is MRX too,** one MRX pass gives both stems.

**Ground-truth test** (`fx_synth.py`, session scratchpad):
- **The mix:** 12 two-second pieces of the original episode audio, taken from seconds where both models put at least 70 % of the energy in the effects stem. They were inserted into the music-only 90 s clip (an anime opening, so music plus singing) at −6 dB under the music.
- **"Leak":** the effects output's energy outside the inserted events, relative to the music there. Lower is better.
- **"Keep":** the SI-SDR of the effects output against the inserted events, inside them. Higher is better.

| Model | Cleanup | Leak | Keep (SI-SDR) | Event energy kept |
|---|---|---|---|---|
| MRX | none | −16.4 dB | −0.3 dB | −1.0 dB |
| MRX | Wiener re-filter with all three stems | −15.9 | −0.9 | −0.5 |
| MRX | **music over-subtraction, β = 1 (now the default for MRX)** | **−21.3** | −1.3 | −3.8 |
| MRX | music over-subtraction, β = 2 | −23.4 | −2.3 | −4.9 |
| MRX | ducking where music outweighs effects by 6 dB | −19.2 | −1.1 | −1.7 |
| CDX23 | **none (default)** | **−24.9** | **+3.6** | −1.4 |
| CDX23 | Wiener re-filter | −24.4 | +3.4 | −1.0 |
| CDX23 | music over-subtraction, β = 1 | −26.3 | +2.3 | −2.7 |

**Real material** gives the same picture (`fx_eval.py`):
- **The music-only clip,** where the effects stem should be silent: it sat at −17 dB vs the input with MRX and −25 dB with CDX23.
- **The 10-min episode's music-dominated seconds:** −14 dB with MRX and −16 dB with CDX23. These seconds are picked by each model's own music stem, so they may also contain real effects.

**Conclusions**
- **CDX23 is the quality option.** It leaks about 8.5 dB less music than MRX and keeps the events better. It costs 47 s per 10 min on the idle machine (80 s on the busy one).
- **MRX leaks music at an audible level** even after the post-mask. Offer it as the fast option and say so in the UI.
- **The Wiener re-filter was removed.** It uses the model's own estimates, so it mostly re-applies masks the model already applied.
- **Effects level control (added 2026-10-23):** Quieter (−6 dB) / Normal / Louder (+3 dB) / Much louder (+6 dB), next to "Keep sound effects". It defaults to Louder for MRX, where +3 dB gave a small but audible lift by ear, and to Normal for CDX23.
  - **What it changes:** only the balance between effects and dialogue. Any leaked music moves with the effects.
  - **Use for Louder (+3 dB):** it roughly restores what MRX's music cleanup takes off the effects.
  - **Live app:** a level knob fits there more naturally, since the result is heard instantly. Live has no effects path yet.
- **Possible next steps** (not built yet):
  - A music-presence gate: cut the effects only where a music detector fires (§3.3 C in `SIMILAR_PROJECTS_RESEARCH.md`).
  - Averaging the MRX and CDX23 music estimates.

---

## Recommended setup for the offline app

1. **Default engine.** Own MDX loop, using OpenVINO with BF16 compute when the CPU supports it, otherwise OpenVINO FP32. Threads = physical cores. Overlap 0.25 for "Best", 0.1 for "Fast".
   - **Desktop CPUs with BF16:** about **35–45 s per 10 min** for separation, with the same output as today.
2. **Model choice shown to the user.** **Best** (Voc_FT) / **Keep effects** (CDX23, about 1–1.5 min per 10 min) / **Fast** (MRX int8, under 20 s).
   - **Recommend automatically:** pick Fast when the CPU lacks BF16 and no GPU is present.
3. **Rest of the pipeline: measured** (see "Full pipeline in the app"). DeepFilterNet (10–18 s) is the next-largest stage after separation. To get margin under 60 s, run it only on regions the VAD gate keeps (about 50 % of an episode).
4. **Later:** a GPU path (CUDA / DirectML) and music-presence skipping.
