# Handover: keeping sound effects in Music Assassin Live

**From** the offline "keep sound effects" work in `Music-Assassin` (commit `71a12d1`, 2026-10-22/23).
**For** whoever picks up SFX-keeping in `music-assassin-live`.

Read alongside the live repo's `docs/ROADMAP.md`, specifically A1, A2, A4, B4 and §11. This doc does not repeat them. It adds what the offline work measured and says which parts carry over to a realtime app.

Labels used below: **measured** (numbers from runs on this machine), **estimate** (worked out, not run), **untested**.

---

## 1. Summary

1. **The offline app now keeps sound effects, and listeners prefer it.**
   - **How:** a second path extracts an effects stem from the original audio and adds it back to the de-musiced dialogue.
   - **By ear (user):** CDX23 and MRX effects both sounded much better than voice-only output. Even the fastest MRX-only setting kept effects, slightly muted.
2. **The models do not transfer to live as-is.**
   - Both effects models (MRX, CDX23) are non-causal: they need seconds of future audio.
   - There is still no public, causal dialogue/music/effects model. The live ROADMAP (§7 note, A4) reached the same conclusion.
   - So SFX-keeping at today's 50–70 ms latency is not available from any existing model.
3. **What does transfer:**
   - The measurements: which model leaks how much music, and which cleanup helps.
   - The mixing design: separate effects path, level matching, add back after any gate.
   - The UI pattern: model choice plus a per-model level default.
   - The test method: a ground-truth leakage test.
4. **Four realistic routes for live**, cheapest first (details in §4):
   - (a) a music-presence gate;
   - (b) a classifier gate on the residual;
   - (c) a "movie mode" running MRX with declared latency;
   - (d) distilling a causal student model, with CDX23 as teacher.

   The browser look-ahead companion can run the offline models unchanged.

---

## 2. What was built and measured offline

### 2.1 Pipeline (`MusicAssassinGUI_v0.4.4.py`, `fast_separation.py`)

```
original ─┬─ gate + HPF + normalize ─▶ dialogue separator ─▶ post-process ─▶ enhancer ─▶ VAD gate ─┐
          │                                                                                        ├─▶ + ─▶ block limiter ─▶ out
          └─ HPF + normalize (NO gate) ─▶ effects model ─▶ [music over-subtraction] ─▶ × gain ───────┘
```

Three design decisions, each made for a measured or observed reason:

- **The effects input skips the noise gate.** The app's −40 dB per-sample gate erases ambience and room tone.
- **Effects are added after the VAD gate.** The gate silences speech gaps, which is exactly where footsteps and doors sit.
- **Level matching:** effects × (post-process normalization gain) × user level.
  - This puts the effects at their original level relative to the dialogue.
  - A 512-sample block limiter then holds peaks ≤ 0.98, so a single explosion doesn't duck the whole file.

### 2.2 Ground-truth leakage test (measured)

**Method.** This is the reusable part; rebuild it in the live repo's `bench_quality.py` corpus terms.

- **Mix:** 12 two-second pieces of real episode audio were inserted into a music-only 90 s anime OP (music plus singing, no effects), each at −6 dB under the music.
- **Which pieces:** seconds where both MRX and CDX23 put ≥ 70 % of the energy in their effects stem.
- **Leak:** effects-output energy outside the inserted pieces, relative to the music there. Lower is better.
- **Keep:** SI-SDR of the effects output against the inserted pieces, inside them. Higher is better.

| Model | Cleanup | Leak | Keep (SI-SDR) | Effects energy kept |
|---|---|---|---|---|
| MRX | none | −16.4 dB | −0.3 dB | −1.0 dB |
| MRX | Wiener re-filter with its own three stems | −15.9 | −0.9 | −0.5 |
| MRX | **music over-subtraction β = 1** | **−21.3** | −1.3 | −3.8 |
| MRX | over-subtraction β = 2 | −23.4 | −2.3 | −4.9 |
| MRX | duck effects where music > effects + 6 dB | −19.2 | −1.1 | −1.7 |
| CDX23 | **none** | **−24.9** | **+3.6** | −1.4 |
| CDX23 | over-subtraction β = 1 | −26.3 | +2.3 | −2.7 |

**Over-subtraction** means: per STFT bin, gain = `max(0.05, 1 − β·|Music| / |Effects|)`, applied to the effects STFT. It uses the model's own music estimate.

**Takeaways:**
- **CDX23 leaks about 8.5 dB less music than MRX** and keeps the effects better.
- **A Wiener re-filter using the model's own stems does nothing.** It moved leakage by less than 0.5 dB.
- **Over-subtraction is the cleanup that works for MRX.** It is now on by default for MRX only.
- **MRX still leaks music at an audible level** after cleanup.
- **Real material agrees.** On the music-only clip, the effects stem sat at −17 dB (MRX) and −25 dB (CDX23) relative to the input.

### 2.3 Cost (measured, 10-min episode, 16 threads, idle machine)

| Path | s per 10 min of audio |
|---|---|
| Voc_FT dialogue (OpenVINO BF16) + DeepFilterNet + VAD gate | 58 |
| + MRX effects | 85 (MRX path ≈ 30 s, of which the model itself ≈ 17–22 s; the rest is STFT/iSTFT) |
| + CDX23 effects | 95 (CDX23 ≈ 47 s) |
| MRX for both dialogue and effects, one pass | 55 |

Earlier MRX model-only figure: **16–18 s per 10 min on 4 threads.** On onnxruntime it took 2.2 s per 60 s, against 1.7 s on OpenVINO.

### 2.4 User-facing results

- **The control:** "Keep sound effects: Off / Fast (MRX, some music bleed) / Quality (CDX23, cleaner, slower)".
- **The level control:** Quieter (−6) / Normal / Louder (+3) / Much louder (+6) dB.
  - **Default is per model:** +3 dB for MRX, which makes up for what over-subtraction removes; Normal for CDX23.
  - **User verdict on +3 dB for MRX:** "small but some increase", accepted as the default.
- **What the level control is for:** it only changes the effects/dialogue balance. Leaked music moves with it, because the model fixes the leak-to-effects ratio. For a realtime app, a live slider makes more sense than a pre-set level.

---

## 3. Why live can't copy it directly

- **MRX is a bidirectional LSTM and needs future audio.** Offline runs used 30 s chunks.
  - Short windows lose quality at the edges, as the live repo's own Spleeter spike (A1) found: **≥ 1.1 s latency**.
  - MRX quality as a function of window length is **untested**. Measure that first (§4.3).
- **CDX23 is a torch HTDemucs.** The live app is onnxruntime-only, and HTDemucs ONNX export is not straightforward (**untested** here). CDX23 is a teacher and browser-companion model, not a live processor.
- **Players don't see the app's delay.** The trap-sink design reports about 0 ms of latency to video players, because PipeWire leaves monitor ports out of latency propagation (`SIMILAR_PROJECTS_RESEARCH.md` §4).
  - Any mode that adds hundreds of milliseconds will break lip-sync until latency is reported correctly.
- **Open product decision:** ROADMAP Q3, the latency ceiling, is still unanswered, and it gates route (c).

---

## 4. Routes for live, cheapest first

### 4.1 Music-presence gate (S, low latency, keeps all effects in music-free scenes)

`out = p·processed + (1 − p)·dry`, where `p` is a music-presence probability.

- **Detector:** EfficientAT mn04 (MIT code) or YAMNet ONNX (Apache-2.0). Run on 1 s windows with a 250 ms hop.
  - Avoid CED: GPL-3.0 in the code repo (`RESEARCH_PRIOR_ART.md` §1.6).
  - Use a **soft** gate with hysteresis: attack ~150 ms, release 1–2 s. Detector F1 is below 60 % on quiet background music (AI-OpenBMAT).
- **Delay rule:** dry and processed must share one constant delay. If they don't, lip-sync jumps whenever the gate moves.
- **Cost:** about 0.3–0.5 s of music leaks at each music onset. Effects *under* music are still lost.
- **Different from the voice-presence gate** in `RESEARCH_PRIOR_ART.md` §1.6 #3: that one closes on music-only passages, while this one opens on music-free passages. They combine.

### 4.2 Classifier gate on the residual (M)

ROADMAP A4's own unpark route.

1. Compute the residual `R = dry − wet`. That is what the enhancer removed: music plus effects.
2. Classify R in 0.5 s windows.
3. Add back the windows that aren't music.

- **Offline precedent:** YAMNet reached 0.5 % leakage with 94 % of effects kept.
- **Why the residual helps:** classifying R is easier than classifying the full mix, because the vocals are already gone.
- **Limits:**
  - Short hits arrive up to 0.5 s late, unless the window delay is declared as latency.
  - Effects under continuous music stay lost.
- **Don't confuse it with the dead end.** ROADMAP §11 records that *pure-DSP* add-back plateaus at ~8 % leakage. This route uses a semantic classifier instead.

### 4.3 Movie mode: MRX effects path with declared latency (M, needs Q3 and the latency fix)

**Design** (estimate; nothing here has run in streaming form):

- **Paths:** keep today's dialogue path (`dpdfnet_hr` or `stemgen`) and add an MRX effects path. Both are delayed by one constant D ≈ 0.5–1.5 s.
- **Windowing:** MRX on 2–3 s windows with a 0.25–0.5 s stride, reading the centre of each window, with a crossfade.
- **Mixing:** the same over-subtraction, level matching and limiter as offline (§5 lists the code).
- **Prerequisite:** latency reporting. On PipeWire 1.4+, a `module-loopback` front end declaring `ProcessLatency`; on 1.0.5, a `pw_filter` with `PW_FILTER_FLAG_CUSTOM_LATENCY`. See `SIMILAR_PROJECTS_RESEARCH.md` §4.
  - Not yet confirmed: that players actually re-time video in response. Verify with mpv, Firefox and Chromium.
- **CPU (estimate):** overlapping windows multiply compute by window ÷ stride, i.e. 4–12×.
  - Against MRX's ≈ 0.05 RTF on 16 threads, that is RTF ≈ 0.2–0.6 for the effects path alone.
  - On 4 threads it is likely over budget. Measure before committing.
- **Questions the spike must answer:**
  1. Effects leak and keep as a function of window length. Use §2.2's method; 30 s windows are the reference.
  2. CPU at the chosen window and stride, under the live app's thread limits. The ONNX over-subscription fix `ae88ab3` caps threads.
  3. Loudness. MRX expects input at about −27 LUFS, and offline used whole-file integrated loudness. Live needs a slow running estimate; watch for gain pumping.
- **Licence:** MRX code and weights are MIT. Whether the DnR training data's terms carry over to the weights is **unchecked**.

### 4.4 Our own causal "music vs everything else" model (M–L)

The only route to effects-keeping at about 20–50 ms. The plan is in `SIMILAR_PROJECTS_RESEARCH.md` §3.4.

- **What this work adds:**
  - **CDX23 is the better teacher:** −25 dB leak against −16 dB for MRX.
  - **The §2.2 test can serve as the student's acceptance check.**
- **Cheapest start:** retrain GTCRN or DPDFNet with clean = dialogue + effects and noise = music. That gives the same runtime and the same processor class, and needs no app changes.

### 4.5 Browser companion (no realtime constraint)

Covered in `RESEARCH_PRIOR_ART.md` §1.5:

- Buffer the audio ahead of the playhead.
- Run CDX23 or MRX exactly as offline does.
- Play the result through Web Audio, locked to the video.

This gives full offline quality on non-DRM video-on-demand, with zero added latency.

---

## 5. Code and assets to reuse

Everything is in `Music-Assassin` at `71a12d1` or later.

| What | Where | Notes |
|---|---|---|
| MRX inference recipe | `fast_separation._mrx_chunk` | Three STFT resolutions (1024 / 2048 / 8192, hop 256, `normalized=True`, centre, reflect). The ONNX takes the three magnitudes and returns masks `[1, 3, F, T]` in the order **music, speech, sfx**. Masked iSTFTs are **summed** across resolutions. It uses torch STFT; live needs a numpy/torch-free equivalent (the live repo's `processors/stft.py` may cover it). |
| MRX input level | `fast_separation._loudness_gain` | −27 LUFS (pyloudnorm, with an RMS fallback). |
| Music over-subtraction | `fast_separation.clean_effects` | `n_fft` 2048, hop 512, floor 0.05. Defaults per model: `DEFAULT_CLEAN = {"mrx": 1.0, "cdx23": 0.0}`. |
| Effects mix + limiter | `MusicAssassinGUI_v0.4.4.mix_effects` | The block limiter takes each block's gain as the minimum of itself and its neighbours, interpolated between block centres, so no sample can exceed the ceiling. Tested: the ceiling held over 20 random trials. |
| Model downloads + hashes | `fast_separation._MODELS` | MRX int8: `huggingface.co/audiojs/mrx`, 31 MB, sha256 recorded. CDX23: ZFTurbo release `v.1.0.0`, `97d170e1-dbb4db15.th`. |
| Per-model level default | `FX_LEVEL_DEFAULT` in the GUI | MRX → +3 dB, CDX23 → 0 dB. |
| Measurements | `docs/OFFLINE_SPEED_RESEARCH.md`, "Keep sound effects" and "Full pipeline in the app" | |

**Lost, being rebuilt:** the test scripts (`fx_synth.py`, `fx_eval.py`) lived in a session scratchpad that a reboot wiped. §2.2 describes the method in enough detail to rebuild it. As of 2026-10-23, another session is rebuilding it as Tier 2 of `Music-Assassin/quality_harness/` (in progress). A live version still belongs in the live repo's harness.

---

## 6. Do not re-explore

These are already measured or reasoned out, in this repo or the live one:

- **B4's mask shaping (floor, release, exponent) as an effects mechanism.** It preserves the model's own ranking of bins, and drums sit next to effects in that ranking (ROADMAP B4, 2026-10-18 correction).
- **Pure-DSP effects add-back:** plateaus at ~8 % music leakage (ROADMAP §11).
- **StemgenRT for effects:** its stems are drums / bass / vocals. Effects end up in the unmodelled residual alongside melodic music (ROADMAP A2/A4).
- **A Wiener re-filter of the effects with the model's own stems:** under 0.5 dB change (§2.2 here).
- **TIGER-DnR int8 ONNX:** RTF 1.3–3.9, because the export is 38,817 unfused small ops. A re-export with fused normalisation is **untested** and might change this (`OFFLINE_SPEED_RESEARCH.md`).

---

## 7. Decisions needed before route (c)

1. **ROADMAP Q3, the latency ceiling.** Movie mode needs about 0.5–1.5 s, and is only acceptable once latency is reported correctly.
2. **Whether "effects under music" matters enough** to pay for route (c) or (d). Routes (a) and (b) keep effects only where music isn't playing.
3. **CPU budget for an optional mode:** is 2–8 cores acceptable while movie mode is on?
