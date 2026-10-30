# A2 audit results (2026-10-30)

The A2 audits from HANDOVER §7 (A2-q, A2-s), run through `quality_harness/` with the fast engine (Voc_FT on OpenVINO). They measure each quality-shaping stage by turning it on or off and holding everything else.

Bottom line:
- The pre-gate is now off. It gained nothing at normal levels and damaged dialogue in quiet files.
- DeepFilterNet is the only stage that clearly helps.
- Every other default stays as it was.

## What was measured

**Tier 3 (new, real audio).** One clean narration recording (TTS, 24 kHz source) over the instrumental-only stretches of two real songs: an anime OP (dense pop) and a nasheed (acoustic). The instrumental stretches were found where each song's separated vocal stem is at least 25 dB below the song. Each mix is 40 s long, with 6 s phrases separated by 2 s of music alone, and is tested at two levels:
- typical: music 8 dB under the speech;
- loud: music 2 dB under the speech.

All the numbers below come from these four clips, because the synthetic tier turned out to be unusable for dialogue (see "Harness fixes").

**Metrics.** Both were added for this audit:
- **Contrast gain:** how much the pipeline raised the dialogue-to-music ratio. It ignores overall level.
- **Dialogue magSNR:** STFT-magnitude fidelity to the clean speech. It ignores phase.

Higher is better for both.

Reproduce with:

```
python quality_harness/run_quality.py --separator fast_mdx --audit a2 --clips real \
  --real-speech speech.wav \
  --real-song song1.wav --real-vocals song1-vocals.wav \
  --real-song song2.wav --real-vocals song2-vocals.wav
```

## Results

Each cell is the change from the baseline. The baseline is the app's defaults: Balanced mode, no enhancer, VAD off, overlap 0.25. The range covers all four real clips.

| Stage | Contrast gain | Dialogue magSNR | Time (RTF) | Verdict |
|---|---|---|---|---|
| Baseline (absolute) | 33.6 to 41.5 dB | 17.3 to 19.3 dB | 0.07 | Music is reduced well over 30 dB relative to dialogue |
| Pre-gate off | −0.6 to +0.3 | ±0.03 | same | **Gate removed** (see the level sweep below) |
| Overlap 0.10 | −0.8 to +0.4 | ±0.06 | 0.05–0.06 | Keep 0.25 (see "Overlap" below) |
| Normal mode (10 kHz low-pass) | +0.4 to +0.6 | −0.4 to −0.6 | same | Cuts the top octave of the voice to remove a little more music. A wash; no change |
| Post-processing off | −0.4 to −1.0 | +2.3 to +3.9 | same | Balanced mode is mainly tonal EQ, not cleanup (see below) |
| VAD gate | +0.2 to +0.35 | 0 | +0.01 | Small help at small cost. Keep it as an option |
| DeepFilterNet | +1.2 to +2.2 | 0 | +0.02–0.03 | **Helps, with no dialogue damage** |
| Best (DeepFilterNet + VAD) | +1.3 to +2.3 | 0 | +0.03 | As DeepFilterNet, plus a little |

### The −40 dB pre-gate

The gate zeroed every sample below −40 dBFS. It ran *before* the prefilter normalises the level, so how hard it hit depended on how loud the file was. To test this, real1_typical was run through the fast engine with the gate on and off at four input levels:

| Input level | Samples zeroed | Dialogue magSNR, gate on | Dialogue magSNR, gate off |
|---|---|---|---|
| 0 dB | 9.5% | 19.21 | 19.22 |
| −12 dB | 37.7% | 18.82 | 19.22 |
| −24 dB | 89.7% | 9.39 | 19.22 |
| −30 dB | 97.6% | 2.91 | 19.22 |

With the gate off, the result does not change with input level. With it on, dialogue in a quiet file is badly damaged. The suspected crossover distortion is real, and it is level-dependent. At full scale the gate bought +0.6 dB of contrast at most.

The dialogue path now calls `audio_prefilter(..., gate=False)`, the same as the keep-effects path, and the two paths share one prefiltered array. The function's default is now `gate=False`.

### The Normal-mode 10 kHz low-pass

Voice energy above 10 kHz, relative to the whole voice:

| Voice source | Energy above 10 kHz |
|---|---|
| The sibilant TTS narration | −20 dB |
| The two sung vocal stems | −28 dB and −43 dB |

So the low-pass removes the "air" band, but not much energy. In return, music leaking above 10 kHz drops a further 7 dB (−18.4 dB in Balanced, −25.6 dB in Normal), and the 300 Hz–8 kHz speech band is untouched.

There is no measurable harm to intelligibility. Whether losing the air matters is a listening call, which belongs in the R7.2 listening session.

### Overlap 0.25 vs 0.10

On the real dialogue clips, 0.10 is quality-neutral and cuts total processing time by about 20%. But on the harness's SFX-leak clip the level-dependent metrics got clearly worse. (Correction, 2026-10-31: run without arguments, `build_sfx_leak_mix()` synthesises both the music bed and the effects. This was the synthetic stand-in, not the real OP and episode effects of §2.2, so it is weak evidence.)
- music suppression fell from 21.3 to 13.8 dB;
- effects leak rose from 6.4 to 13.9 dB.

The synthetic clips lost about 2 dB of contrast. Keep 0.25 until a longer real clip with dense music settles it.

### Balanced post-processing

Turning post-processing off *raises* dialogue magSNR by 2.3–3.9 dB and lowers contrast by under 1 dB. Balanced mode's 1–3 kHz presence boost and de-esser colour the voice; they don't remove music.

That may still be a good-sounding choice (presence aids intelligibility), so this is not a defect. It is a question for the listening session, not a reason to change the default.

## Harness fixes made along the way

- **SI-SDR was measuring phase, not damage.** The app's filters are causal IIR filters: the 40 Hz high-pass, the presence EQ and the 10 kHz low-pass. Their phase shift is inaudible, but SI-SDR punishes it hard:
  - the fast engine alone scores 19.9 dB SI-SDR;
  - after the stock filters, the same output scores 0.7 dB;
  - clean speech through only the 40 Hz high-pass scores 8 dB SI-SDR, but 40 dB magSNR.

  Added `metrics.magnitude_snr`. Compare SI-SDR only between configs that use the same filters.
- **Music suppression moved with loudness.** The app peak-normalises twice (prefilter and post-process). A separator that also removes the dialogue leaves a quiet output, normalisation boosts it, and "suppression" drops. Added `metrics.contrast_gain_db`, which ignores overall level.
- **The synthetic tier can't judge dialogue with a real model.** Voc_FT doesn't treat the synthetic harmonic "speech" as a voice and removes it (dialogue magSNR about 0.25 dB in every config). Keep that tier for the DSP-only self-check.
- **Effects columns are not meaningful for A2.** The harness drives the vocal-only path, not the keep-effects stage, so effects columns in an A2 table only show that the vocal path drops effects.

## Environment notes

- The shared build venv (`~/.venvs/music-assassin`) has numpy 1.26.4 with SciPy 1.18.1, and `scipy.signal` fails to import there. The audits ran in a throwaway overlay venv instead:
  - `python -m venv`;
  - `pip install numpy==2.4.6`;
  - a `.pth` file pointing at the shared venv's site-packages.
- In the same env, `test_processors.py` passes 13 of 15 (one is skipped: Spleeter is not installed). The failure is the audio-separator MDX-Net test, with `operator torchvision::nms does not exist`: a torch/torchvision mismatch in the shared venv, unrelated to these changes.
- RTF numbers are from a busy shared machine; only compare them within this table.
