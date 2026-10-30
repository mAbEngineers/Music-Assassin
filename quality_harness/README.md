# Quality test harness (A1)

A fixed, reproducible way to prove a pipeline change actually helped —
scored against **ground-truth stems we own**, not against a demucs reference
(the structural flaw in the live repo's `bench_quality.py`, which can't fairly
score a model that beats demucs because it treats demucs output as truth).

## What it measures

Per clip, against the clean source stems the corpus builds:

| Metric | Meaning | Good |
|---|---|---|
| `contrast_gain_db` | rise in the dialogue-to-music ratio; gain-invariant, the music-removal number to trust | higher |
| `music_suppression_db` | energy removed in music-only regions (moves with the app's peak-normalise) | higher |
| `dialogue_mag_snr_db` | phase-blind magnitude fidelity vs clean speech; the dialogue-damage number to trust | higher |
| `dialogue_sisdr_db` | scale-invariant SDR vs clean speech (also punishes inaudible IIR phase shift) | higher |
| `effects_sisdr_db` / `effects_keep_db` | effects preserved | higher |
| `effects_leak_db` | residual in effects-absent regions (§2.2 leak) | lower |
| `musical_noise` | spectral-warble artefact proxy | lower |
| `rtf` | processing time / audio duration | lower (<1 = realtime) |
| `peak_rss_mb` | process high-water memory | — |

SI-SDR is scale-invariant on purpose: a pipeline that just turns everything
down (A4) does not get a free boost. But it compares waveforms, so the phase
shift of the app's causal filters costs it 10+ dB with no audible change; see
`docs/A2_AUDIT_RESULTS.md`. Judge dialogue by magSNR and music removal by
contrast gain. Stereo clips are scored per channel and averaged.

## Corpus (`corpus.py`)

- **Tier 1 — synthetic true-stem mixes:** deterministic, seeded, no external
  media. 5 fixed clips: `dialogue_over_music`, `buried_vocals`, `music_only`,
  `effects_over_music`, `busy_mix`.
- **Tier 2 — SFX-leak mix:** rebuild of the lost `fx_synth.py`/`fx_eval.py`
  (`docs/LIVE_SFX_HANDOVER.md` §2.2): 12×2 s effects inserted into a 90 s
  music bed at −6 dB. Pass real wavs to `build_sfx_leak_mix(bed_wav=...,
  effect_wavs=[...])` for a realistic test; omit them to synthesise sources
  (the default run uses the synthetic version).
- **Tier 3 — real dialogue:** clean speech over the instrumental-only
  stretches of real songs (found via each song's separated vocal stem), 40 s,
  music 8 dB and 2 dB under the speech. Mono, or stereo with `--stereo`
  (speech centred, the songs' own stereo beds). Pass wavs with
  `--real-speech`, `--real-song`/`--real-vocals` (repeatable pairs).

## Running

```bash
# DSP-only wiring check — no models, runs anywhere (scipy/numpy only)
python3 quality_harness/run_quality.py --self-check

# A2 audits with the real fast engine (needs torch + openvino/onnxruntime
# + the Voc_FT model, i.e. the env where test_processors.py runs):
python3 quality_harness/run_quality.py --separator fast_mdx --audit a2

# A3 stereo: mono pipeline vs stereo, DeepFilterNet mid-gain vs per channel
python3 quality_harness/run_quality.py --separator fast_mdx --audit a3 --clips real --stereo \
  --real-speech speech.wav --real-song song.wav --real-vocals song-vocals.wav

# A4 "turn down, don't delete" music-mixback sweep:
python3 quality_harness/run_quality.py --separator fast_mdx --audit a4
```

Each run writes a timestamped `*.md` (the before/after artefact) and `*.json`
(for diffing across commits) under `quality_harness/reports/`.

## How it couples to the app

The shipping `_process_single_track` is GUI-coupled (does its own ffmpeg
extract/export). The harness instead loads the app module headlessly
(`pipeline.load_app`, same tkinter-mock trick as `test_processors.py`, plus an
ffmpeg stub since the stages never call it) and drives the stages directly on
a numpy array: `audio_prefilter → separate → advanced_postprocess_chunked →
apply_enhancer → apply_vad_gate → mix_effects`. A `StageConfig` turns any
single stage on/off, which is exactly what the A2 audits toggle.

## Known caveats

- DSP-only (`separator=none`) shows ~0 dB music suppression by design — DSP
  can't remove music; only separation can. Use it only to check wiring.
- The synthetic sources are proxies (harmonic buzz / chord pad / transients),
  good for regression and relative A/B, not for absolute perceptual claims.
  With a real separator they are worse than that: Voc_FT does not treat the
  synthetic "speech" as voice and removes it, so judge dialogue on Tier 3.
- The harness drives the vocal-only path. Keep-effects (MRX/CDX23) is not in
  `run_config`; A3 tested it with a stereo §2.2 rebuild (`docs/A3_STEREO_RESULTS.md`).
