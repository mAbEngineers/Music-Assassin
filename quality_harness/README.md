# Quality test harness (A1)

A fixed, reproducible way to prove a pipeline change actually helped —
scored against **ground-truth stems we own**, not against a demucs reference
(the structural flaw in the live repo's `bench_quality.py`, which can't fairly
score a model that beats demucs because it treats demucs output as truth).

## What it measures

Per clip, against the clean source stems the corpus builds:

| Metric | Meaning | Good |
|---|---|---|
| `music_suppression_db` | energy removed in music-only regions | higher |
| `dialogue_sisdr_db` | scale-invariant SDR vs clean speech | higher |
| `effects_sisdr_db` / `effects_keep_db` | effects preserved | higher |
| `effects_leak_db` | residual in effects-absent regions (§2.2 leak) | lower |
| `musical_noise` | spectral-warble artefact proxy | lower |
| `rtf` | processing time / audio duration | lower (<1 = realtime) |
| `peak_rss_mb` | process high-water memory | — |

SI-SDR is scale-invariant on purpose: a pipeline that just turns everything
down (A4) does not get a free boost.

## Corpus (`corpus.py`)

- **Tier 1 — synthetic true-stem mixes:** deterministic, seeded, no external
  media. 5 fixed clips: `dialogue_over_music`, `buried_vocals`, `music_only`,
  `effects_over_music`, `busy_mix`.
- **Tier 2 — SFX-leak mix:** rebuild of the lost `fx_synth.py`/`fx_eval.py`
  (`docs/LIVE_SFX_HANDOVER.md` §2.2): 12×2 s effects inserted into a 90 s
  music bed at −6 dB. Pass real wavs to `build_sfx_leak_mix(bed_wav=...,
  effect_wavs=[...])` for a realistic test; omit them to synthesise sources.

## Running

```bash
# DSP-only wiring check — no models, runs anywhere (scipy/numpy only)
python3 quality_harness/run_quality.py --self-check

# A2 audits with the real fast engine (needs torch + openvino/onnxruntime
# + the Voc_FT model, i.e. the env where test_processors.py runs):
python3 quality_harness/run_quality.py --separator fast_mdx --audit a2

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
  For those, feed real stems into `build_sfx_leak_mix`.
