# A3: stereo output (2026-10-31)

Before A3 every output was mono. Extraction and export both used `ac=1`, so processed videos lost their stereo image. Voc_FT and CDX23 were being fed duplicated mono.

Now a source with two or more channels is processed and exported in stereo; ffmpeg downmixes 5.1 to stereo. A mono source stays mono and goes through exactly the old code path.

Bottom line:
- For **dialogue-only output**, stereo is quality-neutral (measurements below). Centred dialogue gains nothing from the extra channel.
- The real win is **spatial**: with keep-effects on, the effects and ambience now keep their stereo image instead of collapsing to the centre.

## How each stage handles stereo

Arrays are mono `[n]` or stereo `[2, n]`; the pipeline branches on shape.

| Stage | Stereo handling | Why |
|---|---|---|
| Extraction / export | `ac=2` when the source has 2+ channels; the export interleaves one chunk at a time | |
| Prefilter, post-processing, limiter | One shared gain and filter decision for both channels | The image can't shift between channels |
| Voc_FT (fast engine) | Fed real L/R | It is a stereo model, so this costs nothing extra |
| MRX (keep-effects, MRX dialogue) | Masks from the mid (L+R)/2, applied to each channel's STFT | One network pass. Best leak and image of the options below |
| CDX23 (keep-effects, quality) | Runs on the mid. `_share_stems()` then gives each channel's STFT each stem's share of the mid's spectrum | CDX23 is trained on DnR, which is mono. Fed real stereo it leaked 10 dB more music and still returned mono |
| DeepFilterNet and the other enhancers | `stereo_enhance="mid"`: enhance the mid once, then apply its time-frequency gain to each channel (`_transfer_gain`). `"per_channel"` is kept as an option | Same quality as per channel, at about a quarter of the extra cost |
| VAD gate | Detects on the mid; one mask for both channels | |
| Legacy separators (audio-separator, Demucs, Spleeter) | Mono, on the mid | Unchanged paths. With keep-effects on, the effects mixed back in are still stereo |

**Memory.** Stereo is skipped, with a log line, when there isn't enough free memory (`_stereo_fits`). Measured peak RSS with Voc_FT, MRX effects, DeepFilterNet and VAD:

| | Peak RSS |
|---|---|
| Mono | 5.1 GB + 29 MB per minute |
| Stereo | 6.0 GB + 59 MB per minute |

The rule is: stereo needs three stereo copies of the track plus 3 GB free. For a 2-hour film that is about 10.6 GB.

`stereo_output = False` forces the old mono behaviour. There is no UI switch yet.

## Measurements

### Dialogue (harness `--audit a3 --stereo`)

The clips are real speech panned centre, as film dialogue is, over the stereo instrumental stretches of two real songs. The same four clips as A2, now in stereo.

| Config | Contrast gain (dB) | Dialogue magSNR (dB) |
|---|---|---|
| mono (the pre-A3 pipeline, downmixed first) | 32.97 / 34.95 / 38.36 / 41.75 | 19.22 / 17.33 / 19.34 / 17.75 |
| stereo | −0.05 to −0.3 vs mono | identical (±0.003) |
| + DeepFilterNet, mono | +1.3 to +2.0 vs mono | identical |
| + DeepFilterNet, stereo mid-gain | within 0.2 of the mono DeepFilterNet row | identical |
| + DeepFilterNet, stereo per channel | within 0.1 of the mono DeepFilterNet row | identical |

Stereo gives up a fraction of a dB of contrast. A mono downmix discards the music's side channel for free; stereo has to remove it, and Voc_FT nearly does.

### DeepFilterNet cost on stereo (120 s clip, loaded machine)

| Method | Time per 10 min |
|---|---|
| mono | 19 s |
| stereo, mid-gain | 24 s |
| stereo, per channel | 52 s |

This resolves the "stereo's speed catch" item from HANDOVER §7.

The first version of `_transfer_gain` used SciPy's STFT and took as long as DeepFilterNet itself. It now uses torch, with SciPy as a fallback.

### Keep-effects (stereo rebuild of the §2.2 ground-truth test)

The test clip:
- **Effects:** 12 two-second stereo pieces of real episode audio (Charlotte S01E09, seconds where MRX puts at least 70% of the energy in effects). Their side/mid ratio is 0.358.
- **Music:** 90 s of a real stereo anime OP (music plus singing).
- **Level:** the effects sit 6 dB under the music.

The columns:
- **Leak:** effects-output energy outside the inserted pieces, relative to the mix there.
- **Keep:** the effects output's magnitude SNR against the pieces, inside them.
- **Image:** the output's side/mid ratio inside the pieces. The target is 0.358.

| Model | Method | Leak (dB) ↓ | Keep (dB) ↑ | Image | Time (90 s) |
|---|---|---|---|---|---|
| MRX | mono (pre-A3) | −21.8 | 2.12 | 0 | 13.5 s |
| MRX | **mid masks on each channel (shipped)** | −20.9 | 2.18 | 0.359 | 12.3 s |
| MRX | per channel | −18.7 | 2.02 | 0.43 | 22.4 s |
| MRX | stem-share transfer | −21.5 | 2.14 | 0.346 | 12.1 s |
| CDX23 | mono (pre-A3) | −41.4 | 4.52 | 0 | 24.0 s |
| CDX23 | native stereo input | −31.7 | 4.95 | 0.001 | 22.2 s |
| CDX23 | per channel | −38.0 | 4.98 | 0.41 | 48.8 s |
| CDX23 | **stem-share transfer (shipped)** | −42.8 | 4.96 | 0.342 | 23.6 s |

Running a mono-trained model on each channel separately is the worst option for both models. The models judge L or R alone as worse input than the mid.

### End-to-end (the real `_process_single_track`, 40 s episode excerpt)

| Case | Result |
|---|---|
| Stereo video, Best + MRX effects | stereo AAC in MKV |
| Stereo source to M4A, CDX23 effects | stereo AAC |
| Stereo source, `stereo_output=False` | mono |
| Mono source | mono, identical to the forced-mono case |
| Stereo source to WAV, no extras | stereo PCM, channel order and timing verified against the input |

Under load (about 28), the full Best + MRX-effects chain went from 97 s to 118 s per 10 min. Most of the increase is MRX's per-channel STFT work.

`test_processors.py` gained a stereo group: shapes through every stage, dual-mono in gives identical channels out, identity gain transfer, both enhancer modes, and Voc_FT dual-mono equal to the mono path. The mono code path was checked bit-identical to before for Voc_FT, MRX and CDX23.

## Found along the way

- **MKV exports are 1024 samples (23 ms) late.** This is AAC encoder priming, and MKV doesn't compensate for it; M4A does (lag 0). It affects mono output exactly the same way, so it predates A3. It is below the usual lip-sync threshold, but it is a free fix if anyone touches export.
- **The A2 "SFX-leak clip" was synthetic.** `build_sfx_leak_mix()` without arguments synthesises both the bed and the effects. `docs/A2_AUDIT_RESULTS.md` is corrected. The stereo clip above is the real-audio version.

## Open

- **A UI switch for stereo/mono.** Today `_stereo_fits` decides automatically. A user with a large file on a small machine gets mono, with a log line and no visible notice.
- **The legacy separators** could take real stereo too: Demucs `htdemucs` is a stereo model fed `[chunk, chunk]`. This is low priority, because the fast engine is the default.
- **A listening check** of stereo effects on a full episode.
