# Speed: overlap recheck and the R1 model screen

Measured 2026-10-31. Two questions from `HANDOVER.md` §7: does overlap 0.1 still cost quality on a longer, music-heavy clip (A2-s), and is any of the R1 candidates faster than what the app already runs.

**Read the caveat first.** This is not the 4-core box. It is a 32-core machine that was shared with someone else's job (load average about 33, one process at about 2000% CPU). The 4-core figures below pin the process to four cores with `taskset -c 0-3` and `OMP_NUM_THREADS=4`; that matches the core count but not the other box's memory bandwidth or clocks, and the neighbouring job still competes for those cores. `docs/ROADMAP.md` §1.4 says the 4-core machine produces the numbers that go in a comparison table, so treat these as an emulated screen: good for ordering and ratios, not for citing as absolute RTF.

## A2-s: overlap 0.25 against 0.10

Eight 3-minute stereo clips (`quality_harness`, fast engine, Balanced, no enhancer): real narration over a music bed, music 2 dB and 8 dB under the speech.

- **Beds from instrumental stretches** (`long1`, `long2`): 8 s and 12 s of instrumental-only music from each song, looped to 3 minutes.
- **Whole songs with the singing left in** (`song1`, `song2`): the full track as the music. Much harder: the separator has to take a sung voice out from under the speech. Contrast gain is low for that reason (2 to 7 dB), at either overlap.

Change from overlap 0.25 to 0.10:

| Clip | Contrast gain | Dialogue magSNR | Time |
|---|---|---|---|
| long1, music -2 dB | **-0.90 dB** | -0.05 dB | -29% |
| long1, music -8 dB | +0.03 dB | -0.01 dB | -20% |
| long2, music -2 dB | -0.10 dB | +0.02 dB | -14% |
| long2, music -8 dB | -0.13 dB | +0.02 dB | -14% |
| song1, music -2 dB | -0.02 dB | +0.01 dB | -25% |
| song1, music -8 dB | +0.06 dB | +0.03 dB | -12% |
| song2, music -2 dB | 0.00 dB | 0.00 dB | -17% |
| song2, music -8 dB | 0.00 dB | +0.01 dB | -10% |

- **Quality:** neutral. Dialogue magSNR moves by 0.06 dB or less everywhere. Contrast gain is within 0.15 dB on seven of eight clips; the exception is `long1` with loud music, 0.9 dB lower. That is one clip, so it is a flag, not a finding.
- **Speed:** 10 to 29% faster in these runs (19% on average for the first four clips, 16% for the song clips). The times come from the shared machine and are noisy; the same change measured separately on 4 pinned cores, warmed up, was 0.214 to 0.221 RTF at 0.25 against 0.172 to 0.174 at 0.10, about 20% (see below).
- **What this does not clear:** the earlier A2 result on the sound-effects-leak clip, where 0.10 lost about 7.5 dB of music suppression and let more effects through. These clips contain no effects, so they cannot contradict it.

**Recommendation:** keep the default at 0.25. The window already offers 0.10 as **Quality: Faster**, which is the right home for it: neutral on dialogue over music, a visible loss only where sound effects are involved. Making it the default is a one-line change if you would rather have the 20%, but it would trade away the one case where it is measurably worse.

## Fast engine against the generic MDX path, 4 cores

60 s stereo clip, warmed up, Voc_FT in both:

| Path | RTF | Per 10 min of audio |
|---|---|---|
| `fast_separation` (OpenVINO), overlap 0.25 | 0.21 to 0.22 | 128 to 133 s |
| `fast_separation`, overlap 0.10 | 0.17 | 103 to 105 s |
| `audio-separator` (onnxruntime), its default | 0.33 to 0.34 | 197 to 202 s |

So on four cores the fast engine's separation step is about 1.5 times faster than the generic path. The 58 s per 10 minutes in the README is an end-to-end figure on 16 cores; on four cores expect separation alone to take about two minutes, before the other stages. The first call is slower (model load and compile): 19 s instead of 13 s for the same clip.

## R1: speed screen

Each model separates the same 60 s stereo clip through `audio-separator` 0.44.5 on four pinned cores, twice, interleaved with the others. Model download time is not counted.

| Model | RTF (two runs) | Per 10 min | Peak RSS |
|---|---|---|---|
| UVR-MDX-NET-Voc_FT.onnx (the baseline, today's Voc_FT) | 0.329, 0.336 | 197 to 202 s | 3.3 GB |
| UVR-MDX-NET-Inst_HQ_4.onnx | 0.275, 0.295 | 165 to 177 s | 2.8 GB |
| kuielab_a_vocals.onnx | 0.371, 0.377 | 222 to 226 s | 1.1 GB |
| 2_HP-UVR.pth | 0.714, 0.714 | 428 s | 1.9 GB |

- **No candidate is a step change.** Inst_HQ_4 is 12 to 16% faster than Voc_FT on the same runtime, which is inside what a different graph size explains, and it is an instrumental model: its vocals are the mix minus its instrumental, so its quality as a dialogue extractor is unknown and R1 does not test it.
- **`2_HP-UVR` is not 2 times faster here; it is 2 times slower.** The maintainer's 19 s against 36 s for a 3-minute song was an Apple M3 Max, not a CPU-only run. The hypothesis in `ROADMAP.md` R1 does not hold on this hardware.
- **kuielab_a is the smallest in memory** (1.1 GB against 3.3 GB) and slower than Voc_FT; only interesting if RAM, not time, is the limit.
- **Not tried:** running Inst_HQ_4 through `fast_separation`'s OpenVINO path, which is where Voc_FT gets its 1.5 times. It might land near 0.15 RTF at four cores. That is a guess, and it is the only follow-up from this screen worth the effort; quality would need the harness, with the roadmap's H1 reference caveat.

R1 closes as: no model in this set beats the current engine on speed on CPU. R1 is a speed screen and says nothing about quality.
