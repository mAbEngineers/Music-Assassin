# A4: music level (turn the music down, don't delete it)

Measured 2026-10-31 with `python3 quality_harness/run_quality.py --separator fast_mdx --audit a4 --clips real --stereo` and the real-audio clips described in `docs/A2_AUDIT_RESULTS.md` (narration over two real songs' instrumental stretches, music 2 dB and 8 dB under the speech, stereo, 40 s each).

## What it does

After separation the app works out what the separator took out: the original (after the high-pass and normalise) minus the dialogue, and minus the effects stem when "keep sound effects" is on. It adds that residual back after the VAD gate, at a chosen level below the original mix, scaled by the post-process gain the same way the effects stem is. It goes through the same mix stage as the effects (`mix_effects`), so the tail fade and the peak limiter apply to both.

The window offers three choices under **Music**:

| Choice | Level | Default |
|---|---|---|
| Remove it | the music is deleted, as before | yes |
| Turn down a lot | -20 dB | |
| Turn down a little | -12 dB | |

## Results

Mean of the four clips. "Music suppression" and "contrast gain" are in dB, higher means less music left. Dialogue magSNR is how close the dialogue is to the clean speech, higher is closer.

| Setting | Music suppression | Contrast gain | Dialogue magSNR | Musical-noise proxy |
|---|---|---|---|---|
| Remove it | 38.3 | 37.0 | 18.9 | 0.023 |
| -20 dB | 19.6 | 18.4 | 16.7 | 0.024 |
| -12 dB | 11.7 | 10.6 | 14.6 | 0.028 |

- **The control does what it says.** Music suppression drops from 38.3 dB to 19.6 dB at -20 and 11.7 dB at -12. That is within 0.7 dB of the level asked for on every clip (the gap is the post-process gain and the limiter).
- **The cost is on the dialogue score.** Dialogue magSNR falls 2.2 dB at -20 and 4.3 dB at -12. That is expected and is not damage to the voice: the music let back in counts as noise in this metric. It says how much music is audible, not how clean the voice is.
- **The "hides separation artefacts" idea is not confirmed by this proxy.** The musical-noise index barely moves (0.023 to 0.024 to 0.028; it rises slightly, if anything). Whether a quiet bed *sounds* more natural than a hard cut cannot be settled by a number here. It needs a listening session, and that is why "Remove it" stays the default.

## Checked end to end, but not scored

- **Effects kept at the same time as music turned down** (MRX with -12 dB, CDX23 with -20 dB) was run through the real window on a 30 s stereo clip. Both stems are mixed (the log shows `effects ... dB, music ... dB`), the output has the right length, and the limiter stays under 0.1 dB of reduction. The effects stem is subtracted from the music residual so the effects are not counted twice. There is no score for it: the harness does not drive keep-effects.
- A full run of two files through the window (queue, status, Done) and the 20 s preview with the music level on.

## Not tested

- Mono sources and the legacy separators (DEMUCS, MDX-Net, Spleeter). The residual is the same subtraction, but the audit used the fast engine only.
- How it sounds. See above.

## Memory

Turning the control on keeps the pre-separation signal alive until after separation, then subtracts in place, so the cost is one residual array the size of the track until the final mix. Not measured per setting: the harness reports a process-wide high-water mark, which cannot separate the three configs.

## Where the code is

`music_level_db`, `MUSIC_LEVELS`, `set_music()` and the residual block in `_process_single_track`; `mix_effects(..., music=, music_gain=)`. The harness calls the app's own `mix_effects`, so the audit measures the shipped mix stage.
