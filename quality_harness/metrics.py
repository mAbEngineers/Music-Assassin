"""Quality metrics for the Music-Assassin offline separation pipeline.

Everything here is pure numpy/scipy and operates on float32 mono (or
per-channel) signals at a known sample rate. No model or pipeline import, so
these functions are cheap to unit-test and safe to reuse across repos --
this is the "shared, honest metrics module" the live repo wanted, built
around ground-truth stems we own rather than a demucs reference.

Conventions
-----------
* Signals are 1-D float32 numpy arrays, same length, same sample rate.
* Region masks are boolean arrays the same length as the signal, True where
  the named source is active. The synthetic corpus (corpus.py) knows these
  exactly because it builds the mix.
* dB values are 10*log10(power ratio). Higher suppression = better music
  removal; higher retention/SI-SDR = better dialogue/effects preservation.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _as_f32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"expected 1-D signal, got shape {x.shape}")
    return x


def _power(x: np.ndarray) -> float:
    """Mean power (energy per sample) of a signal."""
    if x.size == 0:
        return 0.0
    return float(np.mean(np.square(x, dtype=np.float64)))


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio, in dB.

    Standard definition (Le Roux et al. 2019): project the estimate onto the
    reference, treat the residual as noise. Invariant to overall gain, so it
    is not fooled by a pipeline that simply turns everything down.

    Returns +inf for a perfect match and a large negative number when the
    estimate is essentially uncorrelated with the reference.
    """
    est = _as_f32(estimate)
    ref = _as_f32(reference)
    if est.shape != ref.shape:
        raise ValueError(f"shape mismatch: {est.shape} vs {ref.shape}")

    ref64 = ref.astype(np.float64)
    est64 = est.astype(np.float64)
    ref_energy = float(np.dot(ref64, ref64)) + _EPS
    alpha = float(np.dot(est64, ref64)) / ref_energy
    target = alpha * ref64
    noise = est64 - target

    target_p = float(np.dot(target, target))
    noise_p = float(np.dot(noise, noise))
    if noise_p <= _EPS:
        return float("inf")
    return 10.0 * np.log10((target_p + _EPS) / (noise_p + _EPS))


def magnitude_snr(
    estimate: np.ndarray,
    reference: np.ndarray,
    region_mask: np.ndarray | None = None,
    frame: int = 2048,
    hop: int = 512,
) -> float:
    """Phase-blind fidelity: STFT-magnitude SNR in dB (higher = better).

    SI-SDR compares waveforms, so it heavily penalises the phase shift of the
    app's causal IIR filters (40 Hz high-pass, presence EQ, 10 kHz low-pass)
    even though phase shift is inaudible: clean speech through the 40 Hz
    high-pass alone scores ~8 dB SI-SDR but ~40 dB here. This compares
    magnitude spectrograms instead, after the best overall gain, over frames
    whose centre falls in `region_mask`. Use it to judge audible damage;
    keep SI-SDR for separation-only comparisons.
    """
    est = _as_f32(estimate)
    ref = _as_f32(reference)
    if est.shape != ref.shape:
        raise ValueError(f"shape mismatch: {est.shape} vs {ref.shape}")
    if est.size < frame:
        return float("nan")

    window = np.hanning(frame).astype(np.float32)
    starts = np.arange(0, est.size - frame + 1, hop)
    if region_mask is not None:
        mask = np.asarray(region_mask, dtype=bool)
        if mask.shape != est.shape:
            raise ValueError("region_mask must match signal shape")
        starts = starts[mask[starts + frame // 2]]
    if starts.size == 0:
        return float("nan")

    idx = starts[:, None] + np.arange(frame)[None, :]
    e = np.abs(np.fft.rfft(est[idx] * window, axis=1)).astype(np.float64)
    r = np.abs(np.fft.rfft(ref[idx] * window, axis=1)).astype(np.float64)
    gain = float(np.sum(e * r)) / (float(np.sum(e * e)) + _EPS)
    noise = float(np.sum((gain * e - r) ** 2))
    if noise <= _EPS:
        return float("inf")
    return 10.0 * np.log10((float(np.sum(r * r)) + _EPS) / (noise + _EPS))


def music_suppression_db(
    output: np.ndarray,
    input_mix: np.ndarray,
    music_mask: np.ndarray,
) -> float:
    """How much music energy the pipeline removed, in dB (higher = better).

    Measured over regions where music is the ONLY active source, so any
    energy surviving in the output there is music leak. We compare output
    power to input power in those regions:

        suppression = 10*log10(P_in / P_out)

    0 dB = nothing removed; +20 dB = music reduced to 1% power. This is the
    honest leak number because the regions contain no speech or effects to
    confound it -- the synthetic corpus guarantees that.
    """
    out = _as_f32(output)
    mix = _as_f32(input_mix)
    mask = np.asarray(music_mask, dtype=bool)
    if not (out.shape == mix.shape == mask.shape):
        raise ValueError("output, input_mix and music_mask must share shape")
    if not mask.any():
        return float("nan")

    p_in = _power(mix[mask])
    p_out = _power(out[mask])
    return 10.0 * np.log10((p_in + _EPS) / (p_out + _EPS))


def contrast_gain_db(
    output: np.ndarray,
    input_mix: np.ndarray,
    speech_mask: np.ndarray,
    music_mask: np.ndarray,
) -> float:
    """Gain-invariant music removal: how much the pipeline raised the
    dialogue-to-music ratio, in dB (higher = better).

        contrast = (P_out[speech] / P_out[music-only])
                 / (P_in[speech]  / P_in[music-only])

    `music_suppression_db` compares absolute output and input power, so the
    app's global peak-normalise (prefilter and post-process both rescale to
    0.98) moves it: a separator that also guts the dialogue leaves a quiet
    output that normalisation boosts, and its music "suppression" drops.
    This ratio cancels any overall gain. It is NaN for a clip with no speech
    or no music-only region.
    """
    out = _as_f32(output)
    mix = _as_f32(input_mix)
    s = np.asarray(speech_mask, dtype=bool)
    m = np.asarray(music_mask, dtype=bool)
    if not (out.shape == mix.shape == s.shape == m.shape):
        raise ValueError("signals and masks must share shape")
    if not (s.any() and m.any()):
        return float("nan")
    ratio_out = (_power(out[s]) + _EPS) / (_power(out[m]) + _EPS)
    ratio_in = (_power(mix[s]) + _EPS) / (_power(mix[m]) + _EPS)
    return 10.0 * np.log10(ratio_out / ratio_in)


def retention_db(
    output: np.ndarray,
    reference_stem: np.ndarray,
    region_mask: np.ndarray | None = None,
) -> float:
    """Energy retained relative to the clean stem, in dB (0 = kept exactly).

    For dialogue and effects we care that the wanted source survives. This
    compares output power to clean-stem power over the region where that
    source is active:

        retention = 10*log10(P_out / P_stem)

    ~0 dB means the source came through at its original level. Strongly
    negative means it was eaten. (Positive is possible if other sources leak
    into the region -- pair this with SI-SDR, which penalises that.)
    """
    out = _as_f32(output)
    stem = _as_f32(reference_stem)
    if out.shape != stem.shape:
        raise ValueError(f"shape mismatch: {out.shape} vs {stem.shape}")
    if region_mask is None:
        mask = np.ones(out.shape, dtype=bool)
    else:
        mask = np.asarray(region_mask, dtype=bool)
        if mask.shape != out.shape:
            raise ValueError("region_mask must match signal shape")
    if not mask.any():
        return float("nan")

    p_out = _power(out[mask])
    p_stem = _power(stem[mask])
    return 10.0 * np.log10((p_out + _EPS) / (p_stem + _EPS))


def leak_energy_db(
    output: np.ndarray,
    quiet_mask: np.ndarray,
    reference_level: np.ndarray | None = None,
) -> float:
    """Residual energy in regions that should be silent, in dB.

    This is the §2.2 SFX-leak metric generalised: over regions where the
    wanted source (effects, or speech) is absent, any output energy is leak
    from whatever we were trying to remove. Reported relative to the overall
    reference level so it is comparable across clips.

    Lower (more negative) is better. If `reference_level` is given, it is the
    signal whose full-clip RMS we normalise against (typically the clean
    wanted stem); otherwise we report absolute dBFS-style power.
    """
    out = _as_f32(output)
    mask = np.asarray(quiet_mask, dtype=bool)
    if mask.shape != out.shape:
        raise ValueError("quiet_mask must match signal shape")
    if not mask.any():
        return float("nan")

    p_leak = _power(out[mask])
    if reference_level is None:
        ref_p = 1.0
    else:
        ref = _as_f32(reference_level)
        ref_p = _power(ref) + _EPS
    return 10.0 * np.log10((p_leak + _EPS) / ref_p)


def musical_noise_index(
    output: np.ndarray,
    sr: int,
    frame: int = 1024,
    hop: int = 512,
) -> float:
    """Proxy for "musical noise" / spectral warble artefacts (lower = cleaner).

    Separation leaves isolated, flickering time-frequency bins that the ear
    hears as burbling. We estimate it as the mean temporal variation of the
    log-magnitude spectrogram (kurtosis-weighted), which rises when energy
    appears and vanishes in scattered bins rather than moving smoothly.

    This is a relative indicator for A/B comparisons, not a calibrated score.
    """
    out = _as_f32(output)
    if out.size < frame:
        return float("nan")

    window = np.hanning(frame).astype(np.float32)
    n_frames = 1 + (out.size - frame) // hop
    if n_frames < 3:
        return float("nan")

    spec = np.empty((n_frames, frame // 2 + 1), dtype=np.float64)
    for i in range(n_frames):
        seg = out[i * hop : i * hop + frame] * window
        mag = np.abs(np.fft.rfft(seg))
        spec[i] = np.log1p(mag)

    # Frame-to-frame flux of log-magnitude; musical noise = high, spiky flux.
    flux = np.diff(spec, axis=0)
    per_bin_var = np.var(flux, axis=0)
    return float(np.mean(per_bin_var))


def rtf(processing_seconds: float, audio_seconds: float) -> float:
    """Real-time factor: processing time / audio duration (lower = faster).

    < 1.0 means faster than real time. The offline budget referenced in the
    roadmap is a ~60 s total for a clip, so this plus the clip length tells
    you whether a change (e.g. per-channel DeepFilter for stereo) blows it.
    """
    if audio_seconds <= 0:
        return float("nan")
    return processing_seconds / audio_seconds


def summarise(scores: dict[str, float]) -> dict[str, float]:
    """Round a score dict for stable, diff-friendly report output."""
    out: dict[str, float] = {}
    for k, v in scores.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[k] = float("nan")
        elif isinstance(v, float) and np.isinf(v):
            out[k] = v
        else:
            out[k] = round(float(v), 3)
    return out
