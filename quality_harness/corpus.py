"""Fixed test corpus for the quality harness.

Two tiers, both giving us ground-truth stems we own -- the whole point, so
leak and retention are scored against known-clean sources rather than against
a demucs reference (the flaw in the sibling repo's bench_quality.py).

Tier 1 -- synthetic true-stem mixes
    Deterministic, seeded, dependency-light. Speech-like, music-like and
    effects-like sources with exact per-sample activity masks. No external
    media, so the test set is reproducible on any machine and in CI.

Tier 2 -- SFX-leak mix (rebuild of the lost fx_synth.py / fx_eval.py)
    Per docs/LIVE_SFX_HANDOVER.md §2.2: 12 two-second effects clips inserted
    into a ~90 s music-only bed at -6 dB. Leak = output energy outside the
    inserts; Keep = SI-SDR inside them. `build_sfx_leak_mix` accepts real
    effect/bed wavs when you have them and falls back to synthesised sources
    so the test runs with nothing downloaded.

A `Clip` bundles the mix with the stems and masks the metrics need.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

SR = 44100  # matches the app's fixed sample_rate


@dataclasses.dataclass
class Clip:
    """One test item: the mix plus the ground truth to score against."""

    name: str
    sr: int
    mix: np.ndarray          # what the pipeline receives: mono [n] or stereo [2, n] float32
    speech: np.ndarray       # clean speech stem (same shape as mix)
    music: np.ndarray        # clean music stem (same shape as mix)
    effects: np.ndarray      # clean effects stem (same shape as mix)
    speech_mask: np.ndarray  # True where speech is active (always [n])
    music_mask: np.ndarray   # True where music is active
    effects_mask: np.ndarray  # True where effects are active

    @property
    def music_only_mask(self) -> np.ndarray:
        """Regions with music and nothing else -- the honest leak window."""
        return self.music_mask & ~self.speech_mask & ~self.effects_mask

    @property
    def duration_s(self) -> float:
        return self.mix.shape[-1] / self.sr


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _normalise(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m < 1e-9:
        return x.astype(np.float32)
    return (x * (peak / m)).astype(np.float32)


def _db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


# --------------------------------------------------------------------------
# Synthetic source generators (ground-truth stems).
# --------------------------------------------------------------------------

def _synth_speech(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Formant-modulated harmonic buzz gated into syllable-like bursts."""
    t = np.arange(n) / sr
    f0 = 110.0 + 20.0 * np.sin(2 * np.pi * 0.7 * t)  # pitch wobble
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = np.zeros(n, dtype=np.float64)
    for k, amp in enumerate((1.0, 0.6, 0.4, 0.25, 0.15), start=1):
        sig += amp * np.sin(k * phase)
    # Formant-ish shaping: two resonant bands via smooth gain over time.
    sig *= 0.8 + 0.2 * np.sin(2 * np.pi * 3.0 * t)
    # Syllable envelope: ~4 Hz bursts with gaps (so there are speech-absent gaps).
    env = (np.sin(2 * np.pi * 3.5 * t) > -0.2).astype(np.float64)
    env *= 0.5 * (1 + np.sin(2 * np.pi * 0.4 * t))  # phrase-level swell/pauses
    sig *= env
    return _normalise(sig, 0.7)


def _synth_music(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Chord pad + arpeggio + a steady beat -- continuous, fills the clip."""
    t = np.arange(n) / sr
    root = 220.0
    chord = [1.0, 1.26, 1.5, 2.0]  # root, maj3, 5th, octave ratios
    sig = np.zeros(n, dtype=np.float64)
    for r in chord:
        sig += 0.25 * np.sin(2 * np.pi * root * r * t)
    # Arpeggio on top, switching note every ~0.5 s.
    step = int(0.5 * sr)
    arp_ratios = [2.0, 2.52, 3.0, 2.52]
    for i in range(0, n, step):
        r = arp_ratios[(i // step) % len(arp_ratios)]
        seg = slice(i, min(i + step, n))
        tt = t[seg]
        sig[seg] += 0.2 * np.sin(2 * np.pi * root * r * tt) * np.hanning(tt.size)
    # Kick on every beat (~2 Hz).
    beat = int(0.5 * sr)
    for i in range(0, n, beat):
        env = np.exp(-np.arange(min(beat, n - i)) / (0.08 * sr))
        sig[i : i + env.size] += 0.4 * env * np.sin(
            2 * np.pi * 60 * np.arange(env.size) / sr
        )
    return _normalise(sig, 0.8)


def _synth_effects(
    n: int, sr: int, rng: np.random.Generator, n_events: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse transients/whooshes. Returns (signal, activity mask)."""
    sig = np.zeros(n, dtype=np.float64)
    mask = np.zeros(n, dtype=bool)
    for _ in range(n_events):
        dur = int(rng.uniform(0.15, 0.6) * sr)
        start = int(rng.uniform(0, max(1, n - dur)))
        seg = slice(start, start + dur)
        kind = rng.integers(0, 3)
        env = np.exp(-np.arange(dur) / (0.12 * sr))
        if kind == 0:  # click/impact: filtered noise burst
            ev = rng.standard_normal(dur) * env
        elif kind == 1:  # whoosh: swept noise
            ev = rng.standard_normal(dur) * np.hanning(dur)
        else:  # tonal beep
            f = rng.uniform(400, 2000)
            ev = np.sin(2 * np.pi * f * np.arange(dur) / sr) * env
        sig[seg] += 0.6 * ev
        mask[seg] = True
    return _normalise(sig, 0.7), mask


def _activity_mask(x: np.ndarray, sr: int, thresh_db: float = -45.0) -> np.ndarray:
    """Short-time energy mask: True where the stem is audibly active."""
    win = max(1, int(0.02 * sr))
    power = np.convolve(x.astype(np.float64) ** 2, np.ones(win) / win, mode="same")
    ref = float(np.max(power)) + 1e-12
    return (10.0 * np.log10(power / ref + 1e-12)) > thresh_db


# --------------------------------------------------------------------------
# Tier 1: synthetic true-stem mixes.
# --------------------------------------------------------------------------

def make_synthetic_clip(
    name: str,
    duration_s: float = 20.0,
    sr: int = SR,
    seed: int = 0,
    music_db: float = -3.0,
    speech_db: float = -6.0,
    effects_db: float = -4.0,
    with_speech: bool = True,
    with_effects: bool = True,
) -> Clip:
    """Build one deterministic true-stem mix.

    Levels are relative gains (dB) applied before summing, so you can dial a
    buried-vocals case (quiet speech under loud music) or a music-only case.
    """
    rng = _rng(seed)
    n = int(duration_s * sr)

    music = _synth_music(n, sr, rng) * _db_to_lin(music_db)
    music_mask = np.ones(n, dtype=bool)  # continuous by construction

    if with_speech:
        speech = _synth_speech(n, sr, rng) * _db_to_lin(speech_db)
        speech_mask = _activity_mask(speech, sr)
    else:
        speech = np.zeros(n, dtype=np.float32)
        speech_mask = np.zeros(n, dtype=bool)

    if with_effects:
        fx, fx_mask = _synth_effects(n, sr, rng)
        effects = fx * _db_to_lin(effects_db)
        effects_mask = fx_mask
    else:
        effects = np.zeros(n, dtype=np.float32)
        effects_mask = np.zeros(n, dtype=bool)

    mix = _normalise(music + speech + effects, 0.95)
    return Clip(
        name=name,
        sr=sr,
        mix=mix,
        speech=speech.astype(np.float32),
        music=music.astype(np.float32),
        effects=effects.astype(np.float32),
        speech_mask=speech_mask,
        music_mask=music_mask,
        effects_mask=effects_mask,
    )


def default_synthetic_set() -> list[Clip]:
    """The fixed Tier-1 corpus. Seeds are pinned so results are comparable."""
    return [
        make_synthetic_clip("dialogue_over_music", seed=1),
        make_synthetic_clip(
            "buried_vocals", seed=2, speech_db=-14.0, music_db=-1.0
        ),
        make_synthetic_clip(
            "music_only", seed=3, with_speech=False, with_effects=False
        ),
        make_synthetic_clip(
            "effects_over_music", seed=4, with_speech=False, effects_db=-2.0
        ),
        make_synthetic_clip("busy_mix", seed=5, effects_db=-2.0, speech_db=-4.0),
    ]


# --------------------------------------------------------------------------
# Tier 2: SFX-leak mix (rebuild of the lost fx_synth/fx_eval, §2.2).
# --------------------------------------------------------------------------

def _load_wav_mono(path: Path, sr: int) -> np.ndarray:
    """Load a wav as mono float32 at `sr`. Requires soundfile + scipy."""
    return _load_wav(path, sr, stereo=False)


def _load_wav(path: Path, sr: int, stereo: bool = False) -> np.ndarray:
    """Load a wav at `sr` as mono [n] (channel mean) or stereo [2, n] (a mono
    file is duplicated). Requires soundfile + scipy."""
    import soundfile as sf
    from scipy.signal import resample_poly

    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if stereo:
        x = data.T[:2] if data.shape[1] >= 2 else np.repeat(data.T, 2, axis=0)
    else:
        x = data.mean(axis=1)
    if file_sr != sr:
        from math import gcd

        g = gcd(int(file_sr), int(sr))
        x = resample_poly(x, sr // g, file_sr // g, axis=-1)
    return np.ascontiguousarray(x, dtype=np.float32)


def build_sfx_leak_mix(
    bed_wav: str | Path | None = None,
    effect_wavs: list[str | Path] | None = None,
    sr: int = SR,
    total_s: float = 90.0,
    n_inserts: int = 12,
    insert_s: float = 2.0,
    insert_gain_db: float = -6.0,
    seed: int = 42,
) -> Clip:
    """Build the §2.2 ground-truth SFX leakage clip.

    A music-only bed with `n_inserts` effect clips (each `insert_s` long)
    laid in at `insert_gain_db`, spaced so they do not overlap. The music
    bed is the ground-truth "to remove"; the inserts are the ground-truth
    "to keep". Scoring (in run_quality) then follows §2.2:
        Leak = output energy in the music-only regions (outside inserts)
        Keep = SI-SDR between output and the clean effects stem inside inserts

    Pass real wavs for a realistic test; omit them to synthesise sources so
    the clip builds with nothing downloaded.
    """
    rng = _rng(seed)
    n = int(total_s * sr)

    if bed_wav is not None:
        bed = _load_wav_mono(Path(bed_wav), sr)
        bed = np.resize(bed, n) if bed.size < n else bed[:n]
        music = _normalise(bed, 0.8)
    else:
        music = _synth_music(n, sr, rng) * _db_to_lin(-3.0)

    effects = np.zeros(n, dtype=np.float64)
    effects_mask = np.zeros(n, dtype=bool)
    ins_n = int(insert_s * sr)
    gain = _db_to_lin(insert_gain_db)

    # Evenly spaced, non-overlapping slots with a little jitter.
    slot = n // n_inserts
    loaded: list[np.ndarray] = []
    if effect_wavs:
        for p in effect_wavs:
            clip = _load_wav_mono(Path(p), sr)
            loaded.append(clip)

    for i in range(n_inserts):
        jitter = int(rng.uniform(0, max(1, slot - ins_n)))
        start = i * slot + jitter
        start = min(start, n - ins_n)
        if loaded:
            src = loaded[i % len(loaded)]
            piece = np.resize(src, ins_n) if src.size < ins_n else src[:ins_n]
        else:
            piece, _ = _synth_effects(ins_n, sr, rng, n_events=2)
        piece = piece.astype(np.float64) * np.hanning(ins_n)  # avoid edge clicks
        seg = slice(start, start + ins_n)
        effects[seg] += piece * gain
        effects_mask[seg] = True

    mix = _normalise(music + effects, 0.95)
    return Clip(
        name="sfx_leak_op",
        sr=sr,
        mix=mix,
        speech=np.zeros(n, dtype=np.float32),
        music=music.astype(np.float32),
        effects=effects.astype(np.float32),
        speech_mask=np.zeros(n, dtype=bool),
        music_mask=np.ones(n, dtype=bool),
        effects_mask=effects_mask,
    )


# --------------------------------------------------------------------------
# Tier 3: real dialogue over a real music bed.
# --------------------------------------------------------------------------

def extract_instrumental_bed(
    song: np.ndarray, vocals: np.ndarray, sr: int = SR, min_run_s: float = 1.0,
    vocal_floor_db: float = -25.0,
) -> np.ndarray:
    """Instrumental-only stretches of a real song.

    `vocals` is a separated vocal stem of `song` (time-aligned). Where it is
    more than `vocal_floor_db` below the song itself (and the song is not
    silent), only instruments are playing, so those stretches are a clean
    music bed. Runs shorter than `min_run_s` are dropped; the rest are joined
    with 20 ms crossfades. Mono [n] or stereo [2, n]; detection runs on the
    channel mean, so a stereo bed's mid is exactly the mono bed.
    """
    n = min(song.shape[-1], vocals.shape[-1])
    song = song[..., :n].astype(np.float64)
    s_mono = song if song.ndim == 1 else song.mean(axis=0)
    v_mono = (vocals if vocals.ndim == 1 else vocals.mean(axis=0))[:n].astype(np.float64)
    hop = int(0.05 * sr)
    frames = n // hop
    v_db = 10 * np.log10(np.array([np.mean(v_mono[i * hop:(i + 1) * hop] ** 2)
                                   for i in range(frames)]) + 1e-12)
    s_db = 10 * np.log10(np.array([np.mean(s_mono[i * hop:(i + 1) * hop] ** 2)
                                   for i in range(frames)]) + 1e-12)
    quiet = (v_db < s_db + vocal_floor_db) & (s_db > s_db.max() - 35.0)
    runs, start = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * hop >= min_run_s * sr:
                runs.append(song[..., start * hop:i * hop])
            start = None
    if not runs:
        raise ValueError("no instrumental-only stretch found")
    xf = int(0.02 * sr)
    ramp = np.linspace(0.0, 1.0, xf)
    bed = runs[0]
    for r in runs[1:]:
        bed = np.concatenate([bed[..., :-xf], bed[..., -xf:] * (1 - ramp) + r[..., :xf] * ramp,
                              r[..., xf:]], axis=-1)
    return bed.astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-20))


def build_real_dialogue_mix(
    speech: np.ndarray, bed: np.ndarray, name: str, sr: int = SR,
    duration_s: float = 40.0, music_rel_db: float = -8.0, phrase_s: float = 6.0,
    gap_s: float = 2.0,
) -> Clip:
    """Real speech over a real music bed, with music-only gaps.

    The speech is cut into `phrase_s` phrases separated by `gap_s` of music
    alone (so suppression is measurable). The bed loops to fill the clip and
    is set `music_rel_db` relative to the speech's active level. A stereo
    [2, n] bed gives a stereo clip with the speech panned centre, as film
    dialogue is.
    """
    n = int(duration_s * sr)
    out_speech = np.zeros(n, dtype=np.float64)
    ph, gap = int(phrase_s * sr), int(gap_s * sr)
    pos, src = gap, 0
    while pos + ph <= n and src + ph <= speech.size:
        piece = speech[src:src + ph].astype(np.float64)
        fade = int(0.01 * sr)
        piece[:fade] *= np.linspace(0, 1, fade)
        piece[-fade:] *= np.linspace(1, 0, fade)
        out_speech[pos:pos + ph] = piece
        pos += ph + gap
        src += ph
    speech_mask = _activity_mask(out_speech, sr, thresh_db=-40.0)
    reps = -(-n // bed.shape[-1])
    music = np.tile(bed.astype(np.float64), (1,) * (bed.ndim - 1) + (reps,))[..., :n]
    target = _rms(out_speech[speech_mask]) * _db_to_lin(music_rel_db)
    music *= target / _rms(music)
    out_speech = np.broadcast_to(out_speech, music.shape)
    total = out_speech + music
    peak = float(np.max(np.abs(total)))
    scale = 0.95 / peak if peak > 0.95 else 1.0
    return Clip(
        name=name,
        sr=sr,
        mix=(total * scale).astype(np.float32),
        speech=(out_speech * scale).astype(np.float32),
        music=(music * scale).astype(np.float32),
        effects=np.zeros(music.shape, dtype=np.float32),
        speech_mask=speech_mask,
        music_mask=np.ones(n, dtype=bool),
        effects_mask=np.zeros(n, dtype=bool),
    )


def real_dialogue_set(speech_wav, songs, sr: int = SR, stereo: bool = False) -> list[Clip]:
    """Tier 3: the same real speech over each real instrumental bed, at a
    typical level (music 8 dB under speech) and a hard one (2 dB under).

    `songs` is a list of (song_wav, vocals_wav) pairs; every file must be a
    wav (convert mp3/m4a with ffmpeg first). stereo=True keeps the songs'
    stereo beds (clip names end in "_st"); their mid equals the mono clip.
    """
    speech = _load_wav_mono(Path(speech_wav), sr)
    tag = "_st" if stereo else ""
    clips = []
    for i, (song_wav, vocals_wav) in enumerate(songs, 1):
        bed = extract_instrumental_bed(_load_wav(Path(song_wav), sr, stereo),
                                       _load_wav_mono(Path(vocals_wav), sr), sr)
        clips += [
            build_real_dialogue_mix(speech, bed, f"real{i}_typical{tag}", sr, music_rel_db=-8.0),
            build_real_dialogue_mix(speech, bed, f"real{i}_loud_music{tag}", sr, music_rel_db=-2.0),
        ]
    return clips


def write_clip(clip: Clip, out_dir: str | Path) -> dict[str, str]:
    """Write mix + stems to wavs for listening/inspection (needs soundfile)."""
    import soundfile as sf

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for label, sig in (
        ("mix", clip.mix),
        ("speech", clip.speech),
        ("music", clip.music),
        ("effects", clip.effects),
    ):
        p = out / f"{clip.name}_{label}.wav"
        sf.write(str(p), sig.T if sig.ndim == 2 else sig, clip.sr)
        paths[label] = str(p)
    return paths
