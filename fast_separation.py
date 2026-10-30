"""
Fast separation engine for the offline Music Assassin app.

Three things live here:

1. separate_vocals_mdx()  -- UVR-MDX-NET-Voc_FT with the same maths as
   audio-separator, but run through OpenVINO (BF16 compute on CPUs that
   support it, FP32 otherwise) with an explicit thread count, and over the
   whole file in one pass instead of 60 s chunks. Measured on a Ryzen 9
   9950X3D: ~42 s per 10 min of audio, against ~220 s through
   audio-separator (docs/OFFLINE_SPEED_RESEARCH.md).

2. separate_dme()  -- dialogue / music / effects separation, used by the
   "keep sound effects" pipeline:
     - "mrx"   MERL cocktail-fork MRX, int8 ONNX, MIT  (fast)
     - "cdx23" ZFTurbo HTDemucs trained on DnR, MIT    (quality)

3. clean_effects()  -- spectral over-subtraction of the music stem from the
   effects stem, so music that leaked into the effects is pushed back out
   before they are mixed into the de-musiced audio. On by default for MRX,
   whose effects stem leaks the most (docs/OFFLINE_SPEED_RESEARCH.md).

Falls back to onnxruntime when OpenVINO is not installed. Overrides:
  MA_ENGINE_THREADS=<n>          thread count (default: physical cores)
  MA_ENGINE_PRECISION=f32|bf16   force precision (default: bf16 if supported;
                                 OpenVINO issue #38256 reports bad bf16 GEMM
                                 results on some Ryzen 9000 CPUs -- set f32 if
                                 output ever sounds wrong)
  MA_MODEL_DIR=<path>            model cache directory
  MA_BUNDLED_MODEL_DIR=<path>    read-only models shipped with a packaged
                                 build (set by the PyInstaller runtime hook)
"""

import os
import math
import time
import hashlib
import urllib.request

import numpy as np

SR = 44100

_MODELS = {
    "voc_ft": {
        "file": "UVR-MDX-NET-Voc_FT.onnx",
        "url": "https://github.com/TRvlvr/model_repo/releases/download/"
               "all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx",
        "size": 66762490,
    },
    "mrx": {
        "file": "mrx.int8.onnx",
        "url": "https://huggingface.co/audiojs/mrx/resolve/main/mrx.int8.onnx",
        "size": 31287081,
        "sha256": "d876c92d224f5d8cd2207d528b23369da895a6de71fb24132416fcb16f9f8424",
    },
    "cdx23": {
        "file": "97d170e1-dbb4db15.th",
        "url": "https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing/"
               "releases/download/v.1.0.0/97d170e1-dbb4db15.th",
        "size": 53916935,
    },
}

# UVR-MDX-NET-Voc_FT parameters (TRvlvr model_data.json, hash 77d07b26...).
# audio-separator's defaults: hop 1024, segment 256 (= dim_t), overlap 0.25.
_VOC_FT = dict(n_fft=7680, dim_f=3072, dim_t=256, hop=1024)

_NET_CACHE = {}
_TORCH_CACHE = {}


# ── Environment ─────────────────────────────────────────────────────────────

def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def default_threads():
    env = os.environ.get("MA_ENGINE_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return physical_cores()


def model_dir():
    d = os.environ.get("MA_MODEL_DIR")
    if not d:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        d = os.path.join(base, "music-assassin", "models")
    os.makedirs(d, exist_ok=True)
    return d


def is_available():
    """The engine needs torch (STFT) and either OpenVINO or onnxruntime."""
    import importlib.util as u
    has_rt = u.find_spec("openvino") is not None or u.find_spec("onnxruntime") is not None
    return has_rt and u.find_spec("torch") is not None


def dme_available(source):
    import importlib.util as u
    if source == "cdx23":
        return u.find_spec("torch") is not None and u.find_spec("demucs") is not None
    return is_available()


def model_path(key, log=print):
    """Return the local path of a model, downloading it on first use."""
    m = _MODELS[key]
    path = os.path.join(model_dir(), m["file"])
    candidates = [path]
    bundled = os.environ.get("MA_BUNDLED_MODEL_DIR")
    if bundled:
        candidates.insert(0, os.path.join(bundled, m["file"]))
    if key == "voc_ft":
        # Reuse audio-separator's copy of Voc_FT if it is already on disk.
        candidates.append(os.path.join("/tmp/audio-separator-models", m["file"]))
    for c in candidates:
        if os.path.isfile(c) and os.path.getsize(c) == m["size"]:
            return c
    log(f"  Downloading {m['file']} ({m['size'] / 1e6:.0f} MB)...")
    tmp = path + ".part"
    urllib.request.urlretrieve(m["url"], tmp)
    if os.path.getsize(tmp) != m["size"]:
        os.remove(tmp)
        raise RuntimeError(f"download of {m['file']} is the wrong size")
    if "sha256" in m:
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != m["sha256"]:
            os.remove(tmp)
            raise RuntimeError(f"download of {m['file']} failed its checksum")
    os.replace(tmp, path)
    return path


# ── Inference runtime ───────────────────────────────────────────────────────

class _Net:
    """One ONNX model on OpenVINO (preferred) or onnxruntime."""

    def __init__(self, path, threads, log=print):
        self.backend = None
        want = os.environ.get("MA_ENGINE_PRECISION", "").lower()
        try:
            import openvino as ov
            core = ov.Core()
            caps = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
            prec = want if want in ("f32", "bf16") else ("bf16" if "BF16" in caps else "f32")
            if prec == "bf16" and "BF16" not in caps:
                prec = "f32"
            cfg = {
                "INFERENCE_NUM_THREADS": int(threads),
                "INFERENCE_PRECISION_HINT": prec,
                "PERFORMANCE_HINT": "LATENCY",
            }
            self._compiled = core.compile_model(path, "CPU", cfg)
            self._req = self._compiled.create_infer_request()
            self._outs = list(self._compiled.outputs)
            self.output_names = [o.get_any_name() for o in self._outs]
            self.backend = f"openvino {prec}"
        except Exception as e:  # not installed, or failed to compile
            if not isinstance(e, ImportError):
                log(f"  OpenVINO unavailable ({e}); using onnxruntime")
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(threads)
            so.inter_op_num_threads = 1
            self._sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            self.output_names = [o.name for o in self._sess.get_outputs()]
            self.backend = "onnxruntime f32"

    def run(self, feeds):
        """feeds: {input_name: array}. Returns {output_name: array}."""
        if self.backend.startswith("openvino"):
            res = self._req.infer(feeds)
            return {n: np.array(res[o], copy=True) for n, o in zip(self.output_names, self._outs)}
        outs = self._sess.run(None, feeds)
        return dict(zip(self.output_names, outs))


def _get_net(key, threads, log=print):
    ck = (key, threads, os.environ.get("MA_ENGINE_PRECISION", ""))
    net = _NET_CACHE.get(ck)
    if net is None:
        t0 = time.perf_counter()
        net = _Net(model_path(key, log), threads, log)
        _NET_CACHE[ck] = net
        log(f"  {key}: loaded on {net.backend}, {threads} threads ({time.perf_counter() - t0:.1f}s)")
    return net


# ── MDX-Net (Voc_FT) ────────────────────────────────────────────────────────

def separate_vocals_mdx(wave, sr=SR, overlap=0.25, threads=None, progress=None, log=print):
    """Vocals with UVR-MDX-NET-Voc_FT, from a mono [n] or stereo [2, n]
    waveform; the output has the input's shape.

    Voc_FT is a stereo model. Mono input is fed as L = R and the two output
    channels averaged; stereo input is fed as is, so stereo costs nothing
    extra.

    Same windowing as audio-separator's MDXSeparator.demix(): Hann-weighted
    overlap-add of 261120-sample windows, trimmed by n_fft/2 at each end.
    Differences, both deliberate:
      - the whole file is one pass (no 60 s outer chunks, so no chunk seams);
      - input is scaled once to a 0.9 peak and the output scaled back, where
        audio-separator multiplied each 60 s chunk by that chunk's peak, which
        made dialogue in quiet scenes quieter still.
    Accumulators cover two windows, not the whole file, so memory stays flat.
    """
    import torch

    threads = threads or default_threads()
    net = _get_net("voc_ft", threads, log)
    p = _VOC_FT
    n_fft, hop, dim_f, dim_t = p["n_fft"], p["hop"], p["dim_f"], p["dim_t"]
    n_bins = n_fft // 2 + 1
    trim = n_fft // 2
    chunk = hop * (dim_t - 1)
    gen = chunk - 2 * trim
    hann = torch.hann_window(n_fft, periodic=True)

    x = np.asarray(wave, dtype=np.float32)
    stereo = x.ndim == 2
    if sr != SR:
        x = _resample(x, sr, SR)
    x = x if stereo else x[None]                                   # [C, n]
    ch, n = x.shape
    peak = float(np.max(np.abs(x))) if n else 0.0
    scale = 0.9 / peak if peak > 0.9 else 1.0

    pad = gen + trim - (n % gen)
    mixture = np.pad(x * scale, ((0, 0), (trim, pad)))
    total = mixture.shape[1]
    step = max(1, int((1 - overlap) * chunk))

    out = np.zeros((ch, n), dtype=np.float32)
    span = 2 * chunk
    acc = np.zeros((ch, span), dtype=np.float32)
    div = np.zeros(span, dtype=np.float32)
    base = 0

    def flush(upto):
        nonlocal base
        k = upto - base
        seg = np.divide(acc[:, :k], div[:k], out=np.zeros((ch, k), np.float32), where=div[:k] > 0)
        lo, hi = max(base, trim), min(upto, trim + n)
        if hi > lo:
            out[:, lo - trim:hi - trim] = seg[:, lo - base:hi - base]
        acc[:, :span - k] = acc[:, k:]
        acc[:, span - k:] = 0
        div[:span - k] = div[k:]
        div[span - k:] = 0
        base = upto

    starts = range(0, total, step)
    n_win = len(starts)
    with torch.no_grad():
        for w_idx, i in enumerate(starts):
            end = min(i + chunk, total)
            actual = end - i
            if i + chunk > base + span:
                flush(i)
            part = mixture[:, i:end]
            if actual < chunk:
                part = np.pad(part, ((0, 0), (0, chunk - actual)))

            # STFT exactly as audio-separator's uvr_lib_v5/stft.py:
            # [1, 2ch, T] -> [1, 4 (L re, L im, R re, R im), dim_f, dim_t]
            two = torch.from_numpy(np.ascontiguousarray(part if stereo else np.repeat(part, 2, 0)))
            spec = torch.stft(two, n_fft=n_fft, hop_length=hop, window=hann,
                              center=True, return_complex=False)       # [2, F, T, 2]
            spec = spec.permute(0, 3, 1, 2).reshape(1, 4, n_bins, -1)[:, :, :dim_f, :]
            spec[:, :, :3, :] = 0
            pred = next(iter(net.run({"input": spec.numpy()}).values()))

            pred = torch.from_numpy(pred)
            pred = torch.cat([pred, torch.zeros(1, 4, n_bins - dim_f, pred.shape[-1])], -2)
            pred = pred.reshape(2, 2, n_bins, -1).permute(0, 2, 3, 1)
            wav = torch.istft(torch.complex(pred[..., 0], pred[..., 1]), n_fft=n_fft,
                              hop_length=hop, window=hann, center=True).numpy()
            res = wav[:, :actual] if stereo else wav.mean(axis=0, keepdims=True)[:, :actual]

            if overlap != 0:
                win = np.hanning(actual).astype(np.float32)
                res = res * win
                div[i - base:end - base] += win
            else:
                div[i - base:end - base] += 1
            acc[:, i - base:end - base] += res
            if progress:
                progress(w_idx + 1, n_win)
    flush(total)

    out /= scale
    if sr != SR:
        out = _resample(out, SR, sr)[:, :np.shape(wave)[-1]]
    return out if stereo else out[0]


# ── Dialogue / music / effects ──────────────────────────────────────────────

# Measured on a ground-truth mix (episode effects inserted into a music-only
# clip): subtraction cuts MRX's music leakage by 5 dB for ~3 dB of effects;
# CDX23 already leaks 8.5 dB less and gains only ~1.4 dB, so it stays raw.
DEFAULT_CLEAN = {"mrx": 1.0, "cdx23": 0.0}


def separate_dme(wave, source="mrx", sr=SR, want=("dialogue", "music", "effects"),
                 clean_strength=None, threads=None, progress=None, log=print):
    """Split a mono [n] or stereo [2, n] waveform into dialogue / music /
    effects.

    Runs in outer chunks with a linear crossfade so memory stays bounded on
    feature-length files; only the stems in `want` are kept. When
    clean_strength > 0 the effects stem is cleaned with clean_effects();
    None means the per-model default (DEFAULT_CLEAN).
    Returns {stem: float32 array shaped like the input}.
    """
    threads = threads or default_threads()
    if clean_strength is None:
        clean_strength = DEFAULT_CLEAN.get(source, 0.0)
    cleaning = "effects" in want and clean_strength > 0
    x = np.asarray(wave, dtype=np.float32)
    if sr != SR:
        x = _resample(x, sr, SR)

    if source == "mrx":
        gain = _loudness_gain(_mid(x), -27.0)    # MRX expects DnR-level input
        chunk_sec, overlap_sec = 30, 2
        # Each stem costs three inverse STFTs; only make the ones needed.
        need = set(want) | ({"music"} if cleaning else set())
        run = lambda seg: _mrx_chunk(seg * gain, threads, log, need)
    elif source == "cdx23":
        gain = 1.0
        chunk_sec, overlap_sec = 60, 2
        run = lambda seg: _cdx23_chunk(seg, threads, log)
    else:
        raise ValueError(f"unknown dialogue/music/effects model: {source}")

    def run_and_clean(seg):
        stems = run(seg)
        if gain != 1.0:
            stems = {k: v / gain for k, v in stems.items()}
        if cleaning:
            stems["effects"] = clean_effects(stems, clean_strength)
        return {k: stems[k] for k in want}

    stems = _chunked(x, run_and_clean, chunk_sec, overlap_sec, want, progress)
    if sr != SR:
        stems = {k: _resample(v, SR, sr)[..., :np.shape(wave)[-1]] for k, v in stems.items()}
    return stems


def clean_effects(stems, strength=1.0, floor=0.05, n_fft=2048, hop=512):
    """Spectral over-subtraction of the music stem from the effects stem.

    gain = max(floor, 1 - strength * |Music| / |Effects|) per STFT bin,
    applied to the effects STFT. The model's own music estimate stands in
    for the music that leaked into the effects. (A Wiener re-filter with all
    three stems was tried first and changed leakage by under 0.5 dB.)
    Stereo stems are cleaned per channel.
    """
    import torch
    win = torch.hann_window(n_fft)

    def spec(sig):
        return torch.stft(torch.from_numpy(np.ascontiguousarray(sig, dtype=np.float32)),
                          n_fft, hop, window=win, center=True, return_complex=True)

    F = spec(stems["effects"])
    g = torch.clamp(1 - strength * spec(stems["music"]).abs() / (F.abs() + 1e-9), min=floor)
    out = torch.istft(F * g, n_fft, hop, window=win, center=True, length=stems["effects"].shape[-1])
    return out.numpy().astype(np.float32)


def _mrx_chunk(seg, threads, log, stems=("music", "dialogue", "effects")):
    """MRX on one chunk. Masks (music, speech, sfx) at three STFT resolutions;
    each resolution's masked signal is summed, as in MERL's mrx.py. Only the
    stems asked for are synthesised.

    MRX is a mono model. For stereo [2, n] the masks come from the mid
    (L+R)/2 and are applied to each channel's STFT: one network pass, and
    the stereo image of what each mask keeps is preserved."""
    import torch
    net = _get_net("mrx", threads, log)
    chans = torch.from_numpy(np.ascontiguousarray(np.atleast_2d(seg), dtype=np.float32))  # [C, n]
    n = chans.shape[-1]

    def stft(sig, w):
        return torch.stft(sig, w, 256, window=torch.hann_window(w), win_length=w, normalized=True,
                          center=True, return_complex=True, pad_mode="reflect")

    specs, feeds = {}, {}
    for w in (1024, 2048, 8192):
        s = stft(chans, w)                                          # [C, F, T]
        specs[w] = s
        mid = s if s.shape[0] == 1 else stft(chans.mean(0, keepdim=True), w)
        feeds[f"mag_{w}"] = mid.abs().numpy()
    masks = net.run(feeds)
    order = {"music": 0, "dialogue": 1, "effects": 2}
    out = {k: np.zeros(chans.shape, np.float32) for k in stems}
    for w, s in specs.items():
        m = torch.from_numpy(masks[f"mask_{w}"])                  # [1, 3, F, T]
        for k in stems:
            y = torch.istft(m[:, order[k]] * s, w, 256, window=torch.hann_window(w), win_length=w,
                            normalized=True, center=True, length=n)
            out[k] += y.numpy()
    return {k: v if np.ndim(seg) == 2 else v[0] for k, v in out.items()}


def _cdx23_chunk(seg, threads, log):
    """CDX23 HTDemucs (one checkpoint) on one chunk. shifts=0 and overlap 0.1
    instead of upstream's 1 / 0.8: about 5x less compute, see
    docs/OFFLINE_SPEED_RESEARCH.md.

    CDX23 is trained on DnR, which is mono: fed real stereo it leaked 10 dB
    more music and still returned mono. So stereo [2, n] runs on the mid and
    _share_stems() spreads the stems back over L and R (A3, 2026-10-31)."""
    if np.ndim(seg) == 2:
        return _share_stems(seg, _cdx23_chunk(seg.mean(axis=0), threads, log))
    import torch
    from demucs.apply import apply_model
    model = _TORCH_CACHE.get("cdx23")
    if model is None:
        from demucs.states import load_model
        t0 = time.perf_counter()
        model = load_model(model_path("cdx23", log))
        model.eval()
        _TORCH_CACHE["cdx23"] = model
        log(f"  cdx23: loaded on torch, {threads} threads ({time.perf_counter() - t0:.1f}s)")
    torch.set_num_threads(threads)
    audio = torch.from_numpy(np.stack([seg, seg]).astype(np.float32))[None]
    with torch.no_grad():
        out = apply_model(model, audio, shifts=0, overlap=0.1, split=True, progress=False)[0]
    out = out.mean(dim=1).numpy()                                   # [3, n] mono
    return {"music": out[0], "effects": out[1], "dialogue": out[2]}


def _share_stems(x, stems, n_fft=2048, hop=512):
    """Stereo stems from a mono model's stems of the mid: each stem's share
    of the summed stem magnitudes, applied to each channel's STFT. On the
    stereo ground-truth effects test this kept the effects' stereo image and
    leaked slightly less music than mono (A3, 2026-10-31)."""
    import torch
    win = torch.hann_window(n_fft)

    def spec(sig):
        return torch.stft(torch.from_numpy(np.ascontiguousarray(sig, dtype=np.float32)), n_fft, hop,
                          window=win, center=True, return_complex=True)

    mags = {k: spec(v).abs() for k, v in stems.items()}
    total = sum(mags.values()) + 1e-8
    X = spec(x)                                                     # [C, F, T]
    return {k: torch.istft(X * (m / total), n_fft, hop, window=win, center=True,
                           length=x.shape[-1]).numpy() for k, m in mags.items()}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _chunked(x, fn, chunk_sec, overlap_sec, keys, progress=None):
    """Run fn over overlapping chunks of x ([n] or [C, n], chunked along the
    last axis); linear crossfade in the overlap. The last chunk is
    zero-padded to full length so model input shapes stay fixed (OpenVINO
    compiles once)."""
    n = x.shape[-1]
    chunk = int(chunk_sec * SR)
    ov = int(overlap_sec * SR)
    step = chunk - ov
    out = {k: np.zeros(x.shape, np.float32) for k in keys}
    starts = list(range(0, max(n - ov, 1), step))
    for idx, s in enumerate(starts):
        e = min(s + chunk, n)
        seg = x[..., s:e]
        if seg.shape[-1] < chunk and n > chunk:
            seg = np.pad(seg, [(0, 0)] * (seg.ndim - 1) + [(0, chunk - seg.shape[-1])])
        res = fn(seg)
        length = e - s
        for k in keys:
            y = res[k][..., :length]
            if idx == 0:
                out[k][..., s:e] = y
            else:
                f = min(ov, length)
                ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
                out[k][..., s:s + f] = out[k][..., s:s + f] * (1 - ramp) + y[..., :f] * ramp
                out[k][..., s + f:e] = y[..., f:]
        if progress:
            progress(idx + 1, len(starts))
    return out


def _loudness_gain(x, target_lufs):
    """Gain that brings x to target integrated loudness (BS.1770 via
    pyloudnorm; RMS fallback if it is not installed)."""
    try:
        import pyloudnorm
        lufs = pyloudnorm.Meter(SR).integrated_loudness(x.astype(np.float64))
    except Exception:
        rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
        lufs = 20 * math.log10(rms) - 0.691 if rms > 0 else float("-inf")
    if not np.isfinite(lufs):
        return 1.0
    return float(10 ** ((target_lufs - lufs) / 20))


def _resample(x, sr_in, sr_out):
    from scipy.signal import resample_poly
    g = math.gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g, axis=-1).astype(np.float32)


def _mid(x):
    """Mono mid (L+R)/2 of a [2, n] signal; mono input passes through."""
    return x if x.ndim == 1 else x.mean(axis=0)
