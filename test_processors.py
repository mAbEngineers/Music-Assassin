#!/usr/bin/env python3
"""
Music Assassin v0.4.5 — Safe Processor Test Suite

Tests every filter / enhancer / separator with:
  - A 10-second audio clip for filters, VAD, enhancers
  - A 20-second clip for separators (short enough to limit RAM)
  - RAM logged before and after every step
  - RAM guard before each separator: skips if < 2 GB free
  - Models freed and GC'd between sections
  - Full run log saved to logs/test_processors_TIMESTAMP.log

Usage:
    /home/yoake/Documents/venvs/assassin_venv_v0.4.2/bin/python test_processors.py

    # DSP + VAD + enhancers only (no separators):
    SKIP_SEPARATORS=1 <venv>/bin/python test_processors.py
"""

import os, sys, gc, time, types, random, subprocess, tempfile, logging, traceback
from datetime import datetime

REPO = os.path.dirname(os.path.abspath(__file__))
SAMPLE_VIDEO = (
    "/home/yoake/Videos/Movies/"
    "Charlotte.S01.1080p.BluRay.10-Bit.Dual-Audio.FLAC2.0.x265-YURASUKA/"
    "Charlotte.S01E09.1080p.BluRay.10-Bit.Dual-Audio.FLAC2.0.x265-YURASUKA.mkv"
)
FILTER_CLIP_SEC    = 10   # DSP / VAD / enhancers
SEPARATOR_CLIP_SEC = 30   # longer clip shows separation quality better
CLIP_START_SEC     = 10   # OP theme — clear music + Japanese vocals, good test signal
SAMPLE_RATE        = 44100
SKIP_SEPARATORS    = os.environ.get("SKIP_SEPARATORS", "0") == "1"
RAM_FLOOR_GB       = 2.0  # minimum free RAM before running a separator

# ─── Logging setup ────────────────────────────────────────────────────────────
logs_dir = os.path.join(REPO, "logs")
os.makedirs(logs_dir, exist_ok=True)

run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(logs_dir, f"test_processors_{run_ts}.log")

_log = logging.getLogger("ma_test")
_log.setLevel(logging.DEBUG)

_fmt = logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s",
                          datefmt="%H:%M:%S")

# File handler — full detail
_fh = logging.FileHandler(log_path, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
_log.addHandler(_fh)

# Console handler — same content
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.DEBUG)
_ch.setFormatter(_fmt)
_log.addHandler(_ch)


# Redirect print() through the logger so app module prints also land in the log
class _PrintToLog:
    def __init__(self, orig): self._orig = orig
    def write(self, s):
        s = s.rstrip("\n")
        if s:
            _log.debug(s)
    def flush(self): self._orig.flush()
    def fileno(self): return self._orig.fileno()

sys.stdout = _PrintToLog(sys.__stdout__)
sys.stderr = _PrintToLog(sys.__stderr__)


# ─── Helpers ──────────────────────────────────────────────────────────────────
G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; B = "\033[1m"; E = "\033[0m"
results = {"pass": 0, "skip": 0, "fail": 0}


def free_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**3
    except ImportError:
        return 99.0


def ram_str():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return (f"{vm.available/1024**3:.2f} GB free / "
                f"{vm.total/1024**3:.1f} GB total  "
                f"({vm.percent:.0f}% used)")
    except ImportError:
        return "psutil not installed"


def log_ram(label=""):
    _log.info(f"RAM  {label}: {ram_str()}")


def ok(label, t):
    _log.info(f"{G}PASS{E}  {label}  ({t:.1f}s)")
    results["pass"] += 1

def sk(label, why):
    _log.warning(f"{Y}SKIP{E}  {label}  — {why}")
    results["skip"] += 1

def fl(label, err):
    _log.error(f"{R}FAIL{E}  {label}  → {err}")
    results["fail"] += 1


def run(label, fn):
    log_ram(f"before [{label[:40]}]")
    t0 = time.time()
    try:
        outcome = fn()
        t = time.time() - t0
        log_ram(f"after  [{label[:40]}]")
        if outcome == "skip":
            sk(label, "not installed / model missing")
        else:
            ok(label, t)
    except Exception as ex:
        log_ram(f"after  [{label[:40]}] (FAILED)")
        _log.error(traceback.format_exc())
        fl(label, ex)


def ram_guard(label):
    """Return True (and log skip) if free RAM is below RAM_FLOOR_GB."""
    free = free_ram_gb()
    if free < RAM_FLOOR_GB:
        sk(label, f"only {free:.1f} GB free — need {RAM_FLOOR_GB} GB")
        return True
    _log.info(f"  RAM check OK: {free:.1f} GB free")
    return False


# ─── Mock tkinter for headless import ────────────────────────────────────────
_log.info("=" * 60)
_log.info(f"Music Assassin v0.4.5 Processor Tests  —  {run_ts}")
_log.info(f"Log file: {log_path}")
_log.info("=" * 60)
_log.info(f"System RAM at start: {ram_str()}")
_log.info("")

_log.info("Mocking tkinter for headless import…")
for mod_name in ("tkinter", "tkinter.filedialog", "tkinter.messagebox",
                 "tkinter.scrolledtext", "tkinter.ttk"):
    m = types.ModuleType(mod_name)
    sys.modules[mod_name] = m

tk = sys.modules["tkinter"]
for attr in ("filedialog", "messagebox", "Label", "Frame", "Scrollbar",
             "Listbox", "Button", "Tk", "Entry", "Checkbutton",
             "BooleanVar", "StringVar", "IntVar", "PhotoImage"):
    setattr(tk, attr, lambda *a, **kw: None)
tk.BooleanVar = lambda *a, **kw: types.SimpleNamespace(
    get=lambda: False, set=lambda v: None)

ttk = sys.modules["tkinter.ttk"]
for attr in ("Progressbar", "Combobox", "Style"):
    setattr(ttk, attr, lambda *a, **kw: None)
sys.modules["tkinter.scrolledtext"].ScrolledText = lambda *a, **kw: None

# ─── Import v0.4.4 ────────────────────────────────────────────────────────────
_log.info("Importing MusicAssassinGUI_v0.4.5…")
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "music_assassin_v044",
    os.path.join(REPO, "MusicAssassinGUI_v0.4.5.py"),
)
app = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(app)

# Patch _run_in_background: no GUI event loop → call fn() directly
app._run_in_background = lambda fn, msg="": fn()
# Give root a no-op so DEMUCS progress loop doesn't crash
app.root = types.SimpleNamespace(
    update=lambda: None, update_idletasks=lambda: None)
app.progress = None

import numpy as np
import soundfile as sf

_log.info(f"DEMUCS_AVAILABLE:           {app.DEMUCS_AVAILABLE}")
_log.info(f"AUDIO_SEPARATOR_AVAILABLE:  {app.AUDIO_SEPARATOR_AVAILABLE}")
_log.info(f"SPLEETER_AVAILABLE:         {app.SPLEETER_AVAILABLE}")
_log.info(f"VAD_AVAILABLE:              {app.VAD_AVAILABLE}")
_log.info(f"DF_AVAILABLE:               {app.DF_AVAILABLE}")
_log.info(f"SHERPA_AVAILABLE:           {app.SHERPA_AVAILABLE}")
_log.info(f"OV_AVAILABLE:               {app.OV_AVAILABLE}")
_log.info("")

# ─── Extract audio clips ──────────────────────────────────────────────────────
_log.info(f"{B}[0] Extracting audio clips…{E}")
if not os.path.isfile(SAMPLE_VIDEO):
    _log.error(f"Sample video not found: {SAMPLE_VIDEO}")
    sys.exit(1)

tmp = tempfile.mkdtemp(prefix="ma_test_")

def extract_clip(duration, start=CLIP_START_SEC):
    path = os.path.join(tmp, f"clip_{duration}s.wav")
    if os.path.isfile(path):
        return path
    r = subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start), "-i", SAMPLE_VIDEO,
        "-t", str(duration), "-vn",
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "wav", path,
    ], capture_output=True)
    if r.returncode != 0:
        _log.error(f"ffmpeg error: {r.stderr.decode()}")
        sys.exit(1)
    return path

filter_wav = extract_clip(FILTER_CLIP_SEC)
audio_f, sr_f = sf.read(filter_wav, dtype='float32')
if audio_f.ndim > 1: audio_f = audio_f[:, 0]
_log.info(f"Filter clip:    {filter_wav}  ({len(audio_f)/sr_f:.1f}s)")

if not SKIP_SEPARATORS:
    separator_wav = extract_clip(SEPARATOR_CLIP_SEC)
    audio_s, sr_s = sf.read(separator_wav, dtype='float32')
    if audio_s.ndim > 1: audio_s = audio_s[:, 0]
    _log.info(f"Separator clip: {separator_wav}  ({len(audio_s)/sr_s:.1f}s)")
else:
    separator_wav = audio_s = sr_s = None

_log.info("")


def fresh_f(): return audio_f.copy()
def fresh_s(): return audio_s.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# A — DSP Filters
# ═══════════════════════════════════════════════════════════════════════════════
_log.info(f"{B}[A] DSP Filters  (10 s clip){E}")

def test_prefilter():
    out = app.audio_prefilter(fresh_f(), sr_f)
    assert out.dtype == np.float32
    assert len(out) == len(audio_f)

def test_pp_normal():
    out = app.advanced_postprocess_chunked(fresh_f(), sr_f, mode='normal')
    assert len(out) > 0 and out.dtype == np.float32

def test_pp_balanced():
    out = app.advanced_postprocess_chunked(fresh_f(), sr_f, mode='balanced')
    assert len(out) > 0

def test_pp_aggressive():
    out = app.advanced_postprocess_chunked(fresh_f(), sr_f, mode='aggressive')
    assert len(out) > 0

run("audio_prefilter  (40 Hz high-pass + normalise)", test_prefilter)
run("postprocess — Normal mode     (SciPy filters, 30 s chunks)",  test_pp_normal)
run("postprocess — Balanced mode   (de-esser + formant boost)",    test_pp_balanced)
run("postprocess — Aggressive mode (heavy de-ess + formant boost)",test_pp_aggressive)


# ═══════════════════════════════════════════════════════════════════════════════
# B — VAD Gate
# ═══════════════════════════════════════════════════════════════════════════════
_log.info(f"\n{B}[B] VAD Gate  (Silero VAD via torch hub){E}")

def test_vad():
    if not app.VAD_AVAILABLE:
        return "skip"
    out = app.apply_vad_gate(fresh_f(), sr_f)
    assert out.shape == audio_f.shape
    assert out.dtype == np.float32
    zeroed = int(np.sum(out == 0.0))
    _log.info(f"  VAD zeroed {zeroed}/{len(out)} samples "
              f"({100*zeroed/len(out):.1f}% non-vocal)")

run("apply_vad_gate  (smooth mask, 80 ms padding, 15 ms fade)", test_vad)


# ═══════════════════════════════════════════════════════════════════════════════
# C — Post-Separation Enhancers
# ═══════════════════════════════════════════════════════════════════════════════
_log.info(f"\n{B}[C] Enhancers  (10 s clip — no separator needed){E}")

def test_enhancer_none():
    app.enhancer_choice = 'none'
    out = app.apply_enhancer(fresh_f(), sr_f)
    assert np.array_equal(out, audio_f), "none should be a no-op pass-through"

def _assert_audio(out, ref_len, name):
    """Assert output has correct length, float32 dtype, and is not silent."""
    assert out is not None, f"{name}: returned None"
    assert len(out) == ref_len, f"{name}: length {len(out)} != {ref_len}"
    assert out.dtype == np.float32, f"{name}: dtype {out.dtype} != float32"
    rms = float(np.sqrt(np.mean(out.astype(np.float64)**2)))
    _log.info(f"  {name}: rms={rms:.4f}  max={float(np.abs(out).max()):.4f}")
    assert rms > 1e-4, f"{name}: output is silent (rms={rms:.6f}) — processing failed"

def test_deepfilter():
    if not app.DF_AVAILABLE:
        return "skip"
    out = app.apply_deepfilter(fresh_f(), sr_f)
    _assert_audio(out, len(audio_f), "DeepFilterNet")
    out_path = os.path.join(REPO, "test_deepfilter_output.wav")
    sf.write(out_path, out, sr_f)
    _log.info(f"  output written: {out_path}")

def test_dpdfnet():
    if not app.SHERPA_AVAILABLE:
        return "skip"
    out = app.apply_sherpa_enhancer(fresh_f(), sr_f, model_name='dpdfnet')
    _assert_audio(out, len(audio_f), "DPDFNet")
    out_path = os.path.join(REPO, "test_dpdfnet_output.wav")
    sf.write(out_path, out, sr_f)
    _log.info(f"  output written: {out_path}")

def test_gtcrn():
    if not app.SHERPA_AVAILABLE:
        return "skip"
    out = app.apply_sherpa_enhancer(fresh_f(), sr_f, model_name='gtcrn')
    _assert_audio(out, len(audio_f), "GTCRN")
    out_path = os.path.join(REPO, "test_gtcrn_output.wav")
    sf.write(out_path, out, sr_f)
    _log.info(f"  output written: {out_path}")

def test_speechdenoiser():
    if not os.path.isfile(app._SPEECHDENOISER_PATH):
        return "skip"
    try:
        import onnxruntime  # noqa
    except ImportError:
        return "skip"
    out = app.apply_speechdenoiser(fresh_f(), sr_f)
    _assert_audio(out, len(audio_f), "SpeechDenoiser")
    out_path = os.path.join(REPO, "test_speechdenoiser_output.wav")
    sf.write(out_path, out, sr_f)
    _log.info(f"  output written: {out_path}")

run("apply_enhancer — mode=none  (pass-through)",                    test_enhancer_none)
run("DeepFilterNet  (torch, 60 s chunks; pip install deepfilternet)", test_deepfilter)
run("DPDFNet  (Sherpa-ONNX streaming; pip install sherpa-onnx)",     test_dpdfnet)
run("GTCRN   (Sherpa-ONNX streaming; pip install sherpa-onnx)",      test_gtcrn)
run("SpeechDenoiser ONNX  (models/speechdenoiser/speechdenoiser.onnx)", test_speechdenoiser)


# ═══════════════════════════════════════════════════════════════════════════════
# D — Full DSP pipeline  (no separator)
# ═══════════════════════════════════════════════════════════════════════════════
_log.info(f"\n{B}[D] Full DSP pipeline  (prefilter → VAD → balanced → none){E}")
_log.info("  NOTE: DSP is applied to raw mixed audio here (not post-separation vocals).")
_log.info("  Output sounds processed/filtered — quality check requires a separator first.")

def test_full_dsp():
    a = fresh_f()
    a = app.audio_prefilter(a, sr_f)
    if app.VAD_AVAILABLE:
        a = app.apply_vad_gate(a, sr_f)
    a = app.advanced_postprocess_chunked(a, sr_f, mode='balanced')
    app.enhancer_choice = 'none'
    a = app.apply_enhancer(a, sr_f)
    assert len(a) > 0 and a.dtype == np.float32
    rms = float(np.sqrt(np.mean(a.astype(np.float64)**2)))
    _log.info(f"  DSP pipeline: rms={rms:.4f}  max={float(np.abs(a).max()):.4f}")
    out_path = os.path.join(REPO, "test_dsp_pipeline_output.wav")
    sf.write(out_path, a, sr_f)
    _log.info(f"  output written: {out_path}")

run("prefilter → VAD → balanced postprocess → no enhancer", test_full_dsp)


# ═══════════════════════════════════════════════════════════════════════════════
# S — Stereo  (A3: stereo sources stay stereo end to end)
# ═══════════════════════════════════════════════════════════════════════════════
_log.info(f"\n{B}[S] Stereo  ([2, n] through every stage){E}")

def _stereo_clip():
    path = os.path.join(tmp, "clip_stereo.wav")
    if not os.path.isfile(path):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(CLIP_START_SEC),
                        "-i", SAMPLE_VIDEO, "-t", str(FILTER_CLIP_SEC), "-vn",
                        "-ac", "2", "-ar", str(SAMPLE_RATE), "-f", "wav", path], check=True)
    a, sr = sf.read(path, dtype='float32')
    return np.ascontiguousarray(a.T), sr

def test_stereo_dsp():
    st, sr = _stereo_clip()
    n = st.shape[1]
    a = app.audio_prefilter(st, sr)
    assert a.shape == (2, n)
    if app.VAD_AVAILABLE:
        a = app.apply_vad_gate(a, sr)
        assert a.shape == (2, n)
    for mode in ('normal', 'balanced', 'aggressive'):
        p = app.advanced_postprocess_chunked(a, sr, mode=mode)
        assert p.shape == (2, n) and p.dtype == np.float32, mode
    # Dual-mono in -> identical channels out (stages share gains across channels)
    dm = app.advanced_postprocess_chunked(np.stack([st[0], st[0]]), sr, mode='balanced')
    assert np.array_equal(dm[0], dm[1])
    assert app.mix_effects(p, p * 0.3, 1.0, sr).shape == (2, n)
    assert app.mix_effects(p[0], p * 0.3, 1.0, sr).shape == (2, n)   # mono dialogue + stereo fx

def test_stereo_transfer():
    st, sr = _stereo_clip()
    mid = st.mean(axis=0)
    err = float(np.abs(app._transfer_gain(st, mid, mid, sr) - st).max())
    _log.info(f"  identity gain transfer: max err {err:.2e}")
    assert err < 1e-4

def test_stereo_enhancer():
    if not app.DF_AVAILABLE:
        return "skip"
    st, sr = _stereo_clip()
    app.enhancer_choice = 'deepfilter'
    for how in ('mid', 'per_channel'):
        app.stereo_enhance = how
        out = app.apply_enhancer(st, sr)
        assert out.shape == st.shape and out.dtype == np.float32, how
    app.stereo_enhance, app.enhancer_choice = 'mid', 'none'

def test_stereo_fast_engine():
    if not app.FAST_ENGINE_AVAILABLE:
        return "skip"
    st, sr = _stereo_clip()
    out = app.fast_separation.separate_vocals_mdx(st, sr)
    assert out.shape == st.shape
    # Dual-mono stereo feeds the model exactly what the mono path does
    x = st[0]
    dual = app.fast_separation.separate_vocals_mdx(np.stack([x, x]), sr)
    mono = app.fast_separation.separate_vocals_mdx(x, sr)
    assert np.allclose(dual.mean(axis=0), mono, atol=1e-5)
    fx = app.fast_separation.separate_dme(st, "mrx", sr, want=("effects",))["effects"]
    assert fx.shape == st.shape

def test_stereo_legacy():
    st, sr = _stereo_clip()
    seps = []
    if app.DEMUCS_AVAILABLE:
        app.model_choice, app.quantize_enabled = "htdemucs", True
        def demucs(w):
            random.seed(0)   # apply_model's shifts=1 offsets the input at random
            return app.separate_with_demucs(w, sr)
        seps.append(("DEMUCS htdemucs", demucs))
    if app.AUDIO_SEPARATOR_AVAILABLE:
        seps.append(("MDX-Net Voc_FT", lambda w: app.separate_with_audio_separator(
            w, sr, mdl="UVR-MDX-NET-Voc_FT.onnx")))
    if not seps or ram_guard("stereo legacy separators"):
        return "skip"
    x = st[0]
    for name, sep in seps:
        out = sep(st)
        assert out is not None and out.shape == st.shape, name
        # Mono is fed to these stereo models as dual mono, so dual-mono
        # stereo averaged back must match the mono path
        dual, mono = sep(np.stack([x, x])), sep(x)
        err = float(np.abs(dual.mean(axis=0) - mono).max())
        _log.info(f"  {name}: stereo {out.shape}, dual-mono vs mono max diff {err:.1e}")
        assert err < 1e-3, name
        app._MODEL_CACHE.clear()        # one model in RAM at a time
        app._AUDIO_SEP_CACHE.clear()
        gc.collect()

def test_stereo_spleeter_plumbing():
    # A stand-in with Spleeter's contract: (n, 1) or (n, 2) in, mono
    # duplicated, (n, 2) out. 130 s spans three 60 s chunks, so the
    # overlap-add runs too; vocals = half the input must come back exactly.
    class _Stub:
        def __init__(self, *a, **kw): pass
        def separate(self, w):
            w = np.tile(w, (1, 2)) if w.shape[-1] == 1 else w
            return {"vocals": w * 0.5, "other": w * 0.5}
        def close(self): pass
    mod = types.ModuleType("spleeter.separator")
    mod.Separator = _Stub
    saved = {k: sys.modules.get(k) for k in ("spleeter", "spleeter.separator")}
    sys.modules.update({"spleeter": types.ModuleType("spleeter"), "spleeter.separator": mod})
    avail, app.SPLEETER_AVAILABLE = app.SPLEETER_AVAILABLE, True
    try:
        st, sr = _stereo_clip()
        st = np.tile(st, (1, 13))
        for w in (st, st[0]):
            out = app.separate_with_spleeter(w, sr)
            assert out is not None and out.shape == w.shape
            assert np.allclose(out, w * 0.5, atol=1e-6)
    finally:
        app.SPLEETER_AVAILABLE = avail
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

run("stereo DSP  (prefilter, VAD, post-process, effects mix keep [2, n])", test_stereo_dsp)
run("stereo gain transfer  (identity within 1e-4)",                    test_stereo_transfer)
run("stereo enhancer  (DeepFilterNet, mid and per-channel)",            test_stereo_enhancer)
run("stereo fast engine  (Voc_FT stereo, dual-mono == mono, MRX fx)",   test_stereo_fast_engine)
run("stereo legacy separators  (DEMUCS, MDX-Net: [2, n] out, dual-mono == mono)", test_stereo_legacy)
run("stereo Spleeter plumbing  (stand-in model, 3 chunks, stereo and mono)",     test_stereo_spleeter_plumbing)


# ═══════════════════════════════════════════════════════════════════════════════
# E — Separators  (20 s clip, one at a time, RAM-guarded)
# ═══════════════════════════════════════════════════════════════════════════════
if SKIP_SEPARATORS:
    _log.info(f"\n{Y}[E] Separators skipped (SKIP_SEPARATORS=1){E}")
else:
    _log.info(f"\n{B}[E] Separators  ({SEPARATOR_CLIP_SEC} s clip — tested one at a time){E}")
    _log.info(f"  RAM floor: {RAM_FLOOR_GB} GB free required before each separator")

    # ── E1: DEMUCS htdemucs ──────────────────────────────────────────────────
    # htdemucs loads ~320 MB of weights and expands to ~1-2 GB during inference.
    # We pass segment=6 (6 s internal segments) so demucs.apply.apply_model
    # sub-segments the 20 s tensor into 4 × 6 s windows instead of all at once.
    def test_demucs_htdemucs():
        if not app.DEMUCS_AVAILABLE:
            return "skip"
        if ram_guard("DEMUCS htdemucs"):
            return "skip"

        from torch import set_num_threads, device as torch_device, from_numpy, no_grad
        from demucs.apply import apply_model
        import threading

        _log.info("  Loading htdemucs model (INT8)…")
        app.model_choice     = "htdemucs"
        app.quantize_enabled = True
        n_threads = os.cpu_count() or 4
        set_num_threads(n_threads)
        dev   = torch_device("cpu")
        model = app._load_demucs_model("htdemucs", dev, False, True, n_threads)
        vocals_idx = model.sources.index('vocals')

        log_ram("  after model load")

        waveform  = fresh_s()
        stereo    = np.stack([waveform, waveform], axis=0)
        tensor    = from_numpy(stereo).float().unsqueeze(0)

        _log.info(f"  tensor shape: {list(tensor.shape)}")
        log_ram("  before apply_model")

        _result = [None]
        _err    = [None]

        def _worker():
            try:
                with no_grad():
                    _result[0] = apply_model(
                        model, tensor,
                        device=dev,
                        segment=6,       # 6 s internal segments — limits peak RAM
                        overlap=0.25,
                        progress=False,
                        num_workers=0,
                    )
            except Exception as e:
                _err[0] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join()
        if _err[0]:
            raise _err[0]

        vocals = _result[0][0, vocals_idx].float().mean(0).cpu().numpy()
        log_ram("  after apply_model")
        _assert_audio(vocals, len(vocals), "DEMUCS htdemucs")

        out_path = os.path.join(REPO, "test_demucs_htdemucs_output.wav")
        sf.write(out_path, vocals, sr_s)
        _log.info(f"  output written: {out_path}")

        del _result[0], tensor, stereo, waveform, model
        app._MODEL_CACHE.clear()
        gc.collect()
        log_ram("  after cleanup")

    run("DEMUCS htdemucs  (6 s internal segments, INT8, CPU)", test_demucs_htdemucs)

    # ── E2: MDX-Net ──────────────────────────────────────────────────────────
    def test_mdxnet():
        if not app.AUDIO_SEPARATOR_AVAILABLE:
            return "skip"
        if ram_guard("MDX-Net UVR-MDX-NET-Voc_FT"):
            return "skip"
        out = app.separate_with_audio_separator(
            fresh_s(), sr_s, mdl="UVR-MDX-NET-Voc_FT.onnx")
        _assert_audio(out, len(fresh_s()), "MDX-Net")
        out_path = os.path.join(REPO, "test_mdxnet_output.wav")
        sf.write(out_path, out, sr_s)
        _log.info(f"  output written: {out_path}")
        app._AUDIO_SEP_CACHE.clear()
        gc.collect()
        log_ram("  after cleanup")

    run("MDX-Net UVR-MDX-NET-Voc_FT  (60 s chunks, audio-separator)", test_mdxnet)

    # ── E3: Spleeter ─────────────────────────────────────────────────────────
    def test_spleeter():
        if not app.SPLEETER_AVAILABLE:
            return "skip"
        if ram_guard("Spleeter 4-stems"):
            return "skip"
        out = app.separate_with_spleeter(fresh_s(), sr_s)
        assert out is not None and len(out) > 0
        out_path = os.path.join(REPO, "test_spleeter_output.wav")
        sf.write(out_path, out, sr_s)
        _log.info(f"  output written: {out_path}")
        gc.collect()
        log_ram("  after cleanup")

    run("Spleeter 4-stems  (60 s chunks, TensorFlow)", test_spleeter)


# ═══════════════════════════════════════════════════════════════════════════════
# F — OpenVINO provider check
# ═══════════════════════════════════════════════════════════════════════════════
_log.info(f"\n{B}[F] OpenVINO provider{E}")

def test_openvino():
    try:
        import onnxruntime as ort
    except ImportError:
        return "skip"
    if "OpenVINOExecutionProvider" not in ort.get_all_providers():
        return "skip"
    app.openvino_enabled = True
    app.OV_AVAILABLE = True
    prov = app._get_onnx_providers()
    app.openvino_enabled = False
    assert isinstance(prov[0], tuple), f"expected (name, opts) tuple, got {prov[0]}"
    assert prov[0][0] == "OpenVINOExecutionProvider"
    assert prov[-1] == "CPUExecutionProvider", "CPU fallback missing"
    _log.info(f"  providers: {[p if isinstance(p,str) else p[0] for p in prov]}")

run("OpenVINO in ort.get_all_providers() + _get_onnx_providers() dispatch",
    test_openvino)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
total = sum(results.values())
_log.info("")
_log.info("=" * 50)
_log.info(f"RESULTS:  {results['pass']} passed  "
          f"{results['skip']} skipped  "
          f"{results['fail']} failed  / {total} total")
_log.info(f"Log saved: {log_path}")
_log.info("=" * 50)

if results["fail"] > 0:
    _log.error("Some tests failed — see log above.")
    sys.exit(1)
else:
    _log.info("All runnable tests passed.")
