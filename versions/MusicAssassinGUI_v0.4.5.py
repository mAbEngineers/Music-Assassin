"""
Music Assassin Pro v0.4.5 - MDX-NET / DEMUCS / SPLEETER EDITION
Advanced triple-source separation with intelligent pre/post-filtering
+ Choose between MDX-Net (vocals-only, fast CPU), DEMUCS (natural) or Spleeter (4-stems)
+ 3 processing modes (Normal/Balanced/Aggressive)
+ Advanced pre-filtering (noise gate, high-pass)
+ Advanced post-filtering (de-esser, harmonic rejection, formant enhancement)
+ Optimized chunked processing (low memory usage)
+ Compact formal UI with real-time feedback

v0.4.5 additions (over v0.4.4):
+ Fast separation engine (fast_separation.py): Voc_FT on OpenVINO (BF16 where
  the CPU supports it, onnxruntime fallback) — the new default separator,
  ~4× faster. A 10-min episode runs in about 58 s end to end on a 16-core CPU.
+ Keep sound effects (Off / Fast MRX / Quality CDX23) with an effects level
  control; effects are separated in parallel and mixed back after the VAD gate.
+ "MRX dialogue" fastest separator option.
+ Crash logging so failures leave evidence in the log.

v0.4.4 additions (over v0.4.2):
+ VAD Gate (Silero VAD) — silences non-vocal gaps after separation; removes
  residual music bleed between lyric phrases. Torch-based, CPU-light (~1 MB model).
+ DeepFilterNet enhancement — learned speech denoiser applied after separation;
  cleans up residual artifacts and room noise. Optional: pip install deepfilternet.
  Pipeline order: DEMUCS/MDX-Net → [VAD gate] → DSP post-filter → [DeepFilter]

v0.4.4 audio fixes (inherited):
+ FIX: global post-filter normalization (was per-chunk → caused static at end)
+ FIX: tail fade-out (last 0.5 s → silent) to remove demucs tail noise
+ FIX: last-chunk fade-to-zero padding to reduce model boundary artifacts
"""

import os, sys, gc, time, logging, math, threading

# Set Tcl/Tk library paths if not already set (Linux only — paths differ per OS)
os.environ.setdefault("TCL_LIBRARY", "/usr/share/tcltk/tcl8.6")
os.environ.setdefault("TK_LIBRARY", "/usr/share/tcltk/tk8.6")

from tkinter import filedialog, messagebox, Label, Frame, Scrollbar, Listbox, Button, Tk, Entry, Checkbutton, BooleanVar, PhotoImage
from tkinter.scrolledtext import ScrolledText
from tkinter.ttk import Progressbar, Combobox, Style
from scipy.signal import butter, sosfilt
import numpy as np
from numpy import frombuffer, zeros, swapaxes, newaxis, ceil
import ffmpeg as _ffmpeg
from ffmpeg import input as ffmpeg_input, probe

# Lightweight availability checks — no heavy ML libraries loaded at startup.
# torch/demucs and spleeter/tensorflow are imported on first use inside the
# separator functions, saving ~1.5 GB of RAM at GUI launch.
import importlib.util
DEMUCS_AVAILABLE = (
    importlib.util.find_spec('torch') is not None and
    importlib.util.find_spec('demucs') is not None
)
def _check_spleeter():
    """Cheap package-presence check only — does NOT import tensorflow.
    Importing tensorflow costs 500MB-1.5GB+ RSS, which every process would
    otherwise pay just to populate a UI dropdown. The real tf.estimator
    compatibility check happens lazily in separate_with_spleeter() the first
    time Spleeter is actually selected and run."""
    if importlib.util.find_spec('spleeter') is None:
        return False
    if importlib.util.find_spec('tensorflow') is None:
        return False
    return True

SPLEETER_AVAILABLE = _check_spleeter()
AUDIO_SEPARATOR_AVAILABLE = importlib.util.find_spec('audio_separator') is not None
if not DEMUCS_AVAILABLE:
    print("WARNING: DEMUCS not installed (torch/torchaudio). Spleeter only.")
if not SPLEETER_AVAILABLE:
    print("WARNING: Spleeter not installed. DEMUCS only.")
if not AUDIO_SEPARATOR_AVAILABLE:
    print("WARNING: audio-separator not installed. MDX-Net models unavailable.")

# Fast engine (fast_separation.py): Voc_FT on OpenVINO BF16/FP32, MRX and
# CDX23 dialogue/music/effects. Imported lazily like the other separators.
try:
    import fast_separation
    FAST_ENGINE_AVAILABLE = fast_separation.is_available()
except ImportError:
    fast_separation = None
    FAST_ENGINE_AVAILABLE = False
if not FAST_ENGINE_AVAILABLE:
    print("INFO: fast engine unavailable (needs torch + openvino or onnxruntime).")

VAD_AVAILABLE    = importlib.util.find_spec('torch') is not None        # Silero VAD uses torch hub
DF_AVAILABLE     = importlib.util.find_spec('df') is not None           # pip install deepfilternet
SHERPA_AVAILABLE = importlib.util.find_spec('sherpa_onnx') is not None  # pip install sherpa-onnx
OV_AVAILABLE     = False
try:
    import onnxruntime as _ort_probe
    OV_AVAILABLE = 'OpenVINOExecutionProvider' in _ort_probe.get_all_providers()
    del _ort_probe
except Exception:
    pass
if not VAD_AVAILABLE:
    print("INFO: torch not found — Silero VAD unavailable.")
if not DF_AVAILABLE:
    print("INFO: deepfilternet not installed — DeepFilter unavailable. Run: pip install deepfilternet")
if not SHERPA_AVAILABLE:
    print("INFO: sherpa-onnx not installed — DPDFNet/GTCRN unavailable. Run: pip install sherpa-onnx")
if not OV_AVAILABLE:
    print("INFO: OpenVINO provider not found — using CPUExecutionProvider. Run: pip install onnxruntime-openvino")

# ============================================================================
# GLOBALS & CONFIG
# ============================================================================

files = []
process1 = process2 = separator = do = 0
sample_rate = 44100
b1 = b2 = f1 = a1 = listbox = inprogressframe = progress = status_label = mode_selector = separator_selector = quantize_var = vad_var = df_var = enh_selector = ov_var = tgate_var = fx_selector = fx_level_selector = None
root = None
status_text = "Ready"
processing_mode = "balanced"  # normal, balanced, aggressive
# Must match the first separator_selector option, which is what the UI shows
# at startup -- see setFrames(), which calls set_separator() once.
separator_choice = "demucs"   # fast_mdx | mrx | audio_separator | demucs | spleeter
model_choice = "htdemucs"     # htdemucs | htdemucs_ft | mdx_extra | UVR-MDX-NET-Voc_FT.onnx | Kim_Vocal_2.onnx
fast_overlap = 0.25           # fast_mdx window overlap: 0.25 = same output as MDX-Net, 0.1 = faster
# Keep sound effects: "off" | "mrx" | "cdx23". Extracts an effects stem from
# the original audio in parallel with de-musicing and mixes it back in.
fx_choice = "off"
fx_level_db = 0.0             # effects level relative to the original mix; set_fx() applies per-model defaults
_last_post_gain = 1.0         # gain applied by advanced_postprocess_chunked's normalize
quantize_enabled  = True      # INT8 dynamic quantization (htdemucs only, 2-3x faster on CPU)
vad_enabled       = False     # Silero VAD gate — silence non-vocal gaps
# Enhancer: "none" | "deepfilter" | "dpdfnet" | "gtcrn" | "speechdenoiser"
enhancer_choice       = "none"
openvino_enabled      = False   # Use OpenVINO provider for ONNX enhancer inference
transient_gate_enabled = False  # Apply spectral transient gate after enhancer

# Session model cache — key "name:quant" → loaded torch model.
# Stays alive for the whole session so batch runs don't reload weights per file.
_MODEL_CACHE = {}

# Silero VAD session cache (loaded on first use via torch hub)
_silero_vad_model = None
_silero_get_ts    = None

# ── Logging setup ────────────────────────────────────────────────────────────
# The log must outlive the process. Writing it next to __file__ is wrong in a
# PyInstaller onefile build: __file__ then lives under sys._MEIPASS, a temp dir
# the bootloader deletes on exit -- so every crash took its own log with it.
# Keep it in the XDG state dir instead, which survives the process.
def _resolve_log_dir():
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    for cand in (os.path.join(base, "music-assassin"),
                 os.path.join(os.path.expanduser("~"), ".music-assassin"),
                 os.path.dirname(os.path.abspath(__file__))):
        try:
            os.makedirs(cand, exist_ok=True)
            _probe = os.path.join(cand, ".writable")
            with open(_probe, "w"):
                pass
            os.unlink(_probe)
            return cand
        except OSError:
            continue
    return os.path.abspath(".")


_LOG_PATH = os.path.join(_resolve_log_dir(), "music_assassin.log")

# stderr as well as the file. traceback.print_exc() writes to stderr, and
# sys.stdout is rebound to the GUI widget later, so without this handler a
# crash leaves no trace either on screen or on disk.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.__stderr__),
    ],
)
_log = logging.getLogger("MusicAssassin")


def _log_uncaught(exc_type, exc_value, exc_tb):
    """Last resort: an uncaught exception has to reach the log file."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    _log.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))


sys.excepthook = _log_uncaught

# Separation runs on a worker thread, and a thread killed by an exception never
# reaches sys.excepthook -- it needs its own hook.
import threading as _threading


def _log_uncaught_thread(args):
    if issubclass(args.exc_type, SystemExit):
        return
    _log.critical("Uncaught exception in thread %s",
                  getattr(args.thread, "name", "?"),
                  exc_info=(args.exc_type, args.exc_value, args.exc_traceback))


_threading.excepthook = _log_uncaught_thread

# ffmpeg-python and torch log a lot of noise at WARNING level — keep those quiet
# in our log file so only real warnings from our own code appear.
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

_log.info("=" * 60)
_log.info("Music Assassin v0.4.5  —  session start")
_log.info(f"Log file: {_LOG_PATH}")
_log.info(f"DEMUCS available: {DEMUCS_AVAILABLE}  |  Spleeter: {SPLEETER_AVAILABLE}  |  MDX-Net: {AUDIO_SEPARATOR_AVAILABLE}"
          f"  |  Fast engine: {FAST_ENGINE_AVAILABLE}"
          f" ({'openvino' if importlib.util.find_spec('openvino') else 'onnxruntime'})")


class PrintLogger:
    """Routes print() calls to both the GUI textbox and the log file.

    Lines that look like ffmpeg banner lines (start with spaces, contain
    'built with' / 'configuration:' / 'lib') are filtered from the GUI widget
    so the textbox shows only app-level messages, while the full detail still
    goes to music_assassin.log.
    """

    # Prefixes we skip in the GUI but still write to the log file
    _GUI_SKIP = (
        "  built with", "  configuration:", "  lib", "Press [q]",
        "Stream mapping:", "Output #", "Input #", "  Stream #",
        "  Metadata:", "  Duration:", "  major_brand", "  minor_version",
        "  compatible_brands", "  encoder", "  handler_name", "  vendor_id",
        "[out#", "[in#", "[aac", "[aist", "size=", "Guessed Channel",
        "ffmpeg version",
    )

    def __init__(self, textbox):
        self.textbox = textbox
        self._buf = []

    def write(self, text):
        if not text:
            return
        # Always write to log file
        for line in text.splitlines():
            if line.strip():
                _log.info(line)

        # Only forward non-ffmpeg-banner lines to the GUI widget
        if not any(text.startswith(p) for p in self._GUI_SKIP):
            self._buf.append(text)
            if len(self._buf) >= 5 or "\n" in text:
                self._flush_to_widget()

    def _flush_to_widget(self):
        if not self._buf:
            return
        combined = "".join(self._buf)
        self._buf = []
        self.textbox.configure(state="normal")
        self.textbox.insert("end", combined)
        self.textbox.see("end")
        self.textbox.configure(state="disabled")

    def flush(self):
        self._flush_to_widget()

def update_status(message):
    """Update status label"""
    global status_text
    status_text = message
    if status_label:
        status_label.config(text=f"  {message}")
        root.update_idletasks()

def update_progress(value, total):
    """Update progress bar"""
    if progress:
        progress['value'] = (value / total) * 100
        root.update_idletasks()

# ============================================================================
# AUDIO DSP PIPELINE (Pre-Processing)
# ============================================================================

def audio_prefilter(waveform, sr=44100, gate=False):
    """Advanced pre-filtering: (optional gate) + high-pass + normalize.
    The per-sample -40 dBFS gate is off by default and no caller enables it:
    it wipes out quiet ambience, and since it runs before normalization it
    distorts dialogue in quiet files (A2 audit, quality_harness/)."""
    update_status("Pre-filtering audio...")
    print("Pre-filtering: " + ("gate + " if gate else "") + "high-pass + normalize")

    # Stage 1: Noise gate (< -40dB removed)
    if gate:
        threshold = 10**(-40/20)  # -40dB
        waveform = waveform * (np.abs(waveform) > threshold)

    # Stage 2: High-pass filter (remove subsonic rumble < 40Hz)
    sos_hp = butter(4, 40/(sr/2), btype='highpass', output='sos')
    waveform = sosfilt(sos_hp, waveform)

    # Stage 3: Normalize
    max_val = np.max(np.abs(waveform))
    if max_val > 1e-10:
        waveform = waveform / max_val * 0.98

    return waveform.astype(np.float32)

# ============================================================================
# AUDIO DSP PIPELINE (Post-Processing with Modes)
# ============================================================================

def advanced_postprocess_chunked(vocals, sr=44100, mode='balanced', chunk_size=None):
    """Apply selected post-processing mode with memory-efficient chunking"""
    print(f"\nAdvanced post-processing ({mode} mode - CHUNKED)...")
    update_status(f"Post-processing: {mode} mode...")

    if chunk_size is None:
        chunk_size = int(sr * 30)  # 30 second chunks
    overlap_size = int(sr * 2)  # 2 second overlap

    print(f"Processing in {chunk_size//sr}s chunks with {overlap_size//sr}s overlap...")

    num_chunks = int(np.ceil(len(vocals) / (chunk_size - overlap_size)))
    output = np.zeros_like(vocals, dtype=np.float32)

    for chunk_idx in range(num_chunks):
        update_progress(chunk_idx, num_chunks)
        update_status(f"Post-processing: chunk {chunk_idx+1}/{num_chunks}")

        start = chunk_idx * (chunk_size - overlap_size)
        end = min(start + chunk_size, len(vocals))
        chunk = vocals[start:end].copy()

        # Apply mode-specific processing
        if mode == 'normal':
            chunk = _postprocess_chunk_normal(chunk, sr)
        elif mode == 'balanced':
            chunk = _postprocess_chunk_balanced(chunk, sr)
        elif mode == 'aggressive':
            chunk = _postprocess_chunk_aggressive(chunk, sr)

        # Handle overlap-add for smooth transitions
        if chunk_idx == 0:
            output[start:start+len(chunk)] = chunk
        else:
            fade_len = min(overlap_size, len(chunk))
            if fade_len > 0:
                fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
                fade_out = 1 - fade_in
                output[start:start+fade_len] = (
                    output[start:start+fade_len] * fade_out +
                    chunk[:fade_len] * fade_in
                )
                output[start+fade_len:start+len(chunk)] = chunk[fade_len:]

        del chunk
        gc.collect()
        root.update_idletasks()

    # Global peak-normalize across the full output — avoids quiet tail chunks
    # being amplified to full scale (the cause of static at the end of files).
    # The gain is kept so the effects stem can be mixed back at the same level.
    global _last_post_gain
    _last_post_gain = 1.0
    peak = np.max(np.abs(output))
    if peak > 1e-10:
        _last_post_gain = 0.98 / peak
        output *= _last_post_gain

    # Fade the last 0.5 s to silence — removes demucs tail noise/artifacts.
    fade_samps = min(int(sr * 0.5), len(output))
    if fade_samps > 0:
        output[-fade_samps:] *= np.linspace(1.0, 0.0, fade_samps, dtype=np.float32)

    print(f"Post-processing complete")
    return output

def mix_effects(vocals, effects, gain, sr=44100, ceiling=0.98, block=512):
    """Add the effects stem to the processed dialogue, then limit peaks.

    The limiter works on 512-sample blocks: each block's gain is the lowest
    of its own and its neighbours' required gain, interpolated linearly
    between block centres, so no sample exceeds the ceiling and a single loud
    effect only ducks the ~30 ms around it instead of the whole file.
    """
    print(f"Mixing sound effects back in (gain {20 * np.log10(max(gain, 1e-10)):+.1f} dB)...")
    update_status("Mixing sound effects back in...")
    n = min(len(vocals), len(effects))
    out = vocals[:n] + effects[:n].astype(np.float32) * np.float32(gain)

    # Same tail fade as advanced_postprocess_chunked applied to the dialogue.
    fade_samps = min(int(sr * 0.5), n)
    if fade_samps > 0:
        out[-fade_samps:] *= np.linspace(1.0, 0.0, fade_samps, dtype=np.float32)

    nb = int(np.ceil(n / block))
    padded = np.zeros(nb * block, np.float32)
    padded[:n] = np.abs(out)
    peaks = padded.reshape(nb, block).max(axis=1)
    g = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-10))
    if g.min() < 1.0:
        g = np.minimum(g, np.minimum(np.r_[g[1:], 1.0], np.r_[1.0, g[:-1]]))
        centres = (np.arange(nb) + 0.5) * block
        step = 1 << 20   # in slices: a whole-file index array costs 8 bytes/sample
        for s in range(0, n, step):
            e = min(s + step, n)
            out[s:e] *= np.interp(np.arange(s, e), centres, g).astype(np.float32)
        print(f"  Limiter: {np.mean(g < 1.0) * 100:.1f}% of blocks reduced, "
              f"max {-20 * np.log10(g.min()):.1f} dB")
    return out

def _postprocess_chunk_normal(chunk, sr):
    """Normal mode: minimal processing"""
    sos_hp = butter(4, 80/(sr/2), btype='highpass', output='sos')
    sos_lp = butter(4, 10000/(sr/2), btype='lowpass', output='sos')
    chunk = sosfilt(sos_hp, chunk)
    chunk = sosfilt(sos_lp, chunk)
    # No per-chunk normalization — global normalize applied after all chunks
    return chunk.astype(np.float32)

def _postprocess_chunk_balanced(chunk, sr):
    """Balanced mode: gentle cleanup only. htdemucs already handles instrument/vocal
    separation — additional notch filters and spectral subtraction add more artifacts
    than they remove."""
    # Sub-bass removal (below vocal range)
    sos_low = butter(4, 40/(sr/2), btype='highpass', output='sos')
    chunk = sosfilt(sos_low, chunk)

    # Gentle de-esser: attenuate (not subtract) the 5-8kHz band only when it's
    # abnormally loud. Attenuation avoids the spectral subtraction "musical noise"
    # artifact that subtraction causes.
    sos_sib = butter(4, [5000/(sr/2), 8000/(sr/2)], btype='band', output='sos')
    sibilants = sosfilt(sos_sib, chunk)
    sib_rms = np.sqrt(np.mean(sibilants**2) + 1e-10)
    if sib_rms > 0.20:
        reduction = np.clip(0.20 / sib_rms, 0.5, 1.0)
        # Apply only via gain on the isolated band and blend back
        chunk = chunk - sibilants * (1.0 - reduction) * 0.4

    # Mild vocal presence boost (1-3kHz)
    sos_f = butter(2, [1000/(sr/2), 3000/(sr/2)], btype='band', output='sos')
    formants = sosfilt(sos_f, chunk)
    chunk = chunk + formants * 0.15

    return chunk.astype(np.float32)

def _postprocess_chunk_aggressive(chunk, sr):
    """Aggressive mode: stronger cleanup than balanced, still avoids spectral subtraction."""
    # Wider sub-bass cut
    sos_low = butter(4, 60/(sr/2), btype='highpass', output='sos')
    chunk = sosfilt(sos_low, chunk)

    # Stronger de-esser: attenuate 5-8kHz band when overly prominent
    sos_sib = butter(4, [5000/(sr/2), 8000/(sr/2)], btype='band', output='sos')
    sibilants = sosfilt(sos_sib, chunk)
    sib_rms = np.sqrt(np.mean(sibilants**2) + 1e-10)
    if sib_rms > 0.15:
        reduction = np.clip(0.15 / sib_rms, 0.4, 1.0)
        chunk = chunk - sibilants * (1.0 - reduction) * 0.5

    # Vocal presence boost (1-3.5kHz)
    sos_f = butter(2, [1000/(sr/2), 3500/(sr/2)], btype='band', output='sos')
    formants = sosfilt(sos_f, chunk)
    chunk = chunk + formants * 0.25

    return chunk.astype(np.float32)

# ============================================================================
# SOURCE SEPARATION (DEMUCS or SPLEETER)
# ============================================================================

def _run_in_background(fn, status_msg):
    """Run fn() in a background thread, animating the progress bar until it finishes.
    Returns fn's return value, or re-raises its exception on the main thread."""
    import threading

    result = [None]
    error  = [None]

    def worker():
        try:
            result[0] = fn()
        except Exception as e:
            error[0] = e

    if progress:
        progress.configure(mode='indeterminate')
        progress.start(40)          # 40 ms animation tick
    update_status(status_msg)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while t.is_alive():
        root.update()               # keeps UI alive + drives progress animation
        time.sleep(0.033)           # ~30 fps poll
    t.join()

    if progress:
        progress.stop()
        progress.configure(mode='determinate')
        progress['value'] = 0

    if error[0]:
        raise error[0]
    return result[0]


def _load_demucs_model(name, dev, use_fp16, do_quantize, n_threads):
    """Load a DEMUCS model, caching it for the whole session.

    Speed features:
      * htdemucs (default) is a single transformer model — far faster on CPU
        than mdx_extra's 4-model bag.
      * The loaded model is cached so batch runs reuse weights instead of
        reloading (~5-15 s) per file.
      * INT8 dynamic quantization (htdemucs only, CPU only) gives ~2-3x speedup
        on the Linear layers with negligible quality loss.
    """
    cache_key = f"{name}:{'q' if (do_quantize and 'htdemucs' in name) else 'fp32'}:{dev.type}"
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        print(f"  Model '{cache_key}' reused from cache (no reload)")
        return cached

    from demucs.pretrained import get_model
    print(f"Loading DEMUCS model '{name}'...")
    update_status(f"Loading {name}...")

    model = get_model(name)
    model.eval()
    model.to(dev)

    if use_fp16:
        model = model.half()
    elif do_quantize and 'htdemucs' in name:
        # INT8 dynamic quantization — only meaningful on CPU, only htdemucs has
        # the Linear (transformer) layers that benefit.
        try:
            import torch
            torch.set_num_threads(n_threads)
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("  INT8 quantization applied (Linear layers)")
        except Exception as ex:
            print(f"  Quantization failed ({ex}); using FP32")

    _MODEL_CACHE[cache_key] = model
    print(f"  Cached as '{cache_key}'")
    return model


def separate_with_demucs(waveform, sr=44100):
    """Vocal extraction using DEMUCS v4 with manual pre-chunking.

    The waveform is split into CHUNK_SEC chunks BEFORE creating any tensor,
    so peak RAM stays bounded regardless of file length or model size.
    htdemucs_ft and mdx_extra are both BagOfModels with 4 sub-models — they
    collapsed on long files because the whole-file tensor was in RAM the whole
    time. Chunking fixes that: each chunk is processed and freed individually.
    """
    if not DEMUCS_AVAILABLE:
        print("ERROR: DEMUCS not available")
        return None

    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        print(f"  Available RAM: {available_gb:.1f} GB")
        if available_gb < 0.5:
            print(f"ERROR: Not enough RAM ({available_gb:.1f} GB free). Need at least 0.5 GB.")
            update_status(f"ERROR: Low RAM ({available_gb:.1f} GB). Free up memory first.")
            return None
    except ImportError:
        pass

    try:
        from torch import set_num_threads, device as torch_device, from_numpy, no_grad
        from torch.cuda import is_available as cuda_available, empty_cache as cuda_empty_cache
        from demucs.apply import apply_model
        import threading

        n_threads = os.cpu_count() or 4
        set_num_threads(n_threads)
        print(f"  PyTorch threads: {n_threads}")

        dev = torch_device("cuda" if cuda_available() else "cpu")
        use_fp16 = (dev.type == "cuda")

        model = _load_demucs_model(model_choice, dev, use_fp16, quantize_enabled, n_threads)

        if use_fp16:
            print(f"  Device: {dev}  (fp16 enabled — ~2x faster)")
        else:
            print(f"  Device: CPU  |  model: {model_choice}"
                  f"{'  (INT8)' if (quantize_enabled and 'htdemucs' in model_choice) else ''}")
            print(f"  TIP: Install CUDA + PyTorch-GPU for 10-20x speedup.")

        vocals_idx = model.sources.index('vocals')
        duration_s = len(waveform) / sr

        # Manual pre-chunking — keeps peak RAM bounded for all model sizes.
        # htdemucs_ft / mdx_extra are 4-model bags: without this they load the
        # whole-song tensor × 4 sub-models simultaneously and OOM on low-RAM machines.
        CHUNK_SEC   = 60
        OVERLAP_SEC = 2
        chunk_samp  = int(CHUNK_SEC * sr)
        overlap_samp = int(OVERLAP_SEC * sr)
        step_samp   = chunk_samp - overlap_samp

        total_samp  = len(waveform)
        starts      = list(range(0, total_samp, step_samp))
        n_chunks    = len(starts)
        vocals_out  = np.zeros(total_samp, dtype=np.float32)

        print(f"  duration: {duration_s:.1f}s  |  chunks: {n_chunks} × {CHUNK_SEC}s  |  overlap: {OVERLAP_SEC}s  |  device: {dev}")

        if progress:
            progress.configure(mode='determinate')
            progress['value'] = 0

        t0 = time.time()

        for idx, start in enumerate(starts):
            end   = min(start + chunk_samp, total_samp)
            chunk = waveform[start:end]

            stereo = np.stack([chunk, chunk], axis=0)
            t = from_numpy(stereo).float().unsqueeze(0).to(dev)
            if use_fp16:
                t = t.half()

            _result = [None]
            _err    = [None]

            def _worker(tensor=t):
                try:
                    with no_grad():
                        # No segment= here, deliberately. CHUNK_SEC is OUR outer
                        # chunk — a bound on the numpy array we hold in RAM.
                        # demucs's `segment` is the model's INTERNAL context
                        # window, and conflating the two crashes htdemucs: it is
                        # a transformer with a fixed training_length of 7.8 s, so
                        # segment=60 makes apply_model pad the input out to 60 s
                        # and then reshape it back to 7.8 s, raising
                        #   shape '[1, 4, -1, 343980]' is invalid for input of
                        #   size 21168000                (= 1*4*2*60*44100)
                        # Omitting it lets each model use its native
                        # model.segment and sub-split this chunk itself, which is
                        # both correct and far cheaper — attention cost grows
                        # quadratically with the context length.
                        _result[0] = apply_model(
                            model, tensor,
                            device=dev,
                            overlap=0.25,
                            progress=False,
                            num_workers=0,
                        )
                except Exception as e:
                    _err[0] = e

            chunk_dur = (end - start) / sr
            update_status(f"DEMUCS: chunk {idx+1}/{n_chunks}  ({chunk_dur:.0f}s)...")
            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()

            # Animate progress while chunk processes
            est_chunk = chunk_dur * 0.75
            t_chunk = time.time()
            while thread.is_alive():
                elapsed_chunk = time.time() - t_chunk
                chunk_frac = min(elapsed_chunk / max(est_chunk, 1), 0.99)
                overall_frac = (idx + chunk_frac) / n_chunks
                if progress:
                    progress['value'] = overall_frac * 100
                eta_total = max(0, int((time.time() - t0) / max(overall_frac, 0.01) * (1 - overall_frac)))
                eta_str = f"{eta_total // 60}m {eta_total % 60:02d}s" if eta_total >= 60 else f"{eta_total}s"
                update_status(f"DEMUCS: chunk {idx+1}/{n_chunks}  |  {overall_frac*100:.0f}%  |  ETA: {eta_str}")
                root.update()
                time.sleep(0.033)

            thread.join()
            if _err[0]:
                raise _err[0]

            chunk_vocals = _result[0][0, vocals_idx].float().mean(0).cpu().numpy()
            del _result[0], t, stereo

            # Overlap-add: crossfade the overlap region with the previous chunk
            if idx == 0:
                vocals_out[start:end] = chunk_vocals
            else:
                fade_len = min(overlap_samp, len(chunk_vocals), end - start)
                fade_in  = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                fade_out = 1.0 - fade_in
                vocals_out[start:start + fade_len] = (
                    vocals_out[start:start + fade_len] * fade_out +
                    chunk_vocals[:fade_len] * fade_in
                )
                vocals_out[start + fade_len:end] = chunk_vocals[fade_len:end - start]

            del chunk_vocals, chunk
            if dev.type == "cuda":
                cuda_empty_cache()
            gc.collect()

        elapsed = time.time() - t0
        print(f"  Separation done in {elapsed:.1f}s ({elapsed/duration_s:.2f}× real-time)")

        if progress:
            progress['value'] = 100

        return vocals_out

    except Exception as e:
        print(f"DEMUCS error: {e}")
        import traceback; traceback.print_exc()
        return None


def separate_with_spleeter(waveform, sr=44100):
    """Vocal extraction using Spleeter 4-stems with manual pre-chunking.

    Spleeter loads the entire waveform into TensorFlow at once and expands it
    to ~400-800 MB of tensors. Without chunking, long files OOM. We split into
    60s chunks, run Spleeter on each, and overlap-add the vocals back.
    """
    if not SPLEETER_AVAILABLE:
        print("ERROR: Spleeter not available")
        return None

    print("Loading Spleeter 4-stems model...")
    update_status("Loading Spleeter model...")

    try:
        try:
            import tensorflow as tf
            if not hasattr(tf, 'estimator'):
                print("ERROR: tf.estimator missing — Spleeter needs TensorFlow <2.16 "
                      "or a tf.estimator-compatible build.")
                return None
            n_threads = os.cpu_count() or 4
            tf.config.threading.set_intra_op_parallelism_threads(n_threads)
            tf.config.threading.set_inter_op_parallelism_threads(n_threads)
            print(f"  TensorFlow threads: {n_threads}")
        except Exception:
            pass

        from spleeter.separator import Separator
        separator = Separator('spleeter:4stems')

        CHUNK_SEC    = 60
        OVERLAP_SEC  = 2
        chunk_samp   = int(CHUNK_SEC * sr)
        overlap_samp = int(OVERLAP_SEC * sr)
        step_samp    = chunk_samp - overlap_samp

        total_samp   = len(waveform)
        starts       = list(range(0, total_samp, step_samp))
        n_chunks     = len(starts)
        vocals_out   = np.zeros(total_samp, dtype=np.float32)

        duration_s = total_samp / sr
        print(f"  duration: {duration_s:.1f}s  |  chunks: {n_chunks} × {CHUNK_SEC}s  |  overlap: {OVERLAP_SEC}s")

        if progress:
            progress.configure(mode='determinate')
            progress['value'] = 0

        for idx, start in enumerate(starts):
            end   = min(start + chunk_samp, total_samp)
            chunk = waveform[start:end].reshape(-1, 1)

            update_status(f"Spleeter: chunk {idx+1}/{n_chunks}...")

            _sep_result = [None]

            def _run_sep(c=chunk):
                _sep_result[0] = separator.separate(c)

            _run_in_background(_run_sep, f"Spleeter: chunk {idx+1}/{n_chunks}...")
            chunk_vocals = _sep_result[0]['vocals'].flatten()

            if idx == 0:
                vocals_out[start:end] = chunk_vocals
            else:
                fade_len = min(overlap_samp, len(chunk_vocals), end - start)
                fade_in  = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                fade_out = 1.0 - fade_in
                vocals_out[start:start + fade_len] = (
                    vocals_out[start:start + fade_len] * fade_out +
                    chunk_vocals[:fade_len] * fade_in
                )
                vocals_out[start + fade_len:end] = chunk_vocals[fade_len:end - start]

            del chunk_vocals, chunk, _sep_result[0]
            gc.collect()

            if progress:
                progress['value'] = (idx + 1) / n_chunks * 100

        separator.close()
        gc.collect()
        return vocals_out

    except Exception as e:
        print(f"Spleeter error: {e}")
        return None


# Session cache for audio-separator Separator instances (one per model filename).
_AUDIO_SEP_CACHE = {}

def separate_with_audio_separator(waveform, sr=44100, mdl="UVR-MDX-NET-Voc_FT.onnx"):
    """Vocal extraction using MDX-Net (UVR) models via audio-separator.

    audio-separator operates on files (not arrays), so each chunk is written
    to a temp WAV, run through the model, and the vocals output read back.
    Waveform is manually split into 60 s chunks with 2 s overlap-add so RAM
    stays bounded regardless of file length — the model processes one chunk at
    a time instead of the entire track.
    The Separator instance is cached per model so batch runs don't reload
    weights each time.
    """
    if not AUDIO_SEPARATOR_AVAILABLE:
        print("ERROR: audio-separator not installed. Run: pip install audio-separator onnxruntime")
        return None

    import tempfile, glob, soundfile as sf
    from audio_separator.separator import Separator

    print(f"MDX-Net: model={mdl}")
    update_status(f"MDX-Net: loading {mdl}...")

    try:
        base_tmp = tempfile.mkdtemp(prefix="music_assassin_mdx_")

        # Load (or reuse cached) Separator — always point at base_tmp.
        # We do NOT update output_dir per chunk: audio-separator ignores
        # dynamic reassignment and always writes to the dir set at init time.
        sep = _AUDIO_SEP_CACHE.get(mdl)
        if sep is None:
            sep = Separator(
                output_dir=base_tmp,
                output_format="WAV",
                normalization_threshold=0.9,
                output_single_stem="Vocals",
            )
            sep.load_model(model_filename=mdl)
            _AUDIO_SEP_CACHE[mdl] = sep
            print(f"  MDX-Net model '{mdl}' loaded and cached")
        else:
            # Redirect output to this run's base_tmp
            sep.output_dir = base_tmp
            print(f"  MDX-Net model '{mdl}' reused from cache")

        # ── Chunked separation ──────────────────────────────────────────────
        # Each chunk is written as chunk_{idx}.wav into base_tmp.
        # sep.separate() returns the output filenames (relative to output_dir).
        # We resolve them against base_tmp to get absolute paths.
        CHUNK_SEC    = 60
        OVERLAP_SEC  = 2
        chunk_samp   = int(CHUNK_SEC * sr)
        overlap_samp = int(OVERLAP_SEC * sr)
        step_samp    = chunk_samp - overlap_samp

        total_samp   = len(waveform)
        starts       = list(range(0, total_samp, step_samp))
        n_chunks     = len(starts)
        vocals_out   = np.zeros(total_samp, dtype=np.float32)
        duration_s   = total_samp / sr

        print(f"  duration: {duration_s:.1f}s  |  chunks: {n_chunks} × {CHUNK_SEC}s  |  overlap: {OVERLAP_SEC}s")

        for idx, start in enumerate(starts):
            end   = min(start + chunk_samp, total_samp)
            chunk = waveform[start:end]
            chunk_dur = (end - start) / sr

            update_status(f"MDX-Net: chunk {idx+1}/{n_chunks}  ({chunk_dur:.0f}s)...")
            print(f"  chunk {idx+1}/{n_chunks}  start={start}  end={end}  ({chunk_dur:.0f}s)")

            # Write chunk with a unique name so we can identify its output
            tmp_in = os.path.join(base_tmp, f"chunk_{idx:04d}.wav")
            sf.write(tmp_in, chunk.astype(np.float32), sr)

            # Capture output filenames from sep.separate() return value
            _sep_out = [None]
            def _run_sep(f=tmp_in):
                _sep_out[0] = sep.separate(f)

            _run_in_background(_run_sep, f"MDX-Net: chunk {idx+1}/{n_chunks}...")

            # Resolve returned filenames against base_tmp
            out_names = _sep_out[0] or []
            vocals_files = [
                os.path.join(base_tmp, os.path.basename(p))
                if not os.path.isabs(p) else p
                for p in out_names
                if "Vocals" in os.path.basename(p) or "vocals" in os.path.basename(p).lower()
            ]
            # Fallback: glob in case return value was empty or paths differ
            if not vocals_files:
                vocals_files = glob.glob(
                    os.path.join(base_tmp, f"chunk_{idx:04d}*(Vocals)*"))
            if not vocals_files:
                vocals_files = glob.glob(
                    os.path.join(base_tmp, f"chunk_{idx:04d}*[Vv]ocal*"))

            if not vocals_files:
                print(f"  WARNING: MDX-Net produced no vocals for chunk {idx+1} — using zeros")
                print(f"    sep.separate() returned: {out_names}")
                print(f"    files in base_tmp: {os.listdir(base_tmp)}")
                del chunk
                gc.collect()
                continue

            chunk_voc, out_sr = sf.read(vocals_files[0], dtype='float32')
            if chunk_voc.ndim > 1:
                chunk_voc = chunk_voc.mean(axis=1)

            # Overlap-add
            chunk_len = min(len(chunk_voc), end - start)
            if idx == 0:
                vocals_out[start:start + chunk_len] = chunk_voc[:chunk_len]
            else:
                fade_len = min(overlap_samp, chunk_len)
                fade_in  = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                fade_out = 1.0 - fade_in
                vocals_out[start:start + fade_len] = (
                    vocals_out[start:start + fade_len] * fade_out +
                    chunk_voc[:fade_len] * fade_in
                )
                if chunk_len > fade_len:
                    vocals_out[start + fade_len:start + chunk_len] = chunk_voc[fade_len:chunk_len]

            del chunk_voc, chunk
            # Remove processed chunk files from base_tmp
            for f in [tmp_in] + vocals_files:
                try: os.remove(f)
                except OSError: pass
            gc.collect()

        # Clean base temp dir
        try:
            import shutil
            shutil.rmtree(base_tmp, ignore_errors=True)
        except Exception:
            pass

        print(f"  MDX-Net separation complete: {total_samp} samples @ {sr} Hz")
        return vocals_out

    except Exception as e:
        print(f"MDX-Net error: {e}")
        import traceback; traceback.print_exc()
        return None


# ============================================================================
# VAD GATE  (Silero VAD via torch hub)
# ============================================================================

_silero_lock = threading.Lock()

def _get_silero_vad():
    """Load Silero VAD from torch hub, caching result in module globals.
    Thread-safe: preload_silero_vad() may be loading it already."""
    global _silero_vad_model, _silero_get_ts
    with _silero_lock:
        if _silero_vad_model is not None:
            print("  Silero VAD: reused from cache")
            return _silero_vad_model, _silero_get_ts
        import torch
        print("  Silero VAD: loading from torch hub (downloads ~1 MB on first use)...")
        model, utils = torch.hub.load(
            'snakers4/silero-vad', 'silero_vad',
            force_reload=False, trust_repo=True
        )
        _silero_vad_model = model
        _silero_get_ts    = utils[0]   # get_speech_timestamps
        print("  Silero VAD: loaded and cached")
        return _silero_vad_model, _silero_get_ts

def preload_silero_vad():
    """Start loading Silero in a background thread. The first load costs
    9-20 s, almost all of it Python imports inside torch.hub; separation
    runs in native code that releases the GIL, so the load hides behind it."""
    if not VAD_AVAILABLE or _silero_vad_model is not None:
        return
    def _load():
        try:
            _get_silero_vad()
        except Exception as e:
            print(f"  Silero VAD preload failed ({e}) -- will retry when needed")
    threading.Thread(target=_load, daemon=True).start()


def apply_vad_gate(vocals, sr=44100):
    """Silence non-speech regions using Silero VAD.

    Silero VAD analyses audio at 16 kHz. The detected speech timestamps are
    converted back to the original sample rate and used to build a smooth
    binary mask (with 80 ms padding and 15 ms fade edges around each segment)
    so the gate introduces no audible clicks.
    """
    if not VAD_AVAILABLE:
        print("  VAD gate: torch not available — skipped")
        return vocals
    try:
        import torch
        from scipy.signal import resample_poly

        update_status("VAD gate: analysing speech...")
        print("VAD gate: running Silero VAD...")

        # Resample to 16 kHz (Silero's expected rate)
        vad_sr = 16000
        g      = math.gcd(sr, vad_sr)
        v16k   = resample_poly(vocals, vad_sr // g, sr // g).astype('float32')

        # Load model (cached after first call)
        _run_in_background(_get_silero_vad, "Loading Silero VAD...")
        vad_model = _silero_vad_model
        get_ts    = _silero_get_ts

        tensor     = torch.from_numpy(v16k)
        timestamps = get_ts(
            tensor, vad_model,
            sampling_rate=vad_sr,
            threshold=0.25,               # low: Japanese/singing/high-pitched voice
            min_speech_duration_ms=80,
            min_silence_duration_ms=400,  # require 400 ms silence before closing
        )

        # ── Drop SFX/music false positives via per-frame peak score ───────────
        # Silero can misfire on anime SFX (threshold=0.25 is intentionally loose
        # for high-pitched voices).  Real speech always has ≥1 frame with score
        # ≥ 0.65; music/SFX false positives peak well below that.
        # Each segment is scored independently (state reset) so GRU carry-over
        # from prior genuine speech cannot inflate the score of a false positive.
        SFRAME = 512   # Silero frame size at 16 kHz
        # Mean score discriminates better than peak on DF-processed audio:
        # genuine speech segments score mean ≥ 0.65; SFX false positives
        # score mean ≈ 0.30 because only their first half has elevated scores.
        MIN_MEAN = 0.45
        keep, drop = [], []
        for t in timestamps:
            seg_audio = v16k[t['start']:t['end']]
            vad_model.reset_states()
            seg_probs = []
            for _i in range(0, len(seg_audio) - SFRAME + 1, SFRAME):
                _p = vad_model(
                    torch.from_numpy(seg_audio[_i:_i + SFRAME]).unsqueeze(0),
                    vad_sr).item()
                seg_probs.append(_p)
            mean_score = float(np.mean(seg_probs)) if seg_probs else 0.0
            (keep if mean_score >= MIN_MEAN else drop).append((t, mean_score))
        if drop:
            for t, ms in drop:
                print(f"  VAD: SFX/music false-positive dropped "
                      f"{t['start']/vad_sr:.2f}s–{t['end']/vad_sr:.2f}s "
                      f"(mean={ms:.2f})")
        timestamps = [t for t, _ in keep]

        # ── Silero mask ───────────────────────────────────────────────────────
        scale  = sr / vad_sr
        pad_s  = int(0.25  * sr)   # 250 ms padding around each segment
        fade_s = int(0.030 * sr)   # 30 ms fade edges
        vad_mask = np.zeros(len(vocals), dtype='float32')

        for ts in timestamps:
            s = max(0,           int(ts['start'] * scale) - pad_s)
            e = min(len(vocals), int(ts['end']   * scale) + pad_s)
            vad_mask[s:e] = 1.0
            fi = min(fade_s, e - s)
            vad_mask[s:s + fi] = np.linspace(0.0, 1.0, fi, dtype='float32')
            fo = min(fade_s, e - s)
            vad_mask[e - fo:e] = np.linspace(1.0, 0.0, fo, dtype='float32')

        # ── Energy-based fallback for high-pitched voices ─────────────────────
        # Silero misses F0 > ~300 Hz (high-pitched anime girl voices, falsetto,
        # singing). We only allow energy to EXPAND existing Silero segments —
        # it cannot open new speech regions in sections Silero marked silent.
        # This prevents music-only sections (no Silero speech nearby) from being
        # kept just because they have strong band energy.
        from scipy.signal import butter, sosfilt
        from scipy.ndimage import binary_dilation, uniform_filter1d

        sos_voice = butter(4, [200, 4000], btype='band', fs=sr, output='sos')
        voice_band = sosfilt(sos_voice, vocals)

        hop = max(1, int(0.010 * sr))   # 10 ms hops
        n_fr = (len(voice_band) + hop - 1) // hop
        band_energy = np.array([
            float(np.mean(voice_band[i*hop:(i+1)*hop]**2))
            for i in range(n_fr)
        ], dtype=np.float64)

        nonzero = band_energy[band_energy > 1e-10]
        if len(nonzero) and timestamps:
            # Build a per-frame Silero mask (unpadded segments, frame indices).
            # Threshold is anchored to the median energy of SILERO-DETECTED voiced
            # frames — not all non-silent frames. This prevents quiet music residuals
            # (which dominate the overall median) from clearing the bar.
            silero_frames = np.zeros(n_fr, dtype=bool)
            for ts in timestamps:
                fs = max(0,    int(ts['start'] * scale) // hop)
                fe = min(n_fr, int(ts['end']   * scale) // hop + 1)
                silero_frames[fs:fe] = True

            voiced_energy = band_energy[silero_frames]
            if len(voiced_energy) == 0:
                voiced_energy = nonzero
            # 10% of the median voiced-frame energy: keeps signals within one
            # order of magnitude of detected speech; blocks DF-attenuated music
            # (~2 orders of magnitude quieter than voice).
            e_thresh = float(np.median(voiced_energy)) * 0.10

            voiced_frames = (band_energy > e_thresh).astype(bool)
            # Fill short gaps (≤300 ms)
            fill_frames = max(1, int(0.30 * sr / hop))
            voiced_frames = binary_dilation(voiced_frames,
                                            structure=np.ones(fill_frames, dtype=bool))

            # Expand ±1.5 s around Silero segments; energy fallback only fires
            # within that window, never in pure-music regions far from any speech.
            expand_frames = max(1, int(1.5 * sr / hop))
            silero_expanded = binary_dilation(silero_frames,
                                              structure=np.ones(expand_frames * 2 + 1, dtype=bool))

            # Energy fallback only where Silero sees nearby speech
            voiced_frames = voiced_frames & silero_expanded

            energy_mask = np.repeat(voiced_frames, hop)[:len(vocals)].astype(np.float32)
            smooth = max(1, int(0.020 * sr))   # 20 ms smoothing
            energy_mask = uniform_filter1d(energy_mask.astype(np.float64),
                                           size=smooth, mode='nearest').astype(np.float32)
        else:
            energy_mask = np.zeros(len(vocals), dtype='float32')

        # ── Combine: keep frame if either detector says speech ────────────────
        mask = np.maximum(vad_mask, energy_mask).astype('float32')

        if not timestamps and energy_mask.max() < 0.1:
            print("  VAD: no speech detected — gate skipped")
            return vocals

        retained = mask.mean() * 100
        vad_pct  = vad_mask.mean() * 100
        eng_pct  = energy_mask.mean() * 100
        print(f"  VAD: {len(timestamps)} segments | "
              f"silero={vad_pct:.0f}%  energy={eng_pct:.0f}%  combined={retained:.0f}% retained")
        update_status(f"VAD: {len(timestamps)} segs, {retained:.0f}% retained")
        return (vocals * mask).astype('float32')

    except Exception as ex:
        print(f"  VAD gate error ({ex}) — skipped")
        return vocals


# ============================================================================
# DEEPFILTER ENHANCEMENT  (pip install deepfilternet)
# ============================================================================

def apply_deepfilter(vocals, sr=44100):
    """Post-separation speech enhancement using DeepFilterNet.

    Reduces residual musical noise and artefacts left by the separator.
    Resamples vocals to 48 kHz (DeepFilterNet's native rate), processes in
    60 s chunks with 2 s overlap-add so RAM stays bounded regardless of file
    length, then resamples back to the original rate.
    Requires: pip install deepfilternet
    """
    if not DF_AVAILABLE:
        print("  DeepFilter: not installed — skipped. Run: pip install deepfilternet")
        return vocals
    try:
        import torch
        from df import enhance, init_df
        from scipy.signal import resample_poly

        update_status("DeepFilter: enhancing vocals...")
        print("DeepFilter: initialising model...")

        def _load_df():
            return init_df()

        model_df, df_state, _ = _run_in_background(_load_df, "Loading DeepFilterNet...")
        df_sr = df_state.sr()
        print(f"  DeepFilter native sr: {df_sr} Hz")

        # ── Chunked processing ─────────────────────────────────────────────
        CHUNK_SEC    = 60
        OVERLAP_SEC  = 2
        chunk_samp   = int(CHUNK_SEC * sr)
        overlap_samp = int(OVERLAP_SEC * sr)
        step_samp    = chunk_samp - overlap_samp

        total_samp   = len(vocals)
        starts       = list(range(0, total_samp, step_samp))
        n_chunks     = len(starts)
        output       = np.zeros(total_samp, dtype=np.float32)

        g_up = math.gcd(sr, df_sr)
        g_dn = math.gcd(df_sr, sr)
        up_num, up_den = df_sr // g_up, sr // g_up
        dn_num, dn_den = sr // g_dn, df_sr // g_dn

        print(f"  DeepFilter: {total_samp/sr:.1f}s  |  {n_chunks} × {CHUNK_SEC}s chunks")

        for idx, start in enumerate(starts):
            end   = min(start + chunk_samp, total_samp)
            chunk = vocals[start:end].astype('float32')
            update_status(f"DeepFilter: chunk {idx+1}/{n_chunks}...")

            # Resample chunk → df_sr
            v_up     = resample_poly(chunk, up_num, up_den).astype('float32')
            tensor   = torch.from_numpy(v_up).unsqueeze(0)
            enhanced = enhance(model_df, df_state, tensor, atten_lim_db=20)
            v_enh    = enhanced.squeeze(0).numpy().astype('float32')

            # Resample back → original sr
            v_dn = resample_poly(v_enh, dn_num, dn_den).astype('float32')

            # Trim to expected length (resampling can add/remove a sample)
            expected = end - start
            v_dn = v_dn[:expected]
            if len(v_dn) < expected:
                v_dn = np.pad(v_dn, (0, expected - len(v_dn)))

            # Overlap-add
            if idx == 0:
                output[start:end] = v_dn
            else:
                fade_len = min(overlap_samp, len(v_dn))
                fade_in  = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                fade_out = 1.0 - fade_in
                output[start:start + fade_len] = (
                    output[start:start + fade_len] * fade_out + v_dn[:fade_len] * fade_in
                )
                output[start + fade_len:end] = v_dn[fade_len:]

            del chunk, v_up, v_enh, v_dn
            gc.collect()

        print("  DeepFilter: enhancement complete")
        update_status("DeepFilter: done")
        return output

    except Exception as ex:
        print(f"  DeepFilter error ({ex}) — skipped")
        return vocals


# ============================================================================
# SPECTRAL TRANSIENT GATE
# Post-process for short-duration high-frequency musical events (guitar/piano
# attack taps) that survive DeepFilterNet.  Operates in the HF band only so
# speech fundamentals and lower formants are completely unaffected.
# ============================================================================

def apply_spectral_transient_gate(audio, sr=44100,
                                   freq_floor_hz=2000,
                                   onset_threshold_db=8.0,
                                   gate_hold_ms=50,
                                   suppression_gain=0.10,
                                   bg_smooth_ms=200):
    """Detect and suppress short-duration high-frequency transients.

    Splits the signal into LF (below freq_floor_hz) and HF (above) using
    complementary IIR filters so LF + HF = input exactly.  Frame-level HF
    energy is compared to a smoothed running background; frames whose energy
    exceeds the background by onset_threshold_db are flagged as transients
    and the HF component is attenuated for gate_hold_ms afterward.
    LF is left untouched — speech fundamentals and formants survive.

    Parameters
    ----------
    freq_floor_hz : int
        Lower boundary of the gated band (Hz). Raise this if the gate cuts
        too much speech (e.g. 3000 Hz for maximum safety).
    onset_threshold_db : float
        dB above running average to trigger the gate.  Typical range 6–12 dB.
        Lower = catches quieter transients but risks false-triggering on speech.
    gate_hold_ms : float
        How long the gate stays active after an onset (ms).
        Should cover the full transient decay; 40–80 ms suits most plucked notes.
    suppression_gain : float
        HF gain during gating (0.0 = silence, 1.0 = bypass).
        0.10 = −20 dB is a good starting point; reduce for heavier suppression.
    bg_smooth_ms : float
        Background energy averaging window (ms).  Longer = more stable; shorter
        = more reactive.  Keep well above gate_hold_ms to avoid the gate
        suppressing its own reference level.
    """
    try:
        from scipy.signal import butter, sosfilt
        from scipy.ndimage import uniform_filter1d, binary_dilation

        update_status("Transient gate: detecting musical onsets...")
        print(f"Spectral transient gate: floor={freq_floor_hz} Hz | "
              f"thresh={onset_threshold_db} dB | hold={gate_hold_ms} ms | "
              f"suppress={20*np.log10(max(suppression_gain, 1e-9)):.0f} dB")

        # ── 1. Complementary LF/HF split ─────────────────────────────────────
        # Use a single LP filter; HF = audio − LF guarantees perfect reconstruction.
        sos_lp = butter(8, freq_floor_hz / (sr / 2), btype='low', output='sos')
        lf     = sosfilt(sos_lp, audio).astype(np.float32)
        hf     = (audio - lf).astype(np.float32)

        # ── 2. Frame-level HF energy (5 ms hops) ─────────────────────────────
        hop    = max(1, int(0.005 * sr))
        n_frm  = (len(hf) + hop - 1) // hop
        energy = np.array([
            float(np.mean(hf[i * hop:(i + 1) * hop] ** 2))
            for i in range(n_frm)
        ], dtype=np.float64)

        # ── 3. Running background energy ──────────────────────────────────────
        bg_frm = max(3, int(bg_smooth_ms * 0.001 * sr / hop))
        bg     = uniform_filter1d(energy, size=bg_frm, mode='reflect')
        bg     = np.maximum(bg, 1e-12)

        # ── 4. Onset detection ────────────────────────────────────────────────
        ratio_db = 10.0 * np.log10(energy / bg + 1e-12)
        onsets   = ratio_db > onset_threshold_db
        print(f"  onsets detected: {int(onsets.sum())} frames  "
              f"({int(onsets.sum()) * hop / sr * 1000:.0f} ms total gated)")

        # ── 5. Gate hold + smoothing ──────────────────────────────────────────
        hold_frm  = max(1, int(gate_hold_ms * 0.001 * sr / hop))
        gate_mask = binary_dilation(onsets, structure=np.ones(hold_frm, dtype=bool))
        gate_mask = gate_mask.astype(np.float64)

        # 5 ms fade-in / fade-out on gate edges to avoid clicks
        fade_frm  = max(1, int(5 * 0.001 * sr / hop))
        gate_mask = uniform_filter1d(gate_mask, size=fade_frm, mode='nearest')

        # ── 6. Gain curve: onset → suppression_gain; silence → 1.0 ──────────
        hf_gain = (1.0 - (1.0 - float(suppression_gain)) * gate_mask).astype(np.float64)

        # ── 7. Upsample gain to sample resolution ─────────────────────────────
        centers   = np.arange(n_frm, dtype=np.float64) * hop + hop * 0.5
        samples   = np.arange(len(audio), dtype=np.float64)
        gain_samp = np.interp(samples, centers, hf_gain).astype(np.float32)

        # ── 8. Apply to HF only; add LF back unchanged ────────────────────────
        output = (lf + hf * gain_samp).astype(np.float32)

        update_status("Transient gate: done")
        print("  Spectral transient gate: complete")
        return output

    except Exception as ex:
        print(f"  Spectral transient gate error ({ex}) — skipped")
        import traceback; traceback.print_exc()
        return audio


# ============================================================================
# SHERPA-ONNX ENHANCERS  (pip install sherpa-onnx)
# DPDFNet and GTCRN: causal streaming models — process file block-by-block
# with stateful hidden layers.  No torch needed.
# ============================================================================

_sherpa_session_cache = {}   # model_name → sherpa_onnx.OfflineSpeechDenoiser

# Local ONNX model paths  (placed in models/<source>/ alongside this script)
_MODELS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
_SHERPA_DIR   = os.path.join(_MODELS_DIR, 'sherpa_onnx')
_SHERPA_PATHS = {
    'dpdfnet': os.path.join(_SHERPA_DIR, 'dpdfnet_baseline.onnx'),
    'gtcrn':   os.path.join(_SHERPA_DIR, 'gtcrn_simple.onnx'),
}


def _get_onnx_providers():
    """Return provider list: OpenVINO first when enabled and available, else CPU."""
    import onnxruntime as ort
    if openvino_enabled and OV_AVAILABLE:
        return [
            ('OpenVINOExecutionProvider', {
                'device_type': 'CPU_FP32',
                'num_of_threads': max(1, (os.cpu_count() or 2) // 2),
                'cache_dir': os.path.expanduser('~/.cache/music-assassin/ov_cache'),
            }),
            'CPUExecutionProvider',
        ]
    return ['CPUExecutionProvider']


def apply_sherpa_enhancer(vocals, sr=44100, model_name='gtcrn'):
    """Speech enhancement via Sherpa-ONNX OfflineSpeechDenoiser (DPDFNet or GTCRN).

    Both models operate at 16 kHz mono.  The audio is resampled, processed in
    one call through OfflineSpeechDenoiser, then resampled back.
    Model files are loaded from models/sherpa_onnx/dpdfnet_baseline.onnx or
    models/sherpa_onnx/gtcrn_simple.onnx.
    Requires: pip install sherpa-onnx  +  the .onnx file in models/sherpa_onnx/
    """
    if not SHERPA_AVAILABLE:
        print(f"  Sherpa-ONNX ({model_name}): not installed — run: pip install sherpa-onnx")
        return vocals

    model_path = _SHERPA_PATHS.get(model_name)
    if not model_path or not os.path.isfile(model_path):
        print(f"  Sherpa-ONNX ({model_name}): model file not found at {model_path}")
        print(f"  Download from: https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models")
        return vocals

    try:
        import sherpa_onnx
        from scipy.signal import resample_poly

        update_status(f"Sherpa ({model_name}): loading model...")
        print(f"Sherpa-ONNX: initialising {model_name} from {model_path}...")

        if model_name not in _sherpa_session_cache:
            def _load():
                if model_name == 'dpdfnet':
                    model_cfg = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                        dpdfnet=sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig(
                            model=model_path
                        )
                    )
                else:  # gtcrn
                    model_cfg = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                        gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                            model=model_path
                        )
                    )
                cfg = sherpa_onnx.OfflineSpeechDenoiserConfig(model=model_cfg)
                return sherpa_onnx.OfflineSpeechDenoiser(cfg)

            _sherpa_session_cache[model_name] = _run_in_background(
                _load, f"Loading Sherpa-ONNX {model_name}..."
            )

        denoiser   = _sherpa_session_cache[model_name]
        target_sr  = denoiser.sample_rate   # 16000 for both DPDFNet and GTCRN

        # Resample to model sr
        g_up = math.gcd(sr, target_sr)
        v16  = resample_poly(vocals, target_sr // g_up, sr // g_up).astype('float32')

        update_status(f"Sherpa ({model_name}): enhancing {len(v16)/target_sr:.1f}s...")
        print(f"  Sherpa ({model_name}): {len(v16)/target_sr:.1f}s @ {target_sr} Hz...")

        # Single call — OfflineSpeechDenoiser handles the streaming internally
        result = denoiser(v16, target_sr)
        v_enh  = np.array(result.samples, dtype='float32')

        # Resample back
        g_dn  = math.gcd(target_sr, sr)
        v_out = resample_poly(v_enh, sr // g_dn, target_sr // g_dn).astype('float32')
        v_out = v_out[:len(vocals)]
        if len(v_out) < len(vocals):
            v_out = np.pad(v_out, (0, len(vocals) - len(v_out)))

        print(f"  Sherpa ({model_name}): complete")
        update_status(f"Sherpa ({model_name}): done")
        return v_out

    except Exception as ex:
        print(f"  Sherpa-ONNX ({model_name}) error ({ex}) — skipped")
        import traceback; traceback.print_exc()
        return vocals


# ============================================================================
# SPEECHDENOISER ONNX  (onnxruntime only — no extra package)
# github.com/yuyun2000/SpeechDenoiser
# Download speechdenoiser.onnx and place alongside this script, or set the path.
# ============================================================================

_SPEECHDENOISER_PATH = os.path.join(_MODELS_DIR, 'speechdenoiser', 'speechdenoiser.onnx')
_speechdenoiser_session = None


def apply_speechdenoiser(vocals, sr=44100):
    """Speech denoiser — yuyun2000/SpeechDenoiser via plain ONNX Runtime.

    Model: models/speechdenoiser/speechdenoiser.onnx (48 kHz, 480-sample hop).
    Inputs:  input_frame (480,), states (45304,), atten_lim_db (1,)
    Outputs: enhanced_audio_frame, new_states, lsnr
    Processes audio frame-by-frame maintaining state; no extra packages needed.
    Supports OpenVINO provider when enabled.
    """
    global _speechdenoiser_session
    if not os.path.isfile(_SPEECHDENOISER_PATH):
        print(f"  SpeechDenoiser: model not found at {_SPEECHDENOISER_PATH} — skipped.")
        print("  Download: wget https://github.com/yuyun2000/SpeechDenoiser/raw/main/48k/denoiser_model.onnx"
              f" -O {_SPEECHDENOISER_PATH}")
        return vocals

    try:
        import onnxruntime as ort
        from scipy.signal import resample_poly
        import torch
        from torch.nn import functional as F

        if _speechdenoiser_session is None:
            def _load_sd():
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.intra_op_num_threads    = max(1, (os.cpu_count() or 2) // 2)
                opts.inter_op_num_threads    = 1
                opts.execution_mode          = ort.ExecutionMode.ORT_SEQUENTIAL
                return ort.InferenceSession(_SPEECHDENOISER_PATH, opts,
                                            providers=_get_onnx_providers())
            _speechdenoiser_session = _run_in_background(_load_sd, "Loading SpeechDenoiser ONNX...")

        session = _speechdenoiser_session

        # Model native rate is 48 kHz
        target_sr = 48000
        g_up = math.gcd(sr, target_sr)
        v48  = resample_poly(vocals, target_sr // g_up, sr // g_up).astype('float32')

        # Pad to hop-size boundary (same logic as official inference script)
        HOP  = 480
        FFT  = 960
        orig_len = len(v48)
        pad_hop  = (HOP - orig_len % HOP) % HOP
        padded   = np.pad(v48, (0, FFT + pad_hop))
        frames   = np.array_split(padded, range(HOP, len(padded), HOP))

        update_status("SpeechDenoiser: enhancing...")
        print(f"  SpeechDenoiser: {orig_len/target_sr:.1f}s @ {target_sr} Hz  |  {len(frames)} frames")

        state        = np.zeros(45304, dtype=np.float32)
        atten_lim_db = np.zeros(1, dtype=np.float32)
        enhanced     = []

        for frame in frames:
            frame = frame.astype('float32')
            if len(frame) < HOP:
                frame = np.pad(frame, (0, HOP - len(frame)))
            out   = session.run(None, {
                'input_frame':   frame,
                'states':        state,
                'atten_lim_db':  atten_lim_db,
            })
            enhanced.append(out[0])
            state = out[1]

        # Trim back to original length (same offset as official script)
        v_enh = np.concatenate(enhanced)
        d     = FFT - HOP
        v_enh = v_enh[d: orig_len + d].astype('float32')

        # Resample back to original sr
        g_dn  = math.gcd(target_sr, sr)
        v_out = resample_poly(v_enh, sr // g_dn, target_sr // g_dn).astype('float32')
        v_out = v_out[:len(vocals)]
        if len(v_out) < len(vocals):
            v_out = np.pad(v_out, (0, len(vocals) - len(v_out)))

        print("  SpeechDenoiser: complete")
        update_status("SpeechDenoiser: done")
        return v_out

    except Exception as ex:
        print(f"  SpeechDenoiser error ({ex}) — skipped")
        import traceback; traceback.print_exc()
        return vocals


def apply_enhancer(vocals, sr=44100):
    """Dispatch to whichever enhancer is selected, then optionally apply the
    spectral transient gate if transient_gate_enabled is True."""
    if enhancer_choice == 'deepfilter':
        out = apply_deepfilter(vocals, sr)
    elif enhancer_choice == 'dpdfnet':
        out = apply_sherpa_enhancer(vocals, sr, model_name='dpdfnet')
    elif enhancer_choice == 'gtcrn':
        out = apply_sherpa_enhancer(vocals, sr, model_name='gtcrn')
    elif enhancer_choice == 'speechdenoiser':
        out = apply_speechdenoiser(vocals, sr)
    else:
        out = vocals

    if transient_gate_enabled:
        out = apply_spectral_transient_gate(out, sr)

    return out


def check_streams(file_path):
    """Detect video/audio/subtitle streams. Returns (video, audio_streams, sub_count, duration_sec).
    audio_streams is a list of dicts with index, codec, language, channels.
    sub_count is the number of subtitle streams (always copied to output)."""
    try:
        update_status(f"Analyzing: {os.path.basename(file_path)}")
        probe_data = probe(file_path)
        streams = probe_data["streams"]
        duration = float(probe_data.get("format", {}).get("duration", 0) or 0)
        video = False
        sub_count = 0
        audio_streams = []
        audio_idx = 0
        print(f"Found {len(streams)} streams  |  Duration: {duration:.1f}s")
        for stream in streams:
            codec_type = stream.get("codec_type", "unknown")
            if codec_type == "video":
                video = True
            elif codec_type == "audio":
                lang = stream.get("tags", {}).get("language", "und")
                codec = stream.get("codec_name", "?")
                channels = stream.get("channels", 1)
                title = stream.get("tags", {}).get("title", "")
                audio_streams.append({
                    "index": audio_idx,
                    "codec": codec,
                    "lang": lang,
                    "channels": channels,
                    "title": title,
                })
                audio_idx += 1
            elif codec_type == "subtitle":
                sub_count += 1
        print(f"Audio tracks found: {len(audio_streams)}  |  Subtitle tracks: {sub_count}")
        for t in audio_streams:
            label = f"  Track {t['index']}: {t['codec']} | {t['channels']}ch | lang={t['lang']}"
            if t['title']:
                label += f" | {t['title']}"
            print(label)
        return video, audio_streams, sub_count, duration
    except Exception as e:
        print(f"Stream check failed: {e}")
        return False, [{"index": 0, "codec": "?", "lang": "und", "channels": 1, "title": ""}], 0, 0


def ask_audio_track(audio_streams):
    """Show a track-picker dialog if multiple audio streams exist.
    Returns a list of track indices to process."""
    if len(audio_streams) <= 1:
        return [0]

    import tkinter as tk

    BG     = '#0d0d14'
    CARD   = '#14141f'
    BORDER = '#22223a'
    ACCENT = '#4f7ef0'
    TEXT   = '#cdd2de'
    MUTED  = '#505a72'
    GREEN  = '#27ae60'

    result = [None]
    dialog = tk.Toplevel(root)
    dialog.title("Select Audio Track")
    dialog.configure(bg=BG)
    dialog.resizable(False, False)
    dialog.grab_set()
    dialog.transient(root)

    tk.Label(dialog, text="Multiple audio tracks detected",
             font=("Segoe UI", 10, "bold"), bg=BG, fg=TEXT
             ).pack(padx=20, pady=(16, 4))

    # ── Track selection ──────────────────────────────────────────────────────
    tk.Label(dialog, text="TRACK TO PROCESS", font=("Segoe UI", 7, "bold"),
             bg=BG, fg=MUTED).pack(anchor="w", padx=20, pady=(6, 2))

    track_var = tk.IntVar(value=0)
    for t in audio_streams:
        label = f"Track {t['index']}:  {t['codec'].upper()}  |  {t['channels']}ch  |  {t['lang'].upper()}"
        if t['title']:
            label += f"  \u2014  {t['title']}"
        tk.Radiobutton(
            dialog, text=label, variable=track_var, value=t['index'],
            font=("Segoe UI", 9), bg=CARD, fg=TEXT,
            selectcolor=CARD, activebackground=CARD, activeforeground=ACCENT,
            highlightthickness=0
        ).pack(anchor="w", padx=20, pady=1)

    # ── Buttons ──────────────────────────────────────────────────────────────
    btn_row = tk.Frame(dialog, bg=BG)
    btn_row.pack(fill="x", padx=20, pady=(12, 16))

    def confirm_one():
        result[0] = [track_var.get()]
        dialog.destroy()

    def confirm_all():
        result[0] = [t['index'] for t in audio_streams]
        dialog.destroy()

    tk.Button(btn_row, text="Process Selected", command=confirm_one,
              bg=ACCENT, fg="white", font=("Segoe UI", 9, "bold"),
              relief="flat", pady=7, cursor="hand2"
              ).pack(side="left", expand=True, fill="x", padx=(0, 6))
    tk.Button(btn_row, text=f"Process All  ({len(audio_streams)})", command=confirm_all,
              bg=GREEN, fg="white", font=("Segoe UI", 9, "bold"),
              relief="flat", pady=7, cursor="hand2"
              ).pack(side="right", expand=True, fill="x")

    root.wait_window(dialog)
    chosen = result[0] if result[0] is not None else [0]
    print(f"Tracks to process: {chosen}")
    return chosen

def get_output_params(extension):
    """Universal codec mapping"""
    ext = extension.lower()

    audio_codecs = {
        'mp3': {'acodec': 'mp3', 'ab': '192k'},
        'm4a': {'acodec': 'aac', 'ab': '192k'},
        'wav': {'acodec': 'pcm_s16le'},
        'flac': {'acodec': 'flac'},
        'ogg': {'acodec': 'libvorbis', 'ab': '192k'},
    }

    video_codecs = {
        'mp4': {'vcodec': 'copy', 'acodec': 'aac', 'ab': '192k'},
        'mkv': {'vcodec': 'copy', 'acodec': 'aac', 'ab': '192k'},
        'avi': {'vcodec': 'copy', 'acodec': 'mp3', 'ab': '192k'},
    }

    if ext in audio_codecs:
        return audio_codecs[ext], False
    elif ext in video_codecs:
        return video_codecs[ext], True
    else:
        return {'acodec': 'aac', 'ab': '192k'}, False

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def _process_single_track(file_path, input_, track_idx, output_path, video, audio_streams, sub_count=0, duration=0):
    """Extract, separate, and export one audio track.
    keep_others=True preserves all other original audio tracks in video output."""
    global process1, process2, sample_rate, processing_mode, separator_choice

    print(f"  Extracting track a:{track_idx}...")
    update_status(f"Extracting audio (track {track_idx})...")
    process1 = (input_[f'a:{track_idx}']
                .output('pipe:1', format='f32le', ac=1, ar=sample_rate)
                .overwrite_output()
                .global_args('-loglevel', 'error')
                .run_async(pipe_stdin=False, pipe_stdout=True, pipe_stderr=True))

    # Chunked read with live progress bar
    # float32 mono: 4 bytes/sample × sample_rate samples/sec × duration sec
    expected_bytes = int(duration * sample_rate * 4) if duration > 0 else 0
    chunks = []
    bytes_read = 0
    chunk_size = 8 * 1024 * 1024   # 8 MB per read — large chunks keep Python overhead low
    ui_every = 4                    # update UI every N chunks (~32 MB intervals)
    iteration = 0
    while True:
        chunk = process1.stdout.read(chunk_size)
        if not chunk:
            break
        chunks.append(chunk)
        bytes_read += len(chunk)
        iteration += 1
        if iteration % ui_every == 0 and expected_bytes > 0 and progress:
            progress['value'] = min((bytes_read / expected_bytes) * 100, 99)
            root.update_idletasks()
    process1.wait()
    process1.kill()
    out = b''.join(chunks)
    del chunks

    if progress:
        progress['value'] = 0
    print(f"  Audio extracted: {len(out)//4} samples")

    waveform = frombuffer(out, dtype='<f4').flatten()
    del out
    gc.collect()

    if vad_enabled:
        preload_silero_vad()

    t_track = time.perf_counter()
    t_stage = t_track
    def _lap(name):
        nonlocal t_stage
        now = time.perf_counter()
        print(f"  [time] {name}: {now - t_stage:.1f}s")
        t_stage = now

    # High-pass + normalize, without the -40 dBFS per-sample gate. The gate
    # runs before normalization, so it bites by file level: the A2 audit
    # (2026-10-30) found it bought nothing at full scale but cost dialogue
    # ~10 dB magnitude SNR on a file 24 dB down, and erased it at -30 dB.
    # The keep-sound-effects path shares this input (the gate would erase
    # ambience there too).
    fx_on = fx_choice != 'off' and FAST_ENGINE_AVAILABLE and fast_separation.dme_available(fx_choice)
    if fx_choice != 'off' and not fx_on:
        print(f"WARNING: keep sound effects ({fx_choice}) unavailable -- skipped")
    waveform = audio_prefilter(waveform, sample_rate, gate=False)
    fx_input = waveform if fx_on else None
    effects = None
    _lap("prefilter")

    def _sep_progress(i, n):
        update_progress(i, n)

    if separator_choice == 'fast_mdx' and FAST_ENGINE_AVAILABLE:
        update_status("Separating vocals (Voc_FT, fast engine)...")
        vocals = fast_separation.separate_vocals_mdx(waveform, sample_rate, overlap=fast_overlap,
                                                     progress=_sep_progress)
    elif separator_choice == 'mrx' and FAST_ENGINE_AVAILABLE:
        update_status("Separating dialogue (MRX)...")
        if fx_on and fx_choice == 'mrx':
            # One MRX pass gives both stems. It runs on the un-gated input
            # so the effects keep their ambience.
            stems = fast_separation.separate_dme(fx_input, "mrx", sample_rate,
                                                 want=("dialogue", "effects"),
                                                 progress=_sep_progress)
            vocals, effects = stems["dialogue"], stems["effects"]
            del stems
        else:
            vocals = fast_separation.separate_dme(waveform, "mrx", sample_rate, want=("dialogue",),
                                                  progress=_sep_progress)["dialogue"]
    elif separator_choice == 'audio_separator' and AUDIO_SEPARATOR_AVAILABLE:
        vocals = separate_with_audio_separator(waveform, sample_rate, model_choice)
    elif separator_choice == 'demucs' and DEMUCS_AVAILABLE:
        vocals = separate_with_demucs(waveform, sample_rate)
    elif separator_choice == 'spleeter' and SPLEETER_AVAILABLE:
        vocals = separate_with_spleeter(waveform, sample_rate)
    elif DEMUCS_AVAILABLE:
        print("WARNING: Falling back to DEMUCS")
        vocals = separate_with_demucs(waveform, sample_rate)
    elif SPLEETER_AVAILABLE:
        print("WARNING: Falling back to Spleeter")
        vocals = separate_with_spleeter(waveform, sample_rate)
    else:
        print("No separator available")
        return False

    # Free the raw waveform immediately — the separator has finished with it
    del waveform
    gc.collect()

    if vocals is None:
        print(f"  Separation failed for track {track_idx}")
        return False

    # ascontiguousarray skips the copy when the array is already contiguous
    vocals = np.ascontiguousarray(vocals, dtype=np.float32)
    gc.collect()
    _lap(f"separation ({separator_choice})")

    # Effects stem from the original audio. Runs after separation rather
    # than in a thread: both stages already use every physical core.
    if fx_on and effects is None:
        update_status(f"Extracting sound effects ({fx_choice.upper()})...")
        effects = fast_separation.separate_dme(fx_input, fx_choice, sample_rate, want=("effects",),
                                               progress=_sep_progress)["effects"]
    if fx_on:
        del fx_input
        _lap(f"effects ({fx_choice})")

    vocals = advanced_postprocess_chunked(vocals, sample_rate, processing_mode)
    gc.collect()
    _lap(f"post-process ({processing_mode})")

    # Optional enhancer — deepfilter | dpdfnet | gtcrn | speechdenoiser
    if enhancer_choice != 'none':
        vocals = apply_enhancer(vocals, sample_rate)
        gc.collect()
        _lap(f"enhancer ({enhancer_choice})")

    # VAD gate runs LAST, after the enhancer. DeepFilterNet needs a continuous
    # signal to build a stable noise model — running VAD first fragments the
    # audio and causes DF to mishandle the gaps. Gating after enhancement also
    # lets DF attenuate residual music enough that VAD's energy fallback can
    # tell real (even high-pitched) voice from leftover instrumental.
    if vad_enabled:
        vocals = apply_vad_gate(vocals, sample_rate)
        gc.collect()
        _lap("VAD gate")

    # Effects go back in after the VAD gate, so sounds between lines of
    # dialogue survive it. Scaled by the post-process gain, they sit at the
    # same level relative to the dialogue as in the original mix.
    if effects is not None:
        vocals = mix_effects(vocals, effects, _last_post_gain * 10 ** (fx_level_db / 20), sample_rate)
        del effects
        gc.collect()
        _lap("effects mix")

    print(f"  Exporting: {os.path.basename(output_path)}")
    update_status(f"Exporting: {os.path.basename(output_path)}")

    ext = os.path.splitext(output_path)[1][1:]
    _, is_video = get_output_params(ext)
    vocals_pipe = ffmpeg_input('pipe:0', format='f32le', ar=sample_rate, ac=1)

    if is_video and video:
        other_indices = [t['index'] for t in audio_streams if t['index'] != track_idx]
        # Always keep all other audio tracks (stream-copy) and all subtitle tracks
        out_streams = [vocals_pipe.audio, input_.video]
        out_kwargs = {'vcodec': 'copy', 'c:a:0': 'aac', 'b:a:0': '192k'}
        if other_indices:
            print(f"  Preserving {len(other_indices)} other audio track(s): {other_indices}")
            out_streams += [input_[f'a:{i}'] for i in other_indices]
            for n in range(len(other_indices)):
                out_kwargs[f'c:a:{n+1}'] = 'copy'
        if sub_count > 0:
            print(f"  Preserving {sub_count} subtitle track(s)")
            out_streams += [input_[f's:{i}'] for i in range(sub_count)]
            for n in range(sub_count):
                out_kwargs[f'c:s:{n}'] = 'copy'
        process2 = (_ffmpeg.output(*out_streams, output_path, **out_kwargs)
                    .overwrite_output()
                    .global_args('-loglevel', 'error')
                    .run_async(pipe_stdin=True, pipe_stderr=True))
    else:
        params, _ = get_output_params(ext)
        process2 = (vocals_pipe.output(output_path, **params)
                    .overwrite_output()
                    .global_args('-loglevel', 'error')
                    .run_async(pipe_stdin=True, pipe_stderr=True))

    # Stream vocals to ffmpeg in 8 MB chunks — avoids holding a full copy
    # of the output in RAM alongside the numpy array
    _chunk = 8 * 1024 * 1024
    _view = memoryview(vocals.astype('<f4', copy=False))
    del vocals
    for _i in range(0, len(_view), _chunk):
        process2.stdin.write(_view[_i:_i + _chunk])
    process2.stdin.close()
    process2.wait()
    process2.kill()
    _lap("export")
    total = time.perf_counter() - t_track
    if duration > 0:
        print(f"  [time] total: {total:.1f}s for {duration / 60:.1f} min of audio "
              f"({total / duration * 600:.0f}s per 10 min)")
    else:
        print(f"  [time] total: {total:.1f}s")
    print(f"  SUCCESS: {output_path}")
    return True


def demusic_pro(file_path):
    """Main music removal processor with dual separator support"""
    global progress, root

    try:
        file_path = file_path.replace("\\\\", "/")
        name = os.path.splitext(os.path.basename(file_path))[0]
        extension = os.path.splitext(file_path)[1][1:]
        path = os.path.dirname(file_path)

        print(f"Processing: {name}.{extension}")
        print(f"Separator: {separator_choice.upper()}  |  Mode: {processing_mode}")
        update_status(f"Processing: {name}")

        input_ = ffmpeg_input(file_path)
        video, audio_streams, sub_count, duration = check_streams(file_path)
        if not audio_streams:
            print("No audio track found")
            return False

        track_indices = ask_audio_track(audio_streams)
        multi = len(track_indices) > 1

        all_ok = True
        for track_idx in track_indices:
            if multi:
                output_path = f"{path}/{name}-vocals-t{track_idx}.{extension}"
                print(f"\n--- Track {track_idx} of {len(track_indices)} ---")
            else:
                output_path = f"{path}/{name}-vocals.{extension}"

            ok = _process_single_track(file_path, input_, track_idx, output_path, video, audio_streams, sub_count, duration)
            if not ok:
                all_ok = False

        progress['value'] = 0
        if all_ok:
            update_status(f"Done: {name} ({len(track_indices)} track(s))")
        else:
            update_status(f"Partial failure: {name}")
        return all_ok

    except Exception as e:
        print(f"ERROR: {e}")
        update_status(f"Error: {str(e)[:50]}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# GUI SETUP
# ============================================================================

def setFrames(root):
    """Compact formal dark GUI"""
    global b1, b2, inprogressframe, listbox, f1, a1, progress, status_label, mode_selector, separator_selector, quantize_var, vad_var, df_var, enh_selector, ov_var, tgate_var, fx_selector, fx_level_selector

    BG      = '#0d0d14'
    CARD    = '#14141f'
    BORDER  = '#22223a'
    ACCENT  = '#4f7ef0'
    ACCENT2 = '#3a6bd6'
    TEXT    = '#cdd2de'
    MUTED   = '#505a72'
    GREEN   = '#27ae60'
    RED     = '#c0392b'

    style = Style(root)
    style.theme_use('clam')
    style.configure('App.TCombobox',
        fieldbackground=CARD, background=CARD, foreground=TEXT,
        selectbackground='#1c2a4a', selectforeground=ACCENT,
        bordercolor=BORDER, arrowcolor=ACCENT, padding=4
    )
    style.map('App.TCombobox',
        fieldbackground=[('readonly', CARD)],
        foreground=[('readonly', TEXT)],
    )
    style.configure('App.Horizontal.TProgressbar',
        background=ACCENT, troughcolor='#0a0a12',
        bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT2
    )

    root.configure(background=BG)

    def divider(parent=root, color=BORDER, h=1, padx=0):
        Frame(parent, bg=color, height=h).pack(fill="x", padx=padx)

    def section_label(text):
        Label(root, text=text, font=("Segoe UI", 7, "bold"),
              bg=BG, fg=MUTED, anchor="w").pack(fill="x", padx=16, pady=(9, 2))

    def mk_card(parent, pad_inner=0):
        f = Frame(parent, bg=CARD, highlightthickness=1,
                  highlightbackground=BORDER, highlightcolor=ACCENT)
        return f

    # ── HEADER ──────────────────────────────────────────────────────────────
    hdr = Frame(root, bg='#0f0f1c')
    hdr.pack(fill="x")

    title_row = Frame(hdr, bg='#0f0f1c')
    title_row.pack(fill="x", padx=16, pady=(10, 8))

    Label(title_row, text="Music Assassin Pro",
          font=("Segoe UI", 15, "bold"), bg='#0f0f1c', fg=TEXT).pack(side="left")
    Label(title_row, text="v0.4.5",
          font=("Segoe UI", 10), bg='#0f0f1c', fg=ACCENT).pack(side="left", padx=(6, 0), pady=4)
    Label(title_row, text="MDX-Net / DEMUCS / Spleeter  \u2022  VAD  \u2022  DeepFilter  \u2022  INT8",
          font=("Segoe UI", 8), bg='#0f0f1c', fg=MUTED).pack(side="right", pady=4)

    Frame(root, bg=ACCENT, height=2).pack(fill="x")

    # ── SEPARATOR + MODE (side-by-side) ─────────────────────────────────────
    section_label("CONFIGURATION")
    top_row = Frame(root, bg=BG)
    top_row.pack(fill="x", padx=16, pady=(0, 0))

    sep_card = mk_card(top_row)
    sep_card.pack(side="left", expand=True, fill="both", padx=(0, 6))
    Label(sep_card, text="SEPARATOR / MODEL", font=("Segoe UI", 7, "bold"),
          bg=CARD, fg=MUTED).pack(anchor="w", padx=10, pady=(7, 2))
    sep_options = []
    if FAST_ENGINE_AVAILABLE:
        sep_options.append(FAST_SEP_BEST)
        sep_options.append(FAST_SEP_QUICK)
        sep_options.append(FAST_SEP_MRX)
    if AUDIO_SEPARATOR_AVAILABLE:
        sep_options.append("MDX-Net Voc_FT   (Vocals · Fast · 66 MB)")
        sep_options.append("MDX-Net Kim_v2   (Vocals · Best CPU · 130 MB)")
    if DEMUCS_AVAILABLE:
        sep_options.append("DEMUCS htdemucs  (Fast)")
        sep_options.append("DEMUCS htdemucs_ft  (Quality)")
        sep_options.append("DEMUCS mdx_extra  (Best)")
    if SPLEETER_AVAILABLE:
        sep_options.append("Spleeter  (4-Stems)")
    if sep_options:
        separator_selector = Combobox(sep_card, values=sep_options, state="readonly",
                                      font=("Segoe UI", 9), style='App.TCombobox')
        separator_selector.set(sep_options[0])
        separator_selector.pack(fill="x", padx=10, pady=(0, 4))
        separator_selector.bind("<<ComboboxSelected>>", lambda e: set_separator())
        quantize_var = BooleanVar(value=True)
        Checkbutton(sep_card, text="INT8 quantize (2-3x faster, htdemucs)",
                    variable=quantize_var, command=set_quantize,
                    bg=CARD, fg=TEXT, selectcolor=CARD, activebackground=CARD,
                    activeforeground=ACCENT, highlightthickness=0,
                    font=("Segoe UI", 7)).pack(anchor="w", padx=8, pady=(0, 7))
    else:
        Label(sep_card, text="No separator available",
              font=("Segoe UI", 8), fg=RED, bg=CARD).pack(padx=10, pady=(0, 8), anchor="w")

    mode_card = mk_card(top_row)
    mode_card.pack(side="right", expand=True, fill="both")
    Label(mode_card, text="PROCESSING MODE", font=("Segoe UI", 7, "bold"),
          bg=CARD, fg=MUTED).pack(anchor="w", padx=10, pady=(7, 2))
    mode_selector = Combobox(mode_card, values=["Normal", "Balanced", "Aggressive"],
                             state="readonly", font=("Segoe UI", 9), style='App.TCombobox')
    mode_selector.set("Balanced")
    mode_selector.pack(fill="x", padx=10, pady=(0, 3))
    mode_selector.bind("<<ComboboxSelected>>", lambda e: set_mode())
    Label(mode_card, text="60-70%  \u2022  75-85%  \u2022  90-95% removal",
          font=("Segoe UI", 7), bg=CARD, fg=MUTED).pack(anchor="w", padx=10, pady=(0, 7))

    # ── ENHANCEMENT ──────────────────────────────────────────────────────────
    enh_card = mk_card(root)
    enh_card.pack(fill="x", padx=16, pady=(6, 0))
    Label(enh_card, text="ENHANCEMENT", font=("Segoe UI", 7, "bold"),
          bg=CARD, fg=MUTED).pack(anchor="w", padx=10, pady=(7, 4))

    # Row 0 — Keep sound effects
    fx_row = Frame(enh_card, bg=CARD)
    fx_row.pack(fill="x", padx=10, pady=(0, 4))
    Label(fx_row, text="Keep sound effects:", bg=CARD, fg=TEXT,
          font=("Segoe UI", 8)).pack(side="left", padx=(0, 4))
    fx_options = [FX_OFF]
    if FAST_ENGINE_AVAILABLE:
        fx_options.append(FX_MRX)
        if fast_separation.dme_available("cdx23"):
            fx_options.append(FX_CDX23)
    fx_selector = Combobox(fx_row, values=fx_options, width=30,
                           state="readonly" if len(fx_options) > 1 else "disabled",
                           font=("Segoe UI", 8), style='App.TCombobox')
    fx_selector.set(FX_OFF)
    fx_selector.pack(side="left")
    fx_selector.bind("<<ComboboxSelected>>", lambda e: set_fx())
    Label(fx_row, text="Level:", bg=CARD, fg=TEXT,
          font=("Segoe UI", 8)).pack(side="left", padx=(10, 4))
    fx_level_selector = Combobox(fx_row, values=list(FX_LEVELS), width=16,
                                 state="disabled", font=("Segoe UI", 8),
                                 style='App.TCombobox')
    fx_level_selector.set("Normal")
    fx_level_selector.pack(side="left")
    fx_level_selector.bind("<<ComboboxSelected>>", lambda e: set_fx_level())
    Label(fx_row, text="footsteps, doors, ambience",
          bg=CARD, fg=MUTED, font=("Segoe UI", 7)).pack(side="left", padx=(8, 0))

    # Row 1 — VAD toggle + enhancer selector
    enh_row = Frame(enh_card, bg=CARD)
    enh_row.pack(fill="x", padx=10, pady=(0, 4))

    vad_var = BooleanVar(value=False)
    ck_kw = dict(bg=CARD, fg=TEXT, selectcolor=CARD, activebackground=CARD,
                 activeforeground=ACCENT, highlightthickness=0, font=("Segoe UI", 8))

    vad_lbl = "VAD Gate"
    if not VAD_AVAILABLE:
        vad_lbl += " [needs torch]"
    Checkbutton(enh_row, text=vad_lbl,
                variable=vad_var, command=set_vad,
                state="normal" if VAD_AVAILABLE else "disabled",
                **ck_kw).pack(side="left", padx=(0, 14))

    Label(enh_row, text="Enhancer:", bg=CARD, fg=MUTED,
          font=("Segoe UI", 8)).pack(side="left", padx=(0, 4))

    # Build option list based on what's installed
    enh_options = ["None"]
    if DF_AVAILABLE:
        enh_options.append("DeepFilter  (torch)")
    else:
        enh_options.append("DeepFilter  [pip install deepfilternet]")
    if SHERPA_AVAILABLE:
        enh_options += ["DPDFNet  (Sherpa-ONNX)", "GTCRN  (Sherpa-ONNX, fastest)"]
    else:
        enh_options += ["DPDFNet  [pip install sherpa-onnx]",
                        "GTCRN  [pip install sherpa-onnx]"]
    sdpath_ok = os.path.isfile(_SPEECHDENOISER_PATH)
    enh_options.append("SpeechDenoiser ONNX" if sdpath_ok
                       else "SpeechDenoiser ONNX  [model not found]")

    enh_selector = Combobox(enh_row, values=enh_options, width=32,
                            state="readonly", font=("Segoe UI", 8),
                            style='App.TCombobox')
    enh_selector.set("None")
    enh_selector.pack(side="left")
    enh_selector.bind("<<ComboboxSelected>>", lambda e: set_enhancer())

    # Row 2 — OpenVINO toggle (affects ONNX-based enhancers only)
    ov_row = Frame(enh_card, bg=CARD)
    ov_row.pack(fill="x", padx=10, pady=(0, 4))
    ov_var = BooleanVar(value=False)
    ov_lbl = "OpenVINO acceleration  (AVX2/AVX-512, ONNX enhancers only)"
    if not OV_AVAILABLE:
        ov_lbl += "  [pip install onnxruntime-openvino]"
    Checkbutton(ov_row, text=ov_lbl,
                variable=ov_var, command=set_openvino,
                state="normal" if OV_AVAILABLE else "disabled",
                **ck_kw).pack(side="left")

    # Row 3 — Spectral transient gate
    tgate_row = Frame(enh_card, bg=CARD)
    tgate_row.pack(fill="x", padx=10, pady=(0, 7))
    tgate_var = BooleanVar(value=False)
    Checkbutton(tgate_row,
                text="Spectral transient gate  (suppresses residual guitar/piano taps above 2 kHz)",
                variable=tgate_var, command=set_transient_gate,
                **ck_kw).pack(side="left")

    # ── STATUS ───────────────────────────────────────────────────────────────
    status_bar = Frame(root, bg='#0e1a2e', highlightthickness=1,
                       highlightbackground=BORDER)
    status_bar.pack(fill="x", padx=16, pady=(8, 0))
    status_label = Label(status_bar, text="  Ready \u2014 add files to begin",
                         font=("Segoe UI", 9), bg='#0e1a2e', fg=GREEN, anchor="w")
    status_label.pack(fill="x", padx=4, pady=5)

    # ── FILE BUTTONS ─────────────────────────────────────────────────────────
    btn_row = Frame(root, bg=BG)
    btn_row.pack(fill="x", padx=16, pady=(8, 0))
    b1 = Button(btn_row, text="Browse Files", bg=ACCENT2, fg="white",
                font=("Segoe UI", 9, "bold"), pady=7, relief="flat", cursor="hand2",
                activebackground='#2f59c0', activeforeground='white', bd=0)
    b1.pack(side="left", expand=True, fill="x", padx=(0, 6))
    b2 = Button(btn_row, text="Remove Selected", bg='#5c1a1a', fg='#e8a0a0',
                font=("Segoe UI", 9, "bold"), pady=7, relief="flat", cursor="hand2",
                activebackground='#7a1f1f', activeforeground='white', bd=0)
    b2.pack(side="right", expand=True, fill="x")

    # ── QUEUE ────────────────────────────────────────────────────────────────
    section_label("PROCESSING QUEUE")
    listboxframe = mk_card(root)
    listboxframe.pack(fill="both", expand=True, padx=16)
    scrollbar_y = Scrollbar(listboxframe, orient="vertical",
                            bg=BORDER, troughcolor=CARD, activebackground=ACCENT,
                            width=10)
    listbox = Listbox(listboxframe, yscrollcommand=scrollbar_y.set, selectmode="multiple",
                      height=4, font=("Segoe UI", 9), bg=CARD, fg=TEXT,
                      selectbackground='#1c2a4a', selectforeground=ACCENT,
                      relief="flat", bd=0, activestyle='none', highlightthickness=0)
    scrollbar_y.config(command=listbox.yview)
    scrollbar_y.pack(side="right", fill="y", pady=1)
    listbox.pack(side="left", fill="both", expand=True, padx=8, pady=6)

    inprogressframe = Frame(root, bg=BG)
    inprogressframe.pack(fill="x", padx=16)

    # ── PROGRESS ─────────────────────────────────────────────────────────────
    section_label("PROGRESS")
    progress = Progressbar(root, orient="horizontal", mode='determinate',
                           style='App.Horizontal.TProgressbar')
    progress.pack(fill="x", padx=16)

    # ── ACTION BUTTONS ───────────────────────────────────────────────────────
    action_row = Frame(root, bg=BG)
    action_row.pack(fill="x", padx=16, pady=(8, 0))
    f1 = Button(action_row, text="Start Processing", bg='#145233', fg='#7de8b8',
                font=("Segoe UI", 10, "bold"), pady=9, relief="flat", cursor="hand2",
                activebackground='#0f3d26', activeforeground='white', bd=0)
    f1.pack(side="left", expand=True, fill="x", padx=(0, 6))
    a1 = Button(action_row, text="Clear Queue", bg='#1e1410', fg='#c8a96e',
                font=("Segoe UI", 10, "bold"), pady=9, relief="flat", cursor="hand2",
                activebackground='#2a1c0e', activeforeground='white', bd=0)
    a1.pack(side="right", expand=True, fill="x")

    # ── LOG ──────────────────────────────────────────────────────────────────
    section_label("PROCESSING LOG")
    log_widget = ScrolledText(root, height=5, font=("Consolas", 8),
                              bg='#07070d', fg='#3eb87a', relief="flat", bd=0,
                              insertbackground=ACCENT, selectbackground='#1c2a4a',
                              highlightthickness=1, highlightbackground=BORDER)
    log_widget.pack(fill="both", expand=True, padx=16, pady=(0, 12))

    sys.stdout = PrintLogger(log_widget)

    # The globals' defaults don't know which separators are installed; make
    # them match the option the selector is actually showing.
    if sep_options:
        set_separator()

FAST_SEP_BEST  = "Voc_FT \u00b7 fast engine   (Best)"
FAST_SEP_QUICK = "Voc_FT \u00b7 fast engine, low overlap   (Faster)"
FAST_SEP_MRX   = "MRX dialogue   (Fastest \u00b7 lower quality)"
FX_OFF   = "Off"
FX_MRX   = "Fast  (MRX, some music bleed)"
FX_CDX23 = "Quality  (CDX23, cleaner, slower)"
# Effects level relative to the original mix. Raising it raises any music
# that leaked into the effects by the same amount. Picking an effects
# model resets the level to that model's default: +3 dB for MRX, to make up
# for what its music cleanup takes off the effects; CDX23 gets no cleanup.
FX_LEVELS = {"Quieter  (-6 dB)": -6.0, "Normal": 0.0,
             "Louder  (+3 dB)": 3.0, "Much louder  (+6 dB)": 6.0}
FX_LEVEL_DEFAULT = {"mrx": "Louder  (+3 dB)", "cdx23": "Normal"}

def set_separator():
    """Update separator engine + model choice"""
    global separator_choice, model_choice, fast_overlap
    sep = separator_selector.get()
    if sep in (FAST_SEP_BEST, FAST_SEP_QUICK):
        separator_choice = "fast_mdx"
        model_choice = "UVR-MDX-NET-Voc_FT.onnx"
        fast_overlap = 0.25 if sep == FAST_SEP_BEST else 0.1
        update_status(f"Voc_FT fast engine (overlap {fast_overlap})")
        print(f"Separator: Voc_FT fast engine  |  overlap: {fast_overlap}")
    elif sep == FAST_SEP_MRX:
        separator_choice = "mrx"
        update_status("MRX dialogue separation")
        print("Separator: MRX (dialogue / music / effects)")
    elif "MDX-Net" in sep:
        separator_choice = "audio_separator"
        model_choice = "UVR-MDX-NET-Voc_FT.onnx" if "Voc_FT" in sep else "Kim_Vocal_2.onnx"
        update_status(f"MDX-Net: {model_choice}")
        print(f"Separator: MDX-Net  |  model: {model_choice}")
    elif "DEMUCS" in sep:
        separator_choice = "demucs"
        model_choice = sep.split()[1]
        update_status(f"DEMUCS: {model_choice}")
        print(f"Separator: DEMUCS  |  model: {model_choice}")
    else:
        separator_choice = "spleeter"
        update_status("Separator: SPLEETER")
        print("Separator: SPLEETER")

def set_fx():
    """Keep sound effects: off | mrx | cdx23"""
    global fx_choice
    sel = fx_selector.get()
    fx_choice = {FX_MRX: "mrx", FX_CDX23: "cdx23"}.get(sel, "off")
    fx_level_selector.configure(state="disabled" if fx_choice == "off" else "readonly")
    if fx_choice != "off":
        fx_level_selector.set(FX_LEVEL_DEFAULT[fx_choice])
        set_fx_level()
    print(f"Keep sound effects: {fx_choice}")
    update_status(f"Keep sound effects: {sel}")

def set_fx_level():
    global fx_level_db
    sel = fx_level_selector.get()
    fx_level_db = FX_LEVELS.get(sel, 0.0)
    print(f"Effects level: {fx_level_db:+.0f} dB")
    update_status(f"Effects level: {sel}")

def set_quantize():
    """Toggle INT8 quantization — clears cache so the model reloads accordingly"""
    global quantize_enabled
    quantize_enabled = bool(quantize_var.get())
    _MODEL_CACHE.clear()
    print(f"INT8 quantize: {quantize_enabled}  (model cache cleared)")
    update_status(f"INT8 quantize: {quantize_enabled}")

def set_vad():
    global vad_enabled
    vad_enabled = bool(vad_var.get())
    print(f"VAD gate: {vad_enabled}")
    update_status(f"VAD gate: {'ON' if vad_enabled else 'OFF'}")

_ENH_LABEL_MAP = {
    'None': 'none',
    'DeepFilter': 'deepfilter',
    'DPDFNet': 'dpdfnet',
    'GTCRN': 'gtcrn',
    'SpeechDenoiser': 'speechdenoiser',
}

def set_enhancer():
    global enhancer_choice
    sel = enh_selector.get().split('  ')[0].strip()   # strip install hints
    enhancer_choice = _ENH_LABEL_MAP.get(sel, 'none')
    # Clear Sherpa session cache when switching models
    _sherpa_session_cache.clear()
    print(f"Enhancer: {enhancer_choice}")
    update_status(f"Enhancer: {sel}")

def set_openvino():
    global openvino_enabled, _speechdenoiser_session
    openvino_enabled = bool(ov_var.get())
    # Clear caches so sessions are recreated with the new provider
    _sherpa_session_cache.clear()
    _speechdenoiser_session = None
    print(f"OpenVINO: {openvino_enabled}")
    update_status(f"OpenVINO: {'ON' if openvino_enabled else 'OFF'}")

def set_transient_gate():
    global transient_gate_enabled
    transient_gate_enabled = bool(tgate_var.get())
    print(f"Transient gate: {transient_gate_enabled}")
    update_status(f"Transient gate: {'ON' if transient_gate_enabled else 'OFF'}")

def set_mode():
    """Update processing mode"""
    global processing_mode
    mode = mode_selector.get()
    processing_mode = mode.lower()
    update_status(f"Mode: {mode}")
    print(f"Processing mode: {processing_mode}")

def get_file_path():
    """Browse files"""
    global files, listbox, do, root

    if do == 1:
        do = 0

    selected = []

    # Try zenity (GTK native dialog - reliable directory navigation on Linux)
    try:
        import subprocess
        result = subprocess.run(
            ["zenity", "--file-selection", "--multiple",
             "--title=Select Audio/Video Files",
             "--file-filter=Media Files (mp4 mp3 wav avi mkv flac) | *.mp4 *.mp3 *.wav *.avi *.mkv *.flac",
             "--file-filter=All Files | *"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            selected = result.stdout.strip().split("|")
    except FileNotFoundError:
        # zenity not available, fall back to tkinter dialog
        browse = filedialog.askopenfilenames(
            title="Select Audio/Video Files",
            parent=root,
            filetypes=[("Media", ("*.mp4", "*.mp3", "*.wav", "*.avi", "*.mkv", "*.flac")), ("All", "*.*")]
        )
        selected = list(browse)

    for i in selected:
        i = i.strip()
        if i and i not in files:
            files.append(i)
            listbox.insert("end", os.path.basename(i))
    print(f"Queue: {len(files)} files")
    update_status(f"Queue: {len(files)} file(s)")

def remove_item():
    """Remove selected files"""
    global listbox, files
    selected = list(listbox.curselection())[::-1]
    for idx in selected:
        listbox.delete(idx)
        files.pop(idx)
    update_status(f"Queue: {len(files)} file(s)")

def demusic_list():
    """Process queue"""
    global do, files

    if do == 1 or not files:
        if not files:
            update_status("Add files to start")
        return

    do = 1
    completed = []
    failed = []
    update_status(f"Processing {len(files)} file(s)...")

    def process_next():
        global do
        if do == 0 or not files:
            if not files:
                do = 0
                print(f"\n=== COMPLETE ===\nProcessed: {len(completed)}\nFailed: {len(failed)}")
                update_status(f"Complete \u2014 {len(completed)} processed, {len(failed)} failed")
            return

        file_path = files.pop(0)
        if listbox.size() > 0:
            listbox.delete(0)

        filename = os.path.basename(file_path)
        print(f"\nProcessing: {filename}")
        update_status(f"Processing: {filename}")

        success = demusic_pro(file_path)

        if success:
            completed.append(filename)
            Label(inprogressframe, text=f"  \u2713  {filename}",
                 font=("Segoe UI", 8), bg='#071a0f', fg='#4ade80',
                 anchor="w", pady=2).pack(fill="x", pady=1)
        else:
            failed.append(filename)
            Label(inprogressframe, text=f"  \u2717  {filename}",
                 font=("Segoe UI", 8), bg='#1a0707', fg='#f87171',
                 anchor="w", pady=2).pack(fill="x", pady=1)

        root.after(1000, process_next)

    root.after(100, process_next)

def on_closing():
    """Clean exit"""
    global do, process1, process2
    if do == 0:
        sys.stdout = sys.__stdout__
        root.destroy()
        return
    if messagebox.askokcancel("Quit", "Processing is active. Exit anyway?"):
        sys.stdout = sys.__stdout__
        if process1: process1.kill()
        if process2: process2.kill()
        root.destroy()

# The window class must equal StartupWMClass in the .deb's .desktop file
# (build/build.sh), or GNOME/XWayland can't match the window to its entry and
# the taskbar shows a generic icon instead of ours.
WM_CLASS = "music-assassin"


def set_window_icon(root):
    """Window icon from assets/icon*.png (made by build/draw_icon.py). Found
    in the PyInstaller bundle, next to this file, or one level up when run
    from versions/. Missing icons are not an error."""
    for base in (getattr(sys, "_MEIPASS", None),
                 os.path.dirname(os.path.abspath(__file__)),
                 os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        if not base:
            continue
        paths = [os.path.join(base, "assets", n)
                 for n in ("icon.png", "icon_64.png", "icon_48.png", "icon_32.png")]
        if all(os.path.isfile(p) for p in paths):
            try:
                root._icons = [PhotoImage(file=p) for p in paths]  # keep refs alive
                root.iconphoto(True, *root._icons)
            except Exception as e:
                print(f"Window icon not set: {e}")
            return


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    root = Tk(className=WM_CLASS)
    root.wm_title("Music Assassin Pro v0.4.5")
    root.geometry("720x660")
    root.configure(background='#0d0d14')
    root.minsize(640, 560)

    try:
        root.update()
    except:
        os.environ['TCL_LIBRARY'] = '/usr/share/tcltk/tcl8.6'
        os.environ['TK_LIBRARY'] = '/usr/share/tcltk/tk8.6'
        root.destroy()
        root = Tk(className=WM_CLASS)
        root.wm_title("Music Assassin Pro v0.4.5")
        root.geometry("720x660")
        root.configure(background='#0d0d14')
        root.minsize(640, 560)

    set_window_icon(root)
    setFrames(root)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    b1.configure(command=get_file_path)
    b2.configure(command=remove_item)
    f1.configure(command=demusic_list)
    a1.configure(command=lambda: [files.clear(), listbox.delete(0, 'end'), update_status("Queue cleared")])

    print("=" * 55)
    print("Music Assassin Pro v0.4.5 - MDX-Net/DEMUCS/SPLEETER")
    print("=" * 55)
    print(f"Fast engine:         {FAST_ENGINE_AVAILABLE}  (OpenVINO: {importlib.util.find_spec('openvino') is not None})")
    print(f"MDX-Net available:   {AUDIO_SEPARATOR_AVAILABLE}")
    print(f"DEMUCS available:    {DEMUCS_AVAILABLE}")
    print(f"Spleeter available:  {SPLEETER_AVAILABLE}")
    print(f"VAD available:       {VAD_AVAILABLE}  (Silero via torch hub)")
    print(f"DeepFilter avail:    {DF_AVAILABLE}  (pip install deepfilternet)")
    print(f"Sherpa-ONNX avail:   {SHERPA_AVAILABLE}  (DPDFNet/GTCRN, pip install sherpa-onnx)")
    print(f"SpeechDenoiser:      {os.path.isfile(_SPEECHDENOISER_PATH)}  (models/speechdenoiser/speechdenoiser.onnx)")
    print(f"OpenVINO provider:   {OV_AVAILABLE}  (pip install onnxruntime-openvino)")
    print(f"CPU cores:           {os.cpu_count()}")
    print("Models:       htdemucs (fast) / htdemucs_ft / mdx_extra")
    print("Speed:        model cache + INT8 quantize (htdemucs)")
    print("Enhancers:    DeepFilter | DPDFNet | GTCRN | SpeechDenoiser | None")
    print("Pipeline:     pre-filter \u2192 separate \u2192 [VAD] \u2192 DSP \u2192 [enhancer]")
    print("Modes:        Normal / Balanced / Aggressive")
    print("Memory:       chunked processing (<180MB)")
    print("=" * 55)
    update_status("Ready \u2014 select separator and add files")

    root.mainloop()
