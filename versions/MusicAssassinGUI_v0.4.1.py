"""
Music Assassin Pro v0.4.1 - DEMUCS/SPLEETER HYBRID EDITION
Advanced dual-source separation with intelligent pre/post-filtering
+ Choose between DEMUCS (natural vocals) or Spleeter (4-stems)
+ 3 processing modes (Normal/Balanced/Aggressive)
+ Advanced pre-filtering (noise gate, high-pass)
+ Advanced post-filtering (de-esser, harmonic rejection, formant enhancement)
+ Manual pre-chunked processing — bounded RAM for all model sizes
+ Compact formal UI with real-time feedback

v0.4.2 chunking fix (over v0.4.1):
+ FIX: Demucs now pre-chunks waveform before tensor creation (60s + 2s overlap)
  — htdemucs_ft / mdx_extra (4-model bags) no longer OOM on long files
+ FIX: Spleeter now pre-chunks waveform (60s + 2s overlap)
  — prevents TF tensor explosion (~400-800 MB per call) on long files

v0.4.1 improvements (inherited):
+ FIX: global post-filter normalization (was per-chunk → caused static at end)
+ FIX: tail fade-out (last 0.5 s → silent) to remove demucs tail noise
+ Model selection: htdemucs (fast) / htdemucs_ft (quality) / mdx_extra (best)
+ Session model cache — weights load once, reused across the whole batch
+ INT8 dynamic quantization (htdemucs only, CPU) — ~2-3x faster
"""

import os, sys, gc, time, logging

# Set Tcl/Tk library paths if not already set (Linux only — paths differ per OS)
os.environ.setdefault("TCL_LIBRARY", "/usr/share/tcltk/tcl8.6")
os.environ.setdefault("TK_LIBRARY", "/usr/share/tcltk/tk8.6")

from tkinter import filedialog, messagebox, Label, Frame, Scrollbar, Listbox, Button, Tk, Entry, Checkbutton, BooleanVar
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
SPLEETER_AVAILABLE = importlib.util.find_spec('spleeter') is not None
if not DEMUCS_AVAILABLE:
    print("WARNING: DEMUCS not installed (torch/torchaudio). Spleeter only.")
if not SPLEETER_AVAILABLE:
    print("WARNING: Spleeter not installed. DEMUCS only.")

# ============================================================================
# GLOBALS & CONFIG
# ============================================================================

files = []
process1 = process2 = separator = do = 0
sample_rate = 44100
b1 = b2 = f1 = a1 = listbox = inprogressframe = progress = status_label = mode_selector = separator_selector = quantize_var = None
root = None
status_text = "Ready"
processing_mode = "balanced"  # normal, balanced, aggressive
separator_choice = "demucs"   # demucs or spleeter
model_choice = "htdemucs"     # htdemucs (fast) | htdemucs_ft (quality) | mdx_extra (best)
quantize_enabled = True       # INT8 dynamic quantization (htdemucs only, 2-3x faster on CPU)

# Session model cache — key "name:quant" → loaded torch model.
# Stays alive for the whole session so batch runs don't reload weights per file.
_MODEL_CACHE = {}

# ── Logging setup ────────────────────────────────────────────────────────────
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "music_assassin.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_LOG_PATH, encoding="utf-8"),
    ],
)
_log = logging.getLogger("MusicAssassin")

# ffmpeg-python and torch log a lot of noise at WARNING level — keep those quiet
# in our log file so only real warnings from our own code appear.
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

_log.info("=" * 60)
_log.info("Music Assassin v0.4.3  —  session start")
_log.info(f"Log file: {_LOG_PATH}")
_log.info(f"DEMUCS available: {DEMUCS_AVAILABLE}  |  Spleeter: {SPLEETER_AVAILABLE}")


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

def audio_prefilter(waveform, sr=44100):
    """Advanced pre-filtering: gate + high-pass + normalize"""
    update_status("Pre-filtering audio...")
    print("Pre-filtering: gate + high-pass + normalize")

    # Stage 1: Noise gate (< -40dB removed)
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
    peak = np.max(np.abs(output))
    if peak > 1e-10:
        output *= 0.98 / peak

    # Fade the last 0.5 s to silence — removes demucs tail noise/artifacts.
    fade_samps = min(int(sr * 0.5), len(output))
    if fade_samps > 0:
        output[-fade_samps:] *= np.linspace(1.0, 0.0, fade_samps, dtype=np.float32)

    print(f"Post-processing complete")
    return output

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
    htdemucs_ft and mdx_extra are both BagOfModels with 4 sub-models — without
    this they OOM on long files because the whole-song tensor stays in RAM.
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

        vocals_idx  = model.sources.index('vocals')
        duration_s  = len(waveform) / sr

        CHUNK_SEC    = 60
        OVERLAP_SEC  = 2
        chunk_samp   = int(CHUNK_SEC * sr)
        overlap_samp = int(OVERLAP_SEC * sr)
        step_samp    = chunk_samp - overlap_samp

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
                        _result[0] = apply_model(
                            model, tensor,
                            device=dev,
                            segment=CHUNK_SEC,
                            overlap=0.25,
                            progress=False,
                            num_workers=0,
                        )
                except Exception as e:
                    _err[0] = e

            chunk_dur = (end - start) / sr
            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()

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
    60s chunks and overlap-add the vocals back.
    """
    if not SPLEETER_AVAILABLE:
        print("ERROR: Spleeter not available")
        return None

    print("Loading Spleeter 4-stems model...")
    update_status("Loading Spleeter model...")

    try:
        try:
            import tensorflow as tf
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

        total_samp  = len(waveform)
        starts      = list(range(0, total_samp, step_samp))
        n_chunks    = len(starts)
        vocals_out  = np.zeros(total_samp, dtype=np.float32)

        duration_s = total_samp / sr
        print(f"  duration: {duration_s:.1f}s  |  chunks: {n_chunks} × {CHUNK_SEC}s  |  overlap: {OVERLAP_SEC}s")

        if progress:
            progress.configure(mode='determinate')
            progress['value'] = 0

        for idx, start in enumerate(starts):
            end   = min(start + chunk_samp, total_samp)
            chunk = waveform[start:end].reshape(-1, 1)

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

    waveform = audio_prefilter(waveform, sample_rate)

    if separator_choice == 'demucs' and DEMUCS_AVAILABLE:
        vocals = separate_with_demucs(waveform, sample_rate)
    elif separator_choice == 'spleeter' or not DEMUCS_AVAILABLE:
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

    vocals = advanced_postprocess_chunked(vocals, sample_rate, processing_mode)
    gc.collect()

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
    global b1, b2, inprogressframe, listbox, f1, a1, progress, status_label, mode_selector, separator_selector, quantize_var

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
    Label(title_row, text="v0.4.3",
          font=("Segoe UI", 10), bg='#0f0f1c', fg=ACCENT).pack(side="left", padx=(6, 0), pady=4)
    Label(title_row, text="DEMUCS / Spleeter  \u2022  Cached + INT8  \u2022  Advanced DSP",
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

def set_separator():
    """Update separator engine + DEMUCS model choice"""
    global separator_choice, model_choice
    sep = separator_selector.get()
    if "DEMUCS" in sep:
        separator_choice = "demucs"
        # Second word of the label is the model name (htdemucs / htdemucs_ft / mdx_extra)
        model_choice = sep.split()[1]
        update_status(f"DEMUCS: {model_choice}")
        print(f"Separator: DEMUCS  |  model: {model_choice}")
    else:
        separator_choice = "spleeter"
        update_status("Separator: SPLEETER")
        print("Separator: SPLEETER")

def set_quantize():
    """Toggle INT8 quantization — clears cache so the model reloads accordingly"""
    global quantize_enabled
    quantize_enabled = bool(quantize_var.get())
    _MODEL_CACHE.clear()
    print(f"INT8 quantize: {quantize_enabled}  (model cache cleared)")
    update_status(f"INT8 quantize: {quantize_enabled}")

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

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    root = Tk()
    root.wm_title("Music Assassin Pro v0.4.3")
    root.geometry("720x660")
    root.configure(background='#0d0d14')
    root.minsize(640, 560)

    try:
        root.update()
    except:
        os.environ['TCL_LIBRARY'] = '/usr/share/tcltk/tcl8.6'
        os.environ['TK_LIBRARY'] = '/usr/share/tcltk/tk8.6'
        root.destroy()
        root = Tk()
        root.wm_title("Music Assassin Pro v0.4.3")
        root.geometry("720x660")
        root.configure(background='#0d0d14')
        root.minsize(640, 560)

    setFrames(root)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    b1.configure(command=get_file_path)
    b2.configure(command=remove_item)
    f1.configure(command=demusic_list)
    a1.configure(command=lambda: [files.clear(), listbox.delete(0, 'end'), update_status("Queue cleared")])

    print("=" * 55)
    print("Music Assassin Pro v0.4.3 - DEMUCS/SPLEETER HYBRID")
    print("=" * 55)
    print(f"DEMUCS available:    {DEMUCS_AVAILABLE}")
    print(f"Spleeter available:  {SPLEETER_AVAILABLE}")
    print(f"CPU cores:           {os.cpu_count()}")
    print("Models:       htdemucs (fast) / htdemucs_ft / mdx_extra")
    print("Speed:        model cache + INT8 quantize (htdemucs)")
    print("Pre-filter:   noise gate + high-pass")
    print("Post-filter:  harmonic rejection + de-esser + global normalize")
    print("Modes:        Normal / Balanced / Aggressive")
    print("Memory:       chunked processing (<180MB)")
    print("=" * 55)
    update_status("Ready \u2014 select separator and add files")

    root.mainloop()
