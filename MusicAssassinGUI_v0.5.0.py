"""
Music Assassin Pro v0.5.0  —  CPU-Optimised Streaming Edition
=============================================================
+ DEMUCS-only (no Spleeter / TensorFlow — saves ~1.5 GB at startup)
+ Model cache: htdemucs/htdemucs_ft/mdx_extra loaded once per session
+ Outer 60 s chunked ML processing — peak RAM never holds the full-song
  4-stem output in memory; only the vocals slice is accumulated
+ INT8 dynamic quantisation for htdemucs (2–3× faster on CPU, optional)
+ Determinate chunk-level progress bar with ETA
+ Pre-filter and post-filter individually togglable
+ memoryview streaming export — no extra copy of vocals during encode
+ All CPU cores used via set_num_threads

Memory target: < 500 MB for a 3-minute song on htdemucs INT8
"""

import os, sys, gc, time, math, threading, socket, subprocess
import http.server, importlib.util

# Tcl/Tk paths — commented out; the runtime hook sets these for bundled builds
# os.environ.setdefault("TCL_LIBRARY", "/usr/share/tcltk/tcl8.6")
# os.environ.setdefault("TK_LIBRARY",  "/usr/share/tcltk/tk8.6")

from tkinter import (filedialog, messagebox, Label, Frame, Scrollbar, Listbox,
                     Button, Tk, BooleanVar, Checkbutton, IntVar, Radiobutton)
from tkinter.scrolledtext import ScrolledText
from tkinter.ttk import Progressbar, Combobox, Style
from scipy.signal import butter, sosfilt
import numpy as np
from numpy import frombuffer
import ffmpeg as _ffmpeg
from ffmpeg import input as ffmpeg_input, probe

# ── Availability check (zero-cost at startup) ─────────────────────────────────
DEMUCS_AVAILABLE = (
    importlib.util.find_spec('torch')  is not None and
    importlib.util.find_spec('demucs') is not None
)
ONNX_AVAILABLE = importlib.util.find_spec('onnxruntime') is not None
if not DEMUCS_AVAILABLE:
    print("WARNING: torch / demucs not found. No separator available.")

# Directory where exported .onnx files are cached between runs
_ONNX_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'music-assassin')

# ── Model wrapper ─────────────────────────────────────────────────────────────
# Unified interface so the rest of the code doesn't care whether inference
# is running through PyTorch or ONNX Runtime.

class _ModelWrapper:
    """Thin wrapper around a torch model or an ONNX Runtime session.

    Attributes
    ----------
    kind          : 'onnx' | 'torch'
    obj           : ort.InferenceSession  or  torch.nn.Module
    sources       : list[str]  e.g. ['drums','bass','other','vocals']
    chunk_samples : int — samples per forward call
        torch  → 60 s outer chunk (apply_model handles inner segmentation)
        onnx   → model's own segment length (one forward pass per chunk)
    """
    def __init__(self, kind, obj, sources, chunk_samples):
        self.kind          = kind
        self.obj           = obj
        self.sources       = sources
        self.chunk_samples = chunk_samples

    def infer(self, chunk_mono: np.ndarray, n_threads: int) -> np.ndarray:
        """Run one forward pass on a mono float32 chunk.
        chunk_mono must be padded to self.chunk_samples before calling.
        Returns vocals as a float32 array of the same length."""
        v_idx = self.sources.index('vocals')
        if self.kind == 'onnx':
            stereo = np.stack([chunk_mono, chunk_mono])[np.newaxis]  # (1,2,N)
            out    = self.obj.run(None, {'mix': stereo.astype(np.float32)})[0]
            return out[0, v_idx].mean(0).astype(np.float32)
        else:
            from torch import from_numpy, no_grad
            from demucs.apply import apply_model
            stereo = np.stack([chunk_mono, chunk_mono], axis=0)
            tensor = from_numpy(stereo).float().unsqueeze(0)
            with no_grad():
                try:
                    sources = apply_model(self.obj, tensor, overlap=0.05,
                                          num_workers=0, progress=False)
                except TypeError:
                    sources = apply_model(self.obj, tensor)
            result = sources[0, v_idx].float().mean(0).cpu().numpy()
            del sources, tensor, stereo
            return result

# ── Model cache ───────────────────────────────────────────────────────────────
# Keys: cache_key string.  Values: _ModelWrapper instances.
# Stays alive for the entire session — avoids reloading between files.
_MODEL_CACHE: dict = {}

# ── Separation progress (written by worker thread, read by main thread) ───────
_sep_state = {'step': 0, 'total': 1, 'status': 'Separating...', 'eta': ''}

# ── Globals & runtime config ──────────────────────────────────────────────────
files: list = []
process1 = process2 = do = 0
sample_rate = 44100

# Widget handles (set in setFrames)
b1 = b2 = f1 = a1 = listbox = inprogressframe = progress = status_label = None
mode_selector = model_selector = keep_others_selector = None
pre_filter_var = post_filter_var = quantize_var = None
root = None

# Settings (updated by GUI callbacks)
status_text     = "Ready"
processing_mode = "balanced"    # normal | balanced | aggressive
model_choice    = "htdemucs"    # htdemucs | htdemucs_ft | mdx_extra
keep_others_choice = True
pre_filter_enabled  = True
post_filter_enabled = True
quantize_enabled    = True      # INT8 quantisation (htdemucs only)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

class PrintLogger:
    """Buffered stdout → GUI textbox."""
    def __init__(self, textbox):
        self.textbox = textbox
        self.buffer  = []

    def write(self, text):
        self.buffer.append(text)
        if len(self.buffer) >= 20:
            self.textbox.configure(state="normal")
            self.textbox.insert("end", ''.join(self.buffer))
            self.textbox.see("end")
            self.textbox.configure(state="disabled")
            self.buffer = []

    def flush(self):
        if self.buffer:
            self.textbox.configure(state="normal")
            self.textbox.insert("end", ''.join(self.buffer))
            self.textbox.see("end")
            self.textbox.configure(state="disabled")
            self.buffer = []


def update_status(message):
    global status_text
    status_text = message
    if status_label:
        status_label.config(text=f"  {message}")
        root.update_idletasks()


def update_progress(value, total):
    if progress and total > 0:
        progress['value'] = (value / total) * 100
        root.update_idletasks()


# ── VLC IPC SERVER ────────────────────────────────────────────────────────────

IPC_PORT    = 19876
STREAM_PORT = 8765
_AUDIO_PIPE = '/tmp/music-assassin-vocals.pcm'   # named FIFO for live audio

_stream_source   = None    # file currently being streamed
_ffmpeg_stream   = None    # FFmpeg mux subprocess
_streaming_active = False  # True while streaming is in progress

def _vocals_processed(file_path):
    """Return True if this file has already had vocals embedded (sidecar exists)."""
    return os.path.exists(file_path + '.ma')

def _queue_file(path):
    """Add a file to the queue from any source (thread-safe via root.after)."""
    if not path or not os.path.exists(path):
        print(f"[VLC] File not found: {path}")
        return
    if path in files:
        update_status(f"Already in queue: {os.path.basename(path)}")
        return
    files.append(path)
    listbox.insert("end", os.path.basename(path))
    update_status(f"Queued from VLC: {os.path.basename(path)}")
    print(f"[VLC] Queued: {path}")

def start_ipc_server():
    """Daemon thread that listens for file paths from the VLC extension."""
    def _serve():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(('127.0.0.1', IPC_PORT))
            srv.listen(5)
            print(f"[VLC] IPC server listening on port {IPC_PORT}")
            while True:
                conn, _ = srv.accept()
                try:
                    data = conn.recv(4096).decode('utf-8').strip()
                    if data:
                        if data.startswith('STREAM:'):
                            threading.Thread(
                                target=handle_stream_ipc,
                                args=(data[7:],), daemon=True).start()
                        else:
                            root.after(0, lambda p=data: _queue_file(p))
                finally:
                    conn.close()
        except Exception as e:
            print(f"[VLC] IPC server error: {e}")
        finally:
            srv.close()

    threading.Thread(target=_serve, daemon=True).start()


# ── HTTP STREAM SERVER ────────────────────────────────────────────────────────

class _StreamHandler(http.server.BaseHTTPRequestHandler):
    """Serves the live MPEG-TS mux of source video + vocals FIFO."""

    def log_message(self, *args): pass  # silence request logs

    def do_GET(self):
        global _ffmpeg_stream

        if self.path != '/stream':
            self.send_error(404)
            return

        if not _stream_source or not os.path.exists(_stream_source):
            self.send_error(503, "No stream active")
            return

        self.send_response(200)
        self.send_header('Content-Type', 'video/mp2t')
        self.end_headers()

        # FFmpeg: source video (real-time) + FIFO audio → MPEG-TS
        # Opening the FIFO for reading here unblocks the processing thread
        # which was waiting for a reader before writing.
        cmd = [
            'ffmpeg', '-loglevel', 'error',
            '-re', '-i', _stream_source,                         # video at real-time speed
            '-f', 'f32le', '-ar', '44100', '-ac', '1',
            '-i', _AUDIO_PIPE,                                   # growing audio from FIFO
            '-map', '0:v', '-map', '1:a',
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
            '-f', 'mpegts', 'pipe:1',
        ]
        _ffmpeg_stream = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        try:
            while True:
                data = _ffmpeg_stream.stdout.read(65536)
                if not data:
                    break
                try:
                    self.wfile.write(data)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
        finally:
            _ffmpeg_stream.kill()
            _ffmpeg_stream = None


def _start_http_stream_server():
    """Start the HTTP stream server as a daemon thread (once per session)."""
    srv = http.server.HTTPServer(('127.0.0.1', STREAM_PORT), _StreamHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[Stream] HTTP server listening on port {STREAM_PORT}")


def handle_stream_ipc(path):
    """Called when IPC receives STREAM:{path}. Sets up FIFO + triggers processing."""
    global _stream_source, _streaming_active

    # Already processed — Lua opens the original file directly (it has both tracks)
    if _vocals_processed(path):
        print(f"[Stream] Already processed — Lua will open original: {path}")
        return

    if _streaming_active:
        print("[Stream] Already streaming — request ignored")
        return

    _stream_source = path

    # Re-create the FIFO (remove stale one if present)
    if os.path.exists(_AUDIO_PIPE):
        os.remove(_AUDIO_PIPE)
    os.mkfifo(_AUDIO_PIPE)
    print(f"[Stream] FIFO created: {_AUDIO_PIPE}")
    print(f"[Stream] Waiting for VLC to connect to http://127.0.0.1:{STREAM_PORT}/stream …")

    # Queue the file for normal processing — the FIFO flag makes
    # advanced_postprocess_chunked also write chunks to the pipe.
    root.after(0, lambda p=path: _queue_file(p))


# ─────────────────────────────────────────────────────────────────────────────
# DSP — PRE-FILTER
# ─────────────────────────────────────────────────────────────────────────────

def audio_prefilter(waveform, sr=44100):
    """Noise gate + high-pass + normalise.  Runs once on the full waveform."""
    update_status("Pre-filtering…")
    print("Pre-filter: gate + high-pass + normalise")

    # Noise gate: silence samples below -40 dB
    waveform = waveform * (np.abs(waveform) > 10 ** (-40 / 20))

    # High-pass: remove sub-40 Hz rumble
    sos_hp = butter(4, 40 / (sr / 2), btype='highpass', output='sos')
    waveform = sosfilt(sos_hp, waveform)

    # Peak-normalise to 0.98
    peak = np.max(np.abs(waveform))
    if peak > 1e-10:
        waveform = waveform / peak * 0.98

    return waveform.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DSP — POST-FILTER (chunked)
# ─────────────────────────────────────────────────────────────────────────────

def advanced_postprocess_chunked(vocals, sr=44100, mode='balanced'):
    """Overlap-add chunked post-processing.  Never materialises the full array
    in more than one extra copy at a time."""
    print(f"\nPost-processing ({mode} mode, chunked)…")
    update_status(f"Post-processing: {mode}…")

    chunk_size   = int(sr * 30)      # 30 s chunks
    overlap_size = int(sr * 2)       # 2 s crossfade
    hop_size     = chunk_size - overlap_size

    n = len(vocals)
    output  = np.zeros(n, dtype=np.float32)
    n_chunks = max(1, math.ceil((n - overlap_size) / hop_size))

    fade_in  = np.linspace(0, 1, overlap_size, dtype=np.float32)
    fade_out = 1.0 - fade_in

    for idx in range(n_chunks):
        update_progress(idx, n_chunks)
        start = idx * hop_size
        end   = min(start + chunk_size, n)
        chunk = vocals[start:end].copy()

        if mode == 'normal':
            chunk = _pp_normal(chunk, sr)
        elif mode == 'balanced':
            chunk = _pp_balanced(chunk, sr)
        else:
            chunk = _pp_aggressive(chunk, sr)

        actual = end - start
        if idx == 0:
            output[start:end] = chunk[:actual]
        else:
            ov = min(overlap_size, actual)
            output[start:start + ov] = (
                output[start:start + ov] * fade_out[:ov] +
                chunk[:ov] * fade_in[:ov]
            )
            if ov < actual:
                output[start + ov:end] = chunk[ov:actual]

        # Write this chunk to the stream FIFO if streaming is active.
        # Opening the FIFO blocks until FFmpeg (the reader) has connected —
        # that happens once VLC opens the HTTP stream URL.
        if _streaming_active and os.path.exists(_AUDIO_PIPE):
            try:
                hop_start = start if idx == 0 else start + overlap_size
                hop_end   = end
                with open(_AUDIO_PIPE, 'ab') as _fifo:
                    _fifo.write(output[hop_start:hop_end].astype('<f4').tobytes())
            except OSError:
                pass  # FIFO gone or reader disconnected — keep processing normally

        del chunk
        gc.collect()
        root.update_idletasks()

    # Signal FIFO end-of-stream
    if _streaming_active and os.path.exists(_AUDIO_PIPE):
        try:
            os.remove(_AUDIO_PIPE)
        except OSError:
            pass

    return output


def _pp_normal(chunk, sr):
    sos_hp = butter(4, 80      / (sr / 2), btype='highpass', output='sos')
    sos_lp = butter(4, 10000   / (sr / 2), btype='lowpass',  output='sos')
    chunk  = sosfilt(sos_hp, chunk)
    chunk  = sosfilt(sos_lp, chunk)
    peak   = np.max(np.abs(chunk))
    if peak > 1e-10:
        chunk = chunk / peak * 0.95
    return chunk.astype(np.float32)


def _pp_balanced(chunk, sr):
    sos_low = butter(4, 40 / (sr / 2), btype='highpass', output='sos')
    chunk   = sosfilt(sos_low, chunk)

    for freq in [82, 110, 146, 196, 247]:
        if freq < sr / 2 - 10:
            sos_n = butter(2, [max(20, freq - 15) / (sr / 2),
                               min(sr / 2 - 1, freq + 15) / (sr / 2)],
                           btype='bandstop', output='sos')
            chunk = sosfilt(sos_n, chunk)

    sos_sib = butter(4, [4000 / (sr / 2), 7000 / (sr / 2)], btype='band', output='sos')
    sibs    = sosfilt(sos_sib, chunk)
    sib_rms = np.sqrt(np.mean(sibs ** 2) + 1e-10)
    if sib_rms > 0.15:
        sibs = np.tanh(sibs / (4 * sib_rms + 1e-10)) * 2 * sib_rms
    chunk -= sibs * 0.6

    sos_f   = butter(2, [700 / (sr / 2), 3200 / (sr / 2)], btype='band', output='sos')
    chunk  += sosfilt(sos_f, chunk) * 0.35

    peak = np.max(np.abs(chunk))
    if peak > 1e-10:
        chunk = chunk / peak * 0.98
    return chunk.astype(np.float32)


def _pp_aggressive(chunk, sr):
    sos_low = butter(4, 40 / (sr / 2), btype='highpass', output='sos')
    chunk   = sosfilt(sos_low, chunk)

    for freq in [82, 110, 146, 196, 247, 330, 440]:
        if freq < sr / 2 - 20:
            sos_n = butter(3, [max(20, freq - 20) / (sr / 2),
                               min(sr / 2 - 1, freq + 20) / (sr / 2)],
                           btype='bandstop', output='sos')
            chunk = sosfilt(sos_n, chunk)

    sos_sib = butter(4, [4000 / (sr / 2), 7000 / (sr / 2)], btype='band', output='sos')
    sibs    = sosfilt(sos_sib, chunk)
    sib_rms = np.sqrt(np.mean(sibs ** 2) + 1e-10)
    if sib_rms > 0.15:
        sibs = np.tanh(sibs / (4 * sib_rms + 1e-10)) * 2 * sib_rms
    chunk -= sibs * 0.6

    sos_f   = butter(2, [700 / (sr / 2), 3200 / (sr / 2)], btype='band', output='sos')
    chunk  += sosfilt(sos_f, chunk) * 0.40

    peak = np.max(np.abs(chunk))
    if peak > 1e-10:
        chunk = chunk / peak * 0.98
    return chunk.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CACHE
# ─────────────────────────────────────────────────────────────────────────────

def _get_or_load_model(name: str, n_threads: int, do_quantize: bool) -> _ModelWrapper:
    """Load the named DEMUCS model and return a _ModelWrapper.

    Priority order (CPU):
      1. ONNX Runtime  — fastest; exported once to ~/.cache/music-assassin/
      2. INT8 quant    — torch with quantised Linear layers (htdemucs only)
      3. FP32 torch    — baseline fallback

    The wrapper is cached so subsequent files in the same session skip loading.
    """
    from torch import set_num_threads
    from demucs.pretrained import get_model

    # ONNX and quantise are mutually exclusive — ONNX wins when available
    use_onnx = ONNX_AVAILABLE
    ck_suffix = 'onnx' if use_onnx else ('q' if do_quantize else 'fp32')
    cache_key = f"{name}:{ck_suffix}"

    if cache_key in _MODEL_CACHE:
        print(f"  Wrapper '{cache_key}' already cached — reusing.")
        return _MODEL_CACHE[cache_key]

    print(f"  Loading torch model '{name}'…")
    set_num_threads(n_threads)
    torch_model = get_model(name)
    torch_model.eval()

    sr = sample_rate

    # ── Try ONNX export + session ────────────────────────────────────────────
    if use_onnx:
        wrapper = _try_build_onnx_wrapper(torch_model, name, sr)
        if wrapper is not None:
            _MODEL_CACHE[cache_key] = wrapper
            return wrapper
        print("  ONNX failed — falling back to torch")

    # ── INT8 quantisation (torch path) ───────────────────────────────────────
    if do_quantize and 'htdemucs' in name:
        try:
            import torch
            torch_model = torch.quantization.quantize_dynamic(
                torch_model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("  Quantisation: INT8 applied (Linear layers)")
        except Exception as ex:
            print(f"  Quantisation failed ({ex}), using FP32")

    # torch mode: outer 60-s chunks; apply_model handles inner segmentation
    wrapper = _ModelWrapper('torch', torch_model, list(torch_model.sources),
                            chunk_samples=60 * sr)
    _MODEL_CACHE[cache_key] = wrapper
    print(f"  Cached as '{cache_key}' (torch)")
    return wrapper


def _try_build_onnx_wrapper(torch_model, name: str, sr: int):
    """Export the torch model to ONNX (once), load an ORT session, and return
    a _ModelWrapper — or return None if anything fails."""
    os.makedirs(_ONNX_CACHE_DIR, exist_ok=True)
    onnx_path = os.path.join(_ONNX_CACHE_DIR, f"{name}.onnx")

    # ── Export if not cached ─────────────────────────────────────────────────
    if not os.path.exists(onnx_path):
        print(f"  ONNX: exporting '{name}' → {onnx_path}")
        print("  (first-time only — takes ~1 min)")
        try:
            import torch

            # Determine the segment length the model expects
            seg_sec     = float(getattr(torch_model, 'segment', 7.8))
            seg_samples = int(seg_sec * sr)

            # valid_length pads to the nearest size the conv stack accepts
            if hasattr(torch_model, 'valid_length'):
                try:
                    seg_samples = int(torch_model.valid_length(seg_samples))
                except Exception:
                    pass

            dummy = torch.zeros(1, 2, seg_samples)

            torch.onnx.export(
                torch_model,
                dummy,
                onnx_path,
                input_names=['mix'],
                output_names=['sources'],
                # Fixed shape — more reliable than dynamic_axes for transformer
                # models whose positional encodings are length-sensitive.
                opset_version=17,
                do_constant_folding=True,
                training=torch.onnx.TrainingMode.EVAL,
            )
            print(f"  ONNX: export OK  ({seg_samples} samples / {seg_sec:.1f} s per chunk)")
        except Exception as ex:
            print(f"  ONNX export failed: {ex}")
            if os.path.exists(onnx_path):
                os.remove(onnx_path)   # remove partial file
            return None
    else:
        print(f"  ONNX: using cached {onnx_path}")

    # ── Load ORT session ─────────────────────────────────────────────────────
    try:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads      = os.cpu_count() or 4
        opts.inter_op_num_threads      = 1
        opts.graph_optimization_level  = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode            = ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=['CPUExecutionProvider'],
        )

        # Derive chunk_samples from the model's input shape in the ONNX graph
        in_shape      = session.get_inputs()[0].shape   # [1, 2, N]
        chunk_samples = int(in_shape[2]) if isinstance(in_shape[2], int) else int(
            getattr(torch_model, 'valid_length',
                    lambda x: x)(int(getattr(torch_model, 'segment', 7.8) * sr)))

        print(f"  ONNX: session ready  (chunk={chunk_samples} samples, "
              f"provider=CPUExecutionProvider)")

        return _ModelWrapper('onnx', session, list(torch_model.sources), chunk_samples)

    except Exception as ex:
        print(f"  ONNX session load failed: {ex}")
        return None


def unload_model():
    """Free all cached models and reclaim RAM."""
    _MODEL_CACHE.clear()
    gc.collect()
    print("Model cache cleared.")
    update_status("Model cache cleared — will reload on next run")


# ─────────────────────────────────────────────────────────────────────────────
# UI BACKGROUND RUNNER (for model load — indeterminate spinner)
# ─────────────────────────────────────────────────────────────────────────────

def _run_in_background(fn, status_msg):
    """Run fn() in a daemon thread; animate the progress bar until done.
    Re-raises any exception on the calling (main) thread."""
    result = [None]
    error  = [None]

    def worker():
        try:
            result[0] = fn()
        except Exception as e:
            error[0] = e

    if progress:
        progress.configure(mode='indeterminate')
        progress.start(40)
    update_status(status_msg)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while t.is_alive():
        root.update()
        time.sleep(0.033)
    t.join()

    if progress:
        progress.stop()
        progress.configure(mode='determinate')
        progress['value'] = 0

    if error[0]:
        raise error[0]
    return result[0]


# ─────────────────────────────────────────────────────────────────────────────
# DEMUCS — CHUNKED SEPARATION
# ─────────────────────────────────────────────────────────────────────────────

_OVERLAP_SEC = 2   # crossfade seconds between consecutive chunks

def _separate_chunked_worker(waveform, wrapper: _ModelWrapper, n_threads: int):
    """
    Chunked separation — backend-agnostic via _ModelWrapper.

    Torch mode : outer 60-s chunks; apply_model does inner segmentation.
    ONNX mode  : chunks = model's native segment length (~7.8 s for htdemucs);
                 each chunk is one ORT forward pass.

    Writes live progress into _sep_state for the main thread to read.
    Returns a mono float32 array of the same length as waveform.
    """
    sr        = sample_rate
    chunk_s   = wrapper.chunk_samples
    overlap_s = min(_OVERLAP_SEC * sr, chunk_s // 4)  # never exceed 25% of chunk
    hop_s     = chunk_s - overlap_s

    n           = len(waveform)
    vocals_out  = np.zeros(n, dtype=np.float32)
    n_chunks    = max(1, math.ceil((n - overlap_s) / hop_s))

    fade_in  = np.linspace(0, 1, overlap_s, dtype=np.float32)
    fade_out = 1.0 - fade_in

    _sep_state['total'] = n_chunks
    _sep_state['step']  = 0

    chunk_times: list = []

    for idx in range(n_chunks):
        t_start   = time.time()
        start     = idx * hop_s
        end       = min(start + chunk_s, n)
        actual    = end - start

        # Pad to chunk_s so the model always sees a fixed-size block
        chunk = np.zeros(chunk_s, dtype=np.float32)
        chunk[:actual] = waveform[start:end]

        eta_str = ''
        if chunk_times:
            avg = sum(chunk_times) / len(chunk_times)
            rem = avg * (n_chunks - idx)
            eta_str = f"  ETA {int(rem // 60)}m {int(rem % 60)}s"

        _sep_state['status'] = (
            f"Separating: chunk {idx + 1}/{n_chunks}"
            f"  ({start // sr}–{end // sr} s){eta_str}"
        )

        chunk_vocals = wrapper.infer(chunk, n_threads)
        del chunk

        # Overlap-add crossfade at chunk boundary
        if idx == 0:
            vocals_out[start:end] = chunk_vocals[:actual]
        else:
            ov = min(overlap_s, actual)
            vocals_out[start:start + ov] = (
                vocals_out[start:start + ov] * fade_out[:ov] +
                chunk_vocals[:ov] * fade_in[:ov]
            )
            if ov < actual:
                vocals_out[start + ov:end] = chunk_vocals[ov:actual]

        del chunk_vocals
        gc.collect()

        chunk_times.append(time.time() - t_start)
        _sep_state['step'] = idx + 1

    return vocals_out


def separate_with_demucs(waveform, sr=44100):
    """
    Load model (cached), then run chunked separation in a background thread
    while the main thread drives the progress bar and keeps the UI alive.
    """
    if not DEMUCS_AVAILABLE:
        print("ERROR: torch / demucs not available.")
        return None

    n_threads = os.cpu_count() or 4

    # ── Load model (once, cached) ────────────────────────────────────────────
    def _load():
        return _get_or_load_model(model_choice, n_threads, quantize_enabled)

    try:
        wrapper = _run_in_background(_load, f"Loading {model_choice}…")
    except Exception as e:
        print(f"Model load failed: {e}")
        return None

    backend = wrapper.kind.upper()
    print(f"  Backend: {backend}  |  chunk={wrapper.chunk_samples} samples "
          f"({wrapper.chunk_samples / sample_rate:.1f} s)")

    # ── Chunked separation — background thread + main-thread progress ────────
    result  = [None]
    err     = [None]

    def _worker():
        try:
            result[0] = _separate_chunked_worker(waveform, wrapper, n_threads)
        except Exception as e:
            err[0] = e

    if progress:
        progress.configure(mode='determinate')
        progress['value'] = 0

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    while t.is_alive():
        step  = _sep_state['step']
        total = _sep_state['total']
        msg   = _sep_state['status']
        if progress and total > 0:
            progress['value'] = (step / total) * 100
        if msg:
            update_status(msg)
        root.update()
        time.sleep(0.033)

    t.join()

    if progress:
        progress['value'] = 0

    if err[0]:
        print(f"Separation error: {err[0]}")
        return None

    return result[0]


# ─────────────────────────────────────────────────────────────────────────────
# STREAM ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def check_streams(file_path):
    """Return (has_video, audio_streams, has_subs, duration_sec)."""
    try:
        update_status(f"Analysing: {os.path.basename(file_path)}")
        data     = probe(file_path)
        streams  = data["streams"]
        duration = float(data.get("format", {}).get("duration", 0) or 0)
        video = subs = False
        audio_streams = []
        audio_idx = 0
        print(f"Streams: {len(streams)}  |  Duration: {duration:.1f} s")
        for s in streams:
            ct = s.get("codec_type", "")
            if ct == "video":
                video = True
            elif ct == "audio":
                audio_streams.append({
                    "index":    audio_idx,
                    "codec":    s.get("codec_name", "?"),
                    "lang":     s.get("tags", {}).get("language", "und"),
                    "channels": s.get("channels", 1),
                    "title":    s.get("tags", {}).get("title", ""),
                })
                audio_idx += 1
            elif ct == "subtitle":
                subs = True
        for t in audio_streams:
            line = (f"  Track {t['index']}: {t['codec']} | {t['channels']}ch"
                    f" | lang={t['lang']}")
            if t['title']:
                line += f" | {t['title']}"
            print(line)
        return video, audio_streams, subs, duration
    except Exception as e:
        print(f"Stream check failed: {e}")
        return False, [{"index": 0, "codec": "?", "lang": "und",
                        "channels": 1, "title": ""}], False, 0


def ask_audio_track(audio_streams):
    """Show a track-picker dialog when multiple streams exist."""
    if len(audio_streams) <= 1:
        return [0]

    import tkinter as tk

    BG = '#0d0d14'; CARD = '#14141f'; BORDER = '#22223a'
    ACCENT = '#4f7ef0'; TEXT = '#cdd2de'; MUTED = '#505a72'
    GREEN = '#27ae60'

    result = [None]
    dlg = tk.Toplevel(root)
    dlg.title("Select Audio Track")
    dlg.configure(bg=BG)
    dlg.resizable(False, False)
    dlg.grab_set()
    dlg.transient(root)

    tk.Label(dlg, text="Multiple audio tracks detected",
             font=("Segoe UI", 10, "bold"), bg=BG, fg=TEXT
             ).pack(padx=20, pady=(16, 4))
    tk.Label(dlg, text="TRACK TO PROCESS",
             font=("Segoe UI", 7, "bold"), bg=BG, fg=MUTED
             ).pack(anchor="w", padx=20, pady=(6, 2))

    track_var = tk.IntVar(value=0)
    for t in audio_streams:
        lbl = (f"Track {t['index']}:  {t['codec'].upper()}"
               f"  |  {t['channels']}ch  |  {t['lang'].upper()}")
        if t['title']:
            lbl += f"  \u2014  {t['title']}"
        tk.Radiobutton(
            dlg, text=lbl, variable=track_var, value=t['index'],
            font=("Segoe UI", 9), bg=CARD, fg=TEXT,
            selectcolor=CARD, activebackground=CARD, activeforeground=ACCENT,
            highlightthickness=0
        ).pack(anchor="w", padx=20, pady=1)

    row = tk.Frame(dlg, bg=BG)
    row.pack(fill="x", padx=20, pady=(12, 16))

    def _one():
        result[0] = [track_var.get()]; dlg.destroy()
    def _all():
        result[0] = [t['index'] for t in audio_streams]; dlg.destroy()

    tk.Button(row, text="Process Selected", command=_one,
              bg=ACCENT, fg="white", font=("Segoe UI", 9, "bold"),
              relief="flat", pady=7, cursor="hand2"
              ).pack(side="left", expand=True, fill="x", padx=(0, 6))
    tk.Button(row, text=f"Process All  ({len(audio_streams)})", command=_all,
              bg=GREEN, fg="white", font=("Segoe UI", 9, "bold"),
              relief="flat", pady=7, cursor="hand2"
              ).pack(side="right", expand=True, fill="x")

    root.wait_window(dlg)
    return result[0] if result[0] is not None else [0]


def get_output_params(extension):
    ext = extension.lower()
    audio = {'mp3':  {'acodec': 'mp3',        'ab': '192k'},
             'm4a':  {'acodec': 'aac',         'ab': '192k'},
             'wav':  {'acodec': 'pcm_s16le'},
             'flac': {'acodec': 'flac'},
             'ogg':  {'acodec': 'libvorbis',   'ab': '192k'}}
    video = {'mp4':  {'vcodec': 'copy', 'acodec': 'aac', 'ab': '192k'},
             'mkv':  {'vcodec': 'copy', 'acodec': 'aac', 'ab': '192k'},
             'avi':  {'vcodec': 'copy', 'acodec': 'mp3', 'ab': '192k'}}
    if ext in audio:
        return audio[ext], False
    if ext in video:
        return video[ext], True
    return {'acodec': 'aac', 'ab': '192k'}, False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _process_single_track(file_path, input_, track_idx, output_path,
                           video, audio_streams, keep_others=True, duration=0):
    global process1, process2, _streaming_active
    _streaming_active = os.path.exists(_AUDIO_PIPE)

    print(f"  Extracting track a:{track_idx}…")
    update_status(f"Extracting audio (track {track_idx})…")

    process1 = (input_[f'a:{track_idx}']
                .output('pipe:1', format='f32le', ac=1, ar=sample_rate)
                .overwrite_output()
                .run_async(pipe_stdin=False, pipe_stdout=True))

    # Chunked read with live progress
    expected = int(duration * sample_rate * 4) if duration > 0 else 0
    chunks   = []
    read_so_far = 0
    READ_CHUNK  = 8 * 1024 * 1024   # 8 MB
    iteration   = 0
    while True:
        blk = process1.stdout.read(READ_CHUNK)
        if not blk:
            break
        chunks.append(blk)
        read_so_far += len(blk)
        iteration   += 1
        if iteration % 4 == 0 and expected > 0 and progress:
            progress['value'] = min((read_so_far / expected) * 100, 99)
            root.update_idletasks()
    process1.wait()
    process1.kill()

    raw = b''.join(chunks)
    del chunks
    if progress:
        progress['value'] = 0
    print(f"  Extracted: {len(raw) // 4} samples")

    waveform = frombuffer(raw, dtype='<f4').copy()
    del raw
    gc.collect()

    # ── Pre-filter ──────────────────────────────────────────────────────────
    if pre_filter_enabled:
        waveform = audio_prefilter(waveform, sample_rate)

    # ── Separation ──────────────────────────────────────────────────────────
    vocals = separate_with_demucs(waveform, sample_rate)

    del waveform
    gc.collect()

    if vocals is None:
        print(f"  Separation failed for track {track_idx}")
        return False

    vocals = np.ascontiguousarray(vocals, dtype=np.float32)
    gc.collect()

    # ── Post-filter ─────────────────────────────────────────────────────────
    if post_filter_enabled:
        vocals = advanced_postprocess_chunked(vocals, sample_rate, processing_mode)
        gc.collect()

    # ── Export ──────────────────────────────────────────────────────────────
    print(f"  Exporting: {os.path.basename(output_path)}")
    update_status(f"Exporting: {os.path.basename(output_path)}")

    ext        = os.path.splitext(output_path)[1][1:]
    _, is_vid  = get_output_params(ext)
    vp         = ffmpeg_input('pipe:0', format='f32le', ar=sample_rate, ac=1)

    # Write to a temp file first — rename to original atomically on success
    # so we never leave the original in a partial/corrupt state.
    temp_path = output_path + '.ma_temp'

    if is_vid and video:
        n_orig = len(audio_streams)
        if keep_others:
            out_streams = ([input_.video]
                           + [input_[f'a:{i}'] for i in range(n_orig)]
                           + [vp.audio])
            kw = {'vcodec': 'copy'}
            for i in range(n_orig):
                kw[f'c:a:{i}']           = 'copy'
                kw[f'disposition:a:{i}'] = '0'        # remove default from originals
            kw[f'c:a:{n_orig}']           = 'aac'
            kw[f'b:a:{n_orig}']           = '192k'
            kw[f'disposition:a:{n_orig}'] = 'default'  # vocals track is default
            print(f"  Vocals added as track {n_orig} (default)  ({n_orig} original track(s) preserved)")
        else:
            out_streams = [input_.video, vp.audio]
            kw = {'vcodec': 'copy', 'c:a:0': 'aac', 'b:a:0': '192k'}
            print(f"  Vocals-only output (original audio discarded)")
        process2 = (_ffmpeg.output(*out_streams, temp_path, **kw)
                    .overwrite_output().run_async(pipe_stdin=True))
    else:
        params, _ = get_output_params(ext)
        process2  = (vp.output(temp_path, **params)
                     .overwrite_output().run_async(pipe_stdin=True))

    # Stream via memoryview — no extra copy of the vocals array
    WRITE_CHUNK = 8 * 1024 * 1024
    view        = memoryview(vocals.astype('<f4', copy=False))
    del vocals
    for i in range(0, len(view), WRITE_CHUNK):
        process2.stdin.write(view[i:i + WRITE_CHUNK])
    process2.stdin.close()
    process2.wait()
    process2.kill()

    # Atomically replace the original file with the processed version
    os.replace(temp_path, output_path)
    # Write sidecar marker so the VLC extension knows this file is already processed
    with open(output_path + '.ma', 'w') as _f:
        _f.write('')

    _streaming_active = False
    print(f"  SUCCESS: {output_path}  (vocals embedded as new audio track)")
    return True


def demusic_pro(file_path):
    global progress, root

    try:
        file_path = file_path.replace("\\\\", "/")
        name      = os.path.splitext(os.path.basename(file_path))[0]
        ext       = os.path.splitext(file_path)[1][1:]
        path      = os.path.dirname(file_path)

        print(f"\nProcessing: {name}.{ext}")
        print(f"Model: {model_choice}  |  Mode: {processing_mode}"
              f"  |  Quantise: {quantize_enabled}"
              f"  |  Pre: {pre_filter_enabled}  Post: {post_filter_enabled}")
        update_status(f"Processing: {name}")

        input_    = ffmpeg_input(file_path)
        video, audio_streams, subs, duration = check_streams(file_path)
        if not audio_streams:
            print("No audio track found")
            return False

        track_indices = ask_audio_track(audio_streams)
        multi         = len(track_indices) > 1
        all_ok        = True

        for track_idx in track_indices:
            if multi:
                print(f"\n--- Track {track_idx}/{len(track_indices)} ---")
            # Output = same file as input; _process_single_track writes to
            # a temp then atomically renames back to the original.
            out_path = file_path

            ok = _process_single_track(file_path, input_, track_idx, out_path,
                                       video, audio_streams, keep_others_choice, duration)
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
        import traceback; traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

def setFrames(root):
    global b1, b2, f1, a1, listbox, inprogressframe, progress, status_label
    global mode_selector, model_selector, keep_others_selector
    global pre_filter_var, post_filter_var, quantize_var

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
        bordercolor=BORDER, arrowcolor=ACCENT, padding=4)
    style.map('App.TCombobox',
        fieldbackground=[('readonly', CARD)],
        foreground=[('readonly', TEXT)])
    style.configure('App.Horizontal.TProgressbar',
        background=ACCENT, troughcolor='#0a0a12',
        bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT2)

    root.configure(background=BG)

    def divider(parent=root, color=BORDER, h=1):
        Frame(parent, bg=color, height=h).pack(fill="x")

    def section_label(text):
        Label(root, text=text, font=("Segoe UI", 7, "bold"),
              bg=BG, fg=MUTED, anchor="w").pack(fill="x", padx=16, pady=(9, 2))

    def card(parent):
        return Frame(parent, bg=CARD, highlightthickness=1,
                     highlightbackground=BORDER, highlightcolor=ACCENT)

    # ── HEADER ───────────────────────────────────────────────────────────────
    hdr = Frame(root, bg='#0f0f1c')
    hdr.pack(fill="x")
    row = Frame(hdr, bg='#0f0f1c')
    row.pack(fill="x", padx=16, pady=(10, 8))
    Label(row, text="Music Assassin Pro",
          font=("Segoe UI", 15, "bold"), bg='#0f0f1c', fg=TEXT).pack(side="left")
    Label(row, text="v0.5.0",
          font=("Segoe UI", 10), bg='#0f0f1c', fg=ACCENT).pack(side="left", padx=(6, 0), pady=4)
    Label(row, text="DEMUCS  \u2022  CPU Streaming  \u2022  Model Cache",
          font=("Segoe UI", 8), bg='#0f0f1c', fg=MUTED).pack(side="right", pady=4)
    Frame(root, bg=ACCENT, height=2).pack(fill="x")

    # ── MODEL + MODE ─────────────────────────────────────────────────────────
    section_label("CONFIGURATION")
    top_row = Frame(root, bg=BG)
    top_row.pack(fill="x", padx=16)

    model_card = card(top_row)
    model_card.pack(side="left", expand=True, fill="both", padx=(0, 6))
    Label(model_card, text="DEMUCS MODEL", font=("Segoe UI", 7, "bold"),
          bg=CARD, fg=MUTED).pack(anchor="w", padx=10, pady=(7, 2))

    model_opts = []
    if DEMUCS_AVAILABLE:
        model_opts = [
            "htdemucs  \u2014  Fast \u00b7 320 MB",
            "htdemucs_ft  \u2014  Quality \u00b7 320 MB",
            "mdx_extra  \u2014  Best \u00b7 600 MB",
        ]
    if model_opts:
        model_selector = Combobox(model_card, values=model_opts, state="readonly",
                                  font=("Segoe UI", 9), style='App.TCombobox')
        model_selector.set(model_opts[0])
        model_selector.pack(fill="x", padx=10, pady=(0, 4))
        model_selector.bind("<<ComboboxSelected>>", lambda e: set_model())
        Label(model_card, text="Cached for session — loads once",
              font=("Segoe UI", 7), bg=CARD, fg=MUTED
              ).pack(anchor="w", padx=10, pady=(0, 7))
    else:
        Label(model_card, text="No model available",
              font=("Segoe UI", 8), fg=RED, bg=CARD).pack(padx=10, pady=(0, 8), anchor="w")

    mode_card = card(top_row)
    mode_card.pack(side="right", expand=True, fill="both")
    Label(mode_card, text="PROCESSING MODE", font=("Segoe UI", 7, "bold"),
          bg=CARD, fg=MUTED).pack(anchor="w", padx=10, pady=(7, 2))
    mode_selector = Combobox(mode_card, values=["Normal", "Balanced", "Aggressive"],
                             state="readonly", font=("Segoe UI", 9),
                             style='App.TCombobox')
    mode_selector.set("Balanced")
    mode_selector.pack(fill="x", padx=10, pady=(0, 3))
    mode_selector.bind("<<ComboboxSelected>>", lambda e: set_mode())
    Label(mode_card, text="60\u201370%  \u2022  75\u201385%  \u2022  90\u201395% removal",
          font=("Segoe UI", 7), bg=CARD, fg=MUTED).pack(anchor="w", padx=10, pady=(0, 7))

    # ── FILTERS + QUANTISE ───────────────────────────────────────────────────
    filt_card = card(root)
    filt_card.pack(fill="x", padx=16, pady=(6, 0))
    Label(filt_card, text="OPTIONS", font=("Segoe UI", 7, "bold"),
          bg=CARD, fg=MUTED).pack(anchor="w", padx=10, pady=(7, 4))

    opt_row = Frame(filt_card, bg=CARD)
    opt_row.pack(fill="x", padx=10, pady=(0, 7))

    pre_filter_var  = BooleanVar(value=True)
    post_filter_var = BooleanVar(value=True)
    quantize_var    = BooleanVar(value=True)

    ck_style = dict(bg=CARD, fg=TEXT, selectcolor=CARD,
                    activebackground=CARD, activeforeground=ACCENT,
                    highlightthickness=0, font=("Segoe UI", 9))

    Checkbutton(opt_row, text="Pre-filter",  variable=pre_filter_var,
                command=set_pre_filter,  **ck_style).pack(side="left", padx=(0, 18))
    Checkbutton(opt_row, text="Post-filter", variable=post_filter_var,
                command=set_post_filter, **ck_style).pack(side="left", padx=(0, 18))
    Checkbutton(opt_row, text="INT8 Quantise  (2\u20133\u00d7 faster, htdemucs)",
                variable=quantize_var, command=set_quantize, **ck_style).pack(side="left")

    # ── OTHER TRACKS ─────────────────────────────────────────────────────────
    trk_card = card(root)
    trk_card.pack(fill="x", padx=16, pady=(6, 0))
    Label(trk_card, text="AUDIO OUTPUT  (video files)",
          font=("Segoe UI", 7, "bold"), bg=CARD, fg=MUTED
          ).pack(anchor="w", padx=10, pady=(7, 4))
    keep_var = IntVar(value=1)
    keep_others_selector = keep_var
    rb_row = Frame(trk_card, bg=CARD)
    rb_row.pack(fill="x", padx=10, pady=(0, 7))
    Radiobutton(rb_row, text="Keep original + add vocals as new track",
                variable=keep_var, value=1, command=set_keep_others,
                bg=CARD, fg=TEXT, selectcolor=CARD, activebackground=CARD,
                activeforeground=ACCENT, highlightthickness=0,
                font=("Segoe UI", 9)).pack(side="left", padx=(0, 20))
    Radiobutton(rb_row, text="Vocals only  (replace audio)",
                variable=keep_var, value=0, command=set_keep_others,
                bg=CARD, fg=TEXT, selectcolor=CARD, activebackground=CARD,
                activeforeground=ACCENT, highlightthickness=0,
                font=("Segoe UI", 9)).pack(side="left")

    # ── STATUS BAR ───────────────────────────────────────────────────────────
    sb = Frame(root, bg='#0e1a2e', highlightthickness=1,
               highlightbackground=BORDER)
    sb.pack(fill="x", padx=16, pady=(8, 0))
    status_label = Label(sb, text="  Ready \u2014 add files to begin",
                         font=("Segoe UI", 9), bg='#0e1a2e', fg=GREEN, anchor="w")
    status_label.pack(fill="x", padx=4, pady=5)

    # ── FILE BUTTONS ─────────────────────────────────────────────────────────
    btn_row = Frame(root, bg=BG)
    btn_row.pack(fill="x", padx=16, pady=(8, 0))
    b1 = Button(btn_row, text="Browse Files", bg=ACCENT2, fg="white",
                font=("Segoe UI", 9, "bold"), pady=7, relief="flat",
                cursor="hand2", activebackground='#2f59c0', activeforeground='white', bd=0)
    b1.pack(side="left", expand=True, fill="x", padx=(0, 6))
    b2 = Button(btn_row, text="Remove Selected", bg='#5c1a1a', fg='#e8a0a0',
                font=("Segoe UI", 9, "bold"), pady=7, relief="flat",
                cursor="hand2", activebackground='#7a1f1f', activeforeground='white', bd=0)
    b2.pack(side="right", expand=True, fill="x")

    # ── QUEUE ─────────────────────────────────────────────────────────────────
    section_label("PROCESSING QUEUE")
    qcard = card(root)
    qcard.pack(fill="both", expand=True, padx=16)
    scrollbar = Scrollbar(qcard, orient="vertical", bg=BORDER,
                          troughcolor=CARD, activebackground=ACCENT, width=10)
    listbox = Listbox(qcard, yscrollcommand=scrollbar.set, selectmode="multiple",
                      height=4, font=("Segoe UI", 9), bg=CARD, fg=TEXT,
                      selectbackground='#1c2a4a', selectforeground=ACCENT,
                      relief="flat", bd=0, activestyle='none', highlightthickness=0)
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side="right", fill="y", pady=1)
    listbox.pack(side="left", fill="both", expand=True, padx=8, pady=6)

    inprogressframe = Frame(root, bg=BG)
    inprogressframe.pack(fill="x", padx=16)

    # ── PROGRESS ─────────────────────────────────────────────────────────────
    section_label("PROGRESS")
    progress = Progressbar(root, orient="horizontal", mode='determinate',
                           style='App.Horizontal.TProgressbar')
    progress.pack(fill="x", padx=16)

    # ── ACTION BUTTONS ────────────────────────────────────────────────────────
    act_row = Frame(root, bg=BG)
    act_row.pack(fill="x", padx=16, pady=(8, 0))
    f1 = Button(act_row, text="Start Processing", bg='#145233', fg='#7de8b8',
                font=("Segoe UI", 10, "bold"), pady=9, relief="flat",
                cursor="hand2", activebackground='#0f3d26', activeforeground='white', bd=0)
    f1.pack(side="left", expand=True, fill="x", padx=(0, 4))
    a1 = Button(act_row, text="Clear Queue", bg='#1e1410', fg='#c8a96e',
                font=("Segoe UI", 10, "bold"), pady=9, relief="flat",
                cursor="hand2", activebackground='#2a1c0e', activeforeground='white', bd=0)
    a1.pack(side="left", expand=True, fill="x", padx=(0, 4))
    Button(act_row, text="Unload Model", bg='#1a1030', fg='#8899cc',
           font=("Segoe UI", 10, "bold"), pady=9, relief="flat",
           cursor="hand2", activebackground='#221540', activeforeground='white', bd=0,
           command=unload_model).pack(side="right", expand=True, fill="x")

    # ── LOG ───────────────────────────────────────────────────────────────────
    section_label("LOG")
    log_widget = ScrolledText(root, height=5, font=("Consolas", 8),
                              bg='#07070d', fg='#3eb87a', relief="flat", bd=0,
                              insertbackground=ACCENT, selectbackground='#1c2a4a',
                              highlightthickness=1, highlightbackground=BORDER)
    log_widget.pack(fill="both", expand=True, padx=16, pady=(0, 12))
    sys.stdout = PrintLogger(log_widget)


# ─────────────────────────────────────────────────────────────────────────────
# GUI CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def set_model():
    global model_choice
    raw = model_selector.get().split()[0]   # "htdemucs" | "htdemucs_ft" | "mdx_extra"
    model_choice = raw
    # Changing model invalidates any cached version
    _MODEL_CACHE.clear()
    print(f"Model: {model_choice}  (cache cleared)")
    update_status(f"Model: {model_choice}")

def set_mode():
    global processing_mode
    processing_mode = mode_selector.get().lower()
    update_status(f"Mode: {processing_mode}")

def set_keep_others():
    global keep_others_choice
    keep_others_choice = bool(keep_others_selector.get())
    print(f"Other tracks: {'keep' if keep_others_choice else 'discard'}")

def set_pre_filter():
    global pre_filter_enabled
    pre_filter_enabled = bool(pre_filter_var.get())
    print(f"Pre-filter: {pre_filter_enabled}")

def set_post_filter():
    global post_filter_enabled
    post_filter_enabled = bool(post_filter_var.get())
    print(f"Post-filter: {post_filter_enabled}")

def set_quantize():
    global quantize_enabled
    quantize_enabled = bool(quantize_var.get())
    _MODEL_CACHE.clear()   # must reload with new quantisation setting
    print(f"INT8 quantise: {quantize_enabled}  (cache cleared)")


def get_file_path():
    global files, listbox, do
    if do == 1:
        do = 0
    selected = []
    try:
        import subprocess
        r = subprocess.run(
            ["zenity", "--file-selection", "--multiple",
             "--title=Select Audio/Video Files",
             "--file-filter=Media Files | *.mp4 *.mp3 *.wav *.avi *.mkv *.flac",
             "--file-filter=All Files | *"],
            capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            selected = r.stdout.strip().split("|")
    except FileNotFoundError:
        browse = filedialog.askopenfilenames(
            title="Select Audio/Video Files", parent=root,
            filetypes=[("Media", ("*.mp4","*.mp3","*.wav","*.avi","*.mkv","*.flac")),
                       ("All", "*.*")])
        selected = list(browse)
    for f in selected:
        f = f.strip()
        if f and f not in files:
            files.append(f)
            listbox.insert("end", os.path.basename(f))
    update_status(f"Queue: {len(files)} file(s)")


def remove_item():
    global listbox, files
    for idx in reversed(list(listbox.curselection())):
        listbox.delete(idx)
        files.pop(idx)
    update_status(f"Queue: {len(files)} file(s)")


def demusic_list():
    global do, files
    if do == 1 or not files:
        if not files:
            update_status("Add files to start")
        return
    do = 1
    completed, failed = [], []
    update_status(f"Processing {len(files)} file(s)…")

    def process_next():
        global do
        if do == 0 or not files:
            if not files:
                do = 0
                print(f"\n=== COMPLETE ===\nDone: {len(completed)}  |  Failed: {len(failed)}")
                update_status(f"Complete \u2014 {len(completed)} done, {len(failed)} failed")
            return
        fp = files.pop(0)
        if listbox.size() > 0:
            listbox.delete(0)
        print(f"\nProcessing: {os.path.basename(fp)}")
        ok = demusic_pro(fp)
        color_bg = '#071a0f' if ok else '#1a0707'
        color_fg = '#4ade80' if ok else '#f87171'
        sym      = '\u2713' if ok else '\u2717'
        (completed if ok else failed).append(os.path.basename(fp))
        Label(inprogressframe,
              text=f"  {sym}  {os.path.basename(fp)}",
              font=("Segoe UI", 8), bg=color_bg, fg=color_fg,
              anchor="w", pady=2).pack(fill="x", pady=1)
        root.after(500, process_next)

    root.after(100, process_next)


def on_closing():
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = Tk()
    root.wm_title("Music Assassin Pro v0.5.0")
    root.geometry("720x680")
    root.configure(background='#0d0d14')
    root.minsize(640, 580)

    try:
        root.update()
    except Exception:
        os.environ['TCL_LIBRARY'] = '/usr/share/tcltk/tcl8.6'
        os.environ['TK_LIBRARY']  = '/usr/share/tcltk/tk8.6'
        root.destroy()
        root = Tk()
        root.wm_title("Music Assassin Pro v0.5.0")
        root.geometry("720x680")
        root.configure(background='#0d0d14')
        root.minsize(640, 580)

    setFrames(root)
    start_ipc_server()
    _start_http_stream_server()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    b1.configure(command=get_file_path)
    b2.configure(command=remove_item)
    f1.configure(command=demusic_list)
    a1.configure(command=lambda: [files.clear(), listbox.delete(0, 'end'),
                                  update_status("Queue cleared")])

    # ── Startup info ─────────────────────────────────────────────────────────
    try:
        import psutil
        ram_avail = psutil.virtual_memory().available // (1024 * 1024)
        ram_total = psutil.virtual_memory().total    // (1024 * 1024)
        ram_info  = f"RAM: {ram_avail} MB free / {ram_total} MB total"
    except ImportError:
        ram_info = "RAM: install psutil for memory info"

    if ONNX_AVAILABLE:
        onnx_status = "onnxruntime installed — ONNX acceleration ACTIVE"
    else:
        onnx_status = "onnxruntime NOT installed — torch fallback (INT8 quant)"

    print("=" * 58)
    print("Music Assassin Pro v0.5.0  —  CPU Streaming Edition")
    print("=" * 58)
    print(f"DEMUCS available:  {DEMUCS_AVAILABLE}")
    print(f"CPU cores:         {os.cpu_count()}")
    print(ram_info)
    print(f"Inference:  {onnx_status}")
    print("Models:   htdemucs (fast) / htdemucs_ft / mdx_extra")
    print("Pipeline: chunk-based overlap-add  |  ETA per chunk")
    print("Cache:    model stays loaded between files")
    if ONNX_AVAILABLE:
        print(f"ONNX dir: {_ONNX_CACHE_DIR}")
    print("=" * 58)
    update_status("Ready \u2014 select model and add files")

    root.mainloop()
