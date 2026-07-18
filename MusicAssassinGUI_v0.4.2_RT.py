"""
Music Assassin Pro v0.4.2-RT  —  REAL-TIME MODEL TESTING BRANCH
Based on v0.4.2 (MDX-Net / DEMUCS / SPLEETER edition).

[NEW — SpeechBrain DSP+AI pipeline integration]
  HPSS Pre-filter (checkbox):
    librosa.effects.hpss() splits input into harmonic + percussive components
    before the AI separator. Feeds only the harmonic component (vocals + melody)
    to the model — removes drum/rhythm transients which confuse the separator.
    Mirrors SpeechBrain's beamformer→neural chain: DSP cleans gross interference,
    AI handles fine separation. No new dependencies (librosa already required).

  MetricGAN+ Post-filter (checkbox):
    SpeechBrain's speechbrain/metricgan-plus-voicebank runs AFTER the AI separator
    to clean residual music bleed from extracted vocals. Trained to remove non-speech
    content from speech signals — the exact problem after source separation.
    Requires: pip install speechbrain (~300 MB installed, ~50 MB model).
    Resamples 44.1 kHz → 16 kHz → enhance → resample back.

  Together these implement the full SpeechBrain DSP→AI→DSP pipeline:
    [DSP: HPSS pre-filter] → [AI: MDX-Net / Demucs] → [DSP: MetricGAN+ enhance]

What's new in this branch (RT = Real-Time research additions):
─────────────────────────────────────────────────────────────────
[NEW MODELS via audio-separator]
  MDX23C-8KFFT-InstVoc_HQ.ckpt  — MDXC architecture, 448 MB torch model
  MDX23C-8KFFT-InstVoc_HQ_2.ckpt — improved variant, same size
  Both outperform MDX-Net Voc_FT on vocals (SDR ~10 vs ~8.x)
  NOTE: require torch (not just onnxruntime) → slower on CPU than ONNX models

[NEW MODELS — Demucs extras]
  htdemucs_6s  — 6-stem Demucs (adds piano + guitar isolation)
  hdemucs_mmi  — Hybrid Demucs v3 retrained on MusDB HQ
  mdx_q        — Quantized MDX (smaller, ~300 MB, slightly lower quality)
  mdx_extra_q  — Quantized MDX Extra (best quality + quantized)

[RT ARCHITECTURE STUBS — infrastructure only, no public weights yet]
  HS-TasNet    — 23 ms latency hybrid model (pip install HS-TasNet).
                 No pretrained weights available publicly as of Jun 2026.
                 Place a .pt checkpoint in pretrained_models/ to activate.
                 Train on MUSDB18: github.com/lucidrains/HS-TasNet
  RT-STT       — Real-Time Single-Path TFC-TDF UNet (arxiv 2511.13146, Nov 2025).
                 No public implementation yet. Placeholder for when released.

[QUALITY BOOST — polarity-flip TTA]
  Optional Test-Time Augmentation: runs the model twice (normal + polarity-
  flipped), averages results. +0.1–0.3 dB SDR at the cost of 2× runtime.
  Works with any separator. Tick "TTA (2× slower, +quality)" to enable.

CPU LIMITS — important notes:
  ONNX models (MDX-Net Voc_FT, Kim_v2): fastest on CPU, use onnxruntime
  MDX23C .ckpt: requires torch, ~2–4× slower than ONNX MDX-Net on CPU
  Demucs: requires torch, ~3–6× real-time on modern CPU (chunked)
  HS-TasNet (when weights available): designed for <50 ms latency on CPU
  All models benefit from INT8 quantization checkbox where applicable.

Open-source references:
  Demucs:        github.com/facebookresearch/demucs (MIT)
  audio-separator: github.com/nomadkaraoke/python-audio-separator (MIT)
  MDX23C weights: huggingface.co/Blane187/all_public_uvr_models (open)
  HS-TasNet:     github.com/lucidrains/HS-TasNet (MIT)
  ZFTurbo models: github.com/ZFTurbo/Music-Source-Separation-Training (Apache)
"""

import os, sys, gc, time, logging

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

import importlib.util
DEMUCS_AVAILABLE          = (importlib.util.find_spec('torch') is not None and
                              importlib.util.find_spec('demucs') is not None)
SPLEETER_AVAILABLE        = importlib.util.find_spec('spleeter') is not None
AUDIO_SEPARATOR_AVAILABLE = importlib.util.find_spec('audio_separator') is not None
HS_TASNET_AVAILABLE       = importlib.util.find_spec('hs_tasnet') is not None
SPEECHBRAIN_AVAILABLE     = importlib.util.find_spec('speechbrain') is not None
LIBROSA_AVAILABLE         = importlib.util.find_spec('librosa') is not None

if not DEMUCS_AVAILABLE:       print("WARNING: DEMUCS not installed.")
if not SPLEETER_AVAILABLE:     print("WARNING: Spleeter not installed.")
if not AUDIO_SEPARATOR_AVAILABLE: print("WARNING: audio-separator not installed. MDX-Net / MDX23C unavailable.")
if not HS_TASNET_AVAILABLE:    print("INFO: hs-tasnet not installed. Run: pip install HS-TasNet")
if not SPEECHBRAIN_AVAILABLE:  print("INFO: speechbrain not installed. MetricGAN+ post-filter unavailable. Run: pip install speechbrain")
if not LIBROSA_AVAILABLE:      print("INFO: librosa not installed. HPSS pre-filter unavailable. Run: pip install librosa")

# ============================================================================
# GLOBALS & CONFIG
# ============================================================================

files = []
process1 = process2 = separator = do = 0
sample_rate = 44100
b1 = b2 = f1 = a1 = listbox = inprogressframe = progress = None
status_label = mode_selector = separator_selector = quantize_var = tta_var = None
root = None
status_text  = "Ready"
processing_mode  = "balanced"       # normal | balanced | aggressive
separator_choice = "audio_separator" # audio_separator | demucs | spleeter | hs_tasnet | rt_stt
model_choice     = "UVR-MDX-NET-Voc_FT.onnx"
quantize_enabled      = True
tta_enabled           = False   # polarity-flip TTA (+quality, 2× slower)
hpss_prefilter_enabled = False  # SpeechBrain DSP stage 1: HPSS harmonic pre-filter
speech_enhance_enabled = False  # SpeechBrain DSP stage 2: MetricGAN+ post-filter

_MODEL_CACHE     = {}               # torch model cache — survives batch runs
_AUDIO_SEP_CACHE = {}               # audio-separator Separator cache

# pretrained_models dir — used for HS-TasNet checkpoints etc.
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models")

# ── Logging ──────────────────────────────────────────────────────────────────
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "music_assassin.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(_LOG_PATH, encoding="utf-8")],
)
_log = logging.getLogger("MusicAssassin")
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

_log.info("=" * 60)
_log.info("Music Assassin v0.4.2-RT — RT model testing branch")
_log.info(f"DEMUCS: {DEMUCS_AVAILABLE} | Spleeter: {SPLEETER_AVAILABLE} | "
          f"audio-sep: {AUDIO_SEPARATOR_AVAILABLE} | HS-TasNet: {HS_TASNET_AVAILABLE}")


class PrintLogger:
    _GUI_SKIP = (
        "  built with", "  configuration:", "  lib", "Press [q]",
        "Stream mapping:", "Output #", "Input #", "  Stream #",
        "  Metadata:", "  Duration:", "  major_brand", "  minor_version",
        "  compatible_brands", "  encoder", "  handler_name", "  vendor_id",
        "[out#", "[in#", "[aac", "[aist", "size=", "Guessed Channel", "ffmpeg version",
    )

    def __init__(self, textbox):
        self.textbox = textbox
        self._buf = []

    def write(self, text):
        if not text: return
        for line in text.splitlines():
            if line.strip(): _log.info(line)
        if not any(text.startswith(p) for p in self._GUI_SKIP):
            self._buf.append(text)
            if len(self._buf) >= 5 or "\n" in text:
                self._flush_to_widget()

    def _flush_to_widget(self):
        if not self._buf: return
        combined = "".join(self._buf)
        self._buf = []
        self.textbox.configure(state="normal")
        self.textbox.insert("end", combined)
        self.textbox.see("end")
        self.textbox.configure(state="disabled")

    def flush(self): self._flush_to_widget()


def update_status(message):
    global status_text
    status_text = message
    if status_label:
        status_label.config(text=f"  {message}")
        root.update_idletasks()

def update_progress(value, total):
    if progress:
        progress['value'] = (value / total) * 100
        root.update_idletasks()

# ============================================================================
# AUDIO DSP — Pre/Post Processing (unchanged from v0.4.2)
# ============================================================================

def audio_prefilter(waveform, sr=44100):
    update_status("Pre-filtering audio...")
    print("Pre-filtering: gate + high-pass + normalize")
    threshold = 10**(-40/20)
    waveform = waveform * (np.abs(waveform) > threshold)
    sos_hp = butter(4, 40/(sr/2), btype='highpass', output='sos')
    waveform = sosfilt(sos_hp, waveform)
    max_val = np.max(np.abs(waveform))
    if max_val > 1e-10:
        waveform = waveform / max_val * 0.98
    return waveform.astype(np.float32)


def advanced_postprocess_chunked(vocals, sr=44100, mode='balanced', chunk_size=None):
    print(f"\nAdvanced post-processing ({mode} mode - CHUNKED)...")
    update_status(f"Post-processing: {mode} mode...")
    if chunk_size is None:
        chunk_size = int(sr * 30)
    overlap_size = int(sr * 2)
    num_chunks = int(np.ceil(len(vocals) / (chunk_size - overlap_size)))
    output = np.zeros_like(vocals, dtype=np.float32)

    for chunk_idx in range(num_chunks):
        update_progress(chunk_idx, num_chunks)
        update_status(f"Post-processing: chunk {chunk_idx+1}/{num_chunks}")
        start = chunk_idx * (chunk_size - overlap_size)
        end   = min(start + chunk_size, len(vocals))
        chunk = vocals[start:end].copy()

        if mode == 'normal':
            chunk = _postprocess_chunk_normal(chunk, sr)
        elif mode == 'balanced':
            chunk = _postprocess_chunk_balanced(chunk, sr)
        elif mode == 'aggressive':
            chunk = _postprocess_chunk_aggressive(chunk, sr)

        if chunk_idx == 0:
            output[start:start+len(chunk)] = chunk
        else:
            fade_len = min(overlap_size, len(chunk))
            if fade_len > 0:
                fade_in  = np.linspace(0, 1, fade_len, dtype=np.float32)
                fade_out = 1 - fade_in
                output[start:start+fade_len] = (
                    output[start:start+fade_len] * fade_out + chunk[:fade_len] * fade_in
                )
                output[start+fade_len:start+len(chunk)] = chunk[fade_len:]
        del chunk
        gc.collect()
        root.update_idletasks()

    peak = np.max(np.abs(output))
    if peak > 1e-10:
        output *= 0.98 / peak
    fade_samps = min(int(sr * 0.5), len(output))
    if fade_samps > 0:
        output[-fade_samps:] *= np.linspace(1.0, 0.0, fade_samps, dtype=np.float32)
    print("Post-processing complete")
    return output


def _postprocess_chunk_normal(chunk, sr):
    sos_hp = butter(4, 80/(sr/2), btype='highpass', output='sos')
    sos_lp = butter(4, 10000/(sr/2), btype='lowpass', output='sos')
    return sosfilt(sos_lp, sosfilt(sos_hp, chunk)).astype(np.float32)

def _postprocess_chunk_balanced(chunk, sr):
    sos_low = butter(4, 40/(sr/2), btype='highpass', output='sos')
    chunk   = sosfilt(sos_low, chunk)
    sos_sib = butter(4, [5000/(sr/2), 8000/(sr/2)], btype='band', output='sos')
    sibilants = sosfilt(sos_sib, chunk)
    sib_rms   = np.sqrt(np.mean(sibilants**2) + 1e-10)
    if sib_rms > 0.20:
        reduction = np.clip(0.20 / sib_rms, 0.5, 1.0)
        chunk = chunk - sibilants * (1.0 - reduction) * 0.4
    sos_f    = butter(2, [1000/(sr/2), 3000/(sr/2)], btype='band', output='sos')
    formants = sosfilt(sos_f, chunk)
    return (chunk + formants * 0.15).astype(np.float32)

def _postprocess_chunk_aggressive(chunk, sr):
    sos_low = butter(4, 60/(sr/2), btype='highpass', output='sos')
    chunk   = sosfilt(sos_low, chunk)
    sos_sib = butter(4, [5000/(sr/2), 8000/(sr/2)], btype='band', output='sos')
    sibilants = sosfilt(sos_sib, chunk)
    sib_rms   = np.sqrt(np.mean(sibilants**2) + 1e-10)
    if sib_rms > 0.15:
        reduction = np.clip(0.15 / sib_rms, 0.4, 1.0)
        chunk = chunk - sibilants * (1.0 - reduction) * 0.5
    sos_f    = butter(2, [1000/(sr/2), 3500/(sr/2)], btype='band', output='sos')
    formants = sosfilt(sos_f, chunk)
    return (chunk + formants * 0.25).astype(np.float32)

# ============================================================================
# TTA — polarity-flip Test-Time Augmentation
# ============================================================================

def apply_tta_polarity(separate_fn, waveform, *args, **kwargs):
    """Run separation twice (normal + polarity-flipped), average results.

    For a well-trained model f:  f(-x) ≈ -f(x).
    Averaging  (f(x) - f(-x)) / 2  cancels asymmetric model biases.
    Effect: +0.1–0.3 dB SDR. Cost: 2× inference time.
    """
    print("  TTA pass 1/2 (normal polarity)...")
    result_pos = separate_fn(waveform, *args, **kwargs)
    if result_pos is None:
        return None

    print("  TTA pass 2/2 (inverted polarity)...")
    result_neg = separate_fn(-waveform, *args, **kwargs)
    if result_neg is None:
        print("  TTA pass 2 failed — using pass 1 only")
        return result_pos

    averaged = (result_pos - result_neg) / 2.0
    print("  TTA: averaged both passes")
    return averaged.astype(np.float32)

# ============================================================================
# SPEECHBRAIN DSP+AI PIPELINE — Stage 1: HPSS Pre-filter
# ============================================================================

def apply_hpss_prefilter(waveform, sr=44100):
    """Harmonic-Percussive Source Separation pre-filter (SpeechBrain DSP Stage 1).

    Chunked: 30 s chunks, 2 s overlap, linear crossfade at boundaries.
    HPSS uses a non-causal median filter so boundary effects are real — overlap
    ensures each output sample has clean context on both sides.

    margin=3.0: H/P separation sharpness.
    Blend 85% harmonic + 15% original: preserves transient naturalness.
    """
    if not LIBROSA_AVAILABLE:
        print("HPSS: librosa not installed (pip install librosa) — skipping")
        return waveform

    import librosa

    CHUNK_SEC   = 30
    OVERLAP_SEC = 2
    chunk_samp  = int(CHUNK_SEC * sr)
    overlap_samp = int(OVERLAP_SEC * sr)
    step_samp   = chunk_samp - overlap_samp
    total_samp  = len(waveform)
    starts      = list(range(0, total_samp, step_samp))
    n_chunks    = len(starts)
    result      = np.zeros(total_samp, dtype=np.float32)

    print(f"HPSS pre-filter: {total_samp/sr:.1f}s | {n_chunks} × {CHUNK_SEC}s chunks")

    for idx, start in enumerate(starts):
        end   = min(start + chunk_samp, total_samp)
        chunk = waveform[start:end]
        update_status(f"HPSS: chunk {idx+1}/{n_chunks}")
        root.update_idletasks()

        try:
            harmonic, percussive = librosa.effects.hpss(chunk, margin=3.0)
            p_rms = np.sqrt(np.mean(percussive**2) + 1e-10)
            h_rms = np.sqrt(np.mean(harmonic**2) + 1e-10)
            blend = 0.95 if (p_rms / max(h_rms, 1e-10)) < 0.1 else 0.85
            chunk_out = (harmonic * blend + chunk * (1.0 - blend)).astype(np.float32)
        except Exception as e:
            print(f"  HPSS chunk {idx+1} error: {e} — using original")
            chunk_out = chunk.astype(np.float32)

        if idx == 0:
            result[start:end] = chunk_out
        else:
            fade_len = min(overlap_samp, len(chunk_out), end - start)
            fade_in  = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade_out = 1.0 - fade_in
            result[start:start+fade_len] = (
                result[start:start+fade_len] * fade_out + chunk_out[:fade_len] * fade_in
            )
            result[start+fade_len:end] = chunk_out[fade_len:end-start]

        del chunk_out
        gc.collect()

    print(f"  HPSS complete ({n_chunks} chunks)")
    return result


# ============================================================================
# SPEECHBRAIN DSP+AI PIPELINE — Stage 2: MetricGAN+ Post-filter
# ============================================================================

def apply_metricgan_enhance(vocals, sr=44100):
    """MetricGAN+ post-filter (SpeechBrain DSP Stage 2).

    Runs after the AI separator to remove residual music bleed from extracted vocals.
    Trained on Voicebank-DEMAND: removes broadband noise/interference from speech.
    Optimised for perceptual quality (PESQ metric via adversarial training).

    Model: speechbrain/metricgan-plus-voicebank (~50 MB, Apache 2.0)
    Input: 16 kHz mono — resamples 44.1 kHz → 16 kHz → enhance → resample back.
    Install: pip install speechbrain

    Pipeline position:
      [AI separator output] → [MetricGAN+ 16 kHz] → [advanced_postprocess_chunked]

    This completes the SpeechBrain DSP→AI→DSP chain:
      HPSS (optional) → MDX-Net/Demucs → MetricGAN+ (optional) → existing postprocess
    """
    if not SPEECHBRAIN_AVAILABLE:
        print("MetricGAN+: speechbrain not installed (pip install speechbrain) — skipping")
        return vocals

    import torch
    import torchaudio

    ENHANCE_SR = 16000
    cache_key  = "metricgan_plus_voicebank"

    try:
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
    except ImportError:
        print("MetricGAN+: speechbrain installed but inference module missing — skipping")
        return vocals

    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        print("Loading MetricGAN+ (speechbrain/metricgan-plus-voicebank)...")
        update_status("Loading MetricGAN+ (~50 MB)...")
        try:
            savedir = os.path.join(_MODELS_DIR, "metricgan-plus-voicebank")
            model = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir=savedir,
                run_opts={"device": "cpu"},
            )
            _MODEL_CACHE[cache_key] = model
            print(f"  MetricGAN+ loaded and cached (saved to {savedir})")
        except Exception as e:
            print(f"MetricGAN+ load error: {e} — skipping enhancement")
            return vocals
    else:
        print("  MetricGAN+ reused from cache")

    try:
        # Chunked: process 30 s at a time at 44.1 kHz (= ~7.3 s at 16 kHz).
        # MetricGAN+ is stateless so no overlap needed for correctness, but
        # a short 0.5 s crossfade smooths any level discontinuities.
        CHUNK_SEC    = 30
        OVERLAP_SEC  = 0.5
        chunk_samp   = int(CHUNK_SEC * sr)
        overlap_samp = int(OVERLAP_SEC * sr)
        step_samp    = chunk_samp - overlap_samp
        total_samp   = len(vocals)
        starts       = list(range(0, total_samp, step_samp))
        n_chunks     = len(starts)
        result       = np.zeros(total_samp, dtype=np.float32)

        print(f"  MetricGAN+: {total_samp/sr:.1f}s | {n_chunks} × {CHUNK_SEC}s chunks → 16 kHz")

        for idx, start in enumerate(starts):
            end   = min(start + chunk_samp, total_samp)
            chunk = vocals[start:end]
            update_status(f"MetricGAN+: chunk {idx+1}/{n_chunks}")
            root.update_idletasks()

            chunk_t    = torch.from_numpy(chunk).float().unsqueeze(0)
            chunk_16k  = torchaudio.functional.resample(chunk_t, sr, ENHANCE_SR)

            with torch.no_grad():
                enh_16k = model.enhance_batch(
                    chunk_16k, lengths=torch.tensor([1.0])
                )

            enh = torchaudio.functional.resample(enh_16k, ENHANCE_SR, sr)
            chunk_out = enh.squeeze(0).numpy().astype(np.float32)

            # trim/pad for resampling rounding
            target_len = end - start
            if len(chunk_out) > target_len:
                chunk_out = chunk_out[:target_len]
            elif len(chunk_out) < target_len:
                chunk_out = np.pad(chunk_out, (0, target_len - len(chunk_out)))

            if idx == 0:
                result[start:end] = chunk_out
            else:
                fade_len = min(overlap_samp, len(chunk_out), end - start)
                fade_in  = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                fade_out = 1.0 - fade_in
                result[start:start+fade_len] = (
                    result[start:start+fade_len] * fade_out + chunk_out[:fade_len] * fade_in
                )
                result[start+fade_len:end] = chunk_out[fade_len:end-start]

            del chunk_t, chunk_16k, enh_16k, enh, chunk_out
            gc.collect()

        before_rms = np.sqrt(np.mean(vocals**2) + 1e-10)
        after_rms  = np.sqrt(np.mean(result**2)  + 1e-10)
        print(f"  MetricGAN+ complete: RMS {before_rms:.4f} → {after_rms:.4f}")
        return result

    except Exception as e:
        print(f"MetricGAN+ enhance error: {e} — returning unenhanced vocals")
        import traceback; traceback.print_exc()
        return vocals


# ============================================================================
# SEPARATION — shared helpers
# ============================================================================

def _run_in_background(fn, status_msg):
    import threading
    result = [None]
    error  = [None]
    def worker():
        try:   result[0] = fn()
        except Exception as e: error[0] = e
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
    if error[0]: raise error[0]
    return result[0]

# ============================================================================
# SEPARATION — DEMUCS (unchanged core, expanded model list)
# ============================================================================

def _load_demucs_model(name, dev, use_fp16, do_quantize, n_threads):
    cache_key = f"{name}:{'q' if (do_quantize and 'htdemucs' in name) else 'fp32'}:{dev.type}"
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        print(f"  Model '{cache_key}' reused from cache")
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
        try:
            import torch
            torch.set_num_threads(n_threads)
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("  INT8 quantization applied")
        except Exception as ex:
            print(f"  Quantization failed ({ex}); using FP32")

    _MODEL_CACHE[cache_key] = model
    print(f"  Cached as '{cache_key}'")
    return model


def separate_with_demucs(waveform, sr=44100):
    if not DEMUCS_AVAILABLE:
        print("ERROR: DEMUCS not available")
        return None
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024**3)
        print(f"  Available RAM: {avail_gb:.1f} GB")
        if avail_gb < 0.5:
            print(f"ERROR: Low RAM ({avail_gb:.1f} GB). Free up memory.")
            update_status(f"ERROR: Low RAM ({avail_gb:.1f} GB)")
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

        dev      = torch_device("cuda" if cuda_available() else "cpu")
        use_fp16 = (dev.type == "cuda")
        model    = _load_demucs_model(model_choice, dev, use_fp16, quantize_enabled, n_threads)

        if use_fp16:
            print(f"  Device: {dev} (fp16)")
        else:
            print(f"  Device: CPU | model: {model_choice}"
                  f"{'  (INT8)' if (quantize_enabled and 'htdemucs' in model_choice) else ''}")

        # vocals index — htdemucs_6s has 6 sources; others have 4
        try:
            vocals_idx = model.sources.index('vocals')
        except (AttributeError, ValueError):
            vocals_idx = 0  # fallback
        duration_s   = len(waveform) / sr
        CHUNK_SEC    = 60
        OVERLAP_SEC  = 2
        chunk_samp   = int(CHUNK_SEC * sr)
        overlap_samp = int(OVERLAP_SEC * sr)
        step_samp    = chunk_samp - overlap_samp
        total_samp   = len(waveform)
        starts       = list(range(0, total_samp, step_samp))
        n_chunks     = len(starts)
        vocals_out   = np.zeros(total_samp, dtype=np.float32)

        print(f"  {duration_s:.1f}s | {n_chunks} × {CHUNK_SEC}s chunks | overlap {OVERLAP_SEC}s | {dev}")
        if progress: progress.configure(mode='determinate'); progress['value'] = 0
        t0 = time.time()

        for idx, start in enumerate(starts):
            end   = min(start + chunk_samp, total_samp)
            chunk = waveform[start:end]
            stereo = np.stack([chunk, chunk], axis=0)
            t = from_numpy(stereo).float().unsqueeze(0).to(dev)
            if use_fp16: t = t.half()

            _result = [None]; _err = [None]
            def _worker(tensor=t):
                try:
                    with no_grad():
                        _result[0] = apply_model(
                            model, tensor, device=dev,
                            segment=CHUNK_SEC, overlap=0.25, progress=False, num_workers=0,
                        )
                except Exception as e: _err[0] = e

            chunk_dur = (end - start) / sr
            update_status(f"DEMUCS: chunk {idx+1}/{n_chunks}  ({chunk_dur:.0f}s)...")
            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()

            est_chunk = chunk_dur * 0.75
            t_chunk = time.time()
            while thread.is_alive():
                elapsed = time.time() - t_chunk
                frac    = min(elapsed / max(est_chunk, 1), 0.99)
                overall = (idx + frac) / n_chunks
                if progress: progress['value'] = overall * 100
                eta = max(0, int((time.time()-t0) / max(overall,0.01) * (1-overall)))
                eta_str = f"{eta//60}m {eta%60:02d}s" if eta >= 60 else f"{eta}s"
                update_status(f"DEMUCS: chunk {idx+1}/{n_chunks} | {overall*100:.0f}% | ETA {eta_str}")
                root.update(); time.sleep(0.033)

            thread.join()
            if _err[0]: raise _err[0]

            chunk_vocals = _result[0][0, vocals_idx].float().mean(0).cpu().numpy()
            del _result[0], t, stereo

            if idx == 0:
                vocals_out[start:end] = chunk_vocals
            else:
                fade_len = min(overlap_samp, len(chunk_vocals), end - start)
                fade_in  = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                fade_out = 1.0 - fade_in
                vocals_out[start:start+fade_len] = (
                    vocals_out[start:start+fade_len] * fade_out +
                    chunk_vocals[:fade_len] * fade_in
                )
                vocals_out[start+fade_len:end] = chunk_vocals[fade_len:end-start]

            del chunk_vocals, chunk
            if dev.type == "cuda": cuda_empty_cache()
            gc.collect()

        elapsed = time.time() - t0
        print(f"  Separation done in {elapsed:.1f}s ({elapsed/duration_s:.2f}× real-time)")
        if progress: progress['value'] = 100
        return vocals_out

    except Exception as e:
        print(f"DEMUCS error: {e}")
        import traceback; traceback.print_exc()
        return None

# ============================================================================
# SEPARATION — SPLEETER (unchanged from v0.4.2)
# ============================================================================

def separate_with_spleeter(waveform, sr=44100):
    if not SPLEETER_AVAILABLE:
        print("ERROR: Spleeter not available"); return None
    print("Loading Spleeter 4-stems..."); update_status("Loading Spleeter...")
    try:
        try:
            import tensorflow as tf
            n = os.cpu_count() or 4
            tf.config.threading.set_intra_op_parallelism_threads(n)
            tf.config.threading.set_inter_op_parallelism_threads(n)
        except Exception: pass

        from spleeter.separator import Separator as SplSep
        sep = SplSep('spleeter:4stems')

        CHUNK_SEC    = 60; OVERLAP_SEC = 2
        chunk_samp   = int(CHUNK_SEC * sr); overlap_samp = int(OVERLAP_SEC * sr)
        step_samp    = chunk_samp - overlap_samp
        total_samp   = len(waveform); starts = list(range(0, total_samp, step_samp))
        n_chunks     = len(starts); vocals_out = np.zeros(total_samp, dtype=np.float32)

        print(f"  {total_samp/sr:.1f}s | {n_chunks} × {CHUNK_SEC}s chunks")
        if progress: progress.configure(mode='determinate'); progress['value'] = 0

        for idx, start in enumerate(starts):
            end   = min(start + chunk_samp, total_samp)
            chunk = waveform[start:end].reshape(-1, 1)
            update_status(f"Spleeter: chunk {idx+1}/{n_chunks}...")
            _r = [None]
            _run_in_background(lambda c=chunk: _r.__setitem__(0, sep.separate(c)), f"Spleeter: chunk {idx+1}/{n_chunks}...")
            chunk_vocals = _r[0]['vocals'].flatten()
            if idx == 0:
                vocals_out[start:end] = chunk_vocals
            else:
                fade_len = min(overlap_samp, len(chunk_vocals), end - start)
                fi = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                fo = 1.0 - fi
                vocals_out[start:start+fade_len] = (
                    vocals_out[start:start+fade_len] * fo + chunk_vocals[:fade_len] * fi
                )
                vocals_out[start+fade_len:end] = chunk_vocals[fade_len:end-start]
            del chunk_vocals, chunk, _r[0]; gc.collect()
            if progress: progress['value'] = (idx+1)/n_chunks * 100

        sep.close(); gc.collect()
        return vocals_out
    except Exception as e:
        print(f"Spleeter error: {e}"); return None

# ============================================================================
# SEPARATION — MDX-Net ONNX + MDX23C .ckpt  (via audio-separator)
# ============================================================================

def separate_with_audio_separator(waveform, sr=44100, mdl="UVR-MDX-NET-Voc_FT.onnx"):
    """Handles both MDX-Net ONNX models AND MDX23C .ckpt models via audio-separator.

    Chunked: 60 s chunks, 2 s overlap, linear crossfade. Each chunk is written
    to a temp WAV, separated, and the vocals output is read back. The Separator
    object is cached across chunks (model stays loaded in memory).

    MDX-Net ONNX (Voc_FT, Kim_Vocal_2): onnxruntime, fast CPU, 66–130 MB.
    MDX23C .ckpt (InstVoc_HQ, HQ_2): torch MDXC, 448 MB, slower on CPU.
    """
    if not AUDIO_SEPARATOR_AVAILABLE:
        print("ERROR: audio-separator not installed. Run: pip install audio-separator onnxruntime")
        return None

    import tempfile, glob, soundfile as sf
    from audio_separator.separator import Separator

    is_mdx23c  = mdl.endswith('.ckpt')
    arch_label = "MDX23C (torch)" if is_mdx23c else "MDX-Net (ONNX)"
    print(f"{arch_label}: model={mdl}")
    if is_mdx23c:
        print(f"  NOTE: MDX23C is ~2–4× slower than ONNX MDX-Net on CPU")
    update_status(f"{arch_label}: loading {mdl}...")

    try:
        # ── Load / cache model once ──────────────────────────────────────────
        sep = _AUDIO_SEP_CACHE.get(mdl)
        if sep is None:
            tmp_init = tempfile.mkdtemp(prefix="music_assassin_init_")
            sep = Separator(
                output_dir=tmp_init,
                output_format="WAV",
                normalization_threshold=0.9,
                output_single_stem="Vocals",
            )
            sep.load_model(model_filename=mdl)
            _AUDIO_SEP_CACHE[mdl] = sep
            print(f"  Model '{mdl}' loaded and cached")
        else:
            print(f"  Model '{mdl}' reused from cache")

        # ── Chunked separation ───────────────────────────────────────────────
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

        print(f"  {duration_s:.1f}s | {n_chunks} × {CHUNK_SEC}s chunks | overlap {OVERLAP_SEC}s")
        if progress: progress.configure(mode='determinate'); progress['value'] = 0

        for idx, start in enumerate(starts):
            end   = min(start + chunk_samp, total_samp)
            chunk = waveform[start:end]

            tmp_dir = tempfile.mkdtemp(prefix="music_assassin_")
            tmp_in  = os.path.join(tmp_dir, "chunk.wav")
            sf.write(tmp_in, chunk.astype(np.float32), sr)
            sep.output_dir = tmp_dir

            update_status(f"{arch_label}: chunk {idx+1}/{n_chunks}  ({(end-start)/sr:.0f}s)...")
            _run_in_background(lambda d=tmp_dir, f=tmp_in: sep.separate(f),
                               f"{arch_label}: chunk {idx+1}/{n_chunks}...")

            vocals_files = glob.glob(os.path.join(tmp_dir, "*Vocals*"))
            if not vocals_files:
                print(f"  WARNING: chunk {idx+1} produced no vocals — using silence")
                chunk_vocals = np.zeros(end - start, dtype=np.float32)
            else:
                chunk_vocals, out_sr = sf.read(vocals_files[0], dtype='float32')
                if chunk_vocals.ndim > 1:
                    chunk_vocals = chunk_vocals.mean(axis=1)
                chunk_vocals = chunk_vocals.astype(np.float32)

            # Trim/pad for resampler rounding
            target_len = end - start
            if len(chunk_vocals) > target_len:
                chunk_vocals = chunk_vocals[:target_len]
            elif len(chunk_vocals) < target_len:
                chunk_vocals = np.pad(chunk_vocals, (0, target_len - len(chunk_vocals)))

            # Overlap-add crossfade
            if idx == 0:
                vocals_out[start:end] = chunk_vocals
            else:
                fade_len = min(overlap_samp, len(chunk_vocals), end - start)
                fade_in  = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                fade_out = 1.0 - fade_in
                vocals_out[start:start+fade_len] = (
                    vocals_out[start:start+fade_len] * fade_out +
                    chunk_vocals[:fade_len] * fade_in
                )
                vocals_out[start+fade_len:end] = chunk_vocals[fade_len:end-start]

            # Clean up temp files for this chunk
            for f in glob.glob(os.path.join(tmp_dir, "*")):
                try: os.remove(f)
                except OSError: pass
            try: os.rmdir(tmp_dir)
            except OSError: pass

            del chunk, chunk_vocals
            gc.collect()
            if progress: progress['value'] = (idx+1) / n_chunks * 100
            root.update_idletasks()

        if progress: progress['value'] = 100
        print(f"  {arch_label} complete: {total_samp} samples")
        return vocals_out

    except Exception as e:
        print(f"{arch_label} error: {e}")
        import traceback; traceback.print_exc()
        return None

# ============================================================================
# SEPARATION — HS-TasNet  (23 ms latency architecture, no public weights yet)
# ============================================================================

def separate_with_hs_tasnet(waveform, sr=44100):
    """HS-TasNet: Hybrid Spectrogram-TasNet for real-time low-latency separation.

    Architecture: hybrid waveform + spectrogram branches, causal convolutions.
    Target latency: 23 ms @ 44100 Hz.
    SDR: 4.65 on MusDB (vs 8.5 for HTDemucs — quality/latency trade-off).
    Paper: arxiv 2402.17701 (L-Acoustics, Feb 2024).
    GitHub: github.com/lucidrains/HS-TasNet  (MIT license)
    Install: pip install HS-TasNet

    STATUS: No public pretrained weights as of June 2026.
    To activate: train on MUSDB18 and place checkpoint (.pt) in:
      {pretrained_models}/hs_tasnet_musdb.pt
    OR any .pt file matching *hs*tasnet*.pt or *hs_tasnet* in that directory.
    """
    if not HS_TASNET_AVAILABLE:
        print("HS-TasNet not installed. Run: pip install HS-TasNet")
        update_status("HS-TasNet: not installed — pip install HS-TasNet")
        return None

    # Look for a checkpoint in pretrained_models/
    ckpt_path = None
    if os.path.isdir(_MODELS_DIR):
        candidates = [
            f for f in os.listdir(_MODELS_DIR)
            if f.endswith('.pt') and (
                'hs_tasnet' in f.lower() or
                ('hs' in f.lower() and 'tasnet' in f.lower())
            )
        ]
        if candidates:
            ckpt_path = os.path.join(_MODELS_DIR, sorted(candidates)[0])
            print(f"  HS-TasNet checkpoint: {ckpt_path}")

    if ckpt_path is None:
        print("=" * 55)
        print("HS-TasNet: NO PRETRAINED WEIGHTS FOUND")
        print(f"  Place a trained checkpoint (.pt) in:")
        print(f"  {_MODELS_DIR}/")
        print("  Expected name pattern: *hs_tasnet*.pt")
        print("")
        print("  To get weights:")
        print("  1. Train on MUSDB18:")
        print("     github.com/lucidrains/HS-TasNet")
        print("     python train.py --dataset_path /path/to/musdb18hq")
        print("  2. Use the ICASSP 2024 reference implementation:")
        print("     github.com/temismink/HS-Tasnet")
        print("  3. Wait for community-trained weights (none public yet)")
        print("=" * 55)
        update_status("HS-TasNet: no weights — see log for instructions")
        return None

    try:
        from hs_tasnet import HSTasNet
        import torch

        print("Loading HS-TasNet from checkpoint...")
        update_status("Loading HS-TasNet checkpoint...")

        cache_key = f"hs_tasnet:{ckpt_path}"
        model = _MODEL_CACHE.get(cache_key)
        if model is None:
            model = HSTasNet.init_and_load_from(ckpt_path)
            model.eval()
            _MODEL_CACHE[cache_key] = model
            print("  HS-TasNet loaded and cached")
        else:
            print("  HS-TasNet reused from cache")

        # HS-TasNet is causal and designed for streaming, so small chunks are fine.
        # Use 10 s chunks (low RAM) with 0.5 s overlap.
        CHUNK_SEC    = 10
        OVERLAP_SEC  = 0.5
        chunk_samp   = int(CHUNK_SEC * sr)
        overlap_samp = int(OVERLAP_SEC * sr)
        step_samp    = chunk_samp - overlap_samp
        total_samp   = len(waveform)
        starts       = list(range(0, total_samp, step_samp))
        n_chunks     = len(starts)
        vocals_out   = np.zeros(total_samp, dtype=np.float32)

        print(f"  HS-TasNet: {total_samp/sr:.1f}s | {n_chunks} × {CHUNK_SEC}s chunks")
        update_status("HS-TasNet: separating...")

        for idx, start in enumerate(starts):
            end   = min(start + chunk_samp, total_samp)
            chunk = waveform[start:end]

            # [1, 2, T] — fake stereo (model expects 2-channel input)
            stereo = np.stack([chunk, chunk], axis=0)
            tensor = torch.from_numpy(stereo).float().unsqueeze(0)

            with torch.no_grad():
                separated, _ = model(tensor)
                chunk_vocals = separated[0, 0].mean(0).cpu().numpy().astype(np.float32)

            target_len = end - start
            if len(chunk_vocals) > target_len:
                chunk_vocals = chunk_vocals[:target_len]
            elif len(chunk_vocals) < target_len:
                chunk_vocals = np.pad(chunk_vocals, (0, target_len - len(chunk_vocals)))

            if idx == 0:
                vocals_out[start:end] = chunk_vocals
            else:
                fade_len = min(overlap_samp, len(chunk_vocals), end - start)
                fade_in  = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                fade_out = 1.0 - fade_in
                vocals_out[start:start+fade_len] = (
                    vocals_out[start:start+fade_len] * fade_out +
                    chunk_vocals[:fade_len] * fade_in
                )
                vocals_out[start+fade_len:end] = chunk_vocals[fade_len:end-start]

            del tensor, chunk_vocals
            update_status(f"HS-TasNet: chunk {idx+1}/{n_chunks}")
            root.update_idletasks()

        print(f"  HS-TasNet complete: {total_samp} samples")
        return vocals_out

    except Exception as e:
        print(f"HS-TasNet error: {e}")
        import traceback; traceback.print_exc()
        return None

# ============================================================================
# SEPARATION — RT-STT  (placeholder — no public implementation yet)
# ============================================================================

def separate_with_rt_stt(waveform, sr=44100):
    """RT-STT: Real-Time Single-Path TFC-TDF UNet.

    Paper: arxiv 2511.13146 (Nov 2025).
    Status: No public GitHub or pretrained weights as of June 2026.
    This is a placeholder — check the paper for updates.

    When released:
      - Expected install: pip install rt-stt (or similar)
      - Architecture: single-path TFC-TDF UNet + channel expansion fusion
      - Uses INT8 quantization by default for CPU speed
      - Should outperform HS-TasNet at similar latency
    """
    print("=" * 55)
    print("RT-STT: NOT YET AVAILABLE")
    print("  Paper: arxiv.org/abs/2511.13146  (Nov 2025)")
    print("  Status: No public implementation as of June 2026.")
    print("  Action: Check github.com for 'RT-STT' or 'rt_stt_music'")
    print("          or search arxiv 2511.13146 authors for code release.")
    print("=" * 55)
    update_status("RT-STT: not yet released — see log")
    return None

# ============================================================================
# FILE HANDLING (unchanged from v0.4.2)
# ============================================================================

def check_streams(file_path):
    try:
        update_status(f"Analyzing: {os.path.basename(file_path)}")
        probe_data = probe(file_path)
        streams    = probe_data["streams"]
        duration   = float(probe_data.get("format", {}).get("duration", 0) or 0)
        video = False; sub_count = 0; audio_streams = []; audio_idx = 0
        print(f"Found {len(streams)} streams | Duration: {duration:.1f}s")
        for stream in streams:
            ct = stream.get("codec_type", "unknown")
            if ct == "video": video = True
            elif ct == "audio":
                audio_streams.append({
                    "index": audio_idx,
                    "codec": stream.get("codec_name", "?"),
                    "lang": stream.get("tags", {}).get("language", "und"),
                    "channels": stream.get("channels", 1),
                    "title": stream.get("tags", {}).get("title", ""),
                })
                audio_idx += 1
            elif ct == "subtitle": sub_count += 1
        for t in audio_streams:
            lbl = f"  Track {t['index']}: {t['codec']} | {t['channels']}ch | lang={t['lang']}"
            if t['title']: lbl += f" | {t['title']}"
            print(lbl)
        return video, audio_streams, sub_count, duration
    except Exception as e:
        print(f"Stream check failed: {e}")
        return False, [{"index": 0, "codec": "?", "lang": "und", "channels": 1, "title": ""}], 0, 0


def ask_audio_track(audio_streams):
    if len(audio_streams) <= 1: return [0]
    import tkinter as tk
    BG='#0d0d14'; CARD='#14141f'; BORDER='#22223a'; ACCENT='#4f7ef0'
    TEXT='#cdd2de'; MUTED='#505a72'; GREEN='#27ae60'
    result = [None]
    dialog = tk.Toplevel(root)
    dialog.title("Select Audio Track"); dialog.configure(bg=BG)
    dialog.resizable(False, False); dialog.grab_set(); dialog.transient(root)
    tk.Label(dialog, text="Multiple audio tracks detected",
             font=("Segoe UI", 10, "bold"), bg=BG, fg=TEXT).pack(padx=20, pady=(16,4))
    tk.Label(dialog, text="TRACK TO PROCESS", font=("Segoe UI",7,"bold"),
             bg=BG, fg=MUTED).pack(anchor="w", padx=20, pady=(6,2))
    track_var = tk.IntVar(value=0)
    for t in audio_streams:
        lbl = f"Track {t['index']}:  {t['codec'].upper()}  |  {t['channels']}ch  |  {t['lang'].upper()}"
        if t['title']: lbl += f"  —  {t['title']}"
        tk.Radiobutton(dialog, text=lbl, variable=track_var, value=t['index'],
                       font=("Segoe UI",9), bg=CARD, fg=TEXT, selectcolor=CARD,
                       activebackground=CARD, activeforeground=ACCENT,
                       highlightthickness=0).pack(anchor="w", padx=20, pady=1)
    btn_row = tk.Frame(dialog, bg=BG)
    btn_row.pack(fill="x", padx=20, pady=(12,16))
    def confirm_one(): result[0]=[track_var.get()]; dialog.destroy()
    def confirm_all(): result[0]=[t['index'] for t in audio_streams]; dialog.destroy()
    tk.Button(btn_row, text="Process Selected", command=confirm_one,
              bg=ACCENT, fg="white", font=("Segoe UI",9,"bold"),
              relief="flat", pady=7, cursor="hand2").pack(side="left", expand=True, fill="x", padx=(0,6))
    tk.Button(btn_row, text=f"Process All ({len(audio_streams)})", command=confirm_all,
              bg=GREEN, fg="white", font=("Segoe UI",9,"bold"),
              relief="flat", pady=7, cursor="hand2").pack(side="right", expand=True, fill="x")
    root.wait_window(dialog)
    return result[0] if result[0] is not None else [0]


def get_output_params(extension):
    ext = extension.lower()
    audio_codecs = {
        'mp3': {'acodec':'mp3','ab':'192k'}, 'm4a':{'acodec':'aac','ab':'192k'},
        'wav': {'acodec':'pcm_s16le'}, 'flac':{'acodec':'flac'},
        'ogg': {'acodec':'libvorbis','ab':'192k'},
    }
    video_codecs = {
        'mp4': {'vcodec':'copy','acodec':'aac','ab':'192k'},
        'mkv': {'vcodec':'copy','acodec':'aac','ab':'192k'},
        'avi': {'vcodec':'copy','acodec':'mp3','ab':'192k'},
    }
    if ext in audio_codecs:   return audio_codecs[ext], False
    elif ext in video_codecs: return video_codecs[ext], True
    else:                     return {'acodec':'aac','ab':'192k'}, False

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def _run_separation(waveform, sr):
    """Route to the correct separator and optionally apply TTA."""
    global separator_choice, model_choice, tta_enabled

    def _core_sep(w):
        if separator_choice == 'audio_separator' and AUDIO_SEPARATOR_AVAILABLE:
            return separate_with_audio_separator(w, sr, model_choice)
        elif separator_choice == 'demucs' and DEMUCS_AVAILABLE:
            return separate_with_demucs(w, sr)
        elif separator_choice == 'spleeter' and SPLEETER_AVAILABLE:
            return separate_with_spleeter(w, sr)
        elif separator_choice == 'hs_tasnet':
            return separate_with_hs_tasnet(w, sr)
        elif separator_choice == 'rt_stt':
            return separate_with_rt_stt(w, sr)
        elif DEMUCS_AVAILABLE:
            print("WARNING: Falling back to DEMUCS")
            return separate_with_demucs(w, sr)
        elif SPLEETER_AVAILABLE:
            print("WARNING: Falling back to Spleeter")
            return separate_with_spleeter(w, sr)
        else:
            print("No separator available"); return None

    if tta_enabled:
        print("TTA enabled: running polarity-flip augmentation (2× slower)")
        return apply_tta_polarity(_core_sep, waveform)
    else:
        return _core_sep(waveform)


def _process_single_track(file_path, input_, track_idx, output_path, video,
                          audio_streams, sub_count=0, duration=0):
    global process1, process2, sample_rate, processing_mode

    print(f"  Extracting track a:{track_idx}...")
    update_status(f"Extracting audio (track {track_idx})...")
    process1 = (input_[f'a:{track_idx}']
                .output('pipe:1', format='f32le', ac=1, ar=sample_rate)
                .overwrite_output()
                .global_args('-loglevel', 'error')
                .run_async(pipe_stdin=False, pipe_stdout=True, pipe_stderr=True))

    expected_bytes = int(duration * sample_rate * 4) if duration > 0 else 0
    chunks = []; bytes_read = 0; chunk_size = 8*1024*1024; iteration = 0
    while True:
        chunk = process1.stdout.read(chunk_size)
        if not chunk: break
        chunks.append(chunk); bytes_read += len(chunk); iteration += 1
        if iteration % 4 == 0 and expected_bytes > 0 and progress:
            progress['value'] = min((bytes_read/expected_bytes)*100, 99)
            root.update_idletasks()
    process1.wait(); process1.kill()
    out = b''.join(chunks); del chunks
    if progress: progress['value'] = 0

    waveform = frombuffer(out, dtype='<f4').flatten()
    del out; gc.collect()

    # ── DSP Stage 0: existing noise gate + high-pass pre-filter ─────────────
    waveform = audio_prefilter(waveform, sample_rate)

    # ── DSP Stage 1: HPSS harmonic pre-filter (SpeechBrain pattern) ─────────
    if hpss_prefilter_enabled:
        waveform = apply_hpss_prefilter(waveform, sample_rate)

    # ── AI Stage: primary source separator ───────────────────────────────────
    vocals = _run_separation(waveform, sample_rate)
    del waveform; gc.collect()

    if vocals is None:
        print(f"  Separation failed for track {track_idx}"); return False

    vocals = np.ascontiguousarray(vocals, dtype=np.float32); gc.collect()

    # ── DSP Stage 2: MetricGAN+ post-filter (SpeechBrain pattern) ───────────
    if speech_enhance_enabled:
        vocals = apply_metricgan_enhance(vocals, sample_rate)
        vocals = np.ascontiguousarray(vocals, dtype=np.float32); gc.collect()

    # ── DSP Stage 3: existing de-esser + formant enhancement ─────────────────
    vocals = advanced_postprocess_chunked(vocals, sample_rate, processing_mode); gc.collect()

    print(f"  Exporting: {os.path.basename(output_path)}")
    update_status(f"Exporting: {os.path.basename(output_path)}")

    ext = os.path.splitext(output_path)[1][1:]
    _, is_video = get_output_params(ext)
    vocals_pipe = ffmpeg_input('pipe:0', format='f32le', ar=sample_rate, ac=1)

    if is_video and video:
        other_indices = [t['index'] for t in audio_streams if t['index'] != track_idx]
        out_streams = [vocals_pipe.audio, input_.video]
        out_kwargs  = {'vcodec':'copy', 'c:a:0':'aac', 'b:a:0':'192k'}
        if other_indices:
            out_streams += [input_[f'a:{i}'] for i in other_indices]
            for n in range(len(other_indices)): out_kwargs[f'c:a:{n+1}'] = 'copy'
        if sub_count > 0:
            out_streams += [input_[f's:{i}'] for i in range(sub_count)]
            for n in range(sub_count): out_kwargs[f'c:s:{n}'] = 'copy'
        process2 = (_ffmpeg.output(*out_streams, output_path, **out_kwargs)
                    .overwrite_output().global_args('-loglevel','error')
                    .run_async(pipe_stdin=True, pipe_stderr=True))
    else:
        params, _ = get_output_params(ext)
        process2 = (vocals_pipe.output(output_path, **params)
                    .overwrite_output().global_args('-loglevel','error')
                    .run_async(pipe_stdin=True, pipe_stderr=True))

    _chunk = 8*1024*1024
    _view  = memoryview(vocals.astype('<f4', copy=False))
    del vocals
    for _i in range(0, len(_view), _chunk):
        process2.stdin.write(_view[_i:_i+_chunk])
    process2.stdin.close(); process2.wait(); process2.kill()
    print(f"  SUCCESS: {output_path}")
    return True


def demusic_pro(file_path):
    global progress, root
    try:
        file_path = file_path.replace("\\\\", "/")
        name      = os.path.splitext(os.path.basename(file_path))[0]
        extension = os.path.splitext(file_path)[1][1:]
        path      = os.path.dirname(file_path)
        print(f"Processing: {name}.{extension}")
        print(f"Separator: {separator_choice.upper()} | Model: {model_choice} | Mode: {processing_mode}")
        if tta_enabled: print("TTA: enabled (2× slower, +quality)")
        update_status(f"Processing: {name}")

        input_ = ffmpeg_input(file_path)
        video, audio_streams, sub_count, duration = check_streams(file_path)
        if not audio_streams:
            print("No audio track found"); return False

        track_indices = ask_audio_track(audio_streams)
        multi = len(track_indices) > 1
        all_ok = True

        for track_idx in track_indices:
            if multi:
                output_path = f"{path}/{name}-vocals-t{track_idx}.{extension}"
                print(f"\n--- Track {track_idx} of {len(track_indices)} ---")
            else:
                output_path = f"{path}/{name}-vocals.{extension}"
            ok = _process_single_track(file_path, input_, track_idx, output_path,
                                       video, audio_streams, sub_count, duration)
            if not ok: all_ok = False

        progress['value'] = 0
        update_status(f"Done: {name} ({len(track_indices)} track(s))" if all_ok
                      else f"Partial failure: {name}")
        return all_ok

    except Exception as e:
        print(f"ERROR: {e}"); update_status(f"Error: {str(e)[:50]}")
        import traceback; traceback.print_exc()
        return False

# ============================================================================
# GUI
# ============================================================================

# Map display label → (separator_choice, model_choice)
_SEP_MAP = {}

def _build_sep_options():
    """Build the ordered list of separator display strings and populate _SEP_MAP."""
    opts = []
    if AUDIO_SEPARATOR_AVAILABLE:
        # ── Fast ONNX (CPU-optimised) ─────────────────────────────────────
        opts.append("── FAST · ONNX (CPU-optimised) ──")
        _SEP_MAP["── FAST · ONNX (CPU-optimised) ──"] = None  # section header

        e = "MDX-Net Voc_FT   [ONNX · 66 MB · ~200 ms CPU]"
        _SEP_MAP[e] = ("audio_separator", "UVR-MDX-NET-Voc_FT.onnx")
        opts.append(e)

        e = "MDX-Net Kim_v2   [ONNX · 130 MB · ~300 ms CPU]"
        _SEP_MAP[e] = ("audio_separator", "Kim_Vocal_2.onnx")
        opts.append(e)

        # ── MDXC (torch, higher quality) ─────────────────────────────────
        opts.append("── MDXC · torch (higher quality) ──")
        _SEP_MAP["── MDXC · torch (higher quality) ──"] = None

        e = "MDX23C InstVoc HQ    [MDXC · torch · 448 MB]"
        _SEP_MAP[e] = ("audio_separator", "MDX23C-8KFFT-InstVoc_HQ.ckpt")
        opts.append(e)

        e = "MDX23C InstVoc HQ 2  [MDXC · torch · 448 MB]"
        _SEP_MAP[e] = ("audio_separator", "MDX23C-8KFFT-InstVoc_HQ_2.ckpt")
        opts.append(e)

    # ── Real-time architecture stubs ─────────────────────────────────────
    opts.append("── RT ARCHITECTURES (experimental) ──")
    _SEP_MAP["── RT ARCHITECTURES (experimental) ──"] = None

    e = "HS-TasNet   [23 ms target · NEEDS WEIGHTS · see log]"
    _SEP_MAP[e] = ("hs_tasnet", "hs_tasnet")
    opts.append(e)

    e = "RT-STT      [COMING SOON · Nov 2025 paper · no weights]"
    _SEP_MAP[e] = ("rt_stt", "rt_stt")
    opts.append(e)

    if DEMUCS_AVAILABLE:
        # ── Demucs ───────────────────────────────────────────────────────
        opts.append("── DEMUCS · torch (high quality) ──")
        _SEP_MAP["── DEMUCS · torch (high quality) ──"] = None

        for label, mname in [
            ("DEMUCS htdemucs        [torch · 600 MB · Fast]",          "htdemucs"),
            ("DEMUCS htdemucs_ft     [torch · 600 MB · Quality]",       "htdemucs_ft"),
            ("DEMUCS htdemucs_6s     [torch · 600 MB · 6-stem]",        "htdemucs_6s"),
            ("DEMUCS hdemucs_mmi     [torch · 600 MB · v3-retrained]",  "hdemucs_mmi"),
            ("DEMUCS mdx_q           [torch · ~300 MB · quantized]",    "mdx_q"),
            ("DEMUCS mdx_extra       [torch · 600 MB · Best MDX]",      "mdx_extra"),
            ("DEMUCS mdx_extra_q     [torch · ~300 MB · Best+quant]",   "mdx_extra_q"),
        ]:
            _SEP_MAP[label] = ("demucs", mname)
            opts.append(label)

    if SPLEETER_AVAILABLE:
        opts.append("── SPLEETER ──")
        _SEP_MAP["── SPLEETER ──"] = None
        e = "Spleeter  [TF · 4-stems · legacy]"
        _SEP_MAP[e] = ("spleeter", "spleeter")
        opts.append(e)

    return opts


def setFrames(root):
    global b1, b2, inprogressframe, listbox, f1, a1, progress, status_label
    global mode_selector, separator_selector, quantize_var, tta_var
    global hpss_var, enhance_var

    BG='#0d0d14'; CARD='#14141f'; BORDER='#22223a'; ACCENT='#4f7ef0'
    ACCENT2='#3a6bd6'; TEXT='#cdd2de'; MUTED='#505a72'; GREEN='#27ae60'; RED='#c0392b'
    WARN='#c8a02a'

    style = Style(root)
    style.theme_use('clam')
    style.configure('App.TCombobox',
        fieldbackground=CARD, background=CARD, foreground=TEXT,
        selectbackground='#1c2a4a', selectforeground=ACCENT,
        bordercolor=BORDER, arrowcolor=ACCENT, padding=4)
    style.map('App.TCombobox',
        fieldbackground=[('readonly', CARD)], foreground=[('readonly', TEXT)])
    style.configure('App.Horizontal.TProgressbar',
        background=ACCENT, troughcolor='#0a0a12',
        bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT2)

    root.configure(background=BG)

    def divider(parent=root, color=BORDER, h=1, padx=0):
        Frame(parent, bg=color, height=h).pack(fill="x", padx=padx)

    def section_label(text):
        Label(root, text=text, font=("Segoe UI",7,"bold"),
              bg=BG, fg=MUTED, anchor="w").pack(fill="x", padx=16, pady=(9,2))

    def mk_card(parent):
        return Frame(parent, bg=CARD, highlightthickness=1,
                     highlightbackground=BORDER, highlightcolor=ACCENT)

    # ── HEADER ───────────────────────────────────────────────────────────────
    hdr = Frame(root, bg='#0f0f1c')
    hdr.pack(fill="x")
    title_row = Frame(hdr, bg='#0f0f1c')
    title_row.pack(fill="x", padx=16, pady=(10,8))
    Label(title_row, text="Music Assassin Pro",
          font=("Segoe UI",15,"bold"), bg='#0f0f1c', fg=TEXT).pack(side="left")
    Label(title_row, text="v0.4.2-RT",
          font=("Segoe UI",10), bg='#0f0f1c', fg=ACCENT).pack(side="left", padx=(6,0), pady=4)
    Label(title_row,
          text="RT MODEL TESTING  •  MDX23C + HS-TasNet  •  TTA  •  HPSS + MetricGAN+",
          font=("Segoe UI",8), bg='#0f0f1c', fg=WARN).pack(side="right", pady=4)
    Frame(root, bg=ACCENT, height=2).pack(fill="x")

    # ── CPU INFO BAR ─────────────────────────────────────────────────────────
    cpu_bar = Frame(root, bg='#0c0c18')
    cpu_bar.pack(fill="x", padx=16, pady=(6,0))
    cpu_cores = os.cpu_count() or 1
    Label(cpu_bar,
          text=f"  CPU cores: {cpu_cores}  •  "
               f"ONNX fast  •  MDX23C/.ckpt = 2–4× slower (torch)  •  "
               f"Demucs = 3–6× real-time on CPU",
          font=("Segoe UI",7), bg='#0c0c18', fg=WARN, anchor="w"
          ).pack(fill="x", padx=4, pady=3)

    # ── SEPARATOR + MODE (side-by-side) ─────────────────────────────────────
    section_label("CONFIGURATION")
    top_row = Frame(root, bg=BG)
    top_row.pack(fill="x", padx=16)

    sep_card = mk_card(top_row)
    sep_card.pack(side="left", expand=True, fill="both", padx=(0,6))
    Label(sep_card, text="SEPARATOR / MODEL", font=("Segoe UI",7,"bold"),
          bg=CARD, fg=MUTED).pack(anchor="w", padx=10, pady=(7,2))

    sep_options = _build_sep_options()
    # Filter out section headers for default selection
    first_real = next((o for o in sep_options if _SEP_MAP.get(o) is not None), None)

    separator_selector = Combobox(sep_card, values=sep_options, state="readonly",
                                  font=("Segoe UI",9), style='App.TCombobox')
    separator_selector.set(first_real or sep_options[0])
    separator_selector.pack(fill="x", padx=10, pady=(0,2))
    separator_selector.bind("<<ComboboxSelected>>", lambda e: set_separator())

    # Option checkboxes
    opt_row = Frame(sep_card, bg=CARD)
    opt_row.pack(fill="x", padx=6, pady=(0,7))

    quantize_var = BooleanVar(value=True)
    Checkbutton(opt_row, text="INT8 quantize (2–3× faster, htdemucs)",
                variable=quantize_var, command=set_quantize,
                bg=CARD, fg=TEXT, selectcolor=CARD, activebackground=CARD,
                activeforeground=ACCENT, highlightthickness=0,
                font=("Segoe UI",7)).pack(side="left", anchor="w", padx=2)

    tta_var = BooleanVar(value=False)
    Checkbutton(opt_row, text="TTA (2× slower, +quality)",
                variable=tta_var, command=set_tta,
                bg=CARD, fg=WARN, selectcolor=CARD, activebackground=CARD,
                activeforeground=WARN, highlightthickness=0,
                font=("Segoe UI",7)).pack(side="left", anchor="w", padx=8)

    # ── SpeechBrain pipeline row ──────────────────────────────────────────────
    sb_row = Frame(sep_card, bg=CARD)
    sb_row.pack(fill="x", padx=6, pady=(0,8))

    TEAL = '#2ab5a0'
    TEAL_DARK = '#1a7a6e'

    hpss_var = BooleanVar(value=False)
    hpss_cb = Checkbutton(
        sb_row,
        text="HPSS pre-filter  [DSP stage 1 · no dep]" if LIBROSA_AVAILABLE
             else "HPSS pre-filter  [needs: pip install librosa]",
        variable=hpss_var, command=set_hpss,
        bg=CARD, fg=TEAL if LIBROSA_AVAILABLE else MUTED,
        selectcolor=CARD, activebackground=CARD, activeforeground=TEAL,
        highlightthickness=0, font=("Segoe UI",7),
        state="normal" if LIBROSA_AVAILABLE else "disabled",
    )
    hpss_cb.pack(side="left", anchor="w", padx=2)

    enhance_var = BooleanVar(value=False)
    enhance_cb = Checkbutton(
        sb_row,
        text="MetricGAN+ post  [DSP stage 2 · ~50 MB]" if SPEECHBRAIN_AVAILABLE
             else "MetricGAN+ post  [needs: pip install speechbrain]",
        variable=enhance_var, command=set_speech_enhance,
        bg=CARD, fg=TEAL if SPEECHBRAIN_AVAILABLE else MUTED,
        selectcolor=CARD, activebackground=CARD, activeforeground=TEAL,
        highlightthickness=0, font=("Segoe UI",7),
        state="normal" if SPEECHBRAIN_AVAILABLE else "disabled",
    )
    enhance_cb.pack(side="left", anchor="w", padx=10)

    mode_card = mk_card(top_row)
    mode_card.pack(side="right", expand=True, fill="both")
    Label(mode_card, text="PROCESSING MODE", font=("Segoe UI",7,"bold"),
          bg=CARD, fg=MUTED).pack(anchor="w", padx=10, pady=(7,2))
    mode_selector = Combobox(mode_card, values=["Normal","Balanced","Aggressive"],
                             state="readonly", font=("Segoe UI",9), style='App.TCombobox')
    mode_selector.set("Balanced")
    mode_selector.pack(fill="x", padx=10, pady=(0,3))
    mode_selector.bind("<<ComboboxSelected>>", lambda e: set_mode())
    Label(mode_card, text="60–70%  •  75–85%  •  90–95% removal",
          font=("Segoe UI",7), bg=CARD, fg=MUTED).pack(anchor="w", padx=10, pady=(0,7))

    # ── STATUS ───────────────────────────────────────────────────────────────
    status_bar = Frame(root, bg='#0e1a2e', highlightthickness=1, highlightbackground=BORDER)
    status_bar.pack(fill="x", padx=16, pady=(8,0))
    status_label = Label(status_bar, text="  Ready — add files to begin",
                         font=("Segoe UI",9), bg='#0e1a2e', fg=GREEN, anchor="w")
    status_label.pack(fill="x", padx=4, pady=5)

    # ── FILE BUTTONS ─────────────────────────────────────────────────────────
    btn_row = Frame(root, bg=BG)
    btn_row.pack(fill="x", padx=16, pady=(8,0))
    b1 = Button(btn_row, text="Browse Files", bg=ACCENT2, fg="white",
                font=("Segoe UI",9,"bold"), pady=7, relief="flat", cursor="hand2",
                activebackground='#2f59c0', activeforeground='white', bd=0)
    b1.pack(side="left", expand=True, fill="x", padx=(0,6))
    b2 = Button(btn_row, text="Remove Selected", bg='#5c1a1a', fg='#e8a0a0',
                font=("Segoe UI",9,"bold"), pady=7, relief="flat", cursor="hand2",
                activebackground='#7a1f1f', activeforeground='white', bd=0)
    b2.pack(side="right", expand=True, fill="x")

    # ── QUEUE ────────────────────────────────────────────────────────────────
    section_label("PROCESSING QUEUE")
    listboxframe = mk_card(root)
    listboxframe.pack(fill="both", expand=True, padx=16)
    scrollbar_y = Scrollbar(listboxframe, orient="vertical",
                            bg=BORDER, troughcolor=CARD, activebackground=ACCENT, width=10)
    listbox = Listbox(listboxframe, yscrollcommand=scrollbar_y.set, selectmode="multiple",
                      height=4, font=("Segoe UI",9), bg=CARD, fg=TEXT,
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
    action_row.pack(fill="x", padx=16, pady=(8,0))
    f1 = Button(action_row, text="Start Processing", bg='#145233', fg='#7de8b8',
                font=("Segoe UI",10,"bold"), pady=9, relief="flat", cursor="hand2",
                activebackground='#0f3d26', activeforeground='white', bd=0)
    f1.pack(side="left", expand=True, fill="x", padx=(0,6))
    a1 = Button(action_row, text="Clear Queue", bg='#1e1410', fg='#c8a96e',
                font=("Segoe UI",10,"bold"), pady=9, relief="flat", cursor="hand2",
                activebackground='#2a1c0e', activeforeground='white', bd=0)
    a1.pack(side="right", expand=True, fill="x")

    # ── LOG ──────────────────────────────────────────────────────────────────
    section_label("PROCESSING LOG")
    log_widget = ScrolledText(root, height=5, font=("Consolas",8),
                              bg='#07070d', fg='#3eb87a', relief="flat", bd=0,
                              insertbackground=ACCENT, selectbackground='#1c2a4a',
                              highlightthickness=1, highlightbackground=BORDER)
    log_widget.pack(fill="both", expand=True, padx=16, pady=(0,12))
    sys.stdout = PrintLogger(log_widget)

# ============================================================================
# CALLBACKS
# ============================================================================

def set_separator():
    global separator_choice, model_choice
    sel = separator_selector.get()
    mapping = _SEP_MAP.get(sel)
    if mapping is None:
        # Section header selected — revert to first real option
        first_real = next((o for o in _SEP_MAP if _SEP_MAP[o] is not None), None)
        if first_real:
            separator_selector.set(first_real)
            mapping = _SEP_MAP[first_real]
        else:
            return

    separator_choice, model_choice = mapping

    # Warn about heavy models on CPU
    if model_choice and model_choice.endswith('.ckpt'):
        print(f"Separator: MDX23C  |  model: {model_choice}")
        print("  NOTE: MDX23C (.ckpt) requires torch — ~2–4× slower than ONNX on CPU")
        print("  First run auto-downloads from Hugging Face (~448 MB)")
        update_status(f"MDX23C: {model_choice}  [torch — see CPU note in log]")
    elif separator_choice == 'hs_tasnet':
        print("HS-TasNet selected — will check for weights at run time")
        print(f"  Look in: {_MODELS_DIR}/  (any *hs_tasnet*.pt file)")
        update_status("HS-TasNet: will check for weights at run time")
    elif separator_choice == 'rt_stt':
        print("RT-STT selected — no implementation available yet (see log at run time)")
        update_status("RT-STT: coming soon — no weights yet")
    elif separator_choice == 'demucs':
        print(f"Separator: DEMUCS  |  model: {model_choice}")
        update_status(f"DEMUCS: {model_choice}")
    else:
        print(f"Separator: MDX-Net  |  model: {model_choice}")
        update_status(f"MDX-Net: {model_choice}")


def set_quantize():
    global quantize_enabled
    quantize_enabled = bool(quantize_var.get())
    _MODEL_CACHE.clear()
    print(f"INT8 quantize: {quantize_enabled}  (model cache cleared)")
    update_status(f"INT8 quantize: {quantize_enabled}")


def set_tta():
    global tta_enabled
    tta_enabled = bool(tta_var.get())
    status = "ON (2× slower, +0.1–0.3 dB quality)" if tta_enabled else "OFF"
    print(f"TTA: {status}")
    update_status(f"TTA: {status}")


def set_hpss():
    global hpss_prefilter_enabled
    hpss_prefilter_enabled = bool(hpss_var.get())
    if hpss_prefilter_enabled:
        print("HPSS pre-filter: ON")
        print("  Stage 1 DSP: librosa HPSS will extract harmonic component before AI separator")
        update_status("HPSS pre-filter: ON (harmonic-only input to separator)")
    else:
        print("HPSS pre-filter: OFF")
        update_status("HPSS pre-filter: OFF")


def set_speech_enhance():
    global speech_enhance_enabled
    speech_enhance_enabled = bool(enhance_var.get())
    if speech_enhance_enabled:
        if not SPEECHBRAIN_AVAILABLE:
            print("MetricGAN+: NOT available — run: pip install speechbrain")
            update_status("MetricGAN+: needs pip install speechbrain")
            enhance_var.set(False); speech_enhance_enabled = False
            return
        print("MetricGAN+ post-filter: ON")
        print("  Stage 2 DSP: SpeechBrain MetricGAN+ will enhance vocals after AI separator")
        print("  First run: downloads speechbrain/metricgan-plus-voicebank (~50 MB)")
        update_status("MetricGAN+ post-filter: ON (downloads ~50 MB on first run)")
    else:
        print("MetricGAN+ post-filter: OFF")
        update_status("MetricGAN+ post-filter: OFF")


def set_mode():
    global processing_mode
    processing_mode = mode_selector.get().lower()
    update_status(f"Mode: {processing_mode}")
    print(f"Processing mode: {processing_mode}")


def get_file_path():
    global files, listbox, do, root
    if do == 1: do = 0
    selected = []
    try:
        import subprocess
        result = subprocess.run(
            ["zenity","--file-selection","--multiple",
             "--title=Select Audio/Video Files",
             "--file-filter=Media Files | *.mp4 *.mp3 *.wav *.avi *.mkv *.flac",
             "--file-filter=All Files | *"],
            capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            selected = result.stdout.strip().split("|")
    except FileNotFoundError:
        browse = filedialog.askopenfilenames(
            title="Select Audio/Video Files", parent=root,
            filetypes=[("Media",("*.mp4","*.mp3","*.wav","*.avi","*.mkv","*.flac")),("All","*.*")])
        selected = list(browse)
    for i in selected:
        i = i.strip()
        if i and i not in files:
            files.append(i)
            listbox.insert("end", os.path.basename(i))
    print(f"Queue: {len(files)} files")
    update_status(f"Queue: {len(files)} file(s)")


def remove_item():
    global listbox, files
    selected = list(listbox.curselection())[::-1]
    for idx in selected:
        listbox.delete(idx); files.pop(idx)
    update_status(f"Queue: {len(files)} file(s)")


def demusic_list():
    global do, files
    if do == 1 or not files:
        if not files: update_status("Add files to start")
        return
    do = 1; completed = []; failed = []
    update_status(f"Processing {len(files)} file(s)...")

    def process_next():
        global do
        if do == 0 or not files:
            if not files:
                do = 0
                print(f"\n=== COMPLETE ===\nProcessed: {len(completed)}\nFailed: {len(failed)}")
                update_status(f"Complete — {len(completed)} processed, {len(failed)} failed")
            return
        file_path = files.pop(0)
        if listbox.size() > 0: listbox.delete(0)
        filename = os.path.basename(file_path)
        print(f"\nProcessing: {filename}"); update_status(f"Processing: {filename}")
        success = demusic_pro(file_path)
        if success:
            completed.append(filename)
            Label(inprogressframe, text=f"  ✓  {filename}",
                  font=("Segoe UI",8), bg='#071a0f', fg='#4ade80',
                  anchor="w", pady=2).pack(fill="x", pady=1)
        else:
            failed.append(filename)
            Label(inprogressframe, text=f"  ✗  {filename}",
                  font=("Segoe UI",8), bg='#1a0707', fg='#f87171',
                  anchor="w", pady=2).pack(fill="x", pady=1)
        root.after(1000, process_next)

    root.after(100, process_next)


def on_closing():
    global do, process1, process2
    if do == 0:
        sys.stdout = sys.__stdout__; root.destroy(); return
    if messagebox.askokcancel("Quit", "Processing active. Exit anyway?"):
        sys.stdout = sys.__stdout__
        if process1: process1.kill()
        if process2: process2.kill()
        root.destroy()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    root = Tk()
    root.wm_title("Music Assassin Pro v0.4.2-RT")
    root.geometry("760x700")
    root.configure(background='#0d0d14')
    root.minsize(680, 600)

    try:
        root.update()
    except Exception:
        os.environ['TCL_LIBRARY'] = '/usr/share/tcltk/tcl8.6'
        os.environ['TK_LIBRARY']  = '/usr/share/tcltk/tk8.6'
        root.destroy()
        root = Tk()
        root.wm_title("Music Assassin Pro v0.4.2-RT")
        root.geometry("760x700")
        root.configure(background='#0d0d14')
        root.minsize(680, 600)

    setFrames(root)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    b1.configure(command=get_file_path)
    b2.configure(command=remove_item)
    f1.configure(command=demusic_list)
    a1.configure(command=lambda: [files.clear(), listbox.delete(0,'end'),
                                   update_status("Queue cleared")])
    set_separator()  # initialise globals from first dropdown entry

    print("=" * 60)
    print("Music Assassin Pro v0.4.2-RT  —  Real-Time Model Testing Branch")
    print("=" * 60)
    print(f"audio-separator:  {AUDIO_SEPARATOR_AVAILABLE}")
    print(f"DEMUCS (torch):   {DEMUCS_AVAILABLE}")
    print(f"Spleeter (TF):    {SPLEETER_AVAILABLE}")
    print(f"HS-TasNet pkg:    {HS_TASNET_AVAILABLE}")
    print(f"SpeechBrain:      {SPEECHBRAIN_AVAILABLE}  ← MetricGAN+ post-filter")
    print(f"librosa:          {LIBROSA_AVAILABLE}  ← HPSS pre-filter")
    print(f"CPU cores:        {os.cpu_count()}")
    print("")
    print("NEW MODELS (this branch):")
    print("  MDX23C-8KFFT-InstVoc_HQ.ckpt   (torch, 448 MB, auto-download)")
    print("  MDX23C-8KFFT-InstVoc_HQ_2.ckpt (torch, 448 MB, auto-download)")
    print("  DEMUCS htdemucs_6s / hdemucs_mmi / mdx_q / mdx_extra_q")
    print("")
    print("RT STUBS (infrastructure only):")
    print("  HS-TasNet  — pip install HS-TasNet, then provide .pt weights")
    print(f"               Place in: {_MODELS_DIR}/")
    print("  RT-STT     — awaiting public release (arxiv 2511.13146)")
    print("")
    print("TTA: polarity-flip averaging (+quality, 2× slower) — checkbox in UI")
    print("")
    print("SPEECHBRAIN DSP+AI PIPELINE (SpeechBrain integration):")
    print("  HPSS pre-filter  — librosa HPSS harmonic extraction before AI")
    print(f"    Status: {'ready' if LIBROSA_AVAILABLE else 'NEEDS pip install librosa'}")
    print("  MetricGAN+ post  — residual music bleed removal after AI")
    print(f"    Status: {'ready (downloads ~50 MB on first use)' if SPEECHBRAIN_AVAILABLE else 'NEEDS pip install speechbrain'}")
    print("  Full DSP→AI→DSP chain: HPSS → separator → MetricGAN+ → postprocess")
    print("=" * 60)
    update_status("Ready — select model and add files")

    root.mainloop()
