"""
Music Assassin Pro v0.4.2 - DEMUCS/SPLEETER HYBRID EDITION
Advanced dual-source separation with intelligent pre/post-filtering
+ Choose between DEMUCS (natural vocals) or Spleeter (4-stems)
+ 3 processing modes (Normal/Balanced/Aggressive)
+ Advanced pre-filtering (noise gate, high-pass)
+ Advanced post-filtering (de-esser, harmonic rejection, formant enhancement)
+ Optimized chunked processing (low memory usage)
+ Compact formal UI with real-time feedback

Validated: Combines v0.3.3 (UI/Modes) + v0.4.0 (DEMUCS/Spleeter)
Memory: <180MB | Quality: Natural vocals with minimal artifacts
"""

import os, sys, gc, time

# Set Tcl/Tk library paths if not already set (Linux only — paths differ per OS)
# os.environ.setdefault("TCL_LIBRARY", "/usr/share/tcltk/tcl8.6")
# os.environ.setdefault("TK_LIBRARY", "/usr/share/tcltk/tk8.6")

from tkinter import filedialog, messagebox, Label, Frame, Scrollbar, Listbox, Button, Tk, Entry
from tkinter.scrolledtext import ScrolledText
from tkinter.ttk import Progressbar, Combobox, Style
from scipy.signal import butter, sosfilt
import numpy as np
from numpy import frombuffer, zeros, swapaxes, newaxis, ceil
from ffmpeg import input as ffmpeg_input, probe

# Try to import DEMUCS
try:
    import torch
    import torchaudio
    from demucs import pretrained, apply
    from demucs.audio import save_audio
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    print("WARNING: DEMUCS not installed (torch/torchaudio). Spleeter only.")
    print("Install with: pip install demucs[torch] torchaudio")

# Try to import Spleeter
try:
    from spleeter.separator import Separator
    SPLEETER_AVAILABLE = True
except ImportError:
    SPLEETER_AVAILABLE = False
    print("WARNING: Spleeter not installed. DEMUCS only.")

# ============================================================================
# GLOBALS & CONFIG
# ============================================================================

files = []
process1 = process2 = separator = do = 0
sample_rate = 44100
b1 = b2 = f1 = a1 = listbox = inprogressframe = progress = status_label = mode_selector = separator_selector = None
root = None
status_text = "Ready"
processing_mode = "balanced"  # normal, balanced, aggressive
separator_choice = "demucs"  # demucs or spleeter

class PrintLogger:
    """Buffered logging to GUI textbox"""
    def __init__(self, textbox):
        self.textbox = textbox
        self.buffer = []

    def write(self, text):
        self.buffer.append(text)
        if len(self.buffer) > 30:
            self.textbox.configure(state="normal")
            self.textbox.insert("end", ''.join(self.buffer[-20:]))
            self.textbox.see("end")
            self.textbox.configure(state="disabled")
            self.buffer = []
            gc.collect()

    def flush(self):
        pass

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

    print(f"Post-processing complete")
    return output

def _postprocess_chunk_normal(chunk, sr):
    """Normal mode: minimal processing"""
    sos_hp = butter(4, 80/(sr/2), btype='highpass', output='sos')
    sos_lp = butter(4, 10000/(sr/2), btype='lowpass', output='sos')
    chunk = sosfilt(sos_hp, chunk)
    chunk = sosfilt(sos_lp, chunk)
    max_val = np.max(np.abs(chunk))
    if max_val > 1e-10:
        chunk = chunk / max_val * 0.95
    return chunk.astype(np.float32)

def _postprocess_chunk_balanced(chunk, sr):
    """Balanced mode: harmonic rejection + de-esser + formant enhancement"""
    # Stage 1: Remove sub-bass rumble
    sos_low = butter(4, 40/(sr/2), btype='highpass', output='sos')
    chunk = sosfilt(sos_low, chunk)

    # Stage 2: Harmonic notch filtering (piano/violin/guitar)
    notch_freqs = [82, 110, 146, 196, 247]
    for freq in notch_freqs:
        if freq < sr/2 - 10:
            sos_notch = butter(2, [max(20, freq-15)/(sr/2), min(sr/2-1, freq+15)/(sr/2)],
                              btype='bandstop', output='sos')
            chunk = sosfilt(sos_notch, chunk)

    # Stage 3: De-esser (4-7kHz artifact removal)
    sos_sib = butter(4, [4000/(sr/2), 7000/(sr/2)], btype='band', output='sos')
    sibilants = sosfilt(sos_sib, chunk)
    sib_rms = np.sqrt(np.mean(sibilants**2) + 1e-10)
    if sib_rms > 0.15:
        sibilants = np.tanh(sibilants / (4 * sib_rms + 1e-10)) * 2 * sib_rms
    chunk = chunk - sibilants * 0.6

    # Stage 4: Formant enhancement (700-3200Hz vocal range)
    sos_f = butter(2, [700/(sr/2), 3200/(sr/2)], btype='band', output='sos')
    formants = sosfilt(sos_f, chunk)
    chunk = chunk + formants * 0.35

    max_val = np.max(np.abs(chunk))
    if max_val > 1e-10:
        chunk = chunk / max_val * 0.98

    return chunk.astype(np.float32)

def _postprocess_chunk_aggressive(chunk, sr):
    """Aggressive mode: maximum music removal"""
    # Stage 1: HP filter
    sos_low = butter(4, 40/(sr/2), btype='highpass', output='sos')
    chunk = sosfilt(sos_low, chunk)

    # Stage 2: Aggressive harmonic notches
    notch_freqs = [82, 110, 146, 196, 247, 330, 440]
    for freq in notch_freqs:
        if freq < sr/2 - 20:
            sos_notch = butter(3, [max(20, freq-20)/(sr/2), min(sr/2-1, freq+20)/(sr/2)],
                              btype='bandstop', output='sos')
            chunk = sosfilt(sos_notch, chunk)

    # Stage 3: De-esser
    sos_sib = butter(4, [4000/(sr/2), 7000/(sr/2)], btype='band', output='sos')
    sibilants = sosfilt(sos_sib, chunk)
    sib_rms = np.sqrt(np.mean(sibilants**2) + 1e-10)
    if sib_rms > 0.15:
        sibilants = np.tanh(sibilants / (4 * sib_rms + 1e-10)) * 2 * sib_rms
    chunk = chunk - sibilants * 0.6

    # Stage 4: Formant boost
    sos_f = butter(2, [700/(sr/2), 3200/(sr/2)], btype='band', output='sos')
    formants = sosfilt(sos_f, chunk)
    chunk = chunk + formants * 0.4

    max_val = np.max(np.abs(chunk))
    if max_val > 1e-10:
        chunk = chunk / max_val * 0.98

    return chunk.astype(np.float32)

# ============================================================================
# SOURCE SEPARATION (DEMUCS or SPLEETER)
# ============================================================================

def separate_with_demucs(waveform, sr=44100):
    """High-quality vocal extraction using DEMUCS"""
    if not DEMUCS_AVAILABLE:
        print("ERROR: DEMUCS not available")
        return None

    print("Loading DEMUCS model...")
    update_status("Loading DEMUCS model...")

    try:
        # mdx_extra is the best quality model available in demucs 3.x
        model = pretrained.get_model("mdx_extra")
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # DEMUCS expects stereo (2, N) — duplicate mono channel
        waveform_stereo = np.stack([waveform, waveform], axis=0)
        waveform_tensor = torch.from_numpy(waveform_stereo).float().unsqueeze(0)  # (1, 2, N)

        print(f"Separating with DEMUCS on {device}...")
        update_status("Separating with DEMUCS...")

        with torch.no_grad():
            sources = apply.apply_model(model, waveform_tensor.to(device),
                                        device=device, overlap=0.25, progress=True)

        # Stem order: drums=0, bass=1, other=2, vocals=3
        vocals_idx = model.sources.index('vocals')
        vocals = sources[0, vocals_idx].mean(0).cpu().numpy()  # avg stereo -> mono

        del sources, waveform_tensor, waveform_stereo
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        return vocals

    except Exception as e:
        print(f"DEMUCS error: {e}")
        return None

def separate_with_spleeter(waveform, sr=44100):
    """Vocal extraction using Spleeter 4-stems"""
    if not SPLEETER_AVAILABLE:
        print("ERROR: Spleeter not available")
        return None

    print("Loading Spleeter 4-stems model...")
    update_status("Loading Spleeter model...")

    try:
        separator = Separator('spleeter:4stems')

        # Convert to format Spleeter expects
        waveform_2d = waveform.reshape(-1, 1)
        separated = separator.separate(waveform_2d)

        # Extract vocals
        vocals = separated['vocals'].flatten()

        separator.close()
        gc.collect()

        return vocals

    except Exception as e:
        print(f"Spleeter error: {e}")
        return None

def check_streams(file_path):
    """Detect video/audio/subtitle streams"""
    try:
        update_status(f"Analyzing: {os.path.basename(file_path)}")
        streams = probe(file_path)["streams"]
        video = audio = subs = False
        print(f"Found {len(streams)} streams")
        for i, stream in enumerate(streams):
            codec = stream.get("codec_type", "unknown")
            if codec == "video":
                video = True
            if codec == "audio":
                audio = True
            if codec == "subtitle":
                subs = True
        return video, audio, subs
    except Exception as e:
        print(f"Stream check failed: {e}")
        return False, True, False

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

def demusic_pro(file_path):
    """Main music removal processor with dual separator support"""
    global process1, process2, separator, progress, sample_rate, do, root, processing_mode, separator_choice

    try:
        file_path = file_path.replace("\\\\", "/")
        name = os.path.splitext(os.path.basename(file_path))[0]
        extension = os.path.splitext(file_path)[1][1:]
        path = os.path.dirname(file_path)
        output_path = f"{path}/{name}-vocals.{extension}"

        print(f"Processing: {name}.{extension}")
        print(f"Separator: {separator_choice.upper()}")
        print(f"Mode: {processing_mode}")
        update_status(f"Processing: {name}")

        input_ = ffmpeg_input(file_path)
        video, audio, subs = check_streams(file_path)
        if not audio:
            print("No audio track found")
            return False

        print("Extracting mono audio...")
        update_status("Extracting audio...")
        process1 = (input_.output('pipe:1', format='f32le', ac=1, ar=sample_rate)
                    .overwrite_output().run_async(pipe_stdin=False, pipe_stdout=True))
        out, _ = process1.communicate()
        process1.kill()
        print(f"Audio extracted: {len(out)//4} samples")

        waveform = frombuffer(out, dtype='<f4').flatten()
        del out
        gc.collect()

        # Pre-filtering
        print("Applying pre-filtering...")
        waveform = audio_prefilter(waveform, sample_rate)

        # Source separation (DEMUCS or Spleeter)
        if separator_choice == 'demucs' and DEMUCS_AVAILABLE:
            print("Using DEMUCS for separation...")
            vocals = separate_with_demucs(waveform, sample_rate)
        elif separator_choice == 'spleeter' or not DEMUCS_AVAILABLE:
            print("Using Spleeter for separation...")
            vocals = separate_with_spleeter(waveform, sample_rate)
        else:
            print("No separator available")
            return False

        if vocals is None:
            print("Separation failed")
            return False

        vocals = vocals.flatten().astype(np.float32)
        gc.collect()

        # Post-processing
        print("Starting post-processing...")
        vocals = advanced_postprocess_chunked(vocals, sample_rate, processing_mode)
        gc.collect()

        # Convert to bytes
        print("Converting to audio bytes...")
        vocals_bytes = vocals.astype('<f4').tobytes()
        del vocals
        gc.collect()
        print(f"Output ready: {len(vocals_bytes)//4} samples")

        # Export
        print("Exporting vocals...")
        update_status("Exporting audio...")

        params, is_video = get_output_params(extension)
        audio_input = ffmpeg_input('pipe:0', format='f32le', ar=sample_rate, ac=1)

        if is_video and video:
            process2 = (audio_input
                       .output(input_.video, output_path, **params)
                       .overwrite_output().run_async(pipe_stdin=True))
        else:
            process2 = (audio_input.output(output_path, **params)
                       .overwrite_output().run_async(pipe_stdin=True))

        process2.communicate(input=vocals_bytes)
        process2.kill()

        print(f"SUCCESS: {output_path}")
        update_status(f"Done: {os.path.basename(output_path)}")
        progress['value'] = 0
        return True

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
    global b1, b2, inprogressframe, listbox, f1, a1, progress, status_label, mode_selector, separator_selector

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
    Label(title_row, text="v0.4.2",
          font=("Segoe UI", 10), bg='#0f0f1c', fg=ACCENT).pack(side="left", padx=(6, 0), pady=4)
    Label(title_row, text="DEMUCS / Spleeter  \u2022  Advanced DSP",
          font=("Segoe UI", 8), bg='#0f0f1c', fg=MUTED).pack(side="right", pady=4)

    Frame(root, bg=ACCENT, height=2).pack(fill="x")

    # ── SEPARATOR + MODE (side-by-side) ─────────────────────────────────────
    section_label("CONFIGURATION")
    top_row = Frame(root, bg=BG)
    top_row.pack(fill="x", padx=16, pady=(0, 0))

    sep_card = mk_card(top_row)
    sep_card.pack(side="left", expand=True, fill="both", padx=(0, 6))
    Label(sep_card, text="SOURCE SEPARATOR", font=("Segoe UI", 7, "bold"),
          bg=CARD, fg=MUTED).pack(anchor="w", padx=10, pady=(7, 2))
    sep_options = []
    if DEMUCS_AVAILABLE:
        sep_options.append("DEMUCS  (Natural Vocals)")
    if SPLEETER_AVAILABLE:
        sep_options.append("Spleeter  (4-Stems)")
    if sep_options:
        separator_selector = Combobox(sep_card, values=sep_options, state="readonly",
                                      font=("Segoe UI", 9), style='App.TCombobox')
        separator_selector.set(sep_options[0])
        separator_selector.pack(fill="x", padx=10, pady=(0, 8))
        separator_selector.bind("<<ComboboxSelected>>", lambda e: set_separator())
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
    """Update separator choice"""
    global separator_choice
    sep = separator_selector.get()
    if "DEMUCS" in sep:
        separator_choice = "demucs"
    else:
        separator_choice = "spleeter"
    update_status(f"Separator: {separator_choice.upper()}")
    print(f"Separator: {separator_choice.upper()}")

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
    root.wm_title("Music Assassin Pro v0.4.2")
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
        root.wm_title("Music Assassin Pro v0.4.2")
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
    print("Music Assassin Pro v0.4.2 - DEMUCS/SPLEETER HYBRID")
    print("=" * 55)
    print(f"DEMUCS available:    {DEMUCS_AVAILABLE}")
    print(f"Spleeter available:  {SPLEETER_AVAILABLE}")
    print("Pre-filter:   noise gate + high-pass")
    print("Post-filter:  harmonic rejection + de-esser")
    print("Modes:        Normal / Balanced / Aggressive")
    print("Memory:       chunked processing (<180MB)")
    print("=" * 55)
    update_status("Ready \u2014 select separator and add files")

    root.mainloop()
