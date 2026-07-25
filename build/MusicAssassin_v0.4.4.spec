# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec File for Music-Assassin v0.4.4
Demucs + MDX-Net separation, with Silero VAD gate and optional
DeepFilterNet / sherpa-onnx enhancer cleanup.

Spleeter/TensorFlow are intentionally NOT bundled: PyInstaller's tensorflow
hook segfaults (SIGSEGV, exit -11) collecting submodules for
'keras.src.backend.torch' when keras's multi-backend detection sees torch
also installed in the same env. Spleeter is an optional, lazily-imported
engine in the app (SPLEETER_AVAILABLE check) — demucs/MDX-Net are the
primary engines and remain fully bundled. This mirrors v0.5.0, which
dropped Spleeter/TF for the same class of packaging pain.

Target: Windows (.exe) / Linux standalone
Python: 3.9-3.12 (3.12 tested)
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

block_cipher = None

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(SPEC)))

# ============================================================================
# TCL/TK FIX (Linux / uv Python)
# ============================================================================
if sys.platform == 'linux':
    os.environ.setdefault('TCL_LIBRARY', '/usr/share/tcltk/tcl8.6')
    os.environ.setdefault('TK_LIBRARY', '/usr/share/tcltk/tk8.6')

# ============================================================================
# PLATFORM-SPECIFIC BINARIES
# ============================================================================

def _find_bin(name):
    """Return (src_path, dest_name) for a binary, or None if not found."""
    import shutil
    path = shutil.which(name)
    if path:
        return (path, 'bin')
    return None

_ffmpeg  = _find_bin('ffmpeg')
_ffprobe = _find_bin('ffprobe')
_binaries = [b for b in [_ffmpeg, _ffprobe] if b is not None]

if sys.platform == 'linux':
    _tcl_libs = [
        ('/usr/lib/x86_64-linux-gnu/libtcl8.6.so.0', '.'),
        ('/usr/lib/x86_64-linux-gnu/libtk8.6.so.0',  '.'),
    ]
    _binaries += [(src, dst) for src, dst in _tcl_libs if os.path.exists(src)]

# ============================================================================
# BUNDLED MODEL DATA
# ============================================================================

# DEMUCS checkpoints + Silero VAD hub repo both live under torch's hub cache
# so both need to land at {bundle}/torch_home/hub/... (see runtime_hook).
_torch_hub = os.path.expanduser('~/.cache/torch/hub')
_torch_hub_data = []
if os.path.isdir(os.path.join(_torch_hub, 'checkpoints')):
    _torch_hub_data.append((os.path.join(_torch_hub, 'checkpoints'),
                             'torch_home/hub/checkpoints'))
if os.path.isdir(os.path.join(_torch_hub, 'snakers4_silero-vad_master')):
    _torch_hub_data.append((os.path.join(_torch_hub, 'snakers4_silero-vad_master'),
                             'torch_home/hub/snakers4_silero-vad_master'))

# demucs>=4.1 fetches htdemucs/htdemucs_ft/mdx_extra from the HuggingFace hub
# (adefossez/*) rather than torch.hub — bundle the default 'htdemucs' repo so
# the app works fully offline out of the box. HF_HOME is set by the runtime
# hook. htdemucs_ft/mdx_extra are NOT bundled (960MB combined) — like the
# MDX-Net/UVR models, they auto-download on first use if the user switches to
# them; this keeps the release artifact under GitHub's 2GB asset limit.
_hf_hub = os.path.expanduser('~/.cache/huggingface/hub')
_hf_hub_data = []
if os.path.isdir(_hf_hub):
    for _entry in os.listdir(_hf_hub):
        if _entry == 'models--adefossez--HTDemucs':
            _hf_hub_data.append((os.path.join(_hf_hub, _entry),
                                  os.path.join('hf_home/hub', _entry)))

# Repo-local ONNX enhancer models (sherpa-onnx, SpeechDenoiser, ...) are NOT
# bundled (355MB, all optional/off-by-default — enhancer_choice defaults to
# 'none') to stay under the release size limit. Each enhancer function
# already prints a one-time download link when its model file is missing.
_local_models = []

# DeepFilterNet's downloaded checkpoint — seeded into the real OS cache dir
# by runtime_hook_v0.4.4.py on first launch (appdirs path isn't overridable
# by env var, unlike torch/spleeter).
_df_cache = []
_df_cache_src = os.path.expanduser('~/.cache/DeepFilterNet')
if os.path.isdir(_df_cache_src):
    _df_cache = [(_df_cache_src, 'deepfilternet_cache')]

# collect_all resolves the circular import that ffmpeg-python triggers inside
# PyInstaller's partial module initialisation (hidden imports alone are not
# enough — the package data and the correct import order both matter). The
# same approach is used below for the heavier ML packages added in v0.4.4.
_ffmpeg_pkg_datas, _ffmpeg_pkg_bins, _ffmpeg_pkg_hidden = collect_all('ffmpeg')
_ort_datas, _ort_bins, _ort_hidden = collect_all('onnxruntime')
_sherpa_datas, _sherpa_bins, _sherpa_hidden = collect_all('sherpa_onnx')
_aseg_datas, _aseg_bins, _aseg_hidden = collect_all('audio_separator')
_df_pkg_datas, _df_pkg_bins, _df_pkg_hidden = collect_all('df')
_libdf_datas, _libdf_bins, _libdf_hidden = collect_all('libdf')

# ============================================================================
# ANALYSIS
# ============================================================================

a = Analysis(
    [os.path.join(_repo_root, 'versions', 'MusicAssassinGUI_v0.4.4.py')],
    pathex=[_repo_root],
    binaries=(
        _binaries
        + _ffmpeg_pkg_bins
        + _ort_bins
        + _sherpa_bins
        + _aseg_bins
        + _df_pkg_bins
        + _libdf_bins
    ),
    datas=[
        *collect_data_files('demucs'),
        *collect_data_files('scipy'),
        *_ffmpeg_pkg_datas,
        *_ort_datas,
        *_sherpa_datas,
        *_aseg_datas,
        *_df_pkg_datas,
        *_libdf_datas,
        *_torch_hub_data,
        *_hf_hub_data,
        *_local_models,
        *_df_cache,
        *([('/usr/share/tcltk/tcl8.6', 'tcl86'),
           ('/usr/share/tcltk/tk8.6',  'tk86')]
          if sys.platform == 'linux' else []),
    ],
    hiddenimports=[
        *_ffmpeg_pkg_hidden,
        *_ort_hidden,
        *_sherpa_hidden,
        *_aseg_hidden,
        *_df_pkg_hidden,
        *_libdf_hidden,
        *collect_submodules('audio_separator'),
        'torch',
        'torch.nn',
        'torch.utils',
        'torch.utils.data',
        'torchaudio',
        'torchaudio.functional',
        'torchvision',
        'demucs',
        'demucs.pretrained',
        'demucs.apply',
        'demucs.audio',
        'demucs.states',
        'demucs.spec',
        'scipy',
        'scipy.signal',
        'scipy.fftpack',
        'scipy.optimize',
        'scipy.special._ufuncs_cxx',
        'scipy.ndimage',
        'numpy',
        'numpy.random',
        'numpy.fft',
        'skimage',
        'skimage.filters',
        'skimage.morphology',
        'librosa',
        'librosa.core',
        'librosa.util',
        'appdirs',
        'loguru',
        'sympy',
        'onnx2torch',
        'rotary_embedding_torch',
        'einops',
        'julius',
        'diffq',
        'ml_collections',
        'resampy',
        'samplerate',
        'pydub',
        'beartype',
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.ttk',
        'tkinter.scrolledtext',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(_repo_root, 'build', 'runtime_hook_v0.4.4.py')],
    excludes=[
        'matplotlib',
        'pytest',
        'IPython',
        'tensorflow',
        'keras',
        'spleeter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MusicAssassin_v0.4.4',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
