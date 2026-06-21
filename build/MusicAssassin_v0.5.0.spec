# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec File for Music-Assassin v0.5.0
CPU-Optimised Streaming Edition — DEMUCS only (no Spleeter / TensorFlow)

Target: Linux .deb / Windows .exe
Python: 3.8+
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_all

block_cipher = None

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(SPEC)))

# ============================================================================
# TCL/TK FIX (Linux / uv Python)
# ============================================================================
if sys.platform == 'linux':
    os.environ.setdefault('TCL_LIBRARY', '/usr/share/tcltk/tcl8.6')
    os.environ.setdefault('TK_LIBRARY',  '/usr/share/tcltk/tk8.6')

# ============================================================================
# PLATFORM-SPECIFIC BINARIES
# ============================================================================

def _find_bin(name):
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

# DEMUCS checkpoints — torch.hub looks for them at {TORCH_HOME}/hub/checkpoints/
# The runtime hook sets TORCH_HOME = sys._MEIPASS/torch_home.
# Include ALL checkpoints so the user can switch between htdemucs/mdx_extra
# without internet access.
_torch_cache   = os.path.expanduser('~/.cache/torch/hub/checkpoints')
_demucs_models = []
if os.path.isdir(_torch_cache):
    _demucs_models = [(_torch_cache, 'torch_home/hub/checkpoints')]

# collect_all fixes the circular import that ffmpeg-python triggers inside
# PyInstaller's partial module initialisation.
_ffmpeg_pkg_datas, _ffmpeg_pkg_bins, _ffmpeg_pkg_hidden = collect_all('ffmpeg')

# onnxruntime ships native shared libraries (.so / .dll) that PyInstaller
# won't find via hiddenimports alone — collect_all bundles them properly.
# If onnxruntime is not installed the call returns empty lists gracefully.
try:
    _ort_datas, _ort_bins, _ort_hidden = collect_all('onnxruntime')
except Exception:
    _ort_datas = _ort_bins = _ort_hidden = []

# ============================================================================
# ANALYSIS
# ============================================================================

a = Analysis(
    [os.path.join(_repo_root, 'MusicAssassinGUI_v0.5.0.py')],
    pathex=[_repo_root],
    binaries=_binaries + _ffmpeg_pkg_bins + _ort_bins,
    datas=[
        *collect_data_files('demucs'),
        *collect_data_files('scipy'),
        *_ffmpeg_pkg_datas,
        *_ort_datas,
        *_demucs_models,
        *([('/usr/share/tcltk/tcl8.6', 'tcl86'),
           ('/usr/share/tcltk/tk8.6',  'tk86')]
          if sys.platform == 'linux' else []),
    ],
    hiddenimports=[
        *_ffmpeg_pkg_hidden,
        # PyTorch
        'torch',
        'torch.nn',
        'torch.nn.modules',
        'torch.quantization',
        'torch.utils',
        'torch.utils.data',
        'torchaudio',
        'torchaudio.functional',
        # DEMUCS
        'demucs',
        'demucs.pretrained',
        'demucs.apply',
        'demucs.audio',
        'demucs.states',
        'demucs.spec',
        'demucs.htdemucs',
        'demucs.hdemucs',
        'demucs.model',
        # ONNX Runtime (optional — app works without it via torch fallback)
        'onnxruntime',
        'onnxruntime.capi',
        'onnxruntime.capi.onnxruntime_inference_collection',
        # SciPy / NumPy
        'scipy',
        'scipy.signal',
        'scipy.fftpack',
        'scipy.optimize',
        'scipy.special._ufuncs_cxx',
        'numpy',
        'numpy.random',
        'numpy.fft',
        # Tkinter
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.ttk',
        'tkinter.scrolledtext',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(_repo_root, 'build', 'runtime_hook_v0.5.0.py')],
    excludes=[
        # Removed entirely in v0.5.0 — saves ~1.5 GB in the bundle
        'tensorflow',
        'tensorflow_core',
        'spleeter',
        # Other unused heavyweights
        'matplotlib',
        'pytest',
        'IPython',
        'notebook',
        'pandas',
        'sklearn',
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
    name='MusicAssassin_v0.5.0',
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
