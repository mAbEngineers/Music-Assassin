# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec File for Music-Assassin v0.4.2
DEMUCS/Spleeter Hybrid with Advanced Filtering

Generated for: Current production release
Target: Windows (.exe) / Linux standalone
Python: 3.8+
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

_torch_cache = os.path.expanduser('~/.cache/torch/hub/checkpoints')
_demucs_models = []
if os.path.isdir(_torch_cache):
    _demucs_models = [(_torch_cache, 'torch_home/hub/checkpoints')]

_spleeter_models = []
_spleeter_src = os.path.join(_repo_root, 'pretrained_models')
if os.path.isdir(_spleeter_src):
    _spleeter_models = [(_spleeter_src, 'pretrained_models')]

# collect_all resolves the circular import that ffmpeg-python triggers inside
# PyInstaller's partial module initialisation (hidden imports alone are not
# enough — the package data and the correct import order both matter).
_ffmpeg_pkg_datas, _ffmpeg_pkg_bins, _ffmpeg_pkg_hidden = collect_all('ffmpeg')

# ============================================================================
# ANALYSIS
# ============================================================================

a = Analysis(
    [os.path.join(_repo_root, 'MusicAssassinGUI_v0.4.2.py')],
    pathex=[_repo_root],
    binaries=_binaries + _ffmpeg_pkg_bins,
    datas=[
        *collect_data_files('demucs'),
        *collect_data_files('spleeter'),
        *collect_data_files('scipy'),
        *_ffmpeg_pkg_datas,
        *_demucs_models,
        *_spleeter_models,
        *([('/usr/share/tcltk/tcl8.6', 'tcl86'),
           ('/usr/share/tcltk/tk8.6',  'tk86')]
          if sys.platform == 'linux' else []),
    ],
    hiddenimports=[
        *_ffmpeg_pkg_hidden,
        'torch',
        'torch.nn',
        'torch.utils',
        'torch.utils.data',
        'torchaudio',
        'torchaudio.functional',
        'demucs',
        'demucs.pretrained',
        'demucs.apply',
        'demucs.audio',
        'demucs.states',
        'demucs.spec',
        'spleeter',
        'spleeter.separator',
        'spleeter.model',
        'spleeter.audio',
        'spleeter.audio.adapter',
        'scipy',
        'scipy.signal',
        'scipy.fftpack',
        'scipy.optimize',
        'scipy.special._ufuncs_cxx',
        'numpy',
        'numpy.random',
        'numpy.fft',
        'skimage',
        'skimage.filters',
        'skimage.morphology',
        'librosa',
        'librosa.core',
        'librosa.util',
        'tensorflow',
        'tensorflow.python',
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.ttk',
        'tkinter.scrolledtext',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(_repo_root, 'build', 'runtime_hook_v0.4.2.py')],
    excludes=[
        'matplotlib',
        'pytest',
        'IPython',
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
    name='MusicAssassin_v0.4.2',
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
