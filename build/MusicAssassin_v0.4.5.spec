# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec File for Music-Assassin v0.4.5
Fast engine (OpenVINO Voc_FT / MRX / CDX23, "keep sound effects") + Demucs +
MDX-Net separation, with Silero VAD gate and optional DeepFilterNet /
sherpa-onnx enhancer cleanup.

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
# Only the DEFAULT model's checkpoint, mirroring the HuggingFace filter below.
# Bundling this whole directory is what made the 2026-07 artifact 1.5 GB: on
# the build machine it held 1.4 GB of .th weights -- htdemucs_ft's four
# sub-models and mdx_extra's three -- i.e. exactly the models the release
# notes say are NOT bundled, plus any unrelated model another project on the
# same machine happened to cache there (one build host had a stray 1.26 GB
# model.pt). demucs>=4.1 fetches htdemucs from the HuggingFace hub anyway, so
# on a clean machine this directory contributes nothing; the filter exists so
# a machine with a populated torch-hub cache produces the same artifact as a
# clean one.
_HTDEMUCS_CKPT_PREFIX = '955717e8'   # demucs/remote/htdemucs.yaml -> models: ['955717e8']
_torch_ckpt_dir = os.path.join(_torch_hub, 'checkpoints')
if os.path.isdir(_torch_ckpt_dir):
    for _ck in sorted(os.listdir(_torch_ckpt_dir)):
        if _ck.startswith(_HTDEMUCS_CKPT_PREFIX):
            _torch_hub_data.append((os.path.join(_torch_ckpt_dir, _ck),
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
# by runtime_hook_v0.4.5.py on first launch (appdirs path isn't overridable
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

# Fast engine (fast_separation.py): OpenVINO runs Voc_FT and MRX on the CPU.
# PyInstaller has no hook for openvino, so it is collected here. libs/ has to
# stay together -- libopenvino looks for its plugins next to itself. Dropped:
# the GPU/NPU plugins, the GPU kernel-tuning cache.json, and the non-ONNX
# model frontends (~75 MB); the engine only ever loads ONNX models on "CPU".
_OV_UNUSED = ('intel_gpu_plugin', 'intel_npu_plugin', 'tensorflow', 'pytorch',
              'paddle', 'jax', 'gguf')

def _ov_unused(path):
    name = os.path.basename(path)
    return 'openvino' in path and (name == 'cache.json' or any(u in name for u in _OV_UNUSED))

_ov_datas, _ov_bins, _ov_hidden = collect_all('openvino')
_ov_hidden = [m for m in _ov_hidden
              if not any(m.startswith('openvino.frontend.' + u) for u in _OV_UNUSED)]

# Fast-engine models: bundled so the default separator and "Keep sound
# effects" work offline (runtime hook sets MA_BUNDLED_MODEL_DIR). Taken from
# the build machine's download cache; files of the wrong size are skipped so
# a half-finished download is never shipped. Warm the cache first:
#   python -c "import fast_separation as f; [f.model_path(k) for k in f._MODELS]"
sys.path.insert(0, _repo_root)
import fast_separation as _fs
_ma_models = []
for _key, _m in _fs._MODELS.items():
    _p = os.path.join(_fs.model_dir(), _m['file'])
    if os.path.isfile(_p) and os.path.getsize(_p) == _m['size']:
        _ma_models.append((_p, 'ma_models'))
    else:
        print(f"WARNING: fast-engine model {_m['file']} not in {_fs.model_dir()} -- "
              f"the build will download it on first use")

# ============================================================================
# ANALYSIS
# ============================================================================

a = Analysis(
    [os.path.join(_repo_root, 'versions', 'MusicAssassinGUI_v0.4.5.py')],
    pathex=[_repo_root],
    binaries=(
        _binaries
        + _ffmpeg_pkg_bins
        + _ort_bins
        + _sherpa_bins
        + _aseg_bins
        + _df_pkg_bins
        + _libdf_bins
        + [b for b in _ov_bins if not _ov_unused(b[0])]
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
        *[d for d in _ov_datas if not _ov_unused(d[0])],
        *_ma_models,
        *_torch_hub_data,
        *_hf_hub_data,
        *_local_models,
        *_df_cache,
        # Tk window icons (build/draw_icon.py); found via sys._MEIPASS/assets
        *[(os.path.join(_repo_root, 'assets', n), 'assets')
          for n in ('icon.png', 'icon_64.png', 'icon_48.png', 'icon_32.png')],
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
        *_ov_hidden,
        *collect_submodules('audio_separator'),
        'fast_separation',
        'pyloudnorm',
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
    runtime_hooks=[os.path.join(_repo_root, 'build', 'runtime_hook_v0.4.5.py')],
    excludes=[
        'matplotlib',
        'pytest',
        'IPython',
        'tensorflow',
        'keras',
        'spleeter',
        *[f'openvino.frontend.{u}' for u in ('tensorflow', 'pytorch', 'paddle', 'jax')],
        # GPU compiler for torch.compile; the CPU torch wheel doesn't need it and
        # `import torch` doesn't load it, but a stray install in the build venv
        # added ~240 MB to the v0.4.5 bundle.
        'triton',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Dependency analysis can re-add an unused OpenVINO library on its own
# (e.g. as a dependency of a collected extension); drop those here too.
a.binaries = [b for b in a.binaries if not _ov_unused(b[1])]
a.datas = [d for d in a.datas if not _ov_unused(d[1])]

# Same for triton's native libs and torch's test binaries (~270 MB together),
# which hooks can collect even with the package excluded.
def _torch_unused(dest):
    return dest.replace('\\', '/').startswith(('triton/', 'torch/test/'))

a.binaries = [b for b in a.binaries if not _torch_unused(b[0])]
a.datas = [d for d in a.datas if not _torch_unused(d[0])]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MusicAssassin_v0.4.5',
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
    icon=os.path.join(_repo_root, 'build', 'icon.ico'),  # Windows; ignored on Linux
)
