"""
PyInstaller runtime hook for Music-Assassin v0.4.5
Runs before any app code — sets env vars so bundled assets are found
without any internet access.
"""
import os
import sys

if getattr(sys, 'frozen', False):
    base = sys._MEIPASS

    # ---- FFmpeg -------------------------------------------------------
    # The ffmpeg (and ffprobe) binaries are bundled in the root of the
    # bundle. Prepend that directory to PATH so ffmpeg-python finds them.
    os.environ['PATH'] = os.path.join(base, 'bin') + os.pathsep + os.environ.get('PATH', '')

    # ---- DEMUCS + Silero VAD models -------------------------------------
    # torch.hub resolves checkpoints under {TORCH_HOME}/hub/checkpoints/ and
    # cloned repos (e.g. snakers4/silero-vad) under {TORCH_HOME}/hub/<repo>/.
    # The spec copies both into that sub-path so torch.hub.load() never
    # needs network access.
    os.environ['TORCH_HOME'] = os.path.join(base, 'torch_home')

    # ---- Demucs (HuggingFace hub) models -------------------------------
    # demucs>=4.1 downloads htdemucs/htdemucs_ft/mdx_extra from the HF hub
    # (adefossez/*) instead of torch.hub. HF_HOME relocates huggingface_hub's
    # entire cache root, so the bundled repo snapshots are found offline.
    os.environ['HF_HOME'] = os.path.join(base, 'hf_home')

    # ---- Fast engine models (fast_separation.py) -----------------------
    # Voc_FT, MRX and CDX23 ship in the bundle so the default separator and
    # "Keep sound effects" work offline; fast_separation checks this
    # read-only directory before its download cache.
    os.environ['MA_BUNDLED_MODEL_DIR'] = os.path.join(base, 'ma_models')

    # ---- Spleeter models ----------------------------------------------
    # Spleeter reads MODEL_PATH to locate pretrained model directories.
    os.environ['MODEL_PATH'] = os.path.join(base, 'pretrained_models')

    # ---- Tcl/Tk (Linux) -----------------------------------------------
    # uv Python 3.8 has _tkinter as a builtin; PyInstaller can't auto-detect
    # the Tcl/Tk data paths, so we bundle them and point to them here.
    if sys.platform == 'linux':
        os.environ.setdefault('TCL_LIBRARY', os.path.join(base, 'tcl86'))
        os.environ.setdefault('TK_LIBRARY',  os.path.join(base, 'tk86'))

    # ---- DeepFilterNet cache --------------------------------------------
    # df.utils.get_cache_dir() always resolves to the OS user-cache dir via
    # appdirs (not overridable by env var), so on first launch seed it from
    # the bundled copy — avoids a network download the first time the
    # optional DeepFilterNet enhancer is selected.
    try:
        from appdirs import user_cache_dir
        _df_cache = user_cache_dir('DeepFilterNet')
    except ImportError:
        _df_cache = os.path.expanduser('~/.cache/DeepFilterNet')

    _bundled_df = os.path.join(base, 'deepfilternet_cache')
    if os.path.isdir(_bundled_df) and not os.path.isdir(_df_cache):
        import shutil
        try:
            shutil.copytree(_bundled_df, _df_cache)
        except OSError:
            pass
