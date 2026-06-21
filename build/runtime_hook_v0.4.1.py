"""
PyInstaller runtime hook for Music-Assassin v0.4.1
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
    os.environ['PATH'] = base + os.pathsep + os.environ.get('PATH', '')

    # ---- DEMUCS models ------------------------------------------------
    # torch.hub resolves checkpoints under {TORCH_HOME}/hub/checkpoints/
    # The spec copies ~/.cache/torch/hub/checkpoints/ to that sub-path.
    os.environ['TORCH_HOME'] = os.path.join(base, 'torch_home')

    # ---- Spleeter models ----------------------------------------------
    # Spleeter reads MODEL_PATH to locate pretrained model directories.
    os.environ['MODEL_PATH'] = os.path.join(base, 'pretrained_models')

    # ---- Tcl/Tk (Linux) -----------------------------------------------
    # uv Python 3.8 has _tkinter as a builtin; PyInstaller can't auto-detect
    # the Tcl/Tk data paths, so we bundle them and point to them here.
    if sys.platform == 'linux':
        os.environ.setdefault('TCL_LIBRARY', os.path.join(base, 'tcl86'))
        os.environ.setdefault('TK_LIBRARY',  os.path.join(base, 'tk86'))
