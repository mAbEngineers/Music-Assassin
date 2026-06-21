"""
PyInstaller runtime hook for Music-Assassin v0.5.0
Runs before any app code — sets env vars so bundled assets are found
without any internet access.
"""
import os
import sys

if getattr(sys, 'frozen', False):
    base = sys._MEIPASS

    # ---- FFmpeg -------------------------------------------------------
    os.environ['PATH'] = os.path.join(base, 'bin') + os.pathsep + os.environ.get('PATH', '')

    # ---- DEMUCS models ------------------------------------------------
    os.environ['TORCH_HOME'] = os.path.join(base, 'torch_home')

    # ---- Tcl/Tk (Linux) -----------------------------------------------
    if sys.platform == 'linux':
        os.environ.setdefault('TCL_LIBRARY', os.path.join(base, 'tcl86'))
        os.environ.setdefault('TK_LIBRARY',  os.path.join(base, 'tk86'))
