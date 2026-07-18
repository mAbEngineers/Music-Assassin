# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec File for Music-Assassin v0.3.3 - Enhanced Edition
Harmonic Rejection + Advanced Filtering + YouTube Support

Generated for: Production release
Target: Windows/Linux/macOS executable
Python: 3.8+
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

a = Analysis(
    ['versions/MusicAssassinGUI_v0.3.3.py'],
    pathex=[],
    binaries=[],
    datas=[
        *collect_data_files('spleeter'),
    ],
    hiddenimports=[
        # Core audio processing
        'numpy',
        'scipy',
        'scipy.signal',
        'scipy.fftpack',
        
        # Spleeter
        'spleeter',
        'spleeter.separator',
        'spleeter.model',
        'spleeter.audio',
        'spleeter.audio.adapter',
        
        # Image/signal processing
        'skimage',
        'skimage.filters',
        'skimage.morphology',
        
        # Audio analysis
        'librosa',
        'librosa.core',
        'librosa.util',
        
        # TensorFlow (for Spleeter)
        'tensorflow',
        'tensorflow.python',
        
        # FFmpeg
        'ffmpeg',
        
        # GUI
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.ttk',
        'tkinter.scrolledtext',
        
        # YouTube (optional)
        'yt_dlp',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludedimports=[
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
    name='MusicAssassin_v0.3.3',
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
