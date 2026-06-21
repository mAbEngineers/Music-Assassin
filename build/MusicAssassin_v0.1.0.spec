# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec File for Music-Assassin v0.1.0
FFMPEG Transition - Improved Performance over MoviePy

Generated for: FFMPEG improvement phase
Target: Windows/Linux/macOS executable
Python: 3.6+
"""

import sys
import os

block_cipher = None

a = Analysis(
    ['MusicAssassinGUI_v0.1.0.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.ttk',
        'ffmpeg',
        'spleeter',
        'spleeter.separator',
        'numpy',
        'scipy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludedimports=['matplotlib', 'pytest'],
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
    name='MusicAssassin_v0.1.0',
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
