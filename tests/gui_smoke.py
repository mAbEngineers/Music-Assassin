"""Builds the real window under a display and drives it. Run by
tests/test_ci.py through xvfb-run; exits non-zero on the first failure.

A separate process because the headless tests stub tkinter in sys.modules,
and this one needs the real thing."""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
cfg = tempfile.mkdtemp(prefix="ma_cfg_")
os.environ["XDG_CONFIG_HOME"] = cfg
os.environ["APPDATA"] = cfg

spec = importlib.util.spec_from_file_location("ma", REPO / "MusicAssassinGUI_v0.4.5.py")
ma = importlib.util.module_from_spec(spec)
sys.modules["ma"] = ma
spec.loader.exec_module(ma)

from tkinter import Tk


def build():
    root = Tk(className="music-assassin")
    ma.root = root
    ma.setFrames(root)
    ma.apply_settings()
    root.update()
    return root


def check(cond, msg):
    if not cond:
        print("FAIL:", msg)
        sys.exit(1)
    print("ok:", msg)


root = build()
check(ma.separator_choice in ("fast_mdx", "demucs", "audio_separator", "spleeter", "mrx"),
      "a separator is selected at startup")
check(ma.music_level_db is None, "music is removed by default")

# queue
ma.files.append("/tmp/a.mkv")
ma.queue_view.add("/tmp/a.mkv")
ma.files.append("/tmp/b.mp3")
ma.queue_view.add("/tmp/b.mp3")
check(ma.queue_view.paths() == ["/tmp/a.mkv", "/tmp/b.mp3"], "files show in the queue")
ma._queue_removed("/tmp/a.mkv")
ma.queue_view.remove("/tmp/a.mkv")
check(ma.files == ["/tmp/b.mp3"], "removing a file takes it out of the work list")
ma.queue_view.set_state("/tmp/b.mp3", "done")
root.update()

# controls drive the globals and are saved
ma.music_selector.set(ma.MUSIC_HIGH)
check(ma.music_level_db == -12.0, "'Turn down a little' sets -12 dB")
ma.music_selector.set(ma.MUSIC_LOW)
check(ma.music_level_db == -20.0, "'Turn down a lot' sets -20 dB")
ma.mode_selector.set("Aggressive  (strongest)")
check(ma.processing_mode == "aggressive", "strength dropdown sets the mode")
ma.stereo_var.set(False)
check(ma.stereo_output is False, "stereo switch sets stereo_output")
saved = json.loads(Path(ma._settings_path()).read_text())
check(saved["music"] == ma.MUSIC_LOW and saved["mode"].startswith("Aggressive")
      and saved["stereo"] is False, "changes are written to the settings file")

# advanced window opens and closes
ma.ui["toggle_advanced"]()
root.update()
ma.ui["toggle_advanced"]()
ma.ui["toggle_details"]()
root.update()

# buttons repaint for the running state
ma.do = 1
ma.refresh_buttons()
ma.do = 0
ma.refresh_buttons()

# a second launch reads the settings back
sys.stdout = sys.__stdout__     # the log widget is about to go away
root.destroy()
ma.music_level_db = None
ma.stereo_output = True
root = build()
check(ma.music_selector.get() == ma.MUSIC_LOW and ma.music_level_db == -20.0,
      "music level is remembered across launches")
check(ma.stereo_output is False and ma.processing_mode == "aggressive",
      "stereo and strength are remembered across launches")
sys.stdout = sys.__stdout__
root.destroy()
print("GUI smoke test passed")
