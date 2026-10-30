"""Headless access to the real Music-Assassin stages, driven on arrays.

The shipping entry point `_process_single_track` is GUI-coupled and does its
own ffmpeg extract/export, so the harness does not call it. Instead we load
the app module headlessly (same tkinter-mocking trick as test_processors.py)
and call the individual stages on a numpy array:

    prefilter -> separate -> postprocess -> enhancer -> VAD gate -> mix fx

Driving the stages (rather than the GUI driver) is also what the A2 audits
need: flip one stage, hold the rest, measure the delta. A `StageConfig`
captures exactly which stages run and with what knobs, so a before/after is
just two configs over the same corpus.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import resource
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
APP_FILE = REPO / "MusicAssassinGUI_v0.4.5.py"

_app = None  # cached module


def load_app():
    """Import the app module headlessly, once per process."""
    global _app
    if _app is not None:
        return _app

    for mod_name in (
        "tkinter",
        "tkinter.filedialog",
        "tkinter.messagebox",
        "tkinter.scrolledtext",
        "tkinter.ttk",
    ):
        sys.modules.setdefault(mod_name, types.ModuleType(mod_name))
    tk = sys.modules["tkinter"]
    for attr in (
        "filedialog", "messagebox", "Label", "Frame", "Scrollbar", "Listbox",
        "Button", "Tk", "Entry", "Checkbutton", "BooleanVar", "StringVar",
        "IntVar", "PhotoImage",
    ):
        setattr(tk, attr, lambda *a, **kw: None)
    tk.BooleanVar = lambda *a, **kw: types.SimpleNamespace(
        get=lambda: False, set=lambda v: None
    )
    ttk = sys.modules["tkinter.ttk"]
    for attr in ("Progressbar", "Combobox", "Style"):
        setattr(ttk, attr, lambda *a, **kw: None)
    sys.modules["tkinter.scrolledtext"].ScrolledText = lambda *a, **kw: None

    # ffmpeg-python is a hard top-level import but is only used by the GUI
    # extract/export driver, never by the stages the harness calls. Stub it so
    # the harness runs in a minimal env without ffmpeg-python installed.
    if importlib.util.find_spec("ffmpeg") is None:
        ff = types.ModuleType("ffmpeg")
        ff.input = lambda *a, **kw: None
        ff.probe = lambda *a, **kw: {}
        sys.modules["ffmpeg"] = ff

    spec = importlib.util.spec_from_file_location(
        "music_assassin_v044", str(APP_FILE)
    )
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)

    # Neutralise the GUI hooks the stages may touch.
    app._run_in_background = lambda fn, msg="": fn()
    app.root = types.SimpleNamespace(
        update=lambda: None, update_idletasks=lambda: None
    )
    app.progress = None
    _app = app
    return app


@dataclasses.dataclass
class StageConfig:
    """One pipeline configuration. Defaults mirror the app's own defaults."""

    name: str = "default"
    # Prefilter
    prefilter: bool = True
    prefilter_gate: bool = False       # the -40 dB per-sample pre-gate (off since A2)
    # Separation
    separator: str = "fast_mdx"        # fast_mdx | demucs | none
    model_choice: str = "htdemucs"
    fast_overlap: float = 0.25         # 0.25 vs 0.1 (A2)
    # Post / enhance / gate
    postprocess_mode: str = "balanced"  # normal(10kHz LPF) | balanced | aggressive
    postprocess: bool = True
    enhancer: str = "none"             # none | deepfilter | dpdfnet | ...
    vad_gate: bool = False
    # A4 music mixback: add this fraction of the removed music back (0 = delete)
    music_mixback_db: float | None = None  # None/Off, or e.g. -20, -12
    # A3 stereo: False downmixes a stereo clip to mono first (as ffmpeg ac=1
    # does), i.e. the pre-A3 pipeline. stereo_enhance mirrors the app global.
    stereo: bool = True
    stereo_enhance: str = "mid"            # mid | per_channel

    def describe(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class RunResult:
    config: StageConfig
    output: np.ndarray
    proc_seconds: float
    peak_rss_mb: float
    stages_run: list[str]
    error: str | None = None


def _peak_rss_mb() -> float:
    # ru_maxrss is KB on Linux, bytes on macOS. Assume Linux (platform=linux).
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _separate(app, waveform: np.ndarray, sr: int, cfg: StageConfig):
    """Run the chosen separator. Returns (vocals, music_estimate_or_None)."""
    if cfg.separator == "none":
        # DSP-only path: treat the (prefiltered) mix as the vocals stem. Lets
        # the pre-gate / low-pass audits run without loading a model.
        return waveform, None
    if cfg.separator == "fast_mdx":
        import fast_separation
        vocals = fast_separation.separate_vocals_mdx(
            waveform, sr, overlap=cfg.fast_overlap
        )
        # Music estimate for A4 mixback = what the separator removed.
        music = waveform[..., : vocals.shape[-1]] - vocals
        return vocals, music
    if cfg.separator == "demucs":
        app.model_choice = cfg.model_choice
        vocals = app.separate_with_demucs(app._mid(waveform), sr)
        music = app._mid(waveform)[: len(vocals)] - vocals
        return vocals, music
    raise ValueError(f"unsupported separator for harness: {cfg.separator!r}")


def run_config(mix: np.ndarray, sr: int, cfg: StageConfig) -> RunResult:
    """Run `mix` through the stages selected by `cfg`, timed and measured."""
    app = load_app()
    app.sample_rate = sr
    stages: list[str] = []
    x = np.asarray(mix, dtype=np.float32)
    if x.ndim == 2 and not cfg.stereo:
        x = x.mean(axis=0)
        stages.append("downmix")
    app.stereo_enhance = cfg.stereo_enhance
    t0 = time.perf_counter()
    try:
        if cfg.prefilter:
            x = app.audio_prefilter(x, sr, gate=cfg.prefilter_gate)
            stages.append("prefilter(gate=%s)" % cfg.prefilter_gate)

        vocals, music = _separate(app, x, sr, cfg)
        stages.append("separate:%s" % cfg.separator)

        if cfg.postprocess:
            vocals = app.advanced_postprocess_chunked(
                vocals, sr, mode=cfg.postprocess_mode
            )
            stages.append("postprocess:%s" % cfg.postprocess_mode)

        if cfg.enhancer and cfg.enhancer != "none":
            app.enhancer_choice = cfg.enhancer
            vocals = app.apply_enhancer(vocals, sr)
            stages.append("enhancer:%s" % cfg.enhancer)

        if cfg.vad_gate:
            app.vad_enabled = True
            vocals = app.apply_vad_gate(vocals, sr)
            stages.append("vad_gate")

        if cfg.music_mixback_db is not None and music is not None:
            gain = float(10.0 ** (cfg.music_mixback_db / 20.0))
            n = min(vocals.shape[-1], music.shape[-1])
            vocals = vocals[..., :n] + gain * music[..., :n]
            stages.append("music_mixback:%gdB" % cfg.music_mixback_db)

        proc = time.perf_counter() - t0
        return RunResult(cfg, np.asarray(vocals, dtype=np.float32), proc,
                         _peak_rss_mb(), stages)
    except Exception as exc:  # noqa: BLE001 - harness reports, never crashes
        proc = time.perf_counter() - t0
        return RunResult(cfg, x, proc, _peak_rss_mb(), stages, error=repr(exc))
