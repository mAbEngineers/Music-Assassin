"""Tests that need no models, no GPU and no real media: they run on every
push (.github/workflows/tests.yml). The model-heavy checks live in
test_processors.py and quality_harness/, which need the weights and a sample
video and are run by hand.

    python -m unittest discover -s tests -v
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "quality_harness"))

from quality_harness.pipeline import load_app   # noqa: E402  (stubs tkinter)

app = load_app()
SR = 44100


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12)


class MixTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.voice = (0.1 * rng.standard_normal(SR * 3)).astype(np.float32)
        self.music = (0.1 * rng.standard_normal(SR * 3)).astype(np.float32)

    def test_music_only_lands_at_the_requested_level(self):
        silent = np.zeros_like(self.voice)
        out = app.mix_effects(silent, None, 0.0, SR, music=self.music, music_gain=10 ** (-20 / 20))
        body = slice(SR, 2 * SR)    # clear of the tail fade
        self.assertAlmostEqual(rms_db(out[body]) - rms_db(self.music[body]), -20.0, delta=0.1)

    def test_effects_only_matches_the_old_behaviour(self):
        out = app.mix_effects(self.voice, self.music, 0.5, SR)
        want = self.voice + 0.5 * self.music
        body = slice(SR, 2 * SR)
        np.testing.assert_allclose(out[body], want[body], atol=1e-5)

    def test_both_stems_add(self):
        fx = self.music * 0.5
        out = app.mix_effects(self.voice, fx, 1.0, SR, music=self.music, music_gain=0.1)
        want = self.voice + fx + 0.1 * self.music
        body = slice(SR, 2 * SR)
        np.testing.assert_allclose(out[body], want[body], atol=1e-5)

    def test_stereo_music_over_mono_voice_broadcasts(self):
        stereo = np.stack([self.music, self.music * 0.5])
        out = app.mix_effects(self.voice, None, 0.0, SR, music=stereo, music_gain=0.1)
        self.assertEqual(out.shape, (2, len(self.voice)))

    def test_limiter_holds_the_ceiling(self):
        loud = np.full(SR * 2, 0.9, np.float32)
        out = app.mix_effects(loud, None, 0.0, SR, music=loud, music_gain=1.0, ceiling=0.98)
        self.assertLessEqual(float(np.abs(out).max()), 0.98 + 1e-4)

    def test_levels_table(self):
        self.assertIsNone(app.MUSIC_LEVELS[app.MUSIC_REMOVE])
        self.assertEqual(app.MUSIC_LEVELS[app.MUSIC_LOW], -20.0)
        self.assertEqual(app.MUSIC_LEVELS[app.MUSIC_HIGH], -12.0)


class SettingsTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "settings.json")
            self.assertTrue(app.save_settings({"music": "x", "stereo": False}, path))
            self.assertEqual(app.load_settings(path), {"music": "x", "stereo": False})

    def test_missing_and_damaged_files_give_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(app.load_settings(os.path.join(d, "nope.json")), {})
            bad = os.path.join(d, "bad.json")
            Path(bad).write_text("{not json")
            self.assertEqual(app.load_settings(bad), {})
            Path(bad).write_text("[1, 2]")
            self.assertEqual(app.load_settings(bad), {})

    def test_unwritable_location_is_not_fatal(self):
        self.assertFalse(app.save_settings({"a": 1}, "/proc/nope/settings.json"))

    def test_pick_valid_ignores_stale_choices(self):
        self.assertEqual(app.pick_valid({"k": "a"}, "k", ["a", "b"], "b"), "a")
        self.assertEqual(app.pick_valid({"k": "gone"}, "k", ["a", "b"], "b"), "b")
        self.assertEqual(app.pick_valid({}, "k", ["a", "b"], "b"), "b")


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        self._saved = (app.separator_choice, app.model_choice,
                       dict(app._MODEL_CACHE), dict(app._AUDIO_SEP_CACHE))

    def tearDown(self):
        app.separator_choice, app.model_choice = self._saved[:2]
        app._MODEL_CACHE.clear(); app._MODEL_CACHE.update(self._saved[2])
        app._AUDIO_SEP_CACHE.clear(); app._AUDIO_SEP_CACHE.update(self._saved[3])

    def fill(self):
        app._MODEL_CACHE.update({"htdemucs:q:cpu": 1, "htdemucs_ft:fp32:cpu": 2})
        app._AUDIO_SEP_CACHE.update({"a.onnx": 3, "b.onnx": 4})

    def test_switching_to_fast_engine_drops_the_legacy_models(self):
        self.fill()
        app.separator_choice = "fast_mdx"
        app._free_unused_models()
        self.assertFalse(app._MODEL_CACHE)
        self.assertFalse(app._AUDIO_SEP_CACHE)

    def test_switching_demucs_model_keeps_only_that_model(self):
        self.fill()
        app.separator_choice, app.model_choice = "demucs", "htdemucs"
        app._free_unused_models()
        self.assertEqual(list(app._MODEL_CACHE), ["htdemucs:q:cpu"])   # not htdemucs_ft
        self.assertFalse(app._AUDIO_SEP_CACHE)

    def test_mdx_keeps_only_the_chosen_model(self):
        self.fill()
        app.separator_choice, app.model_choice = "audio_separator", "b.onnx"
        app._free_unused_models()
        self.assertEqual(list(app._AUDIO_SEP_CACHE), ["b.onnx"])
        self.assertFalse(app._MODEL_CACHE)


@unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
class MkvDelayTests(unittest.TestCase):
    """The premise of the MKV export fix: AAC in Matroska comes out
    AAC_PRIMING samples late, and skipping that many input samples cancels
    it. M4A needs no such help. If a future ffmpeg compensates in the muxer,
    this fails and the skip in _process_single_track should go."""

    @staticmethod
    def lag(ext, skip):
        n = SR * 3
        x = np.zeros(n, np.float32)
        x[SR:SR + 2000] = np.random.default_rng(0).standard_normal(2000) * 0.5
        y = np.concatenate([x[skip:], np.zeros(skip, np.float32)])
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "t." + ext)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "f32le", "-ar", str(SR),
                            "-ac", "1", "-i", "pipe:0", "-c:a", "aac", "-b:a", "192k", out],
                           input=y.astype("<f4").tobytes(), check=True)
            dec = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", out, "-f", "f32le", "-ar",
                                  str(SR), "-ac", "1", "pipe:1"], capture_output=True, check=True).stdout
        z = np.frombuffer(dec, "<f4")[:n]
        m = 2 * n
        c = np.fft.irfft(np.fft.rfft(z, m) * np.conj(np.fft.rfft(x, m)), m)
        k = int(np.argmax(c))
        return k if k < m // 2 else k - m

    def test_mkv_is_late_without_the_skip_and_on_time_with_it(self):
        self.assertEqual(self.lag("mkv", 0), app.AAC_PRIMING)
        self.assertEqual(self.lag("mkv", app.AAC_PRIMING), 0)

    def test_m4a_is_on_time_without_it(self):
        self.assertEqual(self.lag("m4a", 0), 0)


class RepoTests(unittest.TestCase):
    def test_every_python_file_compiles(self):
        skip = {".git", "build", "dist", "logs", "__pycache__", "test_outputs", "node_modules"}
        for path in REPO.rglob("*.py"):
            if skip & set(path.relative_to(REPO).parts) or ".venv" in str(path):
                continue
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_harness_self_check(self):
        r = subprocess.run([sys.executable, str(REPO / "quality_harness" / "run_quality.py"),
                            "--self-check"], capture_output=True, text=True, cwd=REPO)
        self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-800:])

    @unittest.skipUnless(shutil.which("xvfb-run"), "needs xvfb-run")
    def test_window_builds_and_remembers_settings(self):
        r = subprocess.run(["xvfb-run", "-a", sys.executable, str(REPO / "tests" / "gui_smoke.py")],
                           capture_output=True, text=True, cwd=REPO, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])


if __name__ == "__main__":
    unittest.main()
