"""Quality harness runner: score pipeline configs against ground-truth stems.

Usage
-----
    python3 quality_harness/run_quality.py --self-check
        DSP-only smoke run (separator=none), no models needed. Proves the
        harness wiring end to end and emits a report.

    python3 quality_harness/run_quality.py --separator fast_mdx --audit a2
        Run the A2 audits (pre-gate on/off, overlap 0.25 vs 0.1, Normal-mode
        10 kHz low-pass, VAD gate on/off, DeepFilterNet, Best) with the real
        fast engine and emit a before/after table.

    ... --real-speech speech.wav --real-song song.wav --real-vocals vox.wav
        Add Tier 3: real speech over the instrumental-only stretches of a
        real song (found via its separated vocal stem). --real-song and
        --real-vocals repeat in pairs. All inputs must be wav.

Every run writes a timestamped markdown report + a JSON sidecar under
quality_harness/reports/. The markdown is the before/after artefact A1 is
supposed to produce; the JSON is for diffing across commits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402
import metrics  # noqa: E402
from pipeline import RunResult, StageConfig, run_config  # noqa: E402

REPORTS = Path(__file__).resolve().parent / "reports"


def _masked(sig: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return sig[mask] if mask.any() else sig[:0]


def score_run(clip: corpus.Clip, res: RunResult) -> dict:
    """All metrics for one (clip, config) pair, as a flat dict."""
    out = res.output
    n = min(out.size, clip.mix.size)
    out = out[:n]
    mix = clip.mix[:n]
    speech = clip.speech[:n]
    music = clip.music[:n]
    effects = clip.effects[:n]
    speech_mask = clip.speech_mask[:n]
    effects_mask = clip.effects_mask[:n]
    music_only = clip.music_only_mask[:n]

    scores: dict[str, float] = {}
    scores["music_suppression_db"] = metrics.music_suppression_db(
        out, mix, music_only
    )
    if speech_mask.any():
        scores["contrast_gain_db"] = metrics.contrast_gain_db(
            out, mix, speech_mask, music_only
        )
        scores["dialogue_sisdr_db"] = metrics.si_sdr(
            _masked(out, speech_mask), _masked(speech, speech_mask)
        )
        scores["dialogue_mag_snr_db"] = metrics.magnitude_snr(
            out, speech, speech_mask
        )
        scores["dialogue_retention_db"] = metrics.retention_db(
            out, speech, speech_mask
        )
    if effects_mask.any():
        scores["effects_sisdr_db"] = metrics.si_sdr(
            _masked(out, effects_mask), _masked(effects, effects_mask)
        )
        scores["effects_mag_snr_db"] = metrics.magnitude_snr(
            out, effects, effects_mask
        )
        scores["effects_keep_db"] = metrics.retention_db(
            out, effects, effects_mask
        )
        # Leak into effects-absent regions, relative to the clean effects stem.
        scores["effects_leak_db"] = metrics.leak_energy_db(
            out, ~effects_mask & music_only, reference_level=effects
        )
    scores["musical_noise"] = metrics.musical_noise_index(out, clip.sr)
    scores["rtf"] = metrics.rtf(res.proc_seconds, clip.duration_s)
    scores["peak_rss_mb"] = res.peak_rss_mb
    return metrics.summarise(scores)


def run_matrix(
    clips: list[corpus.Clip], configs: list[StageConfig]
) -> list[dict]:
    rows: list[dict] = []
    for clip in clips:
        for cfg in configs:
            res = run_config(clip.mix, clip.sr, cfg)
            row = {
                "clip": clip.name,
                "config": cfg.name,
                "stages": res.stages_run,
                "error": res.error,
                "scores": score_run(clip, res),
            }
            rows.append(row)
            status = "ERR" if res.error else "ok"
            print(f"  [{status}] {clip.name:22s} / {cfg.name:20s} "
                  f"contrast={row['scores'].get('contrast_gain_db')}dB "
                  f"mag={row['scores'].get('dialogue_mag_snr_db')}dB "
                  f"rtf={row['scores'].get('rtf')}")
            if res.error:
                print(f"        -> {res.error}")
    return rows


# --------------------------------------------------------------------------
# Config sets.
# --------------------------------------------------------------------------

def audit_a2_configs(separator: str) -> list[StageConfig]:
    """The A2 cheap audits as paired before/after configs."""
    base = dict(separator=separator)
    return [
        StageConfig(name="baseline", **base),
        StageConfig(name="pregate_on", prefilter_gate=True, **base),
        StageConfig(name="overlap_0.10", fast_overlap=0.10, **base),
        StageConfig(name="postproc_normal", postprocess_mode="normal", **base),
        StageConfig(name="postproc_off", postprocess=False, **base),
        StageConfig(name="vad_on", vad_gate=True, **base),
        StageConfig(name="deepfilter_on", enhancer="deepfilter", **base),
        StageConfig(name="best(df+vad)", enhancer="deepfilter", vad_gate=True, **base),
    ]


def a4_mixback_configs(separator: str) -> list[StageConfig]:
    """A4 'turn down, don't delete' sweep."""
    base = dict(separator=separator)
    return [
        StageConfig(name="mixback_off", music_mixback_db=None, **base),
        StageConfig(name="mixback_-20dB", music_mixback_db=-20.0, **base),
        StageConfig(name="mixback_-12dB", music_mixback_db=-12.0, **base),
    ]


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------

_COLS = [
    ("contrast_gain_db", "Contrast gain (dB)↑"),
    ("music_suppression_db", "Music supp (dB)↑"),
    ("dialogue_retention_db", "Dialog ret. (dB)"),
    ("dialogue_mag_snr_db", "Dialog magSNR↑"),
    ("dialogue_sisdr_db", "Dialog SI-SDR↑"),
    ("effects_mag_snr_db", "FX magSNR↑"),
    ("effects_sisdr_db", "FX SI-SDR↑"),
    ("effects_leak_db", "FX leak (dB)↓"),
    ("musical_noise", "Mus.noise↓"),
    ("rtf", "RTF↓"),
    ("peak_rss_mb", "Peak RSS MB"),
]


def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float) and (v != v):  # NaN
        return "–"
    if isinstance(v, float) and v == float("inf"):
        return "∞"
    return f"{v:g}"


def report_markdown(rows: list[dict], title: str) -> str:
    lines = [f"# {title}", "",
             f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_", "",
             "Ground-truth stems from `quality_harness/corpus.py` "
             "(true-stem tier — scored against clean sources we own, not a "
             "demucs reference). ↑ higher is better, ↓ lower is better.",
             "magSNR is phase-blind and tracks audible damage; SI-SDR also "
             "penalises the inaudible phase shift of the app's IIR filters, "
             "so compare it only between configs with the same filters. "
             "Contrast gain is gain-invariant; music supp and dialogue "
             "retention move with the app's peak-normalise.", ""]
    clips = sorted({r["clip"] for r in rows})
    for clip in clips:
        lines.append(f"## {clip}")
        lines.append("")
        header = "| Config | " + " | ".join(h for _, h in _COLS) + " |"
        sep = "|" + "---|" * (len(_COLS) + 1)
        lines.append(header)
        lines.append(sep)
        for r in (x for x in rows if x["clip"] == clip):
            cells = [r["config"]]
            for key, _ in _COLS:
                cells.append(_fmt(r["scores"].get(key)))
            row = "| " + " | ".join(cells) + " |"
            if r["error"]:
                row += "  <!-- ERROR: %s -->" % r["error"]
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


def write_reports(rows: list[dict], title: str, slug: str) -> tuple[Path, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    md_path = REPORTS / f"{slug}_{ts}.md"
    json_path = REPORTS / f"{slug}_{ts}.json"
    md_path.write_text(report_markdown(rows, title))
    json_path.write_text(json.dumps(rows, indent=2))
    return md_path, json_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true",
                    help="DSP-only run (separator=none), no models needed")
    ap.add_argument("--separator", default="fast_mdx",
                    choices=["none", "fast_mdx", "demucs"])
    ap.add_argument("--audit", choices=["a2", "a4"],
                    help="which audit config set to run")
    ap.add_argument("--clips", default="all",
                    choices=["synthetic", "sfx", "real", "all"])
    ap.add_argument("--real-speech", help="clean speech recording (Tier 3)")
    ap.add_argument("--real-song", action="append", default=[],
                    help="a song with vocals (Tier 3 music bed); repeatable")
    ap.add_argument("--real-vocals", action="append", default=[],
                    help="separated vocals of the matching --real-song; its "
                    "silent stretches mark the instrumental-only bed")
    args = ap.parse_args(argv)

    clips = []
    if args.clips in ("synthetic", "all"):
        clips += corpus.default_synthetic_set()
    if args.clips in ("sfx", "all"):
        clips.append(corpus.build_sfx_leak_mix())
    if args.real_speech:
        if not args.real_song or len(args.real_song) != len(args.real_vocals):
            ap.error("--real-speech needs matching --real-song/--real-vocals pairs")
        if args.clips in ("real", "all"):
            clips += corpus.real_dialogue_set(
                args.real_speech, list(zip(args.real_song, args.real_vocals)))
    elif args.clips == "real":
        ap.error("--clips real needs --real-speech, --real-song and --real-vocals")

    if args.self_check:
        sep = "none"
        configs = [
            StageConfig(name="dsp_baseline", separator=sep),
            StageConfig(name="pregate_on", separator=sep, prefilter_gate=True),
            StageConfig(name="postproc_normal", separator=sep,
                        postprocess_mode="normal"),
        ]
        title, slug = "Quality harness self-check (DSP-only)", "selfcheck"
    elif args.audit == "a2":
        configs = audit_a2_configs(args.separator)
        title = f"A2 audits ({args.separator})"
        slug = f"a2_{args.separator}"
    elif args.audit == "a4":
        configs = a4_mixback_configs(args.separator)
        title = f"A4 music mixback ({args.separator})"
        slug = f"a4_{args.separator}"
    else:
        configs = [StageConfig(name="baseline", separator=args.separator)]
        title, slug = f"Baseline ({args.separator})", "baseline"

    print(f"Running {len(configs)} config(s) over {len(clips)} clip(s)...")
    rows = run_matrix(clips, configs)
    md_path, json_path = write_reports(rows, title, slug)
    print(f"\nReport:  {md_path}")
    print(f"JSON:    {json_path}")
    errors = [r for r in rows if r["error"]]
    if errors:
        print(f"\n{len(errors)} run(s) errored — see report comments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
