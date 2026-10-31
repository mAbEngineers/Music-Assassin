#!/usr/bin/env python3
"""Draw the pipeline diagrams as static SVG.

    python3 docs/pipeline/draw_pipelines.py      # writes docs/pipeline/*.svg

No dependencies. GitHub renders Mermaid with pan/zoom/copy buttons over the
diagram; a plain SVG image has none, so the README and
docs/PIPELINE_EVOLUTION.md embed these instead.

Each diagram is a grid: nodes sit at (row, col) and edges are drawn straight
(same row or column) or as one elbow. Keep a branch's column clear between
its two ends. The text in each diagram was derived from the real per-file
processing function of that version; see docs/PIPELINE_EVOLUTION.md.
"""

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent
FONT, LINE_H, CHAR_W, PAD_X, PAD_Y = 13, 17, 7.0, 14, 11
COL_GAP, ROW_GAP, LABEL_GAP, MARGIN = 46, 50, 74, 24

STYLE = {  # kind -> (fill, stroke)
    "stage":  ("#f8f9fa", "#495057"),
    "choice": ("#f3f0ff", "#5f3dc4"),
    "end":    ("#e7f5ff", "#1971c2"),
    "note":   ("#fff9db", "#e67700"),
}
NEW = ("#ebfbee", "#2b8a3e")


class N:
    def __init__(self, row, col, *lines, kind="stage", opt=False, new=False):
        self.row, self.col, self.lines = row, col, list(lines)
        self.kind, self.opt, self.new = kind, opt, new

    def size(self):
        w = max(len(s) for s in self.lines) * CHAR_W + 2 * PAD_X
        if self.kind == "choice":
            w += 28
        return max(w, 112), len(self.lines) * LINE_H + 2 * PAD_Y


def E(a, b, label=None, dashed=False):
    return (a, b, label, dashed)


def render(nodes, edges, caption=None):
    rows = max(n.row for n in nodes.values()) + 1
    cols = max(n.col for n in nodes.values()) + 1
    col_w = [max([n.size()[0] for n in nodes.values() if n.col == c] or [0]) for c in range(cols)]
    row_h = [max([n.size()[1] for n in nodes.values() if n.row == r] or [0]) for r in range(rows)]
    xs, x = [], MARGIN
    for w in col_w:
        xs.append(x)
        x += w + COL_GAP
    # A row entered by a labelled edge from another row gets a taller gap
    # above it, so the label has its own stretch of line.
    gap = [ROW_GAP] * rows
    for a, b, label, _ in edges:
        if label and nodes[a].row != nodes[b].row:
            gap[nodes[b].row] = LABEL_GAP
    ys, y = [], MARGIN
    for r, h in enumerate(row_h):
        y += gap[r] if r else 0
        ys.append(y)
        y += h
    width = x - COL_GAP + MARGIN
    height = y + MARGIN + (LINE_H + 6 if caption else 0)

    box = {}
    for k, n in nodes.items():
        w, h = col_w[n.col], n.size()[1]
        cx, cy = xs[n.col] + col_w[n.col] / 2, ys[n.row] + row_h[n.row] / 2
        box[k] = (cx - w / 2, cy - h / 2, w, h)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
           f'width="{width:.0f}" height="{height:.0f}" font-family="Helvetica, Arial, sans-serif" '
           f'font-size="{FONT}">',
           '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
           'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#495057"/></marker></defs>',
           f'<rect width="{width:.0f}" height="{height:.0f}" rx="8" fill="#ffffff"/>']

    fanout = {}
    for a, b, _, _ in edges:
        fanout[a] = fanout.get(a, 0) + 1
    labels = []
    for a, b, label, dashed in edges:
        ax, ay, aw, ah = box[a]
        bx, by, bw, bh = box[b]
        acx, acy, bcx, bcy = ax + aw / 2, ay + ah / 2, bx + bw / 2, by + bh / 2
        na, nb = nodes[a], nodes[b]
        if na.row == nb.row:
            x1, x2 = (ax + aw, bx) if bcx > acx else (ax, bx + bw)
            pts = [(x1, acy), (x2, bcy)]
            lab = ((x1 + x2) / 2, acy - 9)
        elif na.col == nb.col:
            down = nb.row > na.row
            y1, y2 = (ay + ah, by) if down else (ay, by + bh)
            pts = [(acx, y1), (bcx, y2)]
            lab = (acx, (y1 + y2) / 2)
        else:
            # One elbow: bend just below the source when it fans out (so each
            # branch's label sits on its own vertical run), else just above
            # the target (so merging edges converge).
            y1, y2 = ay + ah, by
            ym = (y1 + 16) if fanout[a] > 1 else (y2 - 20)
            pts = [(acx, y1), (acx, ym), (bcx, ym), (bcx, y2)]
            lab = (bcx, (ym + y2) / 2)
        d = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        out.append(f'<polyline points="{d}" fill="none" stroke="#495057" stroke-width="1.4"'
                   f'{dash} marker-end="url(#ar)"/>')
        if label:
            labels.append((lab, label))

    for k, n in nodes.items():
        x, y, w, h = box[k]
        fill, stroke = NEW if n.new else STYLE[n.kind]
        dash = ' stroke-dasharray="6 4"' if n.opt else ""
        attrs = f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"{dash}'
        if n.kind == "choice":
            i = 14
            p = [(x + i, y), (x + w - i, y), (x + w, y + h / 2), (x + w - i, y + h), (x + i, y + h), (x, y + h / 2)]
            out.append(f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in p)}" {attrs}/>')
        else:
            rx = h / 2 if n.kind == "end" else 6
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" {attrs}/>')
        ty = y + h / 2 - (len(n.lines) - 1) * LINE_H / 2 + 4.5
        for i, s in enumerate(n.lines):
            weight = ' font-weight="700"' if i == 0 else ""
            out.append(f'<text x="{x + w / 2:.1f}" y="{ty + i * LINE_H:.1f}" text-anchor="middle" '
                       f'fill="#212529"{weight}>{escape(s)}</text>')

    for (lx, ly), s in labels:
        w = len(s) * 6.2 + 10
        out.append(f'<rect x="{lx - w / 2:.1f}" y="{ly - 9:.1f}" width="{w:.1f}" height="17" rx="3" '
                   f'fill="#ffffff" stroke="#ced4da"/>')
        out.append(f'<text x="{lx:.1f}" y="{ly + 3.5:.1f}" text-anchor="middle" font-size="11" '
                   f'fill="#495057">{escape(s)}</text>')
    if caption:
        out.append(f'<text x="{MARGIN}" y="{height - MARGIN + 4:.0f}" font-size="12" '
                   f'fill="#495057">{escape(caption)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def chain(*keys, dashed=False):
    return [E(a, b, dashed=dashed) for a, b in zip(keys, keys[1:])]


IN = lambda r, c: N(r, c, "Input audio / video", kind="end")

DIAGRAMS = {
    # The current pipeline (main), laid out in three rows for the README.
    "current": (dict(
        IN=N(0, 0, "Input audio / video", kind="end"),
        EXT=N(0, 1, "Extract", "ffmpeg, stereo", "(mono sources: mono)"),
        PRE=N(0, 2, "Pre-filter", "high-pass + normalise"),
        SEP=N(0, 3, "Separate vocals", "Voc_FT (default), MRX,", "MDX-Net, Demucs, Spleeter"),
        FX=N(1, 1, "Separate effects", "MRX or CDX23", "(keep sound effects)", opt=True),
        POST=N(1, 3, "Post-filter", "EQ by mode: band-limit,", "or de-esser + presence"),
        ENH=N(2, 3, "DeepFilterNet", "or another enhancer", opt=True),
        VAD=N(2, 2, "VAD gate, Silero", "after the enhancer", opt=True),
        MIX=N(2, 1, "Mix effects back in", "level-matched, limiter", opt=True),
        OUT=N(2, 0, "Export", "video, subtitles and", "other audio tracks kept", kind="end"),
    ), chain("IN", "EXT", "PRE", "SEP", "POST", "ENH", "VAD", "MIX", "OUT")
       + [E("PRE", "FX", dashed=True), E("FX", "MIX", dashed=True)],
       "Dashed: optional. Stereo sources stay stereo throughout (Stereo output checkbox)."),

    "legend": (dict(
        A=N(0, 0, "Stage", "always runs"),
        B=N(0, 1, "Choice", "between options", kind="choice"),
        C=N(0, 2, "Optional", "checkbox-gated", opt=True),
        D=N(0, 3, "Input or", "output", kind="end"),
        F=N(0, 4, "New in", "this version", new=True),
        G=N(0, 5, "Partial stage", "or side note", kind="note"),
    ), [], None),

    "v0.0.1": (dict(
        IN=IN(0, 0),
        EXT=N(1, 0, "Extract + decode", "MoviePy to_soundarray"),
        CH=N(2, 0, "Chunk", "60 s, no overlap"),
        SEP=N(3, 0, "Separate", "Spleeter 2stems"),
        OUT=N(4, 0, "Export", "MoviePy AudioArrayClip", "-> write_videofile", kind="end"),
    ), chain("IN", "EXT", "CH", "SEP", "OUT"), None),

    "v0.1.0": (dict(
        IN=IN(0, 0),
        EXT=N(1, 0, "Extract", "ffmpeg-python pipe"),
        CH=N(2, 0, "Chunk", "60 s, no overlap"),
        SEP=N(3, 0, "Separate", "Spleeter 2stems"),
        OUT=N(4, 0, "Export", "ffmpeg mux, video passthrough", "(vcodec copy)", kind="end"),
    ), chain("IN", "EXT", "CH", "SEP", "OUT"), None),

    "v0.3.0": (dict(
        IN=IN(0, 1),
        EXT=N(1, 1, "Extract", "ffmpeg pipe"),
        CH=N(2, 1, "Chunk", "30 s, 3 s overlap"),
        SEP=N(3, 1, "Separate", "Spleeter 2stems"),
        BR=N(4, 1, "Has video / subtitles?", kind="choice"),
        O1=N(5, 0, "Export", "audio", kind="end"),
        O2=N(5, 1, "Export", "video passthrough", "+ vocals", kind="end"),
        O3=N(5, 2, "Export", "video + subtitle", "passthrough + vocals", kind="end"),
    ), chain("IN", "EXT", "CH", "SEP", "BR")
       + [E("BR", "O1", "audio only"), E("BR", "O2", "video, no subs"), E("BR", "O3", "video + subs")], None),

    "v0.3.3": (dict(
        IN=IN(0, 0),
        YT=N(0, 1, "v0.3.4: paste a YouTube URL", "yt-dlp downloads it, then", "the same pipeline runs", kind="note"),
        EXT=N(1, 0, "Extract", "ffmpeg pipe"),
        PRE=N(2, 0, "Pre-filter", "high-pass 60 Hz + normalize", new=True),
        CH=N(3, 0, "Chunk", "20 s, 2 s overlap"),
        SEP=N(4, 0, "Separate", "Spleeter 4stems"),
        POST=N(5, 0, "Post-filter", "Normal / Balanced / Aggressive", "notch filters + de-esser", "+ formant boost", new=True),
        OUT=N(6, 0, "Export", "video / subtitle passthrough", kind="end"),
    ), chain("IN", "EXT", "PRE", "CH", "SEP", "POST", "OUT") + [E("YT", "IN", dashed=True)], None),

    "v0.4.0": (dict(
        IN=IN(0, 0),
        EXT=N(1, 0, "Extract", "torchaudio.load + resample"),
        CH=N(2, 0, "Chunk", "30 s, no overlap"),
        SEP=N(3, 0, "Separate", "Demucs apply_model"),
        POST=N(4, 0, "Post-DSP (partial)", "compression + saturation", "+ normalize only", kind="note"),
        OUT=N(5, 0, "Export", "subprocess ffmpeg mux", "-> new *-natural_vocals.mp4", kind="end"),
    ), chain("IN", "EXT", "CH", "SEP", "POST", "OUT"), None),

    "v0.4.1": (dict(
        IN=IN(0, 1),
        EXT=N(1, 1, "Extract", "ffmpeg pipe, 8 MB chunked read"),
        PRE=N(2, 1, "Pre-filter", "audio_prefilter:", "gate + high-pass + normalize"),
        SEP=N(3, 1, "Separator choice", kind="choice"),
        S1=N(4, 0, "htdemucs family", "60 s / 2 s-overlap chunks"),
        S2=N(4, 1, "Spleeter 4stems", "60 s / 2 s-overlap chunks"),
        S3=N(4, 2, "MDX-Net, audio-separator", "whole file; the library", "does its own chunking"),
        POST=N(5, 1, "Post-filter", "gentle de-ess + presence boost;", "notch filters dropped"),
        OUT=N(6, 1, "Export", "video / subtitle passthrough", "+ other audio tracks kept (new)", kind="end"),
    ), chain("IN", "EXT", "PRE", "SEP")
       + [E("SEP", "S1", "Demucs"), E("SEP", "S2", "Spleeter"), E("SEP", "S3", "v0.4.2 adds", dashed=True),
          E("S1", "POST"), E("S2", "POST"), E("S3", "POST"), E("POST", "OUT")], None),

    "v0.4.2-RT": (dict(
        IN=IN(0, 1),
        EXT=N(1, 1, "Extract", "ffmpeg pipe"),
        PRE=N(2, 1, "Pre-filter", "audio_prefilter, always on"),
        HPSS=N(3, 1, "HPSS pre-filter", "optional, off by default", opt=True, new=True),
        SEP=N(4, 1, "Separator choice", kind="choice"),
        S1=N(5, 0, "Selected engine", "MDX-Net / Demucs / Spleeter"),
        TTA=N(5, 2, "TTA: run twice,", "polarity-flipped,", "average the outputs", opt=True, new=True),
        MGAN=N(6, 1, "MetricGAN+ post-filter", "optional, off by default", opt=True, new=True),
        POST=N(7, 1, "Post-filter", "same de-ess / formant", "modes as v0.4.1"),
        OUT=N(8, 1, "Export", kind="end"),
    ), chain("IN", "EXT", "PRE", "HPSS", "SEP")
       + [E("SEP", "S1"), E("SEP", "TTA", "optional", dashed=True), E("S1", "MGAN"), E("TTA", "MGAN"),
          E("MGAN", "POST"), E("POST", "OUT")], None),

    "v0.4.4": (dict(
        IN=IN(0, 0),
        EXT=N(1, 0, "Extract", "ffmpeg pipe"),
        PRE=N(2, 0, "Pre-filter", "audio_prefilter"),
        SEP=N(3, 0, "Separate", "MDX-Net / Demucs / Spleeter"),
        POST=N(4, 0, "Post-filter", "advanced_postprocess_chunked"),
        ENH=N(5, 0, "DeepFilterNet cleanup", "optional", opt=True, new=True),
        VAD=N(6, 0, "VAD gate, Silero", "optional, runs last", opt=True, new=True),
        OUT=N(7, 0, "Export", kind="end"),
    ), chain("IN", "EXT", "PRE", "SEP", "POST", "ENH", "VAD", "OUT"), None),

    "v0.4.5": (dict(
        IN=IN(0, 0),
        EXT=N(1, 0, "Extract audio track", "ffmpeg, mono"),
        PRE=N(2, 0, "Pre-filter", "noise gate + high-pass"),
        FXP=N(2, 1, "Pre-filter, no gate", "high-pass + normalise", opt=True, new=True),
        SEP=N(3, 0, "Separate vocals", "Voc_FT fast engine (default),", "MRX, MDX-Net, Demucs, Spleeter"),
        FX=N(3, 1, "Separate effects", "MRX or CDX23,", "after the vocals", opt=True, new=True),
        POST=N(4, 0, "Post-filter", "EQ by mode: band-limit,", "or de-esser + presence"),
        ENH=N(5, 0, "DeepFilterNet cleanup", "optional", opt=True),
        VAD=N(6, 0, "VAD gate, Silero", "optional, runs last", opt=True),
        MIX=N(7, 0, "Mix effects back in", "level-matched, limiter", opt=True, new=True),
        OUT=N(8, 0, "Export", "video + subtitles preserved", kind="end"),
    ), chain("IN", "EXT", "PRE", "SEP", "POST", "ENH", "VAD", "MIX", "OUT")
       + [E("EXT", "FXP", "keep sound effects", dashed=True), E("FXP", "FX", dashed=True),
          E("FX", "MIX", dashed=True)], None),

    "v0.5.0": (dict(
        IN=IN(0, 1),
        EXT=N(1, 1, "Extract", "ffmpeg pipe, 8 MB chunked read"),
        PRE=N(2, 1, "Pre-filter", "optional, on by default", opt=True),
        SEP=N(3, 1, "Separate", "Demucs only, model selectable;", "one background thread,", "still sequential"),
        POST=N(4, 1, "Post-filter", "optional, on by default", opt=True),
        KEEP=N(5, 1, "Keep original audio tracks?", "default: yes", kind="choice"),
        MERGE=N(6, 0, "Append vocals as a new track", "alongside the originals"),
        REPL=N(6, 2, "Vocals-only output"),
        OUT=N(7, 1, "Export", "atomic in-place overwrite", "+ optional live preview to VLC", kind="end"),
    ), chain("IN", "EXT", "PRE", "SEP", "POST", "KEEP")
       + [E("KEEP", "MERGE", "yes"), E("KEEP", "REPL", "no"), E("MERGE", "OUT"), E("REPL", "OUT")], None),
}


if __name__ == "__main__":
    for name, (nodes, edges, caption) in DIAGRAMS.items():
        path = OUT / f"{name}.svg"
        path.write_text(render(nodes, edges, caption))
        print(f"wrote {path.relative_to(OUT.parent.parent)}")
