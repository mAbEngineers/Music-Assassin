#!/usr/bin/env python3
"""Draws the app icon: a dark ink treble clef on an ivory plate, sliced once
by a thin red cut. The piece above the cut lifts away and fades, so the slice
still reads at taskbar sizes. Rendered at 4x and downsampled for clean edges.

The clef is the treble-clef glyph (U+1D11E) from the Noto Music font (SIL
Open Font License), thickened slightly so it holds up at 32 px. Install it
with your distro's Noto fonts package, or pass its path with --font.

Writes:
  build/icon.png          1024 px master; build.sh renders the .deb's hicolor
                          sizes from it
  build/icon.ico          Windows executable icon (spec: icon=)
  assets/icon*.png        Tk window icons (128, 64, 48, 32 px), bundled by
                          the spec

Usage: python build/draw_icon.py [--font /path/to/NotoMusic-Regular.ttf]
"""
import argparse
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SIZE = 1024
SS = 4  # supersampling factor
S = SIZE * SS

FONT_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoMusic-Regular.ttf",
    "/usr/share/fonts/noto/NotoMusic-Regular.ttf",
    "/usr/share/fonts/google-noto/NotoMusic-Regular.ttf",
]
CLEF = "\U0001D11E"

PLATE_TOP, PLATE_BOTTOM = (248, 245, 238), (224, 218, 206)
INK_TOP, INK_BOTTOM = (40, 46, 86), (22, 26, 52)
RED_TOP, RED_BOTTOM = (214, 44, 58), (168, 24, 40)

# Clef box and the cut, as fractions of the canvas
CLEF_CX, CLEF_TOP, CLEF_H = 0.50, 0.12, 0.76
CLEF_THICKEN = 0.016
CUT_P = (0.50, CLEF_TOP + 0.42 * CLEF_H)
CUT_ANGLE = -24       # degrees; the cut rises to the right
CUT_GAP = 0.02        # width removed from the clef along the cut
CUT_LINE = 0.016      # width of the red line at its middle
CUT_HALF_LEN = 0.37
LIFT = 0.045          # how far the top piece moves away from the cut
TOP_ALPHA = 110       # the lifted piece fades (0-255)


def px(v):
    return int(v * S)


def _gradient(top, bottom):
    col = Image.new("RGB", (1, S))
    for y in range(S):
        t = y / (S - 1)
        col.putpixel((0, y), tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)))
    return col.resize((S, S)).convert("RGBA")


def _fill(img, mask, top, bottom, alpha=255, glow=0.0):
    """Composite a vertical gradient through `mask` onto `img`, with an
    optional soft glow of the bottom colour behind it."""
    if alpha < 255:
        mask = mask.point(lambda v: v * alpha // 255)
    if glow:
        g = Image.new("RGBA", (S, S), bottom + (0,))
        g.putalpha(mask.point(lambda v: v * 100 // 255).filter(ImageFilter.GaussianBlur(glow * S)))
        img = Image.alpha_composite(img, g)
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    layer.paste(_gradient(top, bottom), (0, 0), mask)
    return Image.alpha_composite(img, layer)


def _clef_mask(font_path):
    font = ImageFont.truetype(font_path, 1000)
    tmp = Image.new("L", (2600, 2600), 0)
    ImageDraw.Draw(tmp).text((300, 300), CLEF, font=font, fill=255)
    g = tmp.crop(tmp.getbbox())
    h = px(CLEF_H)
    g = g.resize((round(g.width * h / g.height), h), Image.LANCZOS)
    out = Image.new("L", (S, S), 0)
    out.paste(g, (px(CLEF_CX) - g.width // 2, px(CLEF_TOP)))
    return out.filter(ImageFilter.MaxFilter(px(CLEF_THICKEN) | 1))


def _strip(width, half_len=3.0, taper=False):
    """Band of `width` along the cut line; tapered to points if `taper`."""
    a = math.radians(CUT_ANGLE)
    ux, uy = math.cos(a), math.sin(a)
    nx, ny = -uy * width / 2, ux * width / 2
    x, y = CUT_P
    if taper:
        pts = [(x - half_len * ux, y - half_len * uy), (x + nx, y + ny),
               (x + half_len * ux, y + half_len * uy), (x - nx, y - ny)]
    else:
        pts = [(x - half_len * ux + nx, y - half_len * uy + ny), (x + half_len * ux + nx, y + half_len * uy + ny),
               (x + half_len * ux - nx, y + half_len * uy - ny), (x - half_len * ux - nx, y - half_len * uy - ny)]
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).polygon([(px(a_), px(b_)) for a_, b_ in pts], fill=255)
    return m


def _above_cut():
    a = math.radians(CUT_ANGLE)
    ux, uy = math.cos(a), math.sin(a)
    nx, ny = uy, -ux  # normal pointing up, away from the cut
    x, y = CUT_P
    pts = [(x - 3 * ux, y - 3 * uy), (x + 3 * ux, y + 3 * uy),
           (x + 3 * ux + 3 * nx, y + 3 * uy + 3 * ny), (x - 3 * ux + 3 * nx, y - 3 * uy + 3 * ny)]
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).polygon([(px(a_), px(b_)) for a_, b_ in pts], fill=255)
    return m


def draw(font_path) -> Image.Image:
    plate = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    plate_mask = Image.new("L", (S, S), 0)
    m = int(0.03 * S)
    ImageDraw.Draw(plate_mask).rounded_rectangle((m, m, S - m, S - m), radius=int(0.2 * S), fill=255)
    plate.paste(_gradient(PLATE_TOP, PLATE_BOTTOM), (0, 0), plate_mask)

    clef = ImageChops.subtract(_clef_mask(font_path), _strip(CUT_GAP))
    above = _above_cut()
    top = ImageChops.multiply(clef, above)
    low = ImageChops.subtract(clef, above)
    a = math.radians(CUT_ANGLE)
    dx, dy = math.sin(a) * LIFT, -math.cos(a) * LIFT
    top = top.transform((S, S), Image.AFFINE, (1, 0, -px(dx), 0, 1, -px(dy)))

    img = _fill(plate, low, INK_TOP, INK_BOTTOM)
    img = _fill(img, top, INK_TOP, INK_BOTTOM, alpha=TOP_ALPHA)
    img = _fill(img, _strip(CUT_LINE, CUT_HALF_LEN, taper=True), RED_TOP, RED_BOTTOM, glow=0.022)
    # keep the glow inside the plate
    img.putalpha(ImageChops.multiply(img.split()[3], plate_mask))
    return img.resize((SIZE, SIZE), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--font", help="path to NotoMusic-Regular.ttf")
    args = ap.parse_args()
    font = args.font or next((p for p in FONT_PATHS if Path(p).is_file()), None)
    if not font:
        raise SystemExit("Noto Music font not found; install it or pass --font /path/to/NotoMusic-Regular.ttf")

    icon = draw(font)
    icon.save(ROOT / "build/icon.png")
    icon.save(ROOT / "build/icon.ico",
              sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    for name, size in (("icon.png", 128), ("icon_64.png", 64),
                       ("icon_48.png", 48), ("icon_32.png", 32)):
        icon.resize((size, size), Image.LANCZOS).save(assets / name)
    print("wrote build/icon.png, build/icon.ico and 4 window icons in assets/")


if __name__ == "__main__":
    main()
