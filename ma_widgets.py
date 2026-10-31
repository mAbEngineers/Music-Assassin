"""Small stdlib-only Canvas widgets for the Music Assassin window.

Ported from music-assassin-live (assassin_live/ui/widgets.py) so both apps
look and feel the same: capsule switches, pill dropdowns, a pill button and
antialiased rounded shapes, drawn on a Canvas with numpy. No image assets and
no extra dependencies. ttk cannot be styled into these shapes; a Canvas can.

These widgets own their drawing and nothing else. Colours, fonts and pixel
sizes are all passed in; the palette and display scaling live in the app.

Added here, not in the live app: Segmented (a 2-4 way choice), ProgressBar
(determinate and indeterminate, mimics the bits of ttk.Progressbar the
processing code uses) and Dropdown.set_enabled.
"""

import base64
import struct
import tkinter as tk
import zlib

import numpy as np

# Rendered shapes, keyed by every argument that changes their pixels. Tk
# garbage-collects a PhotoImage the moment nothing references it -- and then
# draws nothing, with no error -- so this cache is also what keeps them alive.
_SHAPES: dict = {}


def _rgb(colour: str) -> tuple:
    return tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))


def _rrect_sdf(w: int, h: int, r: float) -> np.ndarray:
    """Signed distance to a rounded rectangle, per pixel centre.

    The reason for the whole detour: the Tk canvas has no antialiasing. Its
    ovals and smoothed polygons are filled with whole pixels, so every
    circle in the panel — switch knob, fader handle, status dot — gets a
    stepped edge, which at these sizes reads as a smudge rather than as a
    circle. A distance field gives exact edge coverage instead, and numpy is
    already a hard dependency, so it costs nothing to ship.

    A circle is this with r = min(w, h) / 2, so one function draws both.
    """
    ys, xs = np.mgrid[0:h, 0:w]
    px = np.abs(xs + 0.5 - w / 2.0) - (w / 2.0 - r)
    py = np.abs(ys + 0.5 - h / 2.0) - (h / 2.0 - r)
    return (np.hypot(np.maximum(px, 0.0), np.maximum(py, 0.0))
            + np.minimum(np.maximum(px, py), 0.0) - r)


def aa_shape(master, width: int, height: int, *, bg: str, fill: str,
             radius: float | None = None, outline: str | None = None,
             outline_px: float = 1.0) -> tk.PhotoImage:
    """An antialiased rounded rectangle (or circle) as a PhotoImage.

    Composited against `bg` rather than carrying alpha: every one of these
    sits on a known flat colour, and Tk's own PhotoImage alpha handling
    varies by build in ways a control panel should not depend on.
    """
    width, height = max(1, int(width)), max(1, int(height))
    r = min(width, height) / 2.0 if radius is None else float(radius)
    # Keyed by the interpreter as well: a PhotoImage belongs to the Tcl
    # interpreter that made it, and handing one to a widget in another
    # (which only happens in tests, but happens) raises rather than draws.
    key = (master.tk, width, height, bg, fill, r, outline, outline_px)
    if key in _SHAPES:
        return _SHAPES[key]

    d = _rrect_sdf(width, height, r)
    cover = np.clip(0.5 - d, 0.0, 1.0)[..., None]
    img = np.full((height, width, 3), _rgb(bg), dtype=np.float64)
    img += (np.array(_rgb(fill), dtype=np.float64) - img) * cover
    if outline:
        ring = (cover - np.clip(0.5 - (d + outline_px), 0.0, 1.0)[..., None])
        img += (np.array(_rgb(outline), dtype=np.float64) - img) * ring

    photo = tk.PhotoImage(master=master, data=base64.b64encode(
        _png(np.clip(img + 0.5, 0, 255).astype(np.uint8))).decode("ascii"))
    _SHAPES[key] = photo
    return photo


def _png(rgb: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array as PNG.

    Tk 8.6 reads PNG from base64 `-data` on every build we ship to; its PPM
    reader wants a file. Twenty lines of zlib is cheaper than a dependency
    or a temp file per shape.
    """
    h, w, _ = rgb.shape
    # one filter byte (0 = none) per scanline
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def aa_disc(master, diameter: int, *, bg: str, fill: str) -> tk.PhotoImage:
    return aa_shape(master, diameter, diameter, bg=bg, fill=fill)


def round_rect(canvas: tk.Canvas, x0, y0, x1, y1, r, **kw) -> int:
    """A rounded rectangle, as a smoothed polygon.

    Kept for shapes that are drawn once at a size nobody looks at closely.
    Anything with a visible curved edge should use aa_shape() instead — see
    _rrect_sdf for why.
    """
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
           x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
           x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return canvas.create_polygon(pts, smooth=True, **kw)




class Switch(tk.Canvas):
    """A capsule toggle: label lives outside, state lives here.

    Replaces three full-width `tk.Button`s that read `Mid/Side Prefilter:
    OFF`. Every shipping audio tool looked at for this (SoundSource, Easy
    Effects, NVIDIA Broadcast, Krisp) uses a switch in a list, and none of
    them paints a bar across the window to say a filter is on — which the
    band-limit button did, making the calmest state in the app the loudest
    thing in it.
    """

    STEPS = 5           # frames of knob travel; ~60 ms total
    STEP_MS = 12

    def __init__(self, parent, *, bg, track, accent, knob_off, knob_on,
                 value=False, command=None, width=40, height=22):
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self.w, self.h = width, height
        self.bg_colour = bg
        self.track, self.accent = track, accent
        self.knob_off, self.knob_on = knob_off, knob_on
        self.value = bool(value)
        self.command = command
        self.enabled = True
        self._pos = 1.0 if self.value else 0.0
        self._anim = None
        self.bind("<Button-1>", self._on_click)
        self._draw()

    def set_enabled(self, enabled: bool):
        """Grey out and stop responding, for a setting the selected model
        cannot honour. The value is kept, not cleared: re-enabling restores
        what the user last chose rather than silently resetting it."""
        enabled = bool(enabled)
        if enabled == self.enabled:
            return
        self.enabled = enabled
        self.configure(cursor="hand2" if enabled else "")
        self._draw()

    def _draw(self):
        self.delete("all")
        on = self._pos > 0.5
        # Disabled draws in the off colours whatever the value, so a switch
        # that cannot act never shows the accent that means "this is doing
        # something". The caller greys the label beside it.
        track = self.accent if (on and self.enabled) else self.track
        self.create_image(0, 0, anchor="nw", image=aa_shape(
            self, self.w, self.h, bg=self.bg_colour, fill=track))
        # The knob's own edge is blended against the track it sits on, not
        # against the widget background it never touches.
        knob = max(8, int(round(self.h - 6)))
        cx = self.h / 2 + (self.w - self.h) * self._pos
        self.create_image(cx - knob / 2, (self.h - knob) / 2, anchor="nw",
                          image=aa_disc(self, knob, bg=track,
                                        fill=self.knob_on if (on and self.enabled)
                                        else self.knob_off))

    def _on_click(self, _evt=None):
        if not self.enabled:
            return
        self.set(not self.value)
        if self.command:
            self.command(self.value)

    def set(self, value: bool, animate: bool = True):
        """Move to `value`. Never fires the command — callers that flip a
        switch programmatically are reporting state, not requesting it."""
        self.value = bool(value)
        target = 1.0 if self.value else 0.0
        if self._anim is not None:
            self.after_cancel(self._anim)
            self._anim = None
        if not animate:
            self._pos = target
            self._draw()
            return
        self._step(target)

    def _step(self, target: float):
        delta = (target - self._pos)
        if abs(delta) < 1e-3:
            self._pos = target
            self._anim = None
            self._draw()
            return
        self._pos += delta / max(1, self.STEPS) * 2
        self._pos = max(0.0, min(1.0, self._pos))
        self._draw()
        self._anim = self.after(self.STEP_MS, lambda: self._step(target))


class PillButton(tk.Canvas):
    """The power control: a rounded pill with a status dot and a word.

    C9 gave the button four states and put its work on a thread; what was
    left was the shape — a flat rectangle with a `●` glyph standing in for a
    light. The states and their colours still come from the caller, because
    the judgement about what "on" means belongs with the engine, not here.
    """

    def __init__(self, parent, *, bg, font, width, height, command=None):
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0)
        self.w, self.h = width, height
        self.bg_colour = bg
        self.font = font
        self.command = command
        self._enabled = True
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", lambda _e: self._hover(True))
        self.bind("<Leave>", lambda _e: self._hover(False))
        self._state = None
        self._hovering = False

    def _on_click(self, _evt=None):
        if self._enabled and self.command:
            self.command()

    def _hover(self, on: bool):
        self._hovering = on
        if self._state:
            self.paint(*self._state)

    def paint(self, text, fill, fg, dot=None, enabled=True,
              cursor="hand2", outline=""):
        self._state = (text, fill, fg, dot, enabled, cursor, outline)
        self._enabled = enabled
        self.config(cursor=cursor)
        self.delete("all")
        shade = _lighten(fill, 0.12) if (enabled and self._hovering) else fill
        self.create_image(0, 0, anchor="nw", image=aa_shape(
            self, self.w, self.h, bg=self.bg_colour, fill=shade,
            outline=outline or None, outline_px=max(1.0, self.h / 26.0)))
        # Laid out from the measured text rather than from fixed offsets:
        # "stopping…" is twice the width of "ON" and would otherwise sit on
        # top of its own status dot.
        dia = max(7, int(round(self.h * 0.22)))
        gap = dia + int(round(self.h * 0.2))
        span = self.font.measure(text) + (gap if dot else 0)
        x = (self.w - span) / 2.0
        if dot:
            self.create_image(x, (self.h - dia) / 2, anchor="nw",
                              image=aa_disc(self, dia, bg=shade, fill=dot))
            x += gap
        self.create_text(x, self.h / 2, text=text, fill=fg, font=self.font,
                         anchor="w")



class _Popup(tk.Toplevel):
    """The list a Dropdown opens, drawn rather than delegated.

    `tk.Menu` was the second attempt and the second thing to be themed out
    from under us: it took our colours on one machine and the system menu
    palette on another, and it sizes and places itself, so it came up
    narrower than the field it belongs to and offset from it. A borderless
    toplevel with a canvas in it looks the same everywhere, lines up with
    the field exactly, and can say which item is the current one — which a
    picker showing five model names rather badly needs.
    """

    MAX_ROWS = 9        # taller than this and it scrolls instead

    def __init__(self, owner, values, current, *, surface, border, fg, muted,
                 accent, hover, font, width, row_h, on_pick):
        super().__init__(owner)
        self.withdraw()
        self.overrideredirect(True)
        self.configure(bg=border)          # the 1 px frame around the canvas
        self.values, self.current = list(values), current
        self.fg, self.muted, self.accent = fg, muted, accent
        self.surface, self.hover = surface, hover
        self.font, self.row_h, self.on_pick = font, row_h, on_pick
        self.hover_index = self.values.index(current) if current in self.values else -1

        inner_w = max(1, width - 2)
        rows = max(1, min(len(self.values), self.MAX_ROWS))
        self.canvas = tk.Canvas(self, bg=surface, highlightthickness=0, bd=0,
                                width=inner_w, height=rows * row_h,
                                scrollregion=(0, 0, inner_w,
                                              len(self.values) * row_h))
        self.canvas.pack(padx=1, pady=1)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Button-1>", self._on_click)
        for wheel_up, wheel_down in (("<Button-4>", "<Button-5>"),):
            self.canvas.bind(wheel_up, lambda _e: self._scroll(-1))
            self.canvas.bind(wheel_down, lambda _e: self._scroll(1))
        self.bind("<Escape>", lambda _e: self.dismiss())
        self.bind("<Up>", lambda _e: self._step(-1))
        self.bind("<Down>", lambda _e: self._step(1))
        self.bind("<Return>", lambda _e: self._pick(self.hover_index))
        self._draw()

    # -- placement -----------------------------------------------------------
    def show(self, x: int, y: int, above_y: int) -> None:
        """Below the field, or above it when there is no room below."""
        self.update_idletasks()
        h = self.winfo_reqheight()
        if y + h > self.winfo_screenheight() and above_y - h > 0:
            y = above_y - h
        self.geometry(f"+{x}+{y}")
        self.deiconify()
        self.lift()
        self.focus_set()
        # A global grab, so a click anywhere outside dismisses instead of
        # leaving a list floating over an app that has moved on.
        try:
            self.grab_set_global()
        except tk.TclError:
            self.grab_set()
        self.bind("<Button-1>", self._maybe_outside, add="+")

    def _maybe_outside(self, event) -> None:
        if event.widget is not self.canvas:
            self.dismiss()

    def dismiss(self) -> None:
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()

    # -- drawing -------------------------------------------------------------
    def _draw(self) -> None:
        self.canvas.delete("all")
        w = int(self.canvas.cget("width"))
        pad = max(8, int(round(self.row_h * 0.34)))
        for i, value in enumerate(self.values):
            top = i * self.row_h
            chosen = value == self.current
            if i == self.hover_index:
                self.canvas.create_rectangle(0, top, w, top + self.row_h,
                                             fill=self.hover, outline="")
            if chosen:
                # A bar rather than a tick: it survives every font, and it
                # marks the row without competing with the label.
                self.canvas.create_rectangle(0, top + 3, 3, top + self.row_h - 3,
                                             fill=self.accent, outline="")
            self.canvas.create_text(pad, top + self.row_h / 2, text=value,
                                    anchor="w", font=self.font,
                                    fill=self.accent if chosen else self.fg)

    def _index_at(self, event) -> int:
        y = self.canvas.canvasy(event.y)
        i = int(y // self.row_h)
        return i if 0 <= i < len(self.values) else -1

    def _on_motion(self, event) -> None:
        i = self._index_at(event)
        if i != self.hover_index:
            self.hover_index = i
            self._draw()

    def _on_click(self, event) -> None:
        self._pick(self._index_at(event))

    def _pick(self, index: int) -> None:
        if 0 <= index < len(self.values):
            value = self.values[index]
            self.dismiss()
            self.on_pick(value)
        else:
            self.dismiss()

    def _step(self, delta: int) -> None:
        if not self.values:
            return
        self.hover_index = max(0, min(len(self.values) - 1,
                                      self.hover_index + delta))
        self._scroll_to(self.hover_index)
        self._draw()

    def _scroll(self, units: int) -> None:
        self.canvas.yview_scroll(units, "units")

    def _scroll_to(self, index: int) -> None:
        total = len(self.values) * self.row_h
        if total <= int(self.canvas.cget("height")):
            return
        self.canvas.yview_moveto(max(0.0, (index * self.row_h - self.row_h) / total))


class Dropdown(tk.Canvas):
    """A filled pill with a chevron, opening a drawn list (_Popup).

    `ttk.Combobox` was styled through `option_add("*TCombobox*Listbox...")`
    incantations that reach into a widget's internals by name and silently
    do nothing when they miss; `tk.Menu` replaced it and took the system
    palette on some machines. This owns its pixels.

    Reads and writes a `StringVar`, so callers that already hold one keep it.
    """

    def __init__(self, parent, *, variable, values=(), bg, field, border, fg,
                 muted, accent, font, width, height=30, command=None):
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self.w, self.h = width, height
        self.bg_colour = bg
        self.field, self.border, self.fg = field, border, fg
        self.muted, self.accent = muted, accent
        self.font = font
        self.var = variable
        self.values = list(values)
        self.command = command
        self.enabled = True
        self._open = None
        self.bind("<Button-1>", self._popup)
        self.var.trace_add("write", lambda *_: self._draw())
        self._draw()

    def set_values(self, values) -> None:
        self.values = list(values)

    def set_enabled(self, enabled: bool) -> None:
        """Grey out and stop opening, for a setting that does not apply yet
        (the value is kept)."""
        enabled = bool(enabled)
        if enabled != self.enabled:
            self.enabled = enabled
            self.configure(cursor="hand2" if enabled else "")
            self._draw()

    def _draw(self):
        self.delete("all")
        # The field brightens while its list is up, so the two read as one
        # control. Deliberately not the accent: the accent is already doing
        # a job inside the list, marking the current item, and spending it
        # twice in one glance makes neither use mean anything.
        edge = _lighten(self.border, 0.45) if self._open else self.border
        self.create_image(0, 0, anchor="nw", image=aa_shape(
            self, self.w, self.h, bg=self.bg_colour, fill=self.field,
            radius=max(4.0, self.h * 0.22), outline=edge,
            outline_px=max(1.0, self.h / 28.0)))
        pad = max(8, int(round(self.h * 0.34)))
        text = self._fit(self.var.get() or "—", self.w - pad - self.h)
        self.create_text(pad, self.h / 2, text=text, anchor="w",
                         fill=self.fg if self.enabled else self.muted, font=self.font)
        # A glyph rather than two canvas lines: text is the one thing Tk
        # does antialias, and a 1 px diagonal is the one thing it does worst.
        self.create_text(self.w - pad, self.h / 2 + 1, text="▾", anchor="e",
                         fill=self.muted, font=self.font)

    def _fit(self, text: str, room: int) -> str:
        """Device descriptions are long and the field is not. Truncate with
        an ellipsis rather than letting the text run under the chevron."""
        probe = self.create_text(0, -50, text=text, font=self.font, anchor="w")
        try:
            while text and self.bbox(probe)[2] - self.bbox(probe)[0] > room:
                text = text[:-1]
                self.itemconfig(probe, text=text + "…")
            return self.itemcget(probe, "text")
        finally:
            self.delete(probe)

    def _popup(self, _evt=None):
        if self._open is not None or not self.enabled:
            return
        # An empty picker still opens, and says why. Doing nothing on click
        # is indistinguishable from a broken control.
        values, pick = ((self.values, self._choose) if self.values
                        else (["no devices found"], lambda _v: None))
        self._open = _Popup(
            self, values, self.var.get(), surface=self.field,
            border=self.border, fg=self.fg, muted=self.muted,
            accent=self.accent, hover=_lighten(self.field, 0.10),
            font=self.font, width=self.w, row_h=max(24, int(self.h * 0.95)),
            on_pick=pick)
        self._open.bind("<Destroy>", self._closed, add="+")
        self._draw()
        self._open.show(self.winfo_rootx(),
                        self.winfo_rooty() + self.h + max(2, self.h // 8),
                        self.winfo_rooty() - max(2, self.h // 8))

    def _closed(self, event=None):
        if event is None or event.widget is self._open:
            self._open = None
            self._draw()

    def _choose(self, value: str):
        if value == self.var.get():
            return
        self.var.set(value)
        if self.command:
            self.command(value)




def _lighten(hex_colour: str, amount: float) -> str:
    """Hover shade. Kept here so widgets never need the palette."""
    try:
        r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    except (ValueError, IndexError):
        return hex_colour
    mix = lambda c: min(255, int(c + (255 - c) * amount))  # noqa: E731
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


# Vertical counterpart of HSlider — used for the main mix fader and the


class Segmented(tk.Canvas):
    """A 2-4 way choice drawn as one pill with the chosen part filled.

    Reads and writes a StringVar. `options` is a list of (value, label).
    Used where a non-technical user should see every choice at once instead
    of opening a list.
    """

    def __init__(self, parent, *, variable, options, bg, track, accent,
                 fg_on, fg_off, font, width, height=34, command=None):
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self.w, self.h = width, height
        self.bg_colour = bg
        self.track, self.accent = track, accent
        self.fg_on, self.fg_off = fg_on, fg_off
        self.font = font
        self.var = variable
        self.options = list(options)
        self.command = command
        self.enabled = True
        self.bind("<Button-1>", self._on_click)
        self.var.trace_add("write", lambda *_: self._draw())
        self._draw()

    def set_enabled(self, enabled: bool) -> None:
        if bool(enabled) != self.enabled:
            self.enabled = bool(enabled)
            self.configure(cursor="hand2" if self.enabled else "")
            self._draw()

    def _cell(self, i):
        n = len(self.options)
        return self.w * i / n, self.w * (i + 1) / n

    def _draw(self):
        self.delete("all")
        self.create_image(0, 0, anchor="nw", image=aa_shape(
            self, self.w, self.h, bg=self.bg_colour, fill=self.track,
            radius=self.h * 0.3))
        cur = self.var.get()
        inset = 3
        for i, (value, label) in enumerate(self.options):
            x0, x1 = self._cell(i)
            on = value == cur
            if on:
                self.create_image(x0 + inset, inset, anchor="nw", image=aa_shape(
                    self, int(x1 - x0) - 2 * inset, self.h - 2 * inset,
                    bg=self.track, fill=self.accent if self.enabled else _lighten(self.track, 0.2),
                    radius=(self.h - 2 * inset) * 0.3))
            self.create_text((x0 + x1) / 2, self.h / 2, text=label, font=self.font,
                             fill=self.fg_on if on else self.fg_off)

    def _on_click(self, evt):
        if not self.enabled:
            return
        i = min(len(self.options) - 1, max(0, int(evt.x / self.w * len(self.options))))
        value = self.options[i][0]
        if value != self.var.get():
            self.var.set(value)
            if self.command:
                self.command(value)


class ProgressBar(tk.Canvas):
    """A thin pill progress bar with the slice of the ttk.Progressbar API the
    processing code uses: pb['value'] = 0..100, configure(mode=...),
    start(ms) / stop() for the indeterminate sweep."""

    def __init__(self, parent, *, bg, track, fill, width, height=8):
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0)
        self.w, self.h = width, height
        self.bg_colour = bg
        self.track, self.fill = track, fill
        self._value = 0.0
        self._mode = "determinate"
        self._tick = None
        self._phase = 0.0
        self.bind("<Configure>", self._resize)
        self._draw()

    def _resize(self, evt):
        if evt.width > 8 and evt.width != self.w:
            self.w = evt.width
            self._draw()

    # ttk-compatible surface ------------------------------------------------
    def configure(self, cnf=None, **kw):
        if "mode" in kw:
            self._mode = kw.pop("mode")
        if "value" in kw:
            self._value = max(0.0, min(100.0, float(kw.pop("value"))))
        if kw or cnf:
            super().configure(cnf, **kw)
        self._draw()
    config = configure

    def cget(self, key):
        return self._value if key == "value" else self._mode if key == "mode" else super().cget(key)

    def __setitem__(self, key, value):
        self.configure(**{key: value})

    def __getitem__(self, key):
        return self.cget(key)

    def start(self, interval=40):
        self.stop()
        def step():
            self._phase = (self._phase + 0.02) % 1.0
            self._draw()
            self._tick = self.after(interval, step)
        self._tick = self.after(interval, step)

    def stop(self):
        if self._tick is not None:
            self.after_cancel(self._tick)
            self._tick = None

    # drawing ---------------------------------------------------------------
    def _draw(self):
        self.delete("all")
        w, h = max(self.w, 10), self.h
        self.create_image(0, 0, anchor="nw", image=aa_shape(
            self, w, h, bg=self.bg_colour, fill=self.track))
        if self._mode == "indeterminate" and self._tick is not None:
            seg = max(h * 3, w // 5)
            x = int((w + seg) * self._phase) - seg
            x0, x1 = max(0, x), min(w, x + seg)
            if x1 - x0 > 2:
                self.create_image(x0, 0, anchor="nw", image=aa_shape(
                    self, x1 - x0, h, bg=self.track, fill=self.fill, radius=h / 2))
            return
        if self._value <= 0:
            return
        # Quantised to 4 px so the cache of rendered shapes stays small.
        fw = max(h, int(w * self._value / 100.0) // 4 * 4)
        self.create_image(0, 0, anchor="nw", image=aa_shape(
            self, fw, h, bg=self.track, fill=self.fill))


class QueueList(tk.Canvas):
    """The file queue: one row per file with a status on the right and an
    x to take it out. Drawn on a Canvas (no child widgets), scrolls with the
    mouse wheel. Replaces a multi-select Listbox plus a 'Remove Selected'
    button, which asked the user to select, then remove, then guess."""

    STATES = ("waiting", "working", "done", "failed")

    def __init__(self, parent, *, bg, row, hairline, fg, muted, accent, good,
                 bad, font, small, width, height, on_remove=None, empty_text=""):
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0)
        self.bg_colour, self.row_colour, self.hairline = bg, row, hairline
        self.fg, self.muted, self.accent = fg, muted, accent
        self.good, self.bad = good, bad
        self.font, self.small = font, small
        self.on_remove = on_remove
        self.empty_text = empty_text
        self.row_h = max(30, int(font.metrics("linespace") * 2.0))
        self.rows = []          # dicts: path, state, note
        self.removable = True
        self.bind("<Button-1>", self._click)
        self.bind("<Configure>", lambda _e: self._draw())
        for up, down in (("<Button-4>", "<Button-5>"),):
            self.bind(up, lambda _e: self.yview_scroll(-1, "units"))
            self.bind(down, lambda _e: self.yview_scroll(1, "units"))
        self.bind("<MouseWheel>", lambda e: self.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self._draw()

    # model -----------------------------------------------------------------
    def add(self, path):
        self.rows.append({"path": path, "state": "waiting", "note": ""})
        self._draw()

    def remove(self, path):
        self.rows = [r for r in self.rows if r["path"] != path]
        self._draw()

    def clear(self):
        self.rows = []
        self._draw()

    def set_state(self, path, state, note=""):
        for r in self.rows:
            if r["path"] == path:
                r["state"], r["note"] = state, note
        self._draw()

    def paths(self):
        return [r["path"] for r in self.rows]

    # view ------------------------------------------------------------------
    def _draw(self):
        import os
        self.delete("all")
        w = max(self.winfo_width(), int(self.cget("width")))
        h = int(self.winfo_height()) if self.winfo_height() > 1 else int(self.cget("height"))
        if not self.rows:
            self.create_text(w / 2, h / 2, text=self.empty_text, fill=self.muted,
                             font=self.font, justify="center", width=w - 40)
            self.configure(scrollregion=(0, 0, w, h))
            return
        for i, r in enumerate(self.rows):
            top = i * self.row_h
            mid = top + self.row_h / 2
            if i % 2 == 0:
                self.create_rectangle(0, top, w, top + self.row_h, fill=self.row_colour, outline="")
            name = os.path.basename(r["path"])
            colour = {"waiting": self.fg, "working": self.accent,
                      "done": self.good, "failed": self.bad}[r["state"]]
            label = {"waiting": "", "working": "Working...", "done": "Done",
                     "failed": "Failed"}[r["state"]]
            if r["note"]:
                label = f"{label}  {r['note']}".strip()
            right = w - 14
            if self.removable and r["state"] == "waiting":
                self.create_text(right, mid, text="\u2715", anchor="e", fill=self.muted, font=self.font,
                                 tags=("rm", f"rm{i}"))
                right -= 26
            if label:
                self.create_text(right, mid, text=label, anchor="e", fill=colour, font=self.small)
                right -= self.small.measure(label) + 14
            self.create_text(14, mid, text=self._fit(name, right - 14), anchor="w",
                             fill=colour if r["state"] != "waiting" else self.fg, font=self.font)
        self.configure(scrollregion=(0, 0, w, max(h, len(self.rows) * self.row_h)))

    def _fit(self, text, room):
        if self.font.measure(text) <= room:
            return text
        while text and self.font.measure(text + "...") > room:
            text = text[:-1]
        return text + "..."

    def see(self, path):
        for i, r in enumerate(self.rows):
            if r["path"] == path:
                total = max(1, len(self.rows) * self.row_h)
                self.yview_moveto(max(0.0, (i * self.row_h - self.row_h) / total))

    def _click(self, evt):
        if not self.removable:
            return
        w = max(self.winfo_width(), 1)
        y = self.canvasy(evt.y)
        i = int(y // self.row_h)
        if 0 <= i < len(self.rows) and evt.x > w - 44 and self.rows[i]["state"] == "waiting":
            path = self.rows[i]["path"]
            self.remove(path)
            if self.on_remove:
                self.on_remove(path)
