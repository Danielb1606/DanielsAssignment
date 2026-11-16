"""Daniels Assignment - desktop app (Tkinter)."""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import base64

import daniels_assignment as core
import updater

# ---------------------------------------------------------------- theme
BG = "#14110e"
PANEL = "#201b16"
PANEL_2 = "#2a231c"
BORDER = "#3a3026"
GOLD = "#c9a45c"
TEXT = "#ece4d4"
MUTED = "#9c917f"
GREEN = "#3fae5a"
RED = "#c94a43"
ORANGE = "#d9822b"
YELLOW = "#f2cf86"
GREY = "#4a4339"

TITLE_FONT = ("Segoe UI Semibold", 20)
H_FONT = ("Segoe UI Semibold", 11)
FONT = ("Segoe UI", 10)
SMALL = ("Segoe UI", 9)
PILL_FONT = ("Segoe UI Semibold", 11)

THUMB_W, THUMB_H = 120, 165
KIND_LABEL = core.KIND_NAMES


MODIFIER_KEYSYMS = ("Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Meta_L", "Meta_R",
                    "Caps_Lock", "Win_L", "Win_R", "Super_L", "Super_R")


def combo_from_event(e) -> str:
    """A Tk key event -> 'shift+q' style combination."""
    import sys as _sys
    win = _sys.platform == "win32"
    name = core.vk_to_name(e.keycode) if win else None
    if not name:
        name = e.keysym.lower()
    alt_mask = 0x20000 if win else 0x8
    mods = [m for m, bit in (("ctrl", 0x4), ("alt", alt_mask), ("shift", 0x1)) if e.state & bit]
    return "+".join(mods + [name])


def to_photo(bgr: np.ndarray, max_w: int, max_h: int, upscale=True) -> tk.PhotoImage:
    """OpenCV image -> Tk image. Uses Tk's own PNG support, so no extra
    imaging library is needed (that library caused the old start-up crash)."""
    h, w = bgr.shape[:2]
    s = min(max_w / w, max_h / h)
    if not upscale:
        s = min(s, 1.0)
    img = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))),
                     interpolation=cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", img)
    return tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode("ascii"))


def button(parent, text, cmd, kind="normal", **kw):
    colours = {
        "normal": (PANEL_2, TEXT, "#3a3128"),
        "primary": (GOLD, "#1a140c", "#dcb873"),
        "danger": (RED, "white", "#d9605a"),
        "go": (GREEN, "white", "#4fc46b"),
        "ghost": (PANEL, MUTED, PANEL_2),
    }[kind]
    b = tk.Label(parent, text=text, bg=colours[0], fg=colours[1], font=kw.pop("font", H_FONT),
                 padx=kw.pop("padx", 14), pady=kw.pop("pady", 7), cursor="hand2", **kw)
    b._colours = colours
    b._enabled = True
    b._cmd = cmd

    def enter(_):
        if b._enabled:
            b.config(bg=b._colours[2])

    def leave(_):
        if b._enabled:
            b.config(bg=b._colours[0])

    def click(_):
        if b._enabled and b._cmd:
            b._cmd()

    b.bind("<Enter>", enter)
    b.bind("<Leave>", leave)
    b.bind("<Button-1>", click)
    return b


def set_button(b, text=None, kind=None, enabled=None):
    if kind:
        b._colours = {"normal": (PANEL_2, TEXT, "#3a3128"), "primary": (GOLD, "#1a140c", "#dcb873"),
                      "danger": (RED, "white", "#d9605a"), "ghost": (PANEL, MUTED, PANEL_2),
                      "go": (GREEN, "white", "#4fc46b")}[kind]
    if text is not None:
        b.config(text=text)
    if enabled is not None:
        b._enabled = enabled
    if b._enabled:
        b.config(bg=b._colours[0], fg=b._colours[1], cursor="hand2")
    else:
        b.config(bg=PANEL, fg=GREY, cursor="arrow")


# ================================================================ engine
class Engine:
    """Runs test mode or the controller in a background thread."""

    def __init__(self, log_q: "queue.Queue[str]"):
        self.log_q = log_q
        self.mode = "idle"            # idle | test | run
        self.thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.latest: Optional[Dict[int, core.SlotReading]] = None
        self.crops: Dict[int, np.ndarray] = {}
        self.controller: Optional[core.FlaskController] = None
        self.error: Optional[str] = None
        self.hotkeys = None
        self.reader: Optional[core.FlaskReader] = None
        self.trig_values: Dict[int, Optional[float]] = {}
        self.trig_crops: Dict[int, np.ndarray] = {}
        self.poll_interval = core.POLL_INTERVAL

    def log(self, msg: str) -> None:
        self.log_q.put(msg)

    def _reader(self, cfg) -> core.FlaskReader:
        loc = core.WindowLocator(cfg["window_title"], cfg.get("screen_origin", (0, 0)))
        return core.FlaskReader(cfg, loc, core.ScreenGrabber()), loc

    def _read_and_store(self, reader):
        r = reader.read()
        with self.lock:
            self.latest = r
            self.crops = {k: c.copy() for k, (c, _) in reader.last_crops.items()} if r else {}
            self.trig_values = dict(reader.trigger_values)
            self.trig_crops = {k: c.copy() for k, c in reader.trigger_crops.items()}
        return r

    def start(self, mode: str) -> None:
        self.stop()
        cfg = core.load_config()
        try:
            reader, loc = self._reader(cfg)
        except SystemExit as e:
            self.error = str(e.code)
            return
        self.error = None
        try:
            self._start(mode, cfg, reader, loc)
        except Exception as e:
            import traceback
            tb = traceback.extract_tb(e.__traceback__)[-1]
            self.error = f"Couldn't start: {e.__class__.__name__}: {e} ({tb.name}, line {tb.lineno})"
            self.log(self.error)
            self.stop_flag.set()
            self.controller = None
            self.reader = None
            self.thread = None
            self.mode = "idle"

    def _start(self, mode, cfg, reader, loc) -> None:
        self.stop_flag.clear()
        self.reader = reader
        settings = core.get_settings(cfg)
        self.poll_interval = 1.0 / settings["checks_per_second"]
        if mode == "test":
            target = self._test_loop
            args = (reader,)
        else:
            sender = core.KeySender(cfg, loc)
            if getattr(sender, "fallback_reason", ""):
                self.log(f"Key sending: using the fallback method ({sender.fallback_reason}).")
            self.controller = core.FlaskController(
                lambda: self._read_and_store(reader), sender, log=self.log, settings=settings,
                triggers=[t for t in cfg.get("triggers", []) if t["id"] in reader.trigger_models],
                read_triggers=lambda: reader.trigger_values, press_key=sender.press_combo,
                **core.controller_options(cfg))
            self.controller.on_town = lambda on: self.log_q.put("__TOWN_ON__" if on else "__TOWN_OFF__")
            target = self._run_loop
            args = ()
        self.thread = threading.Thread(target=target, args=args, daemon=True)
        self.mode = mode
        self.thread.start()

    def _close_capture(self, reader) -> None:
        g = getattr(reader, "grabber", None)
        if g is not None and hasattr(g, "close"):
            try:
                g.close()                # release this thread's screen capture
            except Exception:
                pass

    def _test_loop(self, reader) -> None:
        last_err = None
        while not self.stop_flag.is_set():
            try:
                self._read_and_store(reader)
            except Exception as e:      # keep the UI alive
                core.note_error("test loop")
                if str(e) != last_err:
                    last_err = str(e)
                    self.log(f"Read error: {e}")
            time.sleep(self.poll_interval)
        self._close_capture(reader)

    def _run_loop(self) -> None:
        self.log("Started.")
        try:
            self.controller.run()
        except Exception as e:
            core.note_error("run loop")
            self.log(f"Stopped because of an error: {e}")
        self._close_capture(self.reader)
        self.log("Stopped.")

    def stop(self) -> None:
        self.stop_flag.set()
        if self.controller:
            self.controller.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.thread = None
        self.controller = None
        self.reader = None
        self.mode = "idle"
        with self.lock:
            self.latest = None
            self.crops = {}
            self.trig_values = {}
            self.trig_crops = {}


class Loupe:
    """Magnifier for picking colours: shows the pixels around the cursor large,
    with the exact pixel that a click would pick outlined, and its colour."""
    R = 7            # pixels each side of the centre
    Z = 10           # size of each pixel in the magnifier

    def __init__(self, canvas: tk.Canvas, get_img, to_img, active):
        self.c, self.get_img, self.to_img, self.active = canvas, get_img, to_img, active
        self._photo = None
        canvas.bind("<Motion>", self._move, add="+")
        canvas.bind("<Leave>", lambda e: self.hide(), add="+")

    def hide(self):
        self.c.delete("loupe")

    def _move(self, e):
        self.hide()
        if not self.active():
            return
        img = self.get_img()
        ix, iy = self.to_img(e.x, e.y)
        h, w = img.shape[:2]
        if not (0 <= ix < w and 0 <= iy < h):
            return
        R, Z = self.R, self.Z
        pad = cv2.copyMakeBorder(img, R, R, R, R, cv2.BORDER_CONSTANT, value=(12, 10, 8))
        crop = pad[iy:iy + 2 * R + 1, ix:ix + 2 * R + 1]
        size = (2 * R + 1) * Z
        self._photo = to_photo(crop, size, size)
        cw, ch = int(self.c.cget("width")), int(self.c.cget("height"))
        # sit in the corner furthest from the cursor so it never covers what you're aiming at
        x0 = 8 if e.x > cw / 2 else max(8, cw - size - 8)
        y0 = 8 if e.y > ch / 2 else max(8, ch - size - 34)
        self.c.create_rectangle(x0 - 3, y0 - 3, x0 + size + 3, y0 + size + 29, fill="#0c0a08", outline=GOLD,
                                width=2, tags="loupe")
        self.c.create_image(x0, y0, image=self._photo, anchor="nw", tags="loupe")
        cx, cy = x0 + R * Z, y0 + R * Z
        self.c.create_rectangle(cx, cy, cx + Z, cy + Z, outline="#ffffff", width=2, tags="loupe")
        self.c.create_rectangle(cx - 1, cy - 1, cx + Z + 1, cy + Z + 1, outline="#000000", width=1, tags="loupe")
        b, g, r = (int(v) for v in img[iy, ix])
        self.c.create_rectangle(x0, y0 + size + 6, x0 + 20, y0 + size + 24, fill=f"#{r:02x}{g:02x}{b:02x}",
                                outline=BORDER, tags="loupe")
        self.c.create_text(x0 + 28, y0 + size + 15, text="this pixel", anchor="w", fill=TEXT, font=SMALL, tags="loupe")


class HandFix:
    """Optional hand fixes on top of a colour pick: paint pixels in or out with a
    brush (click and hold), or with a box. Fixes are kept per area as an array:
    0 = as picked, 1 = always included, 2 = always left out."""
    TOOLS = (("pick", "Click colour"), ("add", "Paint in"), ("remove", "Paint out"),
             ("box_add", "Box in"), ("box_remove", "Box out"))

    def __init__(self, parent, on_change):
        self.on_change = on_change
        self.edits: Optional[np.ndarray] = None
        self.tool = tk.StringVar(value="pick")
        self.size = tk.IntVar(value=1)
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(row, text="Fix by hand", bg=PANEL, fg=GOLD, font=H_FONT, width=10, anchor="w").pack(side="left")
        for val, label in self.TOOLS:
            tk.Radiobutton(row, text=label, value=val, variable=self.tool, bg=PANEL, fg=TEXT, selectcolor=PANEL_2,
                           activebackground=PANEL, activeforeground=TEXT, font=FONT, highlightthickness=0,
                           indicatoron=False, padx=8, pady=2, relief="flat", offrelief="flat",
                           command=lambda: self.on_change(False)).pack(side="left", padx=(0, 4))
        tk.Label(row, text="Brush", bg=PANEL, fg=MUTED, font=SMALL).pack(side="left", padx=(10, 4))
        for n in (1, 2, 3):
            tk.Radiobutton(row, text=f"{n}px", value=n, variable=self.size, bg=PANEL, fg=TEXT, selectcolor=PANEL_2,
                           activebackground=PANEL, activeforeground=TEXT, font=SMALL, highlightthickness=0,
                           indicatoron=False, padx=5, pady=1, relief="flat", offrelief="flat").pack(side="left")
        self.clear_btn = button(row, "Clear fixes", self.clear, kind="ghost", font=SMALL, pady=2)
        self.clear_btn.pack(side="right")
        self._box0 = None

    @property
    def painting(self) -> bool:
        return self.tool.get() != "pick"

    def ensure(self, shape):
        if self.edits is None or self.edits.shape != tuple(shape[:2]):
            self.edits = np.zeros(shape[:2], np.uint8)

    def brush(self, ix: int, iy: int):
        if self.edits is None:
            return
        val = core.EDIT_ADD if self.tool.get() == "add" else core.EDIT_REMOVE
        r = self.size.get() - 1
        h, w = self.edits.shape
        self.edits[max(0, iy - r):min(h, iy + r + 1), max(0, ix - r):min(w, ix + r + 1)] = val

    def rect(self, a: int, b: int, c: int, d: int):
        if self.edits is None:
            return
        val = core.EDIT_ADD if self.tool.get() == "box_add" else core.EDIT_REMOVE
        h, w = self.edits.shape
        self.edits[max(0, b):min(h, d + 1), max(0, a):min(w, c + 1)] = val

    def clear(self):
        if self.edits is not None:
            self.edits[:] = 0
        self.on_change(True)

    def result(self) -> Optional[np.ndarray]:
        return self.edits.copy() if self.edits is not None and self.edits.any() else None


def shade_removed(crop: np.ndarray, edits: Optional[np.ndarray]) -> np.ndarray:
    """Darken pixels you painted out, so you can see them."""
    if edits is None or edits.shape != crop.shape[:2]:
        return crop
    out = crop.copy()
    rem = edits == core.EDIT_REMOVE
    out[rem] = (out[rem] * 0.3 + np.array([30, 30, 150]) * 0.7).astype(np.uint8)
    return out


# ================================================================ flask card
class FlaskCard(tk.Frame):
    def __init__(self, parent, slot: int, kind: str, on_recal, on_kind=None):
        super().__init__(parent, bg=PANEL, highlightthickness=2, highlightbackground=BORDER)
        self.slot, self.kind = slot, kind
        title = tk.Label(self, text=f"{slot}  ·  {KIND_LABEL.get(kind, kind)}  ▼", bg=PANEL, fg=GOLD, font=H_FONT,
                         cursor="hand2")
        title.pack(pady=(8, 4))
        if on_kind:
            title.bind("<Button-1>", lambda e: on_kind(slot, e.x_root, e.y_root))
        # A Label's width/height are in text characters unless it shows an
        # image, so it always shows one: a blank placeholder until calibrated.
        self._blank = tk.PhotoImage(width=THUMB_W, height=THUMB_H)
        self.canvas = tk.Label(self, bg="#0c0a08", image=self._blank, borderwidth=0)
        self.canvas.pack(padx=10)
        self.pill = tk.Label(self, text="NOT CALIBRATED", bg=GREY, fg=TEXT, font=PILL_FONT, pady=4)
        self.pill.pack(fill="x", padx=10, pady=(8, 2))
        self.detail = tk.Label(self, text="", bg=PANEL, fg=MUTED, font=SMALL)
        self.detail.pack()
        self.run_pill = None
        if kind in core.BAR_KINDS:
            self.run_pill = tk.Label(self, text="not running", bg=PANEL, fg=MUTED, font=PILL_FONT, pady=3,
                                     highlightthickness=1, highlightbackground=GREY)
            self.run_pill.pack(fill="x", padx=10, pady=(4, 0))
        button(self, "Recalibrate", lambda: on_recal(slot), kind="ghost", font=SMALL, pady=3).pack(pady=8)
        self._photo = None

    def show_image(self, bgr: Optional[np.ndarray]) -> None:
        if bgr is None or bgr.size == 0:
            self.canvas.config(image=self._blank)
            return
        # centre the flask on a fixed-size tile so every card is the same width
        h, w = bgr.shape[:2]
        sc = min(THUMB_W / w, THUMB_H / h)
        small = cv2.resize(bgr, (max(1, int(w * sc)), max(1, int(h * sc))),
                           interpolation=cv2.INTER_NEAREST if sc > 1 else cv2.INTER_AREA)
        tile = np.full((THUMB_H, THUMB_W, 3), (8, 10, 12), np.uint8)
        y, x = (THUMB_H - small.shape[0]) // 2, (THUMB_W - small.shape[1]) // 2
        tile[y:y + small.shape[0], x:x + small.shape[1]] = small
        self._photo = to_photo(tile, THUMB_W, THUMB_H)
        self.canvas.config(image=self._photo)

    def set_state(self, label: str, colour: str, detail: str = "", running: Optional[bool] = None,
                  left: Optional[float] = None) -> None:
        self.pill.config(text=label, bg=colour)
        self.config(highlightbackground=colour if colour != GREY else BORDER)
        self.detail.config(text=detail)
        if self.run_pill is not None:
            if running is None:
                self.run_pill.config(text="—", bg=PANEL, fg=MUTED, highlightbackground=GREY)
            elif running:
                txt = "RUNNING" if left is None else f"RUNNING · {left:.0%} left"
                self.run_pill.config(text=txt, bg=YELLOW, fg="#1a140c", highlightbackground=YELLOW)
            else:
                self.run_pill.config(text="not running", bg=PANEL, fg=MUTED, highlightbackground=GREY)


# ================================================================ per-flask settings
class FlaskSettingsDialog(tk.Toplevel):
    """Own delay range, globe threshold and (utility) refresh % for one flask."""

    def __init__(self, parent, cfg: dict, slot: int):
        super().__init__(parent, bg=BG)
        self.cfg, self.slot = cfg, slot
        self.sc = core.slot_cfg(cfg, slot)
        self.saved = ""
        self.title(f"{core.APP} | Flask {slot}")
        self.transient(parent)
        self.resizable(False, False)
        st = core.get_settings(cfg)
        kind = self.sc["kind"]
        tk.Label(self, text=f"Flask {slot}  ·  {KIND_LABEL.get(kind, kind)}", bg=BG, fg=GOLD, font=TITLE_FONT).pack(
            anchor="w", padx=20, pady=(16, 8))
        self.vars = {}

        def section(title, sub):
            box = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
            box.pack(fill="x", padx=20, pady=(0, 10))
            tk.Label(box, text=title, bg=PANEL, fg=GOLD, font=H_FONT).pack(anchor="w", padx=12, pady=(10, 0))
            tk.Label(box, text=sub, bg=PANEL, fg=MUTED, font=SMALL, justify="left", wraplength=420).pack(anchor="w", padx=12)
            inner = tk.Frame(box, bg=PANEL)
            inner.pack(fill="x", padx=12, pady=(6, 12))
            return inner

        def field(parent, key, label, value, unit):
            row = tk.Frame(parent, bg=PANEL)
            row.pack(anchor="w", pady=2)
            tk.Label(row, text=label, bg=PANEL, fg=TEXT, font=FONT, width=26, anchor="w").pack(side="left")
            v = tk.StringVar(value=value)
            tk.Entry(row, textvariable=v, width=7, bg=PANEL_2, fg=TEXT, insertbackground=TEXT, relief="flat",
                     font=H_FONT, justify="center", highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=GOLD).pack(side="left", ipady=2)
            tk.Label(row, text=unit, bg=PANEL, fg=MUTED, font=FONT).pack(side="left", padx=6)
            self.vars[key] = v

        b = section("Delay before pressing", f"A random wait in this range before this flask is pressed. "
                    f"Leave it at {st['delay_min_ms']}-{st['delay_max_ms']} ms to use the main setting.")
        field(b, "dmin", "Shortest", self.sc.get("delay_min_ms", st["delay_min_ms"]), "ms")
        field(b, "dmax", "Longest", self.sc.get("delay_max_ms", st["delay_max_ms"]), "ms")
        self.global_delay = (st["delay_min_ms"], st["delay_max_ms"])

        g = self.sc.get("globe") or {}
        if g.get("box"):
            b = section("Globe", "Pressed whenever the linked globe drops below this much of its colour "
                        "(as well as its normal rule). The card shows the globe % live in Test mode.")
            field(b, "below", "Press when the globe is below", g.get("below", core.GLOBE_BELOW), "%")
            field(b, "again", "If still low, press again after", g.get("again_ms", core.GLOBE_AGAIN_MS) / 1000, "seconds")
        else:
            b = section("Globe", "Not linked. Use “Link to a globe…” on the flask's name menu to press "
                        "this flask when your life or mana globe gets low.")

        if kind == "utility":
            b = section("Utility", "Used when it's full and less than this much of its effect is left.")
            field(b, "refresh", "Use again below", self.sc.get("refresh_below", core.UTILITY_REFRESH_BELOW), "% left")

        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=20, pady=(4, 16))
        self.err = tk.Label(bar, text="", bg=BG, fg=RED, font=SMALL)
        self.err.pack(side="left")
        button(bar, "Save", self.save, kind="primary").pack(side="right")
        button(bar, "Cancel", self.destroy, kind="ghost").pack(side="right", padx=8)
        self.grab_set()

    def _num(self, key, lo, hi, name):
        try:
            v = float(self.vars[key].get().strip())
        except ValueError:
            raise ValueError(f"{name}: enter a number")
        if not lo <= v <= hi:
            raise ValueError(f"{name} must be {lo:g}-{hi:g}")
        return v

    def save(self):
        notes = []
        try:
            dmin, dmax = self._num("dmin", 0, 5000, "Shortest delay"), self._num("dmax", 0, 5000, "Longest delay")
            if dmin > dmax:
                raise ValueError("Shortest delay can't be more than the longest")
            below = again = refresh = None
            if "below" in self.vars:
                below = self._num("below", 1, 99, "Globe %")
                again = self._num("again", 0.2, 60, "Press again after")
            if "refresh" in self.vars:
                refresh = self._num("refresh", 1, 100, "Use again below")
        except ValueError as e:
            self.err.config(text=str(e))
            return
        if (int(dmin), int(dmax)) != tuple(int(x) for x in self.global_delay):
            self.sc["delay_min_ms"], self.sc["delay_max_ms"] = int(dmin), int(dmax)
            notes.append(f"delay {int(dmin)}-{int(dmax)} ms")
        else:
            self.sc.pop("delay_min_ms", None)
            self.sc.pop("delay_max_ms", None)
        if below is not None:
            self.sc["globe"]["below"] = below
            self.sc["globe"]["again_ms"] = int(again * 1000)
            notes.append(f"globe below {below:g}%")
        if refresh is not None:
            self.sc["refresh_below"] = refresh
            notes.append(f"used again below {refresh:g}% left")
        self.saved = f"Flask {self.slot} saved" + (": " + ", ".join(notes) if notes else "") + "."
        self.destroy()


# ================================================================ calibration dialog
class CalibrationDialog(tk.Toplevel):
    """Step-by-step: box each flask, pick its liquid with a click + Similarity
    slider, let the app find quicksilvers 4 and 5, then the timer strips."""

    def __init__(self, parent, img: np.ndarray, cfg: dict, slots: List[int], preset: Optional[dict] = None):
        super().__init__(parent, bg=BG)
        self.title(f"{core.APP} | Calibration")
        self.preset = preset or {}
        self.transient(parent)
        self.grab_set()
        self.img, self.cfg, self.slots = img, cfg, slots
        self.result = None

        self.steps: List[Tuple[str, int]] = []
        if sorted(slots) == [1, 2, 3, 4, 5]:
            for s in (1, 2, 3):
                self.steps += [("box", s), ("liquid", s)]
            self.steps += [("find", 4)]
        else:
            for s in slots:
                self.steps += [("box", s), ("liquid", s)]
        self.steps += [("bar", s) for s in slots if core.slot_kinds(cfg).get(s) in core.BAR_KINDS]
        self.i = 0
        self.boxes: Dict[int, List[int]] = {}
        self.points: Dict[int, List[Tuple[int, int]]] = {}
        self.liq: Dict[int, dict] = {}
        for s in slots:
            sc = core.slot_cfg(cfg, s)
            keep_old = len(slots) == 1 and sc.get("similarity")
            self.liq[s] = {"similarity": int(sc.get("similarity", core.DEFAULT_SIMILARITY)) if keep_old
                           else core.DEFAULT_SIMILARITY,
                           "mode": sc.get("pick_mode", "shape") if keep_old else "shape"}
            if s in self.preset:                       # start from the auto-detected guess
                pr = self.preset[s]
                self.liq[s] = {"similarity": int(pr.get("similarity", core.DEFAULT_SIMILARITY)),
                               "mode": pr.get("mode", "shape")}
                if pr.get("points"):
                    self.points[s] = [tuple(p) for p in pr["points"]]
        self.bars: Dict[int, List[int]] = {}
        self.found: List[List[int]] = []

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.max_w, self.max_h = min(1280, sw - 120), min(640, sh - 320)

        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=16, pady=(14, 6))
        self.step_lbl = tk.Label(top, text="", bg=BG, fg=GOLD, font=H_FONT)
        self.step_lbl.pack(anchor="w")
        self.instr = tk.Label(top, text="", bg=BG, fg=TEXT, font=("Segoe UI", 12), justify="left",
                              wraplength=760)
        self.instr.pack(anchor="w", pady=(2, 0))

        self.canvas = tk.Canvas(self, bg="#0c0a08", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=16, pady=6)
        self.canvas.bind("<ButtonPress-1>", self._down)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._up)
        Loupe(self.canvas, lambda: self.img, self._to_img,
              lambda: self.steps[self.i][0] == "liquid")

        # liquid controls (only on liquid steps)
        self.liq_frame = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        r1 = tk.Frame(self.liq_frame, bg=PANEL)
        r1.pack(fill="x", padx=12, pady=(8, 2))
        tk.Label(r1, text="Similarity", bg=PANEL, fg=GOLD, font=H_FONT, width=10, anchor="w").pack(side="left")
        tk.Label(r1, text="strict", bg=PANEL, fg=MUTED, font=SMALL).pack(side="left")
        self.sim_var = tk.IntVar(value=core.DEFAULT_SIMILARITY)
        st = ttk.Style(self)
        st.configure("Gold.Horizontal.TScale", troughcolor=PANEL_2, background=GOLD, bordercolor=BORDER,
                     lightcolor=GOLD, darkcolor="#a8853f", sliderthickness=18)
        self.slider = ttk.Scale(r1, from_=1, to=100, orient="horizontal", length=340,
                                style="Gold.Horizontal.TScale",
                                command=lambda v: self._slider_moved(v))
        self.slider.pack(side="left", padx=8, pady=4)
        tk.Label(r1, text="loose", bg=PANEL, fg=MUTED, font=SMALL).pack(side="left")
        self.sim_lbl = tk.Label(r1, text="", bg=PANEL, fg=TEXT, font=H_FONT, width=4)
        self.sim_lbl.pack(side="left", padx=(10, 0))
        r2 = tk.Frame(self.liq_frame, bg=PANEL)
        r2.pack(fill="x", padx=12, pady=(2, 8))
        tk.Label(r2, text="Pick from", bg=PANEL, fg=GOLD, font=H_FONT, width=10, anchor="w").pack(side="left")
        self.mode_var = tk.StringVar(value="shape")
        for val, label in (("shape", "Inside flask shape"), ("box", "Anywhere in box")):
            tk.Radiobutton(r2, text=label, value=val, variable=self.mode_var, bg=PANEL, fg=TEXT,
                           selectcolor=PANEL_2, activebackground=PANEL, activeforeground=TEXT, font=FONT,
                           highlightthickness=0, command=self._on_liquid_setting).pack(side="left", padx=(0, 14))
        self.undo_btn = button(r2, "Undo click", self._undo_click, kind="ghost", font=SMALL, pady=3)
        self.undo_btn.pack(side="right")
        self.fix = HandFix(self.liq_frame, self._fix_changed)
        self.edit_store: Dict[int, Optional[np.ndarray]] = {}

        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=16, pady=(4, 14), side="bottom")
        button(bar, "Cancel", self.destroy, kind="ghost").pack(side="left")
        self.back_btn = button(bar, "◀  Back", self._back)
        self.back_btn.pack(side="left", padx=8)
        self.manual_btn = button(bar, "Draw them myself", self._find_manual)
        self.next_btn = button(bar, "Next  ▶", self._accept_default, kind="primary")
        self.next_btn.pack(side="right")

        self._drag_start = None
        self._rect = None
        self._pending_box = None
        self._render()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.focus_set()

    # -- view helpers -------------------------------------------------------
    def _set_view(self, region: Tuple[int, int, int, int], overlay: Optional[np.ndarray] = None,
                  removed: Optional[np.ndarray] = None) -> None:
        """Show part of the screenshot (x, y, w, h) scaled to the canvas,
        optionally with a mask shaded pink (and painted-out pixels darkened)."""
        x, y, w, h = region
        crop = shade_removed(self.img[y:y + h, x:x + w], removed)
        if overlay is not None and overlay.any():
            crop = crop.copy()
            crop[overlay] = (crop[overlay] * 0.35 + np.array([255, 0, 255]) * 0.65).astype(np.uint8)
        self.view = region
        self.scale = min(self.max_w / w, self.max_h / h, 8.0)
        self._photo = to_photo(crop, int(w * self.scale), int(h * self.scale))
        self.canvas.config(width=self._photo.width(), height=self._photo.height())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")

    def _to_img(self, cx, cy) -> Tuple[int, int]:
        return int(self.view[0] + cx / self.scale), int(self.view[1] + cy / self.scale)

    def _to_canvas(self, box) -> Tuple[float, float, float, float]:
        x, y, w, h = box
        s = self.scale
        return ((x - self.view[0]) * s, (y - self.view[1]) * s,
                (x + w - self.view[0]) * s, (y + h - self.view[1]) * s)

    def _draw_box(self, box, colour, dash=None, width=2, tag=None, label=None):
        x0, y0, x1, y1 = self._to_canvas(box)
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=colour, width=width, dash=dash, tags=tag)
        if label:
            self.canvas.create_text(x0 + 4, y0 + 2, text=label, anchor="nw", fill=colour,
                                    font=("Segoe UI Semibold", 11), tags=tag)

    # -- steps ----------------------------------------------------------------
    def _render(self) -> None:
        kind, slot = self.steps[self.i]
        name = KIND_LABEL[core.slot_kinds(self.cfg)[slot]]
        what = "Flasks 4 and 5 (Quicksilver)" if kind == "find" else f"Flask {slot} ({name})"
        self.step_lbl.config(text=f"Step {self.i + 1} of {len(self.steps)}  ·  {what}")
        set_button(self.back_btn, enabled=self.i > 0)
        if kind == "liquid":
            self.liq_frame.pack(fill="x", padx=16, pady=(0, 4))
        else:
            self.liq_frame.pack_forget()
        if kind == "find":
            self.manual_btn.pack(side="right", padx=8)
        else:
            self.manual_btn.pack_forget()
        self._pending_box = None
        old = core.slot_cfg(self.cfg, slot)
        last = self.i == len(self.steps) - 1

        if kind == "box":
            self._set_view((0, 0, self.img.shape[1], self.img.shape[0]))
            for s, b in self.boxes.items():
                self._draw_box(b, GREEN, label=str(s))
            guess = self.preset.get(slot, {}).get("box")
            existing = self.boxes.get(slot) or guess or (old["box"] if old["box"] and self._fits(old["box"]) else None)
            self._pending_box = existing
            if existing:
                self._draw_box(existing, ORANGE, dash=(4, 3), tag="sel")
            self.instr.config(text=f"Drag a tight box around flask {slot}."
                              + ("  The orange box is the current one - click Next to keep it." if existing else ""))
            set_button(self.next_btn, text="Next  ▶", enabled=existing is not None)
        elif kind == "liquid":
            self.sim_var.set(self.liq[slot]["similarity"])
            self.slider.set(self.liq[slot]["similarity"])
            self.sim_lbl.config(text=str(self.liq[slot]["similarity"]))
            self.mode_var.set(self.liq[slot]["mode"])
            self._show_liquid(slot)
        elif kind == "find":
            self._set_view((0, 0, self.img.shape[1], self.img.shape[0]))
            for s, b in self.boxes.items():
                self._draw_box(b, GREEN, label=str(s))
            self.found = core.find_similar(self.img, self.boxes[3], count=2)
            if len(self.found) == 2:
                for n, b in zip((4, 5), self.found):
                    self._draw_box(b, ORANGE, label=str(n))
                self.instr.config(text="Found flasks 4 and 5 (orange). They'll use the same liquid settings as "
                                       "flask 3.\nClick Next if the boxes are right, or Draw them myself.")
                set_button(self.next_btn, text="Next  ▶", enabled=True)
            else:
                self.instr.config(text="Couldn't find flasks 4 and 5 automatically - click Draw them myself.")
                set_button(self.next_btn, enabled=False)
        else:  # bar
            x, y, w, h = self.boxes[slot]
            pad = int(h * 0.6)
            vx, vy = max(0, x - w), max(0, y - 10)
            vw = min(self.img.shape[1] - vx, w * 3)
            vh = min(self.img.shape[0] - vy, h + pad + 10)
            self._set_view((vx, vy, vw, vh))
            self._draw_box(self.boxes[slot], GREEN)
            prev = self.bars.get(slot) or self.preset.get(slot, {}).get("bar_box") or \
                (old["bar_box"] if (old["bar_box"] and len(self.slots) == 1) else None)
            if prev is None and slot != 3 and 3 in self.bars and 3 in self.boxes:
                # same position under the flask as flask 3's bar
                b3, bx3 = self.bars[3], self.boxes[3]
                prev = [x + (b3[0] - bx3[0]), y + (b3[1] - bx3[1]), b3[2], b3[3]]
            self._pending_box = prev or core.default_bar_box(self.boxes[slot])
            self._draw_box(self._pending_box, ORANGE, tag="sel")
            self.instr.config(text=f"The orange box is where flask {slot}'s timer bar is read.\n"
                                   "Click Next to use it, or drag a new box where the bar appears.")
            set_button(self.next_btn, text="Next  ▶", enabled=True)
        if last and kind in ("bar", "liquid"):
            set_button(self.next_btn, text="Finish  ✓")

    def _edits(self, slot):
        """Hand fixes for this flask (sized to its box)."""
        x, y, w, h = self.boxes[slot]
        e = self.edit_store.get(slot)
        if e is None or e.shape != (h, w):
            e = None
            old = core.slot_cfg(self.cfg, slot).get("box")
            if old and list(old) == list(self.boxes[slot]):     # same box as last time: keep its fixes
                e = core.load_edits(f"slot{slot}", (h, w))
            if e is None and slot in self.preset and self.preset[slot].get("edits") is not None:
                e = self.preset[slot]["edits"]
            e = e.copy() if e is not None and e.shape == (h, w) else np.zeros((h, w), np.uint8)
            self.edit_store[slot] = e
        return e

    def _liquid(self, slot):
        x, y, w, h = self.boxes[slot]
        rel = [(px - x, py - y) for px, py in self.points.get(slot, [])]
        st = self.liq[slot]
        return core.apply_edits(core.liquid_selection(self.img[y:y + h, x:x + w], rel, st["similarity"], st["mode"]),
                                self._edits(slot))

    def _fix_changed(self, redraw=True):
        kind, slot = self.steps[self.i]
        if kind == "liquid" and redraw:
            self._show_liquid(slot)

    def _slider_moved(self, v):
        val = int(round(float(v)))
        if val == self.sim_var.get() and self.sim_lbl.cget("text") == str(val):
            return
        self.sim_var.set(val)
        self.sim_lbl.config(text=str(val))
        self._on_liquid_setting()

    def _on_liquid_setting(self):
        kind, slot = self.steps[self.i]
        if kind != "liquid":
            return
        self.liq[slot] = {"similarity": int(self.sim_var.get()), "mode": self.mode_var.get()}
        self._show_liquid(slot)

    def _show_liquid(self, slot):
        name = KIND_LABEL[core.slot_kinds(self.cfg)[slot]].lower()
        mask = self._liquid(slot)
        self.fix.edits = self._edits(slot)
        self._set_view(tuple(self.boxes[slot]), overlay=mask, removed=self.fix.edits)
        for px, py in self.points.get(slot, []):
            cx, cy = (px - self.view[0]) * self.scale, (py - self.view[1]) * self.scale
            self.canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, outline=YELLOW, width=2)
        share = float(mask.mean())
        if not self.points.get(slot):
            text = f"Click on the {name} liquid inside flask {slot}. The liquid it picks turns pink."
        elif share < 0.03:
            text = ("Hardly anything is pink. Move Similarity to the right, or click on the liquid itself "
                    "(Undo click removes a wrong click).")
        else:
            text = (f"Pink = the liquid it will watch ({share:.0%} of the box). Slide Similarity until the pink "
                    "covers the liquid - including its top edge - but not the glass, label or decorations. "
                    "Click any patch it still misses. To fix single pixels, use Fix by hand: Paint in / Paint out "
                    "(click and hold), or Box in / Box out (drag a box).")
        self.instr.config(text=text)
        set_button(self.undo_btn, enabled=bool(self.points.get(slot)))
        set_button(self.next_btn, enabled=share >= 0.03)

    def _undo_click(self):
        kind, slot = self.steps[self.i]
        if kind == "liquid" and self.points.get(slot):
            self.points[slot].pop()
            self._show_liquid(slot)

    def _find_manual(self):
        kind, _ = self.steps[self.i]
        if kind != "find":
            return
        self.steps[self.i:self.i + 1] = [("box", 4), ("liquid", 4), ("box", 5), ("liquid", 5)]
        for s in (4, 5):
            self.liq[s] = dict(self.liq.get(3, {"similarity": core.DEFAULT_SIMILARITY, "mode": "shape"}))
        self._render()

    def _fits(self, box) -> bool:
        x, y, w, h = box
        return x >= 0 and y >= 0 and x + w <= self.img.shape[1] and y + h <= self.img.shape[0]

    def _down(self, e):
        kind, slot = self.steps[self.i]
        if kind == "find":
            return
        if kind == "liquid":
            if not self.fix.painting:
                self.points.setdefault(slot, []).append(self._to_img(e.x, e.y))
                self._show_liquid(slot)
                return
            if self.fix.tool.get() in ("add", "remove"):
                self._paint_at(slot, e)
                return
        self._drag_start = (e.x, e.y)
        self.canvas.delete("sel")
        self._rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline=YELLOW, width=2, tags="sel")

    def _paint_at(self, slot, e):
        ix, iy = self._to_img(e.x, e.y)
        x, y, w, h = self.boxes[slot]
        self.fix.edits = self._edits(slot)
        self.fix.brush(ix - x, iy - y)
        self._show_liquid(slot)

    def _drag(self, e):
        kind, slot = self.steps[self.i]
        if kind == "liquid" and self.fix.tool.get() in ("add", "remove"):
            self._paint_at(slot, e)
            return
        if self._drag_start and self._rect:
            x0, y0 = self._drag_start
            self.canvas.coords(self._rect, x0, y0, e.x, e.y)

    def _up(self, e):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        self._drag_start = None
        a, b = self._to_img(min(x0, e.x), min(y0, e.y))
        c, d = self._to_img(max(x0, e.x), max(y0, e.y))
        kind, slot = self.steps[self.i]
        if kind == "liquid":                        # Box in / Box out
            x, y, w, h = self.boxes[slot]
            self.fix.edits = self._edits(slot)
            self.fix.rect(a - x, b - y, c - x, d - y)
            self._show_liquid(slot)
            return
        if c - a < 4 or d - b < 4:
            self.canvas.delete("sel")
            return
        self._pending_box = [a, b, c - a, d - b]
        set_button(self.next_btn, enabled=True)

    def _accept_default(self):
        kind, slot = self.steps[self.i]
        if kind == "box":
            if not self._pending_box:
                return
            self.boxes[slot] = self._pending_box
        elif kind == "liquid":
            if float(self._liquid(slot).mean()) < 0.03:
                return
        elif kind == "find":
            if len(self.found) != 2:
                return
            b3 = self.boxes[3]
            for n, b in zip((4, 5), self.found):
                self.boxes[n] = b
                self.points[n] = [(px - b3[0] + b[0], py - b3[1] + b[1]) for px, py in self.points.get(3, [])]
                self.liq[n] = dict(self.liq[3])
        else:
            self.bars[slot] = self._pending_box
        if self.i == len(self.steps) - 1:
            for sl in self.boxes:
                if sl in self.liq:
                    e = self.edit_store.get(sl)
                    self.liq[sl]["edits"] = e.copy() if e is not None and e.any() and \
                        e.shape == (self.boxes[sl][3], self.boxes[sl][2]) else None
            self.result = (self.boxes, self.bars, self.points, self.liq)
            self.destroy()
            return
        self.i += 1
        self._render()

    def _back(self):
        if self.i > 0:
            self.i -= 1
            self._render()


# ================================================================ review (auto-detected)
class ReviewDialog(tk.Toplevel):
    """Shows what auto-detection found for all five flasks. Finish if it's
    right; click a flask (or its Fix button) to correct just that one."""

    def __init__(self, parent, img: np.ndarray, cfg: dict, detected: Dict[int, dict]):
        super().__init__(parent, bg=BG)
        self.title(f"{core.APP} | Calibration")
        self.transient(parent)
        self.grab_set()
        self.img, self.cfg = img, cfg
        self.entries: Dict[int, dict] = {k: dict(v) for k, v in detected.items()}
        self.result = None
        self.manual = False

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.max_w, self.max_h = min(1280, sw - 120), min(600, sh - 330)

        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(top, text="Check the flasks", bg=BG, fg=GOLD, font=TITLE_FONT).pack(anchor="w")
        self.instr = tk.Label(top, text="", bg=BG, fg=TEXT, font=("Segoe UI", 12), justify="left", wraplength=900)
        self.instr.pack(anchor="w", pady=(2, 0))

        self.canvas = tk.Canvas(self, bg="#0c0a08", highlightthickness=0, cursor="hand2")
        self.canvas.pack(padx=16, pady=6)
        self.canvas.bind("<ButtonPress-1>", self._click)

        self.row = tk.Frame(self, bg=BG)
        self.row.pack(fill="x", padx=16, pady=(2, 4))
        self.fix_btns: Dict[int, tk.Label] = {}
        for slot in range(1, 6):
            b = button(self.row, "", lambda s=slot: self.fix(s), font=FONT, padx=10, pady=4)
            b.pack(side="left", padx=(0, 8))
            self.fix_btns[slot] = b

        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=16, pady=(6, 14))
        button(bar, "Cancel", self.destroy, kind="ghost").pack(side="left")
        button(bar, "Do it step by step instead", self._manual, kind="ghost").pack(side="left", padx=8)
        self.finish_btn = button(bar, "Looks right - Finish  ✓", self._finish, kind="primary")
        self.finish_btn.pack(side="right")
        self._render()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.focus_set()

    def _status(self, slot):
        e = self.entries.get(slot)
        if not e:
            return "missing"
        return "unsure" if e.get("unsure") else "ok"

    def _render(self):
        h, w = self.img.shape[:2]
        boxes = [e["box"] for e in self.entries.values()] + \
                [e["bar_box"] for e in self.entries.values() if e.get("bar_box")]
        if boxes and len(self.entries) == 5:   # zoom to the flask belt with some margin
            x0 = max(0, min(b[0] for b in boxes) - 40); y0 = max(0, min(b[1] for b in boxes) - 30)
            x1 = min(w, max(b[0] + b[2] for b in boxes) + 40); y1 = min(h, max(b[1] + b[3] for b in boxes) + 30)
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        self.view = (x0, y0, x1 - x0, y1 - y0)
        view = self.img[y0:y1, x0:x1].copy()
        for slot, e in self.entries.items():
            bx, by, bw, bh = e["box"]
            rel = [(px - bx, py - by) for px, py in e.get("points", [])]
            m = core.liquid_selection(self.img[by:by + bh, bx:bx + bw], rel, e["similarity"], e["mode"])
            sub = view[by - y0:by - y0 + bh, bx - x0:bx - x0 + bw]
            mm = m[:sub.shape[0], :sub.shape[1]]
            sub[mm] = (sub[mm] * 0.35 + np.array([255, 0, 255]) * 0.65).astype(np.uint8)
        self.scale = min(self.max_w / view.shape[1], self.max_h / view.shape[0], 6.0)
        self._photo = to_photo(view, int(view.shape[1] * self.scale), int(view.shape[0] * self.scale))
        self.canvas.config(width=self._photo.width(), height=self._photo.height())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        sc = self.scale
        for slot, e in self.entries.items():
            col = ORANGE if e.get("unsure") else GREEN
            bx, by, bw, bh = e["box"]
            self.canvas.create_rectangle((bx - x0) * sc, (by - y0) * sc, (bx + bw - x0) * sc, (by + bh - y0) * sc,
                                         outline=col, width=3 if e.get("unsure") else 2)
            self.canvas.create_text((bx - x0) * sc + 6, (by - y0) * sc + 4, anchor="nw", text=str(slot),
                                    fill=col, font=("Segoe UI Semibold", 13))
            if e.get("bar_box"):
                bx, by, bw, bh = e["bar_box"]
                self.canvas.create_rectangle((bx - x0) * sc, (by - y0) * sc, (bx + bw - x0) * sc,
                                             (by + bh - y0) * sc, outline=YELLOW, width=1, dash=(3, 2))

        missing = [s for s in range(1, 6) if self._status(s) == "missing"]
        unsure = [s for s in range(1, 6) if self._status(s) == "unsure"]
        for slot, b in self.fix_btns.items():
            st = self._status(slot)
            label = {"ok": f"✓  Flask {slot}", "unsure": f"?  Check flask {slot}",
                     "missing": f"+  Draw flask {slot}"}[st]
            b.config(text=label, fg={"ok": GREEN, "unsure": ORANGE, "missing": RED}[st])
        if missing:
            text = (f"Couldn't find flask{'s' if len(missing) > 1 else ''} {', '.join(map(str, missing))} - "
                    "click its button below to draw it.")
        elif unsure:
            text = (f"Found all five. Please check flask{'s' if len(unsure) > 1 else ''} "
                    f"{', '.join(map(str, unsure))} (orange). Pink = the liquid it will watch; dashed yellow = "
                    "where it reads the timer bar. Click any flask to fix it.")
        else:
            text = ("Found all five. Pink = the liquid it will watch; dashed yellow = where it reads the timer bar. "
                    "If it all looks right, click Finish. Otherwise click the flask to fix it.")
        self.instr.config(text=text)
        set_button(self.finish_btn, enabled=not missing)

    def _click(self, e):
        x0, y0 = self.view[0], self.view[1]
        ix, iy = x0 + e.x / self.scale, y0 + e.y / self.scale
        for slot, en in self.entries.items():
            bx, by, bw, bh = en["box"]
            if bx <= ix <= bx + bw and by <= iy <= by + bh:
                self.fix(slot)
                return

    def fix(self, slot: int):
        preset = {slot: self.entries[slot]} if slot in self.entries else {}
        dlg = CalibrationDialog(self, self.img, self.cfg, [slot], preset=preset)
        self.wait_window(dlg)
        if not dlg.result:
            return
        boxes, bars, points, liq = dlg.result
        self.entries[slot] = {"box": boxes[slot], "points": points.get(slot, []),
                              "similarity": liq[slot]["similarity"], "mode": liq[slot]["mode"],
                              "edits": liq[slot].get("edits"),
                              "bar_box": bars.get(slot) if core.slot_kinds(self.cfg).get(slot) in core.BAR_KINDS else None,
                              "unsure": False, "source": "you"}
        self._render()

    def _manual(self):
        self.manual = True
        self.destroy()

    def _finish(self):
        if any(self._status(s) == "missing" for s in range(1, 6)):
            return
        boxes = {s: e["box"] for s, e in self.entries.items()}
        bars = {s: e["bar_box"] for s, e in self.entries.items() if core.slot_kinds(self.cfg).get(s) in core.BAR_KINDS and e.get("bar_box")}
        points = {s: e.get("points", []) for s, e in self.entries.items()}
        liq = {s: {"similarity": e["similarity"], "mode": e["mode"], "edits": e.get("edits")}
               for s, e in self.entries.items()}
        self.result = (boxes, bars, points, liq)
        self.destroy()


# ================================================================ triggers
class TriggerEditDialog(tk.Toplevel):
    """Every setting for one trigger."""

    def __init__(self, parent, trig: dict, is_new: bool):
        super().__init__(parent, bg=BG)
        self.title(f"{core.APP} | Trigger")
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)
        self.t = dict(trig)
        self.result = None          # dict, or "delete"
        tk.Label(self, text="New trigger" if is_new else "Edit trigger", bg=BG, fg=GOLD,
                 font=TITLE_FONT).pack(anchor="w", padx=20, pady=(16, 0))
        tk.Label(self, text=("Set the key and timing here. After Save you'll draw a box around the area and "
                              "click the colour to watch." if is_new else
                              "To change the area or colour, use the trigger's Calibrate button."),
                 bg=BG, fg=MUTED, font=FONT, wraplength=560, justify="left").pack(anchor="w", padx=20)
        f = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        f.pack(fill="x", padx=20, pady=12)
        self.vars = {}

        def row(label, help_=""):
            r = tk.Frame(f, bg=PANEL)
            r.pack(fill="x", padx=12, pady=4)
            tk.Label(r, text=label, bg=PANEL, fg=TEXT, font=FONT, width=30, anchor="w").pack(side="left")
            return r

        def entry(r, key, width=7):
            v = tk.StringVar(value=str(self.t[key]))
            tk.Entry(r, textvariable=v, width=width, bg=PANEL_2, fg=TEXT, insertbackground=TEXT, relief="flat",
                     font=FONT, justify="right" if width < 10 else "left", highlightthickness=1,
                     highlightbackground=BORDER, highlightcolor=GOLD).pack(side="left", ipady=2)
            self.vars[key] = v

        entry(row("Name"), "name", width=24)
        r = row("Key to press")
        self.key_val = self.t["key"]
        self.key_lbl = tk.Label(r, text=core.hotkey_label(self.key_val), bg=PANEL_2, fg=TEXT,
                                font=("Segoe UI Semibold", 10), width=16, pady=3)
        self.key_lbl.pack(side="left")
        button(r, "Change", self._record, font=SMALL, pady=3).pack(side="left", padx=8)
        self._recording = False
        self.bind("<KeyPress>", self._on_key)
        r = row("Start pressing when below (%)", )
        entry(r, "fire_below")
        tk.Label(r, text="of the colour still showing", bg=PANEL, fg=MUTED, font=SMALL).pack(side="left", padx=8)
        r = row("Stop when back above (%)")
        entry(r, "rearm_above")
        tk.Label(r, text="re-arms for the next time", bg=PANEL, fg=MUTED, font=SMALL).pack(side="left", padx=8)
        r = row("Keep pressing until it's back")
        self.repeat = tk.BooleanVar(value=bool(self.t.get("repeat", True)))
        tk.Checkbutton(r, variable=self.repeat, bg=PANEL, selectcolor=PANEL_2, activebackground=PANEL,
                       highlightthickness=0).pack(side="left")
        tk.Label(r, text="(off = press once each time)", bg=PANEL, fg=MUTED, font=SMALL).pack(side="left")
        r = row("Repeat every - min (ms)")
        entry(r, "repeat_min_ms")
        r = row("Repeat every - max (ms)")
        entry(r, "repeat_max_ms")
        tk.Label(r, text="random in this range", bg=PANEL, fg=MUTED, font=SMALL).pack(side="left", padx=8)
        r = row("Enabled")
        self.enabled = tk.BooleanVar(value=bool(self.t.get("enabled", True)))
        tk.Checkbutton(r, variable=self.enabled, bg=PANEL, selectcolor=PANEL_2, activebackground=PANEL,
                       highlightthickness=0).pack(side="left")
        tk.Label(f, text="The first press waits the normal press delay; key hold time is the same as the flasks'.",
                 bg=PANEL, fg=MUTED, font=SMALL).pack(anchor="w", padx=12, pady=(2, 10))

        self.err = tk.Label(self, text="", bg=BG, fg=RED, font=FONT, justify="left")
        self.err.pack(anchor="w", padx=20)
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=20, pady=(4, 16))
        if not is_new:
            button(bar, "Delete trigger", self._delete, kind="ghost").pack(side="left")
        button(bar, "Save and pick the colour  ▶" if is_new else "Save", self._save, kind="primary").pack(side="right")
        button(bar, "Cancel", self.destroy).pack(side="right", padx=8)

    def _record(self):
        self._recording = True
        self.key_lbl.config(text="Press keys… (Esc cancels)", bg=GOLD, fg="#1a140c")
        self.focus_set()

    def _on_key(self, e):
        if not self._recording or e.keysym in MODIFIER_KEYSYMS:
            return
        self._recording = False
        if e.keysym != "Escape":
            self.key_val = combo_from_event(e)
        self.key_lbl.config(text=core.hotkey_label(self.key_val), bg=PANEL_2, fg=TEXT)
        return "break"

    def _delete(self):
        if messagebox.askyesno(core.APP, f"Delete the trigger “{self.t['name']}”?", parent=self):
            self.result = "delete"
            self.destroy()

    def _save(self):
        t = dict(self.t)
        t["name"] = self.vars["name"].get().strip()
        t["key"] = core.normalize_hotkey(self.key_val)
        try:
            for k in ("fire_below", "rearm_above", "repeat_min_ms", "repeat_max_ms"):
                t[k] = int(self.vars[k].get().strip())
        except ValueError:
            self.err.config(text="Use whole numbers for the percentages and times.")
            return
        t["repeat"] = bool(self.repeat.get())
        t["enabled"] = bool(self.enabled.get())
        errs = core.validate_trigger(t)
        if errs:
            self.err.config(text="\n".join(errs))
            return
        self.result = t
        self.destroy()


class TriggerCalibDialog(tk.Toplevel):
    """Draw a box around the area to watch, then click the colour (with the Similarity slider)."""

    def __init__(self, parent, img: np.ndarray, trig: dict, edits_key: Optional[str] = None):
        super().__init__(parent, bg=BG)
        self.title(f"{core.APP} | {trig['name']}")
        self.edits_key = edits_key
        self._old_box = list(trig["box"]) if trig.get("box") else None
        self.transient(parent)
        self.grab_set()
        self.img, self.t = img, trig
        self.result = None
        self.step = "box"
        self.box = trig.get("box") if trig.get("box") and self._fits(trig["box"]) else None
        self.points: List[Tuple[int, int]] = []
        self.sim = int(trig.get("similarity", core.DEFAULT_SIMILARITY))
        self.mode = trig.get("pick_mode", "shape")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.max_w, self.max_h = min(1280, sw - 120), min(600, sh - 320)
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=16, pady=(14, 6))
        self.step_lbl = tk.Label(top, text="", bg=BG, fg=GOLD, font=H_FONT)
        self.step_lbl.pack(anchor="w")
        self.instr = tk.Label(top, text="", bg=BG, fg=TEXT, font=("Segoe UI", 12), justify="left", wraplength=760)
        self.instr.pack(anchor="w")
        self.canvas = tk.Canvas(self, bg="#0c0a08", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=16, pady=6)
        self.canvas.bind("<ButtonPress-1>", self._down)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._up)
        Loupe(self.canvas, lambda: self.img, self._to_img, lambda: self.step == "pick")
        self.ctrl = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        r1 = tk.Frame(self.ctrl, bg=PANEL)
        r1.pack(fill="x", padx=12, pady=(8, 2))
        tk.Label(r1, text="Similarity", bg=PANEL, fg=GOLD, font=H_FONT, width=10, anchor="w").pack(side="left")
        tk.Label(r1, text="strict", bg=PANEL, fg=MUTED, font=SMALL).pack(side="left")
        self.slider = ttk.Scale(r1, from_=1, to=100, orient="horizontal", length=340,
                                style="Gold.Horizontal.TScale", command=self._slid)
        self.slider.pack(side="left", padx=8, pady=4)
        tk.Label(r1, text="loose", bg=PANEL, fg=MUTED, font=SMALL).pack(side="left")
        self.sim_lbl = tk.Label(r1, text="", bg=PANEL, fg=TEXT, font=H_FONT, width=4)
        self.sim_lbl.pack(side="left", padx=(10, 0))
        r2 = tk.Frame(self.ctrl, bg=PANEL)
        r2.pack(fill="x", padx=12, pady=(2, 8))
        tk.Label(r2, text="Pick from", bg=PANEL, fg=GOLD, font=H_FONT, width=10, anchor="w").pack(side="left")
        self.mode_var = tk.StringVar(value=self.mode)
        for val, label in (("shape", "Inside the shape"), ("box", "Anywhere in box")):
            tk.Radiobutton(r2, text=label, value=val, variable=self.mode_var, bg=PANEL, fg=TEXT, selectcolor=PANEL_2,
                           activebackground=PANEL, activeforeground=TEXT, font=FONT, highlightthickness=0,
                           command=self._render).pack(side="left", padx=(0, 14))
        self.undo = button(r2, "Undo click", self._undo, kind="ghost", font=SMALL, pady=3)
        self.undo.pack(side="right")
        self.fix = HandFix(self.ctrl, lambda redraw=True: redraw and self._render())
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=16, pady=(4, 14), side="bottom")
        button(bar, "Cancel", self.destroy, kind="ghost").pack(side="left")
        self.back = button(bar, "◀  Back", self._back)
        self.back.pack(side="left", padx=8)
        self.next_btn = button(bar, "Next  ▶", self._next, kind="primary")
        self.next_btn.pack(side="right")
        st = ttk.Style(self)
        st.configure("Gold.Horizontal.TScale", troughcolor=PANEL_2, background=GOLD, bordercolor=BORDER,
                     lightcolor=GOLD, darkcolor="#a8853f", sliderthickness=18)
        self._drag0 = None
        self._render()

    def _fits(self, b):
        x, y, w, h = b
        return x >= 0 and y >= 0 and x + w <= self.img.shape[1] and y + h <= self.img.shape[0]

    def _show(self, region, overlay=None, removed=None):
        x, y, w, h = region
        crop = shade_removed(self.img[y:y + h, x:x + w], removed).copy()
        if overlay is not None and overlay.any():
            crop[overlay] = (crop[overlay] * 0.35 + np.array([255, 0, 255]) * 0.65).astype(np.uint8)
        self.view = region
        self.scale = min(self.max_w / w, self.max_h / h, 6.0)
        self._photo = to_photo(crop, int(w * self.scale), int(h * self.scale))
        self.canvas.config(width=self._photo.width(), height=self._photo.height())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")

    def _to_img(self, cx, cy):
        return int(self.view[0] + cx / self.scale), int(self.view[1] + cy / self.scale)

    def _edits(self):
        x, y, w, h = self.box
        if self.fix.edits is None or self.fix.edits.shape != (h, w):
            e = None
            if self.edits_key and self._old_box == list(self.box):     # same area as last time: keep its fixes
                e = core.load_edits(self.edits_key, (h, w))
            self.fix.edits = e.copy() if e is not None else np.zeros((h, w), np.uint8)
        return self.fix.edits

    def _sel(self):
        x, y, w, h = self.box
        return core.apply_edits(
            core.liquid_selection(self.img[y:y + h, x:x + w], [(px - x, py - y) for px, py in self.points],
                                  self.sim, self.mode_var.get()), self._edits())

    def _render(self):
        set_button(self.back, enabled=self.step == "pick")
        if self.step == "box":
            self.ctrl.pack_forget()
            self.step_lbl.config(text=f"Step 1 of 2  ·  {self.t['name']}")
            self.instr.config(text="Drag a box around the area to watch (e.g. the whole life globe)."
                              + ("  The orange box is the current one." if self.box else ""))
            self._show((0, 0, self.img.shape[1], self.img.shape[0]))
            if self.box:
                bx, by, bw, bh = self.box
                s = self.scale
                self.canvas.create_rectangle(bx * s, by * s, (bx + bw) * s, (by + bh) * s, outline=ORANGE,
                                             width=2, dash=(4, 3), tags="sel")
            set_button(self.next_btn, text="Next  ▶", enabled=bool(self.box))
        else:
            self.ctrl.pack(fill="x", padx=16, pady=(0, 4))
            self.slider.set(self.sim)
            self.sim_lbl.config(text=str(self.sim))
            self.step_lbl.config(text=f"Step 2 of 2  ·  {self.t['name']}")
            m = self._sel()
            self._show(tuple(self.box), overlay=m, removed=self.fix.edits)
            for px, py in self.points:
                cx, cy = (px - self.view[0]) * self.scale, (py - self.view[1]) * self.scale
                self.canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, outline=YELLOW, width=2)
            share = float(m.mean())
            self.instr.config(text="With the colour fully showing, click it. Pink = what will be watched. Slide "
                              "Similarity until the pink covers it but nothing else." if not self.points else
                              f"Pink = what will be watched ({share:.0%} of the box). When most of it is gone, "
                              f"{core.hotkey_label(self.t['key'])} gets pressed.")
            set_button(self.undo, enabled=bool(self.points))
            set_button(self.next_btn, text="Finish  ✓", enabled=share >= 0.005)

    def _slid(self, v):
        val = int(round(float(v)))
        if val != self.sim:
            self.sim = val
            self.sim_lbl.config(text=str(val))
            self._render()

    def _down(self, e):
        if self.step == "pick":
            tool = self.fix.tool.get()
            if tool == "pick":
                self.points.append(self._to_img(e.x, e.y))
                self._render()
                return
            if tool in ("add", "remove"):
                self._paint(e)
                return
        self._drag0 = (e.x, e.y)
        self.canvas.delete("sel")
        self._rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline=YELLOW, width=2, tags="sel")

    def _paint(self, e):
        ix, iy = self._to_img(e.x, e.y)
        self._edits()
        self.fix.brush(ix - self.box[0], iy - self.box[1])
        self._render()

    def _drag(self, e):
        if self.step == "pick" and self.fix.tool.get() in ("add", "remove"):
            self._paint(e)
            return
        if self._drag0:
            self.canvas.coords(self._rect, *self._drag0, e.x, e.y)

    def _up(self, e):
        if not self._drag0:
            return
        x0, y0 = self._drag0
        self._drag0 = None
        a, b = self._to_img(min(x0, e.x), min(y0, e.y))
        c, d = self._to_img(max(x0, e.x), max(y0, e.y))
        if self.step == "pick":                      # Box in / Box out
            self._edits()
            self.fix.rect(a - self.box[0], b - self.box[1], c - self.box[0], d - self.box[1])
            self._render()
            return
        if c - a >= 4 and d - b >= 4:
            self.box = [a, b, c - a, d - b]
            self.points = []
            self.fix.edits = None
            set_button(self.next_btn, enabled=True)

    def _undo(self):
        if self.points:
            self.points.pop()
            self._render()

    def _back(self):
        self.step = "box"
        self._render()

    def _next(self):
        if self.step == "box":
            if not self.box:
                return
            self.step = "pick"
            self._render()
            return
        if float(self._sel().mean()) < 0.005:
            return
        self.result = {"box": self.box, "points": self.points, "similarity": self.sim, "mode": self.mode_var.get(),
                       "edits": self.fix.result()}
        self.destroy()


class TriggerRow(tk.Frame):
    def __init__(self, parent, app: "App", trig: dict):
        super().__init__(parent, bg=PANEL, highlightthickness=2, highlightbackground=BORDER)
        self.app, self.tid = app, trig["id"]
        self._blank = tk.PhotoImage(width=64, height=64)
        self.img = tk.Label(self, image=self._blank, bg="#0c0a08", borderwidth=0)
        self.img.pack(side="left", padx=8, pady=8)
        mid = tk.Frame(self, bg=PANEL)
        mid.pack(side="left", fill="both", expand=True, padx=4, pady=6)
        self.title = tk.Label(mid, text="", bg=PANEL, fg=GOLD, font=H_FONT, anchor="w")
        self.title.pack(anchor="w")
        self.sub = tk.Label(mid, text="", bg=PANEL, fg=MUTED, font=SMALL, anchor="w")
        self.sub.pack(anchor="w")
        self.bar = tk.Canvas(mid, width=260, height=12, bg=PANEL_2, highlightthickness=0)
        self.bar.pack(anchor="w", pady=(4, 0))
        right = tk.Frame(self, bg=PANEL)
        right.pack(side="right", padx=10)
        self.pill = tk.Label(right, text="", bg=GREY, fg=TEXT, font=PILL_FONT, width=22, pady=4)
        self.pill.pack(pady=(6, 4))
        btns = tk.Frame(right, bg=PANEL)
        btns.pack()
        button(btns, "Calibrate", lambda: app.calibrate_trigger(self.tid), font=SMALL, pady=3).pack(side="left", padx=2)
        button(btns, "Edit", lambda: app.edit_trigger(self.tid), font=SMALL, pady=3).pack(side="left", padx=2)
        self._photo = None

    def update_view(self, trig: dict, value: Optional[float], crop: Optional[np.ndarray], gone: Optional[bool],
                    live: bool, testing: bool = False):
        self.title.config(text=trig["name"])
        self.sub.config(text=f"presses {core.hotkey_label(trig['key'])} below {trig['fire_below']}% · "
                             f"stops above {trig['rearm_above']}% · "
                             + (f"every {trig['repeat_min_ms']}-{trig['repeat_max_ms']} ms" if trig.get("repeat", True)
                                else "once"))
        if crop is None and trig.get("box") and core.trigger_ref_path(trig["id"]).exists():
            crop = cv2.imread(str(core.trigger_ref_path(trig["id"])))
        if crop is not None and crop.size:
            self._photo = to_photo(crop, 64, 64)
            self.img.config(image=self._photo)
        self.bar.delete("all")
        if not trig.get("enabled", True):
            self.pill.config(text="OFF", bg=GREY, fg=TEXT)
        elif not (trig.get("box") and core.trigger_ref_path(trig["id"]).exists()):
            self.pill.config(text="NOT SET UP - Calibrate", bg=ORANGE, fg="white")
        elif not live or value is None:
            self.pill.config(text="ready", bg=GREY, fg=TEXT)
        else:
            colour = GREEN if value >= trig["rearm_above"] else (RED if value < trig["fire_below"] else ORANGE)
            self.bar.create_rectangle(0, 0, 260 * value / 100.0, 12, fill=colour, width=0)
            for pct in (trig["fire_below"], trig["rearm_above"]):
                self.bar.create_line(260 * pct / 100.0, 0, 260 * pct / 100.0, 12, fill=TEXT)
            if gone:
                verb = "would press" if testing else "pressing"
                self.pill.config(text=f"GONE {value:.0f}% · {verb} {core.hotkey_label(trig['key'])}", bg=RED, fg="white")
            else:
                self.pill.config(text=f"SHOWING {value:.0f}%", bg=GREEN if value >= trig["rearm_above"] else ORANGE,
                                 fg="white")


# ================================================================ report dialog
def show_report(parent) -> None:
    win = tk.Toplevel(parent, bg=BG)
    win.title(f"{core.APP} | Calibration results")
    win.transient(parent)
    results = core.LAST_REPORT
    all_ok = all(r["ok"] for r in results)
    tk.Label(win, text="Calibration results", bg=BG, fg=GOLD, font=TITLE_FONT).pack(anchor="w", padx=20, pady=(16, 2))
    tk.Label(win, text="Everything looks good." if all_ok else "Some flasks need attention - see below.",
             bg=BG, fg=GREEN if all_ok else ORANGE, font=H_FONT).pack(anchor="w", padx=20)

    body = tk.Frame(win, bg=BG)
    body.pack(padx=20, pady=12, fill="both")
    rows = tk.Frame(body, bg=BG)
    rows.pack(side="left", fill="y", anchor="n")
    for r in results:
        row = tk.Frame(rows, bg=PANEL, highlightthickness=1, highlightbackground=GREEN if r["ok"] else ORANGE)
        row.pack(fill="x", pady=4)
        tk.Label(row, text="✓" if r["ok"] else "!", bg=GREEN if r["ok"] else ORANGE, fg="white",
                 font=("Segoe UI Semibold", 14), width=3).pack(side="left", fill="y")
        txt = tk.Frame(row, bg=PANEL)
        txt.pack(side="left", padx=10, pady=6, anchor="w")
        title = f"Flask {r['slot']}  ·  {KIND_LABEL[r['kind']]}"
        tk.Label(txt, text=title, bg=PANEL, fg=TEXT, font=H_FONT).pack(anchor="w")
        detail = f"{r['colour'].capitalize()} liquid  ·  reads {'FULL' if r['full'] else 'NOT FULL'}"
        if r["has_bar"]:
            detail += "  ·  timer bar area set"
        tk.Label(txt, text=detail, bg=PANEL, fg=MUTED, font=SMALL).pack(anchor="w")
        for issue in r["issues"]:
            tk.Label(txt, text="•  " + issue, bg=PANEL, fg=ORANGE, font=SMALL, wraplength=380,
                     justify="left").pack(anchor="w")

    prev_path = core.DATA_DIR / "calibration_preview.png"
    if prev_path.exists():
        pv = tk.Frame(body, bg=BG)
        pv.pack(side="left", padx=(18, 0), anchor="n")
        win._photo = to_photo(cv2.imread(str(prev_path)), 460, 360)
        tk.Label(pv, image=win._photo, bg="#0c0a08").pack()
        tk.Label(pv, text="Pink = the top of the liquid it checks for FULL.\nOrange = where it looks for timer bars.",
                 bg=BG, fg=MUTED, font=SMALL, justify="left").pack(anchor="w", pady=6)

    tk.Label(win, text="Next: turn on Test mode and use a flask in your program - its card should switch to "
                       "NOT FULL (and RUNNING for quicksilvers).",
             bg=BG, fg=TEXT, font=FONT, wraplength=820, justify="left").pack(anchor="w", padx=20)
    button(win, "Done", win.destroy, kind="primary").pack(anchor="e", padx=20, pady=14)


# ================================================================ settings dialog
class SettingsDialog(tk.Toplevel):
    FIELDS = [
        # key, label, help
        ("checks_per_second", "Checks per second", "How often the flasks are read."),
        ("delay_min_ms", "Press delay - min (ms)", "Random wait before each press. Your course requires 150-300."),
        ("delay_max_ms", "Press delay - max (ms)", ""),
        ("key_hold_min_ms", "Key hold - min (ms)", "Each key is held a random time in this range."),
        ("key_hold_max_ms", "Key hold - max (ms)", ""),
        ("refire_cooldown_ms", "Same-flask cooldown (ms)", "Minimum gap between presses of one flask."),
        ("fails_for_town", "Presses with no effect before town mode", "Lower = pauses sooner in town."),
        ("verify_ms", "Time allowed for a press to work (ms)", "Raise this if town mode turns on by mistake."),
        ("town_retry_secs", "In town mode, retry a quicksilver every (s)", "0 = off: wait for a loading screen instead."),
    ]

    def __init__(self, parent, cfg: dict, running: bool):
        super().__init__(parent, bg=BG)
        self.title(f"{core.APP} | Settings")
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)
        self.cfg = cfg
        self.saved = False
        st = core.get_settings(cfg)

        tk.Label(self, text="Settings", bg=BG, fg=GOLD, font=TITLE_FONT).pack(anchor="w", padx=20, pady=(16, 0))
        if running:
            tk.Label(self, text="Changes apply the next time you press Start.", bg=BG, fg=ORANGE,
                     font=SMALL).pack(anchor="w", padx=20)

        # keys
        box = self._section("Flask keys")
        self.key_vars = {}
        row = tk.Frame(box, bg=PANEL)
        row.pack(anchor="w", pady=(2, 4))
        for slot in "12345":
            kind = KIND_LABEL[core.slot_kinds(self.cfg)[int(slot)]]
            col = tk.Frame(row, bg=PANEL)
            col.pack(side="left", padx=(0, 14))
            tk.Label(col, text=f"{slot} · {kind}", bg=PANEL, fg=MUTED, font=SMALL).pack(anchor="w")
            v = tk.StringVar(value=st["keys"][slot])
            e = tk.Entry(col, textvariable=v, width=4, bg=PANEL_2, fg=TEXT, insertbackground=TEXT,
                         relief="flat", font=("Segoe UI Semibold", 12), justify="center",
                         highlightthickness=1, highlightbackground=BORDER, highlightcolor=GOLD)
            e.pack(anchor="w", ipady=3)
            self.key_vars[slot] = v

        # numbers
        timing = self._section("Timing and town mode")
        self.num_vars = {}
        for key, label, help_ in self.FIELDS:
            r = tk.Frame(timing, bg=PANEL)
            r.pack(fill="x", pady=3)
            tk.Label(r, text=label, bg=PANEL, fg=TEXT, font=FONT, width=34, anchor="w").pack(side="left")
            v = tk.StringVar(value=str(st[key]))
            tk.Entry(r, textvariable=v, width=7, bg=PANEL_2, fg=TEXT, insertbackground=TEXT, relief="flat",
                     font=FONT, justify="right", highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=GOLD).pack(side="left", ipady=2)
            if help_:
                tk.Label(r, text=help_, bg=PANEL, fg=MUTED, font=SMALL).pack(side="left", padx=10)
            self.num_vars[key] = v

        hk = self._section("Hotkeys (work while your program is focused too)")
        self.hk_vals = {k: st[k] for k in core.HOTKEY_NAMES}
        self.hk_lbls = {}
        self._recording = None
        for key, name in core.HOTKEY_NAMES.items():
            r = tk.Frame(hk, bg=PANEL)
            r.pack(fill="x", pady=3)
            tk.Label(r, text=name, bg=PANEL, fg=TEXT, font=FONT, width=34, anchor="w").pack(side="left")
            lbl = tk.Label(r, text=core.hotkey_label(self.hk_vals[key]), bg=PANEL_2, fg=TEXT,
                           font=("Segoe UI Semibold", 10), width=16, pady=3)
            lbl.pack(side="left")
            self.hk_lbls[key] = lbl
            button(r, "Change", lambda k=key: self._record(k), font=SMALL, pady=3).pack(side="left", padx=(8, 4))
            button(r, "Clear", lambda k=key: self._set_hotkey(k, ""), kind="ghost", font=SMALL, pady=3).pack(side="left")
        self.bind("<KeyPress>", self._on_key)

        km = self._section("Key presses")
        r = tk.Frame(km, bg=PANEL)
        r.pack(fill="x", pady=3)
        tk.Label(r, text="How keys are sent", bg=PANEL, fg=TEXT, font=FONT, width=34, anchor="w").pack(side="left")
        self.method_var = tk.StringVar(value=core.KEY_METHODS[st.get("key_method", "scancode")])
        ttk.Combobox(r, textvariable=self.method_var, values=list(core.KEY_METHODS.values()), state="readonly",
                     width=34, style="Dark.TCombobox", font=FONT).pack(side="left")
        r2 = tk.Frame(km, bg=PANEL)
        r2.pack(fill="x", pady=(6, 0))
        button(r2, "Test keys", self._test_keys, font=FONT, pady=3).pack(side="left")
        tk.Label(r2, text="Presses 1-5 once each in your program so you can check they arrive.",
                 bg=PANEL, fg=MUTED, font=SMALL).pack(side="left", padx=10)

        alerts = self._section("Alerts")
        self.sound = tk.BooleanVar(value=bool(st["sound_alerts"]))
        tk.Checkbutton(alerts, text="Play a sound when town mode starts or stops", variable=self.sound,
                       bg=PANEL, fg=TEXT, selectcolor=PANEL_2, activebackground=PANEL, activeforeground=TEXT,
                       font=FONT, highlightthickness=0).pack(anchor="w")

        self.err = tk.Label(self, text="", bg=BG, fg=RED, font=FONT, justify="left")
        self.err.pack(anchor="w", padx=20)
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=20, pady=(6, 16))
        button(bar, "Reset to defaults", self._reset, kind="ghost").pack(side="left")
        button(bar, "Save", self._save, kind="primary").pack(side="right")
        button(bar, "Cancel", self.destroy).pack(side="right", padx=8)

    def _section(self, title):
        f = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        f.pack(fill="x", padx=20, pady=(12, 0))
        tk.Label(f, text=title, bg=PANEL, fg=GOLD, font=H_FONT).pack(anchor="w", padx=12, pady=(8, 2))
        inner = tk.Frame(f, bg=PANEL)
        inner.pack(fill="x", padx=12, pady=(0, 10))
        return inner

    # -- hotkey recording ---------------------------------------------------
    def _record(self, key):
        self._recording = key
        for k, l in self.hk_lbls.items():
            l.config(text=core.hotkey_label(self.hk_vals[k]), bg=PANEL_2)
        self.hk_lbls[key].config(text="Press keys…", bg=GOLD, fg="#1a140c")
        self.focus_set()

    def _set_hotkey(self, key, combo):
        self.hk_vals[key] = combo
        self.hk_lbls[key].config(text=core.hotkey_label(combo), bg=PANEL_2, fg=TEXT)
        self._recording = None

    def _on_key(self, e):
        if not self._recording:
            return
        if e.keysym in ("Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Meta_L", "Meta_R",
                        "Caps_Lock", "Win_L", "Win_R", "Super_L", "Super_R"):
            return "break"                       # wait for the main key
        if e.keysym == "Escape":
            self._set_hotkey(self._recording, self.hk_vals[self._recording])
            return "break"
        self._set_hotkey(self._recording, combo_from_event(e))
        return "break"

    def _collect(self) -> Optional[dict]:
        st = {"keys": {k: v.get().strip() for k, v in self.key_vars.items()},
              "sound_alerts": bool(self.sound.get()),
              "key_method": next(k for k, v in core.KEY_METHODS.items() if v == self.method_var.get())}
        st.update(self.hk_vals)
        for key, v in self.num_vars.items():
            try:
                st[key] = int(v.get().strip())
            except ValueError:
                self.err.config(text=f"“{v.get()}” isn't a whole number.")
                return None
        problems = core.validate_settings(st)
        if problems:
            self.err.config(text="\n".join(problems))
            return None
        self.err.config(text="")
        return st

    def _test_keys(self):
        st = self._collect()
        if st is None:
            return
        cfg = dict(self.cfg)
        cfg["settings"] = st
        if not cfg.get("window_title"):
            messagebox.showwarning(core.APP, "Pick your program window first.", parent=self)
            return
        if not messagebox.askokcancel(core.APP, "After you click OK, click into your program. In 3 seconds it "
                                      "will press your flask keys 1-5, one per second, using the settings above.\n\n"
                                      "Watch whether each flask is used.", parent=self):
            return
        loc = core.WindowLocator(cfg["window_title"], cfg.get("screen_origin", (0, 0)))
        parent = self.master

        def work():
            try:
                sender = core.KeySender(cfg, loc)
                time.sleep(3)
                for slot in range(1, 6):
                    if not loc.is_focused():
                        parent.log_q.put(f"Test keys: stopped at flask {slot} - your program wasn't focused.")
                        return
                    sender(slot, check_focus=False)
                    parent.log_q.put(f"Test keys: pressed '{st['keys'][str(slot)]}' (flask {slot}).")
                    time.sleep(1)
                parent.log_q.put("Test keys finished. If nothing happened in your program, try another "
                                 "'How keys are sent' option, or run Daniels Assignment as administrator.")
            except Exception as e:
                parent.log_q.put(f"Test keys error: {e}")

        threading.Thread(target=work, daemon=True).start()

    def _reset(self):
        d = core.DEFAULT_SETTINGS
        for slot, v in self.key_vars.items():
            v.set(d["keys"][slot])
        for key, v in self.num_vars.items():
            v.set(str(d[key]))
        self.sound.set(d["sound_alerts"])
        self.method_var.set(core.KEY_METHODS[d["key_method"]])
        for k in core.HOTKEY_NAMES:
            self._set_hotkey(k, d[k])
        self.err.config(text="")

    def _save(self):
        st = self._collect()
        if st is None:
            return
        self.cfg["settings"] = st
        core.save_config(self.cfg)
        self.saved = True
        self.destroy()


# ================================================================ compact mode
class CompactBar(tk.Toplevel):
    """Small always-on-top strip: five status dots, status text, start/stop."""

    DOT = 22

    def __init__(self, app: "App"):
        super().__init__(app, bg=GOLD)
        self.app = app
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        inner = tk.Frame(self, bg=PANEL)
        inner.pack(padx=1, pady=1)
        grip = tk.Label(inner, text="⠿", bg=PANEL, fg=MUTED, font=("Segoe UI", 14), cursor="fleur")
        grip.pack(side="left", padx=(8, 4))
        self.canvas = tk.Canvas(inner, width=5 * (self.DOT + 8), height=self.DOT + 8, bg=PANEL,
                                highlightthickness=0)
        self.canvas.pack(side="left", padx=4, pady=6)
        self.status = tk.Label(inner, text="Idle", bg=PANEL, fg=MUTED, font=H_FONT, width=13, anchor="w",
                               cursor="fleur")
        self.status.pack(side="left", padx=6)
        self.run_btn = button(inner, "▶", app.toggle_run, kind="primary", font=H_FONT, padx=10, pady=2)
        self.run_btn.pack(side="left", padx=4)
        button(inner, "⤢", app.leave_compact, font=H_FONT, padx=8, pady=2).pack(side="left", padx=(0, 8))
        for w in (grip, self.status):
            w.bind("<ButtonPress-1>", self._start_move)
            w.bind("<B1-Motion>", self._move)
            w.bind("<ButtonRelease-1>", self._save_pos)
        pos = app.cfg.get("compact_pos")
        if pos:
            self.geometry(f"+{pos[0]}+{pos[1]}")
        else:
            self.geometry(f"+{app.winfo_screenwidth() // 2 - 170}+20")
        self._off = (0, 0)

    def _start_move(self, e):
        self._off = (e.x_root - self.winfo_x(), e.y_root - self.winfo_y())

    def _move(self, e):
        self.geometry(f"+{e.x_root - self._off[0]}+{e.y_root - self._off[1]}")

    def _save_pos(self, _):
        self.app.cfg["compact_pos"] = [self.winfo_x(), self.winfo_y()]
        core.save_config(self.app.cfg)

    def update_view(self, readings, status: str, colour: str, mode: str):
        c, d = self.canvas, self.DOT
        c.delete("all")
        for i in range(5):
            slot = i + 1
            x, y = 4 + i * (d + 8), 4
            rd = readings.get(slot) if readings else None
            if rd is None:
                fill = GREY
            elif not rd.visible:
                fill = ORANGE
            else:
                fill = GREEN if rd.full else RED
            ring = YELLOW if (rd is not None and rd.active) else PANEL
            c.create_oval(x - 2, y - 2, x + d + 2, y + d + 2, outline=ring, width=3)
            c.create_oval(x, y, x + d, y + d, fill=fill, outline="")
            c.create_text(x + d / 2, y + d / 2, text=str(slot), fill="white", font=("Segoe UI Semibold", 9))
        self.status.config(text=status, fg=colour)
        set_button(self.run_btn, text="■" if mode == "run" else "▶",
                   kind="danger" if mode == "run" else "go",
                   enabled=mode == "run" or self.app.can_start())

    def flash(self, colour: str, times: int = 6):
        def step(n):
            if not self.winfo_exists():
                return
            self.config(bg=colour if n % 2 == 0 else GOLD)
            if n > 0:
                self.after(250, step, n - 1)
        step(times)


# ================================================================ main window
class App(tk.Tk):
    def __init__(self, updated_info: Optional[dict] = None):
        super().__init__()
        self.title(core.APP)
        self.configure(bg=BG)
        self.minsize(820, 640)
        self.log_q: "queue.Queue[str]" = queue.Queue()
        self.engine = Engine(self.log_q)
        self.cfg = core.load_config()
        self.mini: Optional[CompactBar] = None
        self._resized = False
        self._status = ("Idle", MUTED)
        self._tick_n = 0
        self._style()
        self._build()
        self.report_callback_exception = self._ui_error
        self.refresh_cards_idle()
        self.after(250, self._tick)
        self.hotkeys = core.HotkeyListener(core.get_settings(self.cfg),
                                           lambda name: self.log_q.put("__HK__" + name))
        if not self.hotkeys.start():
            self.write_log("Hotkeys unavailable on this computer.")
        self._update_hotkey_hint()
        self._update_info = None
        if updater.is_packaged() and updater.UPDATE_REPO:
            threading.Thread(target=self._check_update, daemon=True).start()
        if updated_info:
            self.after(400, lambda: self._show_updated(updated_info))
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _style(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("Dark.TCombobox", fieldbackground=PANEL_2, background=PANEL_2, foreground=TEXT,
                     arrowcolor=GOLD, bordercolor=BORDER, lightcolor=PANEL_2, darkcolor=PANEL_2)
        st.map("Dark.TCombobox", fieldbackground=[("readonly", PANEL_2)], foreground=[("readonly", TEXT)])
        self.option_add("*TCombobox*Listbox.background", PANEL_2)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", GOLD)

    # -- layout -------------------------------------------------------------
    def _build(self):
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=22, pady=(18, 6))
        tk.Label(head, text=core.APP, bg=BG, fg=GOLD, font=TITLE_FONT).pack(side="left")
        tk.Label(head, text=f"v{core.__version__}", bg=BG, fg=MUTED, font=SMALL).pack(side="left", padx=(10, 2), pady=(10, 0))
        self.check_lbl = tk.Label(head, text="Check for updates", bg=BG, fg=GOLD, font=("Segoe UI", 9, "underline"),
                                  cursor="hand2")
        self.check_lbl.pack(side="left", padx=6, pady=(10, 0))
        self.check_lbl.bind("<Button-1>", lambda _: self._manual_check())
        self.status = tk.Label(head, text="●  Idle", bg=PANEL_2, fg=MUTED, font=H_FONT, padx=14, pady=5)
        self.status.pack(side="right")
        button(head, "▭  Compact", self.enter_compact, font=FONT, padx=10, pady=5).pack(side="right", padx=(0, 10))
        button(head, "Report a problem", self.report_problem, kind="ghost", font=SMALL, padx=8, pady=5).pack(
            side="right", padx=(0, 8))
        button(head, "⚙  Settings", self.open_settings, font=FONT, padx=10, pady=5).pack(side="right", padx=(0, 8))

        # update banner (hidden until an update is found)
        self.banner = tk.Frame(self, bg=GOLD)
        self.banner_lbl = tk.Label(self.banner, text="", bg=GOLD, fg="#1a140c", font=H_FONT)
        self.banner_lbl.pack(side="left", padx=14, pady=7)
        self.banner_btn = button(self.banner, "Update now", self._do_update, kind="normal", font=H_FONT, pady=4)
        self.banner_btn.pack(side="right", padx=10, pady=5)
        self._head = head

        # resize warning (hidden unless the window size changed since calibration)
        self.resize_banner = tk.Frame(self, bg=ORANGE)
        self.resize_lbl = tk.Label(self.resize_banner, text="", bg=ORANGE, fg="#1a140c", font=FONT,
                                   justify="left", wraplength=640)
        self.resize_lbl.pack(side="left", padx=14, pady=7)
        button(self.resize_banner, "Recalibrate", self._recal_after_resize, font=H_FONT, pady=4).pack(
            side="right", padx=10, pady=5)

        # setup guide (hidden once you're set up)
        self.guide = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        steps_row = tk.Frame(self.guide, bg=PANEL)
        steps_row.pack(fill="x", padx=12, pady=(10, 4))
        self.guide_steps = []
        for i, name in enumerate(["Pick your program", "Calibrate flasks", "Check in Test mode", "Start"]):
            if i:
                tk.Label(steps_row, text="›", bg=PANEL, fg=GREY, font=H_FONT).pack(side="left", padx=4)
            lbl = tk.Label(steps_row, text=f"{i + 1}  {name}", bg=PANEL, fg=MUTED, font=H_FONT, padx=10, pady=3)
            lbl.pack(side="left")
            self.guide_steps.append(lbl)
        self.guide_hint = tk.Label(self.guide, text="", bg=PANEL, fg=TEXT, font=FONT, anchor="w", justify="left")
        self.guide_hint.pack(fill="x", padx=16, pady=(0, 10))
        self.guide.pack(fill="x", padx=22, pady=(6, 0))

        # window picker
        wbox = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        wbox.pack(fill="x", padx=22, pady=8)
        tk.Label(wbox, text="Program window", bg=PANEL, fg=GOLD, font=H_FONT).pack(side="left", padx=(14, 10), pady=10)
        self.win_var = tk.StringVar(value=self.cfg.get("window_title", ""))
        self.win_combo = ttk.Combobox(wbox, textvariable=self.win_var, width=46, style="Dark.TCombobox", font=FONT)
        self.win_combo.pack(side="left", pady=10)
        self.win_combo.bind("<<ComboboxSelected>>", lambda _: self._save_window())
        self.win_combo.bind("<FocusOut>", lambda _: self._save_window())
        self.win_combo.bind("<Return>", lambda _: self._save_window())
        button(wbox, "⟳", self._refresh_windows, font=H_FONT, padx=10, pady=3).pack(side="left", padx=6)
        self.win_status = tk.Label(wbox, text="", bg=PANEL, fg=MUTED, font=FONT)
        self.win_status.pack(side="left", padx=8)
        self._refresh_windows()

        # flask cards
        self.cards_frame = tk.Frame(self, bg=BG)
        self.cards_frame.pack(padx=22, pady=8)
        self.cards: Dict[int, FlaskCard] = {}
        self._build_cards()

        # triggers
        tf = tk.Frame(self, bg=BG)
        tf.pack(fill="x", padx=22, pady=(4, 0))
        th = tk.Frame(tf, bg=BG)
        th.pack(fill="x")
        tk.Label(th, text="Triggers", bg=BG, fg=GOLD, font=H_FONT).pack(side="left")
        tk.Label(th, text="press a key when a colour disappears", bg=BG, fg=MUTED, font=SMALL).pack(side="left", padx=8)
        button(th, "+  Add trigger", self.add_trigger, font=SMALL, pady=3).pack(side="right")
        self.trig_frame = tk.Frame(tf, bg=BG)
        self.trig_frame.pack(fill="x", pady=(4, 0))
        self.trig_rows: Dict[int, TriggerRow] = {}
        self._rebuild_triggers()

        # controls
        ctl = tk.Frame(self, bg=BG)
        ctl.pack(fill="x", padx=22, pady=10)
        self.cal_btn = button(ctl, "Calibrate all flasks", lambda: self.calibrate([1, 2, 3, 4, 5]))
        self.cal_btn.pack(side="left")
        self.test_btn = button(ctl, "Test mode", self.toggle_test)
        self.test_btn.pack(side="left", padx=10)
        self.town_btn = button(ctl, "Leave town mode", self._leave_town, kind="ghost")
        self.pause_btn = button(ctl, "Pause", self._toggle_pause, kind="ghost")
        self.run_btn = button(ctl, "▶  Start", self.toggle_run, kind="go", font=("Segoe UI Semibold", 13),
                              padx=26, pady=9)
        self.run_btn.pack(side="right")
        self.pause_btn.pack(side="right", padx=8)
        self.hk_hint = tk.Label(self, text="", bg=BG, fg=MUTED, font=SMALL, anchor="e")
        self.hk_hint.pack(fill="x", padx=24, pady=(0, 2), after=ctl)

        # log
        logf = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        logf.pack(fill="both", expand=True, padx=22, pady=(4, 18))
        tk.Label(logf, text="Activity", bg=PANEL, fg=GOLD, font=H_FONT).pack(anchor="w", padx=12, pady=(8, 0))
        self.log = tk.Text(logf, bg=PANEL, fg=TEXT, font=("Consolas", 9), relief="flat", height=8,
                           highlightthickness=0, borderwidth=0,
                           padx=10, pady=6, state="disabled", wrap="none")
        self.log.pack(fill="both", expand=True, padx=2, pady=(0, 6))
        self.log.tag_config("town", foreground=ORANGE)
        self.log.tag_config("press", foreground=GREEN)
        self.log.tag_config("err", foreground=RED)
        self._update_buttons()
        self.write_log(f"{core.APP} ready. Calibrate your flasks, check them in Test mode, then press Start.")

    # -- helpers -----------------------------------------------------------
    def write_log(self, msg: str) -> None:
        tag = "town" if "TOWN" in msg else "press" if "pressed" in msg else \
            "err" if ("error" in msg.lower() or "isn't focused" in msg) else None
        self.log.config(state="normal")
        self.log.insert("end", time.strftime("%H:%M:%S  ") + msg + "\n", tag)
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 400:
            self.log.delete("1.0", f"{lines - 400}.0")
        self.log.see("end")
        self.log.config(state="disabled")

    def _locator(self) -> core.WindowLocator:
        return core.WindowLocator(self.cfg.get("window_title", ""), self.cfg.get("screen_origin", (0, 0)))

    def _refresh_windows(self):
        titles = core.WindowLocator("").list_titles()
        self.win_combo["values"] = titles
        self._update_window_status()

    def _save_window(self):
        title = self.win_var.get().strip()
        if title != self.cfg.get("window_title", ""):
            self.cfg["window_title"] = title
            core.save_config(self.cfg)
            self.write_log(f"Program window set to “{title}”." if title else "Program window cleared.")
        self._update_window_status()
        self._update_buttons()

    def _update_window_status(self):
        title = self.win_var.get().strip()
        loc = core.WindowLocator(title)
        if not title:
            self.win_status.config(text="Pick your program from the list", fg=MUTED)
        elif not loc.win:
            self.win_status.config(text="(window tracking needs Windows)", fg=MUTED)
        elif loc.find():
            self.win_status.config(text="●  Found", fg=GREEN)
        else:
            self.win_status.config(text="●  Not open", fg=RED)

    def _calibrated(self) -> bool:
        return all(s["box"] and core.ref_path(s["slot"]).exists() for s in self.cfg["slots"])

    def _update_buttons(self):
        mode = self.engine.mode
        ready = self._calibrated()
        set_button(self.cal_btn, enabled=mode == "idle")
        set_button(self.test_btn, text="Stop test" if mode == "test" else "Test mode",
                   kind="danger" if mode == "test" else "normal", enabled=ready and mode != "run")
        has_window = bool(self.cfg.get("window_title"))
        set_button(self.run_btn, text="■  Stop" if mode == "run" else "▶  Start",
                   kind="danger" if mode == "run" else "go",
                   enabled=(ready and has_window and mode != "test" and not self._resized) or mode == "run")
        if mode == "run":
            self.pause_btn.pack(side="right", padx=8)
        else:
            self.pause_btn.pack_forget()

    def _set_status(self, text: str, colour: str):
        flashing = self.status.cget("bg") != PANEL_2
        self.status.config(text=f"●  {text}", fg="#1a140c" if flashing else colour)
        self._status = (text, colour)

    def can_start(self) -> bool:
        return self.engine.mode == "idle" and self._calibrated() and bool(self.cfg.get("window_title")) \
            and not self._resized

    # -- setup guide ---------------------------------------------------------
    def _update_guide(self):
        title = self.cfg.get("window_title", "")
        done = [bool(title), self._calibrated(), bool(self.cfg.get("tested")), self.engine.mode == "run"]
        current = next((i for i, d in enumerate(done) if not d), 3)
        for i, lbl in enumerate(self.guide_steps):
            if done[i] and i < 3:
                lbl.config(bg=PANEL, fg=GREEN, text="✓  " + lbl.cget("text")[3:])
            elif i == current:
                lbl.config(bg=GOLD, fg="#1a140c", text=f"{i + 1}  " + lbl.cget("text")[3:])
            else:
                lbl.config(bg=PANEL, fg=MUTED, text=f"{i + 1}  " + lbl.cget("text")[3:])
        hints = [
            "Choose your program in “Program window” below (open it first, then press ⟳ if it's not listed).",
            "Make every flask full in your program, then click “Calibrate all flasks”.",
            "Click “Test mode” and use a flask in your program — its card should turn NOT FULL.",
            "All set. Click Start (or use Compact mode to keep a small status bar on top).",
        ]
        hint = hints[current]
        if current == 3 and not self.can_start() and self.engine.mode != "run":
            if self.engine.mode == "test":
                hint = "Stop Test mode to enable Start."
            elif self._resized:
                hint = "Start is off because your program's window changed size — recalibrate first."
        self.guide_hint.config(text=hint)
        if all(done[:3]) and self.guide.winfo_ismapped():
            self.guide.pack_forget()
        elif not all(done[:3]) and not self.guide.winfo_ismapped():
            self.guide.pack(fill="x", padx=22, pady=(6, 0), after=self._head)

    # -- resize warning -------------------------------------------------------
    def _check_resize(self):
        expected = self.cfg.get("window_size")
        size = None
        if self.engine.reader is not None:
            mm = self.engine.reader.size_mismatch
            size = mm[0] if mm else None
        elif expected and self.cfg.get("window_title"):
            cur = self._locator().client_size()
            if cur and list(cur) != list(expected):
                size = cur
        was = self._resized
        self._resized = size is not None
        if self._resized:
            self.resize_lbl.config(
                text=f"Your program's window is now {size[0]}×{size[1]}, but it was calibrated at "
                     f"{expected[0]}×{expected[1]}. Flasks won't be pressed until you resize it back "
                     "or recalibrate.")
            if not self.resize_banner.winfo_ismapped():
                self.resize_banner.pack(fill="x", padx=22, pady=(0, 4), after=self._head)
            if not was:
                self.write_log("Window size changed since calibration — pressing is paused.")
        elif self.resize_banner.winfo_ismapped():
            self.resize_banner.pack_forget()
            self.write_log("Window size matches calibration again.")
        if was != self._resized:
            self._update_buttons()

    def _recal_after_resize(self):
        if self.engine.mode != "idle":
            self.engine.stop()
            self.refresh_cards_idle()
        self.calibrate([1, 2, 3, 4, 5])

    # -- town alert ------------------------------------------------------------
    def _alert(self, on: bool):
        if core.get_settings(self.cfg)["sound_alerts"]:
            try:
                import winsound
                winsound.MessageBeep(0x30 if on else 0x40)
            except Exception:
                self.bell()
        colour = ORANGE if on else GREEN

        def flash(n):
            lit = n % 2 == 0 and n > 0
            self.status.config(bg=colour if lit else PANEL_2,
                               fg="#1a140c" if lit else self._status[1])
            if n > 0:
                self.after(250, flash, n - 1)
        flash(7)
        if self.mini is not None:
            self.mini.flash(colour)

    # -- settings / compact ----------------------------------------------------
    def open_settings(self):
        self.hotkeys.paused = True
        dlg = SettingsDialog(self, self.cfg, running=self.engine.mode != "idle")
        self.wait_window(dlg)
        self.hotkeys.paused = False
        if dlg.saved:
            self.cfg = core.load_config()
            self.hotkeys.update(core.get_settings(self.cfg))
            self._update_hotkey_hint()
            st = core.get_settings(self.cfg)
            keys = " ".join(st["keys"][k] for k in "12345")
            self.write_log(f"Settings saved (keys {keys}, delay {st['delay_min_ms']}-{st['delay_max_ms']} ms, "
                           f"{st['checks_per_second']} checks/s).")

    def enter_compact(self):
        if self.mini is not None:
            return
        self.mini = CompactBar(self)
        self.withdraw()

    def leave_compact(self):
        if self.mini is not None:
            self.mini.destroy()
            self.mini = None
        self.deiconify()
        self.lift()

    # -- cards ------------------------------------------------------------
    def refresh_cards_idle(self):
        for s in self.cfg["slots"]:
            card = self.cards[s["slot"]]
            p = core.ref_path(s["slot"])
            if s["box"] and p.exists():
                card.show_image(cv2.imread(str(p)))
                m = core.model_for(s)
                if m.problems:
                    card.set_state("CHECK", ORANGE, m.problems[0][:34] + "…")
                else:
                    card.set_state("CALIBRATED", GREY, f"{core.hue_name(m.hue)} liquid", None)
            else:
                card.show_image(None)
                card.set_state("NOT CALIBRATED", GREY, "use Calibrate", None)

    def _show_live(self):
        with self.engine.lock:
            r = self.engine.latest
            crops = dict(self.engine.crops)
        if r is None:
            why = (self.engine.reader.problem if self.engine.reader is not None else "") or "starting…"
            label = {"window not found": "NO WINDOW", "window resized": "RESIZED"}.get(why, "CAN'T READ")
            for card in self.cards.values():
                card.set_state(label, ORANGE, why, None)
            return
        for slot, rd in r.items():
            card = self.cards[slot]
            if slot in crops:
                card.show_image(crops[slot])
            run = rd.active if card.kind in core.BAR_KINDS else None
            left = rd.remaining if (card.kind == "utility" and rd.active) else None
            if not rd.visible:
                card.set_state("NOT VISIBLE", ORANGE, "belt hidden / covered", run, left)
            elif rd.brightness < core.PAUSE_DIM:
                card.set_state("PAUSED", GREY, "program looks paused (dimmed)", run, left)
            else:
                gtxt = f" · globe {rd.globe:.0f}%" if rd.globe is not None else ""
                if rd.full:
                    card.set_state("FULL", GREEN, f"about {rd.fill:.0%}{gtxt}", run, left)
                else:
                    card.set_state("NOT FULL", RED, f"about {rd.fill:.0%}{gtxt}", run, left)

    # -- actions ------------------------------------------------------------
    def _capture(self) -> Optional[np.ndarray]:
        loc = self._locator()
        self.withdraw()
        self.update()
        try:
            loc.bring_to_front()
            time.sleep(0.6)
            img, origin = core.ScreenGrabber().grab_window_or_screen(loc)
            if not (self.cfg.get("window_title") and loc.win):
                self.cfg["screen_origin"] = list(origin)
            return img
        except SystemExit as e:
            messagebox.showerror(core.APP, str(e.code))
            return None
        finally:
            self.deiconify()
            self.lift()
            self.focus_force()

    def calibrate(self, slots: List[int]):
        if self.engine.mode != "idle":
            return
        self._save_window()
        which = "every flask" if len(slots) == 5 else f"flask {slots[0]}"
        if not messagebox.askokcancel(
                core.APP, f"Make sure {which} is FULL in your program.\n\n"
                          "When you click OK this window hides for a moment and takes a picture "
                          "of your program."):
            return
        img = self._capture()
        if img is None:
            return
        detected = core.auto_detect(img, self.cfg)
        if len(slots) == 5:
            dlg = ReviewDialog(self, img, self.cfg, detected)
            self.wait_window(dlg)
            if dlg.manual:                               # "Do it step by step instead"
                dlg = CalibrationDialog(self, img, self.cfg, slots)
                self.wait_window(dlg)
        else:
            preset = {slots[0]: detected[slots[0]]} if slots[0] in detected else {}
            dlg = CalibrationDialog(self, img, self.cfg, slots, preset=preset)
            self.wait_window(dlg)
        if not dlg.result:
            self.write_log("Calibration cancelled.")
            return
        boxes, bars, points, liq = dlg.result
        found = sum(1 for d in detected.values() if not d.get("unsure"))
        self.write_log(f"Auto-detection found {len(detected)} of 5 flasks ({found} confidently).")
        loc = self._locator()
        size = (img.shape[1], img.shape[0]) if (self.cfg.get("window_title") and loc.win) else None
        core.calibrate_from_image(img, self.cfg, boxes, bars, points, liquid_settings=liq, window_size=size)
        self.cfg = core.load_config()
        self.refresh_cards_idle()
        self._update_buttons()
        self.write_log(f"Calibrated {which}.")
        self._resized = False
        self._check_resize()
        show_report(self)

    def recalibrate(self, slot: int):
        self.calibrate([slot])

    def _build_cards(self):
        for w in self.cards_frame.winfo_children():
            w.destroy()
        self.cards = {}
        for i, s in enumerate(self.cfg["slots"]):
            c = FlaskCard(self.cards_frame, s["slot"], s["kind"], self.recalibrate, self.kind_menu)
            c.grid(row=0, column=i, padx=6, sticky="n")
            self.cards[s["slot"]] = c

    def kind_menu(self, slot: int, x: int, y: int):
        """Click a flask's name: choose what that flask does."""
        sc = core.slot_cfg(self.cfg, slot)
        m = tk.Menu(self, tearoff=0, bg=PANEL, fg=TEXT, activebackground=PANEL_2, activeforeground=TEXT, font=FONT)
        tips = {"life": "use when full", "mana": "use when full", "quicksilver": "take turns, one running at a time",
                "utility": f"use when full and under {sc.get('refresh_below', core.UTILITY_REFRESH_BELOW):g}% left",
                "off": "never pressed"}
        for k, name in core.KIND_NAMES.items():
            mark = "✓ " if sc["kind"] == k else "   "
            m.add_command(label=f"{mark}{name}  ({tips[k]})", command=lambda k=k: self.set_kind(slot, k))
        m.add_separator()
        g = sc.get("globe") or {}
        if g.get("box"):
            m.add_command(label=f"Globe linked: pressed below {g.get('below', core.GLOBE_BELOW):g}%  (change in Settings…)",
                          state="disabled")
            m.add_command(label="Pick the globe again…", command=lambda: self.pick_globe(slot))
            m.add_command(label="Unlink the globe", command=lambda: self.unlink_globe(slot))
        else:
            m.add_command(label="Link to a globe (red life / blue mana)…", command=lambda: self.pick_globe(slot))
        m.add_command(label=f"Settings for flask {slot}…", command=lambda: self.flask_settings(slot))
        m.tk_popup(x, y)

    def _apply_live(self) -> str:
        """Push flask changes into the running controller. Returns a note for the log."""
        c = self.engine.controller
        if self.engine.mode == "run" and c is not None:
            c.update_options(**core.controller_options(self.cfg))
            return " Applied straight away."
        return ""

    def pick_globe(self, slot: int):
        if self.engine.mode != "idle":
            messagebox.showinfo(core.APP, "Stop first, then link the globe.")
            return
        self._save_window()
        sc = core.slot_cfg(self.cfg, slot)
        g = dict(sc.get("globe") or {})
        if not messagebox.askokcancel(core.APP, "Make sure the globe is completely full in your program.\n\n"
                                      "When you click OK this window hides for a moment and takes a picture. Then draw a "
                                      "box around the globe and click its colour (red for life, blue for mana)."):
            return
        img = self._capture()
        if img is None:
            return
        pseudo = {"name": f"Flask {slot} globe", "box": g.get("box"), "similarity": g.get("similarity", core.DEFAULT_SIMILARITY),
                  "pick_mode": g.get("pick_mode", "shape")}
        dlg = TriggerCalibDialog(self, img, pseudo, edits_key=f"globe{slot}")
        self.wait_window(dlg)
        if not dlg.result:
            return
        x, y, w, h = dlg.result["box"]
        core.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(core.globe_ref_path(slot)), img[y:y + h, x:x + w])
        core.save_edits(f"globe{slot}", dlg.result.get("edits"))
        g.update(box=dlg.result["box"], liquid_points=[[px - x, py - y] for px, py in dlg.result["points"]],
                 similarity=dlg.result["similarity"], pick_mode=dlg.result["mode"])
        g.setdefault("below", core.GLOBE_BELOW)
        g.setdefault("again_ms", core.GLOBE_AGAIN_MS)
        sc["globe"] = g
        core.save_config(self.cfg)
        self._build_cards()
        self.write_log(f"Flask {slot} is linked to the globe: it's pressed when the globe is below {g['below']:g}%. "
                       f"Change the % in Settings for flask {slot}. Check it in Test mode (the card shows the globe %).")
        self.flask_settings(slot)

    def unlink_globe(self, slot: int):
        sc = core.slot_cfg(self.cfg, slot)
        sc.pop("globe", None)
        try:
            core.globe_ref_path(slot).unlink()
        except OSError:
            pass
        core.save_edits(f"globe{slot}", None)
        core.save_config(self.cfg)
        self.write_log(f"Flask {slot} is no longer linked to a globe." + self._apply_live())

    def flask_settings(self, slot: int):
        dlg = FlaskSettingsDialog(self, self.cfg, slot)
        self.wait_window(dlg)
        if dlg.saved:
            core.save_config(self.cfg)
            self._build_cards()
            self.write_log(dlg.saved + self._apply_live())

    def set_kind(self, slot: int, kind: str):
        sc = core.slot_cfg(self.cfg, slot)
        if sc["kind"] == kind:
            return
        sc["kind"] = kind
        if kind in core.BAR_KINDS and sc.get("box") and not sc.get("bar_box"):
            sc["bar_box"] = core.default_bar_box(sc["box"])
        if kind == "utility":
            sc.setdefault("refresh_below", core.UTILITY_REFRESH_BELOW)
        core.save_config(self.cfg)
        self._build_cards()
        msg = f"Flask {slot} is now: {core.KIND_NAMES[kind]}."
        if kind in core.BAR_KINDS and sc.get("box"):
            msg += " If its timer strip isn't picked up, press Recalibrate on it."
        self.write_log(msg + self._apply_live())

    def toggle_test(self):
        if self.engine.mode == "test":
            self.engine.stop()
            self.write_log("Test mode stopped.")
            self.refresh_cards_idle()
        elif self.engine.mode == "idle":
            self.engine.start("test")
            if self.engine.error:
                messagebox.showerror(core.APP, self.engine.error)
            else:
                self.write_log("Test mode: reading flasks live. Nothing will be pressed.")
        self._update_buttons()

    def toggle_run(self):
        if self.engine.mode == "run":
            self.engine.stop()
            self.refresh_cards_idle()
        elif self.engine.mode == "idle":
            self.engine.start("run")
            if self.engine.error:
                messagebox.showerror(core.APP, self.engine.error)
        self._update_buttons()

    def report_problem(self):
        c = self.engine.controller
        state = {"mode": self.engine.mode, "error": self.engine.error,
                 "reader_problem": getattr(self.engine.reader, "problem", None),
                 "has_latest_reading": self.engine.latest is not None,
                 "thread_alive": bool(self.engine.thread and self.engine.thread.is_alive()),
                 "town_mode": getattr(c, "town_mode", None), "paused": getattr(c, "paused", None),
                 "resized_flag": self._resized}
        self.write_log("Making a problem report…")
        self.update_idletasks()
        try:
            path = core.write_report(core.DATA_DIR.parent, self.log.get("1.0", "end"), state)
        except Exception as e:
            core.note_error("report")
            messagebox.showerror(core.APP, f"Couldn't make the report: {e}")
            return
        self.write_log(f"Report saved: {path}")
        messagebox.showinfo(core.APP, f"Saved a problem report:\n\n{path}\n\n"
                                      "Send this file to Claude. It contains your settings, the activity log, "
                                      "recent errors and a picture of your program's window.")
        try:
            import subprocess, sys as _sys
            if _sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", str(path)])
        except Exception:
            pass

    def _ui_error(self, exc, val, tb):
        import traceback
        where = traceback.extract_tb(tb)[-1] if tb else None
        msg = f"Something went wrong: {exc.__name__}: {val}"
        if where:
            msg += f"  ({where.name}, line {where.lineno})"
        self.write_log(msg)
        try:
            messagebox.showerror(core.APP, msg + "\n\nPlease send a screenshot of this to get it fixed.")
        except Exception:
            pass
        self._update_buttons()

    # -- triggers ------------------------------------------------------------------
    def _rebuild_triggers(self):
        for w in self.trig_frame.winfo_children():
            w.destroy()
        self.trig_rows = {}
        for t in self.cfg.get("triggers", []):
            row = TriggerRow(self.trig_frame, self, t)
            row.pack(fill="x", pady=3)
            self.trig_rows[t["id"]] = row
            row.update_view(t, None, None, None, False)

    def _save_trigger(self, t: dict):
        trigs = self.cfg.setdefault("triggers", [])
        for i, old in enumerate(trigs):
            if old["id"] == t["id"]:
                trigs[i] = t
                break
        else:
            trigs.append(t)
        core.save_config(self.cfg)
        self._rebuild_triggers()

    def add_trigger(self):
        if self.engine.mode != "idle":
            messagebox.showinfo(core.APP, "Stop first, then add the trigger.")
            return
        t = core.new_trigger(self.cfg)
        dlg = TriggerEditDialog(self, t, is_new=True)
        self.wait_window(dlg)
        if isinstance(dlg.result, dict):
            self._save_trigger(dlg.result)
            self.write_log(f"Added trigger “{dlg.result['name']}”. Now click its Calibrate button.")
            self.calibrate_trigger(dlg.result["id"])

    def edit_trigger(self, tid: int):
        if self.engine.mode != "idle":
            messagebox.showinfo(core.APP, "Stop first, then edit the trigger.")
            return
        t = next(x for x in self.cfg["triggers"] if x["id"] == tid)
        dlg = TriggerEditDialog(self, t, is_new=False)
        self.wait_window(dlg)
        if dlg.result == "delete":
            self.cfg["triggers"] = [x for x in self.cfg["triggers"] if x["id"] != tid]
            core.save_config(self.cfg)
            try:
                core.trigger_ref_path(tid).unlink()
            except OSError:
                pass
            self._rebuild_triggers()
            self.write_log(f"Deleted trigger “{t['name']}”.")
        elif isinstance(dlg.result, dict):
            self._save_trigger(dlg.result)
            self.write_log(f"Saved trigger “{dlg.result['name']}”.")

    def calibrate_trigger(self, tid: int):
        if self.engine.mode != "idle":
            messagebox.showinfo(core.APP, "Stop first, then calibrate.")
            return
        self._save_window()
        t = next(x for x in self.cfg["triggers"] if x["id"] == tid)
        if not messagebox.askokcancel(core.APP, f"Make sure “{t['name']}” is fully showing in your program.\n\n"
                                      "When you click OK this window hides for a moment and takes a picture."):
            return
        img = self._capture()
        if img is None:
            return
        dlg = TriggerCalibDialog(self, img, t, edits_key=f"trigger{tid}")
        self.wait_window(dlg)
        if not dlg.result:
            return
        x, y, w, h = dlg.result["box"]
        core.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(core.trigger_ref_path(tid)), img[y:y + h, x:x + w])
        core.save_edits(f"trigger{tid}", dlg.result.get("edits"))
        t = dict(t, box=dlg.result["box"], liquid_points=[[px - x, py - y] for px, py in dlg.result["points"]],
                 similarity=dlg.result["similarity"], pick_mode=dlg.result["mode"])
        self._save_trigger(t)
        self.write_log(f"Calibrated trigger “{t['name']}”. Check it in Test mode.")

    def _show_triggers_live(self):
        live = self.engine.mode in ("test", "run")
        with self.engine.lock:
            vals = dict(self.engine.trig_values)
            crops = dict(self.engine.trig_crops)
        c = self.engine.controller
        for t in self.cfg.get("triggers", []):
            row = self.trig_rows.get(t["id"])
            if row is None:
                continue
            v = vals.get(t["id"])
            gone = None
            if c is not None and t["id"] in c.tstate:
                gone = c.tstate[t["id"]]["gone"]
            elif v is not None:
                gone = v < t["fire_below"]
            row.update_view(t, v, crops.get(t["id"]) if live else None, gone, live, self.engine.mode == "test")

    def _hotkey(self, name: str):
        if name == "hotkey_toggle":
            if self.engine.mode == "test":
                self.toggle_test()
            if self.engine.mode == "run" or self.can_start():
                self.toggle_run()
                self.write_log("Started (hotkey)." if self.engine.mode == "run" else "Stopped (hotkey).")
            else:
                self.write_log("Hotkey: can't start yet - " + self.guide_hint.cget("text"))
        elif name == "hotkey_pause" and self.engine.mode == "run":
            self._toggle_pause()
        elif name == "hotkey_town" and self.engine.mode == "run":
            self._leave_town()

    def _update_hotkey_hint(self):
        st = core.get_settings(self.cfg)
        parts = [f"{core.hotkey_label(st[k])} {label.lower()}" for k, label in core.HOTKEY_NAMES.items() if st[k]]
        self.hk_hint.config(text="   ·   ".join(parts) if parts else "No hotkeys set")

    def _toggle_pause(self):
        c = self.engine.controller
        if c:
            c.paused = not c.paused
            self.write_log("Paused." if c.paused else "Resumed.")

    def _leave_town(self):
        c = self.engine.controller
        if c:
            c.leave_town("button pressed")

    # -- periodic UI update --------------------------------------------------
    def _tick(self):
        while not self.log_q.empty():
            msg = self.log_q.get_nowait()
            if msg == "__STOP__":
                self.toggle_run()
            elif msg == "__UPDATE__":
                self.check_lbl.config(text="Check for updates")
                self._show_banner()
            elif msg.startswith("__PROGRESS__"):
                self.banner_lbl.config(text=f"Downloading v{self._update_info['version']}…  {msg[12:]}")
            elif msg == "__INSTALLING__":
                self.banner_lbl.config(text=f"Starting v{self._update_info['version']}…")
            elif msg == "__UPDATED__":
                if self.mini is not None:
                    self.mini.destroy()
                self.destroy()
                return
            elif msg == "__ROLLED_BACK__":
                set_button(self.banner_btn, text="Try again", enabled=True)
                self.banner_lbl.config(text="The new version didn't start, so this version was kept.")
                messagebox.showwarning(core.APP, "The new version didn't start properly, so the update was "
                                                 f"undone and you're still on v{core.__version__}.\n\n"
                                                 "You can try again, or download it from the Releases page.")
            elif msg == "__UPDATE_FAILED__":
                set_button(self.banner_btn, text="Try again", enabled=True)
            elif msg == "__NO_UPDATE__":
                self.check_lbl.config(text="Check for updates")
                messagebox.showinfo(core.APP, f"You're on the latest version (v{core.__version__}).")
            elif msg == "__CHECK_FAILED__":
                self.check_lbl.config(text="Check for updates")
                messagebox.showwarning(core.APP, "Couldn't reach GitHub to check for updates. "
                                                 "Check your internet connection and try again.")
            elif msg in ("__TOWN_ON__", "__TOWN_OFF__"):
                self._alert(msg == "__TOWN_ON__")
            elif msg.startswith("__HK__"):
                self._hotkey(msg[6:])
            else:
                self.write_log(msg)
        mode = self.engine.mode
        if mode != getattr(self, "_last_mode", None):
            self._last_mode = mode
            self._update_buttons()
        if mode == "idle":
            self._set_status("Idle", MUTED)
            self.town_btn.pack_forget()
        elif mode == "test":
            self._set_status("Test mode · nothing is pressed", "#7fb4e6")
            self._show_live()
            if not self.cfg.get("tested") and self.engine.latest:
                self.cfg["tested"] = True
                core.save_config(self.cfg)
        else:
            c = self.engine.controller
            if c and c.program_paused:
                self._set_status("Program paused · waiting", "#7fb4e6")
                self.town_btn.pack_forget()
            elif c and c.town_mode:
                self._set_status("Town mode · not pressing", ORANGE)
                self.town_btn.pack(side="right", padx=8)
            elif c and c.paused:
                self._set_status("Paused", YELLOW)
                self.town_btn.pack_forget()
            else:
                self._set_status("Running", GREEN)
                self.town_btn.pack_forget()
            if c:
                set_button(self.pause_btn, text="Resume" if c.paused else "Pause")
            self._show_live()
        if self._resized and mode == "run":
            self._set_status("Window resized · not pressing", ORANGE)
        self._tick_n += 1
        if self._tick_n % 8 == 0:                     # every ~2 s
            self._update_window_status()
            self._check_resize()
        elif self.engine.reader is not None and bool(self.engine.reader.size_mismatch) != self._resized:
            self._check_resize()
        self._update_guide()
        if self.engine.mode in ("test", "run") or getattr(self, "_trig_was_live", False):
            self._show_triggers_live()
        self._trig_was_live = self.engine.mode in ("test", "run")
        if self.mini is not None:
            with self.engine.lock:
                r = self.engine.latest
            short = {"Test mode · nothing is pressed": "Test mode", "Town mode · not pressing": "Town mode",
                     "Program paused · waiting": "Paused (program)",
                     "Window resized · not pressing": "Resized!"}.get(self._status[0], self._status[0])
            self.mini.update_view(r if mode != "idle" else None, short, self._status[1], mode)
        self.after(250, self._tick)

    # -- updates --------------------------------------------------------------
    def _check_update(self, manual: bool = False):
        try:
            info = updater.check(core.__version__)
        except Exception:
            if manual:
                self.log_q.put("__CHECK_FAILED__")
            return                               # offline etc. - try next launch
        if info:
            self._update_info = info
            self.log_q.put("__UPDATE__")
        elif manual:
            self.log_q.put("__NO_UPDATE__")

    def _manual_check(self):
        if not updater.is_packaged():
            messagebox.showinfo(core.APP, "Updates are only installed by the .exe version of the app.")
            return
        self.check_lbl.config(text="Checking…")
        threading.Thread(target=self._check_update, args=(True,), daemon=True).start()

    def _show_banner(self):
        v = self._update_info["version"]
        self.banner_lbl.config(text=f"A new version (v{v}) is available.")
        set_button(self.banner_btn, text="Update now", enabled=True)
        if not self.banner.winfo_ismapped():
            self.banner.pack(fill="x", padx=22, pady=(0, 4), after=self._head)
        self.write_log(f"Update available: v{v}.")

    def _show_updated(self, info: dict):
        notes = _clean_notes(info.get("notes", ""))
        self.write_log(f"Updated to v{core.__version__}.")
        messagebox.showinfo(core.APP, f"Updated to v{core.__version__}" +
                            (f" (from v{info.get('from')})" if info.get("from") else "") +
                            (f"\n\nWhat's new:\n{notes}" if notes else ""))

    def _do_update(self):
        info = self._update_info
        if not info or not messagebox.askokcancel(
                core.APP, f"Update to v{info['version']} now?\n\nThe app will download it "
                          f"({info.get('size', 0) / 1e6:.0f} MB), close and reopen. Your calibration is kept."):
            return
        self.engine.stop()
        set_button(self.banner_btn, text="Updating…", enabled=False)

        def progress(f):
            self.log_q.put(f"__PROGRESS__{f:.0%}")
            if f >= 1.0:
                self.log_q.put("__INSTALLING__")

        def work():
            try:
                ok = updater.install_and_restart(info, core.DATA_DIR, core.__version__, progress)
                self.log_q.put("__UPDATED__" if ok else "__ROLLED_BACK__")
            except Exception as e:
                self.log_q.put(f"Update failed: {e}")
                self.log_q.put("__UPDATE_FAILED__")

        threading.Thread(target=work, daemon=True).start()

    def _close(self):
        self.hotkeys.stop()
        self.engine.stop()
        if self.mini is not None:
            self.mini.destroy()
        self.destroy()


def _dpi_aware():
    """Decide DPI handling once, before any window exists (changing it later
    makes Windows rescale the app unpredictably)."""
    import sys
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def _clean_notes(notes: str) -> str:
    keep = [l for l in notes.splitlines()
            if l.strip() and not l.startswith(("Co-Authored-By", "Claude-Session", "**Full Changelog"))]
    return "\n".join(keep[:12])


def launch():
    _dpi_aware()
    updated = None
    if updater.is_packaged():
        try:
            updated = updater.on_startup(core.DATA_DIR, core.__version__)
        except Exception:
            updated = None
    app = App(updated)
    app.update_idletasks()
    w = min(app.winfo_reqwidth(), app.winfo_screenwidth() - 40)
    h = min(app.winfo_reqheight(), app.winfo_screenheight() - 80)
    app.geometry(f"{w}x{h}+{max(0, (app.winfo_screenwidth() - w) // 2)}+{max(0, (app.winfo_screenheight() - h) // 3)}")
    app.mainloop()


if __name__ == "__main__":
    launch()
