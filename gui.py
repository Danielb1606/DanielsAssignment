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
from PIL import Image, ImageTk

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
KIND_LABEL = {"life": "Life", "mana": "Mana", "quicksilver": "Quicksilver"}


def to_photo(bgr: np.ndarray, max_w: int, max_h: int, upscale=True) -> ImageTk.PhotoImage:
    h, w = bgr.shape[:2]
    s = min(max_w / w, max_h / h)
    if not upscale:
        s = min(s, 1.0)
    img = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))),
                     interpolation=cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))


def button(parent, text, cmd, kind="normal", **kw):
    colours = {
        "normal": (PANEL_2, TEXT, "#3a3128"),
        "primary": (GOLD, "#1a140c", "#dcb873"),
        "danger": (RED, "white", "#d9605a"),
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
                      "danger": (RED, "white", "#d9605a"), "ghost": (PANEL, MUTED, PANEL_2)}[kind]
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
        self.stop_flag.clear()
        self.mode = mode
        self.reader = reader
        settings = core.get_settings(cfg)
        self.poll_interval = 1.0 / settings["checks_per_second"]
        if mode == "test":
            target = self._test_loop
            args = (reader,)
        else:
            sender = core.KeySender(cfg, loc)
            self.controller = core.FlaskController(
                lambda: self._read_and_store(reader), sender, log=self.log, settings=settings)
            self.controller.on_town = lambda on: self.log_q.put("__TOWN_ON__" if on else "__TOWN_OFF__")
            target = self._run_loop
            args = ()
            self._start_hotkeys()
        self.thread = threading.Thread(target=target, args=args, daemon=True)
        self.thread.start()

    def _test_loop(self, reader) -> None:
        while not self.stop_flag.is_set():
            try:
                self._read_and_store(reader)
            except Exception as e:      # keep the UI alive
                self.log(f"Read error: {e}")
            time.sleep(self.poll_interval)

    def _run_loop(self) -> None:
        self.log("Started. F8 pause/resume, F9 leave town mode, F10 stop.")
        try:
            self.controller.run()
        except Exception as e:
            self.log(f"Stopped because of an error: {e}")
        self.log("Stopped.")

    def _start_hotkeys(self) -> None:
        try:
            from pynput import keyboard
        except Exception:
            return

        def on_press(key):
            c = self.controller
            if not c or self.mode != "run":
                return
            if key == keyboard.Key.f8:
                c.paused = not c.paused
                self.log("Paused." if c.paused else "Resumed.")
            elif key == keyboard.Key.f9:
                c.leave_town("F9 pressed")
            elif key == keyboard.Key.f10:
                self.log_q.put("__STOP__")

        self.hotkeys = keyboard.Listener(on_press=on_press)
        self.hotkeys.daemon = True
        self.hotkeys.start()

    def stop(self) -> None:
        self.stop_flag.set()
        if self.controller:
            self.controller.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.hotkeys:
            self.hotkeys.stop()
            self.hotkeys = None
        self.thread = None
        self.controller = None
        self.reader = None
        self.mode = "idle"
        with self.lock:
            self.latest = None
            self.crops = {}


# ================================================================ flask card
class FlaskCard(tk.Frame):
    def __init__(self, parent, slot: int, kind: str, on_recal):
        super().__init__(parent, bg=PANEL, highlightthickness=2, highlightbackground=BORDER)
        self.slot, self.kind = slot, kind
        tk.Label(self, text=f"{slot}  ·  {KIND_LABEL[kind]}", bg=PANEL, fg=GOLD, font=H_FONT).pack(pady=(8, 4))
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
        if kind == "quicksilver":
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

    def set_state(self, label: str, colour: str, detail: str = "", running: Optional[bool] = None) -> None:
        self.pill.config(text=label, bg=colour)
        self.config(highlightbackground=colour if colour != GREY else BORDER)
        self.detail.config(text=detail)
        if self.run_pill is not None:
            if running is None:
                self.run_pill.config(text="—", bg=PANEL, fg=MUTED, highlightbackground=GREY)
            elif running:
                self.run_pill.config(text="RUNNING", bg=YELLOW, fg="#1a140c", highlightbackground=YELLOW)
            else:
                self.run_pill.config(text="not running", bg=PANEL, fg=MUTED, highlightbackground=GREY)


# ================================================================ calibration dialog
class CalibrationDialog(tk.Toplevel):
    """Step-by-step: box each flask, click its liquid, then the timer strips."""

    def __init__(self, parent, img: np.ndarray, cfg: dict, slots: List[int]):
        super().__init__(parent, bg=BG)
        self.title(f"{core.APP} | Calibration")
        self.transient(parent)
        self.grab_set()
        self.img, self.cfg, self.slots = img, cfg, slots
        self.result = None

        self.steps: List[Tuple[str, int]] = []
        for s in slots:
            self.steps += [("box", s), ("liquid", s)]
        self.steps += [("bar", s) for s in slots if s in core.QUICKSILVER_SLOTS]
        self.i = 0
        self.boxes: Dict[int, List[int]] = {}
        self.points: Dict[int, Tuple[int, int]] = {}
        self.bars: Dict[int, List[int]] = {}

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.max_w, self.max_h = min(1280, sw - 120), min(720, sh - 260)

        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=16, pady=(14, 6))
        self.step_lbl = tk.Label(top, text="", bg=BG, fg=GOLD, font=H_FONT)
        self.step_lbl.pack(anchor="w")
        self.instr = tk.Label(top, text="", bg=BG, fg=TEXT, font=("Segoe UI", 12), justify="left")
        self.instr.pack(anchor="w", pady=(2, 0))

        self.canvas = tk.Canvas(self, bg="#0c0a08", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=16, pady=6)
        self.canvas.bind("<ButtonPress-1>", self._down)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._up)

        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=16, pady=(4, 14))
        button(bar, "Cancel", self.destroy, kind="ghost").pack(side="left")
        self.back_btn = button(bar, "◀  Back", self._back)
        self.back_btn.pack(side="left", padx=8)
        self.next_btn = button(bar, "Use this  ▶", self._accept_default, kind="primary")
        self.next_btn.pack(side="right")

        self._drag_start = None
        self._rect = None
        self._pending_box = None
        self._render()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.focus_set()

    # -- view helpers -------------------------------------------------------
    def _set_view(self, region: Tuple[int, int, int, int]) -> None:
        """Show part of the screenshot (x, y, w, h) scaled to the canvas."""
        x, y, w, h = region
        crop = self.img[y:y + h, x:x + w]
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

    def _draw_box(self, box, colour, dash=None, width=2, tag=None):
        self.canvas.create_rectangle(*self._to_canvas(box), outline=colour, width=width, dash=dash, tags=tag)

    # -- steps ----------------------------------------------------------------
    def _render(self) -> None:
        kind, slot = self.steps[self.i]
        name = KIND_LABEL[core.SLOT_KINDS[slot]]
        self.step_lbl.config(text=f"Step {self.i + 1} of {len(self.steps)}  ·  Flask {slot} ({name})")
        set_button(self.back_btn, enabled=self.i > 0)
        self._pending_box = None
        old = core.slot_cfg(self.cfg, slot)

        if kind == "box":
            self._set_view((0, 0, self.img.shape[1], self.img.shape[0]))
            for s, b in self.boxes.items():
                self._draw_box(b, GREEN)
            existing = old["box"] if old["box"] and self._fits(old["box"]) else None
            self._pending_box = existing
            if existing:
                self._draw_box(existing, ORANGE, dash=(4, 3), tag="sel")
            self.instr.config(text=f"Drag a tight box around flask {slot}."
                              + ("  The orange box is the current one." if existing else ""))
            set_button(self.next_btn, text="Next  ▶", enabled=existing is not None)
        elif kind == "liquid":
            x, y, w, h = self.boxes[slot]
            self._set_view((x, y, w, h))
            self.instr.config(text=f"Click on the {name.lower()} liquid inside flask {slot}.\n"
                                   "Avoid labels, wings, highlights and glass.")
            set_button(self.next_btn, text="Next  ▶", enabled=False)
        else:  # bar
            x, y, w, h = self.boxes[slot]
            pad = int(h * 0.6)
            vx, vy = max(0, x - w), max(0, y - 10)
            vw = min(self.img.shape[1] - vx, w * 3)
            vh = min(self.img.shape[0] - vy, h + pad + 10)
            self._set_view((vx, vy, vw, vh))
            self._draw_box(self.boxes[slot], GREEN)
            prev = old["bar_box"] if (old["bar_box"] and len(self.slots) == 1) else None
            self._pending_box = prev or core.default_bar_box(self.boxes[slot])
            self._draw_box(self._pending_box, ORANGE, tag="sel")
            self.instr.config(text=f"The orange box is where flask {slot}'s timer bar is read.\n"
                                   "Click Next to use it, or drag a new box where the bar appears.")
            set_button(self.next_btn, text="Next  ▶", enabled=True)
        if self.i == len(self.steps) - 1 and kind == "bar":
            set_button(self.next_btn, text="Finish  ✓")

    def _fits(self, box) -> bool:
        x, y, w, h = box
        return x >= 0 and y >= 0 and x + w <= self.img.shape[1] and y + h <= self.img.shape[0]

    def _down(self, e):
        kind, slot = self.steps[self.i]
        if kind == "liquid":
            px, py = self._to_img(e.x, e.y)
            self.points[slot] = (px, py)
            self.canvas.delete("dot")
            self.canvas.create_oval(e.x - 6, e.y - 6, e.x + 6, e.y + 6, outline=YELLOW, width=3, tags="dot")
            hue = core.hue_at(self.img, (px, py))
            self.instr.config(text=f"Got it - {core.hue_name(hue)} liquid. Click Next, or click again to change.")
            set_button(self.next_btn, text="Next  ▶", enabled=True)
            return
        self._drag_start = (e.x, e.y)
        self.canvas.delete("sel")
        self._rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline=YELLOW, width=2, tags="sel")

    def _drag(self, e):
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
            if slot not in self.points:
                return
        else:
            self.bars[slot] = self._pending_box
        if self.i == len(self.steps) - 1:
            self.result = (self.boxes, self.bars, self.points)
            self.destroy()
            return
        self.i += 1
        self._render()

    def _back(self):
        if self.i > 0:
            self.i -= 1
            self._render()


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
        ("key_hold_ms", "Key hold (ms)", "How long each key is held down."),
        ("refire_cooldown_ms", "Same-flask cooldown (ms)", "Minimum gap between presses of one flask."),
        ("fails_for_town", "Presses with no effect before town mode", "Lower = pauses sooner in town."),
        ("verify_ms", "Time allowed for a press to work (ms)", "Raise this if town mode turns on by mistake."),
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
            kind = KIND_LABEL[core.SLOT_KINDS[int(slot)]]
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

    def _reset(self):
        d = core.DEFAULT_SETTINGS
        for slot, v in self.key_vars.items():
            v.set(d["keys"][slot])
        for key, v in self.num_vars.items():
            v.set(str(d[key]))
        self.sound.set(d["sound_alerts"])
        self.err.config(text="")

    def _save(self):
        st = {"keys": {k: v.get().strip() for k, v in self.key_vars.items()},
              "sound_alerts": bool(self.sound.get())}
        for key, v in self.num_vars.items():
            try:
                st[key] = int(v.get().strip())
            except ValueError:
                self.err.config(text=f"“{v.get()}” isn't a whole number.")
                return
        problems = core.validate_settings(st)
        if problems:
            self.err.config(text="\n".join(problems))
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
                   kind="danger" if mode == "run" else "primary",
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
    def __init__(self):
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
        self.refresh_cards_idle()
        self.after(250, self._tick)
        self._update_info = None
        if updater.is_packaged() and updater.UPDATE_REPO:
            threading.Thread(target=self._check_update, daemon=True).start()
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
        tk.Label(head, text=f"v{core.__version__}", bg=BG, fg=MUTED, font=SMALL).pack(side="left", padx=10, pady=(10, 0))
        self.status = tk.Label(head, text="●  Idle", bg=PANEL_2, fg=MUTED, font=H_FONT, padx=14, pady=5)
        self.status.pack(side="right")
        button(head, "▭  Compact", self.enter_compact, font=FONT, padx=10, pady=5).pack(side="right", padx=(0, 10))
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
        cards = tk.Frame(self, bg=BG)
        cards.pack(padx=22, pady=8)
        self.cards: Dict[int, FlaskCard] = {}
        for i, s in enumerate(self.cfg["slots"]):
            c = FlaskCard(cards, s["slot"], s["kind"], self.recalibrate)
            c.grid(row=0, column=i, padx=6, sticky="n")
            self.cards[s["slot"]] = c

        # controls
        ctl = tk.Frame(self, bg=BG)
        ctl.pack(fill="x", padx=22, pady=10)
        self.cal_btn = button(ctl, "Calibrate all flasks", lambda: self.calibrate([1, 2, 3, 4, 5]))
        self.cal_btn.pack(side="left")
        self.test_btn = button(ctl, "Test mode", self.toggle_test)
        self.test_btn.pack(side="left", padx=10)
        self.town_btn = button(ctl, "Leave town mode", self._leave_town, kind="ghost")
        self.pause_btn = button(ctl, "Pause", self._toggle_pause, kind="ghost")
        self.run_btn = button(ctl, "▶  Start", self.toggle_run, kind="primary", font=("Segoe UI Semibold", 13),
                              padx=26, pady=9)
        self.run_btn.pack(side="right")
        self.pause_btn.pack(side="right", padx=8)

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
                   kind="danger" if mode == "run" else "primary",
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
        dlg = SettingsDialog(self, self.cfg, running=self.engine.mode != "idle")
        self.wait_window(dlg)
        if dlg.saved:
            self.cfg = core.load_config()
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
            for card in self.cards.values():
                card.set_state("NO WINDOW", ORANGE, "program not found", None)
            return
        for slot, rd in r.items():
            card = self.cards[slot]
            if slot in crops:
                card.show_image(crops[slot])
            run = rd.active if card.kind == "quicksilver" else None
            if not rd.visible:
                card.set_state("NOT VISIBLE", ORANGE, "belt hidden / covered", run)
            elif rd.full:
                card.set_state("FULL", GREEN, f"about {rd.fill:.0%}", run)
            else:
                card.set_state("NOT FULL", RED, f"about {rd.fill:.0%}", run)

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
        dlg = CalibrationDialog(self, img, self.cfg, slots)
        self.wait_window(dlg)
        if not dlg.result:
            self.write_log("Calibration cancelled.")
            return
        boxes, bars, points = dlg.result
        loc = self._locator()
        size = (img.shape[1], img.shape[0]) if (self.cfg.get("window_title") and loc.win) else None
        core.calibrate_from_image(img, self.cfg, boxes, bars, points, window_size=size)
        self.cfg = core.load_config()
        self.refresh_cards_idle()
        self._update_buttons()
        self.write_log(f"Calibrated {which}.")
        self._resized = False
        self._check_resize()
        show_report(self)

    def recalibrate(self, slot: int):
        self.calibrate([slot])

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
                self._show_banner()
            elif msg.startswith("__PROGRESS__"):
                self.banner_lbl.config(text=f"Downloading v{self._update_info['version']}…  {msg[12:]}")
            elif msg == "__UPDATED__":
                self.destroy()
                return
            elif msg == "__UPDATE_FAILED__":
                set_button(self.banner_btn, text="Try again", enabled=True)
            elif msg in ("__TOWN_ON__", "__TOWN_OFF__"):
                self._alert(msg == "__TOWN_ON__")
            else:
                self.write_log(msg)
        mode = self.engine.mode
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
            if c and c.town_mode:
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
        if self.mini is not None:
            with self.engine.lock:
                r = self.engine.latest
            short = {"Test mode · nothing is pressed": "Test mode", "Town mode · not pressing": "Town mode",
                     "Window resized · not pressing": "Resized!"}.get(self._status[0], self._status[0])
            self.mini.update_view(r if mode != "idle" else None, short, self._status[1], mode)
        self.after(250, self._tick)

    # -- updates --------------------------------------------------------------
    def _check_update(self):
        try:
            info = updater.check(core.__version__)
        except Exception:
            return                               # offline etc. - try next launch
        if info:
            self._update_info = info
            self.log_q.put(f"__UPDATE__")

    def _show_banner(self):
        v = self._update_info["version"]
        self.banner_lbl.config(text=f"A new version (v{v}) is available.")
        self.banner.pack(fill="x", padx=22, pady=(0, 4), after=self._head)
        self.write_log(f"Update available: v{v}.")

    def _do_update(self):
        info = self._update_info
        if not info or not messagebox.askokcancel(
                core.APP, f"Update to v{info['version']} now?\n\nThe app will close, update and reopen. "
                          "Your calibration is kept."):
            return
        self.engine.stop()
        set_button(self.banner_btn, text="Downloading…", enabled=False)

        def work():
            try:
                updater.download_and_restart(
                    info, lambda f: self.log_q.put(f"__PROGRESS__{f:.0%}"))
                self.log_q.put("__UPDATED__")
            except Exception as e:
                self.log_q.put(f"Update failed: {e}")
                self.log_q.put("__UPDATE_FAILED__")

        threading.Thread(target=work, daemon=True).start()

    def _close(self):
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


def launch():
    _dpi_aware()
    app = App()
    app.update_idletasks()
    w = min(app.winfo_reqwidth(), app.winfo_screenwidth() - 40)
    h = min(app.winfo_reqheight(), app.winfo_screenheight() - 80)
    app.geometry(f"{w}x{h}+{max(0, (app.winfo_screenwidth() - w) // 2)}+{max(0, (app.winfo_screenheight() - h) // 3)}")
    app.mainloop()


if __name__ == "__main__":
    launch()
