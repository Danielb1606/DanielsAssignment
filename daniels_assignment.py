"""
Daniels Assignment
==================

Screen-reading flask controller for a PoE-style UI running in its own window.

Modes
-----
  calibrate   Drag a box around each flask (and the timer strip under each
              quicksilver). Saves each flask's full-state image as its
              reference and prints a report of what is and isn't working.
              --slot N redoes one flask only (e.g. after its art changes size).
  test        Live indicator window: FULL / NOT FULL / RUNNING / NOT VISIBLE
              for every flask. Never presses anything.
  run         The controller.

Rules (run mode)
----------------
  * Checks the flasks 4 times a second.
  * Quicksilver (slots 3-5): a full one is used only when no quicksilver is
    running (timer bar visible, press queued, or pressed within 0.5 s).
  * Life / mana (slots 1-2): used whenever full (Instilling Orb style).
  * Every press is sent 150-300 ms (random, uniform) after the decision.
  * Keys 1-5 are sent as ordinary key presses, only while your program's
    window is focused.
  * Town mode: if 2 presses in a row have no effect (flask stays full and no
    timer bar appears), the tool stops pressing. It resumes when the flask
    belt disappears and comes back (area change / loading screen), when you
    use a flask yourself, or when you press F9.

Hotkeys (run mode): F8 pause/resume, F9 leave town mode, F10 quit.

Install:  pip install mss numpy opencv-python pynput
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

APP = "Daniels Assignment"
import collections, traceback
RECENT_ERRORS: "collections.deque[str]" = collections.deque(maxlen=20)   # for "Report a problem"


def note_error(where: str) -> None:
    """Remember the current exception (with its full traceback) for diagnostics."""
    RECENT_ERRORS.append(f"[{time.strftime('%H:%M:%S')}] {where}\n{traceback.format_exc()}")
__version__ = "1.7.0"
# Saved next to the .exe when packaged (a one-file .exe unpacks itself to a
# temporary folder, so __file__ would point somewhere that gets deleted).
_BASE = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
DATA_DIR = _BASE / "daniels_assignment_data"
CONFIG_PATH = DATA_DIR / "config.json"

# ---------------------------------------------------------------- timing
POLL_INTERVAL = 0.25                 # 4 checks per second
INPUT_DELAY_RANGE = (0.150, 0.300)   # course requirement
KEY_HOLD = 0.03                      # how long each key is held down
ACTIVATION_GRACE = 0.5               # treat a just-pressed quicksilver as running
REFIRE_COOLDOWN = 0.5                # min gap between presses of the same slot
VERIFY_WINDOW = 1.2                  # time for a press to show an effect
FAILS_FOR_TOWN = 2                   # presses with no effect before town mode
HIDDEN_POLLS_FOR_AREA_CHANGE = 2     # belt hidden this many checks = area change
TRIGGER_TAG = 1000       # queue ids >= this are trigger presses, not flasks
PAUSE_DIM = 0.8          # belt this much darker than calibration = your program is paused
CHANGE_MIN = 6.0                     # avg grey-level change that counts as "the press did something"

# Everything above can be changed in the app's Settings panel; these are the defaults.
DEFAULT_SETTINGS = {
    "keys": {"1": "1", "2": "2", "3": "3", "4": "4", "5": "5"},
    "checks_per_second": 4,
    "delay_min_ms": 150,
    "delay_max_ms": 300,
    "key_hold_min_ms": 20,
    "key_hold_max_ms": 90,
    "key_method": "scancode",
    "refire_cooldown_ms": 500,
    "verify_ms": 1200,
    "fails_for_town": 2,
    "town_retry_secs": 0,     # off: town mode ends at the next loading screen (or F9 / using a flask)
    "sound_alerts": True,
    "hotkey_toggle": "f7",
    "hotkey_pause": "f8",
    "hotkey_town": "f9",
}
HOTKEY_NAMES = {"hotkey_toggle": "Start / stop", "hotkey_pause": "Pause / resume", "hotkey_town": "Leave town mode"}
MODIFIERS = ("ctrl", "alt", "shift")


def vk_to_name(vk: int) -> Optional[str]:
    """Windows virtual-key code -> hotkey key name."""
    if 65 <= vk <= 90:
        return chr(vk).lower()
    if 48 <= vk <= 57:
        return chr(vk)
    if 112 <= vk <= 123:
        return f"f{vk - 111}"
    if 96 <= vk <= 105:
        return f"num{vk - 96}"
    return {32: "space", 9: "tab", 13: "enter", 8: "backspace", 45: "insert", 46: "delete", 36: "home",
            35: "end", 33: "page_up", 34: "page_down", 37: "left", 38: "up", 39: "right", 40: "down",
            192: "`", 189: "-", 187: "=", 219: "[", 221: "]", 220: "\\", 186: ";", 222: "'", 188: ",",
            190: ".", 191: "/"}.get(vk)


def normalize_hotkey(text: str) -> str:
    """'Shift + E' / 'shift+e' / 'F7' -> 'shift+e' / 'f7' (modifiers in a fixed order). '' = none."""
    if not text or str(text).strip().lower() in ("none", ""):
        return ""
    parts = [p.strip().lower() for p in str(text).replace(" ", "").split("+") if p.strip()]
    mods = [m for m in MODIFIERS if m in parts]
    keys = [p for p in parts if p not in MODIFIERS]
    if len(keys) != 1:
        return ""
    return "+".join(mods + keys)


def hotkey_label(combo: str) -> str:
    if not combo:
        return "None"
    nice = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift"}
    return " + ".join(nice.get(p, p.upper() if len(p) <= 3 else p.replace("_", " ").title()) for p in combo.split("+"))


KEY_METHODS = {"scancode": "Hardware scan code (most compatible)",
               "virtual": "Virtual key code",
               "typed": "Typed character (old method)"}


def get_settings(cfg: dict) -> dict:
    out = json.loads(json.dumps(DEFAULT_SETTINGS))
    saved = cfg.get("settings", {})
    for k, v in saved.items():
        if k == "keys" and isinstance(v, dict):
            out["keys"].update({str(a): str(b) for a, b in v.items()})
        elif k in HOTKEY_NAMES:
            out[k] = normalize_hotkey(v)
        elif k in out:
            out[k] = v
    return out


def validate_settings(st: dict) -> List[str]:
    """Problems with a settings dict (empty list = OK)."""
    errs = []
    keys = st["keys"]
    for slot in "12345":
        k = str(keys.get(slot, "")).strip()
        if len(k) != 1:
            errs.append(f"Flask {slot} key must be a single key (e.g. 1 or q).")
    if len({str(keys.get(s, "")).lower() for s in "12345"}) < 5:
        errs.append("Each flask needs a different key.")
    if not 1 <= st["checks_per_second"] <= 20:
        errs.append("Checks per second must be between 1 and 20.")
    if not 0 <= st["delay_min_ms"] <= st["delay_max_ms"] <= 5000:
        errs.append("Press delay: min must be ≤ max (0-5000 ms).")
    if not 5 <= st["key_hold_min_ms"] <= st["key_hold_max_ms"] <= 1000:
        errs.append("Key hold: min must be ≤ max (5-1000 ms).")
    hks = [normalize_hotkey(st.get(k, "")) for k in HOTKEY_NAMES]
    used = [h for h in hks if h]
    if len(used) != len(set(used)):
        errs.append("Each hotkey needs a different key combination.")
    flask_keys = {str(st["keys"].get(s_, "")).lower() for s_ in "12345"}
    for name, h in zip(HOTKEY_NAMES.values(), hks):
        if h and h in flask_keys:
            errs.append(f"{name} hotkey is the same as a flask key - add a modifier (e.g. Shift).")
    if st.get("key_method") not in KEY_METHODS:
        errs.append("Unknown key sending method.")
    if not 0 <= st["refire_cooldown_ms"] <= 10000:
        errs.append("Cooldown must be 0-10000 ms.")
    if not 300 <= st["verify_ms"] <= 10000:
        errs.append("Effect check window must be 300-10000 ms.")
    if not 0 <= st.get("town_retry_secs", 3) <= 120:
        errs.append("Town retry must be 0-120 seconds (0 = off).")
    if not 1 <= st["fails_for_town"] <= 10:
        errs.append("Town mode trigger must be 1-10 presses.")
    return errs

# ---------------------------------------------------------------- detection
SLOT_KINDS = {1: "life", 2: "mana", 3: "quicksilver", 4: "quicksilver", 5: "quicksilver"}
QUICKSILVER_SLOTS = (3, 4, 5)
INSTILLING_SLOTS = (1, 2)
# What each flask slot does. Changeable per slot (click the flask's name in the app).
KIND_NAMES = {"life": "Life", "mana": "Mana", "quicksilver": "Quicksilver",
              "utility": "Utility", "off": "Not used"}
BAR_KINDS = ("quicksilver", "utility")      # flasks with a timer strip under them


UTILITY_REFRESH_BELOW = 10    # % of the effect left before a utility flask is used again


GLOBE_BELOW = 50          # % of the globe's colour left before the flask is pressed
GLOBE_AGAIN_MS = 2000     # while the globe stays low, press again after this long


def slot_options(cfg: Optional[dict]) -> Dict[int, dict]:
    """Per-flask extras: own delay range and the globe link."""
    out = {}
    for sc in (cfg or {}).get("slots", []):
        o = {}
        if sc.get("delay_min_ms") is not None and sc.get("delay_max_ms") is not None:
            o["delay"] = (sc["delay_min_ms"] / 1000.0, sc["delay_max_ms"] / 1000.0)
        g = sc.get("globe")
        if g and g.get("box"):
            o["globe_below"] = float(g.get("below", GLOBE_BELOW))
            o["globe_again"] = float(g.get("again_ms", GLOBE_AGAIN_MS)) / 1000.0
        out[int(sc["slot"])] = o
    return out


def slot_refresh(cfg: Optional[dict]) -> Dict[int, float]:
    return {int(sc["slot"]): float(sc.get("refresh_below", UTILITY_REFRESH_BELOW))
            for sc in (cfg or {}).get("slots", [])}


def slot_kinds(cfg: Optional[dict]) -> Dict[int, str]:
    """{slot: kind} from the config, falling back to the original layout."""
    kinds = dict(SLOT_KINDS)
    for sc in (cfg or {}).get("slots", []):
        if sc.get("kind") in KIND_NAMES:
            kinds[int(sc["slot"])] = sc["kind"]
    return kinds
DEFAULT_BAR_RGB = (249, 215, 153)    # measured from your screenshot

TOP_BAND_FRAC = 0.15  # top 15% of the liquid decides "full"
FULL_MATCH_MIN = 0.6
VISIBLE_CORR_MIN = 0.45  # shape match of the non-liquid parts vs the reference
BAR_RGB_TOL = 40
BAR_MIN_PIXELS = 4


# ================================================================ config
def default_config() -> dict:
    return {
        "window_title": "",
        "bar_rgb": list(DEFAULT_BAR_RGB),
        "slots": [
            {"slot": s, "key": str(s), "kind": SLOT_KINDS[s], "box": None, "bar_box": None}
            for s in range(1, 6)
        ],
        "triggers": [],
    }


# ---------------------------------------------------------------- triggers
# "When this colour disappears, press this key." Each trigger watches one area
# (e.g. the blue layer on the life globe) picked like a flask's liquid.
TRIGGER_DEFAULTS = {
    "name": "Blue layer",
    "key": "q",
    "enabled": True,
    "fire_below": 20,         # % of the calibrated colour still showing -> start pressing
    "rearm_above": 50,        # % back -> stop pressing and wait for the next time
    "repeat": True,
    "repeat_min_ms": 200,
    "repeat_max_ms": 600,
    "box": None, "liquid_points": [], "similarity": 55,
    "pick_mode": "shape",
}


def new_trigger(cfg: dict) -> dict:
    ids = [t["id"] for t in cfg.get("triggers", [])]
    t = json.loads(json.dumps(TRIGGER_DEFAULTS))
    t["id"] = max(ids, default=0) + 1
    if ids:
        t["name"] = f"Trigger {t['id']}"
    return t


def trigger_ref_path(tid: int) -> Path:
    return DATA_DIR / f"trigger{tid}_reference.png"


def globe_ref_path(slot: int) -> Path:
    return DATA_DIR / f"globe{slot}_reference.png"


def controller_options(cfg: dict) -> dict:
    """Per-flask choices from the config, for FlaskController(**options)."""
    return {"kinds": slot_kinds(cfg), "refresh_below": slot_refresh(cfg), "slot_opts": slot_options(cfg)}


def validate_trigger(t: dict) -> List[str]:
    errs = []
    if not normalize_hotkey(t.get("key", "")):
        errs.append("Pick a key for the trigger to press.")
    if not 0 <= t["fire_below"] < t["rearm_above"] <= 100:
        errs.append("'Fire below' must be lower than 'Re-arm above' (0-100%).")
    if not 20 <= t["repeat_min_ms"] <= t["repeat_max_ms"] <= 60000:
        errs.append("Repeat: min must be ≤ max (20-60000 ms).")
    if not str(t.get("name", "")).strip():
        errs.append("Give the trigger a name.")
    return errs


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return default_config()


def save_config(cfg: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def slot_cfg(cfg: dict, slot: int) -> dict:
    return next(s for s in cfg["slots"] if s["slot"] == slot)


def ref_path(slot: int) -> Path:
    return DATA_DIR / f"slot{slot}_reference.png"


def default_bar_box(box: List[int]) -> List[int]:
    x, y, w, h = box
    return [x, y + h, w, max(8, int(round(h * 0.22)))]


# ================================================================ window (Windows)
class WindowLocator:
    """Finds your program's window and reports its client-area origin, so
    the boxes keep working when the window moves."""

    def __init__(self, title: str, screen_origin=(0, 0)):
        self.title = title.strip()
        self.screen_origin = tuple(screen_origin)
        self.win = sys.platform == "win32"
        self._own = set()
        if self.win:
            import ctypes
            from ctypes import wintypes
            self.ctypes, self.wintypes = ctypes, wintypes
            self.user32 = ctypes.windll.user32
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    self.user32.SetProcessDPIAware()
                except Exception:
                    pass
            console = ctypes.windll.kernel32.GetConsoleWindow()
            if console:
                self._own.add(console)

    def _title_of(self, hwnd) -> str:
        n = self.user32.GetWindowTextLengthW(hwnd)
        buf = self.ctypes.create_unicode_buffer(n + 1)
        self.user32.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value

    def find(self):
        if not (self.win and self.title):
            return None
        found = []
        needle = self.title.lower()

        @self.ctypes.WINFUNCTYPE(self.wintypes.BOOL, self.wintypes.HWND, self.wintypes.LPARAM)
        def cb(hwnd, _):
            if hwnd in self._own or not self.user32.IsWindowVisible(hwnd):
                return True
            t = self._title_of(hwnd)
            if needle in t.lower() and not t.startswith(APP + " |"):
                found.append(hwnd)
            return True

        self.user32.EnumWindows(cb, 0)
        return found[0] if found else None

    def list_titles(self) -> List[str]:
        """Titles of visible windows, for the picker in the app."""
        if not self.win:
            return []
        titles = []

        @self.ctypes.WINFUNCTYPE(self.wintypes.BOOL, self.wintypes.HWND, self.wintypes.LPARAM)
        def cb(hwnd, _):
            if hwnd not in self._own and self.user32.IsWindowVisible(hwnd):
                t = self._title_of(hwnd).strip()
                if t and not t.startswith(APP) and t not in titles and t != "Program Manager":
                    titles.append(t)
            return True

        self.user32.EnumWindows(cb, 0)
        return titles

    def bring_to_front(self) -> bool:
        hwnd = self.find() if self.win else None
        if not hwnd:
            return False
        self.user32.ShowWindow(hwnd, 9)          # SW_RESTORE
        self.user32.SetForegroundWindow(hwnd)
        return True

    def client_size(self) -> Optional[Tuple[int, int]]:
        """Inner size of your program's window (None = unknown)."""
        if not (self.win and self.title):
            return None
        hwnd = self.find()
        if not hwnd:
            return None
        r = self.wintypes.RECT()
        self.user32.GetClientRect(hwnd, self.ctypes.byref(r))
        return (r.right, r.bottom)

    def origin(self) -> Optional[Tuple[int, int]]:
        """Screen position of the window's client area (None = not found)."""
        if not (self.title and self.win):
            return self.screen_origin         # boxes are plain screen coords
        hwnd = self.find()
        if not hwnd:
            return None
        pt = self.wintypes.POINT(0, 0)
        self.user32.ClientToScreen(hwnd, self.ctypes.byref(pt))
        return (pt.x, pt.y)

    def is_focused(self) -> bool:
        if not (self.win and self.title):
            return True
        hwnd = self.find()
        return bool(hwnd) and self.user32.GetForegroundWindow() == hwnd


# ================================================================ capture
class ScreenGrabber:
    def __init__(self):
        import mss
        self._mss = mss
        self._local = threading.local()

    def _sct(self):
        if not hasattr(self._local, "sct"):
            self._local.sct = self._mss.mss()
        return self._local.sct

    def _reset(self):
        sct = getattr(self._local, "sct", None)
        if sct is not None:
            try:
                sct.close()
            except Exception:
                pass
            del self._local.sct

    def grab(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        region = {"left": left, "top": top, "width": width, "height": height}
        try:
            shot = self._sct().grab(region)
        except Exception:
            # Windows screen capture can go stale (e.g. after the app stopped
            # and started again) - start a fresh capture and try once more
            self._reset()
            shot = self._sct().grab(region)
        return np.ascontiguousarray(np.array(shot)[:, :, :3])

    def close(self):
        self._reset()

    def grab_window_or_screen(self, locator: WindowLocator) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Whole client area of your window (or the whole desktop)."""
        sct = self._sct()
        if locator.title and locator.win:
            hwnd = locator.find()
            if not hwnd:
                raise SystemExit(f'Could not find a window whose title contains "{locator.title}".')
            r = locator.wintypes.RECT()
            locator.user32.GetClientRect(hwnd, locator.ctypes.byref(r))
            ox, oy = locator.origin()
            return self.grab(ox, oy, r.right, r.bottom), (ox, oy)
        mon = sct.monitors[0]
        img = self.grab(mon["left"], mon["top"], mon["width"], mon["height"])
        return img, (mon["left"], mon["top"])


# ================================================================ detection model
def _circ_hue_dist(h: np.ndarray, centre: int) -> np.ndarray:
    d = np.abs(h.astype(int) - centre)
    return np.minimum(d, 180 - d)


HUE_NAMES = [(10, "red"), (22, "orange"), (35, "yellow"), (85, "green"),
             (100, "cyan"), (130, "blue"), (165, "purple"), (180, "red")]


def hue_name(h: int) -> str:
    return next(n for lim, n in HUE_NAMES if h < lim)


@dataclass
class SlotReading:
    visible: bool
    full: bool
    active: bool
    fill: float          # rough 0-1 estimate of how full the flask looks
    top_match: float     # share of the top-of-liquid pixels that still match
    sig: Optional[np.ndarray] = None   # small picture of the liquid area, to spot any change
    brightness: float = 1.0            # frame brightness vs. the calibration picture (pause = dimmer)
    remaining: float = 0.0             # share of the timer strip still showing (0 = not running, 1 = just used)
    globe: Optional[float] = None      # % of the linked globe's colour showing (None = no globe linked)


DEFAULT_SIMILARITY = 55   # calibration slider default (1-100)
SAT_FLOOR = 55            # never count washed-out (grey) pixels as liquid
VAL_FLOOR = 30            # ...or near-black ones
PICK_MODES = ("shape", "box")


def _seed(hsv: np.ndarray, x: int, y: int) -> Tuple[int, float, float]:
    """Liquid colour around a clicked spot: (hue, saturation, brightness)."""
    patch = hsv[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3).astype(float)
    ang = patch[:, 0] * (2 * np.pi / 180)
    hue = int(round((np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) * 180 / (2 * np.pi))) % 180
    return hue, float(np.median(patch[:, 1])), float(np.median(patch[:, 2]))


def colour_match(hsv: np.ndarray, seeds: List[Tuple[int, float, float]], similarity: float) -> np.ndarray:
    """Pixels whose colour is close enough to any clicked liquid colour.
    Higher similarity = wider hue range and darker/paler shades allowed,
    which is what shading on the liquid looks like."""
    f = max(1.0, min(100.0, float(similarity))) / 100.0
    hue_tol = 3 + 37 * f
    H, S, V = (hsv[..., i].astype(np.int16) for i in range(3))
    out = np.zeros(H.shape, bool)
    for h0, s0, v0 in seeds:
        # each click sets its own limits, so a click on a dark edge lets darker
        # liquid in; the floors stop grey glass ever counting as liquid
        out |= ((_circ_hue_dist(H, h0) <= hue_tol) & (S >= max(SAT_FLOOR, s0 * (1 - 0.85 * f)))
                & (V >= max(VAL_FLOOR, v0 * (1 - 0.9 * f))))
    return out


def liquid_selection(ref_bgr: np.ndarray, points: List[Tuple[int, int]],
                     similarity: float = DEFAULT_SIMILARITY, mode: str = "shape") -> np.ndarray:
    """The liquid area for the calibration slider.
    mode 'shape': matching pixels inside the flask's outline - the piece you
      clicked plus any other matching pieces that sit inside the bottle (e.g.
      green above and below the quicksilver wings), but nothing touching the
      edge of the box (arch, background).
    mode 'box': every matching pixel in the box."""
    h, w = ref_bgr.shape[:2]
    pts = [(int(x), int(y)) for x, y in points if 0 <= x < w and 0 <= y < h]
    if not pts:
        return np.zeros((h, w), bool)
    hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)
    match = colour_match(hsv, [_seed(hsv, x, y) for x, y in pts], similarity)
    # (no speck-removal filter here: it also deleted the thin 1-2 px liquid
    # along the glass edge; small loose specks are dropped below instead)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(match.astype(np.uint8), connectivity=8)
    clicked = set()
    for x, y in pts:                     # the piece under (or right next to) each click
        win = lab[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        ids = win[win > 0]
        if ids.size:
            clicked.add(int(np.bincount(ids).argmax()))
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x0, y0, bw, bh, area = stats[i]
        touches_edge = x0 == 0 or y0 == 0 or x0 + bw >= w or y0 + bh >= h
        if mode == "box":
            keep[i] = i in clicked or area >= 5          # everything except loose specks
        elif i in clicked or (area >= 0.004 * h * w and not touches_edge):
            keep[i] = True
    sel = keep[lab]
    if mode == "box" or not sel.any():
        return sel
    ys, xs = np.nonzero(sel)
    hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
    inside = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(inside, hull, 1)
    return match & (inside > 0)


def find_similar(img: np.ndarray, box: List[int], count: int = 2, min_score: float = 0.7) -> List[List[int]]:
    """Find other copies of the flask in `box` (e.g. the other quicksilvers).
    Returns up to `count` boxes, left to right."""
    x, y, w, h = box
    tmpl = img[y:y + h, x:x + w]
    if tmpl.size == 0 or img.shape[0] < h or img.shape[1] < w:
        return []
    res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
    res[max(0, y - h // 2):y + h // 2 + 1, max(0, x - w // 2):x + w // 2 + 1] = -1   # itself
    found = []
    for _ in range(count):
        _, score, _, (mx, my) = cv2.minMaxLoc(res)
        if score < min_score:
            break
        found.append([int(mx), int(my), w, h])
        res[max(0, my - h // 2):my + h // 2 + 1, max(0, mx - w // 2):mx + w // 2 + 1] = -1
    return sorted(found, key=lambda b: b[0])


class SlotModel:
    """Built from a picture of the flask when FULL plus the spots you clicked
    on its liquid. FULL = the top of the liquid still looks like it did in
    that picture, so any flask art and any colour shading works."""

    def __init__(self, ref_bgr: np.ndarray, liquid_points: Optional[List[Tuple[int, int]]] = None,
                 similarity: float = DEFAULT_SIMILARITY, mode: str = "shape"):
        self.ref = ref_bgr
        self.h, self.w = ref_bgr.shape[:2]
        self.problems: List[str] = []
        self.similarity = similarity
        pts = [(int(x), int(y)) for x, y in (liquid_points or []) if 0 <= x < self.w and 0 <= y < self.h]
        ref_hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)
        self.seeds = [_seed(ref_hsv, x, y) for x, y in pts]
        self.mask = liquid_selection(ref_bgr, pts, similarity, mode)
        if not self.mask.any():
            self.problems.append("no liquid found where you clicked - click on the liquid itself")

        hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)
        if self.mask.any():
            ang = hsv[..., 0][self.mask].astype(float) * (2 * np.pi / 180)
            self.hue = int(round((np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi))
                                 * 180 / (2 * np.pi))) % 180
        else:
            self.hue = 0

        # FULL is judged on the body of the liquid: thin 1-2 px strips along the
        # glass (or a thin rim at the top) are part of the pink area but are too
        # fragile to judge by
        core = cv2.erode(self.mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if core.sum() < 0.5 * self.mask.sum():
            core = self.mask
        rows = core.sum(1)
        if rows.max() == 0:
            self.top = self.bottom = 0
        else:
            good = np.nonzero(rows >= max(2, 0.15 * rows.max()))[0]
            self.top, self.bottom = int(good.min()), int(good.max())
        band_h = max(3, int(round(TOP_BAND_FRAC * (self.bottom - self.top + 1))))
        self.band = core.copy()
        self.band[: self.top] = False
        self.band[self.top + band_h:] = False
        if self.band.sum() < 15:                     # very thin liquid: fall back to all of it
            self.band = self.mask.copy()
            self.band[: self.top] = False
            self.band[self.top + band_h:] = False

        grown = cv2.dilate(self.mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.frame = ~grown
        self.ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        bright_px = self.frame & (self.ref_gray > 20)          # ignore pure black background
        self.bright_mask = bright_px if bright_px.sum() >= 30 else self.frame
        self.ref_bright = float(self.ref_gray[self.bright_mask].mean()) if self.bright_mask.any() else 0.0

        self.liquid_share = float(self.mask.mean())
        if self.mask.any() and self.liquid_share < 0.05:
            self.problems.append(f"only {self.liquid_share:.0%} of the box is liquid - "
                                 "click the parts of the liquid it missed")
        if self.mask.any() and self.band.sum() < 15:
            self.problems.append("top of the liquid is very small; 'full' may be unreliable")
        if self.frame.sum() < 0.1 * self.frame.size:
            self.problems.append("box has almost no frame/background; visibility check weak")

    def shape_match(self, gray: np.ndarray) -> float:
        """Correlation of the non-liquid parts (glass, cork, frame, number)
        with the reference. A loading screen or a covered belt scores ~0."""
        if not self.frame.any():
            return 1.0
        a = gray[self.frame].astype(float); b = self.ref_gray[self.frame].astype(float)
        a -= a.mean(); b -= b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den > 1e-6 else 0.0

    # -- live reading ------------------------------------------------------
    def read(self, crop_bgr: np.ndarray, bar_crop: Optional[np.ndarray], bar_bgr) -> SlotReading:
        if crop_bgr.shape[:2] != (self.h, self.w):
            crop_bgr = cv2.resize(crop_bgr, (self.w, self.h))
        # a pixel still counts as liquid if its colour is in the range you
        # chose with the slider - so shading, shimmer or movement inside the
        # liquid don't matter, but grey glass (drained) does
        same = colour_match(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV), self.seeds, self.similarity)

        top_match = float(same[self.band].mean()) if self.band.any() else 0.0
        full = top_match >= FULL_MATCH_MIN

        rows = slice(self.top, self.bottom + 1)
        per_row = self.mask[rows].sum(1)
        hit = (same & self.mask)[rows].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(per_row > 0, hit / np.maximum(per_row, 1), np.nan)
        valid = ~np.isnan(frac)
        fill = float((frac[valid] >= 0.5).mean()) if valid.any() else 0.0

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        visible = self.shape_match(gray) >= VISIBLE_CORR_MIN

        active = False
        remaining = 0.0
        if bar_crop is not None and bar_crop.size:
            dist = np.abs(bar_crop.astype(int) - np.array(bar_bgr)).max(axis=2)
            hits = dist <= BAR_RGB_TOL
            active = int(hits.sum()) >= BAR_MIN_PIXELS
            if active:
                # the strip shrinks along its long side as the effect runs out; compare its
                # length with the longest it has been (a fresh flask fills the strip)
                along = hits.any(axis=0) if hits.shape[1] >= hits.shape[0] else hits.any(axis=1)
                length = int(along.sum())
                self.bar_longest = max(getattr(self, "bar_longest", 0), length)
                remaining = length / max(1, self.bar_longest)

        sig = None
        if self.mask.any():
            ys, xs = np.nonzero(self.mask)
            area = crop_bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            sig = cv2.resize(cv2.cvtColor(area, cv2.COLOR_BGR2GRAY), (12, 20),
                             interpolation=cv2.INTER_AREA).astype(np.float32)
        brightness = 1.0
        if self.ref_bright > 1 and self.bright_mask.any():
            brightness = float(gray[self.bright_mask].mean()) / self.ref_bright
        return SlotReading(visible, full and visible, active and visible, fill, top_match, sig, brightness,
                           remaining if visible else 0.0)


def slot_points(s: dict) -> List[Tuple[int, int]]:
    """Clicked liquid spots for a slot (relative to its box). Older configs
    stored a single spot as liquid_point."""
    pts = s.get("liquid_points")
    if pts:
        return [tuple(p) for p in pts]
    if s.get("liquid_point"):
        return [tuple(s["liquid_point"])]
    return []


class TriggerModel:
    """How much of the calibrated colour is still showing in a trigger's box (0-100%)."""

    def __init__(self, ref_bgr: np.ndarray, points, similarity: float, mode: str):
        self.h, self.w = ref_bgr.shape[:2]
        pts = [(int(x), int(y)) for x, y in points if 0 <= x < self.w and 0 <= y < self.h]
        hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)
        self.seeds = [_seed(hsv, x, y) for x, y in pts]
        self.similarity = similarity
        self.mask = liquid_selection(ref_bgr, pts, similarity, mode)

    def coverage(self, crop_bgr: np.ndarray) -> Optional[float]:
        if not self.mask.any():
            return None
        if crop_bgr.shape[:2] != (self.h, self.w):
            crop_bgr = cv2.resize(crop_bgr, (self.w, self.h))
        hit = colour_match(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV), self.seeds, self.similarity)
        return 100.0 * float(hit[self.mask].mean())


def trigger_model(t: dict) -> Optional[TriggerModel]:
    p = trigger_ref_path(t["id"])
    if not (t.get("box") and p.exists()):
        return None
    ref = cv2.imread(str(p))
    return TriggerModel(ref, [tuple(q) for q in t.get("liquid_points", [])],
                        t.get("similarity", DEFAULT_SIMILARITY), t.get("pick_mode", "shape"))


def model_for(s: dict) -> SlotModel:
    return SlotModel(cv2.imread(str(ref_path(s["slot"]))), slot_points(s),
                     s.get("similarity", DEFAULT_SIMILARITY), s.get("pick_mode", "shape"))


# ================================================================ auto-detection
AUTO_PREV_MIN = 0.8        # how closely a saved flask must match to be reused


def _colour_candidates(mask: np.ndarray, img_shape) -> List[dict]:
    ih, iw = img_shape[:2]
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))   # join pieces split by art
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a < 120 or w > 0.3 * iw or h > 0.6 * ih or w < 8 or h < 12:
            continue                                   # specks, health/mana globes, etc.
        if not 0.6 <= h / w <= 5 or a / (w * h) < 0.25:
            continue
        comp = (lab == i).astype(np.uint8)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        py, px = np.unravel_index(int(np.argmax(dist)), dist.shape)   # deepest point in the liquid
        out.append({"x": x, "y": y, "w": w, "h": h, "a": a, "cx": x + w / 2, "cy": y + h / 2,
                    "point": (int(px), int(py))})
    return out


def _detect_by_colour(img: np.ndarray) -> Dict[int, dict]:
    """Red, blue and three evenly spaced green liquids in a row."""
    import itertools
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = (hsv[..., i].astype(int) for i in range(3))
    sat = (S >= 90) & (V >= 45)
    cands = {
        "life": _colour_candidates(sat & ((H <= 10) | (H >= 170)), img.shape),
        "mana": _colour_candidates(sat & (H >= 100) & (H <= 130), img.shape),
        "quicksilver": _colour_candidates(sat & (H >= 40) & (H <= 90), img.shape),
    }
    best = None
    greens = sorted(cands["quicksilver"], key=lambda d: d["cx"])
    for a, b, c in itertools.combinations(greens, 3):
        p1, p2 = b["cx"] - a["cx"], c["cx"] - b["cx"]
        hs = [a["h"], b["h"], c["h"]]
        if p1 <= 0 or p2 <= 0 or abs(p1 - p2) > 0.2 * max(p1, p2) or max(hs) > 1.4 * min(hs):
            continue
        if max(abs(a["cy"] - b["cy"]), abs(b["cy"] - c["cy"])) > 0.3 * max(hs):
            continue
        pitch = (p1 + p2) / 2
        if pitch < 0.8 * max(a["w"], b["w"], c["w"]):
            continue
        score = -abs(p1 - p2) - float(np.std(hs))
        if best is None or score > best[0]:
            best = (score, [a, b, c], pitch)
    if not best:
        return {}
    _, qs, pitch = best
    liquids = {3: qs[0], 4: qs[1], 5: qs[2]}
    for slot, kind, k in ((2, "mana", 1), (1, "life", 2)):
        tx = qs[0]["cx"] - k * pitch
        near = [d for d in cands[kind]
                if abs(d["cx"] - tx) < 0.4 * pitch and abs(d["cy"] - qs[0]["cy"]) < 0.8 * qs[0]["h"]]
        if near:
            liquids[slot] = max(near, key=lambda d: d["a"])

    # liquid -> whole-bottle box (neck above, base below), one shared row height
    ih, iw = img.shape[:2]
    top = min(d["y"] - 0.6 * d["h"] for d in liquids.values())
    bottom = max(d["y"] + d["h"] * 1.15 for d in liquids.values())
    y0, y1 = max(0, int(top)), min(ih, int(round(bottom)))
    w = int(round(pitch * 0.92))
    out = {}
    for slot, d in liquids.items():
        x0 = max(0, min(iw - w, int(round(d["cx"] - w / 2))))
        out[slot] = {"box": [x0, y0, w, y1 - y0], "point": d["point"], "source": "colour",
                     "confidence": 0.8}
    return out


def auto_detect(img: np.ndarray, cfg: dict) -> Dict[int, dict]:
    """Best guess for every flask: {slot: {box, point (image coords), similarity,
    mode, bar_box, source, confidence, unsure}}. Uses your saved calibration
    first (if the flasks look the same), then colour detection."""
    found: Dict[int, dict] = {}
    for s in cfg["slots"]:
        p = ref_path(s["slot"])
        if not (s.get("box") and p.exists()):
            continue
        ref = cv2.imread(str(p))
        if ref is None or ref.shape[0] > img.shape[0] or ref.shape[1] > img.shape[1]:
            continue
        res = cv2.matchTemplate(img, ref, cv2.TM_CCOEFF_NORMED)
        _, score, _, (mx, my) = cv2.minMaxLoc(res)
        if score >= AUTO_PREV_MIN:
            box = [int(mx), int(my), ref.shape[1], ref.shape[0]]
            pts = slot_points(s)
            entry = {"box": box, "point": (mx + pts[0][0], my + pts[0][1]) if pts else None,
                     "points": [(mx + px, my + py) for px, py in pts],
                     "similarity": s.get("similarity", DEFAULT_SIMILARITY), "mode": s.get("pick_mode", "shape"),
                     "source": "previous", "confidence": float(score)}
            if s.get("bar_box"):
                ob = s["box"]
                b = s["bar_box"]
                entry["bar_box"] = [mx + b[0] - ob[0], my + b[1] - ob[1], b[2], b[3]]
            found[s["slot"]] = entry

    if len(found) < 5:
        for slot, d in _detect_by_colour(img).items():
            if slot not in found:
                found[slot] = d

    for slot, d in found.items():
        d.setdefault("points", [d["point"]] if d.get("point") else [])
        d.setdefault("similarity", DEFAULT_SIMILARITY)
        d.setdefault("mode", "shape")
        if slot_kinds(cfg).get(slot) in BAR_KINDS and not d.get("bar_box"):
            d["bar_box"] = default_bar_box(d["box"])
        # unsure if the liquid it would pick looks too small
        x, y, w, h = d["box"]
        rel = [(px - x, py - y) for px, py in d["points"]]
        share = float(liquid_selection(img[y:y + h, x:x + w], rel, d["similarity"], d["mode"]).mean()) if rel else 0
        d["liquid_share"] = share
        d["unsure"] = d["confidence"] < 0.75 or share < 0.05
    return found


# ================================================================ reader
class FlaskReader:
    """Grabs the flask area once per check and reads every slot."""

    def __init__(self, cfg: dict, locator: Optional[WindowLocator] = None,
                 grabber: Optional[ScreenGrabber] = None,
                 frame_source: Optional[Callable[[], np.ndarray]] = None):
        self.cfg = cfg
        self.locator = locator
        self.grabber = grabber
        self.frame_source = frame_source      # for testing on an image
        self.bar_bgr = tuple(reversed(cfg.get("bar_rgb", DEFAULT_BAR_RGB)))
        self.models: Dict[int, SlotModel] = {}
        for s in cfg["slots"]:
            if s["box"] is None or not ref_path(s["slot"]).exists():
                raise SystemExit(f"Slot {s['slot']} is not calibrated. Run: calibrate")
            self.models[s["slot"]] = model_for(s)
        self.trigger_models: Dict[int, TriggerModel] = {}
        for t in cfg.get("triggers", []):
            m = trigger_model(t) if t.get("enabled", True) else None
            if m is not None:
                self.trigger_models[t["id"]] = m
        self.trigger_boxes = {t["id"]: t["box"] for t in cfg.get("triggers", []) if t["id"] in self.trigger_models}
        self.globe_models: Dict[int, TriggerModel] = {}
        self.globe_boxes: Dict[int, List[int]] = {}
        for s in cfg["slots"]:
            g = s.get("globe")
            if g and g.get("box") and globe_ref_path(s["slot"]).exists():
                self.globe_models[s["slot"]] = TriggerModel(
                    cv2.imread(str(globe_ref_path(s["slot"]))), [tuple(q) for q in g.get("liquid_points", [])],
                    g.get("similarity", DEFAULT_SIMILARITY), g.get("pick_mode", "shape"))
                self.globe_boxes[s["slot"]] = g["box"]
        self.trigger_values: Dict[int, Optional[float]] = {}
        self.trigger_crops: Dict[int, np.ndarray] = {}
        boxes = [s["box"] for s in cfg["slots"]] + [s["bar_box"] for s in cfg["slots"] if s["bar_box"]] + \
            list(self.trigger_boxes.values()) + list(self.globe_boxes.values())
        self.bx0 = min(b[0] for b in boxes)
        self.by0 = min(b[1] for b in boxes)
        self.bx1 = max(b[0] + b[2] for b in boxes)
        self.by1 = max(b[1] + b[3] for b in boxes)
        self.last_crops: Dict[int, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        self.expected_size = tuple(cfg["window_size"]) if cfg.get("window_size") else None
        self.problem = ""          # why the last read returned nothing (shown in the app)
        self.size_mismatch: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

    def _region(self) -> Optional[np.ndarray]:
        if self.frame_source is not None:
            img = self.frame_source()
            return img[self.by0:self.by1, self.bx0:self.bx1]
        origin = self.locator.origin()
        if origin is None:
            return None
        ox, oy = origin
        return self.grabber.grab(ox + self.bx0, oy + self.by0, self.bx1 - self.bx0, self.by1 - self.by0)

    def read(self) -> Optional[Dict[int, SlotReading]]:
        self.trigger_values = {}
        # If the window was resized since calibration the boxes no longer line
        # up: report nothing (so nothing gets pressed) and flag it for the app.
        if self.frame_source is None and self.expected_size and self.locator:
            size = self.locator.client_size()
            if size and tuple(size) != self.expected_size:
                self.size_mismatch = (tuple(size), self.expected_size)
                self.problem = "window resized"
                return None
        self.size_mismatch = None
        try:
            region = self._region()
        except Exception as e:
            note_error("screen capture")
            self.problem = f"screen capture failed ({e.__class__.__name__})"
            return None
        if region is None:
            self.problem = "window not found"
            return None
        self.problem = ""
        out = {}
        for s in self.cfg["slots"]:
            x, y, w, h = s["box"]
            crop = region[y - self.by0:y - self.by0 + h, x - self.bx0:x - self.bx0 + w]
            bar = None
            if s["bar_box"]:
                bx, by, bw, bh = s["bar_box"]
                bar = region[by - self.by0:by - self.by0 + bh, bx - self.bx0:bx - self.bx0 + bw]
            self.last_crops[s["slot"]] = (crop, bar)
            out[s["slot"]] = self.models[s["slot"]].read(crop, bar, self.bar_bgr)
            if s["slot"] in self.globe_boxes:
                gx, gy, gw, gh = self.globe_boxes[s["slot"]]
                gcrop = region[gy - self.by0:gy - self.by0 + gh, gx - self.bx0:gx - self.bx0 + gw]
                out[s["slot"]].globe = self.globe_models[s["slot"]].coverage(gcrop)
        vals = {}
        for tid, (x, y, w, h) in self.trigger_boxes.items():
            crop = region[y - self.by0:y - self.by0 + h, x - self.bx0:x - self.bx0 + w]
            self.trigger_crops[tid] = crop
            vals[tid] = self.trigger_models[tid].coverage(crop)
        self.trigger_values = vals
        return out


# ================================================================ controller
@dataclass(order=True)
class _Pending:
    fire_at: float
    slot: int = field(compare=False)


class FlaskController:
    def __init__(self, read: Callable[[], Optional[Dict[int, SlotReading]]],
                 press: Callable[[int], bool],
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 rng: Optional[random.Random] = None,
                 log: Callable[[str], None] = print,
                 settings: Optional[dict] = None,
                 triggers: Optional[List[dict]] = None,
                 read_triggers: Callable[[], Dict[int, Optional[float]]] = lambda: {},
                 press_key: Optional[Callable[[str], bool]] = None,
                 kinds: Optional[Dict[int, str]] = None,
                 refresh_below: Optional[Dict[int, float]] = None,
                 slot_opts: Optional[Dict[int, dict]] = None):
        self.read, self.press = read, press
        self.kinds = dict(kinds or SLOT_KINDS)
        self.quick = tuple(sorted(sl for sl, k in self.kinds.items() if k == "quicksilver"))
        self.instill = tuple(sorted(sl for sl, k in self.kinds.items() if k in ("life", "mana")))
        self.utility = tuple(sorted(sl for sl, k in self.kinds.items() if k == "utility"))
        self.refresh_below: Dict[int, float] = dict(refresh_below or {})
        self.slot_opts: Dict[int, dict] = dict(slot_opts or {})
        self.last_globe_press: Dict[int, float] = {}
        self.triggers = [t for t in (triggers or []) if t.get("enabled", True)]
        self.read_triggers = read_triggers
        self.press_key = press_key or (lambda combo: False)
        self.tstate: Dict[int, dict] = {t["id"]: {"gone": False, "next": 0.0} for t in self.triggers}
        st = settings or DEFAULT_SETTINGS
        self.poll_interval = 1.0 / float(st["checks_per_second"])
        self.delay_range = (st["delay_min_ms"] / 1000.0, st["delay_max_ms"] / 1000.0)
        self.cooldown = st["refire_cooldown_ms"] / 1000.0
        self.verify_window = st["verify_ms"] / 1000.0
        self.fails_for_town = int(st["fails_for_town"])
        self.town_retry = float(st.get("town_retry_secs", 0))
        self.program_paused = False      # your program is paused (belt dimmed)
        self.probe_slot: Optional[int] = None
        self.last_probe = -1e9
        self.on_town: Callable[[bool], None] = lambda on: None
        self.clock, self.sleep = clock, sleep
        self.rng = rng or random.Random()
        self.log = log

        self.queue: List[_Pending] = []
        self.queued: set = set()
        self.last_press: Dict[int, float] = {}
        self.verify: Dict[int, float] = {}       # slot -> deadline
        self.before: Dict[int, Optional[np.ndarray]] = {}   # slot -> look just before the press
        self.last: Dict[int, SlotReading] = {}
        self.fails = 0
        self.failed_slots: set = set()
        self.wait_refill: set = set()      # used; don't press again until it has read NOT FULL
        self.warned_full: set = set()
        self.town_mode = False
        self.hidden_streak = 0
        self.belt_hidden_in_town = False
        self.prev_full: Dict[int, bool] = {}
        self.paused = False
        self.running = False

    # -- town mode ---------------------------------------------------------
    def enter_town(self, why: str) -> None:
        if not self.town_mode:
            self.town_mode = True
            self.belt_hidden_in_town = False
            self.probe_slot = None
            self.last_probe = self.clock()
            self.queue.clear(); self.queued.clear(); self.verify.clear()
            self.log(f"TOWN MODE ON  - {why}. Not pressing flasks until the next loading screen.")
            self.on_town(True)

    def leave_town(self, why: str) -> None:
        if self.town_mode:
            self.town_mode = False
            self.fails = 0
            self.failed_slots.clear()
            self.log(f"TOWN MODE OFF - {why}.")
            self.on_town(False)

    # -- one check -----------------------------------------------------------
    def poll(self) -> None:
        now = self.clock()
        r = self.read()
        if r is None:
            return

        if sum(x.visible for x in r.values()) < 3:
            self.hidden_streak += 1
            if self.town_mode and self.hidden_streak >= HIDDEN_POLLS_FOR_AREA_CHANGE:
                self.belt_hidden_in_town = True
            self.queue.clear(); self.queued.clear()
            return
        self.hidden_streak = 0
        if self.town_mode and self.belt_hidden_in_town:
            self.leave_town("flask belt was hidden and came back (area change)")

        # paused program: the belt dims. Press nothing, count nothing, resume as soon as it brightens.
        vis = [x.brightness for x in r.values() if x.visible]
        dim = bool(vis) and float(np.median(vis)) < PAUSE_DIM
        if dim != self.program_paused:
            self.program_paused = dim
            self.log("Your program looks paused - waiting." if dim else "Your program is running again - resuming.")
        if dim:
            self.queue.clear(); self.queued.clear(); self.verify.clear()
            self.before.clear()
            self.prev_full = {k: v.full for k, v in r.items()}
            return

        # did recent presses do anything?
        for slot, deadline in list(self.verify.items()):
            if not r[slot].full or r[slot].active or self._changed(slot, r[slot]):
                self.verify.pop(slot, None)
                if slot in self.quick:
                    self.fails = 0
                    self.failed_slots.clear()
                self.wait_refill.add(slot)
                if self.town_mode and slot == self.probe_slot:
                    self.probe_slot = None
                    self.leave_town(f"quicksilver {slot} worked again")
            elif now >= deadline:
                self.verify.pop(slot, None)
                self.wait_refill.discard(slot)
                if self.town_mode:
                    if slot == self.probe_slot:
                        self.probe_slot = None
                    continue
                if slot not in self.quick:
                    continue      # life/mana can't be drunk at full life/mana - not a sign of town
                self.fails += 1
                self.failed_slots.add(slot)
                self.log(f"[{now:8.3f}] slot {slot}: press had no effect ({self.fails}/{self.fails_for_town})")
                # a real town stops every quicksilver, so one misbehaving flask isn't enough
                usable = sum(1 for q in self.quick if r[q].visible and r[q].full)
                if self.fails >= self.fails_for_town and len(self.failed_slots) >= min(2, max(1, usable)):
                    self.enter_town("flasks are not being used")
                    break

        # a flask emptied without us pressing it = flasks work again
        for slot, rd in r.items():
            was_full = self.prev_full.get(slot)
            recently = now - self.last_press.get(slot, -1e9) < self.verify_window + 0.5
            if was_full and not rd.full and rd.visible and not recently and self.town_mode:
                self.leave_town(f"slot {slot} was used manually")
            self.prev_full[slot] = rd.full
            if not rd.full and rd.visible:
                self.wait_refill.discard(slot)        # it emptied - may fire again once full
                self.warned_full.discard(slot)
            elif (slot in self.wait_refill and slot not in self.warned_full and slot not in self.verify
                  and now - self.last_press.get(slot, -1e9) > 5.0):
                self.warned_full.add(slot)
                self.log(f"Flask {slot} still reads FULL after being used - its pink area may include "
                         "empty glass. Recalibrate it with a lower Similarity.")
        self.last = r

        if self.paused:
            return
        if not self.town_mode:
            self._run_triggers(now)
        if self.town_mode:
            # every few seconds, quietly try one quicksilver; if it works, town mode ends
            if (self.town_retry > 0 and self.probe_slot is None and now - self.last_probe >= self.town_retry
                    and not self.queue):
                for slot in sorted(self.quick, key=lambda q: self.last_press.get(q, -1e9)):
                    if r[slot].visible and r[slot].full and slot not in self.verify:
                        self.probe_slot = slot
                        self.last_probe = now
                        self._schedule(slot, now)
                        break
            return

        if not self._quicksilver_running(r, now):
            # after a quicksilver didn't respond, try a different one next
            for slot in sorted(self.quick, key=lambda q: (q in self.failed_slots, q)):
                if r[slot].visible and r[slot].full and self._can_schedule(slot, now):
                    self._schedule(slot, now)
                    break
        for slot in self.instill:
            rd = r[slot]
            if rd.visible and rd.full and self._can_schedule(slot, now):
                self._schedule(slot, now)
        # utility flasks: keep each one's effect going on its own (no taking turns)
        for slot in self.utility:
            rd = r[slot]
            low = (not rd.active) or rd.remaining * 100 < self.refresh_below.get(slot, 10)
            if (rd.visible and rd.full and low and self._can_schedule(slot, now)
                    and now - self.last_press.get(slot, -1e9) >= ACTIVATION_GRACE + 1.0):
                self._schedule(slot, now)
        # globe link: also press a flask whenever its globe drops below the chosen %
        for slot, o in self.slot_opts.items():
            if "globe_below" not in o or self.kinds.get(slot) == "off" or slot not in r:
                continue
            rd = r[slot]
            if (rd.visible and rd.globe is not None and rd.globe < o["globe_below"]
                    and slot not in self.queued and slot not in self.verify
                    and now - self.last_press.get(slot, -1e9) >= self.cooldown
                    and now - self.last_globe_press.get(slot, -1e9) >= o["globe_again"]):
                self.last_globe_press[slot] = now
                self._schedule(slot, now, f"globe at {rd.globe:.0f}% (below {o['globe_below']:g}%)")

    def _run_triggers(self, now: float) -> None:
        values = self.read_triggers() or {}
        for i, t in enumerate(self.triggers):
            c = values.get(t["id"])
            st = self.tstate[t["id"]]
            if c is None:
                continue
            if not st["gone"] and c < t["fire_below"]:
                st["gone"] = True
                st["next"] = now + self.rng.uniform(*self.delay_range)
                self.log(f"[{now:8.3f}] {t['name']}: gone ({c:.0f}%) -> pressing {hotkey_label(t['key'])}")
            elif st["gone"] and c >= t["rearm_above"]:
                st["gone"] = False
                self.log(f"[{now:8.3f}] {t['name']}: back ({c:.0f}%) - stopped pressing")
            tag = TRIGGER_TAG + i
            if st["gone"] and tag not in self.queued and st["next"] != float("inf"):
                fire = max(now, st["next"])           # exact time, not the next check
                heapq.heappush(self.queue, _Pending(fire, tag))
                self.queued.add(tag)
                if t.get("repeat", True):
                    st["next"] = fire + self.rng.uniform(t["repeat_min_ms"], t["repeat_max_ms"]) / 1000.0
                else:
                    st["next"] = float("inf")       # once per disappearance

    def _changed(self, slot: int, rd: SlotReading) -> bool:
        """Did the flask visibly change since just before it was pressed?"""
        before = self.before.get(slot)
        if before is None or rd.sig is None or before.shape != rd.sig.shape:
            return False
        return float(np.abs(rd.sig - before).mean()) > CHANGE_MIN

    def _quicksilver_running(self, r: Dict[int, SlotReading], now: float) -> bool:
        for slot in self.quick:
            if r[slot].active or slot in self.queued or slot in self.verify:
                return True
            if now - self.last_press.get(slot, -1e9) < ACTIVATION_GRACE:
                return True
        return False

    def _can_schedule(self, slot: int, now: float) -> bool:
        if slot in self.queued or slot in self.verify or slot in self.wait_refill:
            return False
        return now - self.last_press.get(slot, -1e9) >= self.cooldown

    def _schedule(self, slot: int, now: float, why: str = "full") -> None:
        delay = self.rng.uniform(*self.slot_opts.get(slot, {}).get("delay", self.delay_range))
        heapq.heappush(self.queue, _Pending(now + delay, slot))
        self.queued.add(slot)
        self.log(f"[{now:8.3f}] slot {slot} {why} -> press in {delay * 1000:.0f} ms")

    def _fire_due(self) -> None:
        now = self.clock()
        while self.queue and self.queue[0].fire_at <= now:
            p = heapq.heappop(self.queue)
            self.queued.discard(p.slot)
            if self.paused or self.program_paused or (self.town_mode and p.slot != self.probe_slot):
                continue
            if p.slot >= TRIGGER_TAG:
                t = self.triggers[p.slot - TRIGGER_TAG]
                if not self.tstate[t["id"]]["gone"]:
                    continue                          # it came back meanwhile
                if self.press_key(t["key"]):
                    self.log(f"[{now:8.3f}] {t['name']}: pressed {hotkey_label(t['key'])}")
                continue
            if self.press(p.slot):
                self.last_press[p.slot] = now
                self.verify[p.slot] = now + self.verify_window
                prev = self.last.get(p.slot)
                self.before[p.slot] = prev.sig if prev is not None else None
                self.log(f"[{now:8.3f}] pressed {p.slot}")
            else:
                self.log(f"[{now:8.3f}] skipped {p.slot} (your program's window isn't focused)")

    def run(self, duration: Optional[float] = None) -> None:
        self.running = True
        start = next_poll = self.clock()
        while self.running:
            now = self.clock()
            if duration is not None and now - start >= duration:
                break
            if now >= next_poll:
                try:
                    self.poll()
                except Exception as e:           # keep going; report once per kind of error
                    note_error("controller check")
                    tb = traceback.extract_tb(e.__traceback__)[-1]
                    msg = f"Error while reading the flasks: {e.__class__.__name__}: {e} ({tb.name}, line {tb.lineno})"
                    if msg != getattr(self, "_last_error", None):
                        self._last_error = msg
                        self.log(msg)
                next_poll += self.poll_interval
                if next_poll < now:                 # fell behind; don't burst
                    next_poll = now + self.poll_interval
            self._fire_due()
            wake = next_poll if not self.queue else min(next_poll, self.queue[0].fire_at)
            self.sleep(max(0.0, wake - self.clock()))


# ================================================================ keyboard
class HotkeyListener:
    """Watches the keyboard everywhere (also while your program has focus)
    and calls on_hotkey(setting_name) when a configured combination is pressed."""

    def __init__(self, settings: dict, on_hotkey: Callable[[str], None]):
        self.on_hotkey = on_hotkey
        self.paused = False                  # e.g. while recording a new hotkey
        self.mods: set = set()
        self.listener = None
        self.update(settings)

    def update(self, settings: dict) -> None:
        self.combos = {normalize_hotkey(settings.get(k, "")): k for k in HOTKEY_NAMES}
        self.combos.pop("", None)

    def start(self) -> bool:
        try:
            from pynput import keyboard
        except Exception:
            return False
        self._kb = keyboard
        self.listener = keyboard.Listener(on_press=self._press, on_release=self._release)
        self.listener.daemon = True
        self.listener.start()
        return True

    def stop(self) -> None:
        if self.listener:
            self.listener.stop()
            self.listener = None

    def _mod(self, key) -> Optional[str]:
        K = self._kb.Key
        if key in (K.shift, K.shift_l, K.shift_r):
            return "shift"
        if key in (K.ctrl, K.ctrl_l, K.ctrl_r):
            return "ctrl"
        if key in (K.alt, K.alt_l, K.alt_r, getattr(K, "alt_gr", K.alt)):
            return "alt"
        return None

    def _name(self, key) -> Optional[str]:
        vk = getattr(key, "vk", None)
        if vk is None and hasattr(key, "value"):
            vk = getattr(key.value, "vk", None)
        if vk is not None:
            n = vk_to_name(int(vk))
            if n:
                return n
        if hasattr(key, "name") and key.name:
            return key.name.lower()
        ch = getattr(key, "char", None)
        return ch.lower() if ch else None

    def _press(self, key):
        m = self._mod(key)
        if m:
            self.mods.add(m)
            return
        if self.paused:
            return
        name = self._name(key)
        if not name:
            return
        combo = "+".join([x for x in MODIFIERS if x in self.mods] + [name])
        if combo in self.combos:
            self.on_hotkey(self.combos[combo])

    def _release(self, key):
        m = self._mod(key)
        if m:
            self.mods.discard(m)


class KeySender:
    """Presses a flask key like a real keyboard would.

    On Windows it uses SendInput with the key's hardware scan code (the
    default), so programs that read raw keyboard input see it too. The old
    method typed the character, which many programs ignore."""

    def __init__(self, cfg: dict, locator: WindowLocator, rng: Optional[random.Random] = None):
        st = get_settings(cfg)
        self.keys = {s["slot"]: st["keys"][str(s["slot"])] for s in cfg["slots"]}
        self.hold_range = (st["key_hold_min_ms"] / 1000.0, st["key_hold_max_ms"] / 1000.0)
        self.method = st.get("key_method", "scancode")
        self.locator = locator
        self.rng = rng or random.Random()
        self.win = sys.platform == "win32" and self.method != "typed"
        self.fallback_reason = ""
        if self.win:
            try:
                self._setup_win()
            except Exception as e:                       # fall back rather than fail to start
                self.win = False
                self.fallback_reason = f"{e.__class__.__name__}: {e}"
        if not self.win:
            from pynput.keyboard import Controller
            self.kb = Controller()

    def _setup_win(self):
        import ctypes
        from ctypes import wintypes
        ULONG_PTR = ctypes.c_size_t

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

        class _U(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("u", _U)]

        self._ct, self._INPUT, self._KEYBDINPUT = ctypes, INPUT, KEYBDINPUT
        self._user32 = ctypes.windll.user32
        self._user32.VkKeyScanW.restype = ctypes.c_short
        self._codes = {}
        for slot, k in self.keys.items():
            vk = self._user32.VkKeyScanW(ord(k)) & 0xFF
            scan = self._user32.MapVirtualKeyW(vk, 0)          # MAPVK_VK_TO_VSC
            self._codes[slot] = (vk, scan)

    def _send_win(self, slot: int, up: bool) -> int:
        vk, scan = self._codes[slot]
        KEYUP, SCANCODE = 0x0002, 0x0008
        if self.method == "scancode":
            ki = self._KEYBDINPUT(0, scan, SCANCODE | (KEYUP if up else 0), 0, 0)
        else:
            ki = self._KEYBDINPUT(vk, scan, KEYUP if up else 0, 0, 0)
        inp = self._INPUT(1)                                   # INPUT_KEYBOARD
        inp.u.ki = ki
        return self._user32.SendInput(1, self._ct.byref(inp), self._ct.sizeof(self._INPUT))

    NAME_TO_VK = {**{chr(c).lower(): c for c in range(65, 91)}, **{chr(c): c for c in range(48, 58)},
                  **{f"f{i}": 111 + i for i in range(1, 13)}, **{f"num{i}": 96 + i for i in range(10)},
                  "space": 32, "tab": 9, "enter": 13, "backspace": 8, "insert": 45, "delete": 46, "home": 36,
                  "end": 35, "page_up": 33, "page_down": 34, "left": 37, "up": 38, "right": 39, "down": 40,
                  "`": 192, "-": 189, "=": 187, "[": 219, "]": 221, "\\": 220, ";": 186, "'": 222, ",": 188,
                  ".": 190, "/": 191, "ctrl": 0x11, "alt": 0x12, "shift": 0x10}

    def _send_vk_win(self, vk: int, up: bool) -> None:
        scan = self._user32.MapVirtualKeyW(vk, 0)
        KEYUP, SCANCODE, EXTENDED = 0x0002, 0x0008, 0x0001
        ext = EXTENDED if vk in (33, 34, 35, 36, 37, 38, 39, 40, 45, 46) else 0
        if self.method == "scancode":
            ki = self._KEYBDINPUT(0, scan, SCANCODE | ext | (KEYUP if up else 0), 0, 0)
        else:
            ki = self._KEYBDINPUT(vk, scan, ext | (KEYUP if up else 0), 0, 0)
        inp = self._INPUT(1)
        inp.u.ki = ki
        self._user32.SendInput(1, self._ct.byref(inp), self._ct.sizeof(self._INPUT))

    def press_combo(self, combo: str, check_focus: bool = True) -> bool:
        """Press any key or combination, e.g. 'q' or 'shift+q'."""
        combo = normalize_hotkey(combo)
        if not combo or (check_focus and not self.locator.is_focused()):
            return False
        parts = combo.split("+")
        mods, key = parts[:-1], parts[-1]
        hold = self.hold_time()
        if self.win:
            vks = [self.NAME_TO_VK[m] for m in mods]
            kvk = self.NAME_TO_VK.get(key)
            if kvk is None:
                return False
            for v in vks:
                self._send_vk_win(v, up=False)
            self._send_vk_win(kvk, up=False)
            time.sleep(hold)
            self._send_vk_win(kvk, up=True)
            for v in reversed(vks):
                self._send_vk_win(v, up=True)
            return True
        from pynput.keyboard import Key
        pk = {"shift": Key.shift, "ctrl": Key.ctrl, "alt": Key.alt}
        main = getattr(Key, key, None) if len(key) > 1 else key
        if main is None:
            return False
        for m in mods:
            self.kb.press(pk[m])
        self.kb.press(main)
        time.sleep(hold)
        self.kb.release(main)
        for m in reversed(mods):
            self.kb.release(pk[m])
        return True

    def hold_time(self) -> float:
        return self.rng.uniform(*self.hold_range)

    def __call__(self, slot: int, check_focus: bool = True) -> bool:
        if check_focus and not self.locator.is_focused():
            return False
        hold = self.hold_time()
        if self.win:
            self._send_win(slot, up=False)
            time.sleep(hold)
            self._send_win(slot, up=True)
        else:
            k = self.keys[slot]
            self.kb.press(k)
            time.sleep(hold)
            self.kb.release(k)
        return True


# ================================================================ diagnostics
def write_report(dest_dir: Path, log_text: str, engine_state: dict) -> Path:
    """Bundle everything useful for debugging into one zip file."""
    import platform, zipfile, io
    cfg = load_config()
    info = {"app_version": __version__, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "windows": platform.platform(), "python": sys.version.split()[0],
            "packaged": bool(getattr(sys, "frozen", False)), "engine": engine_state,
            "settings": get_settings(cfg)}
    loc = WindowLocator(cfg.get("window_title", ""), cfg.get("screen_origin", (0, 0)))
    win = {"title_setting": cfg.get("window_title", ""), "calibrated_size": cfg.get("window_size")}
    try:
        win["found"] = bool(loc.find()) if loc.win and loc.title else None
        win["origin"] = loc.origin()
        win["client_size"] = loc.client_size()
        win["focused"] = loc.is_focused()
        win["open_windows"] = loc.list_titles()[:40]
    except Exception:
        note_error("window info")
    info["window"] = win
    pictures = {}
    try:
        cap = ScreenGrabber()
        try:
            img, _ = cap.grab_window_or_screen(loc)
            pictures["window_now.png"] = img
        except SystemExit as e:
            info["window"]["capture"] = str(e.code)
        try:
            r = FlaskReader(cfg, loc, cap)
            readings = r.read()
            info["reading_now"] = ({k: {"visible": v.visible, "full": v.full, "running": v.active,
                                        "fill": round(v.fill, 2), "top_match": round(v.top_match, 2)}
                                    for k, v in readings.items()} if readings else {"problem": r.problem})
        except SystemExit as e:
            info["reading_now"] = {"problem": str(e.code)}
        cap.close()
    except Exception:
        note_error("report capture")
    info["calibration"] = {k: v for k, v in cfg.items() if k != "settings"}
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"DanielsAssignment-report-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("info.json", json.dumps(info, indent=2, default=str))
        z.writestr("activity_log.txt", log_text)
        z.writestr("errors.txt", "\n\n".join(RECENT_ERRORS) or "(no errors recorded)")
        for name, im in pictures.items():
            ok, png = cv2.imencode(".png", im)
            if ok:
                z.writestr(name, png.tobytes())
        for f in DATA_DIR.glob("*.png"):
            z.write(f, f"data/{f.name}")
    return path


# ================================================================ calibration
def _fit(img: np.ndarray, max_w=1600, max_h=900) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(1.0, max_w / w, max_h / h)
    return (cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img.copy()), s


def _banner(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (30, 30, 30), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _select_box(img: np.ndarray, prompt: str, existing: Optional[List[int]] = None) -> Optional[List[int]]:
    view, s = _fit(img)
    if existing:
        x, y, w, h = [int(v * s) for v in existing]
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 200, 255), 1)
        prompt += "  (Enter with no box = keep the orange one)"
    win = f"{APP} | Calibration"
    rect = cv2.selectROI(win, _banner(view, prompt), showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(win)
    x, y, w, h = rect
    if w < 4 or h < 4:
        return existing
    return [int(round(x / s)), int(round(y / s)), int(round(w / s)), int(round(h / s))]


def _click_liquid(img: np.ndarray, box: List[int], prompt: str) -> Tuple[int, int]:
    """Show the flask enlarged and wait for one click on its liquid."""
    crop = _crop(img, box)
    z = max(1, min(6, 500 // max(crop.shape[:2])))
    view = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
    view = cv2.copyMakeBorder(view, 34, 0, 0, max(0, 520 - view.shape[1]), cv2.BORDER_CONSTANT, value=(30, 30, 30))
    win = f"{APP} | Calibration"
    clicked = []

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and y >= 34 and x < crop.shape[1] * z:
            clicked.append((box[0] + x // z, box[1] + (y - 34) // z))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while not clicked:
        cv2.imshow(win, _banner(view, prompt))
        if cv2.waitKey(30) == 27:
            raise SystemExit("Calibration cancelled.")
    cv2.destroyWindow(win)
    return clicked[0]


def _crop(img: np.ndarray, box: List[int]) -> np.ndarray:
    x, y, w, h = box
    return img[y:y + h, x:x + w]


def hue_at(img: np.ndarray, point: Tuple[int, int]) -> int:
    """Liquid hue around a clicked point (5x5 circular median)."""
    x, y = point
    patch = cv2.cvtColor(img[max(0, y - 2):y + 3, max(0, x - 2):x + 3], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat = patch[patch[:, 1] >= 40]
    hues = (sat if len(sat) else patch)[:, 0].astype(float) * (2 * np.pi / 180)
    ang = np.arctan2(np.sin(hues).mean(), np.cos(hues).mean())
    return int(round((ang % (2 * np.pi)) * 180 / (2 * np.pi))) % 180


def calibrate_from_image(img: np.ndarray, cfg: dict, boxes: Dict[int, List[int]],
                         bar_boxes: Dict[int, List[int]],
                         liquid_points: Optional[Dict[int, object]] = None,
                         liquid_settings: Optional[Dict[int, dict]] = None,
                         window_size: Optional[Tuple[int, int]] = None) -> List[str]:
    """Save references for the given slots and return the report lines.
    (The interactive calibrate command collects the boxes, then calls this.)"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if window_size:
        cfg["window_size"] = list(window_size)
    for slot, box in boxes.items():
        sc = slot_cfg(cfg, slot)
        sc["box"] = box
        if slot in bar_boxes:
            sc["bar_box"] = bar_boxes[slot]
        sc.pop("liquid_hue", None)
        sc.pop("liquid_point", None)
        if liquid_points and slot in liquid_points:
            pts = liquid_points[slot]
            if pts and isinstance(pts[0], (int, np.integer)):     # a single (x, y)
                pts = [pts]
            sc["liquid_points"] = [[int(px) - box[0], int(py) - box[1]] for px, py in pts]
        else:
            sc.pop("liquid_points", None)
        ls = (liquid_settings or {}).get(slot, {})
        sc["similarity"] = int(ls.get("similarity", DEFAULT_SIMILARITY))
        sc["pick_mode"] = ls.get("mode", "shape") if ls.get("mode") in PICK_MODES else "shape"
        cv2.imwrite(str(ref_path(slot)), _crop(img, box))
    save_config(cfg)
    return report(img, cfg, boxes.keys())


LAST_REPORT: List[dict] = []


def report(img: np.ndarray, cfg: dict, slots=None) -> List[str]:
    """Text report; the per-flask results are also left in LAST_REPORT."""
    LAST_REPORT.clear()
    lines = [f"{APP} - calibration report", ""]
    bar_bgr = tuple(reversed(cfg.get("bar_rgb", DEFAULT_BAR_RGB)))
    preview = img.copy()
    all_ok = True
    for s in cfg["slots"]:
        slot = s["slot"]
        if s["box"] is None or not ref_path(slot).exists():
            lines.append(f"  Slot {slot} ({s['kind']}): NOT CALIBRATED")
            all_ok = False
            continue
        m = model_for(s)
        bar = _crop(img, s["bar_box"]) if s["bar_box"] else None
        rd = m.read(_crop(img, s["box"]), bar, bar_bgr)
        ok = not m.problems and rd.full and rd.visible
        all_ok &= ok
        tag = "OK  " if ok else "WARN"
        redone = " (just calibrated)" if slots is not None and slot in slots else ""
        lines.append(f"  [{tag}] Slot {slot} {s['kind']:<11} key '{s['key']}'{redone}")
        lines.append(f"         box {s['box']}  liquid: {hue_name(m.hue)}, "
                     f"{m.liquid_share:.0%} of box, top edge at row {m.top}")
        lines.append(f"         reads now: {'FULL' if rd.full else 'NOT FULL'} "
                     f"(top match {rd.top_match:.0%}), visible={rd.visible}"
                     + (f", timer bar {'SEEN' if rd.active else 'not seen'}" if s['bar_box'] else ""))
        issues = list(m.problems)
        if not rd.full:
            issues.append("reads NOT FULL on the calibration picture - was the flask full?")
        for p in issues:
            lines.append(f"         ! {p}")
        LAST_REPORT.append({"slot": slot, "kind": s["kind"], "ok": ok, "colour": hue_name(m.hue),
                            "full": rd.full, "visible": rd.visible, "top_match": rd.top_match,
                            "issues": issues, "has_bar": bool(s["bar_box"])})

        x, y, w, h = s["box"]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0) if ok else (0, 0, 255), 1)
        tint = preview[y:y + h, x:x + w]
        tint[m.band] = (tint[m.band] * 0.3 + np.array([255, 0, 255]) * 0.7).astype(np.uint8)
        cv2.putText(preview, str(slot), (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if s["bar_box"]:
            bx, by, bw, bh = s["bar_box"]
            cv2.rectangle(preview, (bx, by), (bx + bw, by + bh), (0, 200, 255), 1)

    # overlapping boxes would read the wrong flask
    boxes = [(s["slot"], s["box"]) for s in cfg["slots"] if s["box"]]
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            if not (ba[0] + ba[2] <= bb[0] or bb[0] + bb[2] <= ba[0] or
                    ba[1] + ba[3] <= bb[1] or bb[1] + bb[3] <= ba[1]):
                lines.append(f"  (boxes for slot {a} and slot {b} overlap - that's fine)")

    lines += ["", "  Magenta = the top-of-liquid area that decides FULL. "
                  "Orange = where it looks for the timer bar.",
              "  Next: run 'test' and use a flask - it should flip to NOT FULL "
              "(and RUNNING for quicksilvers).",
              "  Overall: " + ("everything looks good" if all_ok else "see the warnings above")]
    cv2.imwrite(str(DATA_DIR / "calibration_preview.png"), preview)
    lines.append(f"  Preview image saved to {DATA_DIR / 'calibration_preview.png'}")
    return lines


def cmd_calibrate(args) -> None:
    cfg = load_config()
    if args.window is not None:
        cfg["window_title"] = args.window
    if not cfg["window_title"]:
        print("Tip: pass --window \"<part of your program's window title>\" so the tool can\n"
              "follow the window when it moves and only press keys while it's focused.\n"
              "Without it, boxes are fixed screen positions.\n")
    locator = WindowLocator(cfg["window_title"], cfg.get("screen_origin", (0, 0)))

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"Can't read {args.image}")
    else:
        print("Switch to your program and make sure ALL flasks you're calibrating are FULL.")
        for i in range(args.delay, 0, -1):
            print(f"  capturing in {i}...", end="\r", flush=True)
            time.sleep(1)
        img, origin = ScreenGrabber().grab_window_or_screen(locator)
        if not (cfg["window_title"] and locator.win):
            cfg["screen_origin"] = list(origin)
        else:
            cfg["window_size"] = [img.shape[1], img.shape[0]]
        print(" " * 30)

    slots = [args.slot] if args.slot else [1, 2, 3, 4, 5]
    boxes, bar_boxes = {}, {}
    for slot in slots:
        sc = slot_cfg(cfg, slot)
        box = _select_box(img, f"Drag a tight box around flask {slot} ({sc['kind']}), then Enter",
                          sc["box"])
        if box is None:
            raise SystemExit(f"No box drawn for slot {slot}; calibration cancelled.")
        boxes[slot] = box

    points = {}
    for slot in slots:
        points[slot] = _click_liquid(img, boxes[slot],
                                     f"Click on the {slot_cfg(cfg, slot)['kind']} liquid in flask {slot} "
                                     "(not the label/wings/glass)")

    for slot in slots:
        if slot_kinds(cfg).get(slot) not in BAR_KINDS:
            continue
        sc = slot_cfg(cfg, slot)
        old = sc["bar_box"]
        # keep the strip's position relative to the flask if the flask moved/resized
        default = default_bar_box(boxes[slot])
        bar = _select_box(img, f"Drag a box where slot {slot}'s timer bar appears "
                               f"(under the flask), Enter = use the orange default",
                          old if old and old[2] > 0 and args.slot else default)
        bar_boxes[slot] = bar or default

    for line in calibrate_from_image(img, cfg, boxes, bar_boxes, points):
        print(line)


# ================================================================ test mode
def render_panel(cfg: dict, reader: FlaskReader, readings: Optional[Dict[int, SlotReading]]) -> np.ndarray:
    col_w, img_h = 150, 210
    panel = np.full((img_h + 120, col_w * 5, 3), 28, np.uint8)
    cv2.putText(panel, f"{APP} - TEST MODE (no keys are pressed)  |  q to quit",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    if readings is None:
        cv2.putText(panel, "Window not found", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return panel
    for i, s in enumerate(cfg["slots"]):
        slot = s["slot"]
        rd = readings[slot]
        crop, _ = reader.last_crops[slot]
        x0 = i * col_w
        ch, cw = crop.shape[:2]
        sc = min((col_w - 20) / cw, (img_h - 20) / ch)
        thumb = cv2.resize(crop, (max(1, int(cw * sc)), max(1, int(ch * sc))), interpolation=cv2.INTER_NEAREST)
        ty, tx = 30 + (img_h - 20 - thumb.shape[0]) // 2, x0 + (col_w - thumb.shape[1]) // 2
        panel[ty:ty + thumb.shape[0], tx:tx + thumb.shape[1]] = thumb

        if not rd.visible:
            colour, label = (0, 140, 255), "NOT VISIBLE"
        elif rd.full:
            colour, label = (60, 200, 60), "FULL"
        else:
            colour, label = (60, 60, 220), "NOT FULL"
        cv2.rectangle(panel, (x0 + 4, 28), (x0 + col_w - 4, 28 + img_h - 12), colour, 4)
        y = img_h + 40
        cv2.rectangle(panel, (x0 + 6, y - 20), (x0 + col_w - 6, y + 8), colour, -1)
        cv2.putText(panel, label, (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"{slot} {s['kind']}  ~{rd.fill:.0%}", (x0 + 8, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        if s["bar_box"]:
            bcol = (0, 215, 255) if rd.active else (90, 90, 90)
            cv2.rectangle(panel, (x0 + 6, y + 42), (x0 + col_w - 6, y + 70), bcol, -1 if rd.active else 2)
            cv2.putText(panel, "RUNNING" if rd.active else "not running", (x0 + 14, y + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20) if rd.active else (160, 160, 160),
                        2 if rd.active else 1, cv2.LINE_AA)
    return panel


def cmd_test(args) -> None:
    cfg = load_config()
    if args.image:
        img = cv2.imread(args.image)
        reader = FlaskReader(cfg, frame_source=lambda: img)
    else:
        reader = FlaskReader(cfg, WindowLocator(cfg["window_title"], cfg.get("screen_origin", (0, 0))), ScreenGrabber())
    win = f"{APP} | Test mode"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    last_print = ""
    while True:
        readings = reader.read()
        cv2.imshow(win, render_panel(cfg, reader, readings))
        if readings:
            line = "  ".join(
                f"{k}:{'--' if not r.visible else ('FULL' if r.full else 'not')}{'+RUN' if r.active else ''}"
                for k, r in readings.items())
            if line != last_print:
                print(time.strftime("%H:%M:%S"), line)
                last_print = line
        key = cv2.waitKey(int(POLL_INTERVAL * 1000)) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if args.image and args.once:
            cv2.imwrite(str(DATA_DIR / "test_panel.png"), render_panel(cfg, reader, readings))
            break
    cv2.destroyAllWindows()


# ================================================================ run mode
def cmd_run(args) -> None:
    cfg = load_config()
    if not cfg["window_title"]:
        raise SystemExit("Set your program's window title first:  calibrate --window \"<title>\"")
    locator = WindowLocator(cfg["window_title"], cfg.get("screen_origin", (0, 0)))
    reader = FlaskReader(cfg, locator, ScreenGrabber())
    ctl = FlaskController(reader.read, KeySender(cfg, locator), settings=get_settings(cfg), **controller_options(cfg))

    from pynput import keyboard

    def on_press(key):
        if key == keyboard.Key.f8:
            ctl.paused = not ctl.paused
            print("PAUSED" if ctl.paused else "RESUMED")
        elif key == keyboard.Key.f9:
            ctl.leave_town("F9 pressed")
        elif key == keyboard.Key.f10:
            ctl.running = False
            return False

    keyboard.Listener(on_press=on_press).start()
    print(f"{APP} running. F8 pause/resume, F9 leave town mode, F10 quit.")
    ctl.run()
    print("Stopped.")


def main() -> None:
    p = argparse.ArgumentParser(prog="daniels_assignment", description=APP)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="select the flasks and save their full-state images")
    c.add_argument("--slot", type=int, choices=range(1, 6), help="recalibrate just this flask")
    c.add_argument("--window", help="part of your program's window title")
    c.add_argument("--image", help="calibrate from a saved screenshot instead of the screen")
    c.add_argument("--delay", type=int, default=3, help="seconds before capturing (default 3)")
    c.set_defaults(func=cmd_calibrate)

    t = sub.add_parser("test", help="live FULL / NOT FULL indicators, presses nothing")
    t.add_argument("--image", help="read a saved screenshot instead of the screen")
    t.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    t.set_defaults(func=cmd_test)

    r = sub.add_parser("run", help="run the flask controller")
    r.set_defaults(func=cmd_run)

    if len(sys.argv) == 1:                  # double-clicked: open the app
        from gui import launch
        launch()
        return
    args = p.parse_args()
    try:
        args.func(args)
    except SystemExit as e:
        if e.code not in (None, 0):
            print(e.code if isinstance(e.code, str) else f"Exited with code {e.code}")


if __name__ == "__main__":
    main()
