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
__version__ = "1.3.0"
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

# Everything above can be changed in the app's Settings panel; these are the defaults.
DEFAULT_SETTINGS = {
    "keys": {"1": "1", "2": "2", "3": "3", "4": "4", "5": "5"},
    "checks_per_second": 4,
    "delay_min_ms": 150,
    "delay_max_ms": 300,
    "key_hold_ms": 30,
    "refire_cooldown_ms": 500,
    "verify_ms": 1200,
    "fails_for_town": 2,
    "sound_alerts": True,
}


def get_settings(cfg: dict) -> dict:
    out = json.loads(json.dumps(DEFAULT_SETTINGS))
    saved = cfg.get("settings", {})
    for k, v in saved.items():
        if k == "keys" and isinstance(v, dict):
            out["keys"].update({str(a): str(b) for a, b in v.items()})
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
    if not 5 <= st["key_hold_ms"] <= 500:
        errs.append("Key hold must be 5-500 ms.")
    if not 0 <= st["refire_cooldown_ms"] <= 10000:
        errs.append("Cooldown must be 0-10000 ms.")
    if not 300 <= st["verify_ms"] <= 10000:
        errs.append("Effect check window must be 300-10000 ms.")
    if not 1 <= st["fails_for_town"] <= 10:
        errs.append("Town mode trigger must be 1-10 presses.")
    return errs

# ---------------------------------------------------------------- detection
SLOT_KINDS = {1: "life", 2: "mana", 3: "quicksilver", 4: "quicksilver", 5: "quicksilver"}
QUICKSILVER_SLOTS = (3, 4, 5)
INSTILLING_SLOTS = (1, 2)
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
    }


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

    def grab(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        shot = self._sct().grab({"left": left, "top": top, "width": width, "height": height})
        return np.ascontiguousarray(np.array(shot)[:, :, :3])

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


DEFAULT_SIMILARITY = 55   # calibration slider default (1-100)
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
        out |= ((_circ_hue_dist(H, h0) <= hue_tol) & (S >= s0 * (1 - 0.85 * f))
                & (V >= v0 * (1 - 0.9 * f)))
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
    match = cv2.morphologyEx(match.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(match.astype(np.uint8), connectivity=8)
    clicked = {int(lab[y, x]) for x, y in pts if lab[y, x] > 0}
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x0, y0, bw, bh, area = stats[i]
        touches_edge = x0 == 0 or y0 == 0 or x0 + bw >= w or y0 + bh >= h
        if i in clicked or (area >= 0.004 * h * w and (mode == "box" or not touches_edge)):
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

        rows = self.mask.sum(1)
        if rows.max() == 0:
            self.top = self.bottom = 0
        else:
            good = np.nonzero(rows >= max(2, 0.15 * rows.max()))[0]
            self.top, self.bottom = int(good.min()), int(good.max())
        band_h = max(3, int(round(TOP_BAND_FRAC * (self.bottom - self.top + 1))))
        self.band = self.mask.copy()
        self.band[: self.top] = False
        self.band[self.top + band_h:] = False

        grown = cv2.dilate(self.mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.frame = ~grown
        self.ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)

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
        if bar_crop is not None and bar_crop.size:
            dist = np.abs(bar_crop.astype(int) - np.array(bar_bgr)).max(axis=2)
            active = int((dist <= BAR_RGB_TOL).sum()) >= BAR_MIN_PIXELS

        return SlotReading(visible, full and visible, active and visible, fill, top_match)


def slot_points(s: dict) -> List[Tuple[int, int]]:
    """Clicked liquid spots for a slot (relative to its box). Older configs
    stored a single spot as liquid_point."""
    pts = s.get("liquid_points")
    if pts:
        return [tuple(p) for p in pts]
    if s.get("liquid_point"):
        return [tuple(s["liquid_point"])]
    return []


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
        if slot in QUICKSILVER_SLOTS and not d.get("bar_box"):
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
        boxes = [s["box"] for s in cfg["slots"]] + [s["bar_box"] for s in cfg["slots"] if s["bar_box"]]
        self.bx0 = min(b[0] for b in boxes)
        self.by0 = min(b[1] for b in boxes)
        self.bx1 = max(b[0] + b[2] for b in boxes)
        self.by1 = max(b[1] + b[3] for b in boxes)
        self.last_crops: Dict[int, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        self.expected_size = tuple(cfg["window_size"]) if cfg.get("window_size") else None
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
        # If the window was resized since calibration the boxes no longer line
        # up: report nothing (so nothing gets pressed) and flag it for the app.
        if self.frame_source is None and self.expected_size and self.locator:
            size = self.locator.client_size()
            if size and tuple(size) != self.expected_size:
                self.size_mismatch = (tuple(size), self.expected_size)
                return None
        self.size_mismatch = None
        region = self._region()
        if region is None:
            return None
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
                 settings: Optional[dict] = None):
        self.read, self.press = read, press
        st = settings or DEFAULT_SETTINGS
        self.poll_interval = 1.0 / float(st["checks_per_second"])
        self.delay_range = (st["delay_min_ms"] / 1000.0, st["delay_max_ms"] / 1000.0)
        self.cooldown = st["refire_cooldown_ms"] / 1000.0
        self.verify_window = st["verify_ms"] / 1000.0
        self.fails_for_town = int(st["fails_for_town"])
        self.on_town: Callable[[bool], None] = lambda on: None
        self.clock, self.sleep = clock, sleep
        self.rng = rng or random.Random()
        self.log = log

        self.queue: List[_Pending] = []
        self.queued: set = set()
        self.last_press: Dict[int, float] = {}
        self.verify: Dict[int, float] = {}       # slot -> deadline
        self.fails = 0
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
            self.queue.clear(); self.queued.clear(); self.verify.clear()
            self.log(f"TOWN MODE ON  - {why}. Not pressing flasks.")
            self.on_town(True)

    def leave_town(self, why: str) -> None:
        if self.town_mode:
            self.town_mode = False
            self.fails = 0
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

        # did recent presses do anything?
        for slot, deadline in list(self.verify.items()):
            if not r[slot].full or r[slot].active:
                self.verify.pop(slot, None)
                self.fails = 0
            elif now >= deadline:
                self.verify.pop(slot, None)
                self.fails += 1
                self.log(f"[{now:8.3f}] slot {slot}: press had no effect ({self.fails}/{self.fails_for_town})")
                if self.fails >= self.fails_for_town:
                    self.enter_town("flasks are not being used")
                    break

        # a flask emptied without us pressing it = flasks work again
        for slot, rd in r.items():
            was_full = self.prev_full.get(slot)
            recently = now - self.last_press.get(slot, -1e9) < self.verify_window + 0.5
            if was_full and not rd.full and rd.visible and not recently and self.town_mode:
                self.leave_town(f"slot {slot} was used manually")
            self.prev_full[slot] = rd.full

        if self.paused or self.town_mode:
            return

        if not self._quicksilver_running(r, now):
            for slot in QUICKSILVER_SLOTS:
                if r[slot].visible and r[slot].full and self._can_schedule(slot, now):
                    self._schedule(slot, now)
                    break
        for slot in INSTILLING_SLOTS:
            if r[slot].visible and r[slot].full and self._can_schedule(slot, now):
                self._schedule(slot, now)

    def _quicksilver_running(self, r: Dict[int, SlotReading], now: float) -> bool:
        for slot in QUICKSILVER_SLOTS:
            if r[slot].active or slot in self.queued or slot in self.verify:
                return True
            if now - self.last_press.get(slot, -1e9) < ACTIVATION_GRACE:
                return True
        return False

    def _can_schedule(self, slot: int, now: float) -> bool:
        if slot in self.queued or slot in self.verify:
            return False
        return now - self.last_press.get(slot, -1e9) >= self.cooldown

    def _schedule(self, slot: int, now: float) -> None:
        delay = self.rng.uniform(*self.delay_range)
        heapq.heappush(self.queue, _Pending(now + delay, slot))
        self.queued.add(slot)
        self.log(f"[{now:8.3f}] slot {slot} full -> press in {delay * 1000:.0f} ms")

    def _fire_due(self) -> None:
        now = self.clock()
        while self.queue and self.queue[0].fire_at <= now:
            p = heapq.heappop(self.queue)
            self.queued.discard(p.slot)
            if self.paused or self.town_mode:
                continue
            if self.press(p.slot):
                self.last_press[p.slot] = now
                self.verify[p.slot] = now + self.verify_window
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
                self.poll()
                next_poll += self.poll_interval
                if next_poll < now:                 # fell behind; don't burst
                    next_poll = now + self.poll_interval
            self._fire_due()
            wake = next_poll if not self.queue else min(next_poll, self.queue[0].fire_at)
            self.sleep(max(0.0, wake - self.clock()))


# ================================================================ keyboard
class KeySender:
    def __init__(self, cfg: dict, locator: WindowLocator):
        from pynput.keyboard import Controller
        self.kb = Controller()
        st = get_settings(cfg)
        self.keys = {s["slot"]: st["keys"][str(s["slot"])] for s in cfg["slots"]}
        self.hold = st["key_hold_ms"] / 1000.0
        self.locator = locator

    def __call__(self, slot: int) -> bool:
        if not self.locator.is_focused():
            return False
        k = self.keys[slot]
        self.kb.press(k)
        time.sleep(self.hold)
        self.kb.release(k)
        return True


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
        if slot not in QUICKSILVER_SLOTS:
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
    ctl = FlaskController(reader.read, KeySender(cfg, locator), settings=get_settings(cfg))

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
