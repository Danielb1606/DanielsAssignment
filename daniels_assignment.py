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
__version__ = "1.0.0"
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

# ---------------------------------------------------------------- detection
SLOT_KINDS = {1: "life", 2: "mana", 3: "quicksilver", 4: "quicksilver", 5: "quicksilver"}
QUICKSILVER_SLOTS = (3, 4, 5)
INSTILLING_SLOTS = (1, 2)
DEFAULT_BAR_RGB = (249, 215, 153)    # measured from your screenshot

HUE_TOL = 12          # OpenCV hue units (0-180)
LIQ_SAT_MIN = 80      # saturation needed to count as liquid in the reference
LIQ_VAL_MIN = 45
LIVE_SAT_MIN = 55     # slightly looser when reading live
LIVE_VAL_MIN = 35
TOP_BAND_FRAC = 0.15  # top 15% of the liquid decides "full"
FULL_MATCH_MIN = 0.60
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


class SlotModel:
    """Built from a screenshot of the flask when FULL. Works for any flask
    art: it learns the liquid colour and where the top of the liquid is."""

    def __init__(self, ref_bgr: np.ndarray, liquid_hue: Optional[int] = None,
                 liquid_point: Optional[Tuple[int, int]] = None):
        self.ref = ref_bgr
        self.h, self.w = ref_bgr.shape[:2]
        hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        coloured = (S >= LIQ_SAT_MIN) & (V >= LIQ_VAL_MIN)
        self.problems: List[str] = []

        if coloured.sum() < 30:
            self.hue = 0
            self.mask = np.zeros_like(coloured)
            self.problems.append("no coloured liquid found in the box")
        else:
            if liquid_hue is not None:
                self.hue = int(liquid_hue)       # from the click on the liquid
            else:
                hist = np.bincount(H[coloured].ravel(), minlength=180).astype(float)
                hist = np.convolve(np.r_[hist[-4:], hist, hist[:4]], np.ones(9), "valid")
                self.hue = int(np.argmax(hist))
            mask = coloured & (_circ_hue_dist(H, self.hue) <= HUE_TOL)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
            n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
            keep = np.zeros_like(mask)
            main = 0
            if liquid_point is not None:
                px, py = liquid_point
                win = lab[max(0, py - 3):py + 4, max(0, px - 3):px + 4]
                ids = win[win > 0]
                if ids.size:
                    main = int(np.bincount(ids).argmax())
            if main:
                # the clicked blob, plus big pieces inside its outline (e.g. the
                # green above and below the quicksilver wings). Anything outside,
                # like a same-coloured arch or frame, is ignored.
                x, y, w, h = stats[main, :4]
                keep |= lab == main
                for i in range(1, n):
                    cx, cy = cents[i]
                    if (i != main and stats[i, cv2.CC_STAT_AREA] >= 0.1 * stats[main, cv2.CC_STAT_AREA]
                            and x - 3 <= cx <= x + w + 3 and y - h * 0.5 <= cy <= y + h * 1.5):
                        keep |= lab == i
            else:
                total = mask.sum()
                for i in range(1, n):
                    if stats[i, cv2.CC_STAT_AREA] >= 0.05 * total:
                        keep |= lab == i
                if liquid_point is not None:
                    self.problems.append("clicked point isn't on the liquid colour; click the liquid itself")
            self.mask = keep

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
        if self.mask.any() and self.liquid_share < 0.08:
            self.problems.append(f"only {self.liquid_share:.0%} of the box is liquid; "
                                 "draw the box tighter or check the flask is full")
        if self.band.sum() < 15:
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
    def liquid_like(self, crop_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        return ((hsv[..., 1] >= LIVE_SAT_MIN) & (hsv[..., 2] >= LIVE_VAL_MIN)
                & (_circ_hue_dist(hsv[..., 0], self.hue) <= HUE_TOL))

    def read(self, crop_bgr: np.ndarray, bar_crop: Optional[np.ndarray], bar_bgr) -> SlotReading:
        if crop_bgr.shape[:2] != (self.h, self.w):
            crop_bgr = cv2.resize(crop_bgr, (self.w, self.h))
        ll = self.liquid_like(crop_bgr)

        top_match = float(ll[self.band].mean()) if self.band.any() else 0.0
        full = top_match >= FULL_MATCH_MIN

        rows = slice(self.top, self.bottom + 1)
        per_row = self.mask[rows].sum(1)
        hit = (ll & self.mask)[rows].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(per_row > 0, hit / np.maximum(per_row, 1), np.nan)
        valid = ~np.isnan(frac)
        fill = float((frac[valid] >= 0.4).mean()) if valid.any() else 0.0

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        visible = self.shape_match(gray) >= VISIBLE_CORR_MIN

        active = False
        if bar_crop is not None and bar_crop.size:
            dist = np.abs(bar_crop.astype(int) - np.array(bar_bgr)).max(axis=2)
            active = int((dist <= BAR_RGB_TOL).sum()) >= BAR_MIN_PIXELS

        return SlotReading(visible, full and visible, active and visible, fill, top_match)


def model_for(s: dict) -> SlotModel:
    pt = s.get("liquid_point")
    return SlotModel(cv2.imread(str(ref_path(s["slot"]))), s.get("liquid_hue"), tuple(pt) if pt else None)


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
                 log: Callable[[str], None] = print):
        self.read, self.press = read, press
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

    def leave_town(self, why: str) -> None:
        if self.town_mode:
            self.town_mode = False
            self.fails = 0
            self.log(f"TOWN MODE OFF - {why}.")

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
                self.log(f"[{now:8.3f}] slot {slot}: press had no effect ({self.fails}/{FAILS_FOR_TOWN})")
                if self.fails >= FAILS_FOR_TOWN:
                    self.enter_town("flasks are not being used")
                    break

        # a flask emptied without us pressing it = flasks work again
        for slot, rd in r.items():
            was_full = self.prev_full.get(slot)
            recently = now - self.last_press.get(slot, -1e9) < VERIFY_WINDOW + 0.5
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
        return now - self.last_press.get(slot, -1e9) >= REFIRE_COOLDOWN

    def _schedule(self, slot: int, now: float) -> None:
        delay = self.rng.uniform(*INPUT_DELAY_RANGE)
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
                self.verify[p.slot] = now + VERIFY_WINDOW
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
                next_poll += POLL_INTERVAL
                if next_poll < now:                 # fell behind; don't burst
                    next_poll = now + POLL_INTERVAL
            self._fire_due()
            wake = next_poll if not self.queue else min(next_poll, self.queue[0].fire_at)
            self.sleep(max(0.0, wake - self.clock()))


# ================================================================ keyboard
class KeySender:
    def __init__(self, cfg: dict, locator: WindowLocator):
        from pynput.keyboard import Controller
        self.kb = Controller()
        self.keys = {s["slot"]: s["key"] for s in cfg["slots"]}
        self.locator = locator

    def __call__(self, slot: int) -> bool:
        if not self.locator.is_focused():
            return False
        k = self.keys[slot]
        self.kb.press(k)
        time.sleep(KEY_HOLD)
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
                         liquid_points: Optional[Dict[int, Tuple[int, int]]] = None) -> List[str]:
    """Save references for the given slots and return the report lines.
    (The interactive calibrate command collects the boxes, then calls this.)"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for slot, box in boxes.items():
        sc = slot_cfg(cfg, slot)
        sc["box"] = box
        if slot in bar_boxes:
            sc["bar_box"] = bar_boxes[slot]
        if liquid_points and slot in liquid_points:
            px, py = liquid_points[slot]
            sc["liquid_hue"] = hue_at(img, (px, py))
            sc["liquid_point"] = [px - box[0], py - box[1]]
        else:
            sc.pop("liquid_hue", None)
            sc.pop("liquid_point", None)
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
                lines.append(f"  ! boxes for slot {a} and slot {b} overlap")
                all_ok = False
                for r in LAST_REPORT:
                    if r["slot"] in (a, b):
                        r["ok"] = False
                        r["issues"].append(f"box overlaps flask {b if r['slot'] == a else a}'s box")

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
    ctl = FlaskController(reader.read, KeySender(cfg, locator))

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
