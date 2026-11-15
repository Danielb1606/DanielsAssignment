"""Self-update from GitHub Releases.

How an update works (Windows, packaged .exe):
  1. Download the new DanielsAssignment.exe next to the running one (.new.exe)
     and check its size.
  2. Rename the running exe to .old.exe (Windows allows renaming a running
     program, just not overwriting it), then move .new.exe into its place.
  3. Start the new exe with a clean environment. (A one-file PyInstaller exe
     sets private variables pointing at its own temporary folder; if the new
     copy inherits them it looks for Python in a folder that's been deleted -
     the "Failed to load Python DLL" error.)
  4. Wait for the new copy to report that it started. If it does, this copy
     exits. If it doesn't within START_TIMEOUT, put the old exe back and carry on.
  5. Next start-up, the new version shows "Updated to vX" and tidies up .old.exe.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Tuple

# owner/repo on GitHub. Empty = updates switched off.
UPDATE_REPO = "Danielb1606/DanielsAssignment"
ASSET_NAME = "DanielsAssignment.exe"
TIMEOUT = 10
START_TIMEOUT = 45          # seconds the new version has to report it started
MARKER = "update_pending.json"


def _ver(v: str) -> Tuple[int, ...]:
    parts = []
    for p in v.strip().lstrip("vV").split("."):
        num = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts)


def is_packaged() -> bool:
    return bool(getattr(sys, "frozen", False)) and sys.platform == "win32"


def check(current_version: str) -> Optional[dict]:
    """Return {'version', 'url', 'size', 'notes'} if a newer release exists, else None."""
    if not UPDATE_REPO:
        return None
    req = urllib.request.Request(
        f"https://api.github.com/repos/{UPDATE_REPO}/releases/latest",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "DanielsAssignment"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = json.load(r)
    latest = data.get("tag_name", "")
    if _ver(latest) <= _ver(current_version):
        return None
    asset = next((a for a in data.get("assets", []) if a.get("name") == ASSET_NAME), None)
    if not asset:
        return None
    return {"version": latest.lstrip("vV"), "url": asset["browser_download_url"],
            "size": int(asset.get("size") or 0), "notes": (data.get("body") or "").strip()}


def clean_env() -> dict:
    """Environment for starting a fresh copy of a one-file PyInstaller exe."""
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("_MEI") or k.startswith("_PYI") or k == "PYINSTALLER_RESET_ENVIRONMENT")}
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    return env


def _launch(exe: Path) -> subprocess.Popen:
    flags = 0
    if sys.platform == "win32":
        flags = 0x00000008 | 0x00000200          # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen([str(exe)], env=clean_env(), cwd=str(exe.parent),
                            creationflags=flags, close_fds=True)


def download(info: dict, dest: Path, progress: Callable[[float], None] = lambda f: None) -> None:
    req = urllib.request.Request(info["url"], headers={"User-Agent": "DanielsAssignment"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or info.get("size") or 0)
        done = 0
        while chunk := r.read(1 << 16):
            f.write(chunk)
            done += len(chunk)
            if total:
                progress(done / total)
    size = dest.stat().st_size
    expected = info.get("size") or 0
    if size < 1_000_000 or (expected and size != expected):
        dest.unlink(missing_ok=True)
        raise RuntimeError("the download was incomplete")


def install_and_restart(info: dict, data_dir: Path, current_version: str,
                        progress: Callable[[float], None] = lambda f: None,
                        exe: Optional[Path] = None, launch=_launch, fetch=download) -> bool:
    """Swap in the new exe and start it. Returns True when the new version has
    started (the caller should then exit). Returns False after rolling back."""
    exe = Path(exe or sys.executable)
    new, old = exe.with_suffix(".new.exe"), exe.with_suffix(".old.exe")
    marker = Path(data_dir) / MARKER
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    new.unlink(missing_ok=True)
    fetch(info, new, progress)

    old.unlink(missing_ok=True)
    os.replace(exe, old)                 # rename the running program out of the way
    try:
        os.replace(new, exe)
    except OSError:
        os.replace(old, exe)
        raise

    marker.write_text(json.dumps({"from": current_version, "to": info["version"],
                                  "notes": info.get("notes", ""), "started": False,
                                  "time": time.time()}))
    proc = launch(exe)
    deadline = time.time() + START_TIMEOUT
    while time.time() < deadline:
        try:
            if json.loads(marker.read_text()).get("started"):
                return True
        except (OSError, ValueError):
            pass
        if proc.poll() is not None:      # the new version closed or crashed
            break
        time.sleep(0.25)

    # roll back
    try:
        proc.kill()
    except Exception:
        pass
    time.sleep(0.5)
    bad = exe.with_suffix(".failed.exe")
    bad.unlink(missing_ok=True)
    try:
        os.replace(exe, bad)
    except OSError:
        pass
    os.replace(old, exe)
    marker.unlink(missing_ok=True)
    return False


def on_startup(data_dir: Path, current_version: str, exe: Optional[Path] = None) -> Optional[dict]:
    """Call early when the app starts. Confirms a pending update to the copy
    that launched us, tidies leftovers, and returns {'from','to','notes'} the
    first time a new version runs (so the app can say "Updated to vX")."""
    exe = Path(exe or sys.executable)
    for suffix in (".old.exe", ".new.exe", ".failed.exe"):
        try:
            exe.with_suffix(suffix).unlink(missing_ok=True)
        except OSError:
            pass                             # still in use - next time
    marker = Path(data_dir) / MARKER
    try:
        info = json.loads(marker.read_text())
    except (OSError, ValueError):
        return None
    if _ver(info.get("to", "0")) != _ver(current_version):
        return None
    if not info.get("started"):
        info["started"] = True
        marker.write_text(json.dumps(info))  # tells the old copy we're up
        return info
    # second start-up after the update: nothing more to report
    try:
        marker.unlink()
    except OSError:
        pass
    return None
