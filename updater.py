"""Self-update from GitHub Releases.

Each release has DanielsAssignment.exe attached (built by the GitHub Action in
.github/workflows/release.yml). On start-up the app asks GitHub for the latest
release; if it's newer, the app downloads it next to itself, hands over to a
tiny batch script that swaps the files once the app has closed, and relaunches.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Tuple

# owner/repo on GitHub. Empty = updates switched off.
UPDATE_REPO = "Danielb1606/DanielsAssignment"
ASSET_NAME = "DanielsAssignment.exe"
TIMEOUT = 10


def _ver(v: str) -> Tuple[int, ...]:
    parts = []
    for p in v.strip().lstrip("vV").split("."):
        num = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts)


def is_packaged() -> bool:
    return bool(getattr(sys, "frozen", False)) and sys.platform == "win32"


def check(current_version: str) -> Optional[dict]:
    """Return {'version', 'url', 'notes'} if a newer release exists, else None."""
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
            "notes": (data.get("body") or "").strip()}


def download_and_restart(info: dict, progress: Callable[[float], None] = lambda f: None) -> None:
    """Download the new exe, then swap it in after this process exits."""
    exe = Path(sys.executable)
    new = exe.with_suffix(".new.exe")
    req = urllib.request.Request(info["url"], headers={"User-Agent": "DanielsAssignment"})
    with urllib.request.urlopen(req, timeout=60) as r, open(new, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 16):
            f.write(chunk)
            done += len(chunk)
            if total:
                progress(done / total)
    if new.stat().st_size < 1_000_000:
        new.unlink(missing_ok=True)
        raise RuntimeError("Downloaded file looks wrong; update cancelled.")

    bat = Path(tempfile.gettempdir()) / "daniels_assignment_update.bat"
    bat.write_text(
        "@echo off\r\n"
        ":wait\r\n"
        f'tasklist /FI "PID eq {os.getpid()}" | find "{os.getpid()}" >nul && '
        "(timeout /t 1 /nobreak >nul & goto wait)\r\n"
        f'move /y "{new}" "{exe}" >nul\r\n'
        f'start "" "{exe}"\r\n'
        'del "%~f0"\r\n')
    subprocess.Popen(["cmd", "/c", str(bat)], creationflags=0x08000000)  # CREATE_NO_WINDOW
