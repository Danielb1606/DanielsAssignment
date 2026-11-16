# Daniels Assignment - notes for Claude

Read this before changing anything, and update it in the same commit as each change.

## What it is
Dan's class project: a Windows tool that watches a PoE-style UI (his own program's UI, NOT the actual
PoE client) by screen/colour/image detection and presses flask keys. Python 3.12, Tkinter (dark/gold theme),
OpenCV/NumPy, mss, pynput, ctypes SendInput. Packaged as a one-file PyInstaller exe
(`DanielsAssignment.exe`), released by GitHub Actions when `__version__` in daniels_assignment.py changes.
It lives in Dan's Workshop at `apps\Daniels Assignment\` (data in `daniels_assignment_data\` next to the exe).

## Behaviour (Dan's requirements)
- Quicksilvers in slots 3-5 rotate one at a time; life/mana (1-2) used when FULL (Instilling Orb style).
- 4 checks/s; random 150-300 ms delay before a press (professor's requirement); key hold random 20-90 ms;
  keys sent as hardware scan codes (SendInput), pynput fallback.
- Calibration GUI: draw boxes, click the liquid colour, Similarity slider (default 55), shape/box modes,
  auto-find flasks 4/5 and reuse of the previous calibration; review screen with Fix buttons.
- Town mode: only quicksilver failures count (>=2 distinct quicksilvers); waits quietly for a loading
  screen; manual flask use or F9 also ends it. Pause detection by dimmed screen.
- Triggers: press a key when a colour disappears (default Q, fire below 20%, re-arm above 50%,
  repeat every random 200-600 ms), all editable.
- Hotkeys F7/F8/F9 by default, changeable, with modifiers. Start button green, Stop red.
- Self-update: rename-swap of the running exe with handshake + rollback (updater.py).
- Report a problem button zips logs/config for Claude.

## Working with Dan
- Before bigger changes, show suggestions/examples to pick between. Small fixes: just do them, mention after.
- Speed: push and reply without waiting for the build; screenshots only for new screens.
- Never delete whole repos.

## Checks
`python test_offline.py flasks.png` and `python test_updater.py` (both run in CI before the build).
