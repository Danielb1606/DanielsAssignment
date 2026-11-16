# Daniels Assignment - notes for Claude

Read this before changing anything, and update it in the same commit as each change.

## What it is
Dan's class project: a Windows tool that watches a PoE-style UI (his own program's UI, NOT the actual
PoE client) by screen/colour/image detection and presses flask keys. Python 3.12, Tkinter (dark/gold theme),
OpenCV/NumPy, mss, pynput, ctypes SendInput. Packaged as a one-file PyInstaller exe
(`DanielsAssignment.exe`), released by GitHub Actions when `__version__` in daniels_assignment.py changes.
It lives in Dan's Workshop at `apps\Daniels Assignment\` (data in `daniels_assignment_data\` next to the exe).

## Behaviour (Dan's requirements)
- Each slot has a role (click the flask's name on its card): Life / Mana (use when FULL, Instilling Orb style),
  Quicksilver (quicksilvers take turns, one running at a time), Utility (use when FULL and its timer strip
  shows under `refresh_below`% left, default 10, judged against the longest the strip has been), Not used.
  Stored as `kind` / `refresh_below` per slot in config.json. Dan's current belt: 1 life, 2 mana,
  3 utility, 4-5 quicksilver. Town mode only counts quicksilver failures.
- Per flask (name menu → Settings for flask N…): own press delay range (`delay_min_ms`/`delay_max_ms`,
  else the main 150-300 ms), and an optional globe link (`globe`: box + colour like a trigger, `below` %
  chosen by Dan, `again_ms` repeat while still low, ref image `globeN_reference.png`). A linked flask is
  pressed when its normal rule says so OR the globe is below `below`% (Dan's choice: full OR globe low).
- Every check takes ONE screenshot of the rectangle covering all areas (flasks, strips, triggers, globes),
  4 checks/s by default; presses fire at their exact scheduled time between checks.
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
