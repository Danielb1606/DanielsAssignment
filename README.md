# Daniels Assignment

## Make the app (once)
Double-click `build_exe.bat` (needs Python installed, only for this step).
It creates `dist\DanielsAssignment.exe`. Copy it anywhere and double-click it.
Calibration is saved in a `daniels_assignment_data` folder next to the .exe.

## Using it
A setup guide at the top walks you through the first run:
1. **Program window** - pick your program from the list (⟳ refreshes it).
2. **Calibrate all flasks** - with every flask full: drag a box around flasks 1-3, click
   each one's liquid and slide **Similarity** until the pink covers the liquid. The app then
   finds quicksilvers 4 and 5 for you (same settings as flask 3). Accept or redraw the
   timer-bar boxes.
3. **Test mode** - each card shows FULL / NOT FULL / RUNNING live. Nothing is pressed.
4. **Start** - F8 pause/resume, F9 leave town mode, F10 stop.

Also:
- **Recalibrate** under any flask redoes just that one (e.g. new or bigger art).
- **Settings** - flask keys, checks per second, press delay, key hold, cooldown,
  town-mode sensitivity, sound alerts.
- **Compact** - a small always-on-top bar with five status dots, start/stop and ⤢ to
  return to the full window. Drag it by the ⠿ handle; it remembers where you put it.
- If your program's window changes size after calibration, an orange warning appears
  and nothing is pressed until you resize it back or recalibrate.
- Town mode plays a sound and flashes the status when it starts or stops.

## Auto-update
Releases come from https://github.com/Danielb1606/DanielsAssignment/releases.
To publish a new version: raise `__version__` in `daniels_assignment.py` and push to main.
GitHub builds the .exe (`.github/workflows/release.yml`) and the app offers the update
the next time it opens.

## Command line (optional)
    python daniels_assignment.py calibrate | test | run
