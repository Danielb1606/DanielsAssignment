# Daniels Assignment

## Make the app (once)
Double-click `build_exe.bat` (needs Python installed, only for this step).
It creates `dist\DanielsAssignment.exe`. Copy it anywhere and double-click it.
Calibration is saved in a `daniels_assignment_data` folder next to the .exe.

## Using it
A setup guide at the top walks you through the first run:
1. **Program window** - pick your program from the list (⟳ refreshes it).
2. **Calibrate all flasks** - with every flask full. The app finds all five flasks by itself
   (or reuses your last calibration if they look the same) and shows a review screen:
   pink = the liquid it will watch, dashed yellow = timer-bar area. Click **Finish** if it's
   right, or click a flask to fix just that one (box, liquid click and Similarity slider).
3. **Test mode** - each card shows FULL / NOT FULL / RUNNING live. Nothing is pressed.
4. **Start** - default hotkeys: F7 start/stop, F8 pause/resume, F9 leave town mode (change them in Settings).
   Something wrong? Click **Report a problem** and send the file it saves.

Also:
- **Recalibrate** under any flask redoes just that one (e.g. new or bigger art).
- **Settings** - flask keys, checks per second, press delay, key hold, cooldown,
  town-mode sensitivity, sound alerts.
- **Compact** - a small always-on-top bar with five status dots, start/stop and ⤢ to
  return to the full window. Drag it by the ⠿ handle; it remembers where you put it.
- If your program's window changes size after calibration, an orange warning appears
  and nothing is pressed until you resize it back or recalibrate.
- Town mode plays a sound and flashes the status when it starts or stops.

## Triggers
Under the flasks: **+ Add trigger** watches an area (e.g. the blue layer on the life globe) and
presses a key when its colour disappears. Calibrate it while the colour is fully showing (box,
click the colour, Similarity slider). Edit sets the name, key (default Q, combinations allowed),
the "start pressing below" and "stop when back above" percentages (default 20% / 50%), repeat
on/off and the random repeat interval (default 200-600 ms).

## Auto-update
Releases come from https://github.com/Danielb1606/DanielsAssignment/releases.
To publish a new version: raise `__version__` in `daniels_assignment.py` and push to main.
GitHub builds the .exe (`.github/workflows/release.yml`) and the app offers the update
the next time it opens.

## Command line (optional)
    python daniels_assignment.py calibrate | test | run
