# Daniels Assignment

## Make the app (once)
Double-click `build_exe.bat` (needs Python installed, only for this step).
It creates `dist\DanielsAssignment.exe`. Copy it anywhere and double-click it.
Calibration is saved in a `daniels_assignment_data` folder next to the .exe.

## Using it
1. **Program window** - pick your program from the list (⟳ refreshes it).
2. **Calibrate all flasks** - with every flask full: drag a box around each flask,
   click its liquid, then accept or redraw the timer-bar boxes for the quicksilvers.
   A results screen shows what's working.
3. **Recalibrate** under any flask redoes just that one (e.g. new or bigger art).
4. **Test mode** - each card shows FULL / NOT FULL / RUNNING live. Nothing is pressed.
5. **Start** - runs it. F8 pause/resume, F9 leave town mode, F10 stop.

## Auto-update
Set `UPDATE_REPO = "owner/repo"` in `updater.py`, push this folder to that GitHub
repo, then push a version tag (e.g. `git tag v1.0.1 && git push --tags`).
GitHub builds the .exe (`.github/workflows/release.yml`) and the app offers the update
the next time it opens.

## Command line (optional)
    python daniels_assignment.py calibrate | test | run
