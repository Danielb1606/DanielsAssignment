@echo off
rem Builds DanielsAssignment.exe. Needs Python installed on this PC (only for building).
cd /d "%~dp0"
echo Installing build tools...
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt pyinstaller || goto :fail
echo Building...
python -m PyInstaller --onefile --windowed --name DanielsAssignment ^
  --collect-submodules pynput --hidden-import pynput.keyboard._win32 --hidden-import pynput.mouse._win32 ^
  daniels_assignment.py || goto :fail
echo.
echo Done: %~dp0dist\DanielsAssignment.exe
echo You can copy that .exe anywhere; it saves its calibration in a folder next to itself.
pause
exit /b 0
:fail
echo.
echo Build failed - see the messages above.
pause
exit /b 1
