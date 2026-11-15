"""Checks for the self-updater (success, rollback, notice, clean environment)."""
import os, sys, json, subprocess, tempfile, time
from pathlib import Path
HERE = str(Path(__file__).resolve().parent); sys.path.insert(0, HERE)
import updater
updater.START_TIMEOUT = 8

def fake_fetch(content):
    def f(info, dest, progress):
        dest.write_bytes(content); progress(1.0)
    return f

def setup():
    d = Path(tempfile.mkdtemp()); exe = d / "DanielsAssignment.exe"; exe.write_bytes(b"OLD")
    return d, exe, d / "daniels_assignment_data"

# 1. success: new copy starts and confirms
d, exe, data = setup()
starter = f"""
import sys; sys.path.insert(0, r'{HERE}'); import updater
from pathlib import Path
print(updater.on_startup(Path(r'{data}'), '1.2.0', exe=Path(r'{exe}')))
"""
launch_ok = lambda e: subprocess.Popen([sys.executable, "-c", starter], env=updater.clean_env())
ok = updater.install_and_restart({"version": "1.2.0", "notes": "Fixed the updater"}, data, "1.1.1",
                                 exe=exe, launch=launch_ok, fetch=fake_fetch(b"NEW"))
assert ok and exe.read_bytes() == b"NEW", (ok, exe.read_bytes())
# (on Windows .old.exe stays until the old copy exits; Linux lets it be deleted at once)
first = updater.on_startup(data, "1.2.0", exe=exe)          # the app's next start: already confirmed
assert first is None and not (data / updater.MARKER).exists()
assert not (d / "DanielsAssignment.old.exe").exists()       # tidied up
print("success path ok")

# 2. new version crashes on start -> rolled back
d, exe, data = setup()
launch_bad = lambda e: subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(1)"])
ok = updater.install_and_restart({"version": "1.2.0"}, data, "1.1.1", exe=exe, launch=launch_bad,
                                 fetch=fake_fetch(b"BROKEN"))
assert not ok and exe.read_bytes() == b"OLD" and not (data / updater.MARKER).exists()
print("rollback path ok")

# 3. "Updated to" info shown once
d, exe, data = setup(); data.mkdir()
(data / updater.MARKER).write_text(json.dumps({"from": "1.1.1", "to": "1.2.0", "notes": "x", "started": False}))
info = updater.on_startup(data, "1.2.0", exe=exe)
assert info and info["from"] == "1.1.1" and json.loads((data / updater.MARKER).read_text())["started"]
assert updater.on_startup(data, "1.2.0", exe=exe) is None
print("updated-notice ok")

# 4. clean environment drops PyInstaller's private variables
os.environ["_MEIPASS2"] = "C:/x"; os.environ["_PYI_APPLICATION_HOME_DIR"] = "C:/y"
env = updater.clean_env()
assert "_MEIPASS2" not in env and "_PYI_APPLICATION_HOME_DIR" not in env and env["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
print("clean env ok")
print("ALL UPDATER CHECKS PASSED")
