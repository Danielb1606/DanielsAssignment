"""Offline checks for Daniels Assignment using the user's screenshot."""
import random, shutil, sys
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import daniels_assignment as da

shutil.rmtree(da.DATA_DIR, ignore_errors=True)
src = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "flasks.png")
# pad below so there is room for timer bars, like the real UI
img = cv2.copyMakeBorder(src, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=(5, 5, 5))
boxes = {1: [8, 8, 50, 118], 2: [68, 12, 52, 112], 3: [126, 4, 58, 124],
         4: [188, 4, 58, 124], 5: [248, 4, 58, 124]}
bars = {s: [boxes[s][0], 141 + 6, boxes[s][2], 16] for s in (3, 4, 5)}

cfg = da.default_config()
points = {1: (33, 80), 2: (93, 85), 3: (140, 95), 4: (202, 95), 5: (262, 95)}
print("\n".join(da.calibrate_from_image(img, cfg, boxes, bars, points)))
cfg = da.load_config()

def read(image):
    return da.FlaskReader(cfg, frame_source=lambda: image).read()

def show(tag, r):
    print(f"{tag:<34}", "  ".join(f"{k}:{'--' if not v.visible else ('FULL' if v.full else 'not')}"
                                 f"{'+RUN' if v.active else ''}({v.fill:.0%})" for k, v in r.items()))
    return r

# 1. all full
r = show("all full", read(img))
assert all(v.full and v.visible and not v.active for v in r.values())

# 2. drain life to ~70% and quicksilver 4 to ~50% (paint dark glass over the top)
d = img.copy()
m1 = da.model_for(cfg['slots'][0])
x, y, w, h = boxes[1]; cut = m1.top + int(0.3 * (m1.bottom - m1.top))
d[y:y + cut, x:x + w][m1.mask[:cut]] = (40, 38, 36)
m4 = da.model_for(cfg['slots'][3])
x, y, w, h = boxes[4]; cut = m4.top + int(0.5 * (m4.bottom - m4.top))
d[y:y + cut, x:x + w][m4.mask[:cut]] = (40, 38, 36)
r = show("life 70%, qs4 50%", read(d))
assert not r[1].full and r[2].full and not r[4].full and r[3].full

# 3. timer bars (partial lengths like the user's screenshot)
b = d.copy()
for s, length in ((4, 17), (5, 40)):
    bx, by, bw, bh = bars[s]
    cv2.rectangle(b, (bx + 2, by + 4), (bx + 2 + length, by + 11), (153, 215, 249), -1)
cv2.rectangle(b, (bars[3][0] + 2, bars[3][1] + 4), (bars[3][0] + 4, bars[3][1] + 11), (153, 215, 249), -1)  # nearly expired
r = show("bars on 3 (tiny), 4, 5", read(b))
assert r[3].active and r[4].active and r[5].active and not r[1].active

fr = da.FlaskReader(cfg, frame_source=lambda: b)
cv2.imwrite(str(da.DATA_DIR / "test_panel.png"), da.render_panel(cfg, fr, fr.read()))

# 4. belt hidden (loading screen)
hidden = np.full_like(img, 12)
r = show("loading screen", read(hidden))
assert not any(v.visible for v in r.values())

# 5. recalibrate slot 1 with a bigger box (new, larger life flask art)
big = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
lines = da.calibrate_from_image(big, cfg, {1: [4, 2, 58, 128]}, {}, {1: (33, 80)})
print("\n".join(l for l in lines if "Slot 1" in l or "slot 1" in l))
cfg = da.load_config()
r = show("after slot 1 recalibrated bigger", read(big))
assert r[1].full


# 6. controller: town mode + quicksilver rotation, with a simulated UI
class Sim:
    def __init__(s): s.t = 0.0; s.town = False; s.full = {k: True for k in range(1, 6)}; s.active_until = {k: 0 for k in range(1, 6)}; s.hidden_until = 0; s.refill_at = {}
    def now(s): return s.t
    def sleep(s, dt): s.t += dt
    def read(s):
        for k, t in list(s.refill_at.items()):
            if s.t >= t: s.full[k] = True; del s.refill_at[k]
        vis = s.t >= s.hidden_until
        return {k: da.SlotReading(vis, s.full[k] and vis, s.t < s.active_until[k] and vis, 1.0, 1.0) for k in range(1, 6)}
    def press(s, k):
        s.presses.append((round(s.t, 3), k))
        if s.town: return True          # key sent, flask does nothing
        s.full[k] = False; s.refill_at[k] = s.t + (8 if k >= 3 else 3)
        if k >= 3: s.active_until[k] = s.t + 4
        return True

sim = Sim(); sim.presses = []
logs = []
ctl = da.FlaskController(sim.read, sim.press, clock=sim.now, sleep=sim.sleep, rng=random.Random(2), log=logs.append)
ctl.run(12)
qs = [p for p in sim.presses if p[1] >= 3]
assert all(b[0] - a[0] >= 4 for a, b in zip(qs, qs[1:])), qs
delays = [float(l.split("in ")[1].split(" ms")[0]) for l in logs if "press in" in l]
assert all(150 <= d <= 300 for d in delays), delays
print(f"\nfield: {len(sim.presses)} presses, quicksilver presses at {[p[0] for p in qs]}, delays {min(delays):.0f}-{max(delays):.0f} ms")

# enter town: flasks refill and stop working
sim.town = True
for k in range(1, 6): sim.full[k] = True; sim.active_until[k] = 0
sim.refill_at.clear()
n0 = len(sim.presses)
ctl.run(10)
print("town:", [l for l in logs if "TOWN" in l], f"presses in town: {len(sim.presses) - n0}")
assert ctl.town_mode and len(sim.presses) - n0 <= 8   # two quicksilver tries, plus life/mana retries

# leave town via loading screen
sim.town = False; sim.hidden_until = sim.t + 2
ctl.run(6)
print("after area change:", [l for l in logs if "TOWN MODE OFF" in l], f"town_mode={ctl.town_mode}")
assert not ctl.town_mode and len(sim.presses) > n0 + 3


# 7. settings: validation, custom keys/timing reach the controller and key sender
st = da.get_settings({})
assert st["keys"]["3"] == "3" and not da.validate_settings(st)
bad = da.get_settings({"settings": {"keys": {"1": "1", "2": "1"}, "delay_min_ms": 400, "delay_max_ms": 100}})
errs = da.validate_settings(bad)
assert any("different key" in e for e in errs) and any("min must be" in e for e in errs), errs
custom = da.get_settings({"settings": {"delay_min_ms": 200, "delay_max_ms": 250, "fails_for_town": 3}})
c = da.FlaskController(lambda: None, lambda s: True, settings=custom)
assert c.delay_range == (0.2, 0.25) and c.fails_for_town == 3
town_events = []
c.on_town = town_events.append
c.enter_town("x"); c.leave_town("y")
assert town_events == [True, False]

# 8. resize: reader reports nothing (so nothing is pressed) when the window size changed
class Loc:
    size = (322, 181)
    def client_size(self): return self.size
    def origin(self): return (0, 0)
class Grab:
    def grab(self, l, t, w, h): return img[t:t + h, l:l + w]
cfg2 = da.load_config(); cfg2["window_size"] = [322, 181]
loc = Loc(); rd = da.FlaskReader(cfg2, loc, Grab())
assert rd.read() is not None and rd.size_mismatch is None
loc.size = (400, 220)
assert rd.read() is None and rd.size_mismatch == ((400, 220), (322, 181))
print("settings / resize / town-alert checks passed")

# 9. liquid with uneven colour: one click should cover most of each flask's liquid
cfg3 = da.default_config()
da.calibrate_from_image(img, cfg3, boxes, bars, points)
cfg3 = da.load_config()
shares = {s["slot"]: da.model_for(s).liquid_share for s in cfg3["slots"]}
print("liquid share per flask:", {k: f"{v:.0%}" for k, v in shares.items()})
assert all(v >= 0.12 for v in shares.values()), shares
# small brightness noise (rendering differences) must not flip FULL off
rng = np.random.default_rng(0)
noisy = np.clip(img.astype(int) + rng.integers(-12, 13, img.shape), 0, 255).astype(np.uint8)
r = da.FlaskReader(cfg3, frame_source=lambda: noisy).read()
assert all(v.full for v in r.values()), {k: v.top_match for k, v in r.items()}
# a quicksilver drained to half must read NOT FULL even though its liquid is shaded
d2 = img.copy()
m3 = da.model_for(da.slot_cfg(cfg3, 3)); x, y, w, h = boxes[3]
cut = m3.top + int(0.5 * (m3.bottom - m3.top))
glass = img[y + 2:y + 6, x + 2:x + 6].reshape(-1, 3).mean(0).astype(np.uint8)
d2[y:y + cut, x:x + w][m3.mask[:cut]] = glass
r = da.FlaskReader(cfg3, frame_source=lambda: d2).read()
assert not r[3].full and r[4].full, (r[3], r[4])
print("uneven-colour liquid checks passed")
print("ALL CHECKS PASSED")

# 10. shimmering liquid: hue jitter, brightness changes and moved texture inside the
#     liquid must still read FULL; drained must not
cfg4 = da.load_config()
rng = np.random.default_rng(1)
sh = img.copy()
hsv = cv2.cvtColor(sh, cv2.COLOR_BGR2HSV).astype(int)
for slot in range(1, 6):
    sc = da.slot_cfg(cfg4, slot); m = da.model_for(sc); x, y, w, h = sc["box"]
    sub = hsv[y:y + h, x:x + w]
    moved = np.roll(sub, 2, axis=1)                                   # texture moves 2 px...
    inside = m.mask & np.roll(m.mask, 2, axis=1)                      # ...within the liquid
    sub[inside] = moved[inside]
    sub[..., 0][m.mask] = (sub[..., 0][m.mask] + rng.integers(-4, 5, m.mask.sum())) % 180
    sub[..., 2][m.mask] = np.clip(sub[..., 2][m.mask] * rng.uniform(0.8, 1.15, m.mask.sum()), 0, 255)
sh = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
r = da.FlaskReader(cfg4, frame_source=lambda: sh).read()
assert all(v.full for v in r.values()), {k: round(v.top_match, 2) for k, v in r.items()}
print("shimmer: all FULL", {k: f"{v.top_match:.0%}" for k, v in r.items()})

# 11. finding quicksilvers 4 and 5 from flask 3
found = da.find_similar(img, boxes[3], count=2)
assert len(found) == 2 and abs(found[0][0] - boxes[4][0]) <= 2 and abs(found[1][0] - boxes[5][0]) <= 2, found
print("auto-find:", found)

# 12. pick modes: 'shape' keeps out the cork/arch; 'box' takes every match in the box
ref1 = img[boxes[1][1]:boxes[1][1] + boxes[1][3], boxes[1][0]:boxes[1][0] + boxes[1][2]]
p1 = [(33 - boxes[1][0], 80 - boxes[1][1])]
shape = da.liquid_selection(ref1, p1, 55, "shape"); anywhere = da.liquid_selection(ref1, p1, 55, "box")
assert anywhere.sum() >= shape.sum() > 0
assert not shape[:25].any(), "shape mode picked something in the cork/neck area"
ref3 = img[boxes[3][1]:boxes[3][1] + boxes[3][3], boxes[3][0]:boxes[3][0] + boxes[3][2]]
q = da.liquid_selection(ref3, [(140 - boxes[3][0], 95 - boxes[3][1])], 55, "shape")
rows = np.nonzero(q.any(1))[0]
assert rows.min() < 50 and rows.max() > 95, (rows.min(), rows.max())   # above AND below the wings
print("pick modes ok; quicksilver liquid rows", rows.min(), "-", rows.max())
print("ALL CHECKS PASSED")

# 13. auto-detection: colour detection finds all five, boxes contain the right flasks
for path in [sys.argv[1] if len(sys.argv) > 1 else "flasks.png"]:
    raw = cv2.imread(path)
    det = da.auto_detect(raw, da.default_config())
    assert sorted(det) == [1, 2, 3, 4, 5], det.keys()
    for s_, (bx, by, bw, bh) in boxes.items():
        dx, dy, dw, dh = det[s_]["box"]
        cx = dx + dw / 2
        assert bx <= cx <= bx + bw, (s_, det[s_]["box"], boxes[s_])          # centred on the right flask
        assert not det[s_]["unsure"] and det[s_]["liquid_share"] > 0.1, (s_, det[s_])
print("auto-detect (colour): all five found")

# 14. reuse of the saved calibration when the window moved
da.calibrate_from_image(img, da.default_config(), boxes, bars, points)
cfgp = da.load_config()
moved = cv2.copyMakeBorder(img, 37, 0, 55, 0, cv2.BORDER_REPLICATE)          # belt 55 px right, 37 px down
det = da.auto_detect(moved, cfgp)
for s_ in range(1, 6):
    d = det[s_]
    assert d["source"] == "previous" and d["box"][:2] == [boxes[s_][0] + 55, boxes[s_][1] + 37], (s_, d["box"])
    if s_ >= 3:
        assert d["bar_box"][:2] == [bars[s_][0] + 55, bars[s_][1] + 37]
print("auto-detect (saved calibration, window moved): all five reused")
print("ALL CHECKS PASSED")

# 15. town mode: one flask that never visibly changes must NOT trigger it;
#     a flask that drains a little (but still reads FULL) counts as working
class One:
    def __init__(s): s.t = 0.0; s.sig = {k: np.zeros((20, 12), np.float32) for k in range(1, 6)}; s.presses = []
    def now(s): return s.t
    def sleep(s, dt): s.t += dt
    def read(s): return {k: da.SlotReading(True, True, False, 1.0, 1.0, s.sig[k].copy()) for k in range(1, 6)}
    def press(s, k):
        s.presses.append(k)
        if k != 1: s.sig[k] += 20          # visibly drains but still reads FULL
        return True                        # slot 1 (life) never changes
sim = One(); logs = []
c = da.FlaskController(sim.read, sim.press, clock=sim.now, sleep=sim.sleep, rng=random.Random(3), log=logs.append)
c.run(1.6)
assert not c.town_mode, logs
assert sim.presses.count(2) == 1, sim.presses          # not re-pressed until it has read NOT FULL
print("single unresponsive flask: no town mode; used flask waits to refill")
sim2 = One(); sim2.press = lambda k: (sim2.presses.append(k), True)[1]     # nothing ever changes = town
c2 = da.FlaskController(sim2.read, sim2.press, clock=sim2.now, sleep=sim2.sleep, rng=random.Random(3), log=lambda m: None)
c2.run(4)
assert c2.town_mode
print("every flask unresponsive: town mode")
print("ALL CHECKS PASSED")

# 16. full life/mana that can't be drunk never cause town mode; quicksilvers alone decide
sim3 = One()
def press3(k):
    sim3.presses.append(k)
    if k >= 3: sim3.sig[k] += 20; return True      # quicksilvers work
    return True                                      # life/mana do nothing (full life/mana)
sim3.press = press3; logs3 = []
c3 = da.FlaskController(sim3.read, sim3.press, clock=sim3.now, sleep=sim3.sleep, rng=random.Random(4), log=logs3.append)
c3.run(8)
assert not c3.town_mode and sim3.presses.count(1) >= 2, (sim3.presses, logs3[-3:])
print("life/mana unusable at full: no town mode, still retried")
sim4 = One(); sim4.press = lambda k: (sim4.presses.append(k), True)[1]
c4 = da.FlaskController(sim4.read, sim4.press, clock=sim4.now, sleep=sim4.sleep, rng=random.Random(4), log=lambda m: None)
c4.run(6)
assert c4.town_mode and len({k for k in sim4.presses if k >= 3}) >= 2, sim4.presses
print("quicksilvers unresponsive (rotating between them): town mode")
print("ALL CHECKS PASSED")

# 17. key hold: random 20-90 ms by default; settings validate
st = da.get_settings({})
assert (st["key_hold_min_ms"], st["key_hold_max_ms"], st["key_method"]) == (20, 90, "scancode")
assert not da.validate_settings(st)
bad = dict(st); bad["key_hold_min_ms"] = 100; bad["key_hold_max_ms"] = 50
assert any("Key hold" in e for e in da.validate_settings(bad))
ks = da.KeySender.__new__(da.KeySender); ks.hold_range = (0.02, 0.09); ks.rng = random.Random(5)
holds = [ks.hold_time() for _ in range(500)]
assert 0.02 <= min(holds) and max(holds) <= 0.09 and max(holds) - min(holds) > 0.05
print(f"key hold random {min(holds)*1000:.0f}-{max(holds)*1000:.0f} ms")
print("ALL CHECKS PASSED")

# 18. mana flask edges: loosening picks up the thin blue edge, never grey glass
mb = [66, 3, 57, 127]; refm = img[mb[1]:mb[1] + mb[3], mb[0]:mb[0] + mb[2]]
hsvm = cv2.cvtColor(refm, cv2.COLOR_BGR2HSV); Hm, Sm, Vm = (hsvm[..., i].astype(int) for i in range(3))
obvious = (da._circ_hue_dist(Hm, 112) <= 18) & (Sm >= 70) & (Vm >= 50)
for clicks in ([(27, 70)], [(27, 70), (14, 53), (19, 44), (22, 41), (24, 40), (34, 47)]):
    for sim in (55, 75, 95):
        m = da.liquid_selection(refm, clicks, sim, "shape")
        assert int((m & (Sm < 50)).sum()) == 0, (clicks, sim)
    assert int((obvious & ~m).sum()) < 0.01 * obvious.sum(), int((obvious & ~m).sum())
print("mana edges: no grey glass at any setting; loose setting gets the blue edge")
print("ALL CHECKS PASSED")

# 19. hotkeys: names, modifiers, conflicts, matching while modifiers are held
assert da.normalize_hotkey("Shift + E") == "shift+e" and da.normalize_hotkey("e+ctrl+shift") == "ctrl+shift+e"
assert da.normalize_hotkey("F7") == "f7" and da.normalize_hotkey("None") == ""
assert da.hotkey_label("ctrl+shift+e") == "Ctrl + Shift + E" and da.hotkey_label("f7") == "F7"
st = da.get_settings({})
assert (st["hotkey_toggle"], st["hotkey_pause"], st["hotkey_town"]) == ("f7", "f8", "f9")
bad = dict(st); bad["hotkey_pause"] = "f7"
assert any("different key" in e for e in da.validate_settings(bad))
bad = dict(st); bad["hotkey_toggle"] = "1"
assert any("same as a flask key" in e for e in da.validate_settings(bad))
ok = dict(st); ok["hotkey_toggle"] = "shift+1"
assert not da.validate_settings(ok)
class _K:                                  # stand-in for pynput's Key / KeyCode
    def __init__(s, name=None, vk=None): s.name, s.vk, s.char = name, vk, None
class _Keys:
    shift = _K("shift"); shift_l = _K("shift_l"); shift_r = _K("shift_r"); ctrl = _K("ctrl"); ctrl_l = _K("ctrl_l")
    ctrl_r = _K("ctrl_r"); alt = _K("alt"); alt_l = _K("alt_l"); alt_r = _K("alt_r"); alt_gr = _K("alt_gr")
class _KB: Key = _Keys
hits = []
hl = da.HotkeyListener(dict(st, hotkey_toggle="shift+e"), hits.append); hl._kb = _KB
hl._press(_K(vk=69)); assert hits == []                                  # plain E: nothing
hl._press(_Keys.shift_l); hl._press(_K(vk=69)); hl._release(_Keys.shift_l)
assert hits == ["hotkey_toggle"], hits                                   # Shift+E
hl._press(_K(vk=119)); assert hits[-1] == "hotkey_pause"                 # F8
hl.paused = True; hl._press(_K(vk=120)); assert hits[-1] == "hotkey_pause"   # ignored while recording
print("hotkeys: Shift+E style combos, defaults F7/F8/F9, conflict checks ok")
print("ALL CHECKS PASSED")

# 19. hotkeys: names, modifiers, conflicts, matching
assert da.normalize_hotkey("Shift + E") == "shift+e" and da.normalize_hotkey("e+ctrl+shift") == "ctrl+shift+e"
assert da.get_settings({})["hotkey_toggle"] == "f7" and da.hotkey_label("shift+e") == "Shift + E"
st = da.get_settings({"settings": {"hotkey_toggle": "1"}})
assert any("flask key" in e for e in da.validate_settings(st))
st = da.get_settings({"settings": {"hotkey_toggle": "shift+1"}})
assert not da.validate_settings(st)
class K:            # stand-in for pynput keys
    shift = shift_l = shift_r = "SHIFT"; ctrl = ctrl_l = ctrl_r = "CTRL"; alt = alt_l = alt_r = alt_gr = "ALT"
class Key:
    def __init__(s, vk): s.vk = vk
got = []
hl = da.HotkeyListener(da.get_settings({"settings": {"hotkey_toggle": "shift+e"}}), got.append)
hl._kb = type("kb", (), {"Key": K})
hl._press(Key(69))                         # E alone: nothing
hl._press("SHIFT"); hl._press(Key(69)); hl._release("SHIFT")
hl._press(Key(119))                        # F8
assert got == ["hotkey_toggle", "hotkey_pause"], got
print("hotkeys ok")
print("ALL CHECKS PASSED")

# 20. program paused (belt dims to ~50%): detected as paused, nothing pressed, no town mode
cfg5 = da.load_config()
dimmed = (img.astype(np.float32) * 0.5).astype(np.uint8)
r = da.FlaskReader(cfg5, frame_source=lambda: dimmed).read()
assert all(v.visible and v.brightness < da.PAUSE_DIM for v in r.values()), {k: round(v.brightness, 2) for k, v in r.items()}
r = da.FlaskReader(cfg5, frame_source=lambda: img).read()
assert all(0.9 < v.brightness < 1.1 for v in r.values())
class P(One):
    def __init__(s): super().__init__(); s.dim = False
    def read(s): return {k: da.SlotReading(True, True, False, 1.0, 1.0, s.sig[k].copy(), 0.5 if s.dim else 1.0) for k in range(1, 6)}
sp = P(); sp.dim = True
cp = da.FlaskController(sp.read, sp.press, clock=sp.now, sleep=sp.sleep, rng=random.Random(6), log=lambda m: None)
cp.run(10)
assert sp.presses == [] and not cp.town_mode and cp.program_paused
sp.dim = False; cp.run(1)
assert sp.presses and not cp.program_paused
print("paused program: no presses, no town mode, resumes at once")

# 21. town mode retry: once quicksilvers work again, town mode ends within the retry time
class T(One):
    def __init__(s): super().__init__(); s.town = True
    def press(s, k):
        s.presses.append(k)
        if not s.town and k >= 3: s.sig[k] += 25
        return True
stt = T(); logs_t = []
ct = da.FlaskController(stt.read, stt.press, clock=stt.now, sleep=stt.sleep, rng=random.Random(7),
                        log=logs_t.append, settings=da.get_settings({"settings": {"town_retry_secs": 3}}))
ct.run(6)
assert ct.town_mode
n_town = len(stt.presses); stt.town = False; t0 = stt.t
ct.run(8)
assert not ct.town_mode, logs_t[-5:]
left = next(l for l in logs_t if "worked again" in l)
print("town retry: left town mode ->", left.strip(), f"(presses while in town: {n_town})")
print("ALL CHECKS PASSED")

# 22. default: in town mode nothing is pressed until a loading screen (belt hidden, then back)
class L(One):
    def __init__(s): super().__init__(); s.town = True; s.hidden = False
    def read(s): return {k: da.SlotReading(not s.hidden, not s.hidden, False, 1.0, 1.0, s.sig[k].copy()) for k in range(1, 6)}
    def press(s, k):
        s.presses.append(k)
        if not s.town and k >= 3: s.sig[k] += 25
        return True
sl = L()
cl = da.FlaskController(sl.read, sl.press, clock=sl.now, sleep=sl.sleep, rng=random.Random(8), log=lambda m: None)
cl.run(5); assert cl.town_mode
n = len(sl.presses); cl.run(20)
assert len(sl.presses) == n, "pressed while in town mode"
sl.town = False; sl.hidden = True; cl.run(1); sl.hidden = False; cl.run(2)
assert not cl.town_mode and len(sl.presses) > n
print("town mode: silent until a loading screen, then resumes")
print("ALL CHECKS PASSED")

# 23. triggers: coverage of the blue layer, fire / repeat / re-arm
globe = cv2.imread(str(Path(__file__).parent / "globe.png"))
gh, gw = globe.shape[:2]
tm = da.TriggerModel(globe, [(269, 162)], 55, "box")
full_cov = tm.coverage(globe)
gone = globe.copy()
hsvg = cv2.cvtColor(gone, cv2.COLOR_BGR2HSV)
bluepx = (da._circ_hue_dist(hsvg[..., 0].astype(int), 100) <= 25) & (hsvg[..., 1] > 50)
gone[bluepx] = (40, 20, 120)                      # blue layer replaced by the red globe underneath
half = globe.copy(); half[:, : gw // 2][bluepx[:, : gw // 2]] = (40, 20, 120)
print(f"trigger coverage: full {full_cov:.0f}%, gone {tm.coverage(gone):.0f}%, partly {tm.coverage(half):.0f}%")
assert full_cov > 90 and tm.coverage(gone) < 5

class Trig:
    def __init__(s): s.t = 0.0; s.cov = 100.0; s.keys = []
    def now(s): return s.t
    def sleep(s, dt): s.t += dt
    def read(s): return {k: da.SlotReading(True, False, False, 0.5, 0.0, None) for k in range(1, 6)}
trig = dict(da.TRIGGER_DEFAULTS, id=1)
tsim = Trig()
ct = da.FlaskController(tsim.read, lambda k: True, clock=tsim.now, sleep=tsim.sleep, rng=random.Random(9),
                        log=lambda m: None, triggers=[trig], read_triggers=lambda: {1: tsim.cov},
                        press_key=lambda combo: (tsim.keys.append((round(tsim.t, 3), combo)), True)[1])
ct.run(2); assert tsim.keys == []                     # blue showing: nothing
tsim.cov = 10; t0 = tsim.t; ct.run(3)                  # gone: q repeatedly
times = [t for t, _ in tsim.keys]
gaps = [b - a for a, b in zip(times, times[1:])]
assert tsim.keys and all(k == "q" for _, k in tsim.keys)
assert 0.15 <= times[0] - t0 <= 0.55, times[0] - t0        # first press after the 150-300 ms delay (+ check interval)
assert all(0.2 <= g <= 0.61 for g in gaps), gaps            # repeats every 0.2-0.6 s
tsim.cov = 35; n = len(tsim.keys); ct.run(1.5)               # between thresholds: still gone -> keeps pressing
assert len(tsim.keys) > n
tsim.cov = 60; ct.run(0.3); n = len(tsim.keys); ct.run(3)     # back above 50%: stops
assert len(tsim.keys) == n
once = dict(trig, repeat=False); tsim2 = Trig(); tsim2.cov = 5
c2 = da.FlaskController(tsim2.read, lambda k: True, clock=tsim2.now, sleep=tsim2.sleep, rng=random.Random(9),
                        log=lambda m: None, triggers=[once], read_triggers=lambda: {1: tsim2.cov},
                        press_key=lambda combo: (tsim2.keys.append(combo), True)[1])
c2.run(3); assert tsim2.keys == ["q"]
assert not da.validate_trigger(dict(trig)) and da.validate_trigger(dict(trig, fire_below=60))
print(f"trigger: {len(times)} presses, gaps {min(gaps):.2f}-{max(gaps):.2f}s, stops when blue returns; once-mode ok")
print("ALL CHECKS PASSED")

# ---- flask roles: slot 3 as a utility flask (used when full and under 10% left) ----
print("\n=== Flask roles ===")
cfg = da.default_config()
da.slot_cfg(cfg, 3)["kind"] = "utility"
kinds = da.slot_kinds(cfg)
assert kinds == {1: "life", 2: "mana", 3: "utility", 4: "quicksilver", 5: "quicksilver"}, kinds

class RoleSim:
    def __init__(s):
        s.t = 0.0; s.presses = []; s.left = 0.5; s.active3 = True; s.qs_until = {}
    def now(s): return s.t
    def sleep(s, d): s.t += d
    def read(s):
        out = {}
        for k in range(1, 6):
            act = s.active3 if k == 3 else s.t < s.qs_until.get(k, -1)
            out[k] = da.SlotReading(True, True, act, 1.0, 1.0, None, 1.0, s.left if act else 0.0)
        return out
    def press(s, k):
        s.presses.append(k)
        if k in (4, 5): s.qs_until[k] = s.t + 4      # quicksilver runs for 4 s
        return True

rs = RoleSim()
rc = da.FlaskController(rs.read, rs.press, clock=rs.now, sleep=rs.sleep, rng=random.Random(9), log=lambda m: None,
                        kinds=kinds, refresh_below={3: 10})
assert rc.quick == (4, 5) and rc.utility == (3,), (rc.quick, rc.utility)
rc.run(3)
assert 3 not in rs.presses, rs.presses          # 50% left: leave it
assert set(p for p in rs.presses if p >= 3) <= {4, 5}
rs.left = 0.05
rc.run(2)
assert 3 in rs.presses, rs.presses              # under 10% left and full: use it
cfg2 = da.default_config(); da.slot_cfg(cfg2, 3)["kind"] = "off"
rs2 = RoleSim(); rs2.active3 = False
rc2 = da.FlaskController(rs2.read, rs2.press, clock=rs2.now, sleep=rs2.sleep, rng=random.Random(9), log=lambda m: None,
                         kinds=da.slot_kinds(cfg2))
rc2.run(4)
assert 3 not in rs2.presses, rs2.presses        # off: never pressed
print("roles: utility waits for <10% left, quicksilvers 4/5 only, 'not used' never pressed - ok")

# ---- globe link + per-flask delay ----
print("\n=== Globe link and per-flask delay ===")
class GlobeSim:
    def __init__(s):
        s.t = 0.0; s.presses = []; s.globe = 80.0
    def now(s): return s.t
    def sleep(s, d): s.t += d
    def read(s):
        out = {}
        for k in range(1, 6):
            rd = da.SlotReading(True, k != 1, k >= 4, 0.5, 1.0, None, 1.0)   # flask 1 NOT full
            if k == 1: rd.globe = s.globe
            out[k] = rd
        return out
    def press(s, k):
        s.presses.append((round(s.t, 3), k)); return True
gs = GlobeSim(); glog = []
cfg = da.default_config()
sc1 = da.slot_cfg(cfg, 1)
sc1["globe"] = {"box": [0, 0, 10, 10], "below": 40, "again_ms": 2000}
sc1["delay_min_ms"], sc1["delay_max_ms"] = 400, 450
opts = da.slot_options(cfg)
assert opts[1]["globe_below"] == 40 and opts[1]["delay"] == (0.4, 0.45), opts[1]
gc = da.FlaskController(gs.read, gs.press, clock=gs.now, sleep=gs.sleep, rng=random.Random(5), log=glog.append,
                        kinds=da.slot_kinds(cfg), slot_opts=opts)
gc.run(2)
assert not any(k == 1 for _, k in gs.presses), gs.presses          # globe 80% > 40%: flask 1 (not full) left alone
gs.globe = 30.0
t0 = gs.t
gc.run(5)
p1 = [t for t, k in gs.presses if k == 1]
assert p1, "flask 1 should be pressed when the globe is low"
assert 0.4 <= p1[0] - t0 <= 0.45 + 0.3, (p1, t0)                   # its own 400-450 ms delay (+ one check)
assert all(b - a >= 2.0 - 0.06 for a, b in zip(p1, p1[1:])), p1    # again only every 2 s (± the random delay) while still low
d = [float(l.split("in ")[1].split(" ms")[0]) for l in glog if "slot 1 globe" in l]
assert d and all(400 <= x <= 450 for x in d), d
print(f"globe: flask 1 pressed at {p1} (globe 30% < 40%), delays {d} - ok")

# ---- changing a flask's role while running applies straight away ----
rs3 = RoleSim(); rs3.active3 = False
rc3 = da.FlaskController(rs3.read, rs3.press, clock=rs3.now, sleep=rs3.sleep, rng=random.Random(9), log=lambda m: None)
rc3.run(1)
cfgu = da.default_config(); da.slot_cfg(cfgu, 3)["kind"] = "utility"
rc3.update_options(**da.controller_options(cfgu))
n = len(rs3.presses)
rc3.run(2)
assert rc3.utility == (3,) and 3 in rs3.presses[n:], rs3.presses
print("live role change: flask 3 used as utility without restarting - ok")
