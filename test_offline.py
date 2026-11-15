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
assert ctl.town_mode and len(sim.presses) - n0 <= 3

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
    moved = np.roll(sub, 2, axis=1)                                   # texture moves 2 px
    sub[m.mask] = moved[m.mask]
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
