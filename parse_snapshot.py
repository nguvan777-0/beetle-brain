"""
Parse a beetle-brain snapshot.npz and print a human/AI readable summary.

Usage:
    uv run --with numpy python parse_snapshot.py [snapshot.npz]
"""
import sys
import tomllib
import numpy as np
from pathlib import Path

# --- load config ---

cfg_path = Path(__file__).parent / "config.toml"
with open(cfg_path, "rb") as f:
    cfg = tomllib.load(f)

t  = cfg["traits"]
e  = cfg["energy"]
ev = cfg["evolution"]
ag = cfg["aging"]
h  = cfg["hgt"]
b  = cfg["brain"]

N_HIDDEN = b["n_hidden"]
N_RAYS   = b["n_rays"]

TRAITS = [
    # (label, gene_idx, min, max)
    ("speed",         0,  t["speed_min"],           t["speed_max"]),
    ("fov",           1,  np.degrees(t["fov_min"]),  np.degrees(t["fov_max"])),
    ("ray_len",       2,  t["ray_min"],              t["ray_max"]),
    ("size",          3,  t["size_min"],             t["size_max"]),
    ("breed_at",      8,  e["breed_at_min"],         e["breed_at_max"]),
    ("clone_with",    9,  e["clone_with_min"],        e["clone_with_max"]),
    ("mutation_rate", 10, ev["mutation_rate_min"],   ev["mutation_rate_max"]),
    ("mutation_scale",11, ev["mutation_scale_min"],  ev["mutation_scale_max"]),
    ("epigenetic",    12, ev["epigenetic_min"],      ev["epigenetic_max"]),
    ("weight_decay",  13, ag["weight_decay_min"],    ag["weight_decay_max"]),
    ("mouth",         14, e["mouth_min"],             e["mouth_max"]),
    ("pred_ratio",    15, e["pred_ratio_min"],        e["pred_ratio_max"]),
    ("hgt_eat_rate",  16, h["eat_rate_min"],          h["eat_rate_max"]),
    ("hgt_contact",   17, h["contact_rate_min"],      h["contact_rate_max"]),
]

BLOCKS = " ▁▂▃▄▅▆▇█"

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def pct(val, lo, hi):
    return 100 * (val - lo) / (hi - lo) if hi > lo else 0.0

def sparkline(series, width=28):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return BLOCKS[4] * width
    norm = (series - lo) / (hi - lo)
    idxs = np.round(norm * (len(BLOCKS) - 1)).astype(int)
    # downsample to width
    step = max(1, len(idxs) // width)
    sampled = idxs[::step][:width]
    return "".join(BLOCKS[i] for i in sampled)

def find_crossings(series, ticks, threshold):
    """Return tick where series first crosses threshold."""
    for i, v in enumerate(series):
        if v >= threshold:
            return int(ticks[i])
    return None

# --- load snapshot ---

path  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("snapshot.npz")
d     = np.load(path, allow_pickle=True)
s     = sig(d["W_body"])
n     = len(d["x"])
tick  = int(d["tick"].item())
seed  = int(d["seed"].item())
gen   = d["generation"]
age   = d["age"]
eaten = d["eaten"]

# --- decode traits at snapshot ---

decoded = {}
for label, idx, lo, hi in TRAITS:
    decoded[label] = lo + s[:, idx] * (hi - lo)

active_neurons = (s[:, 18] * N_HIDDEN).astype(int)
n_rays         = (s[:, 19] * (N_RAYS + 1)).astype(int)

# --- stats (rich 500-tick samples) ---

import json as _json
samples = None
if 'stats_json' in d:
    payload = _json.loads(d['stats_json'].tobytes().decode())
    samples = payload.get('samples', [])

# --- history (30-tick fallback) ---

hist = d["hist"]  # (T, 7): tick, pop, max_gen, mean_speed, mean_fov, mean_size, mean_mutation_rate
H = {
    "tick":     hist[:, 0],
    "pop":      hist[:, 1],
    "max_gen":  hist[:, 2],
    "speed":    hist[:, 3],
    "fov":      hist[:, 4],
    "size":     hist[:, 5],
    "mut_rate": hist[:, 6],
}
T = len(hist)
mid = T // 2

# ══════════════════════════════════════════════════════════════
print("=" * 60)
print(f"  beetle-brain snapshot")
print(f"  tick {tick:,}  ·  seed {seed}  ·  pop {n}")
print("=" * 60)

print(f"\n  max gen    {gen.max():>6}    mean {gen.mean():.1f}")
print(f"  max age    {age.max():>6,}    mean {age.mean():.0f}")
print(f"  max eaten  {eaten.max():>6}    mean {eaten.mean():.2f}")

# immortal grazer — born at approximately tick - max_age
born_at = tick - int(age.max())
print(f"\n  oldest wight born ~tick {born_at:,}  (survived {age.max():,} ticks)")

print(f"\n  food on map : {len(d['food'])}")
print(f"  vents       : {len(d['vents'])}")
coastline = cfg["world"]["coastline_x"]
for i, v in enumerate(d["vents"]):
    side = "sea" if v[0] < coastline else "land"
    print(f"    vent {i+1}  x={v[0]:.0f}  y={v[1]:.0f}  ({side})")

# ── snapshot traits ────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"  {'trait':<18}  {'mean':>7}  {'min':>7}  {'max':>7}  {'range%':>7}")
print(f"{'─'*60}")

for label, idx, lo, hi in TRAITS:
    vals = decoded[label]
    m    = vals.mean()
    unit = "°" if label == "fov" else ""
    print(f"  {label+unit:<18}  {m:>7.2f}  {vals.min():>7.2f}  {vals.max():>7.2f}"
          f"  {pct(m,lo,hi):>6.0f}%")

an_mean = active_neurons.mean()
nr_mean = n_rays.mean()
print(f"  {'active_neurons':<18}  {an_mean:>7.1f}  {active_neurons.min():>7}  {active_neurons.max():>7}"
      f"  {pct(an_mean,0,N_HIDDEN):>6.0f}%")
print(f"  {'n_rays':<18}  {nr_mean:>7.2f}  {n_rays.min():>7}  {n_rays.max():>7}"
      f"  {pct(nr_mean,0,N_RAYS):>6.0f}%")
print(f"{'─'*60}")

r_col  = (40 + s[:, 4] * 215).mean()
g_col  = (40 + s[:, 5] * 215).mean()
b_col  = (40 + s[:, 6] * 215).mean()
print(f"\n  mean color  r={r_col:.0f}  g={g_col:.0f}  b={b_col:.0f}")

# ── time series ────────────────────────────────────────────────
if samples and len(samples) > 1:
    # rich path: use StatsCollector samples (500-tick, 20+ traits)
    sticks = np.array([s['tick'] for s in samples])
    smid   = len(samples) // 2
    t0s, tms, t1s = int(sticks[0]), int(sticks[smid]), int(sticks[-1])

    print(f"\n{'─'*60}")
    print(f"  trajectory  (tick {t0s:,} → {tms:,} → {t1s:,})  [500-tick samples]")
    print(f"{'─'*60}")

    def _col(key):
        return np.array([s[key] for s in samples])

    rows = [
        ("pop",          _col('pop'),              None, None),
        ("max_gen",      _col('max_gen'),           None, None),
        ("size",         _col('size_mean'),         t["size_min"], t["size_max"]),
        ("speed",        _col('speed_mean'),        t["speed_min"], t["speed_max"]),
        ("fov°",         _col('fov_mean_deg'),      0, 162),
        ("pred_ratio",   _col('pred_ratio_mean'),   e["pred_ratio_min"], e["pred_ratio_max"]),
        ("mut_rate",     _col('mutation_mean'),     ev["mutation_rate_min"], ev["mutation_rate_max"]),
        ("n_rays",       _col('n_rays_mean'),       0, N_RAYS),
        ("active_neur",  _col('active_neurons_mean'), 0, N_HIDDEN),
    ]

    for label, series, lo, hi in rows:
        v0, vm, v1 = series[0], series[smid], series[-1]
        spark = sparkline(series)
        print(f"  {label:<12}  {v0:>6.1f} → {vm:>6.1f} → {v1:>6.1f}  {spark}")

    # drain breakdown
    print(f"\n  drain / tick (mean at end)")
    for key, label in [
        ('drain_kleiber', 'kleiber'),
        ('drain_speed',   'speed'),
        ('drain_size',    'size'),
        ('drain_sensing', 'sensing'),
        ('drain_brain',   'brain'),
    ]:
        if key in samples[-1]:
            series = _col(key)
            print(f"    {label:<10}  {series[-1]:>7.4f}  {sparkline(series)}")

    # key moments
    size_series = _col('size_mean')
    pop_series  = _col('pop')
    print(f"\n  key moments")

    size_80 = find_crossings(size_series, sticks,
                             t["size_min"] + 0.8 * (t["size_max"] - t["size_min"]))
    if size_80:
        print(f"    tick {size_80:>6,}  size crossed 80% of range")

    size_90 = find_crossings(size_series, sticks,
                             t["size_min"] + 0.9 * (t["size_max"] - t["size_min"]))
    if size_90:
        print(f"    tick {size_90:>6,}  size crossed 90% — ceiling locked")

    pop_peak = int(sticks[np.argmax(pop_series)])
    print(f"    tick {pop_peak:>6,}  population peak ({int(pop_series.max())} wights)")

    if born_at > 0:
        print(f"    tick {born_at:>6,}  oldest living wight born")

    size_diff = np.abs(np.diff(size_series))
    fastest_i = np.argmax(size_diff)
    fastest_t = int(sticks[fastest_i])
    print(f"    tick {fastest_t:>6,}  fastest size change ({size_series[fastest_i]:.2f}→{size_series[fastest_i+1]:.2f})")

elif T > 1:
    # fallback: hist (30-tick, 6 traits)
    t0, tm, t1 = int(H["tick"][0]), int(H["tick"][mid]), int(H["tick"][-1])

    print(f"\n{'─'*60}")
    print(f"  trajectory  (tick {t0:,} → {tm:,} → {t1:,})")
    print(f"{'─'*60}")

    rows = [
        ("pop",      H["pop"],      None, None),
        ("max_gen",  H["max_gen"],  None, None),
        ("size",     H["size"],     t["size_min"], t["size_max"]),
        ("speed",    H["speed"],    t["speed_min"], t["speed_max"]),
        ("fov°",     np.degrees(H["fov"]), 0, 162),
        ("mut_rate", H["mut_rate"], ev["mutation_rate_min"], ev["mutation_rate_max"]),
    ]

    for label, series, lo, hi in rows:
        v0, vm, v1 = series[0], series[mid], series[-1]
        spark = sparkline(series)
        print(f"  {label:<10}  {v0:>6.1f} → {vm:>6.1f} → {v1:>6.1f}  {spark}")

    # key moments
    print(f"\n  key moments")

    size_80 = find_crossings(H["size"], H["tick"],
                             t["size_min"] + 0.8 * (t["size_max"] - t["size_min"]))
    if size_80:
        print(f"    tick {size_80:>6,}  size crossed 80% of range")

    size_90 = find_crossings(H["size"], H["tick"],
                             t["size_min"] + 0.9 * (t["size_max"] - t["size_min"]))
    if size_90:
        print(f"    tick {size_90:>6,}  size crossed 90% — ceiling locked")

    pop_peak = int(H["tick"][np.argmax(H["pop"])])
    print(f"    tick {pop_peak:>6,}  population peak ({int(H['pop'].max())} wights)")

    if born_at > 0:
        print(f"    tick {born_at:>6,}  oldest living wight born")

    size_diff = np.abs(np.diff(H["size"]))
    fastest_i = np.argmax(size_diff)
    fastest_t = int(H["tick"][fastest_i])
    print(f"    tick {fastest_t:>6,}  fastest size change ({H['size'][fastest_i]:.2f}→{H['size'][fastest_i+1]:.2f})")

print()
