"""
Benchmark all CoreML compute unit options and print a perf table.

Three measurements per backend:
  start   — first 500-tick interval from a fresh world (pop ~12)
  mature  — last 500-tick interval of the main run (pop has grown)
  maxpop  — full MAX_POP world, positions randomised, timed separately

Usage:
    uv run --with numpy --with coremltools python scripts/bench_compute_units.py
    uv run --with numpy --with coremltools python scripts/bench_compute_units.py --duration 30

Internal:
    python scripts/bench_compute_units.py --worker <duration>
    (called by the script itself as a subprocess for the maxpop phase)
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── worker mode (subprocess entry point for maxpop phase) ─────────────────────

if "--worker" in sys.argv:
    import time
    import numpy as np
    from sim.config import MAX_POP, WIDTH, HEIGHT
    from sim import new_world, init_ane
    from sim.tick import tick as sim_tick
    from game.snapshot import load_snapshot

    duration = float(sys.argv[sys.argv.index("--worker") + 1])

    rng = np.random.default_rng(12345)
    world, _, _, _, _ = load_snapshot(rng)
    if world is None or len(world['pop']['x']) == 0:
        world = new_world(seed=12345)

    pop = world['pop']
    n   = len(pop['x'])
    if n < MAX_POP:
        reps    = (MAX_POP + n - 1) // n
        new_pop = {}
        for k, v in pop.items():
            tiled      = np.tile(v, (reps,) + (1,) * (v.ndim - 1))
            new_pop[k] = tiled[:MAX_POP]
        tile_rng         = np.random.default_rng(0)
        new_pop['x']     = tile_rng.uniform(0, WIDTH,       MAX_POP).astype(np.float32)
        new_pop['y']     = tile_rng.uniform(0, HEIGHT,      MAX_POP).astype(np.float32)
        new_pop['angle'] = tile_rng.uniform(0, 2 * np.pi,  MAX_POP).astype(np.float32)
        world['pop']     = new_pop
        print(f"[bench] population tiled {n} → {MAX_POP}, positions randomised")

    init_ane()
    import brain.coreml_sense_brain as _sb
    t_wait = time.time()
    while not _sb._use_coreml and time.time() - t_wait < 300:
        time.sleep(0.05)

    ticks = 0
    t0    = time.time()
    while time.time() - t0 < duration:
        world  = sim_tick(world, rng)
        ticks += 1

    elapsed = time.time() - t0
    print(f"{ticks:,} ticks  {elapsed:.1f}s  {int(ticks / elapsed):,} t/s")
    sys.exit(0)


# ── orchestrator mode ─────────────────────────────────────────────────────────

OPTIONS = [
    ("numpy fallback", None),
    ("CPU_AND_GPU",    "CPU_AND_GPU"),
    ("CPU_ONLY",       "CPU_ONLY"),
    ("ALL",            "ALL"),
    ("CPU_AND_NE",     "CPU_AND_NE"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=int, default=60,
                    help="seconds per run (default: 60)")
args = parser.parse_args()

THIS = str(Path(__file__).resolve())


def _env(compute_unit):
    env = os.environ.copy()
    if compute_unit:
        env["BEETLE_COMPUTE_UNITS"] = compute_unit
    return env


def _deps(compute_unit):
    deps = ["--with", "numpy"]
    if compute_unit is not None:
        deps += ["--with", "coremltools"]
    return deps


def _run_world(compute_unit, duration):
    cmd = [
        "uv", "run", *_deps(compute_unit),
        "python", "world.py", str(duration), "--new", "--seed", "12345", "--bench",
    ]
    r = subprocess.run(cmd, cwd=ROOT, env=_env(compute_unit),
                       capture_output=True, text=True)
    return r.stdout + r.stderr


def _run_maxpop(compute_unit, duration):
    cmd = [
        "uv", "run", *_deps(compute_unit),
        "python", THIS, "--worker", str(duration),
    ]
    r = subprocess.run(cmd, cwd=ROOT, env=_env(compute_unit),
                       capture_output=True, text=True)
    return r.stdout + r.stderr


def _parse_load(output):
    def _f(pat):
        m = re.search(pat, output)
        return float(m.group(1)) if m else None
    vals = [x for x in [_f(r"\[Brain\] ready \(([0-9.]+)s\)"),
                         _f(r"\[SenseBrain\] ready \(([0-9.]+)s\)")] if x is not None]
    if not vals:
        return "instant", 0.0
    load_s = max(vals)
    return f"~{load_s:.1f}s", load_s


_TICK_RE = re.compile(
    r'^\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)s',
    re.MULTILINE,
)


def _parse_intervals(output):
    lines = _TICK_RE.findall(output)
    if not lines:
        return None, None, None, None
    t0, p0, e0 = int(lines[0][0]), int(lines[0][1]), float(lines[0][2])
    start_tps  = int(t0 / e0) if e0 > 0 else None
    if len(lines) >= 2:
        t1, _,  e1 = int(lines[-2][0]), int(lines[-2][1]), float(lines[-2][2])
        t2, p2, e2 = int(lines[-1][0]), int(lines[-1][1]), float(lines[-1][2])
        dt         = e2 - e1
        mature_tps = int((t2 - t1) / dt) if dt > 0 else None
        mature_pop = p2
    else:
        mature_tps, mature_pop = start_tps, p0
    return start_tps, p0, mature_tps, mature_pop


def _parse_avg_tps(output):
    m = re.search(r"([\d,]+) ticks\s+[\d.]+s\s+([\d,]+) t/s", output)
    return int(m.group(2).replace(",", "")) if m else None


def _fmt(tps):
    return f"{tps} t/s" if tps else "—"


# ── run ───────────────────────────────────────────────────────────────────────

from sim.config import MAX_POP

print(f"\nbench_compute_units  —  {args.duration}s per run, seed 12345\n"
      f"  fresh world: --new    max pop: {MAX_POP} organisms\n")

rows = []
for label, compute_unit in OPTIONS:
    print(f"  {label}:")
    print(f"    fresh ({args.duration}s) ...", flush=True)
    out_fresh = _run_world(compute_unit, args.duration)
    load_str, load_s = _parse_load(out_fresh)
    start_tps, start_pop, mature_tps, mature_pop = _parse_intervals(out_fresh)

    print(f"    maxpop ({args.duration}s) ...", flush=True)
    out_max = _run_maxpop(compute_unit, args.duration)
    max_tps = _parse_avg_tps(out_max)

    rows.append((label, load_str, load_s,
                 start_tps, start_pop,
                 mature_tps, mature_pop,
                 max_tps))

# ── table ─────────────────────────────────────────────────────────────────────

col_w   = max(len(r[0]) for r in rows) + 2
col_tps = 15

start_hdr  = f"start (pop~{rows[0][4]})"  if rows and rows[0][4] else "start"
mature_hdr = f"mature (pop~{rows[0][6]})" if rows and rows[0][6] else "mature"
max_hdr    = f"maxpop ({MAX_POP})"

sep = "─" * (col_w + 10 + col_tps * 3 + 8)
print()
print(sep)
print(f"{'backend':<{col_w}}  {'compile':>8}  {start_hdr:>{col_tps}}  {mature_hdr:>{col_tps}}  {max_hdr:>{col_tps}}")
print(sep)
for (label, load_str, load_s,
     start_tps, start_pop,
     mature_tps, mature_pop,
     max_tps) in sorted(rows, key=lambda r: -(r[5] or 0)):
    marker = " ✓" if label == "CPU_AND_GPU" else "  "
    print(f"{label + marker:<{col_w}}  {load_str:>8}  "
          f"{_fmt(start_tps):>{col_tps}}  "
          f"{_fmt(mature_tps):>{col_tps}}  "
          f"{_fmt(max_tps):>{col_tps}}")

print()
