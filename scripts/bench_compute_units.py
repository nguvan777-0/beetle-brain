"""
Tick-rate benchmark across backends: CoreML compute units and numpy.

Three measurements per backend:
  world start  — first 500-tick interval from a new world (pop ~12)
  grown        — last 500-tick interval of the same run (pop has grown)
  full         — MAX_POP world, positions randomised, timed separately

Build cache is cleared before each backend so compile times are real.
Calls itself as a subprocess for the full-pop phase (--worker flag, internal).
"""
import argparse
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── worker mode (subprocess entry point for full-pop phase) ───────────────────

if "--worker" in sys.argv:
    import time
    import numpy as np
    from sim.config import MAX_POP, WIDTH, HEIGHT
    from sim import new_world, init_ane
    from sim.tick import tick as sim_tick
    from game.snapshot import load_snapshot

    wi       = sys.argv.index("--worker")
    duration = float(sys.argv[wi + 1])
    seed     = int(sys.argv[wi + 2])

    rng = np.random.default_rng(seed)
    world, _, _, _, _ = load_snapshot(rng)
    if world is None or len(world['pop']['x']) == 0:
        world = new_world(seed=seed)

    pop = world['pop']
    n   = len(pop['x'])
    if n < MAX_POP:
        reps    = (MAX_POP + n - 1) // n
        new_pop = {}
        for k, v in pop.items():
            tiled      = np.tile(v, (reps,) + (1,) * (v.ndim - 1))
            new_pop[k] = tiled[:MAX_POP]
        tile_rng         = np.random.default_rng(0)
        new_pop['x']     = tile_rng.uniform(0, WIDTH,      MAX_POP).astype(np.float32)
        new_pop['y']     = tile_rng.uniform(0, HEIGHT,     MAX_POP).astype(np.float32)
        new_pop['angle'] = tile_rng.uniform(0, 2 * np.pi, MAX_POP).astype(np.float32)
        world['pop']     = new_pop
        print(f"[bench] tiled {n} → {MAX_POP}, positions randomised")

    init_ane()

    ticks = 0
    t0    = time.time()
    try:
        while time.time() - t0 < duration:
            world  = sim_tick(world, rng)
            ticks += 1
    except Exception as e:
        print(f"[bench] failed after {ticks} ticks: {e}")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"{ticks:,} ticks  {elapsed:.1f}s  {int(ticks / elapsed):,} t/s")
    sys.exit(0)


# ── orchestrator ──────────────────────────────────────────────────────────────

try:
    import coremltools  # noqa: F401
    _HAS_CT = True
except ImportError:
    _HAS_CT = False

OPTIONS = [
    ("ane",    "CPU_AND_NE"),
    ("gpu",    "CPU_AND_GPU"),
    ("cpu",    "CPU_ONLY"),
    ("all",    "ALL"),
    ("numpy",  None),
]
if not _HAS_CT:
    OPTIONS = [o for o in OPTIONS if o[1] is None]

_MISSING = object()

def _int(s):
    try: return int(s)
    except ValueError: raise argparse.ArgumentTypeError(f"invalid value '{s}'  —  expected a number")

class _Parser(argparse.ArgumentParser):
    def error(self, message):
        super().error(message.replace('argument --', '--', 1))

parser = _Parser(
    prog='bench_compute_units.py',
    usage=argparse.SUPPRESS,
    description=(
        'bench_compute_units  —  tick-rate benchmark across backends\n'
        '\n'
        'Two phases per backend, each timed for --duration seconds:\n'
        '  world run  — fresh world, small starting pop growing over time;\n'
        '               reports t/s at start and at the end of the run\n'
        '  full pop   — MAX_POP wights, positions randomised; reports t/s\n'
        '\n'
        'Usage:\n'
        '  uv run --with coremltools python scripts/bench_compute_units.py\n'
        '  uv run --with numpy python scripts/bench_compute_units.py --duration 5\n'
        '\n'
        'Libraries:\n'
        '  numpy          required — or use coremltools which includes it\n'
        '  coremltools    enables CoreML backends (includes numpy)'
    ),
    formatter_class=argparse.RawTextHelpFormatter,
    add_help=False,
)
parser.add_argument('-h', '--help', action='help', help='show this help message and exit')
parser.add_argument('--duration', type=_int, nargs='?', const=_MISSING, default=60, metavar='N',
                    help='seconds per phase per backend  (default: 60)')
parser.add_argument('--backend', nargs='?', const=_MISSING, default='gpu' if _HAS_CT else 'numpy', metavar='BACKEND',
                    help='backend(s) to bench  (default: gpu)\n'
                         '  ane, gpu, cpu  — CoreML with that compute unit\n'
                         '  all            — CoreML with ANE + GPU + CPU together\n'
                         '  numpy          — no CoreML\n'
                         '  every          — each of the above\n'
                         '  ane,gpu        — comma-separate for a subset')
parser.add_argument('--seed', type=_int, nargs='?', const=_MISSING, default=None, metavar='N',
                    help='random seed  (default: random)')
parser._optionals.title = 'Options'

THIS = str(Path(__file__).resolve())

try:
    from sim.config import MAX_POP
except ModuleNotFoundError:
    parser.error("numpy is required — use --with numpy, or --with coremltools (which includes numpy)")

args = parser.parse_args()

if args.duration is _MISSING:
    parser.error('--duration requires a value  —  expected a number')
if args.seed is _MISSING:
    parser.error('--seed requires a value  —  expected a number')

_ALL_LABELS = [label for label, _ in OPTIONS]
_CHOICES    = _ALL_LABELS + ['every']
if args.backend is _MISSING:
    parser.error(f"--backend requires a value  —  choices: {', '.join(_CHOICES)}")

args.seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)

if args.backend.lower() != 'every':
    _requested = [b.strip().lower() for b in args.backend.split(',')]
    _unknown   = [b for b in _requested if b not in _ALL_LABELS]
    if _unknown:
        parser.error(f"unknown backend: {', '.join(_unknown)}  —  choices: {', '.join(_CHOICES)}")
    OPTIONS = [o for o in OPTIONS if o[0] in _requested]

col_w   = max(len(label) for label, _ in OPTIONS) + 2
col_tps = 16
sep_w   = col_w + 10 + col_tps * 3 + 8


def _clear_cache():
    build = ROOT / "build"
    if build.exists():
        shutil.rmtree(build)


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


def _stream(cmd, env, prefix):
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    buf = []
    for line in proc.stdout:
        line = line.rstrip()
        buf.append(line)
        print(f"  {prefix}  {line}", flush=True)
    proc.wait()
    return "\n".join(buf)


def _parse_compile(output):
    def _f(pat):
        m = re.search(pat, output)
        return float(m.group(1)) if m else None
    vals = [x for x in [
        _f(r"\[Brain\] ready \(([0-9.]+)s\)"),
        _f(r"\[Brain\].*done \(([0-9.]+)s\)"),
        _f(r"\[SenseBrain\] ready \(([0-9.]+)s\)"),
        _f(r"\[SenseBrain\] all done \(([0-9.]+)s\)"),
    ] if x is not None]
    if not vals:
        return "n/a", 0.0
    t = max(vals)
    return f"~{t:.1f}s", t


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
        grown_tps  = int((t2 - t1) / dt) if dt > 0 else None
        grown_pop  = p2
    else:
        grown_tps, grown_pop = start_tps, p0
    return start_tps, p0, grown_tps, grown_pop


def _parse_full_tps(output):
    if "[bench] failed" in output:
        return "failed"
    m = re.search(r"([\d,]+) ticks\s+[\d.]+s\s+([\d,]+) t/s", output)
    return int(m.group(2).replace(",", "")) if m else None


def _fmt(tps):
    if tps == 'failed':
        return "failed"
    return f"{tps:,} t/s" if tps else "n/a"


def _print_header(ws_hdr, gr_hdr):
    sep = "─" * sep_w
    print(sep)
    print(f"{'backend':<{col_w}}  {'compile':>8}  {ws_hdr:>{col_tps}}  {gr_hdr:>{col_tps}}  {f'full ({MAX_POP})':>{col_tps}}")
    print(sep)


def _print_row(label, compile_str, start_tps, grown_tps, full_tps, fastest=False):
    marker = " ✓" if fastest else "  "
    print(f"{label + marker:<{col_w}}  {compile_str:>8}  "
          f"{_fmt(start_tps):>{col_tps}}  "
          f"{_fmt(grown_tps):>{col_tps}}  "
          f"{_fmt(full_tps):>{col_tps}}", flush=True)


# ── run ───────────────────────────────────────────────────────────────────────

print(f"\nbench_compute_units  —  {args.duration}s per phase, seed {args.seed}"
      f"  |  max pop {MAX_POP}  |  cache cleared before each backend\n")

rows        = []
ws_hdr      = "world start"
gr_hdr      = "grown"
hdr_printed = False

for label, compute_unit in OPTIONS:
    print(f"── {label}  {'─' * (sep_w - len(label) - 5)}", flush=True)

    _clear_cache()
    print(f"  {label}  world run ({args.duration}s):", flush=True)
    out_world = _stream(
        ["uv", "run", *_deps(compute_unit),
         "python", "-u", "world.py", "--duration", str(args.duration), "--new", "--seed", str(args.seed), "--no-report",
         "--backend", label if compute_unit is not None else "numpy"],
        _env(compute_unit), "│",
    )
    compile_str, _ = _parse_compile(out_world)
    start_tps, start_pop, grown_tps, grown_pop = _parse_intervals(out_world)

    _clear_cache()
    print(f"  {label}  full pop ({args.duration}s):", flush=True)
    out_full = _stream(
        ["uv", "run", *_deps(compute_unit),
         "python", "-u", THIS, "--worker", str(args.duration), str(args.seed)],
        _env(compute_unit), "│",
    )
    full_tps = _parse_full_tps(out_full)

    row = (label, compile_str, start_tps, start_pop, grown_tps, grown_pop, full_tps)
    rows.append(row)

    if start_pop:
        ws_hdr = f"world start (~{start_pop})"
    if grown_pop:
        gr_hdr = f"grown (~{grown_pop})"

    if not hdr_printed:
        print()
        _print_header(ws_hdr, gr_hdr)
        hdr_printed = True

    _print_row(label, compile_str, start_tps, grown_tps, full_tps)
    if full_tps == 'failed':
        print(f"    ↳ ANE on-chip memory limit — {MAX_POP}-batch model compiles but fails at inference")
    print(flush=True)

# ── final sorted table (only when multiple backends were tested) ──────────────

ane_failed = any(label == 'ane' and full_tps == 'failed' for label, _, _, _, _, _, full_tps in rows)

if len(rows) > 1:
    print(f"── results  {'─' * (sep_w - 12)}\n")
    _print_header(ws_hdr, gr_hdr)
    sorted_rows = sorted(rows, key=lambda r: -(r[4] if isinstance(r[4], int) else 0))
    for i, (label, compile_str, start_tps, start_pop,
            grown_tps, grown_pop, full_tps) in enumerate(sorted_rows):
        _print_row(label, compile_str, start_tps, grown_tps, full_tps, fastest=(i == 0))
if ane_failed:
    print(f"\n  † ANE full ({MAX_POP}): model compiles but exceeds ANE on-chip memory at inference.")
print()
