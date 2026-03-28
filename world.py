"""beetle-brain — one entry point."""
import sys

if sys.version_info < (3, 11):
    print()
    print(f"  beetle-brain requires Python 3.11+  (you're on {sys.version.split()[0]})")
    print()
    print("  with uv:")
    print("    uv run --with coremltools --with pygame --with plotly python world.py")
    print()
    print("  or switch to 3.11+ via pyenv, conda, or your package manager.")
    print()
    sys.exit(1)

import argparse

_MISSING          = object()
_BACKEND_CHOICES  = 'gpu, ane, cpu, all, numpy'

def _float(s):
    try: return float(s)
    except ValueError: raise argparse.ArgumentTypeError(f"invalid value '{s}'  —  expected a number")

def _int(s):
    try: return int(s)
    except ValueError: raise argparse.ArgumentTypeError(f"invalid value '{s}'  —  expected a number")

class _Parser(argparse.ArgumentParser):
    def error(self, message):
        super().error(message.replace('argument --', '--', 1))

parser = _Parser(
    prog='world.py',
    usage=argparse.SUPPRESS,
    description=(
        'beetle-brain  —  evolution on weights\n'
        '\n'
        'Usage:\n'
        '  uv run --with coremltools --with pygame python world.py\n'
        '  uv run --with coremltools python world.py --new\n'
        '\n'
        'Libraries:\n'
        '  numpy          required — or use coremltools which includes it\n'
        '  coremltools    CoreML acceleration on Apple Silicon (includes numpy)\n'
        '  pygame         visual display\n'
        '  plotly         HTML report generation\n'
        '\n'
        'Runs forever by default. Omit pygame to run headless. Snapshot and reports written on exit.'
    ),
    formatter_class=argparse.RawTextHelpFormatter,
    add_help=False,
)
parser.add_argument('-h', '--help', action='help', help='show this help message and exit')
parser.add_argument('--duration', type=_float, nargs='?', const=_MISSING, default=None, metavar='N',
                    help='stop after N seconds  (default: run forever)')
parser.add_argument('--backend', nargs='?', const=_MISSING, default='CPU_AND_GPU', metavar='BACKEND',
                    help='backend  (default: gpu)\n'
                         '  gpu, ane, cpu  — CoreML with that compute unit\n'
                         '  all            — CoreML with ANE + GPU + CPU together\n'
                         '  numpy          — no CoreML')
parser.add_argument('--seed',  type=_int, nargs='?', const=_MISSING, default=None, metavar='N',
                    help='start fresh with seed N  (ignores snapshot)')
parser.add_argument('--new',   action='store_true',
                    help='start fresh with a random seed  (ignores snapshot)')
parser.add_argument('--fork',  type=_int, nargs='?', const=_MISSING, default=None, metavar='N',
                    help='load snapshot, run forward with a different RNG seed')
parser.add_argument('--no-report', action='store_true',
                    help='skip printing the report on exit')
parser._optionals.title = 'Options'

try:
    import numpy  # noqa: F401
except ImportError:
    parser.error("numpy is required — use --with numpy, or --with coremltools (which includes numpy)")

args = parser.parse_args()

if args.duration is _MISSING:
    parser.error('--duration requires a value  —  expected a number')
if args.seed is _MISSING:
    parser.error('--seed requires a value  —  expected a number')
if args.fork is _MISSING:
    parser.error('--fork requires a value  —  expected a number')
if args.backend is _MISSING:
    parser.error(f'--backend requires a value  —  choices: {_BACKEND_CHOICES}')

_BACKENDS = {
    'gpu':         'CPU_AND_GPU',
    'cpu_and_gpu': 'CPU_AND_GPU',
    'ane':         'CPU_AND_NE',
    'cpu_and_ne':  'CPU_AND_NE',
    'cpu':         'CPU_ONLY',
    'cpu_only':    'CPU_ONLY',
    'all':         'ALL',
    'numpy':       'numpy',
}
_backend = _BACKENDS.get(args.backend.lower().replace('-', '_'))
if _backend is None:
    parser.error(f"unknown backend {args.backend!r}  —  choices: {_BACKEND_CHOICES}")
args.backend = _backend

try:
    import pygame
    _has_pygame = True
except ImportError:
    _has_pygame = False

if _has_pygame:
    from game.main import main
    main(new=args.new, seed=args.seed, fork=args.fork, compute_units=args.backend)
else:
    import atexit
    import signal
    import time
    import numpy as np
    from sim import new_world, tick as sim_tick, init_ane
    from sim.stats import StatsCollector, SAMPLE_EVERY
    from game.snapshot import save_snapshot, load_snapshot

    DURATION     = args.duration  # None means run forever
    REPORT_EVERY = 500

    import os
    if args.backend != 'numpy':
        os.environ['BEETLE_COMPUTE_UNITS'] = args.backend
    init_ane()

    force_new = (args.new or args.seed is not None) and args.fork is None

    # load snapshot unless --new or --seed
    world, tick, history, _hf, stats = load_snapshot(np.random.default_rng(args.fork))
    if force_new or world is None or len(world['pop']['x']) == 0:
        seed  = args.seed if args.seed is not None else None
        world = new_world(seed=seed)
        tick    = 0
        history = []
        stats   = StatsCollector()

    from sim.config import SIZE_MIN, SIZE_MAX, N_RAYS, N_HIDDEN

    rng_seed = args.fork if args.fork is not None else world['seed']
    rng      = np.random.default_rng(rng_seed)
    extinct     = False
    next_sample = tick + SAMPLE_EVERY
    next_report = tick + REPORT_EVERY

    t0          = time.time()

    # streaming analytics state
    _prev = {'sz': None, 'rays': None, 'neur': None, 'pop': None}
    _flags = set()  # one-shot events already printed

    def _on_exit():
        elapsed = time.time() - t0
        if not extinct:
            save_snapshot(world, tick, history, [], stats)
        if not stats.run_meta:
            stats.finalize(tick, elapsed,
                           pop=world['pop'] if not extinct else None,
                           phylo_state=world['phylo'], extinct=extinct,
                           seed=world.get('seed'))
        from pathlib import Path
        from reports.report import generate, generate_text, generate_summary, _report_stem
        generate(stats, world=world if not extinct else None, tick=tick)
        txt_path = _report_stem(stats) + ".txt"
        if not Path(txt_path).exists():
            generate_text(stats, txt_path, world=world if not extinct else None, tick=tick)
        if not args.no_report:
            generate_summary(stats, world=world if not extinct else None, tick=tick)

    atexit.register(_on_exit)
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    resume = f"  resuming tick {tick:,}" if tick > 0 else ""
    fork   = f"  fork rng {args.fork}" if args.fork is not None else ""
    print(f"seed {world['seed']}{resume}{fork}")
    print("─" * 72)
    print(f"{'tick':>8} {'pop':>5} {'gen':>5} {'age':>7} {'ate':>5} {'spd':>5} {'sz':>4} {'rays':>5} {'neur':>5}  elapsed")
    print("─" * 72)

    while True:
        elapsed = time.time() - t0
        if DURATION is not None and elapsed >= DURATION:
            break

        world = sim_tick(world, rng)
        pop   = world['pop']
        tick += 1

        if len(pop['x']) == 0:
            print(f"extinction at tick {tick:,}  ({elapsed:.1f}s)")
            extinct = True
            break

        if tick >= next_sample:
            stats.record(tick, pop, world['phylo'])
            next_sample += SAMPLE_EVERY

        if tick % 30 == 0 and len(pop['x']) > 0:
            history.append((
                float(tick), float(len(pop['x'])),
                float(pop['generation'].max()),
                float(pop['speed'].mean()), float(pop['fov'].mean()),
                float(pop['size'].mean()), float(pop['mutation_rate'].mean()),
            ))

        if tick >= next_report:
            N    = len(pop['x'])
            sz   = float(pop['size'].mean())
            rays = float(pop['n_rays'].mean()) if 'n_rays' in pop else 0.0
            neur = float(pop.get('active_neurons', np.zeros(N)).mean())
            print(f"{tick:8d} {N:5d} {int(pop['generation'].max()):5d} "
                  f"{int(pop['age'].max()):7d} {int(pop['eaten'].max()):5d} "
                  f"{pop['speed'].mean():5.2f} {sz:4.1f} {rays:5.1f} {neur:5.1f}  {elapsed:.1f}s")

            # streaming analytics — one-shot flags
            notes = []
            sz_pct = (sz - SIZE_MIN) / (SIZE_MAX - SIZE_MIN)
            if sz_pct >= 0.9 and 'sz90' not in _flags:
                notes.append("size locked — 90% of range")
                _flags.add('sz90')
            elif sz_pct >= 0.8 and 'sz80' not in _flags:
                notes.append("size crossing 80% of range")
                _flags.add('sz80')

            if rays < 1.0 and 'rays0' not in _flags and _prev['rays'] is not None:
                notes.append("vision collapsed — mean n_rays < 1")
                _flags.add('rays0')
            elif _prev['rays'] is not None and rays < _prev['rays'] * 0.7 and 'rays_drop' not in _flags:
                notes.append(f"vision dropping fast  {_prev['rays']:.1f} → {rays:.1f}")
                _flags.add('rays_drop')

            if _prev['neur'] is not None:
                d = neur - _prev['neur']
                if d > 2.0 and 'neur_up' not in _flags:
                    notes.append(f"brain growing  +{d:.1f} neurons")
                    _flags.add('neur_up')
                elif d < -2.0 and 'neur_dn' not in _flags:
                    notes.append(f"brain pruning  {d:.1f} neurons")
                    _flags.add('neur_dn')

            if _prev['pop'] is not None and N < _prev['pop'] * 0.5:
                notes.append(f"population crash  {_prev['pop']} → {N}")
            elif _prev['pop'] is not None and N > _prev['pop'] * 2.0:
                notes.append(f"population boom  {_prev['pop']} → {N}")

            for note in notes:
                print(f"          ↳ {note}")

            _prev['sz']   = sz
            _prev['rays'] = rays
            _prev['neur'] = neur
            _prev['pop']  = N
            next_report  += REPORT_EVERY

    elapsed = time.time() - t0
    print("─" * 72)
    print(f"{tick:,} ticks  {elapsed:.1f}s  {tick/elapsed:,.0f} t/s")
