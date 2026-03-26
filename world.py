"""beetle-brain — one entry point.

with pygame:   uv run --with numpy --with pygame --with coremltools --with plotly python world.py [seconds]
without:       uv run --with numpy --with coremltools --with plotly python world.py [seconds]
"""
import sys

if sys.version_info < (3, 11):
    print()
    print(f"  beetle-brain requires Python 3.11+  (you're on {sys.version.split()[0]})")
    print()
    print("  with uv:")
    print("    uv run --with numpy --with pygame --with coremltools --with plotly python world.py")
    print()
    print("  or switch to 3.11+ via pyenv, conda, or your package manager.")
    print()
    sys.exit(1)

try:
    import numpy  # noqa: F401
except ImportError:
    print()
    print("  missing dependencies — numpy is required, pygame/coremltools/plotly are optional.")
    print()
    print("  with uv:")
    print("    uv run --with numpy --with pygame --with coremltools --with plotly python world.py")
    print()
    print("  or install numpy into your active environment.")
    print()
    sys.exit(1)

try:
    import pygame
    _has_pygame = True
except ImportError:
    _has_pygame = False

if _has_pygame:
    from game.main import main
    main()
else:
    import atexit
    import signal
    import time
    import numpy as np
    from sim import new_world, tick as sim_tick, init_ane
    from sim.stats import StatsCollector, SAMPLE_EVERY
    from game.snapshot import save_snapshot, load_snapshot

    import argparse
    parser = argparse.ArgumentParser(
        prog='world.py',
        description='beetle-brain headless sim',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('duration', nargs='?', type=float, default=30.0,
                        help='run duration in seconds (default: 30)')
    parser.add_argument('--seed',  type=int, metavar='X',
                        help='start a new world with seed X  (ignores snapshot)')
    parser.add_argument('--new',   action='store_true',
                        help='start a new world with a random seed  (ignores snapshot)')
    parser.add_argument('--fork',  type=int, metavar='X',
                        help='load snapshot but run forward with RNG seed X')
    args = parser.parse_args()

    DURATION     = args.duration
    REPORT_EVERY = 500

    init_ane()

    force_new = args.new or args.seed is not None

    # load snapshot unless --new or --seed
    world, tick, history, _hf, stats = load_snapshot(np.random.default_rng())
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
        from report import generate, _report_stem
        generate(stats, world=world if not extinct else None, tick=tick)
        txt_path = _report_stem(stats) + ".txt"
        if Path(txt_path).exists():
            print("\n" + Path(txt_path).read_text())

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
        if elapsed >= DURATION:
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
