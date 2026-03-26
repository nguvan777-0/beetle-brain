"""beetle-brain — one entry point.

with pygame:   uv run --with numpy --with pygame --with coremltools --with plotly python world.py [seconds]
without:       uv run --with numpy --with coremltools --with plotly python world.py [seconds]
"""
import sys

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
    from game.snapshot import save_snapshot

    DURATION     = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    REPORT_EVERY = 500

    init_ane()

    world   = new_world()
    rng     = np.random.default_rng(world['seed'])
    tick    = 0
    stats   = StatsCollector()
    extinct = False
    history = []

    def _on_exit():
        save_snapshot(world, tick, history, [])
        print("\n" + "─" * 60)
        import runpy
        _argv, sys.argv = sys.argv, ["parse_snapshot.py"]
        try:
            runpy.run_path("parse_snapshot.py")
        finally:
            sys.argv = _argv

    atexit.register(_on_exit)
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    print(f"seed {world['seed']}  —  {'tick':>8} {'pop':>5} {'gen':>5} {'age':>7} {'ate':>5} {'spd':>5} {'sz':>4}  elapsed")
    print("─" * 60)

    t0          = time.time()
    next_report = REPORT_EVERY
    next_sample = SAMPLE_EVERY

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
            N = len(pop['x'])
            print(f"{tick:8d} {N:5d} {int(pop['generation'].max()):5d} "
                  f"{int(pop['age'].max()):7d} {int(pop['eaten'].max()):5d} "
                  f"{pop['speed'].mean():5.2f} {pop['size'].mean():4.1f}  {elapsed:.1f}s")
            next_report += REPORT_EVERY

    elapsed = time.time() - t0
    print("─" * 60)
    print(f"{tick:,} ticks  {elapsed:.1f}s  {tick/elapsed:,.0f} t/s")

    stats.finalize(tick, elapsed, pop=pop if not extinct else None,
                   phylo_state=world['phylo'], extinct=extinct, seed=world.get('seed'))

    from report import generate
    generate(stats)
