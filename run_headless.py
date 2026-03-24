"""
run_headless.py — run beetle-brain at max speed, print analysis, generate report
    uv run --with coremltools --with numpy --with plotly python run_headless.py [seconds]
"""
import sys
import time
import numpy as np

import sim
from sim import new_world, tick as sim_tick, init_ane
from sim.stats import StatsCollector, SAMPLE_EVERY

DURATION     = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
REPORT_EVERY = 500

init_ane()

rng    = np.random.default_rng()
world  = new_world(rng)
tick   = 0
stats  = StatsCollector()
extinct = False

print(f"{'tick':>8} {'pop':>5} {'maxGen':>7} {'maxAge':>7} {'maxAte':>7} "
      f"{'avgSpd':>7} {'avgSz':>6} {'avgDrn':>7}  elapsed")
print("─" * 80)

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
        print(f"EXTINCTION at tick {tick:,}  ({elapsed:.1f}s)")
        extinct = True
        break

    if tick >= next_sample:
        stats.record(tick, pop)
        next_sample += SAMPLE_EVERY

    if tick >= next_report:
        N = len(pop['x'])
        print(f"{tick:8d} {N:5d} {int(pop['generation'].max()):7d} "
              f"{int(pop['age'].max()):7d} {int(pop['eaten'].max()):7d} "
              f"{pop['speed'].mean():7.2f} {pop['size'].mean():6.1f} "
              f"{(0.015 * pop['size']**0.75).mean():7.3f}  {elapsed:.1f}s")
        next_report += REPORT_EVERY

elapsed = time.time() - t0
print("─" * 80)
print(f"done: {tick:,} ticks in {elapsed:.1f}s  ({tick/elapsed:,.0f} ticks/sec)")

if not extinct:
    print(f"\nFINAL pop={len(pop['x'])}  maxGen={pop['generation'].max()}  "
          f"maxAge={pop['age'].max()}  maxAte={pop['eaten'].max()}")
    print(f"  avg speed={pop['speed'].mean():.2f}  size={pop['size'].mean():.1f}  "
          f"drain={(0.015 * pop['size']**0.75).mean():.3f}  fov={np.degrees(pop['fov'].mean()):.0f}°")

stats.finalize(tick, elapsed, pop=pop if not extinct else None, extinct=extinct)

from report import generate
generate(stats)
