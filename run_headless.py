"""
run_headless.py — run beetle-brain at max speed, print analysis
    uv run --with coremltools --with numpy --with emoji python run_headless.py [seconds]
"""
import sys
import time
import numpy as np

import sim
from sim import new_world, tick as sim_tick, init_ane

DURATION     = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
REPORT_EVERY = 500

init_ane()

rng            = np.random.default_rng()
pop, food, vents = new_world(rng)
tick           = 0

print(f"{'tick':>8} {'pop':>5} {'maxGen':>7} {'maxAge':>7} {'maxAte':>7} "
      f"{'avgSpd':>7} {'avgSz':>6} {'avgDrn':>7}  elapsed")
print("─" * 80)

t0          = time.time()
next_report = REPORT_EVERY

while True:
    elapsed = time.time() - t0
    if elapsed >= DURATION:
        break

    pop, food = sim_tick(pop, food, vents, rng)
    tick += 1

    if len(pop['x']) == 0:
        print(f"EXTINCTION at tick {tick:,}  ({elapsed:.1f}s)")
        break

    if tick >= next_report:
        N = len(pop['x'])
        print(f"{tick:8d} {N:5d} {int(pop['generation'].max()):7d} "
              f"{int(pop['age'].max()):7d} {int(pop['eaten'].max()):7d} "
              f"{pop['speed'].mean():7.2f} {pop['size'].mean():6.1f} "
              f"{pop['drain'].mean():7.3f}  {elapsed:.1f}s")
        next_report += REPORT_EVERY

elapsed = time.time() - t0
print("─" * 80)
print(f"done: {tick:,} ticks in {elapsed:.1f}s  ({tick/elapsed:,.0f} ticks/sec)")

if len(pop['x']) > 0:
    print(f"\nFINAL pop={len(pop['x'])}  maxGen={pop['generation'].max()}  "
          f"maxAge={pop['age'].max()}  maxAte={pop['eaten'].max()}")
    print(f"  avg speed={pop['speed'].mean():.2f}  size={pop['size'].mean():.1f}  "
          f"drain={pop['drain'].mean():.3f}  fov={np.degrees(pop['fov'].mean()):.0f}°")
