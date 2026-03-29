"""Headless tick-rate benchmark.

Usage:
    uv run --with coremltools python scripts/bench.py [--duration 10]
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--backend",  default="CPU_AND_GPU")
args = parser.parse_args()

if args.backend != "numpy":
    os.environ["BEETLE_COMPUTE_UNITS"] = args.backend

from sim import new_world, tick as sim_tick, init_ane
from sim.seed import random_name

print(f"backend: {args.backend}  duration: {args.duration}s")
init_ane()

world = new_world(seed=random_name())
rng   = np.random.default_rng(42)

# warm-up: 50 ticks (model JIT, caches)
for _ in range(50):
    world = sim_tick(world, rng)

ticks = 0
t0    = time.perf_counter()
deadline = t0 + args.duration

while time.perf_counter() < deadline:
    world = sim_tick(world, rng)
    ticks += 1
    if len(world["pop"]["x"]) == 0:
        world = new_world(seed=random_name())

elapsed = time.perf_counter() - t0
pop_n   = len(world["pop"]["x"])
print(f"ticks: {ticks:,}  elapsed: {elapsed:.2f}s  tps: {ticks/elapsed:.1f}  final_pop: {pop_n}")
