"""Hydrothermal vents: seeded, concentrated food sources."""
import numpy as np
from sim.config import WIDTH, HEIGHT, WORLD_SEED, VENT_COUNT_MIN, VENT_COUNT_MAX, VENT_RADIUS


def make_vents(seed=None):
    """Generate vent positions deterministically from seed.
    Vents are kept VENT_RADIUS away from all edges so food never wraps."""
    rng = np.random.default_rng(WORLD_SEED if seed is None else seed)
    n   = int(rng.integers(VENT_COUNT_MIN, VENT_COUNT_MAX + 1))
    m   = VENT_RADIUS
    return rng.uniform([m, m], [WIDTH - m, HEIGHT - m], size=(n, 2)).astype(np.float32)


def spawn_near_vents(n, vents, rng):
    """Spawn n food items distributed uniformly within vent radii."""
    chosen = rng.integers(0, len(vents), size=n)
    angles = rng.uniform(0, 2 * np.pi, size=n)
    radii  = np.sqrt(rng.uniform(0, 1, size=n)) * VENT_RADIUS
    x = (vents[chosen, 0] + np.cos(angles) * radii) % WIDTH
    y = (vents[chosen, 1] + np.sin(angles) * radii) % HEIGHT
    return np.stack([x, y], axis=1).astype(np.float32)


def refill_vents(food, vents, rng, per_vent_cap):
    """Top up each vent independently to per_vent_cap food items."""
    new_chunks = []
    for v in vents:
        if len(food):
            dx   = food[:, 0] - v[0]
            dy   = food[:, 1] - v[1]
            near = int((dx * dx + dy * dy < VENT_RADIUS * VENT_RADIUS).sum())
        else:
            near = 0
        short = per_vent_cap - near
        if short > 0:
            angles = rng.uniform(0, 2 * np.pi, size=short)
            radii  = np.sqrt(rng.uniform(0, 1, size=short)) * VENT_RADIUS
            xs = (v[0] + np.cos(angles) * radii) % WIDTH
            ys = (v[1] + np.sin(angles) * radii) % HEIGHT
            new_chunks.append(np.stack([xs, ys], axis=1).astype(np.float32))
    if new_chunks:
        new_f = np.vstack(new_chunks)
        food  = np.vstack([food, new_f]) if len(food) else new_f
    return food
