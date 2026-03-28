"""Hydrothermal vents: seeded, concentrated food sources."""
import numpy as np
from sim.config import WIDTH, HEIGHT, WORLD_SEED, VENT_COUNT_MIN, VENT_COUNT_MAX, VENT_RADIUS, COASTLINE_X


def make_vents(seed=None):
    """Generate vent positions deterministically from seed (any string or None)."""
    from sim.seed import to_int, parse
    rng = np.random.default_rng(to_int(parse(seed)))
    n   = int(rng.integers(VENT_COUNT_MIN, VENT_COUNT_MAX + 1))
    m   = VENT_RADIUS
    max_x = COASTLINE_X  # allow vents right up to the beach
    return rng.uniform([m, m], [max_x, HEIGHT - m], size=(n, 2)).astype(np.float32)


_R_MIN = 2.0  # avoid singularity at centre


def spawn_near_vents(n, vents, rng):
    """Spawn n food items with 1/r² density — dense at centre, sparse at edge."""
    chosen = rng.integers(0, len(vents), size=n)
    angles = rng.uniform(0, 2 * np.pi, size=n)
    # log-uniform radius → density ∝ 1/r²
    radii  = _R_MIN * (VENT_RADIUS / _R_MIN) ** rng.uniform(0, 1, size=n)
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
            new_chunks.append(spawn_near_vents(short, v[None, :], rng))
    if new_chunks:
        new_f = np.vstack(new_chunks)
        food  = np.vstack([food, new_f]) if len(food) else new_f
    return food
