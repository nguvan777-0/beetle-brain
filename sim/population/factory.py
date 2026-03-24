"""Create new populations and worlds."""
import numpy as np
from sim.config import N_FOOD, N_START, N_INPUTS, N_HIDDEN, N_OUTPUTS, ENERGY_START, WIDTH, HEIGHT
from sim.population.genome import decode, N_BODY
from sim.vents import make_vents, spawn_near_vents, refill_vents
from sim import phylo


def make_pop(n, rng, phylo_state):
    W_body = rng.standard_normal((n, N_BODY)).astype(np.float32)
    W1     = (rng.standard_normal((n, N_INPUTS, N_HIDDEN)) * 0.8).astype(np.float32)
    W2     = (rng.standard_normal((n, N_HIDDEN, N_OUTPUTS)) * 0.8).astype(np.float32)
    t = decode(W_body)
    return {
        'x':             rng.uniform(0, WIDTH,  n).astype(np.float32),
        'y':             rng.uniform(0, HEIGHT, n).astype(np.float32),
        'angle':         rng.uniform(0, 2 * np.pi, n).astype(np.float32),
        'energy':        np.full(n, ENERGY_START, dtype=np.float32),
        'W_body': W_body, 'W1': W1, 'W2': W2,
        **t,
        'generation':    np.zeros(n, dtype=np.int32),
        'age':           np.zeros(n, dtype=np.int32),
        'eaten':         np.zeros(n, dtype=np.int32),
        'h_state':       np.zeros((n, N_HIDDEN), dtype=np.float32),
        'lineage_id':    np.arange(n, dtype=np.int32),
        'individual_id': np.arange(n, dtype=np.int32),   # founders are 0..n-1
    }


def new_world(rng=None, world_seed=None):
    if rng is None:
        rng = np.random.default_rng()
    vents       = make_vents(world_seed)
    phylo_state = phylo.new_state(N_START)
    pop         = make_pop(N_START, rng, phylo_state)
    food        = refill_vents(np.empty((0, 2), dtype=np.float32), vents, rng, N_FOOD // len(vents))
    return {'pop': pop, 'food': food, 'vents': vents, 'phylo': phylo_state}
