"""Create new populations and worlds."""
import numpy as np
from sim.config import WIDTH, HEIGHT, N_FOOD, N_START, N_INPUTS, N_HIDDEN, N_OUTPUTS, ENERGY_START
from sim.population.genome import decode, N_BODY


def make_pop(n, rng):
    W_body = rng.standard_normal((n, N_BODY)).astype(np.float32)
    W1     = (rng.standard_normal((n, N_INPUTS, N_HIDDEN)) * 0.8).astype(np.float32)
    W2     = (rng.standard_normal((n, N_HIDDEN, N_OUTPUTS)) * 0.8).astype(np.float32)
    speed, fov, ray, size, drain, turn_s, r, g, b, breed_at, clone_with, mut_rate, mut_scale, epigenetic, weight_decay = decode(W_body)
    return {
        'x':             rng.uniform(0, WIDTH,  n).astype(np.float32),
        'y':             rng.uniform(0, HEIGHT, n).astype(np.float32),
        'angle':         rng.uniform(0, 2 * np.pi, n).astype(np.float32),
        'energy':        np.full(n, ENERGY_START, dtype=np.float32),
        'W_body':        W_body, 'W1': W1, 'W2': W2,
        'speed':         speed.astype(np.float32),
        'fov':           fov.astype(np.float32),
        'ray_len':       ray.astype(np.float32),
        'size':          size.astype(np.float32),
        'drain':         drain.astype(np.float32),
        'turn_s':        turn_s.astype(np.float32),
        'breed_at':      breed_at.astype(np.float32),
        'clone_with':    clone_with.astype(np.float32),
        'mutation_rate':  mut_rate.astype(np.float32),
        'mutation_scale': mut_scale.astype(np.float32),
        'epigenetic':     epigenetic.astype(np.float32),
        'weight_decay':   weight_decay.astype(np.float32),
        'r': r, 'g': g, 'b': b,
        'generation': np.zeros(n, dtype=np.int32),
        'age':        np.zeros(n, dtype=np.int32),
        'eaten':      np.zeros(n, dtype=np.int32),
        'h_state':    np.zeros((n, N_HIDDEN), dtype=np.float32),
    }


def new_world(rng=None):
    if rng is None:
        rng = np.random.default_rng()
    pop  = make_pop(N_START, rng)
    food = rng.uniform(0, [WIDTH, HEIGHT], size=(N_FOOD, 2)).astype(np.float32)
    return pop, food
