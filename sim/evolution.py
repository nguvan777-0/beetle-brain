"""Cloning with mutation — the only way new wights enter the world."""
import numpy as np
from sim.config import (
    WIDTH, HEIGHT, N_BODY, N_INPUTS, N_HIDDEN, N_OUTPUTS,
    MUTATION_RATE, MUTATION_SCALE, EPIGENETIC, ENERGY_CLONE,
)
from sim.population.genome import decode


def clone_batch(pop, idx, rng):
    """Return a new pop dict of children cloned (with mutation) from parent indices."""
    n = len(idx)

    noise_r  = rng.random((n, N_BODY)).astype(np.float32)
    noise_W1 = rng.random((n, N_INPUTS, N_HIDDEN)).astype(np.float32)
    noise_W2 = rng.random((n, N_HIDDEN, N_OUTPUTS)).astype(np.float32)

    mut_r  = (rng.standard_normal((n, N_BODY))             * MUTATION_SCALE).astype(np.float32)
    mut_W1 = (rng.standard_normal((n, N_INPUTS, N_HIDDEN)) * MUTATION_SCALE).astype(np.float32)
    mut_W2 = (rng.standard_normal((n, N_HIDDEN, N_OUTPUTS))* MUTATION_SCALE).astype(np.float32)

    W_body = pop['W_body'][idx] + np.where(noise_r  < MUTATION_RATE, mut_r,  0)
    W1     = pop['W1'][idx]     + np.where(noise_W1 < MUTATION_RATE, mut_W1, 0)
    W2     = pop['W2'][idx]     + np.where(noise_W2 < MUTATION_RATE, mut_W2, 0)

    speed, fov, ray, size, drain, turn_s, r, g, b = decode(W_body)
    ang = pop['angle'][idx] + np.pi + rng.uniform(-0.5, 0.5, n).astype(np.float32)

    return {
        'x':          (pop['x'][idx] + np.cos(ang) * (pop['size'][idx] * 2 + 2)) % WIDTH,
        'y':          (pop['y'][idx] + np.sin(ang) * (pop['size'][idx] * 2 + 2)) % HEIGHT,
        'angle':      ang,
        'energy':     np.full(n, ENERGY_CLONE, dtype=np.float32),
        'W_body':     W_body, 'W1': W1, 'W2': W2,
        'speed':      speed.astype(np.float32),
        'fov':        fov.astype(np.float32),
        'ray_len':    ray.astype(np.float32),
        'size':       size.astype(np.float32),
        'drain':      drain.astype(np.float32),
        'turn_s':     turn_s.astype(np.float32),
        'r': r, 'g': g, 'b': b,
        'generation': (pop['generation'][idx] + 1).astype(np.int32),
        'age':        np.zeros(n, dtype=np.int32),
        'eaten':      np.zeros(n, dtype=np.int32),
        'h_state':    pop['h_state'][idx] * EPIGENETIC,
    }
