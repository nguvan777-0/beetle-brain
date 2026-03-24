"""Cloning with mutation — the only way new wights enter the world."""
import numpy as np
from sim.config import WIDTH, HEIGHT, N_INPUTS, N_HIDDEN, N_OUTPUTS
from sim.population.genome import decode, N_BODY


def clone_batch(pop, idx, rng):
    """Return a new pop dict of children cloned (with mutation) from parent indices."""
    n = len(idx)

    mut_rate  = pop['mutation_rate'][idx, None]   # (n, 1)
    mut_scale = pop['mutation_scale'][idx, None]

    noise_r  = rng.random((n, N_BODY)).astype(np.float32)
    noise_W1 = rng.random((n, N_INPUTS, N_HIDDEN)).astype(np.float32)
    noise_W2 = rng.random((n, N_HIDDEN, N_OUTPUTS)).astype(np.float32)

    mut_r  = (rng.standard_normal((n, N_BODY)) * mut_scale).astype(np.float32)

    # germline/somatic split:
    # W_body (body) is inherited from parent and mutated — somatic lineage
    # W1/W2  (brain) reset fresh each generation — germline reset
    W_body = pop['W_body'][idx] + np.where(noise_r < mut_rate, mut_r, 0)
    W1     = (rng.standard_normal((n, N_INPUTS, N_HIDDEN))  * 0.8).astype(np.float32)
    W2     = (rng.standard_normal((n, N_HIDDEN, N_OUTPUTS)) * 0.8).astype(np.float32)

    t   = decode(W_body)
    ang = pop['angle'][idx] + np.pi + rng.uniform(-0.5, 0.5, n).astype(np.float32)

    return {
        'x':          (pop['x'][idx] + np.cos(ang) * (pop['size'][idx] * 2 + 2)) % WIDTH,
        'y':          (pop['y'][idx] + np.sin(ang) * (pop['size'][idx] * 2 + 2)) % HEIGHT,
        'angle':      ang,
        'energy':     pop['clone_with'][idx].copy(),
        'W_body': W_body, 'W1': W1, 'W2': W2,
        **t,
        'generation':  (pop['generation'][idx] + 1).astype(np.int32),
        'age':         np.zeros(n, dtype=np.int32),
        'eaten':       np.zeros(n, dtype=np.int32),
        'h_state':     pop['h_state'][idx] * pop['epigenetic'][idx, None],
        'lineage_id':  pop['lineage_id'][idx].copy(),
    }
