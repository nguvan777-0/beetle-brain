"""Cloning with mutation — the only way new wights enter the world."""
import numpy as np
from sim.config import WIDTH, HEIGHT, N_INPUTS, N_HIDDEN, N_OUTPUTS
from sim.population.genome import decode, N_BODY
from sim import phylo


def clone_batch(pop, idx, rng, phylo_state):
    """Return a new pop dict of children cloned (with mutation) from parent indices."""
    n = len(idx)

    mut_rate  = pop['mutation_rate'][idx]    # (n,)
    mut_scale = pop['mutation_scale'][idx]   # (n,)

    def _mutate(parent, noise_shape, extra_dims):
        noise = rng.random(noise_shape).astype(np.float32)
        delta = (rng.standard_normal(noise_shape) * mut_scale[(...,) + (None,) * extra_dims]).astype(np.float32)
        mask  = noise < mut_rate[(...,) + (None,) * extra_dims]
        return parent + np.where(mask, delta, 0)

    # all weights are genome — inherited and mutated
    W_body = _mutate(pop['W_body'][idx], (n, N_BODY),              1)
    W1     = _mutate(pop['W1'][idx],     (n, N_INPUTS, N_HIDDEN),  2)
    W2     = _mutate(pop['W2'][idx],     (n, N_HIDDEN, N_OUTPUTS), 2)
    Wh     = _mutate(pop['Wh'][idx],     (n, N_HIDDEN, N_HIDDEN),  2)

    t   = decode(W_body)
    ang = pop['angle'][idx] + np.pi + rng.uniform(-0.5, 0.5, n).astype(np.float32)

    return {
        'x':          (pop['x'][idx] + np.cos(ang) * (pop['size'][idx] * 2 + 2)) % WIDTH,
        'y':          (pop['y'][idx] + np.sin(ang) * (pop['size'][idx] * 2 + 2)) % HEIGHT,
        'angle':      ang,
        'energy':     pop['clone_with'][idx].copy(),
        'W_body': W_body, 'W1': W1, 'W2': W2, 'Wh': Wh,
        **t,
        'generation':    (pop['generation'][idx] + 1).astype(np.int32),
        'age':           np.zeros(n, dtype=np.int32),
        'eaten':         np.zeros(n, dtype=np.int32),
        'h_state':       pop['h_state'][idx] * pop['epigenetic'][idx, None],
        'lineage_id':    pop['lineage_id'][idx].copy(),
        'individual_id': phylo.alloc(n, pop['individual_id'][idx], phylo_state, rng),
    }
