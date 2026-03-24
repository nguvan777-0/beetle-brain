"""Cloning with mutation — the only way new wights enter the world."""
import numpy as np
from sim.config import WIDTH, HEIGHT, N_INPUTS, N_HIDDEN, N_OUTPUTS
from sim.population.genome import decode, N_BODY


def clone_batch(pop, idx, rng):
    """Return a new pop dict of children cloned (with mutation) from parent indices."""
    n = len(idx)

    # each parent uses its own mutation_rate and mutation_scale
    mut_rate  = pop['mutation_rate'][idx, None]              # (n, 1) for broadcasting
    mut_scale = pop['mutation_scale'][idx, None]

    noise_r  = rng.random((n, N_BODY)).astype(np.float32)
    noise_W1 = rng.random((n, N_INPUTS, N_HIDDEN)).astype(np.float32)
    noise_W2 = rng.random((n, N_HIDDEN, N_OUTPUTS)).astype(np.float32)

    mut_r  = (rng.standard_normal((n, N_BODY))              * mut_scale).astype(np.float32)
    mut_W1 = (rng.standard_normal((n, N_INPUTS, N_HIDDEN))  * mut_scale[:, :, None]).astype(np.float32)
    mut_W2 = (rng.standard_normal((n, N_HIDDEN, N_OUTPUTS)) * mut_scale[:, :, None]).astype(np.float32)

    W_body = pop['W_body'][idx] + np.where(noise_r  < mut_rate,           mut_r,  0)
    W1     = pop['W1'][idx]     + np.where(noise_W1 < mut_rate[:, :, None], mut_W1, 0)
    W2     = pop['W2'][idx]     + np.where(noise_W2 < mut_rate[:, :, None], mut_W2, 0)

    speed, fov, ray, size, drain, turn_s, r, g, b, breed_at, clone_with, mut_rate_c, mut_scale_c, epigenetic_c, weight_decay_c = decode(W_body)
    ang = pop['angle'][idx] + np.pi + rng.uniform(-0.5, 0.5, n).astype(np.float32)

    return {
        'x':             (pop['x'][idx] + np.cos(ang) * (pop['size'][idx] * 2 + 2)) % WIDTH,
        'y':             (pop['y'][idx] + np.sin(ang) * (pop['size'][idx] * 2 + 2)) % HEIGHT,
        'angle':         ang,
        'energy':        pop['clone_with'][idx].copy(),
        'W_body':        W_body, 'W1': W1, 'W2': W2,
        'speed':         speed.astype(np.float32),
        'fov':           fov.astype(np.float32),
        'ray_len':       ray.astype(np.float32),
        'size':          size.astype(np.float32),
        'drain':         drain.astype(np.float32),
        'turn_s':        turn_s.astype(np.float32),
        'breed_at':      breed_at.astype(np.float32),
        'clone_with':    clone_with.astype(np.float32),
        'mutation_rate':  mut_rate_c.astype(np.float32),
        'mutation_scale': mut_scale_c.astype(np.float32),
        'epigenetic':     epigenetic_c.astype(np.float32),
        'weight_decay':   weight_decay_c.astype(np.float32),
        'r': r, 'g': g, 'b': b,
        'generation':    (pop['generation'][idx] + 1).astype(np.int32),
        'age':           np.zeros(n, dtype=np.int32),
        'eaten':         np.zeros(n, dtype=np.int32),
        'h_state':       pop['h_state'][idx] * pop['epigenetic'][idx, None],
    }
