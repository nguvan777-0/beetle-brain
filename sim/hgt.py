"""Horizontal gene transfer — eating and contact.

Crossover: single random cut point across the full concatenated genome
(W_body ++ W1.flat ++ W2.flat). Recipient takes [0:cut] from self,
[cut:] from donor. Traits re-decoded from new W_body immediately.
"""
import numpy as np
from sim.population.genome import decode
from sim.grid.constants import GRID_SCALE, GH, GW, PRED_R_PIX, _PR_OFF


def _crossover(pop, recipient_idx, donor_idx, rng):
    """In-place single-point crossover of W_body/W1/W2 for given pairs."""
    n = len(recipient_idx)
    if n == 0:
        return

    wb_r = pop['W_body'][recipient_idx]
    wb_d = pop['W_body'][donor_idx]
    w1_r = pop['W1'][recipient_idx]
    w1_d = pop['W1'][donor_idx]
    w2_r = pop['W2'][recipient_idx]
    w2_d = pop['W2'][donor_idx]
    wh_r = pop['Wh'][recipient_idx]
    wh_d = pop['Wh'][donor_idx]

    g_r = np.concatenate([wb_r, w1_r.reshape(n, -1), w2_r.reshape(n, -1), wh_r.reshape(n, -1)], axis=1)
    g_d = np.concatenate([wb_d, w1_d.reshape(n, -1), w2_d.reshape(n, -1), wh_d.reshape(n, -1)], axis=1)
    L   = g_r.shape[1]

    cuts = rng.integers(1, L, size=n)
    mask = np.arange(L)[None, :] >= cuts[:, None]   # True → take from donor
    g_new = np.where(mask, g_d, g_r).astype(np.float32)

    n_wb = wb_r.shape[1]
    n_w1 = w1_r.shape[1] * w1_r.shape[2]
    n_w2 = w2_r.shape[1] * w2_r.shape[2]

    pop['W_body'][recipient_idx] = g_new[:, :n_wb]
    pop['W1'][recipient_idx]     = g_new[:, n_wb            : n_wb + n_w1          ].reshape(w1_r.shape)
    pop['W2'][recipient_idx]     = g_new[:, n_wb + n_w1     : n_wb + n_w1 + n_w2  ].reshape(w2_r.shape)
    pop['Wh'][recipient_idx]     = g_new[:, n_wb + n_w1 + n_w2 :                   ].reshape(wh_r.shape)

    # re-decode body traits from updated W_body
    new_traits = decode(pop['W_body'][recipient_idx])
    for key, val in new_traits.items():
        pop[key][recipient_idx] = val


def eat_hgt(pop, predator_idx, prey_idx, rng):
    """HGT via predation: each (predator, prey) pair rolls against hgt_eat_rate."""
    if len(predator_idx) == 0:
        return
    rolls = rng.random(len(predator_idx))
    take  = rolls < pop['hgt_eat_rate'][predator_idx]
    if take.any():
        _crossover(pop, predator_idx[take], prey_idx[take], rng)


def contact_hgt(pop, idx_grid, rng):
    """HGT via proximity: wights in touching range roll against hgt_contact_rate."""
    N = len(pop['x'])
    if N <= 1:
        return

    oy = np.clip((pop['y'] * GRID_SCALE).astype(np.int32), 0, GH - 1)
    ox = np.clip((pop['x'] * GRID_SCALE).astype(np.int32), 0, GW - 1)

    row_idx = (oy[:, None, None] + _PR_OFF[None, :, None]) % GH
    col_idx = (ox[:, None, None] + _PR_OFF[None, None, :]) % GW
    j_idx   = idx_grid[row_idx, col_idx].reshape(N, -1)

    i_idx = np.arange(N, dtype=np.int32)[:, None]
    valid = (j_idx >= 0) & (j_idx != i_idx)
    j_safe = np.where(valid, j_idx, 0)

    dx   = pop['x'][:, None] - pop['x'][j_safe]
    dy   = pop['y'][:, None] - pop['y'][j_safe]
    dist = np.sqrt(dx * dx + dy * dy)

    in_contact = valid & (dist < (pop['size'][:, None] + pop['size'][j_safe]))
    rolls      = rng.random((N, j_idx.shape[1]))
    do_hgt     = in_contact & (rolls < pop['hgt_contact_rate'][:, None])

    # one donor per recipient (first contact that fired)
    rows, cols = np.where(do_hgt)
    if len(rows) == 0:
        return
    _, first = np.unique(rows, return_index=True)
    rec = rows[first]
    don = j_idx[rec, cols[first]]
    _crossover(pop, rec, don, rng)
