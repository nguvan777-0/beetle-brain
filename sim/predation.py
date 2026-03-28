"""Predation: O(N) patch-based kill detection with energy conservation."""
import numpy as np
from sim.config import SIZE_MAX, CAMO_BONUS, CAMO_ENABLED
from sim.grid.constants import PRED_R_PIX


def predation(pop, idx_grid, j_idx):
    """
    For each wight, sample a (2*PRED_R_PIX+1)² patch of idx_grid to find
    nearby candidates. Kill prey that are smaller and in range.

    j_idx: (N, patch²) int32 pre-computed by tick — shared with contact_hgt.
    Returns (killed mask (N,), prey_gain (N,), pred_idx, prey_idx).
    Energy is split among all predators that kill the same prey.
    """
    N = len(pop['x'])
    if N <= 1:
        return np.zeros(N, dtype=bool), np.zeros(N, dtype=np.float32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    brightness = (pop['r'].astype(np.float32) + pop['g'] + pop['b']) / (3.0 * 255.0)
    detect_r   = pop['size'] + (brightness * CAMO_BONUS if CAMO_ENABLED else 0.0)

    i_idx  = np.arange(N, dtype=np.int32)[:, None]
    valid  = (j_idx >= 0) & (j_idx != i_idx)
    j_safe = np.where(valid, j_idx, 0)

    dx   = pop['x'][:, None] - pop['x'][j_safe]
    dy   = pop['y'][:, None] - pop['y'][j_safe]
    dist = np.sqrt(dx * dx + dy * dy)

    in_range = dist < (pop['size'][:, None] + detect_r[j_safe])
    bigger   = pop['size'][:, None] > pop['size'][j_safe] * pop['pred_ratio'][:, None]
    kills    = valid & in_range & bigger

    # split prey energy among all killers of the same prey
    prey_kill_counts = np.zeros(N, dtype=np.float32)
    np.add.at(prey_kill_counts, j_idx[kills], 1.0)
    counts_per_slot  = np.where(kills, prey_kill_counts[j_safe].clip(min=1), 1.0)
    prey_gain = (kills * pop['energy'][j_safe] / counts_per_slot).sum(axis=1) * 0.3

    killed = np.zeros(N, dtype=bool)
    killed[j_idx[kills]] = True

    pred_idx, slot_idx = np.where(kills)
    prey_idx           = j_idx[pred_idx, slot_idx]

    return killed, prey_gain, pred_idx, prey_idx
