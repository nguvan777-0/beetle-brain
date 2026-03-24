"""Predation: O(N) patch-based kill detection with energy conservation."""
import numpy as np
from sim.config import SIZE_MAX, CAMO_BONUS, CAMO_ENABLED
from sim.grid.constants import GRID_SCALE, GH, GW, PRED_R_PIX, _PR_OFF


def predation(pop, idx_grid):
    """
    For each wight, sample a (2*PRED_R_PIX+1)² patch of idx_grid to find
    nearby candidates. Kill prey that are smaller and in range.

    Returns (killed mask (N,), prey_gain (N,)).
    Energy is split among all predators that kill the same prey.
    """
    N = len(pop['x'])
    if N <= 1:
        return np.zeros(N, dtype=bool), np.zeros(N, dtype=np.float32)

    brightness = (pop['r'].astype(np.float32) + pop['g'] + pop['b']) / (3.0 * 255.0)
    detect_r   = pop['size'] + (brightness * CAMO_BONUS if CAMO_ENABLED else 0.0)

    oy = np.clip((pop['y'] * GRID_SCALE).astype(np.int32), 0, GH - 1)
    ox = np.clip((pop['x'] * GRID_SCALE).astype(np.int32), 0, GW - 1)

    row_idx = (oy[:, None, None] + _PR_OFF[None, :, None]) % GH
    col_idx = (ox[:, None, None] + _PR_OFF[None, None, :]) % GW
    j_idx   = idx_grid[row_idx, col_idx].reshape(N, -1)

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
    prey_gain = (kills * pop['energy'][j_safe] / counts_per_slot).sum(axis=1) * 0.7

    killed = np.zeros(N, dtype=bool)
    killed[j_idx[kills]] = True

    return killed, prey_gain
