"""Predation: O(N) patch-based prey detection with energy conservation."""
import numpy as np
from sim.config import SIZE_MAX, CAMO_BONUS, CAMO_ENABLED
from sim.grid.constants import PRED_R_PIX


def predation(pop, j_idx, valid, j_safe, dist):
    """
    Hunt prey that are smaller and in range.

    j_idx, valid, j_safe, dist: pre-computed patch geometry from tick — shared
    with contact_hgt to avoid recomputing dx/dy/sqrt twice per tick.
    Returns (hunted mask (N,), prey_gain (N,), pred_idx, prey_idx).
    Energy is split among all hunters that hunt the same prey.
    """
    N = len(pop['x'])
    if N <= 1:
        return np.zeros(N, dtype=bool), np.zeros(N, dtype=np.float32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    brightness = (pop['r'].astype(np.float32) + pop['g'] + pop['b']) / (3.0 * 255.0)
    detect_r   = pop['size'] + (brightness * CAMO_BONUS if CAMO_ENABLED else 0.0)

    in_range = dist < (pop['size'][:, None] + detect_r[j_safe])
    bigger   = pop['size'][:, None] > pop['size'][j_safe] * pop['pred_ratio'][:, None]
    hunts    = valid & in_range & bigger

    # split prey energy among all hunters of the same prey
    prey_hunt_counts = np.bincount(j_idx[hunts], minlength=N).astype(np.float32)
    counts_per_slot  = np.where(hunts, prey_hunt_counts[j_safe].clip(min=1), 1.0)
    prey_gain = (hunts * pop['energy'][j_safe] / counts_per_slot).sum(axis=1) * 0.3

    hunted = np.zeros(N, dtype=bool)
    hunted[j_idx[hunts]] = True

    pred_idx, slot_idx = np.where(hunts)
    prey_idx           = j_idx[pred_idx, slot_idx]

    return hunted, prey_gain, pred_idx, prey_idx
