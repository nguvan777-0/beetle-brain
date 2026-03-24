"""Ray-march sensing: each wight casts N_RAYS through the world grid."""
import numpy as np
from sim.config import N_RAYS, ENERGY_MAX
from sim.grid.constants import GW, GH, GRID_SCALE, MAX_STEPS, _STEPS, _RAY_OFFSETS


def sense(pop, grid):
    """
    Ray-march through the world grid for all N organisms simultaneously.
    Returns inputs (N, N_INPUTS).
    """
    N      = len(pop['x'])
    inputs = np.zeros((N, N_RAYS * 2 + 1), dtype=np.float32)

    half_fov   = pop['fov'][:, None] * 0.5
    ray_angles = pop['angle'][:, None] + _RAY_OFFSETS[None, :] * half_fov
    ray_dirs   = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=-1)

    gx = pop['x'] * GRID_SCALE
    gy = pop['y'] * GRID_SCALE

    coords_x = gx[:, None, None] + ray_dirs[:, :, None, 0] * _STEPS[None, None, :]
    coords_y = gy[:, None, None] + ray_dirs[:, :, None, 1] * _STEPS[None, None, :]

    rx = coords_x.astype(np.int32) % GW
    ry = coords_y.astype(np.int32) % GH

    food_hits = grid[0][ry, rx] > 0
    org_hits  = grid[1][ry, rx] > 0

    size_pix = np.ceil(pop['size'] * GRID_SCALE).astype(np.int32)
    step_idx = np.arange(MAX_STEPS, dtype=np.int32)[None, None, :]
    org_hits = org_hits & (step_idx >= size_pix[:, None, None])

    ray_len_pix = pop['ray_len'] * GRID_SCALE

    def _first_hit(hits):
        has_hit  = hits.any(axis=2)
        hit_step = np.argmax(hits, axis=2).astype(np.float32) + 1.0
        dist     = np.where(has_hit, hit_step, ray_len_pix[:, None])
        return np.clip(dist / ray_len_pix[:, None], 0.0, 1.0)

    inputs[:, 0:N_RAYS * 2:2] = 1.0 - _first_hit(food_hits)
    inputs[:, 1:N_RAYS * 2:2] = 1.0 - _first_hit(org_hits)
    inputs[:, -1]             = pop['energy'] / ENERGY_MAX
    return inputs
