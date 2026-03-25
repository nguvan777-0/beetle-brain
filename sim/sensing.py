"""Ray-march sensing: each wight casts N_RAYS through the world grid."""
import numpy as np
from sim.config import N_RAYS, ENERGY_MAX_SCALE
from sim.grid.constants import GW, GH, GRID_SCALE, MAX_STEPS, _STEPS, _RAY_OFFSETS


def sense(pop, grid):
    """
    Ray-march through the world grid for all N organisms simultaneously.
    Returns inputs (N, N_INPUTS).
    Input layout per ray: [food_dist, org_dist, org_r, org_g, org_b] × N_RAYS, then energy.
    """
    N      = len(pop['x'])
    inputs = np.zeros((N, N_RAYS * 5 + 1), dtype=np.float32)

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
    r_vals    = grid[1][ry, rx].astype(np.float32)   # (N, R, S)
    g_vals    = grid[2][ry, rx].astype(np.float32)
    b_vals    = grid[3][ry, rx].astype(np.float32)
    org_hits  = (r_vals + g_vals + b_vals) > 0

    size_pix = np.ceil(pop['size'] * GRID_SCALE).astype(np.int32)
    step_idx = np.arange(MAX_STEPS, dtype=np.int32)[None, None, :]
    org_hits = org_hits & (step_idx >= size_pix[:, None, None])

    ray_len_pix = np.maximum(pop['ray_len'] * GRID_SCALE, 1.0)

    # per-organism ray activity mask: only rays 0..n_rays[i]-1 are active
    ray_active = (np.arange(N_RAYS)[None, :] < pop['n_rays'][:, None]).astype(np.float32)

    def _first_hit(hits):
        has_hit  = hits.any(axis=2)
        hit_step = np.argmax(hits, axis=2).astype(np.float32) + 1.0
        dist     = np.where(has_hit, hit_step, ray_len_pix[:, None])
        return np.clip(dist / ray_len_pix[:, None], 0.0, 1.0)

    # color at first org hit
    has_org_hit = org_hits.any(axis=2)                                      # (N, R)
    hit_step_idx = np.argmax(org_hits, axis=2)                              # (N, R)
    n_idx = np.arange(N)[:, None]
    r_idx = np.arange(N_RAYS)[None, :]
    r_at_hit = np.where(has_org_hit, r_vals[n_idx, r_idx, hit_step_idx] / 255.0, 0.0)
    g_at_hit = np.where(has_org_hit, g_vals[n_idx, r_idx, hit_step_idx] / 255.0, 0.0)
    b_at_hit = np.where(has_org_hit, b_vals[n_idx, r_idx, hit_step_idx] / 255.0, 0.0)

    inputs[:, 0:N_RAYS * 5:5] = (1.0 - _first_hit(food_hits)) * ray_active
    inputs[:, 1:N_RAYS * 5:5] = (1.0 - _first_hit(org_hits))  * ray_active
    inputs[:, 2:N_RAYS * 5:5] = r_at_hit                       * ray_active
    inputs[:, 3:N_RAYS * 5:5] = g_at_hit                       * ray_active
    inputs[:, 4:N_RAYS * 5:5] = b_at_hit                       * ray_active
    inputs[:, -1]             = pop['energy'] / (ENERGY_MAX_SCALE * pop['size'] ** 2)
    return inputs
