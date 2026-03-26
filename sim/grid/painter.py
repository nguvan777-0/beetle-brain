"""Rasterize the world into a spatial grid each tick."""
import numpy as np
from sim.grid.constants import GRID_SCALE, GW, GH, _DISK_OFFSETS


def paint_grid(pop, food):
    """
    Returns (grid, idx_grid):
      grid     (4, GH, GW) uint8 — channel 0: food, channels 1-3: organism r/g/b (1-255, 0=empty)
      idx_grid (GH, GW)    int32 — organism index at each cell, -1 = empty

    Each organism is rasterized as a disk proportional to its size so that
    raycasting sees large organisms from farther away.
    """
    N    = len(pop['x'])
    grid = np.zeros((4, GH, GW), dtype=np.uint8)

    if len(food):
        fy = np.clip((food[:, 1] * GRID_SCALE).astype(np.int32), 0, GH - 1)
        fx = np.clip((food[:, 0] * GRID_SCALE).astype(np.int32), 0, GW - 1)
        grid[0, fy, fx] = 1

    oy      = np.clip((pop['y'] * GRID_SCALE).astype(np.int32), 0, GH - 1)
    ox      = np.clip((pop['x'] * GRID_SCALE).astype(np.int32), 0, GW - 1)
    r_pix   = np.ceil(pop['size'] * GRID_SCALE).astype(np.int32)
    r_clip  = np.clip(pop['r'], 1, 255).astype(np.uint8)
    g_clip  = np.clip(pop['g'], 1, 255).astype(np.uint8)
    b_clip  = np.clip(pop['b'], 1, 255).astype(np.uint8)
    indices = np.arange(N, dtype=np.int32)

    idx_grid = np.full((GH, GW), -1, dtype=np.int32)

    for r, (dy_offs, dx_offs) in _DISK_OFFSETS.items():
        mask = r_pix == r
        if not mask.any():
            continue
        # cy, cx: (M, K) — M organisms of this radius, K cells per disk
        cy = np.clip(oy[mask, None] + dy_offs[None, :], 0, GH - 1)
        cx = np.clip(ox[mask, None] + dx_offs[None, :], 0, GW - 1)
        grid[1, cy, cx] = r_clip[mask, None]
        grid[2, cy, cx] = g_clip[mask, None]
        grid[3, cy, cx] = b_clip[mask, None]
        idx_grid[cy, cx] = indices[mask, None]

    return grid, idx_grid
