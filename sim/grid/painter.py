"""Rasterize the world into spatial grids each tick."""
import numpy as np
from sim.grid.constants import GRID_SCALE, GW, GH, _DISK_OFFSETS, DILATION_R_PIX


def paint_idx_grid(pop):
    """
    Returns idx_grid (GH, GW) int32 — organism index at each cell, -1 = empty.
    Single pixel per organism. Used by eating, predation, and HGT broad-phase.
    """
    N = len(pop['x'])
    oy = np.clip((pop['y'] * GRID_SCALE).astype(np.int32), 0, GH - 1)
    ox = np.clip((pop['x'] * GRID_SCALE).astype(np.int32), 0, GW - 1)
    idx_grid = np.full((GH, GW), -1, dtype=np.int32)
    idx_grid[oy, ox] = np.arange(N, dtype=np.int32)
    return idx_grid


def paint_color_grids(pop, food):
    """
    Returns (r_flat, g_flat, b_flat, food_flat) as (GH*GW,) float32 arrays.
    Organisms are painted as size-proportional disks, vectorized per radius group.
    Passed directly as inputs to the CoreML sense+brain model.
    """
    r_grid    = np.zeros((GH, GW), dtype=np.float32)
    g_grid    = np.zeros((GH, GW), dtype=np.float32)
    b_grid    = np.zeros((GH, GW), dtype=np.float32)
    food_grid = np.zeros((GH, GW), dtype=np.float32)

    if len(food) > 0:
        fy = np.clip((food[:, 1] * GRID_SCALE).astype(np.int32), 0, GH - 1)
        fx = np.clip((food[:, 0] * GRID_SCALE).astype(np.int32), 0, GW - 1)
        food_grid[fy, fx] = 1.0

    N = len(pop['x'])
    if N > 0:
        oy    = np.clip((pop['y'] * GRID_SCALE).astype(np.int32), 0, GH - 1)
        ox    = np.clip((pop['x'] * GRID_SCALE).astype(np.int32), 0, GW - 1)
        r_pix = np.clip(np.ceil(pop['size'] * GRID_SCALE).astype(np.int32), 1, DILATION_R_PIX)
        r_val = np.clip(pop['r'], 1, 255).astype(np.float32)
        g_val = np.clip(pop['g'], 1, 255).astype(np.float32)
        b_val = np.clip(pop['b'], 1, 255).astype(np.float32)

        for r, (dy_offs, dx_offs) in _DISK_OFFSETS.items():
            m = r_pix == r
            if not m.any():
                continue
            cy = np.clip(oy[m, None] + dy_offs[None, :], 0, GH - 1)
            cx = np.clip(ox[m, None] + dx_offs[None, :], 0, GW - 1)
            r_grid[cy, cx] = r_val[m, None]
            g_grid[cy, cx] = g_val[m, None]
            b_grid[cy, cx] = b_val[m, None]

    return r_grid.ravel(), g_grid.ravel(), b_grid.ravel(), food_grid.ravel()
