"""Rasterize the world into spatial grids each tick."""
import numpy as np
from sim.grid.constants import GRID_SCALE, GW, GH, _DISK_OFFSETS, DILATION_R_PIX

# Pre-allocated grid buffers — reused every tick to avoid per-tick malloc.
_idx_buf  = np.full((GH, GW), -1, dtype=np.int32)
_r_buf    = np.zeros((GH, GW), dtype=np.float32)
_g_buf    = np.zeros((GH, GW), dtype=np.float32)
_b_buf    = np.zeros((GH, GW), dtype=np.float32)
_food_buf = np.zeros((GH, GW), dtype=np.float32)


def paint_idx_grid(pop):
    """
    Returns idx_grid (GH, GW) int32 — organism index at each cell, -1 = empty.
    Single pixel per organism. Used by eating, predation, and HGT broad-phase.
    Returned buffer is valid only for the current tick.
    """
    N = len(pop['x'])
    oy = np.clip((pop['y'] * GRID_SCALE).astype(np.int32), 0, GH - 1)
    ox = np.clip((pop['x'] * GRID_SCALE).astype(np.int32), 0, GW - 1)
    _idx_buf.fill(-1)
    _idx_buf[oy, ox] = np.arange(N, dtype=np.int32)
    return _idx_buf


def paint_color_grids(pop, food):
    """
    Returns (r_flat, g_flat, b_flat, food_flat) as (GH*GW,) float32 views.
    Organisms are painted as size-proportional disks, vectorized per radius group.
    Passed directly as inputs to the CoreML sense+brain model.
    Returned views are valid only for the current tick.
    """
    _r_buf.fill(0.0)
    _g_buf.fill(0.0)
    _b_buf.fill(0.0)
    _food_buf.fill(0.0)

    if len(food) > 0:
        fy = np.clip((food[:, 1] * GRID_SCALE).astype(np.int32), 0, GH - 1)
        fx = np.clip((food[:, 0] * GRID_SCALE).astype(np.int32), 0, GW - 1)
        _food_buf[fy, fx] = 1.0

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
            _r_buf[cy, cx] = r_val[m, None]
            _g_buf[cy, cx] = g_val[m, None]
            _b_buf[cy, cx] = b_val[m, None]

    return _r_buf.ravel(), _g_buf.ravel(), _b_buf.ravel(), _food_buf.ravel()
