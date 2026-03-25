"""Rasterize the world into a spatial grid each tick."""
import numpy as np
from sim.grid.constants import GRID_SCALE, GW, GH


def paint_grid(pop, food):
    """
    Returns (grid, idx_grid):
      grid     (4, GH, GW) uint8 — channel 0: food, channels 1-3: organism r/g/b (1-255, 0=empty)
      idx_grid (GH, GW)    int32 — organism index at each cell, -1 = empty
    """
    N    = len(pop['x'])
    grid = np.zeros((4, GH, GW), dtype=np.uint8)

    if len(food):
        fy = np.clip((food[:, 1] * GRID_SCALE).astype(np.int32), 0, GH - 1)
        fx = np.clip((food[:, 0] * GRID_SCALE).astype(np.int32), 0, GW - 1)
        grid[0, fy, fx] = 1

    oy = np.clip((pop['y'] * GRID_SCALE).astype(np.int32), 0, GH - 1)
    ox = np.clip((pop['x'] * GRID_SCALE).astype(np.int32), 0, GW - 1)
    grid[1, oy, ox] = np.clip(pop['r'], 1, 255).astype(np.uint8)
    grid[2, oy, ox] = np.clip(pop['g'], 1, 255).astype(np.uint8)
    grid[3, oy, ox] = np.clip(pop['b'], 1, 255).astype(np.uint8)

    idx_grid = np.full((GH, GW), -1, dtype=np.int32)
    idx_grid[oy, ox] = np.arange(N, dtype=np.int32)

    return grid, idx_grid
