"""Rasterize the world into a spatial grid each tick."""
import numpy as np
from sim.grid.constants import GRID_SCALE, GW, GH


def paint_grid(pop, food):
    """
    Returns (grid, idx_grid):
      grid     (2, GH, GW) uint8 — channel 0: food, channel 1: organism brightness
      idx_grid (GH, GW)    int32 — organism index at each cell, -1 = empty
    """
    N    = len(pop['x'])
    grid = np.zeros((2, GH, GW), dtype=np.uint8)

    if len(food):
        fy = np.clip((food[:, 1] * GRID_SCALE).astype(np.int32), 0, GH - 1)
        fx = np.clip((food[:, 0] * GRID_SCALE).astype(np.int32), 0, GW - 1)
        grid[0, fy, fx] = 1

    oy = np.clip((pop['y'] * GRID_SCALE).astype(np.int32), 0, GH - 1)
    ox = np.clip((pop['x'] * GRID_SCALE).astype(np.int32), 0, GW - 1)
    brightness = ((pop['r'].astype(np.int32) + pop['g'].astype(np.int32)
                   + pop['b'].astype(np.int32)) / 3).astype(np.int32)
    grid[1, oy, ox] = np.clip(brightness, 1, 255).astype(np.uint8)

    idx_grid = np.full((GH, GW), -1, dtype=np.int32)
    idx_grid[oy, ox] = np.arange(N, dtype=np.int32)

    return grid, idx_grid
