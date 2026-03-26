"""Rasterize organism positions into the spatial index grid for collision broad-phase."""
import numpy as np
from sim.grid.constants import GRID_SCALE, GW, GH


def paint_idx_grid(pop):
    """
    Returns idx_grid (GH, GW) int32 — organism index at each cell, -1 = empty.
    Single pixel per organism. Used by eating, predation, and HGT broad-phase.
    Color and food grids are built inside the CoreML graph via scatter + dilation.
    """
    N = len(pop['x'])
    oy = np.clip((pop['y'] * GRID_SCALE).astype(np.int32), 0, GH - 1)
    ox = np.clip((pop['x'] * GRID_SCALE).astype(np.int32), 0, GW - 1)
    idx_grid = np.full((GH, GW), -1, dtype=np.int32)
    idx_grid[oy, ox] = np.arange(N, dtype=np.int32)
    return idx_grid
