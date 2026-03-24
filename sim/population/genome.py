"""Genome encoding: decode W_body weights into physical traits."""
import numpy as np
from sim.config import (
    SPEED_MIN, SPEED_MAX, FOV_MIN, FOV_MAX,
    RAY_MIN, RAY_MAX, SIZE_MIN, SIZE_MAX, DRAIN_MIN, DRAIN_MAX,
)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode(W_body):
    """W_body (N, N_BODY) → trait arrays (all shape (N,))."""
    s      = sig(W_body)
    speed  = SPEED_MIN  + s[:, 0] * (SPEED_MAX  - SPEED_MIN)
    fov    = FOV_MIN    + s[:, 1] * (FOV_MAX    - FOV_MIN)
    ray    = RAY_MIN    + s[:, 2] * (RAY_MAX    - RAY_MIN)
    size   = SIZE_MIN   + s[:, 3] * (SIZE_MAX   - SIZE_MIN)
    drain  = DRAIN_MIN  + s[:, 4] * (DRAIN_MAX  - DRAIN_MIN)
    turn_s = 0.05       + s[:, 8] * 0.25
    r = (40 + s[:, 5] * 215).astype(np.int32)
    g = (40 + s[:, 6] * 215).astype(np.int32)
    b = (40 + s[:, 7] * 215).astype(np.int32)
    return speed, fov, ray, size, drain, turn_s, r, g, b
