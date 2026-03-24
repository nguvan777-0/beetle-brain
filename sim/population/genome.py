"""Genome encoding: decode W_body weights into physical traits."""
import numpy as np
from sim.config import (
    SPEED_MIN, SPEED_MAX, FOV_MIN, FOV_MAX,
    RAY_MIN, RAY_MAX, SIZE_MIN, SIZE_MAX, DRAIN_MIN, DRAIN_MAX,
    BREED_AT_MIN, BREED_AT_MAX, CLONE_WITH_MIN, CLONE_WITH_MAX,
    MUTATION_RATE_MIN, MUTATION_RATE_MAX, MUTATION_SCALE_MIN, MUTATION_SCALE_MAX,
    EPIGENETIC_MIN, EPIGENETIC_MAX,
    WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX,
)

N_BODY = 15  # number of body genome weights


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode(W_body):
    """W_body (N, N_BODY) → trait arrays (all shape (N,))."""
    s              = sig(W_body)
    speed          = SPEED_MIN         + s[:, 0]  * (SPEED_MAX         - SPEED_MIN)
    fov            = FOV_MIN           + s[:, 1]  * (FOV_MAX           - FOV_MIN)
    ray            = RAY_MIN           + s[:, 2]  * (RAY_MAX           - RAY_MIN)
    size           = SIZE_MIN          + s[:, 3]  * (SIZE_MAX          - SIZE_MIN)
    drain          = DRAIN_MIN         + s[:, 4]  * (DRAIN_MAX         - DRAIN_MIN)
    turn_s         = 0.05              + s[:, 8]  * 0.25
    breed_at       = BREED_AT_MIN      + s[:, 9]  * (BREED_AT_MAX      - BREED_AT_MIN)
    clone_with     = CLONE_WITH_MIN    + s[:, 10] * (CLONE_WITH_MAX    - CLONE_WITH_MIN)
    mutation_rate  = MUTATION_RATE_MIN  + s[:, 11] * (MUTATION_RATE_MAX  - MUTATION_RATE_MIN)
    mutation_scale = MUTATION_SCALE_MIN + s[:, 12] * (MUTATION_SCALE_MAX - MUTATION_SCALE_MIN)
    epigenetic    = EPIGENETIC_MIN    + s[:, 13] * (EPIGENETIC_MAX    - EPIGENETIC_MIN)
    weight_decay  = WEIGHT_DECAY_MIN  + s[:, 14] * (WEIGHT_DECAY_MAX  - WEIGHT_DECAY_MIN)
    r = (40 + s[:, 5] * 215).astype(np.int32)
    g = (40 + s[:, 6] * 215).astype(np.int32)
    b = (40 + s[:, 7] * 215).astype(np.int32)
    return speed, fov, ray, size, drain, turn_s, r, g, b, breed_at, clone_with, mutation_rate, mutation_scale, epigenetic, weight_decay
