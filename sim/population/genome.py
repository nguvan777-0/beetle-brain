"""Genome encoding: decode W_body weights into physical traits."""
import numpy as np
from sim.config import (
    SPEED_MIN, SPEED_MAX, FOV_MIN, FOV_MAX,
    RAY_MIN, RAY_MAX, SIZE_MIN, SIZE_MAX, DRAIN_MIN, DRAIN_MAX,
    BREED_AT_MIN, BREED_AT_MAX, CLONE_WITH_MIN, CLONE_WITH_MAX,
    MUTATION_RATE_MIN, MUTATION_RATE_MAX, MUTATION_SCALE_MIN, MUTATION_SCALE_MAX,
    EPIGENETIC_MIN, EPIGENETIC_MAX,
    WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX,
    MOUTH_MIN, MOUTH_MAX,
    ENERGY_MAX_MIN, ENERGY_MAX_MAX,
    PRED_RATIO_MIN, PRED_RATIO_MAX,
    SPEED_SCALE_MIN, SPEED_SCALE_MAX,
)

N_BODY = 19  # number of body genome weights


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode(W_body):
    """W_body (N, N_BODY) → dict of trait arrays, each shape (N,)."""
    s = sig(W_body)
    return {
        'speed':         (SPEED_MIN         + s[:, 0]  * (SPEED_MAX         - SPEED_MIN)).astype(np.float32),
        'fov':           (FOV_MIN           + s[:, 1]  * (FOV_MAX           - FOV_MIN)).astype(np.float32),
        'ray_len':       (RAY_MIN           + s[:, 2]  * (RAY_MAX           - RAY_MIN)).astype(np.float32),
        'size':          (SIZE_MIN          + s[:, 3]  * (SIZE_MAX          - SIZE_MIN)).astype(np.float32),
        'drain':         (DRAIN_MIN         + s[:, 4]  * (DRAIN_MAX         - DRAIN_MIN)).astype(np.float32),
        'turn_s':        (0.05              + s[:, 8]  * 0.25).astype(np.float32),
        'breed_at':      (BREED_AT_MIN      + s[:, 9]  * (BREED_AT_MAX      - BREED_AT_MIN)).astype(np.float32),
        'clone_with':    (CLONE_WITH_MIN    + s[:, 10] * (CLONE_WITH_MAX    - CLONE_WITH_MIN)).astype(np.float32),
        'mutation_rate': (MUTATION_RATE_MIN + s[:, 11] * (MUTATION_RATE_MAX - MUTATION_RATE_MIN)).astype(np.float32),
        'mutation_scale':(MUTATION_SCALE_MIN+ s[:, 12] * (MUTATION_SCALE_MAX- MUTATION_SCALE_MIN)).astype(np.float32),
        'epigenetic':    (EPIGENETIC_MIN    + s[:, 13] * (EPIGENETIC_MAX    - EPIGENETIC_MIN)).astype(np.float32),
        'weight_decay':  (WEIGHT_DECAY_MIN  + s[:, 14] * (WEIGHT_DECAY_MAX  - WEIGHT_DECAY_MIN)).astype(np.float32),
        'mouth':         (MOUTH_MIN         + s[:, 15] * (MOUTH_MAX         - MOUTH_MIN)).astype(np.float32),
        'energy_max':    (ENERGY_MAX_MIN    + s[:, 16] * (ENERGY_MAX_MAX    - ENERGY_MAX_MIN)).astype(np.float32),
        'pred_ratio':    (PRED_RATIO_MIN    + s[:, 17] * (PRED_RATIO_MAX    - PRED_RATIO_MIN)).astype(np.float32),
        'speed_scale':   (SPEED_SCALE_MIN   + s[:, 18] * (SPEED_SCALE_MAX   - SPEED_SCALE_MIN)).astype(np.float32),
        'r':             (40 + s[:, 5] * 215).astype(np.int32),
        'g':             (40 + s[:, 6] * 215).astype(np.int32),
        'b':             (40 + s[:, 7] * 215).astype(np.int32),
    }
