"""Genome encoding: decode W_body weights into physical traits."""
import numpy as np
from sim.config import (
    SPEED_MIN, SPEED_MAX, FOV_MIN, FOV_MAX,
    RAY_MIN, RAY_MAX, SIZE_MIN, SIZE_MAX,
    BREED_AT_MIN, BREED_AT_MAX, CLONE_WITH_MIN, CLONE_WITH_MAX,
    MUTATION_RATE_MIN, MUTATION_RATE_MAX, MUTATION_SCALE_MIN, MUTATION_SCALE_MAX,
    EPIGENETIC_MIN, EPIGENETIC_MAX,
    WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX,
    MOUTH_MIN, MOUTH_MAX,
    PRED_RATIO_MIN, PRED_RATIO_MAX,
    HGT_EAT_MIN, HGT_EAT_MAX, HGT_CONTACT_MIN, HGT_CONTACT_MAX,
    N_HIDDEN, N_RAYS,
)

N_BODY = 20  # number of body genome weights


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
        'turn_s':        (0.05              + s[:, 7]  * 0.25).astype(np.float32),
        'breed_at':      (BREED_AT_MIN      + s[:, 8]  * (BREED_AT_MAX      - BREED_AT_MIN)).astype(np.float32),
        'clone_with':    (CLONE_WITH_MIN    + s[:, 9]  * (CLONE_WITH_MAX    - CLONE_WITH_MIN)).astype(np.float32),
        'mutation_rate': (MUTATION_RATE_MIN + s[:, 10] * (MUTATION_RATE_MAX - MUTATION_RATE_MIN)).astype(np.float32),
        'mutation_scale':(MUTATION_SCALE_MIN+ s[:, 11] * (MUTATION_SCALE_MAX- MUTATION_SCALE_MIN)).astype(np.float32),
        'epigenetic':    (EPIGENETIC_MIN    + s[:, 12] * (EPIGENETIC_MAX    - EPIGENETIC_MIN)).astype(np.float32),
        'weight_decay':  (WEIGHT_DECAY_MIN  + s[:, 13] * (WEIGHT_DECAY_MAX  - WEIGHT_DECAY_MIN)).astype(np.float32),
        'mouth':         (MOUTH_MIN         + s[:, 14] * (MOUTH_MAX         - MOUTH_MIN)).astype(np.float32),
        'pred_ratio':    (PRED_RATIO_MIN    + s[:, 15] * (PRED_RATIO_MAX    - PRED_RATIO_MIN)).astype(np.float32),
        'hgt_eat_rate':  (HGT_EAT_MIN      + s[:, 16] * (HGT_EAT_MAX      - HGT_EAT_MIN)).astype(np.float32),
        'hgt_contact_rate': (HGT_CONTACT_MIN + s[:, 17] * (HGT_CONTACT_MAX - HGT_CONTACT_MIN)).astype(np.float32),
        'active_neurons': (s[:, 18] * N_HIDDEN).astype(np.int32),       # 0..N_HIDDEN; 0 = straight-line rover (no brain)
        'n_rays':        (s[:, 19] * (N_RAYS + 1)).astype(np.int32),        # 0..N_RAYS; 0 = no vision
        'r':             (40 + s[:, 4] * 215).astype(np.int32),
        'g':             (40 + s[:, 5] * 215).astype(np.int32),
        'b':             (40 + s[:, 6] * 215).astype(np.int32),
    }
