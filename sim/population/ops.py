"""Population array operations: filter and concatenate."""
import numpy as np


def filter_pop(pop, mask):
    """Return a new pop dict keeping only entries where mask is True (or indices)."""
    return {k: v[mask] for k, v in pop.items()}


def concat_pop(a, b):
    """Concatenate two population dicts along axis 0."""
    return {k: np.concatenate([a[k], b[k]]) for k in a}
