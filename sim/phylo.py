"""Phylogenetic ancestry via a ring-buffer parent table.

Each wight carries an individual_id (int32, ever-increasing counter).
_parent[id % M] stores that wight's parent_id (-1 for founders).

ancestor_at(ids, depth) walks the chain `depth` steps using vectorized
numpy index ops — O(depth) passes, no Python loops over population.

Ring-buffer safety: M=2**18=262144.  At ~5000 births/sec the ring wraps
every ~52 seconds.  Wights live at most a few thousand ticks (~5 sec),
so a slot is always vacated long before its position is recycled.
"""
import numpy as np

M       = 1 << 18                             # 262144 ring slots
_parent = np.full(M, -1, dtype=np.int32)      # -1 = no parent
_next   = [0]                                 # mutable ID counter


def init(n_founders: int) -> np.ndarray:
    """World creation: reset table, return individual_ids for n founders."""
    _parent[:] = -1
    _next[0]   = n_founders
    return np.arange(n_founders, dtype=np.int32)


def init_from_snapshot(individual_ids: np.ndarray) -> None:
    """After loading a snapshot: treat survivors as founders, continue counter."""
    _parent[:] = -1
    _next[0]   = int(individual_ids.max()) + 1


def alloc(n: int, parent_ids: np.ndarray) -> np.ndarray:
    """Assign n new IDs with the given parent_ids.  Returns int32 array."""
    start       = _next[0]
    _next[0]   += n
    ids         = np.arange(start, start + n, dtype=np.int32)
    _parent[ids % M] = parent_ids
    return ids


def ancestor_at(ids: np.ndarray, depth: int) -> np.ndarray:
    """Return the ancestor depth steps back for each wight.

    Fully vectorized: depth passes of numpy fancy-index + where.
    Stops climbing when it hits a founder (-1 sentinel).
    """
    cur = ids.astype(np.int32).copy()
    for _ in range(depth):
        parents = _parent[cur % M]
        cur     = np.where(parents >= 0, parents, cur)
    return cur
