"""Phylogenetic ancestry via a ring-buffer parent table.

PhyloState is a plain dict: {'parent': int32[M], 'next_id': int}
No module-level mutable state — pass state explicitly everywhere.
Any UI, any language can own and serialize this dict.
"""
import numpy as np

M = 1 << 18   # 262144 ring slots — safe for hours of play


def new_state(n_founders: int) -> dict:
    """Fresh phylo state for n founders (IDs 0..n-1)."""
    return {
        'parent':  np.full(M, -1, dtype=np.int32),
        'next_id': n_founders,
    }


def from_snapshot(individual_ids: np.ndarray) -> dict:
    """Reinitialize after loading — treat survivors as founders, continue counter."""
    return {
        'parent':  np.full(M, -1, dtype=np.int32),
        'next_id': int(individual_ids.max()) + 1,
    }


def alloc(n: int, parent_ids: np.ndarray, state: dict) -> np.ndarray:
    """Assign n new IDs with given parents. Mutates state in place, returns IDs."""
    start = state['next_id']
    ids   = np.arange(start, start + n, dtype=np.int32)
    state['parent'][ids % M] = parent_ids
    state['next_id'] += n
    return ids


def ancestor_at(ids: np.ndarray, depth: int, state: dict) -> np.ndarray:
    """Return ancestor depth steps back for each ID. Vectorized, O(depth) passes."""
    cur    = ids.astype(np.int32).copy()
    parent = state['parent']
    for _ in range(depth):
        parents = parent[cur % M]
        cur     = np.where(parents >= 0, parents, cur)
    return cur
