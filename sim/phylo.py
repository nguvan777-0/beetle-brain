"""Phylogenetic ancestry via a ring-buffer parent table.

PhyloState is a plain dict: {'parent': int32[M], 'hue': float32[M], 'next_id': int}
  hue  — lineage color inherited at birth, tiny random drift each generation.
         Founders are evenly spaced around the color wheel.
No module-level mutable state — pass state explicitly everywhere.
Any UI, any language can own and serialize this dict.
"""
import numpy as np

M = 1 << 18   # 262144 ring slots — safe for hours of play
_HUE_DRIFT = 0.03   # ±drift per generation (~11°); random walk so siblings stay close


def new_state(n_founders: int) -> dict:
    """Fresh phylo state for n founders (IDs 0..n-1)."""
    hue = np.zeros(M, dtype=np.float32)
    hue[:n_founders] = np.linspace(0, 1, n_founders, endpoint=False)
    return {
        'parent':  np.full(M, -1, dtype=np.int32),
        'hue':     hue,
        'next_id': n_founders,
    }


def from_snapshot(individual_ids: np.ndarray, hue_array: np.ndarray = None) -> dict:
    """Reinitialize after loading — restore hue array if available."""
    hue = hue_array.astype(np.float32) if hue_array is not None else np.zeros(M, dtype=np.float32)
    return {
        'parent':  np.full(M, -1, dtype=np.int32),
        'hue':     hue,
        'next_id': int(individual_ids.max()) + 1,
    }


def alloc(n: int, parent_ids: np.ndarray, state: dict, rng) -> np.ndarray:
    """Assign n new IDs with given parents. Inherits hue + tiny drift. Returns IDs."""
    start = state['next_id']
    ids   = np.arange(start, start + n, dtype=np.int32)
    state['parent'][ids % M] = parent_ids
    parent_hues = state['hue'][parent_ids % M]
    drift = rng.uniform(-_HUE_DRIFT, _HUE_DRIFT, n).astype(np.float32)
    state['hue'][ids % M] = (parent_hues + drift) % 1.0
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
