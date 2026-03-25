"""
brain/coreml_sense_brain.py — fused sensing + recurrent brain via CoreML
=========================================================================
Compiles one CoreML program that:
  1. Ray-marches the world grid (sensing) — all N wights in parallel on GPU
  2. Runs the Elman RNN forward pass

Running both ops in a single GPU dispatch eliminates the CPU round-trip
between sensing and brain, making the full perception-action pipeline
O(1) wall-clock in population size (up to MAX_POP hardware limit).

Fallback: numpy sense + CoreML brain (the old path) when compilation fails.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    _HAS_CT = True
except ImportError:
    _HAS_CT = False

from sim.config import (
    N_RAYS, N_INPUTS, N_HIDDEN, N_OUTPUTS, MAX_POP, ENERGY_MAX_SCALE,
)
from sim.grid.constants import GW, GH, GRID_SCALE, MAX_STEPS, _STEPS, _RAY_OFFSETS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH   = PROJECT_ROOT / "build" / "sense_brain.mlpackage"
META_PATH    = PROJECT_ROOT / "build" / "sense_brain_meta.json"

_model      = None
_use_coreml = False


def init_sense_brain() -> bool:
    global _model, _use_coreml

    if not _HAS_CT:
        return False

    if MODEL_PATH.exists() and META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text())
            if (meta.get("max_pop") == MAX_POP and
                    meta.get("n_rays") == N_RAYS and
                    meta.get("max_steps") == MAX_STEPS and
                    meta.get("gw") == GW and meta.get("gh") == GH and
                    meta.get("n_hid") == N_HIDDEN and meta.get("n_out") == N_OUTPUTS and
                    meta.get("has_nrays_mask") is True):
                _model = ct.models.MLModel(str(MODEL_PATH), compute_units=ct.ComputeUnit.ALL)
                _use_coreml = True
                print(f"[SenseBrain] Loaded cached model ({MODEL_PATH.name})")
                return True
        except Exception as e:
            print(f"[SenseBrain] Cache load failed ({e}), rebuilding...")

    print(f"[SenseBrain] Compiling fused sense+brain "
          f"(pop={MAX_POP}, rays={N_RAYS}, steps={MAX_STEPS}, "
          f"grid={GW}×{GH}, {N_INPUTS}→{N_HIDDEN}→{N_OUTPUTS})...",
          end="", flush=True)
    t0 = time.time()
    try:
        _model      = _compile()
        _use_coreml = True
        print(f" done ({time.time() - t0:.1f}s)")
        return True
    except Exception as e:
        print(f" FAILED: {e}")
        return False


def _compile():
    B, R, S   = MAX_POP, N_RAYS, MAX_STEPS
    FLAT      = B * R * S
    GRID_FLAT = GH * GW

    steps_c   = np.array(_STEPS,       dtype=np.float32)   # (S,)
    offsets_c = np.array(_RAY_OFFSETS, dtype=np.float32)   # (R,)

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(B,)),               # x_pos
        mb.TensorSpec(shape=(B,)),               # y_pos
        mb.TensorSpec(shape=(B,)),               # angle
        mb.TensorSpec(shape=(B,)),               # half_fov
        mb.TensorSpec(shape=(B,)),               # ray_len_pix
        mb.TensorSpec(shape=(B,)),               # size_pix
        mb.TensorSpec(shape=(B,)),               # energy_frac
        mb.TensorSpec(shape=(GH, GW)),           # food_grid
        mb.TensorSpec(shape=(GH, GW)),           # org_grid
        mb.TensorSpec(shape=(B, N_INPUTS, N_HIDDEN)),  # W1
        mb.TensorSpec(shape=(B, N_HIDDEN, N_OUTPUTS)), # W2
        mb.TensorSpec(shape=(B, N_HIDDEN)),      # h_prev
        mb.TensorSpec(shape=(B, N_HIDDEN)),      # mask
        mb.TensorSpec(shape=(B, N_RAYS * 2)),    # n_rays_mask: 1 for active ray channels, 0 for inactive
    ])
    def prog(x_pos, y_pos, angle, half_fov,
             ray_len_pix, size_pix, energy_frac,
             food_grid, org_grid, W1, W2, h_prev, mask, n_rays_mask):

        steps_k   = mb.const(val=steps_c)    # (S,)
        offsets_k = mb.const(val=offsets_c)  # (R,)
        zero      = mb.const(val=np.float32(0.0))
        one       = mb.const(val=np.float32(1.0))
        gs        = mb.const(val=np.float32(GRID_SCALE))
        gw_f      = mb.const(val=np.float32(GW))
        gh_f      = mb.const(val=np.float32(GH))
        gw_i      = mb.const(val=np.int32(GW))

        # ── ray angles: (B,) × (R,) → (B, R) ───────────────────────────────
        angle_e   = mb.expand_dims(x=angle,    axes=[1])         # (B, 1)
        hfov_e    = mb.expand_dims(x=half_fov, axes=[1])         # (B, 1)
        offsets_e = mb.expand_dims(x=offsets_k, axes=[0])        # (1, R)
        ray_angles = mb.add(x=angle_e, y=mb.mul(x=offsets_e, y=hfov_e))  # (B, R)

        # ── ray directions ───────────────────────────────────────────────────
        cos_a = mb.cos(x=ray_angles)   # (B, R)
        sin_a = mb.sin(x=ray_angles)   # (B, R)

        # ── world → grid positions ───────────────────────────────────────────
        gx = mb.mul(x=x_pos, y=gs)    # (B,)
        gy = mb.mul(x=y_pos, y=gs)    # (B,)

        # ── sample coordinates: (B, R, S) ────────────────────────────────────
        # steps_k (S,) → (1, 1, S); cos_a (B, R) → (B, R, 1)
        steps_e = mb.expand_dims(x=mb.expand_dims(x=steps_k, axes=[0]), axes=[0])  # (1,1,S)
        cos_e   = mb.expand_dims(x=cos_a, axes=[2])    # (B, R, 1)
        sin_e   = mb.expand_dims(x=sin_a, axes=[2])    # (B, R, 1)
        gx_e    = mb.expand_dims(x=mb.expand_dims(x=gx, axes=[1]), axes=[2])  # (B,1,1)
        gy_e    = mb.expand_dims(x=mb.expand_dims(x=gy, axes=[1]), axes=[2])  # (B,1,1)

        coords_x = mb.add(x=gx_e, y=mb.mul(x=cos_e, y=steps_e))   # (B, R, S)
        coords_y = mb.add(x=gy_e, y=mb.mul(x=sin_e, y=steps_e))   # (B, R, S)

        # ── wrap to grid bounds using floor-div trick (handles negatives) ────
        # rx = (coords_x + GW) - floor((coords_x + GW) / GW) * GW
        def _wrap(coords, gw_or_gh_f):
            shifted    = mb.add(x=coords, y=gw_or_gh_f)
            floor_div  = mb.floor(x=mb.real_div(x=shifted, y=gw_or_gh_f))
            return mb.sub(x=shifted, y=mb.mul(x=floor_div, y=gw_or_gh_f))

        rx_f = _wrap(coords_x, gw_f)   # (B, R, S) float in [0, GW)
        ry_f = _wrap(coords_y, gh_f)   # (B, R, S) float in [0, GH)

        # ── flat grid index ── ry * GW + rx ─────────────────────────────────
        flat_f   = mb.add(x=mb.mul(x=ry_f, y=gw_f), y=rx_f)   # (B, R, S) float
        flat_i   = mb.cast(x=mb.reshape(x=flat_f, shape=[FLAT]), dtype='int32')   # (FLAT,)

        # ── gather food and org values from flat grids ───────────────────────
        food_flat = mb.reshape(x=food_grid, shape=[GRID_FLAT])
        org_flat  = mb.reshape(x=org_grid,  shape=[GRID_FLAT])

        food_raw = mb.gather(x=food_flat, indices=flat_i, axis=0)   # (FLAT,)
        org_raw  = mb.gather(x=org_flat,  indices=flat_i, axis=0)   # (FLAT,)

        # threshold → binary hits (B, R, S)
        food_hits = mb.cast(
            x=mb.greater(x=mb.reshape(x=food_raw, shape=[B, R, S]), y=zero),
            dtype='fp32')
        org_hits  = mb.cast(
            x=mb.greater(x=mb.reshape(x=org_raw,  shape=[B, R, S]), y=zero),
            dtype='fp32')

        # ── near-body exclusion for org_hits ─────────────────────────────────
        # exclude step < size_pix (wight's own body)
        # steps_e (1,1,S) vs size_pix_e (B,1,1)
        size_pix_e = mb.expand_dims(x=mb.expand_dims(x=size_pix, axes=[1]), axes=[2])  # (B,1,1)
        far_enough = mb.cast(
            x=mb.greater_equal(x=steps_e, y=size_pix_e),
            dtype='fp32')   # (B, 1, S) broadcasts to (B, R, S)
        org_hits = mb.mul(x=org_hits, y=far_enough)

        # ── first-hit sensor signal ───────────────────────────────────────────
        ray_len_e = mb.expand_dims(x=ray_len_pix, axes=[1])   # (B, 1) → (B, R)

        def _sensor(hits):
            has_hit  = mb.reduce_max(x=hits, axes=[-1], keep_dims=False)  # (B, R) 1 or 0
            hit_idx  = mb.cast(
                x=mb.reduce_argmax(x=hits, axis=-1, keep_dims=False),
                dtype='fp32')                                           # (B, R)
            hit_step = mb.add(x=hit_idx, y=one)                           # 1-indexed
            dist     = mb.select(
                cond=mb.greater(x=has_hit, y=zero),
                a=hit_step, b=ray_len_e)
            return mb.sub(x=one,
                          y=mb.clip(x=mb.real_div(x=dist, y=ray_len_e),
                                    alpha=0.0, beta=1.0))

        food_sig = _sensor(food_hits)   # (B, R)
        org_sig  = _sensor(org_hits)    # (B, R)

        # ── interleave → inputs (B, N_INPUTS) ────────────────────────────────
        # want: [food₀, org₀, food₁, org₁, …, energy]
        food_e2 = mb.expand_dims(x=food_sig, axes=[2])   # (B, R, 1)
        org_e2  = mb.expand_dims(x=org_sig,  axes=[2])   # (B, R, 1)
        paired  = mb.concat(values=[food_e2, org_e2], axis=2)   # (B, R, 2)
        ray_inp = mb.reshape(x=paired, shape=[B, R * 2])        # (B, R*2)
        ray_inp = mb.mul(x=ray_inp, y=n_rays_mask)              # zero inactive ray channels
        nrg_e   = mb.expand_dims(x=energy_frac, axes=[1])       # (B, 1)
        inputs  = mb.concat(values=[ray_inp, nrg_e], axis=1)    # (B, N_INPUTS)

        # ── Elman RNN ─────────────────────────────────────────────────────────
        x_e   = mb.expand_dims(x=inputs, axes=[-2])                          # (B, 1, N_INPUTS)
        h_raw = mb.squeeze(x=mb.matmul(x=x_e, y=W1), axes=[-2])             # (B, N_HIDDEN)
        h_new = mb.tanh(x=mb.add(x=h_raw, y=h_prev), name='h_new_raw')       # (B, N_HIDDEN)
        
        # Apply biological capacity mask
        h_new = mb.mul(x=h_new, y=mask, name='h_new')                        # (B, N_HIDDEN)

        h_e2  = mb.expand_dims(x=h_new, axes=[-2])                           # (B, 1, N_HIDDEN)
        out   = mb.tanh(x=mb.squeeze(x=mb.matmul(x=h_e2, y=W2), axes=[-2]),
                        name='out')                                           # (B, N_OUTPUTS)
        return h_new, out

    model = ct.convert(prog,
                       compute_units=ct.ComputeUnit.ALL,
                       minimum_deployment_target=ct.target.macOS13)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(str(MODEL_PATH))
    META_PATH.write_text(json.dumps({
        "max_pop": MAX_POP, "n_rays": N_RAYS, "max_steps": MAX_STEPS,
        "gw": GW, "gh": GH, "n_hid": N_HIDDEN, "n_out": N_OUTPUTS,
        "has_nrays_mask": True,
    }))
    return ct.models.MLModel(str(MODEL_PATH), compute_units=ct.ComputeUnit.ALL)


# ── public inference ──────────────────────────────────────────────────────────

def run_sense_brain(pop: dict, food_grid: np.ndarray, org_grid: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Fused sensing + brain for all wights.
    food_grid / org_grid: (GH, GW) float32 from paint_grid.
    Returns (h_new, out) identical to run_brain.

    Always routes to the fused GPU model when available — O(1) wall-clock
    regardless of population size (single MAX_POP kernel dispatch).
    Falls back to numpy sense + CoreML brain only if the model failed to load.
    """
    if _use_coreml and _model is not None:
        return _predict(pop, food_grid, org_grid)
    from sim.sensing import sense
    from brain.coreml_brain import run_brain
    inputs = sense(pop, np.stack([food_grid, org_grid], axis=0))
    return run_brain(inputs, pop['W1'], pop['W2'], pop['h_state'])


def _predict(pop, food_grid, org_grid):
    n           = len(pop['x'])
    energy_max  = ENERGY_MAX_SCALE * pop['size'] ** 2
    energy_frac = (pop['energy'] / energy_max).astype(np.float32)
    half_fov    = (pop['fov'] * 0.5).astype(np.float32)
    ray_len_pix = np.maximum(pop['ray_len'] * GRID_SCALE, 1.0).astype(np.float32)
    size_pix    = np.ceil(pop['size'] * GRID_SCALE).astype(np.float32)

    def _pad1(arr, fill=0.0):
        if n == MAX_POP:
            return arr.astype(np.float32)
        pad = np.full((MAX_POP - n,), fill, dtype=np.float32)
        return np.concatenate([arr.astype(np.float32), pad])

    def _padn(arr):
        if n == MAX_POP:
            return arr.astype(np.float32)
        pad = np.zeros((MAX_POP - n,) + arr.shape[1:], dtype=np.float32)
        return np.concatenate([arr.astype(np.float32), pad])
        
    mask = (np.arange(N_HIDDEN) < pop['active_neurons'][:, None]).astype(np.float32)

    # per-organism ray activity mask: rays 0..n_rays[i]-1 active
    ray_active   = (np.arange(N_RAYS)[None, :] < pop['n_rays'][:, None]).astype(np.float32)  # (n, R)
    n_rays_mask  = np.repeat(ray_active, 2, axis=1)  # (n, R*2): food/org interleaved

    r = _model.predict({
        'x_pos':       _pad1(pop['x']),
        'y_pos':       _pad1(pop['y']),
        'angle':       _pad1(pop['angle']),
        'half_fov':    _pad1(half_fov),
        'ray_len_pix': _pad1(ray_len_pix, fill=1.0),
        'size_pix':    _pad1(size_pix,    fill=1.0),
        'energy_frac': _pad1(energy_frac),
        'food_grid':   food_grid.astype(np.float32),
        'org_grid':    org_grid.astype(np.float32),
        'W1':          _padn(pop['W1']),
        'W2':          _padn(pop['W2']),
        'h_prev':      _padn(pop['h_state']),
        'mask':        _padn(mask),
        'n_rays_mask': _padn(n_rays_mask),
    })
    return r['h_new'][:n], r['out'][:n]
