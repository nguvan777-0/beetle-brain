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
import json, threading, time
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
    if not _HAS_CT:
        return False

    threading.Thread(target=_load_or_compile, daemon=True).start()
    return False  # numpy fallback until thread finishes


def _load_or_compile():
    global _model, _use_coreml

    if MODEL_PATH.exists() and META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text())
            if (meta.get("max_pop") == MAX_POP and
                    meta.get("n_rays") == N_RAYS and
                    meta.get("max_steps") == MAX_STEPS and
                    meta.get("gw") == GW and meta.get("gh") == GH and
                    meta.get("n_hid") == N_HIDDEN and meta.get("n_out") == N_OUTPUTS and
                    meta.get("has_nrays_mask") is True and meta.get("has_wh") == True and
                    meta.get("has_rgb") == True and meta.get("has_bias") == True):
                t0 = time.time()
                model = ct.models.MLModel(str(MODEL_PATH), compute_units=ct.ComputeUnit.CPU_AND_GPU)
                _model = model
                _use_coreml = True
                print(f"[SenseBrain] ready ({time.time()-t0:.1f}s)")
                return
        except Exception as e:
            print(f"[SenseBrain] cache load failed ({e}), rebuilding...")

    print(f"[SenseBrain] Compiling fused sense+brain "
          f"(pop={MAX_POP}, rays={N_RAYS}, steps={MAX_STEPS}, "
          f"grid={GW}×{GH}, {N_INPUTS}→{N_HIDDEN}→{N_OUTPUTS})...",
          end="", flush=True)
    t0 = time.time()
    try:
        model = _compile()
        _model      = model
        _use_coreml = True
        print(f" done ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f" FAILED: {e}")


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
        mb.TensorSpec(shape=(GH, GW)),           # r_grid
        mb.TensorSpec(shape=(GH, GW)),           # g_grid
        mb.TensorSpec(shape=(GH, GW)),           # b_grid
        mb.TensorSpec(shape=(B, N_INPUTS, N_HIDDEN)),  # W1
        mb.TensorSpec(shape=(B, N_HIDDEN, N_OUTPUTS)), # W2
        mb.TensorSpec(shape=(B, N_HIDDEN, N_HIDDEN)),  # Wh
        mb.TensorSpec(shape=(B, N_HIDDEN)),      # h_prev
        mb.TensorSpec(shape=(B, N_HIDDEN)),      # mask
        mb.TensorSpec(shape=(B, N_RAYS * 5)),    # n_rays_mask: 1 for active ray channels, 0 for inactive
        mb.TensorSpec(shape=(B, N_HIDDEN)),      # b1
        mb.TensorSpec(shape=(B, N_OUTPUTS)),     # b2
    ])
    def prog(x_pos, y_pos, angle, half_fov,
             ray_len_pix, size_pix, energy_frac,
             food_grid, r_grid, g_grid, b_grid, W1, W2, Wh, h_prev, mask, n_rays_mask, b1, b2):

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

        # ── gather food and rgb values from flat grids ───────────────────────
        food_flat = mb.reshape(x=food_grid, shape=[GRID_FLAT])
        r_flat    = mb.reshape(x=r_grid,    shape=[GRID_FLAT])
        g_flat    = mb.reshape(x=g_grid,    shape=[GRID_FLAT])
        b_flat    = mb.reshape(x=b_grid,    shape=[GRID_FLAT])

        food_raw = mb.reshape(x=mb.gather(x=food_flat, indices=flat_i, axis=0), shape=[B, R, S])
        r_raw    = mb.reshape(x=mb.gather(x=r_flat,    indices=flat_i, axis=0), shape=[B, R, S])
        g_raw    = mb.reshape(x=mb.gather(x=g_flat,    indices=flat_i, axis=0), shape=[B, R, S])
        b_raw    = mb.reshape(x=mb.gather(x=b_flat,    indices=flat_i, axis=0), shape=[B, R, S])

        # food hits: binary
        food_hits = mb.cast(x=mb.greater(x=food_raw, y=zero), dtype='fp32')

        # org hits: any color channel nonzero
        rgb_sum  = mb.add(x=mb.add(x=r_raw, y=g_raw), y=b_raw)
        org_hits = mb.cast(x=mb.greater(x=rgb_sum, y=zero), dtype='fp32')

        # ── near-body exclusion for org_hits ─────────────────────────────────
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

        food_sig     = _sensor(food_hits)   # (B, R)
        org_dist_sig = _sensor(org_hits)    # (B, R)

        # ── color at first org hit ────────────────────────────────────────────
        has_hit   = mb.reduce_max(x=org_hits, axes=[-1], keep_dims=False)         # (B, R)
        hit_idx_f = mb.cast(x=mb.reduce_argmax(x=org_hits, axis=-1, keep_dims=False), dtype='fp32')  # (B, R)

        # one-hot at first hit step: compare step indices (1,1,S) with hit index (B,R,1)
        step_ids_k = mb.const(val=np.arange(S, dtype=np.float32))                 # (S,)
        step_ids_e = mb.expand_dims(x=mb.expand_dims(x=step_ids_k, axes=[0]), axes=[0])  # (1,1,S)
        hit_idx_e  = mb.expand_dims(x=hit_idx_f, axes=[-1])                       # (B,R,1)
        at_hit     = mb.cast(x=mb.equal(x=step_ids_e, y=hit_idx_e), dtype='fp32') # (B,R,S)

        has_hit_e  = mb.expand_dims(x=has_hit, axes=[-1])                         # (B,R,1)
        hit_mask   = mb.mul(x=at_hit, y=has_hit_e)                                # (B,R,S)

        inv255 = mb.const(val=np.float32(1.0 / 255.0))
        r_sig = mb.mul(x=mb.reduce_sum(x=mb.mul(x=r_raw, y=hit_mask), axes=[-1], keep_dims=False), y=inv255)
        g_sig = mb.mul(x=mb.reduce_sum(x=mb.mul(x=g_raw, y=hit_mask), axes=[-1], keep_dims=False), y=inv255)
        b_sig = mb.mul(x=mb.reduce_sum(x=mb.mul(x=b_raw, y=hit_mask), axes=[-1], keep_dims=False), y=inv255)

        # ── interleave → inputs (B, N_INPUTS) ────────────────────────────────
        # layout per ray: [food_dist, org_dist, r, g, b]
        food_e2     = mb.expand_dims(x=food_sig,     axes=[2])   # (B, R, 1)
        org_dist_e2 = mb.expand_dims(x=org_dist_sig, axes=[2])
        r_e2        = mb.expand_dims(x=r_sig,        axes=[2])
        g_e2        = mb.expand_dims(x=g_sig,        axes=[2])
        b_e2        = mb.expand_dims(x=b_sig,        axes=[2])
        quintet = mb.concat(values=[food_e2, org_dist_e2, r_e2, g_e2, b_e2], axis=2)  # (B, R, 5)
        ray_inp = mb.reshape(x=quintet, shape=[B, R * 5])                              # (B, R*5)
        ray_inp = mb.mul(x=ray_inp, y=n_rays_mask)              # zero inactive ray channels
        nrg_e   = mb.expand_dims(x=energy_frac, axes=[1])       # (B, 1)
        inputs  = mb.concat(values=[ray_inp, nrg_e], axis=1)    # (B, N_INPUTS)

        # ── Elman RNN ─────────────────────────────────────────────────────────
        x_e    = mb.expand_dims(x=inputs, axes=[-2])                                      # (B, 1, N_INPUTS)
        h_raw  = mb.squeeze(x=mb.matmul(x=x_e, y=W1), axes=[-2])                         # (B, N_HIDDEN)
        hp_e   = mb.expand_dims(x=h_prev, axes=[-2])                                      # (B, 1, N_HIDDEN)
        wh_raw = mb.squeeze(x=mb.matmul(x=hp_e, y=Wh), axes=[-2])                        # (B, N_HIDDEN)
        h_new  = mb.tanh(x=mb.add(x=mb.add(x=h_raw, y=wh_raw), y=b1), name='h_new_raw') # (B, N_HIDDEN)

        # Apply biological capacity mask
        h_new = mb.mul(x=h_new, y=mask, name='h_new')                                    # (B, N_HIDDEN)

        h_e2    = mb.expand_dims(x=h_new, axes=[-2])                                      # (B, 1, N_HIDDEN)
        out_raw = mb.squeeze(x=mb.matmul(x=h_e2, y=W2), axes=[-2])                       # (B, N_OUTPUTS)
        out     = mb.tanh(x=mb.add(x=out_raw, y=b2), name='out')                         # (B, N_OUTPUTS)
        return h_new, out

    model = ct.convert(prog,
                       compute_units=ct.ComputeUnit.CPU_AND_GPU,
                       minimum_deployment_target=ct.target.macOS13)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(str(MODEL_PATH))
    META_PATH.write_text(json.dumps({
        "max_pop": MAX_POP, "n_rays": N_RAYS, "max_steps": MAX_STEPS,
        "gw": GW, "gh": GH, "n_hid": N_HIDDEN, "n_out": N_OUTPUTS,
        "has_nrays_mask": True, "has_wh": True, "has_rgb": True, "has_bias": True,
    }))
    return ct.models.MLModel(str(MODEL_PATH), compute_units=ct.ComputeUnit.CPU_AND_GPU)


# ── public inference ──────────────────────────────────────────────────────────

def run_sense_brain(pop: dict, food_grid: np.ndarray,
                    r_grid: np.ndarray, g_grid: np.ndarray, b_grid: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Fused sensing + brain for all wights.
    food_grid / r_grid / g_grid / b_grid: (GH, GW) float32 from paint_grid.
    Returns (h_new, out) identical to run_brain.

    Always routes to the fused GPU model when available — O(1) wall-clock
    regardless of population size (single MAX_POP kernel dispatch).
    Falls back to numpy sense + CoreML brain only if the model failed to load.
    """
    if _use_coreml and _model is not None:
        return _predict(pop, food_grid, r_grid, g_grid, b_grid)
    from sim.sensing import sense
    from brain.coreml_brain import run_brain
    inputs = sense(pop, np.stack([food_grid, r_grid, g_grid, b_grid], axis=0))
    return run_brain(inputs, pop['W1'], pop['W2'], pop['Wh'], pop['b1'], pop['b2'], pop['h_state'])


def _predict(pop, food_grid, r_grid, g_grid, b_grid):
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

    # per-organism ray activity mask: rays 0..n_rays[i]-1 active (5 channels per ray)
    ray_active   = (np.arange(N_RAYS)[None, :] < pop['n_rays'][:, None]).astype(np.float32)  # (n, R)
    n_rays_mask  = np.repeat(ray_active, 5, axis=1)  # (n, R*5)

    r = _model.predict({
        'x_pos':       _pad1(pop['x']),
        'y_pos':       _pad1(pop['y']),
        'angle':       _pad1(pop['angle']),
        'half_fov':    _pad1(half_fov),
        'ray_len_pix': _pad1(ray_len_pix, fill=1.0),
        'size_pix':    _pad1(size_pix,    fill=1.0),
        'energy_frac': _pad1(energy_frac),
        'food_grid':   food_grid.astype(np.float32),
        'r_grid':      r_grid.astype(np.float32),
        'g_grid':      g_grid.astype(np.float32),
        'b_grid':      b_grid.astype(np.float32),
        'W1':          _padn(pop['W1']),
        'W2':          _padn(pop['W2']),
        'Wh':          _padn(pop['Wh']),
        'h_prev':      _padn(pop['h_state']),
        'mask':        _padn(mask),
        'n_rays_mask': _padn(n_rays_mask),
        'b1':          _padn(pop['b1']),
        'b2':          _padn(pop['b2']),
    })
    return r['h_new'][:n], r['out'][:n]
