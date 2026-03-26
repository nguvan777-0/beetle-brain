"""
brain/coreml_sense_brain.py — fused sensing + recurrent brain via CoreML
=========================================================================
Compiles one CoreML program that:
  1. Paints organism / food grids on GPU via scatter_nd + max-pool dilation
  2. Ray-marches the world grid (sensing) — all N wights in parallel on GPU
  3. Runs the Elman RNN forward pass

Running all ops in a single GPU dispatch eliminates every CPU round-trip.
Grid painting happens inside the graph: organism positions and colors are
passed as (B,) vectors; scatter_nd writes single pixels; a max-pool
dilation expands them into size-proportional disks. No CPU scatter work.

Fallback: numpy sense + CoreML brain when compilation fails.
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
    N_RAYS, N_INPUTS, N_HIDDEN, N_OUTPUTS, MAX_POP, ENERGY_MAX_SCALE, N_FOOD,
)
from sim.grid.constants import (
    GW, GH, GRID_SCALE, MAX_STEPS, _STEPS, _RAY_OFFSETS,
    DILATION_R_PIX, DILATION_K,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH   = PROJECT_ROOT / "build" / "sense_brain.mlpackage"
META_PATH    = PROJECT_ROOT / "build" / "sense_brain_meta.json"

_model      = None
_use_coreml = False


def init_sense_brain() -> bool:
    if not _HAS_CT:
        return False
    threading.Thread(target=_load_or_compile, daemon=True).start()
    return False


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
                    meta.get("has_nrays_mask") is True and meta.get("has_wh") is True and
                    meta.get("has_rgb") is True and meta.get("has_bias") is True and
                    meta.get("has_gpu_paint") is True and meta.get("n_food") == N_FOOD):
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
        import traceback; traceback.print_exc()


def _compile():
    B, R, S   = MAX_POP, N_RAYS, MAX_STEPS
    FLAT      = B * R * S
    GRID_FLAT = GH * GW

    steps_c   = np.array(_STEPS,       dtype=np.float32)
    offsets_c = np.array(_RAY_OFFSETS, dtype=np.float32)

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(B,)),                     # x_pos
        mb.TensorSpec(shape=(B,)),                     # y_pos
        mb.TensorSpec(shape=(B,)),                     # angle
        mb.TensorSpec(shape=(B,)),                     # half_fov
        mb.TensorSpec(shape=(B,)),                     # ray_len_pix
        mb.TensorSpec(shape=(B,)),                     # size_pix
        mb.TensorSpec(shape=(B,)),                     # energy_frac
        mb.TensorSpec(shape=(B,)),                     # org_r  — [1..255] float
        mb.TensorSpec(shape=(B,)),                     # org_g
        mb.TensorSpec(shape=(B,)),                     # org_b
        mb.TensorSpec(shape=(N_FOOD,)),                # food_flat_f — flat grid index as float
        mb.TensorSpec(shape=(N_FOOD,)),                # food_val — 1.0=real, 0.0=padding
        mb.TensorSpec(shape=(B, N_INPUTS, N_HIDDEN)),  # W1
        mb.TensorSpec(shape=(B, N_HIDDEN, N_OUTPUTS)), # W2
        mb.TensorSpec(shape=(B, N_HIDDEN, N_HIDDEN)),  # Wh
        mb.TensorSpec(shape=(B, N_HIDDEN)),            # h_prev
        mb.TensorSpec(shape=(B, N_HIDDEN)),            # mask
        mb.TensorSpec(shape=(B, N_RAYS * 5)),          # n_rays_mask
        mb.TensorSpec(shape=(B, N_HIDDEN)),            # b1
        mb.TensorSpec(shape=(B, N_OUTPUTS)),           # b2
    ])
    def prog(x_pos, y_pos, angle, half_fov,
             ray_len_pix, size_pix, energy_frac,
             org_r, org_g, org_b,
             food_flat_f, food_val,
             W1, W2, Wh, h_prev, mask, n_rays_mask, b1, b2):

        steps_k   = mb.const(val=steps_c)
        offsets_k = mb.const(val=offsets_c)
        zero      = mb.const(val=np.float32(0.0))
        one       = mb.const(val=np.float32(1.0))
        gs        = mb.const(val=np.float32(GRID_SCALE))
        gw_f      = mb.const(val=np.float32(GW))
        gh_f      = mb.const(val=np.float32(GH))
        gw_i      = mb.const(val=np.int32(GW))

        # ── world → grid positions ────────────────────────────────────────────
        gx = mb.mul(x=x_pos, y=gs)    # (B,)
        gy = mb.mul(x=y_pos, y=gs)    # (B,)

        # ── paint color grids via scatter_nd + max-pool dilation ──────────────
        # Each organism writes a single pixel; max-pool expands to a disk.
        gx_c = mb.cast(x=mb.clip(x=gx, alpha=0.0, beta=float(GW - 1)), dtype='int32')
        gy_c = mb.cast(x=mb.clip(x=gy, alpha=0.0, beta=float(GH - 1)), dtype='int32')
        flat_org   = mb.add(x=mb.mul(x=gy_c, y=gw_i), y=gx_c)   # (B,) int32 flat index
        flat_org_e = mb.expand_dims(x=flat_org, axes=[-1])        # (B, 1)

        z = mb.const(val=np.zeros(GRID_FLAT, dtype=np.float32))
        r_scattered = mb.scatter_nd(data=z, indices=flat_org_e, updates=org_r, mode='update')
        g_scattered = mb.scatter_nd(data=z, indices=flat_org_e, updates=org_g, mode='update')
        b_scattered = mb.scatter_nd(data=z, indices=flat_org_e, updates=org_b, mode='update')

        def _dilate(flat_ch):
            """Expand single-pixel organism dots into size-proportional disks.
            Cascades DILATION_R_PIX passes of 3×3 max-pool (ANE-native kernel size)
            instead of one large kernel, keeping all ops on the ANE/GPU.
            """
            g = mb.reshape(x=flat_ch, shape=[1, 1, GH, GW])
            for _ in range(DILATION_R_PIX):
                g = mb.max_pool(
                    x=mb.pad(x=g, pad=[0, 0, 0, 0, 1, 1, 1, 1],
                              mode='constant', constant_val=np.float32(0.0)),
                    kernel_sizes=[3, 3], strides=[1, 1], pad_type='valid')
            return mb.reshape(x=g, shape=[GH, GW])

        r_grid = _dilate(r_scattered)
        g_grid = _dilate(g_scattered)
        b_grid = _dilate(b_scattered)

        # ── paint food grid via scatter_nd (no dilation — food is a point) ────
        food_flat_i  = mb.cast(x=food_flat_f, dtype='int32')
        food_flat_ie = mb.expand_dims(x=food_flat_i, axes=[-1])   # (N_FOOD, 1)
        zf = mb.const(val=np.zeros(GRID_FLAT, dtype=np.float32))
        food_grid = mb.reshape(
            x=mb.scatter_nd(data=zf, indices=food_flat_ie, updates=food_val, mode='update'),
            shape=[GH, GW])

        # ── ray angles: (B,) × (R,) → (B, R) ────────────────────────────────
        angle_e    = mb.expand_dims(x=angle,     axes=[1])
        hfov_e     = mb.expand_dims(x=half_fov,  axes=[1])
        offsets_e  = mb.expand_dims(x=offsets_k, axes=[0])
        ray_angles = mb.add(x=angle_e, y=mb.mul(x=offsets_e, y=hfov_e))

        # ── ray directions ────────────────────────────────────────────────────
        cos_a = mb.cos(x=ray_angles)
        sin_a = mb.sin(x=ray_angles)

        # ── sample coordinates: (B, R, S) ─────────────────────────────────────
        steps_e = mb.expand_dims(x=mb.expand_dims(x=steps_k, axes=[0]), axes=[0])
        cos_e   = mb.expand_dims(x=cos_a, axes=[2])
        sin_e   = mb.expand_dims(x=sin_a, axes=[2])
        gx_e    = mb.expand_dims(x=mb.expand_dims(x=gx, axes=[1]), axes=[2])
        gy_e    = mb.expand_dims(x=mb.expand_dims(x=gy, axes=[1]), axes=[2])

        coords_x = mb.add(x=gx_e, y=mb.mul(x=cos_e, y=steps_e))
        coords_y = mb.add(x=gy_e, y=mb.mul(x=sin_e, y=steps_e))

        # ── wrap to grid bounds ───────────────────────────────────────────────
        def _wrap(coords, size_f):
            shifted   = mb.add(x=coords, y=size_f)
            floor_div = mb.floor(x=mb.real_div(x=shifted, y=size_f))
            return mb.sub(x=shifted, y=mb.mul(x=floor_div, y=size_f))

        rx_f = _wrap(coords_x, gw_f)
        ry_f = _wrap(coords_y, gh_f)

        # ── flat grid index ───────────────────────────────────────────────────
        flat_f = mb.add(x=mb.mul(x=ry_f, y=gw_f), y=rx_f)
        flat_i = mb.cast(x=mb.reshape(x=flat_f, shape=[FLAT]), dtype='int32')

        # ── gather food and rgb from grids ────────────────────────────────────
        food_flat_ray = mb.reshape(x=food_grid, shape=[GRID_FLAT])
        r_flat        = mb.reshape(x=r_grid,    shape=[GRID_FLAT])
        g_flat        = mb.reshape(x=g_grid,    shape=[GRID_FLAT])
        b_flat        = mb.reshape(x=b_grid,    shape=[GRID_FLAT])

        food_raw = mb.reshape(x=mb.gather(x=food_flat_ray, indices=flat_i, axis=0), shape=[B, R, S])
        r_raw    = mb.reshape(x=mb.gather(x=r_flat,        indices=flat_i, axis=0), shape=[B, R, S])
        g_raw    = mb.reshape(x=mb.gather(x=g_flat,        indices=flat_i, axis=0), shape=[B, R, S])
        b_raw    = mb.reshape(x=mb.gather(x=b_flat,        indices=flat_i, axis=0), shape=[B, R, S])

        # food hits: binary
        food_hits = mb.cast(x=mb.greater(x=food_raw, y=zero), dtype='fp32')

        # org hits: any color channel nonzero
        rgb_sum  = mb.add(x=mb.add(x=r_raw, y=g_raw), y=b_raw)
        org_hits = mb.cast(x=mb.greater(x=rgb_sum, y=zero), dtype='fp32')

        # ── near-body exclusion ───────────────────────────────────────────────
        # Use max(size_pix, DILATION_R_PIX) so organism doesn't see its own dilated disk
        dil_r_k    = mb.const(val=np.float32(DILATION_R_PIX))
        excl       = mb.maximum(x=size_pix, y=dil_r_k)
        excl_e     = mb.expand_dims(x=mb.expand_dims(x=excl, axes=[1]), axes=[2])
        far_enough = mb.cast(x=mb.greater_equal(x=steps_e, y=excl_e), dtype='fp32')
        org_hits   = mb.mul(x=org_hits, y=far_enough)

        # ── first-hit sensor signal ───────────────────────────────────────────
        ray_len_e = mb.expand_dims(x=ray_len_pix, axes=[1])

        def _sensor(hits):
            has_hit  = mb.reduce_max(x=hits, axes=[-1], keep_dims=False)
            hit_idx  = mb.cast(x=mb.reduce_argmax(x=hits, axis=-1, keep_dims=False), dtype='fp32')
            hit_step = mb.add(x=hit_idx, y=one)
            dist     = mb.select(cond=mb.greater(x=has_hit, y=zero), a=hit_step, b=ray_len_e)
            return mb.sub(x=one, y=mb.clip(x=mb.real_div(x=dist, y=ray_len_e), alpha=0.0, beta=1.0))

        food_sig     = _sensor(food_hits)
        org_dist_sig = _sensor(org_hits)

        # ── color at first org hit ────────────────────────────────────────────
        has_hit   = mb.reduce_max(x=org_hits, axes=[-1], keep_dims=False)
        hit_idx_f = mb.cast(x=mb.reduce_argmax(x=org_hits, axis=-1, keep_dims=False), dtype='fp32')

        step_ids_k = mb.const(val=np.arange(S, dtype=np.float32))
        step_ids_e = mb.expand_dims(x=mb.expand_dims(x=step_ids_k, axes=[0]), axes=[0])
        hit_idx_e  = mb.expand_dims(x=hit_idx_f, axes=[-1])
        at_hit     = mb.cast(x=mb.equal(x=step_ids_e, y=hit_idx_e), dtype='fp32')
        has_hit_e  = mb.expand_dims(x=has_hit, axes=[-1])
        hit_mask   = mb.mul(x=at_hit, y=has_hit_e)

        inv255 = mb.const(val=np.float32(1.0 / 255.0))
        r_sig = mb.mul(x=mb.reduce_sum(x=mb.mul(x=r_raw, y=hit_mask), axes=[-1], keep_dims=False), y=inv255)
        g_sig = mb.mul(x=mb.reduce_sum(x=mb.mul(x=g_raw, y=hit_mask), axes=[-1], keep_dims=False), y=inv255)
        b_sig = mb.mul(x=mb.reduce_sum(x=mb.mul(x=b_raw, y=hit_mask), axes=[-1], keep_dims=False), y=inv255)

        # ── interleave → inputs (B, N_INPUTS) ────────────────────────────────
        food_e2     = mb.expand_dims(x=food_sig,     axes=[2])
        org_dist_e2 = mb.expand_dims(x=org_dist_sig, axes=[2])
        r_e2        = mb.expand_dims(x=r_sig,        axes=[2])
        g_e2        = mb.expand_dims(x=g_sig,        axes=[2])
        b_e2        = mb.expand_dims(x=b_sig,        axes=[2])
        quintet = mb.concat(values=[food_e2, org_dist_e2, r_e2, g_e2, b_e2], axis=2)
        ray_inp = mb.reshape(x=quintet, shape=[B, R * 5])
        ray_inp = mb.mul(x=ray_inp, y=n_rays_mask)
        nrg_e   = mb.expand_dims(x=energy_frac, axes=[1])
        inputs  = mb.concat(values=[ray_inp, nrg_e], axis=1)

        # ── Elman RNN ─────────────────────────────────────────────────────────
        x_e    = mb.expand_dims(x=inputs, axes=[-2])
        h_raw  = mb.squeeze(x=mb.matmul(x=x_e, y=W1), axes=[-2])
        hp_e   = mb.expand_dims(x=h_prev, axes=[-2])
        wh_raw = mb.squeeze(x=mb.matmul(x=hp_e, y=Wh), axes=[-2])
        h_new  = mb.tanh(x=mb.add(x=mb.add(x=h_raw, y=wh_raw), y=b1), name='h_new_raw')
        h_new  = mb.mul(x=h_new, y=mask, name='h_new')
        h_e2    = mb.expand_dims(x=h_new, axes=[-2])
        out_raw = mb.squeeze(x=mb.matmul(x=h_e2, y=W2), axes=[-2])
        out     = mb.tanh(x=mb.add(x=out_raw, y=b2), name='out')
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
        "has_gpu_paint": True, "n_food": N_FOOD,
    }))
    return ct.models.MLModel(str(MODEL_PATH), compute_units=ct.ComputeUnit.CPU_AND_GPU)


# ── public inference ──────────────────────────────────────────────────────────

def run_sense_brain(pop: dict, food: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fused sensing + brain for all wights.
    food: (F, 2) float32 world positions from world['food'].
    Returns (h_new, out).

    Grid painting (scatter + dilation) happens inside the CoreML graph.
    CPU only builds the (B,) position/color vectors — no grid scatter work.
    Falls back to numpy sense + CoreML brain if the fused model failed to load.
    """
    if _use_coreml and _model is not None:
        return _predict(pop, food)
    # numpy fallback: build grids on CPU for legacy sense()
    from sim.grid.constants import GW as _GW, GH as _GH, GRID_SCALE as _GS
    from sim.sensing import sense
    from brain.coreml_brain import run_brain
    F    = len(food)
    grid = np.zeros((4, _GH, _GW), dtype=np.uint8)
    if F > 0:
        fy = np.clip((food[:, 1] * _GS).astype(np.int32), 0, _GH - 1)
        fx = np.clip((food[:, 0] * _GS).astype(np.int32), 0, _GW - 1)
        grid[0, fy, fx] = 1
    oy = np.clip((pop['y'] * _GS).astype(np.int32), 0, _GH - 1)
    ox = np.clip((pop['x'] * _GS).astype(np.int32), 0, _GW - 1)
    grid[1, oy, ox] = np.clip(pop['r'], 1, 255).astype(np.uint8)
    grid[2, oy, ox] = np.clip(pop['g'], 1, 255).astype(np.uint8)
    grid[3, oy, ox] = np.clip(pop['b'], 1, 255).astype(np.uint8)
    inputs = sense(pop, grid)
    return run_brain(inputs, pop['W1'], pop['W2'], pop['Wh'], pop['b1'], pop['b2'], pop['h_state'])


def _predict(pop, food):
    n           = len(pop['x'])
    energy_max  = ENERGY_MAX_SCALE * pop['size'] ** 2
    energy_frac = (pop['energy'] / energy_max).astype(np.float32)
    half_fov    = (pop['fov'] * 0.5).astype(np.float32)
    ray_len_pix = np.maximum(pop['ray_len'] * GRID_SCALE, 1.0).astype(np.float32)
    size_pix    = np.ceil(pop['size'] * GRID_SCALE).astype(np.float32)

    org_r = np.clip(pop['r'], 1, 255).astype(np.float32)
    org_g = np.clip(pop['g'], 1, 255).astype(np.float32)
    org_b = np.clip(pop['b'], 1, 255).astype(np.float32)

    # Food: flat grid indices + validity mask, padded to N_FOOD
    F         = len(food)
    food_flat = np.zeros(N_FOOD, dtype=np.float32)
    food_val  = np.zeros(N_FOOD, dtype=np.float32)
    if F > 0:
        fy = np.clip((food[:, 1] * GRID_SCALE).astype(np.int32), 0, GH - 1)
        fx = np.clip((food[:, 0] * GRID_SCALE).astype(np.int32), 0, GW - 1)
        food_flat[:F] = (fy * GW + fx).astype(np.float32)
        food_val[:F]  = 1.0

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

    mask        = (np.arange(N_HIDDEN) < pop['active_neurons'][:, None]).astype(np.float32)
    ray_active  = (np.arange(N_RAYS)[None, :] < pop['n_rays'][:, None]).astype(np.float32)
    n_rays_mask = np.repeat(ray_active, 5, axis=1)

    r = _model.predict({
        'x_pos':       _pad1(pop['x']),
        'y_pos':       _pad1(pop['y']),
        'angle':       _pad1(pop['angle']),
        'half_fov':    _pad1(half_fov),
        'ray_len_pix': _pad1(ray_len_pix, fill=1.0),
        'size_pix':    _pad1(size_pix,    fill=1.0),
        'energy_frac': _pad1(energy_frac),
        'org_r':       _pad1(org_r,       fill=0.0),
        'org_g':       _pad1(org_g,       fill=0.0),
        'org_b':       _pad1(org_b,       fill=0.0),
        'food_flat_f': food_flat,
        'food_val':    food_val,
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
