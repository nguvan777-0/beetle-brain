"""
brain/coreml_sense_brain.py — fused sensing + recurrent brain via CoreML
=========================================================================
Compiles one CoreML program per population bucket that:
  1. Ray-marches pre-built color grids (sensing) — all N wights in parallel on GPU
  2. Runs the Elman RNN forward pass

Color and food grids are painted on CPU before the model call using
size-proportional disk rasterization (_DISK_OFFSETS). The CoreML graph
receives four flat (GH*GW,) float32 arrays and does only raycasting + RNN —
all GPU-eligible ops, no scatter_nd fallbacks.

Bucket models are compiled at population sizes [16, 64, 256, 1024, MAX_POP].
_predict picks the smallest bucket >= current population so padding waste is
bounded to 4x instead of the 293x worst case of a single MAX_POP model.

Fallback: numpy sense + CoreML brain when compilation fails.
"""
from __future__ import annotations
import json, os, time
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
    WIDTH, HEIGHT,
    DRAIN_SCALE, SPEED_TAX, TURN_TAX, SIZE_TAX, SENSING_TAX, BRAIN_TAX, AGE_TAX,
)
from sim.grid.constants import (
    GW, GH, GRID_SCALE, MAX_STEPS, _STEPS, _RAY_OFFSETS,
)
from sim.grid.painter import paint_color_grids

PROJECT_ROOT = Path(__file__).resolve().parents[1]
META_PATH    = PROJECT_ROOT / "build" / "sense_brain_meta.json"

BUCKETS = [16, 64, 256, 1024, MAX_POP]

_models     = {}   # bucket_size -> ct.models.MLModel
_use_coreml = False


def _model_path(b):
    return PROJECT_ROOT / "build" / f"sense_brain_{b}.mlpackage"


def _compute_unit():
    name = os.environ.get('BEETLE_COMPUTE_UNITS', 'CPU_AND_GPU').upper()
    return getattr(ct.ComputeUnit, name, ct.ComputeUnit.CPU_AND_GPU)


def init_sense_brain() -> bool:
    if not _HAS_CT:
        return False
    cu = os.environ.get('BEETLE_COMPUTE_UNITS', 'CPU_AND_GPU')
    print(f"[SenseBrain] compute unit: {cu}")
    _load_or_compile()
    return _use_coreml


def _load_or_compile():
    global _models, _use_coreml

    meta_ok = False
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text())
            meta_ok = (
                meta.get("n_rays")    == N_RAYS    and
                meta.get("max_steps") == MAX_STEPS and
                meta.get("gw")  == GW  and meta.get("gh")    == GH    and
                meta.get("n_hid")     == N_HIDDEN  and
                meta.get("n_out")     == N_OUTPUTS and
                meta.get("has_nrays_mask") is True and
                meta.get("has_wh")    is True      and
                meta.get("has_rgb")   is True      and
                meta.get("has_bias")  is True      and
                meta.get("has_cpu_paint") is True  and
                meta.get("has_move")   is True      and
                meta.get("has_drain") is True      and
                meta.get("max_pop")   == MAX_POP   and
                meta.get("buckets")   == BUCKETS   and
                meta.get("compute_unit") == os.environ.get('BEETLE_COMPUTE_UNITS', 'CPU_AND_GPU') and
                all(_model_path(b).exists() for b in BUCKETS)
            )
        except Exception:
            pass

    if meta_ok:
        t0 = time.time()
        try:
            loaded = {}
            for b in BUCKETS:
                loaded[b] = ct.models.MLModel(str(_model_path(b)), compute_units=_compute_unit())
            _models     = loaded
            _use_coreml = True
            print(f"[SenseBrain] ready ({time.time()-t0:.1f}s)")
            return
        except Exception as e:
            print(f"[SenseBrain] cache load failed ({e}), rebuilding...")

    print(f"[SenseBrain] Compiling {len(BUCKETS)} bucket models "
          f"(rays={N_RAYS}, steps={MAX_STEPS}, grid={GW}×{GH}, "
          f"{N_INPUTS}→{N_HIDDEN}→{N_OUTPUTS})...", flush=True)
    t0 = time.time()
    try:
        PROJECT_ROOT.joinpath("build").mkdir(exist_ok=True)
        compiled = {}
        for b in BUCKETS:
            print(f"  [{b:>5}] ...", end="", flush=True)
            bt = time.time()
            compiled[b] = _compile(b)
            print(f" {time.time()-bt:.1f}s")

        META_PATH.write_text(json.dumps({
            "max_pop": MAX_POP, "n_rays": N_RAYS, "max_steps": MAX_STEPS,
            "gw": GW, "gh": GH, "n_hid": N_HIDDEN, "n_out": N_OUTPUTS,
            "has_nrays_mask": True, "has_wh": True, "has_rgb": True, "has_bias": True,
            "has_cpu_paint": True, "has_move": True, "has_drain": True,
            "buckets": BUCKETS,
            "compute_unit": os.environ.get('BEETLE_COMPUTE_UNITS', 'CPU_AND_GPU'),
        }))
        _models     = compiled
        _use_coreml = True
        print(f"[SenseBrain] all done ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[SenseBrain] FAILED: {e}")
        import traceback; traceback.print_exc()


def _compile(B):
    R, S      = N_RAYS, MAX_STEPS
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
        mb.TensorSpec(shape=(GRID_FLAT,)),             # r_flat
        mb.TensorSpec(shape=(GRID_FLAT,)),             # g_flat
        mb.TensorSpec(shape=(GRID_FLAT,)),             # b_flat
        mb.TensorSpec(shape=(GRID_FLAT,)),             # food_flat
        mb.TensorSpec(shape=(B, N_INPUTS, N_HIDDEN)),  # W1
        mb.TensorSpec(shape=(B, N_HIDDEN, N_OUTPUTS)), # W2
        mb.TensorSpec(shape=(B, N_HIDDEN, N_HIDDEN)),  # Wh
        mb.TensorSpec(shape=(B, N_HIDDEN)),            # h_prev
        mb.TensorSpec(shape=(B, N_HIDDEN)),            # mask
        mb.TensorSpec(shape=(B, N_RAYS * 5)),          # n_rays_mask
        mb.TensorSpec(shape=(B, N_HIDDEN)),            # b1
        mb.TensorSpec(shape=(B, N_OUTPUTS)),           # b2
        mb.TensorSpec(shape=(B,)),                     # turn_s
        mb.TensorSpec(shape=(B,)),                     # speed_base
        mb.TensorSpec(shape=(B,)),                     # size_world
        mb.TensorSpec(shape=(B,)),                     # n_rays_f
        mb.TensorSpec(shape=(B,)),                     # active_neurons_f
        mb.TensorSpec(shape=(B,)),                     # energy
    ])
    def prog(x_pos, y_pos, angle, half_fov,
             ray_len_pix, size_pix, energy_frac,
             r_flat, g_flat, b_flat, food_flat,
             W1, W2, Wh, h_prev, mask, n_rays_mask, b1, b2,
             turn_s, speed_base, size_world, n_rays_f, active_neurons_f, energy):

        steps_k   = mb.const(val=steps_c)
        offsets_k = mb.const(val=offsets_c)
        zero      = mb.const(val=np.float32(0.0))
        one       = mb.const(val=np.float32(1.0))
        gs        = mb.const(val=np.float32(GRID_SCALE))
        gw_f      = mb.const(val=np.float32(GW))
        gh_f      = mb.const(val=np.float32(GH))

        # ── world → grid positions ────────────────────────────────────────────
        gx = mb.mul(x=x_pos, y=gs)
        gy = mb.mul(x=y_pos, y=gs)

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

        # ── gather food and rgb from pre-built grids ──────────────────────────
        food_raw = mb.reshape(x=mb.gather(x=food_flat, indices=flat_i, axis=0), shape=[B, R, S])
        r_raw    = mb.reshape(x=mb.gather(x=r_flat,    indices=flat_i, axis=0), shape=[B, R, S])
        g_raw    = mb.reshape(x=mb.gather(x=g_flat,    indices=flat_i, axis=0), shape=[B, R, S])
        b_raw    = mb.reshape(x=mb.gather(x=b_flat,    indices=flat_i, axis=0), shape=[B, R, S])

        # food hits: binary
        food_hits = mb.cast(x=mb.greater(x=food_raw, y=zero), dtype='fp32')

        # org hits: any color channel nonzero
        rgb_sum  = mb.add(x=mb.add(x=r_raw, y=g_raw), y=b_raw)
        org_hits = mb.cast(x=mb.greater(x=rgb_sum, y=zero), dtype='fp32')

        # ── near-body exclusion ───────────────────────────────────────────────
        # CPU painting is exact per organism size — exclude exactly size_pix steps
        size_pix_e = mb.expand_dims(x=mb.expand_dims(x=size_pix, axes=[1]), axes=[2])
        far_enough = mb.cast(x=mb.greater_equal(x=steps_e, y=size_pix_e), dtype='fp32')
        org_hits   = mb.mul(x=org_hits, y=far_enough)

        # ── first-hit sensor signals ──────────────────────────────────────────
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

        # ── movement integration ──────────────────────────────────────────────
        w_w  = mb.const(val=np.float32(WIDTH))
        w_h  = mb.const(val=np.float32(HEIGHT))
        two  = mb.const(val=np.float32(2.0))
        turn_raw   = mb.slice_by_index(x=out, begin=[0, 0], end=[B, 1], stride=[1, 1])
        speed_raw  = mb.slice_by_index(x=out, begin=[0, 1], end=[B, 2], stride=[1, 1])
        turn_1d    = mb.squeeze(x=turn_raw,  axes=[-1])
        speed_1d   = mb.squeeze(x=speed_raw, axes=[-1])
        angle_new  = mb.add(x=angle, y=mb.mul(x=turn_1d, y=turn_s), name='angle_new')
        speeds     = mb.mul(x=mb.add(x=speed_1d, y=one), y=speed_base, name='speeds')
        x_raw      = mb.add(x=x_pos, y=mb.mul(x=mb.cos(x=angle_new), y=speeds))
        y_raw      = mb.add(x=y_pos, y=mb.mul(x=mb.sin(x=angle_new), y=speeds))
        # wrap to world bounds via modulo: v - floor(v/W)*W
        def _wmod(v, W):
            return mb.sub(x=v, y=mb.mul(x=mb.floor(x=mb.real_div(x=v, y=W)), y=W))
        x_new = _wmod(mb.add(x=x_raw, y=w_w), w_w)  # add W first so negatives wrap right
        y_new = _wmod(mb.add(x=y_raw, y=w_h), w_h)
        x_new = mb.identity(x=x_new, name='x_new')
        y_new = mb.identity(x=y_new, name='y_new')

        # ── metabolic drain + age decay ───────────────────────────────────────
        ds   = mb.const(val=np.float32(DRAIN_SCALE))
        st   = mb.const(val=np.float32(SPEED_TAX))
        tt   = mb.const(val=np.float32(TURN_TAX))
        szt  = mb.const(val=np.float32(SIZE_TAX))
        sent = mb.const(val=np.float32(SENSING_TAX))
        bt   = mb.const(val=np.float32(BRAIN_TAX))
        at   = mb.const(val=np.float32(1.0 - AGE_TAX))
        gs_c = mb.const(val=np.float32(GRID_SCALE))
        p75  = mb.const(val=np.float32(0.75))
        p2   = mb.const(val=np.float32(2.0))
        p15  = mb.const(val=np.float32(1.5))

        # base drain: DRAIN_SCALE * size^0.75
        base_drain = mb.mul(x=mb.pow(x=size_world, y=p75), y=ds)
        # speed tax: speeds^2 * SPEED_TAX
        spd_drain  = mb.mul(x=mb.mul(x=speeds, y=speeds), y=st)
        # turn tax: |turns| * size * TURN_TAX  (turns = out[:,0] * turn_s)
        turns_mag  = mb.abs(x=mb.mul(x=mb.slice_by_index(x=out, begin=[0,0], end=[B,1], stride=[1,1]), y=mb.expand_dims(x=turn_s, axes=[-1])))
        turns_1d   = mb.squeeze(x=turns_mag, axes=[-1])
        trn_drain  = mb.mul(x=mb.mul(x=turns_1d, y=size_world), y=tt)
        # size tax: size^2 * SIZE_TAX
        sz2_drain  = mb.mul(x=mb.mul(x=size_world, y=size_world), y=szt)
        # sensing tax: n_rays * ray_len * fov * SENSING_TAX
        fov_full   = mb.mul(x=half_fov, y=two)  # two already defined above
        ray_len_w  = mb.real_div(x=ray_len_pix, y=gs_c)
        sns_drain  = mb.mul(x=mb.mul(x=mb.mul(x=n_rays_f, y=ray_len_w), y=fov_full), y=sent)
        # brain tax: active_neurons^1.5 * BRAIN_TAX
        brn_drain  = mb.mul(x=mb.pow(x=active_neurons_f, y=p15), y=bt)

        total_drain = mb.add(x=mb.add(x=mb.add(x=mb.add(x=mb.add(
            x=base_drain, y=spd_drain), y=trn_drain), y=sz2_drain), y=sns_drain), y=brn_drain)
        energy_new  = mb.mul(x=mb.sub(x=energy, y=total_drain), y=at, name='energy_new')

        return h_new, out, x_new, y_new, angle_new, speeds, energy_new

    path  = _model_path(B)
    model = ct.convert(prog,
                       compute_units=_compute_unit(),
                       minimum_deployment_target=ct.target.macOS13)
    model.save(str(path))
    return ct.models.MLModel(str(path), compute_units=_compute_unit())


# ── public inference ──────────────────────────────────────────────────────────

def run_sense_brain(pop: dict, food: np.ndarray):
    """
    Fused sensing + brain for all wights.
    food: (F, 2) float32 world positions from world['food'].
    Returns (h_new, out, x_new, y_new, angle_new, speeds).

    Grid painting happens on CPU (disk rasterization), then the CoreML model
    does raycasting + RNN on GPU. Bucket models keep padding waste to ≤4x.
    Falls back to numpy sense + CoreML brain if the models failed to load.
    """
    if _models:
        return _predict(pop, food)
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
    inputs   = sense(pop, grid)
    h_new, out = run_brain(inputs, pop['W1'], pop['W2'], pop['Wh'], pop['b1'], pop['b2'], pop['h_state'])
    # compute move on CPU for the fallback path
    turns     = out[:, 0] * pop['turn_s']
    speeds    = (out[:, 1] + 1.0) * pop['speed']
    angle_new = pop['angle'] + turns
    x_new     = (pop['x'] + np.cos(angle_new) * speeds) % WIDTH
    y_new     = (pop['y'] + np.sin(angle_new) * speeds) % HEIGHT
    energy_raw = pop['energy'] - (
        DRAIN_SCALE * pop['size'] ** 0.75
        + speeds ** 2 * SPEED_TAX
        + np.abs(turns) * pop['size'] * TURN_TAX
        + pop['size'] ** 2 * SIZE_TAX
        + pop['n_rays'] * pop['ray_len'] * pop['fov'] * SENSING_TAX
        + pop['active_neurons'] ** 1.5 * BRAIN_TAX
    )
    energy_new = (energy_raw * (1.0 - AGE_TAX)).astype(np.float32)
    return h_new, out, x_new, y_new, angle_new, speeds, energy_new


def _predict(pop, food):
    n      = len(pop['x'])
    bucket = next(b for b in BUCKETS if b >= n)
    model  = _models[bucket]

    energy_max  = ENERGY_MAX_SCALE * pop['size'] ** 2
    energy_frac = (pop['energy'] / energy_max).astype(np.float32)
    half_fov    = (pop['fov'] * 0.5).astype(np.float32)
    ray_len_pix = np.maximum(pop['ray_len'] * GRID_SCALE, 1.0).astype(np.float32)
    size_pix    = np.ceil(pop['size'] * GRID_SCALE).astype(np.float32)

    r_flat, g_flat, b_flat, food_flat = paint_color_grids(pop, food)

    def _pad1(arr, fill=0.0):
        if n == bucket:
            return arr.astype(np.float32)
        return np.concatenate([arr.astype(np.float32),
                                np.full((bucket - n,), fill, dtype=np.float32)])

    def _padn(arr):
        if n == bucket:
            return arr.astype(np.float32)
        return np.concatenate([arr.astype(np.float32),
                                np.zeros((bucket - n,) + arr.shape[1:], dtype=np.float32)])

    mask        = (np.arange(N_HIDDEN) < pop['active_neurons'][:, None]).astype(np.float32)
    ray_active  = (np.arange(N_RAYS)[None, :] < pop['n_rays'][:, None]).astype(np.float32)
    n_rays_mask = np.repeat(ray_active, 5, axis=1)

    r = model.predict({
        'x_pos':       _pad1(pop['x']),
        'y_pos':       _pad1(pop['y']),
        'angle':       _pad1(pop['angle']),
        'half_fov':    _pad1(half_fov),
        'ray_len_pix': _pad1(ray_len_pix, fill=1.0),
        'size_pix':    _pad1(size_pix,    fill=1.0),
        'energy_frac': _pad1(energy_frac),
        'r_flat':      r_flat,
        'g_flat':      g_flat,
        'b_flat':      b_flat,
        'food_flat':   food_flat,
        'W1':          _padn(pop['W1']),
        'W2':          _padn(pop['W2']),
        'Wh':          _padn(pop['Wh']),
        'h_prev':      _padn(pop['h_state']),
        'mask':        _padn(mask),
        'n_rays_mask': _padn(n_rays_mask),
        'b1':          _padn(pop['b1']),
        'b2':          _padn(pop['b2']),
        'turn_s':           _pad1(pop['turn_s']),
        'speed_base':       _pad1(pop['speed']),
        'size_world':       _pad1(pop['size']),
        'n_rays_f':         _pad1(pop['n_rays'].astype(np.float32)),
        'active_neurons_f': _pad1(pop['active_neurons'].astype(np.float32)),
        'energy':           _pad1(pop['energy']),
    })
    return (
        r['h_new'][:n],
        r['out'][:n],
        r['x_new'][:n],
        r['y_new'][:n],
        r['angle_new'][:n],
        r['speeds'][:n],
        r['energy_new'][:n],
    )
