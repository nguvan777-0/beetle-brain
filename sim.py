"""
sim.py — vectorized beetle-brain simulation engine
===================================================
The entire population lives in numpy arrays.
No Python loop over wights — one tick = a handful of numpy calls.

Sensing is O(1) per organism via a rasterized world grid:
  - food and organisms are painted into a (2, GH, GW) uint8 grid
  - each organism ray-marches N_RAYS × MAX_STEPS pixels — always the same
  - adding more organisms doesn't slow down sensing at all

Wight genome = W_body (9 floats) + W1 (180 floats) + W2 (24 floats) = 213 floats.
The wight IS its weights. Nothing else.

Population dict keys:
    x, y, angle, energy                    — physics  (N,)
    W_body                                  — body genome (N, N_BODY)
    W1, W2                                  — brain genome (N, N_IN, N_HID), (N, N_HID, N_OUT)
    speed, fov, ray_len, size, drain, turn_s— decoded traits (N,)
    r, g, b                                 — color (N,)
    generation, age, eaten                  — stats (N,) int32
"""

import tomllib
import numpy as np
from pathlib import Path
from brain.coreml_brain import init_brain, run_brain

# ── LOAD CONFIG ───────────────────────────────────────────────────────────────
_cfg_path = Path(__file__).resolve().parent / "config.toml"
with open(_cfg_path, "rb") as _f:
    _cfg = tomllib.load(_f)

WIDTH, HEIGHT   = _cfg["world"]["width"],      _cfg["world"]["height"]
N_FOOD          = _cfg["world"]["food_count"]
N_START         = _cfg["world"]["n_start"]
MAX_POP         = _cfg["world"]["max_pop"]

SPEED_MIN       = _cfg["traits"]["speed_min"]; SPEED_MAX  = _cfg["traits"]["speed_max"]
FOV_MIN         = _cfg["traits"]["fov_min"];   FOV_MAX    = _cfg["traits"]["fov_max"]
RAY_MIN         = _cfg["traits"]["ray_min"];   RAY_MAX    = _cfg["traits"]["ray_max"]
SIZE_MIN        = _cfg["traits"]["size_min"];  SIZE_MAX   = _cfg["traits"]["size_max"]
DRAIN_MIN       = _cfg["traits"]["drain_min"]; DRAIN_MAX  = _cfg["traits"]["drain_max"]

ENERGY_START    = _cfg["energy"]["start"]
ENERGY_MAX      = _cfg["energy"]["max"]
ENERGY_FOOD     = _cfg["energy"]["food"]
ENERGY_BREED    = _cfg["energy"]["breed_at"]
ENERGY_CLONE    = _cfg["energy"]["clone_with"]

N_RAYS          = _cfg["brain"]["n_rays"]
N_HIDDEN        = _cfg["brain"]["n_hidden"]
N_OUTPUTS       = _cfg["brain"]["n_outputs"]
N_INPUTS        = N_RAYS * 2 + 1
N_BODY          = 9

MUTATION_RATE   = _cfg["evolution"]["mutation_rate"]
MUTATION_SCALE  = _cfg["evolution"]["mutation_scale"]
EPIGENETIC      = _cfg["evolution"]["epigenetic"]

AGING_ENABLED   = _cfg["aging"]["enabled"]
WEIGHT_DECAY    = _cfg["aging"]["weight_decay"]

CAMO_ENABLED    = _cfg["camouflage"]["enabled"]
CAMO_BONUS      = _cfg["camouflage"]["detect_bonus"]

# ── WORLD GRID (shared spatial data structure) ────────────────────────────────
# The world is rasterized into a (2, GH, GW) grid each tick.
# Channel 0 = food presence, channel 1 = organism size (1-255).
# Organisms ray-march through this grid: O(1) per organism regardless of N.
GRID_SCALE  = 0.5                        # world units per pixel  (2:1)
GW          = int(WIDTH  * GRID_SCALE)   # 450 pixels wide
GH          = int(HEIGHT * GRID_SCALE)   # 450 pixels tall
MAX_STEPS   = int(RAY_MAX * GRID_SCALE)  # 90 steps — longest ray in pixels
_STEPS      = np.arange(1, MAX_STEPS + 1, dtype=np.float32)  # (90,) reused every tick

# precomputed ray offsets from center
_RAY_OFFSETS = np.linspace(-1, 1, N_RAYS, dtype=np.float32)  # scaled by fov/2 per tick


# ── INIT ──────────────────────────────────────────────────────────────────────
def init_ane():
    ok = init_brain(MAX_POP, N_INPUTS, N_HIDDEN, N_OUTPUTS)
    return ok


# ── GENOME ────────────────────────────────────────────────────────────────────
def _sig(x): return 1.0 / (1.0 + np.exp(-x))

def _decode(W_body):
    """Decode W_body (N, N_BODY) → trait arrays."""
    s = _sig(W_body)
    speed  = SPEED_MIN  + s[:,0] * (SPEED_MAX  - SPEED_MIN)
    fov    = FOV_MIN    + s[:,1] * (FOV_MAX    - FOV_MIN)
    ray    = RAY_MIN    + s[:,2] * (RAY_MAX    - RAY_MIN)
    size   = SIZE_MIN   + s[:,3] * (SIZE_MAX   - SIZE_MIN)
    drain  = DRAIN_MIN  + s[:,4] * (DRAIN_MAX  - DRAIN_MIN)
    turn_s = 0.05       + s[:,8] * 0.25
    r = (40 + s[:,5] * 215).astype(np.int32)
    g = (40 + s[:,6] * 215).astype(np.int32)
    b = (40 + s[:,7] * 215).astype(np.int32)
    return speed, fov, ray, size, drain, turn_s, r, g, b

def _make_pop(n, rng):
    W_body = rng.standard_normal((n, N_BODY)).astype(np.float32)
    W1     = (rng.standard_normal((n, N_INPUTS, N_HIDDEN)) * 0.8).astype(np.float32)
    W2     = (rng.standard_normal((n, N_HIDDEN, N_OUTPUTS)) * 0.8).astype(np.float32)
    speed, fov, ray, size, drain, turn_s, r, g, b = _decode(W_body)
    return {
        'x':     rng.uniform(0, WIDTH,  n).astype(np.float32),
        'y':     rng.uniform(0, HEIGHT, n).astype(np.float32),
        'angle': rng.uniform(0, 2*np.pi, n).astype(np.float32),
        'energy':np.full(n, ENERGY_START, dtype=np.float32),
        'W_body':W_body, 'W1':W1, 'W2':W2,
        'speed':speed.astype(np.float32), 'fov':fov.astype(np.float32),
        'ray_len':ray.astype(np.float32), 'size':size.astype(np.float32),
        'drain':drain.astype(np.float32), 'turn_s':turn_s.astype(np.float32),
        'r':r, 'g':g, 'b':b,
        'generation':np.zeros(n, dtype=np.int32),
        'age':       np.zeros(n, dtype=np.int32),
        'eaten':     np.zeros(n, dtype=np.int32),
        'h_state':   np.zeros((n, N_HIDDEN), dtype=np.float32),
    }

def new_world(rng=None):
    if rng is None: rng = np.random.default_rng()
    pop  = _make_pop(N_START, rng)
    food = rng.uniform(0, [WIDTH, HEIGHT], size=(N_FOOD, 2)).astype(np.float32)
    return pop, food


# ── SENSING (grid-based, O(1) per organism) ────────────────────────────────────
def _paint_grid(pop, food):
    """
    Rasterize the world into a (2, GH, GW) uint8 grid.
    Channel 0 — food:      1 where food exists, 0 elsewhere
    Channel 1 — organisms: normalized size (1-255), 0 = empty

    This is the shared data structure. Every organism writes once;
    every organism reads independently. They never query each other directly.
    """
    grid = np.zeros((2, GH, GW), dtype=np.uint8)

    # paint food
    if len(food):
        fy = np.clip((food[:, 1] * GRID_SCALE).astype(np.int32), 0, GH - 1)
        fx = np.clip((food[:, 0] * GRID_SCALE).astype(np.int32), 0, GW - 1)
        grid[0, fy, fx] = 1

    # paint organisms — encode brightness (r+g+b) so bright organisms are
    # more detectable on other organisms' rays (camouflage pressure)
    oy = np.clip((pop['y'] * GRID_SCALE).astype(np.int32), 0, GH - 1)
    ox = np.clip((pop['x'] * GRID_SCALE).astype(np.int32), 0, GW - 1)
    brightness = ((pop['r'].astype(np.int32) + pop['g'].astype(np.int32) + pop['b'].astype(np.int32)) / 3).astype(np.int32)
    norm = np.clip(brightness, 1, 255).astype(np.uint8)
    grid[1, oy, ox] = norm

    return grid


def _sense(pop, grid):
    """
    Ray-march through the world grid for all N organisms simultaneously.

    Each organism fires N_RAYS rays, each MAX_STEPS pixels long.
    Total reads: N × N_RAYS × MAX_STEPS — a fixed constant per organism.
    N=300 or N=3000 doesn't change the per-organism cost at all.

    Returns inputs (N, N_INPUTS).
    """
    N = len(pop['x'])
    inputs = np.zeros((N, N_INPUTS), dtype=np.float32)

    # ── ray directions: (N, N_RAYS, 2) ──────────────────────────────────────
    half_fov   = pop['fov'][:, None] * 0.5
    ray_angles = pop['angle'][:, None] + _RAY_OFFSETS[None, :] * half_fov
    ray_dirs   = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=-1)

    # ── pixel coordinates along every ray ────────────────────────────────────
    # org positions in grid space: (N,)
    gx = pop['x'] * GRID_SCALE
    gy = pop['y'] * GRID_SCALE

    # (N, N_RAYS, MAX_STEPS) — step each ray out from the organism
    # ray_dirs[:, :, 0] is (N, N_RAYS); _STEPS is (MAX_STEPS,)
    coords_x = gx[:, None, None] + ray_dirs[:, :, None, 0] * _STEPS[None, None, :]
    coords_y = gy[:, None, None] + ray_dirs[:, :, None, 1] * _STEPS[None, None, :]

    # toroidal wrap (world wraps, so grid wraps too)
    rx = coords_x.astype(np.int32) % GW   # (N, N_RAYS, MAX_STEPS)
    ry = coords_y.astype(np.int32) % GH

    # ── sample both grid channels ─────────────────────────────────────────────
    food_hits = grid[0][ry, rx] > 0   # (N, N_RAYS, MAX_STEPS) bool
    org_hits  = grid[1][ry, rx] > 0

    # exclude self: ignore org hits within own radius (organism painted at center)
    size_pix  = np.ceil(pop['size'] * GRID_SCALE).astype(np.int32)    # (N,)
    step_idx  = np.arange(MAX_STEPS, dtype=np.int32)[None, None, :]   # (1, 1, MAX_STEPS)
    org_hits  = org_hits & (step_idx >= size_pix[:, None, None])

    # ── first hit per ray → normalized distance ───────────────────────────────
    ray_len_pix = pop['ray_len'] * GRID_SCALE   # (N,) — organism's actual ray length

    def _first_hit(hits):
        # hits: (N, N_RAYS, MAX_STEPS) bool
        has_hit  = hits.any(axis=2)                                    # (N, N_RAYS)
        hit_step = np.argmax(hits, axis=2).astype(np.float32) + 1.0   # (N, N_RAYS), 1-based
        dist     = np.where(has_hit, hit_step, ray_len_pix[:, None])   # miss → ray_len
        return np.clip(dist / ray_len_pix[:, None], 0.0, 1.0)

    inputs[:, 0:N_RAYS*2:2] = 1.0 - _first_hit(food_hits)
    inputs[:, 1:N_RAYS*2:2] = 1.0 - _first_hit(org_hits)
    inputs[:, -1]           = pop['energy'] / ENERGY_MAX
    return inputs


# ── TICK (fully vectorized) ───────────────────────────────────────────────────
def tick(pop, food, rng):
    N = len(pop['x'])

    # ── paint world grid once, sense all organisms from it ──────────────────
    grid       = _paint_grid(pop, food)
    inputs     = _sense(pop, grid)

    # ── brain forward pass (recurrent) ──────────────────────────────────────
    h_new, out       = run_brain(inputs, pop['W1'], pop['W2'], pop['h_state'])
    pop['h_state']   = h_new                              # store hidden state for next tick
    turns      = out[:, 0] * pop['turn_s']                # (N,)
    speeds     = (out[:, 1] + 1.0) * 0.5 * pop['speed']  # (N,)

    # ── physics ─────────────────────────────────────────────────────────────
    pop['angle'] += turns
    pop['x']      = (pop['x'] + np.cos(pop['angle']) * speeds) % WIDTH
    pop['y']      = (pop['y'] + np.sin(pop['angle']) * speeds) % HEIGHT
    pop['energy'] -= pop['drain'] + speeds * 0.01 + pop['size'] * 0.002
    pop['age']    += 1

    # ── eat food ────────────────────────────────────────────────────────────
    if len(food) > 0:
        org_pos  = np.stack([pop['x'], pop['y']], axis=1)   # (N, 2)
        dist_f   = np.linalg.norm(food[None,:,:] - org_pos[:,None,:], axis=2)  # (N, M)
        eat_mask = dist_f < (pop['size'][:, None] + 3.0)                       # (N, M)
        eaten_food = eat_mask.any(axis=0)                   # (M,) — food eaten by anyone
        gain_per   = eat_mask.sum(axis=1).astype(np.float32) * ENERGY_FOOD
        pop['energy'] = np.minimum(ENERGY_MAX, pop['energy'] + gain_per)
        pop['eaten'] += eat_mask.any(axis=1).astype(np.int32)
        food = food[~eaten_food]

    # ── predation ───────────────────────────────────────────────────────────
    if N > 1:
        org_pos  = np.stack([pop['x'], pop['y']], axis=1)
        dist_o   = np.linalg.norm(org_pos[None,:,:] - org_pos[:,None,:], axis=2)  # (N, N)
        np.fill_diagonal(dist_o, np.inf)
        # bright prey are detectable from further away — camouflage pressure
        brightness = (pop['r'].astype(np.float32) + pop['g'] + pop['b']) / (3.0 * 255.0)  # (N,) 0-1
        detect_r   = pop['size'] + (brightness * CAMO_BONUS if CAMO_ENABLED else 0.0)     # (N,)
        touch    = dist_o < (pop['size'][:, None] + detect_r[None, :])                    # (N, N)
        bigger   = pop['size'][:, None] > pop['size'][None, :] * 1.25             # (N, N)
        kills    = touch & bigger                                                   # (N, N)
        killed   = kills.any(axis=0)                                               # (N,)
        killer   = kills.any(axis=1)                                               # (N,)
        # killer gains energy from prey
        prey_energy = (kills * pop['energy'][None, :]).sum(axis=1)
        pop['energy'] = np.where(killer,
                                  np.minimum(ENERGY_MAX, pop['energy'] + prey_energy * 0.7),
                                  pop['energy'])
        pop['eaten'] += kills.sum(axis=1).astype(np.int32)
    else:
        killed = np.zeros(N, dtype=bool)

    # ── aging: weights decay toward zero each tick ──────────────────────────
    # traits drift toward midpoint, brain softens — old organisms get worse
    if AGING_ENABLED:
        decay = 1.0 - WEIGHT_DECAY
        pop['W_body'] *= decay
        pop['W1']     *= decay
        pop['W2']     *= decay
        # re-decode traits from decayed W_body so physics reflects aging
        (pop['speed'], pop['fov'], pop['ray_len'], pop['size'],
         pop['drain'],  pop['turn_s'],
         pop['r'], pop['g'], pop['b']) = _decode(pop['W_body'])

    # ── death ───────────────────────────────────────────────────────────────
    alive = (pop['energy'] > 0) & (~killed)

    # ── cloning ─────────────────────────────────────────────────────────────
    can_breed = alive & (pop['energy'] >= ENERGY_BREED)
    n_breed   = can_breed.sum()
    pop['energy'] = np.where(can_breed, ENERGY_CLONE, pop['energy'])

    if n_breed > 0 and N < MAX_POP:
        n_children = min(n_breed, MAX_POP - N)
        parent_idx = np.where(can_breed)[0][:n_children]
        children   = _clone_batch(pop, parent_idx, rng)
        pop        = _filter(pop, alive)
        pop        = _concat(pop, children)
    else:
        pop = _filter(pop, alive)

    # ── respawn food ────────────────────────────────────────────────────────
    short = N_FOOD - len(food)
    if short > 0:
        new_f = rng.uniform(0, [WIDTH, HEIGHT], size=(short, 2)).astype(np.float32)
        food  = np.vstack([food, new_f]) if len(food) else new_f

    # ── inject random if population collapses ───────────────────────────────
    if len(pop['x']) < 10:
        fresh = _make_pop(20, rng)
        pop   = _concat(pop, fresh)

    return pop, food


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _filter(pop, mask):
    return {k: v[mask] for k, v in pop.items()}

def _concat(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}

def _clone_batch(pop, idx, rng):
    n = len(idx)
    noise_r  = rng.random((n, N_BODY)).astype(np.float32)
    noise_W1 = rng.random((n, N_INPUTS, N_HIDDEN)).astype(np.float32)
    noise_W2 = rng.random((n, N_HIDDEN, N_OUTPUTS)).astype(np.float32)

    mut_r  = (rng.standard_normal((n, N_BODY))            * MUTATION_SCALE).astype(np.float32)
    mut_W1 = (rng.standard_normal((n, N_INPUTS, N_HIDDEN))* MUTATION_SCALE).astype(np.float32)
    mut_W2 = (rng.standard_normal((n, N_HIDDEN, N_OUTPUTS))* MUTATION_SCALE).astype(np.float32)

    W_body = pop['W_body'][idx] + np.where(noise_r  < MUTATION_RATE, mut_r,  0)
    W1     = pop['W1'][idx]     + np.where(noise_W1 < MUTATION_RATE, mut_W1, 0)
    W2     = pop['W2'][idx]     + np.where(noise_W2 < MUTATION_RATE, mut_W2, 0)

    speed, fov, ray, size, drain, turn_s, r, g, b = _decode(W_body)
    ang = pop['angle'][idx] + np.pi + rng.uniform(-0.5, 0.5, n).astype(np.float32)

    return {
        'x':     (pop['x'][idx] + np.cos(ang) * (pop['size'][idx]*2+2)) % WIDTH,
        'y':     (pop['y'][idx] + np.sin(ang) * (pop['size'][idx]*2+2)) % HEIGHT,
        'angle': ang,
        'energy':np.full(n, ENERGY_CLONE, dtype=np.float32),
        'W_body':W_body, 'W1':W1, 'W2':W2,
        'speed': speed.astype(np.float32), 'fov': fov.astype(np.float32),
        'ray_len':ray.astype(np.float32),  'size':size.astype(np.float32),
        'drain': drain.astype(np.float32), 'turn_s':turn_s.astype(np.float32),
        'r':r, 'g':g, 'b':b,
        'generation':(pop['generation'][idx] + 1).astype(np.int32),
        'age':       np.zeros(n, dtype=np.int32),
        'eaten':     np.zeros(n, dtype=np.int32),
        'h_state':   pop['h_state'][idx] * EPIGENETIC,
    }
