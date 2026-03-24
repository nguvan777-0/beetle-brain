"""One simulation tick: sense → think → move → eat → predate → age → die → breed."""
import numpy as np
from sim.config import (
    WIDTH, HEIGHT, N_FOOD, MAX_POP,
    ENERGY_FOOD,
    ENERGY_MAX_SCALE, DRAIN_SCALE, SIZE_TAX, SPEED_TAX, AGE_TAX,
)
from sim.vents import refill_vents
from sim.grid.painter import paint_grid
from sim.sensing import sense
from sim.predation import predation
from sim.evolution import clone_batch
from sim.population.genome import decode
from sim.population.ops import filter_pop, concat_pop
from brain.coreml_brain import run_brain


def tick(world, rng):
    pop         = world['pop']
    food        = world['food']
    vents       = world['vents']
    phylo_state = world['phylo']
    energy_max  = ENERGY_MAX_SCALE * pop['size'] ** 2   # storage ∝ volume

    # ── sense ────────────────────────────────────────────────────────────────
    grid, idx_grid = paint_grid(pop, food)
    inputs         = sense(pop, grid)

    # ── brain ────────────────────────────────────────────────────────────────
    h_new, out     = run_brain(inputs, pop['W1'], pop['W2'], pop['h_state'])
    pop['h_state'] = h_new
    turns  = out[:, 0] * pop['turn_s']
    speeds = (out[:, 1] + 1.0) * pop['speed']

    # ── move ─────────────────────────────────────────────────────────────────
    pop['angle'] += turns
    pop['x']      = (pop['x'] + np.cos(pop['angle']) * speeds) % WIDTH
    pop['y']      = (pop['y'] + np.sin(pop['angle']) * speeds) % HEIGHT

    # ── metabolic drain ──────────────────────────────────────────────────────
    drain = DRAIN_SCALE * pop['size'] ** 0.75   # Kleiber's law
    pop['energy'] -= (drain
                      + speeds**2      * SPEED_TAX
                      + pop['size']**2 * SIZE_TAX)
    pop['energy'] *= (1.0 - AGE_TAX)
    pop['age']    += 1

    # ── eat food ─────────────────────────────────────────────────────────────
    if len(food) > 0:
        org_pos      = np.stack([pop['x'], pop['y']], axis=1)
        dist_f       = np.linalg.norm(food[None, :, :] - org_pos[:, None, :], axis=2)
        eat_mask     = dist_f < (pop['size'][:, None] + pop['mouth'][:, None])
        eaten_food   = eat_mask.any(axis=0)
        eaters_count = eat_mask.sum(axis=0).clip(min=1).astype(np.float32)
        share        = ENERGY_FOOD / eaters_count
        gain_per     = (eat_mask.astype(np.float32) * share[None, :]).sum(axis=1)
        pop['energy'] = np.minimum(energy_max, pop['energy'] + gain_per)
        pop['eaten'] += eat_mask.any(axis=1).astype(np.int32)
        food = food[~eaten_food]

    # ── predation ────────────────────────────────────────────────────────────
    killed, prey_gain = predation(pop, idx_grid)
    pop['energy'] = np.minimum(energy_max, pop['energy'] + prey_gain)
    pop['eaten'] += (prey_gain > 0).astype(np.int32)

    # ── death ────────────────────────────────────────────────────────────────
    alive = (pop['energy'] > 0) & (~killed)

    # ── breed ────────────────────────────────────────────────────────────────
    can_breed = alive & (pop['energy'] >= pop['breed_at'])
    pop['energy'] = np.where(can_breed, pop['clone_with'], pop['energy'])

    if can_breed.any():
        children = clone_batch(pop, np.where(can_breed)[0], rng, phylo_state)
        pop      = filter_pop(pop, alive)
        pop      = concat_pop(pop, children)
        if len(pop['x']) > MAX_POP:
            keep = np.argsort(-pop['generation'])[:MAX_POP]
            pop  = filter_pop(pop, keep)
    else:
        pop = filter_pop(pop, alive)

    # ── refill each vent independently ───────────────────────────────────────
    food = refill_vents(food, vents, rng, N_FOOD // len(vents))

    world['pop']  = pop
    world['food'] = food
    return world
