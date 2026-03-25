"""One simulation tick: sense → think → move → eat → predate → age → die → breed."""
import numpy as np
from sim.config import (
    WIDTH, HEIGHT, N_FOOD, MAX_POP,
    ENERGY_FOOD,
    ENERGY_MAX_SCALE, DRAIN_SCALE, SIZE_TAX, SPEED_TAX, AGE_TAX, SENSING_TAX,
)
from sim.vents import refill_vents
from sim.grid.painter import paint_grid
from sim.predation import predation
from sim.evolution import clone_batch
from sim.hgt import eat_hgt, contact_hgt
from sim.population.genome import decode
from sim.population.ops import filter_pop, concat_pop
from brain.coreml_sense_brain import run_sense_brain


def tick(world, rng):
    pop         = world['pop']
    food        = world['food']
    vents       = world['vents']
    phylo_state = world['phylo']
    energy_max  = ENERGY_MAX_SCALE * pop['size'] ** 2   # storage ∝ volume

    # ── sense + brain (fused GPU dispatch) ───────────────────────────────────
    grid, idx_grid = paint_grid(pop, food)
    h_new, out     = run_sense_brain(pop, grid[0], grid[1])
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
                      + speeds**2               * SPEED_TAX
                      + pop['size']**2          * SIZE_TAX
                      + pop['ray_len'] * pop['fov'] * SENSING_TAX)
    pop['energy'] *= (1.0 - AGE_TAX)
    pop['age']    += 1

    # ── eat food ─────────────────────────────────────────────────────────────
    if len(food) > 0 and len(pop['x']) > 0:
        # Array Rasterization O(N) Eating
        # Map food to the same integer painter-grid used for vision
        from sim.grid.constants import GRID_SCALE, GH, GW
        
        fy = np.clip((food[:, 1] * GRID_SCALE).astype(np.int32), 0, GH - 1)
        fx = np.clip((food[:, 0] * GRID_SCALE).astype(np.int32), 0, GW - 1)
        
        # Look up which organism is standing on the food's grid cell
        eater_ids = idx_grid[fy, fx] 
        valid_eats = eater_ids >= 0
        
        # Mask out any food that was eaten
        eaten_food = np.zeros(len(food), dtype=bool)
        eaten_food[valid_eats] = True
        
        if valid_eats.any():
            actual_eaters = eater_ids[valid_eats]
            eaten_counts  = np.bincount(actual_eaters, minlength=len(pop['x'])).astype(np.float32)
            
            # Since multiple organisms might theoretically map to the same cell (due to resolution),
            # energy is distributed directly (no sharing division required on grid level)
            gain_per = eaten_counts * ENERGY_FOOD
            
            pop['energy'] = np.minimum(energy_max, pop['energy'] + gain_per)
            pop['eaten'] += (eaten_counts > 0).astype(np.int32)
            
            # Filter remaining food
            food = food[~eaten_food]

    # ── predation ────────────────────────────────────────────────────────────
    killed, prey_gain, pred_idx, prey_idx = predation(pop, idx_grid)
    pop['energy'] = np.minimum(energy_max, pop['energy'] + prey_gain)
    pop['eaten'] += (prey_gain > 0).astype(np.int32)

    # ── horizontal gene transfer ──────────────────────────────────────────────
    eat_hgt(pop, pred_idx, prey_idx, rng)
    contact_hgt(pop, idx_grid, rng)

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
