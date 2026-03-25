"""One simulation tick: sense → think → move → eat → predate → age → die → breed."""
import numpy as np
from sim.config import (
    WIDTH, HEIGHT, N_FOOD, MAX_POP,
    ENERGY_FOOD, COASTLINE_X, ENERGY_SUNLIGHT,
    ENERGY_MAX_SCALE, DRAIN_SCALE, SIZE_TAX, SPEED_TAX, TURN_TAX, AGE_TAX, SENSING_TAX, BRAIN_TAX
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
    h_new, out     = run_sense_brain(pop, grid[0], grid[1], grid[2], grid[3])
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
                      + np.abs(turns) * pop['size'] * TURN_TAX
                      + pop['size']**2          * SIZE_TAX
                      + pop['n_rays'] * pop['ray_len'] * pop['fov'] * SENSING_TAX
                      + pop['active_neurons']**1.5  * BRAIN_TAX)

    # ── sunlight (land only) ─────────────────────────────────────────────────
    if ENERGY_SUNLIGHT > 0:
        land_mask = pop['x'] >= COASTLINE_X
        if land_mask.any():
            from sim.grid.constants import GRID_SCALE, GH, GW
            cx = np.clip((pop['x'][land_mask] * GRID_SCALE).astype(np.int32), 0, GW - 1)
            cy = np.clip((pop['y'][land_mask] * GRID_SCALE).astype(np.int32), 0, GH - 1)

            # Flat grid indices for unique matching
            flat_idx = cy * GW + cx
            unique_cells, inv_idx, counts = np.unique(flat_idx, return_inverse=True, return_counts=True)

            # Divide sunlight fairly among all occupants of the cell
            gain = ENERGY_SUNLIGHT / counts[inv_idx]

            # Add to energy arrays (doing it via mask)
            energy_gain = np.zeros(len(pop['x']), dtype=np.float32)
            energy_gain[land_mask] = gain
            pop['energy'] += energy_gain

    pop['energy'] *= (1.0 - AGE_TAX)
    pop['age']    += 1

    # ── eat food ─────────────────────────────────────────────────────────────
    if len(food) > 0 and len(pop['x']) > 0:
        # Broad Phase: Map food to integer painter-grid, narrow search via bounding patch
        from sim.grid.constants import GRID_SCALE, GH, GW, _PR_OFF
        
        fy = np.clip((food[:, 1] * GRID_SCALE).astype(np.int32), 0, GH - 1)
        fx = np.clip((food[:, 0] * GRID_SCALE).astype(np.int32), 0, GW - 1)
        
        # Patch lookup around each food pellet
        row_idx = np.clip(fy[:, None, None] + _PR_OFF[None, :, None], 0, GH - 1)
        col_idx = np.clip(fx[:, None, None] + _PR_OFF[None, None, :], 0, GW - 1)
        potentials = idx_grid[row_idx, col_idx]
        
        f_idx, ry, rx = np.where(potentials >= 0)
        eater_candidates = potentials[f_idx, ry, rx]
        
        # Narrow Phase: Exact distance checks for physics fidelity
        if len(eater_candidates) > 0:
            food_pos = food[f_idx]
            org_pos = np.stack([pop['x'][eater_candidates], pop['y'][eater_candidates]], axis=1)
            dist = np.linalg.norm(food_pos - org_pos, axis=1)
            
            eat_radius = pop['size'][eater_candidates] + pop['mouth'][eater_candidates]
            actual_eats = dist < eat_radius
            
            eaten_f_idx = f_idx[actual_eats]
            actual_eater_ids = eater_candidates[actual_eats]
            
            unique_f, counts_f = np.unique(eaten_f_idx, return_counts=True)
            
            if len(unique_f) > 0:
                f_sharers = np.zeros(len(food), dtype=np.float32)
                f_sharers[unique_f] = counts_f
                
                # Split energy amongst organisms eating the same pellet
                gain_per_event = ENERGY_FOOD / f_sharers[eaten_f_idx]
                gain_per_org = np.bincount(actual_eater_ids, weights=gain_per_event, minlength=len(pop['x'])).astype(np.float32)
                
                pop['energy'] = np.minimum(energy_max, pop['energy'] + gain_per_org)
                pop['eaten'] += (np.bincount(actual_eater_ids, minlength=len(pop['x'])) > 0).astype(np.int32)
                
                # Filter remaining food
                eaten_food_mask = np.zeros(len(food), dtype=bool)
                eaten_food_mask[unique_f] = True
                food = food[~eaten_food_mask]

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
        
        # Branchless max-pop constraint
        excess_count = max(0, len(pop['x']) - MAX_POP)
        if excess_count > 0:
            keep = np.argsort(-pop['generation'])[:MAX_POP]
            pop  = filter_pop(pop, keep)
    else:
        pop = filter_pop(pop, alive)

    # ── refill each vent independently ───────────────────────────────────────
    food = refill_vents(food, vents, rng, N_FOOD // len(vents))

    world['pop']  = pop
    world['food'] = food
    return world
