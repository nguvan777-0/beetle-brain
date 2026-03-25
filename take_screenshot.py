"""
Render a screenshot of the sim without a display window.
uv run --with coremltools --with numpy --with pygame python take_screenshot.py
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import numpy as np
import pygame
pygame.init()

import sim
from sim import new_world, tick as sim_tick, init_ane, phylo
from sim.config import DRAIN_SCALE
from game.renderer import draw_organism, draw_rays, draw_food
from game.panel import draw_panel, PANEL_W
from game.panel.hud import _anc_color

TICKS   = 3000
OUT     = "screenshot.png"

init_ane()
world = new_world(42)
rng   = np.random.default_rng(42)
tick  = 0
history          = []
lineage_history  = []

print(f"running {TICKS} ticks...")
for i in range(TICKS):
    world = sim_tick(world, rng)
    tick += 1
    pop = world['pop']
    if len(pop['x']) == 0:
        print(f"extinction at tick {tick}")
        break
    if tick % 30 == 0:
        history.append((float(tick), float(len(pop['x'])), float(pop['generation'].max()),
                        float(pop['speed'].mean()), float(pop['fov'].mean()),
                        float(pop['size'].mean()), float(pop['mutation_rate'].mean())))
        depth = max(4, int(pop['generation'].max()) // 3)
        anc = phylo.ancestor_at(pop['individual_id'], depth, world['phylo'])
        import numpy as _np
        u, c = _np.unique(anc, return_counts=True)
        lineage_history.append(dict(zip(u.tolist(), c.tolist())))
        if len(lineage_history) > 300:
            lineage_history.pop(0)
    if tick % 1000 == 0:
        print(f"  tick {tick}  pop={len(pop['x'])}  maxGen={pop['generation'].max()}")

pop   = world['pop']
food  = world['food']
vents = world['vents']
print(f"done: pop={len(pop['x'])}  maxGen={pop['generation'].max()}")

W = sim.WIDTH + PANEL_W
H = sim.HEIGHT
surf = pygame.Surface((W, H))
font    = pygame.font.SysFont("monospace", 12)
font_sm = pygame.font.SysFont("monospace", 10)
font_lg = pygame.font.SysFont("monospace", 14)

surf.fill((10, 10, 18))
draw_food(surf, food, vents)

depth    = max(4, int(pop['generation'].max()) // 3)
anc_ids  = phylo.ancestor_at(pop['individual_id'], depth, world['phylo'])
halo_colors = [_anc_color(int(a), world['phylo']) for a in anc_ids]
for i in range(len(pop['x'])):
    draw_organism(surf, pop['x'][i], pop['y'][i], pop['angle'][i],
                  pop['size'][i], int(pop['r'][i]), int(pop['g'][i]), int(pop['b'][i]),
                  halo_colors[i])

history = []
for t in range(0, tick, 30):
    history.append((float(t), float(len(pop['x'])), float(pop['generation'].max()),
                    float(pop['speed'].mean()), float(pop['fov'].mean()),
                    float(pop['size'].mean()), float(pop['mutation_rate'].mean())))

draw_panel(surf, font, font_sm, font_lg, tick, pop, None,
           history, lineage_history, [], 1, vents=vents, phylo_state=world['phylo'],
           seed=world.get('seed'))

pygame.image.save(surf, OUT)
print(f"saved → {OUT}")
