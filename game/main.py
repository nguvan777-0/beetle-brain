"""Pygame event loop — the only place that knows about the screen."""
import sys
import numpy as np
import pygame

import sim
from sim import new_world, tick as sim_tick, init_ane, DRAIN_SCALE
from game.renderer import draw_organism, draw_rays, draw_food
from game.panel import draw_panel, PANEL_W
from game.snapshot import save_snapshot, load_snapshot

FPS          = 60
SPEED_STEPS  = [1, 5, 20, 0]   # 0 = headless (no render)
TOTAL_W      = sim.WIDTH + PANEL_W
HIST_MAX     = 300


def main():
    ane_ok = init_ane()
    print("[ANE] CoreML brain ready" if ane_ok else "[ANE] Using numpy fallback")

    pygame.init()
    surf    = pygame.display.set_mode((TOTAL_W, sim.HEIGHT))
    pygame.display.set_caption("beetle-brain  |  wight")
    clock   = pygame.time.Clock()
    font    = pygame.font.SysFont("monospace", 12)
    font_sm = pygame.font.SysFont("monospace", 10)
    font_lg = pygame.font.SysFont("monospace", 14)

    rng = np.random.default_rng()

    pop, food, vents, tick, history, hall_fame = load_snapshot(rng)
    if pop is None:
        pop, food, vents = new_world(rng)
        tick             = 0
        history          = []
        hall_fame        = []
    lineage_history = []
    last_pop        = pop

    sel_idx       = None
    sim_speed_idx = 0

    while True:
        # ── game over ────────────────────────────────────────────────────────
        if len(pop['x']) == 0:
            # redraw last known world state so panel stays readable
            surf.fill((10, 10, 18))
            draw_food(surf, food, vents)
            draw_panel(surf, font, font_sm, font_lg, tick, last_pop, sel_idx,
                       history, lineage_history, hall_fame, 0, vents=vents)
            _draw_extinction_overlay(surf, font, font_lg, tick)
            pygame.display.flip()
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                btn_rect = pygame.Rect(sim.WIDTH // 2 - 80, sim.HEIGHT // 2 + 50, 160, 40)
                if (event.type == pygame.KEYDOWN and event.key == pygame.K_r) or \
                   (event.type == pygame.MOUSEBUTTONDOWN and btn_rect.collidepoint(event.pos)):
                    pop, food, vents = new_world(rng)
                    tick = 0; history = []; hall_fame = []; lineage_history = []; sel_idx = None
            continue

        # ── events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                save_snapshot(pop, food, vents, tick, history, hall_fame)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                result = load_snapshot(rng)
                if result[0] is not None:
                    pop, food, vents, tick, history, hall_fame = result
                    sel_idx = None
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if mx < sim.WIDTH and len(pop['x']) > 0:
                    dists   = np.hypot(pop['x'] - mx, pop['y'] - my)
                    idx     = int(dists.argmin())
                    sel_idx = idx if dists[idx] < 40 else None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                sim_speed_idx = (sim_speed_idx + 1) % len(SPEED_STEPS)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                pop, food, vents = new_world(rng)
                tick = 0; history = []; hall_fame = []; lineage_history = []; sel_idx = None

        # ── tick ─────────────────────────────────────────────────────────────
        steps = SPEED_STEPS[sim_speed_idx] or 80
        for _ in range(steps):
            if len(pop['x']) > 0:
                last_pop = pop
            pop, food = sim_tick(pop, food, vents, rng)
            tick += 1
            if len(pop['x']) == 0:
                break
            # ── history (inside loop so no ticks are skipped) ─────────────────
            if tick % 30 == 0:
                history.append((
                    float(tick),
                    float(len(pop['x'])),
                    float(pop['generation'].max()),
                    float(pop['speed'].mean()),
                    float(pop['fov'].mean()),
                    float(pop['size'].mean()),
                    float(pop['mutation_rate'].mean()),
                ))
                lineage_history.append(np.bincount(pop['lineage_id'], minlength=sim.N_START).astype(np.float32))
                if len(lineage_history) > HIST_MAX:
                    lineage_history.pop(0)
                if len(history) > HIST_MAX:
                    history.pop(0)

        if sel_idx is not None and sel_idx >= len(pop['x']):
            sel_idx = None

        if len(pop['x']) > 0:
            top_idx = np.where(pop['eaten'] > 0)[0]
            for i in top_idx:
                hall_fame.append((
                    int(pop['eaten'][i]), int(pop['generation'][i]),
                    int(pop['age'][i]),
                    float(pop['speed'][i]), float(pop['fov'][i]),
                    float(pop['size'][i]),  float(DRAIN_SCALE * pop['size'][i]**0.75),
                    int(pop['r'][i]), int(pop['g'][i]), int(pop['b'][i]),
                ))
            hall_fame.sort(key=lambda x: -x[0])
            hall_fame = hall_fame[:5]

        # ── draw (skip if headless) ──────────────────────────────────────────
        if SPEED_STEPS[sim_speed_idx] == 0 and tick % 300 != 0:
            clock.tick()
            continue

        surf.fill((10, 10, 18))
        draw_food(surf, food, vents)

        if sel_idx is not None and sel_idx < len(pop['x']):
            draw_rays(surf, pop['x'][sel_idx], pop['y'][sel_idx],
                      pop['fov'][sel_idx], pop['angle'][sel_idx], pop['ray_len'][sel_idx])

        for i in range(len(pop['x'])):
            draw_organism(surf, pop['x'][i], pop['y'][i], pop['angle'][i],
                          pop['size'][i], int(pop['r'][i]), int(pop['g'][i]), int(pop['b'][i]),
                          int(pop['lineage_id'][i]))

        if sel_idx is not None and sel_idx < len(pop['x']):
            pygame.draw.circle(surf, (255, 255, 0),
                               (int(pop['x'][sel_idx]), int(pop['y'][sel_idx])),
                               int(pop['size'][sel_idx]) + 3, 1)

        draw_panel(surf, font, font_sm, font_lg, tick, pop, sel_idx,
                   history, lineage_history, hall_fame, SPEED_STEPS[sim_speed_idx],
                   vents=vents)

        pygame.display.flip()
        clock.tick(FPS)


def _draw_extinction_overlay(surf, font, font_lg, tick):
    """Semi-transparent overlay on the sim area only — panel stays readable."""
    cx, cy = sim.WIDTH // 2, sim.HEIGHT // 2
    overlay = pygame.Surface((sim.WIDTH, sim.HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surf.blit(overlay, (0, 0))
    label = font_lg.render("EXTINCTION", True, (220, 80, 80))
    surf.blit(label, label.get_rect(center=(cx, cy - 30)))
    sub = font.render(f"survived {tick:,} ticks", True, (180, 160, 160))
    surf.blit(sub, sub.get_rect(center=(cx, cy + 4)))
    btn_rect = pygame.Rect(cx - 80, cy + 36, 160, 34)
    pygame.draw.rect(surf, (40, 100, 40), btn_rect, border_radius=5)
    btn_lbl = font.render("R  restart", True, (200, 240, 200))
    surf.blit(btn_lbl, btn_lbl.get_rect(center=btn_rect.center))
