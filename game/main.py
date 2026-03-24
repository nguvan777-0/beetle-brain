"""Pygame event loop — the only place that knows about the screen."""
import sys
import numpy as np
import pygame

import sim
from sim import new_world, tick as sim_tick, init_ane
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

    pop, food, tick, history, hall_fame = load_snapshot(rng)
    if pop is None:
        pop, food = new_world(rng)
        tick      = 0
        history   = []
        hall_fame = []

    sel_idx       = None
    sim_speed_idx = 0

    while True:
        # ── game over ────────────────────────────────────────────────────────
        if len(pop['x']) == 0:
            _draw_extinction(surf, font, font_lg, tick, clock, FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                btn_rect = pygame.Rect(TOTAL_W // 2 - 80, sim.HEIGHT // 2 + 50, 160, 40)
                if (event.type == pygame.KEYDOWN and event.key == pygame.K_r) or \
                   (event.type == pygame.MOUSEBUTTONDOWN and btn_rect.collidepoint(event.pos)):
                    pop, food = new_world(rng)
                    tick = 0; history = []; hall_fame = []; sel_idx = None
            continue

        # ── events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                save_snapshot(pop, food, tick, history, hall_fame)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                result = load_snapshot(rng)
                if result[0] is not None:
                    pop, food, tick, history, hall_fame = result
                    sel_idx = None
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if mx < sim.WIDTH and len(pop['x']) > 0:
                    dists   = np.hypot(pop['x'] - mx, pop['y'] - my)
                    idx     = int(dists.argmin())
                    sel_idx = idx if dists[idx] < 40 else None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                sim_speed_idx = (sim_speed_idx + 1) % len(SPEED_STEPS)

        # ── tick ─────────────────────────────────────────────────────────────
        steps = SPEED_STEPS[sim_speed_idx] or 80
        for _ in range(steps):
            pop, food = sim_tick(pop, food, rng)
            tick += 1
            if len(pop['x']) == 0:
                break

        if sel_idx is not None and sel_idx >= len(pop['x']):
            sel_idx = None

        # ── history ──────────────────────────────────────────────────────────
        if tick % 30 == 0 and len(pop['x']) > 0:
            history.append((
                float(tick),
                float(len(pop['x'])),
                float(pop['generation'].max()),
                float(pop['speed'].mean()),
                float(pop['fov'].mean()),
                float(pop['size'].mean()),
                float(pop['drain'].mean()),
            ))
            if len(history) > HIST_MAX:
                history.pop(0)
            top_idx = np.where(pop['eaten'] > 0)[0]
            for i in top_idx:
                hall_fame.append((
                    int(pop['eaten'][i]), int(pop['generation'][i]),
                    int(pop['age'][i]),
                    float(pop['speed'][i]), float(pop['fov'][i]),
                    float(pop['size'][i]),  float(pop['drain'][i]),
                    int(pop['r'][i]), int(pop['g'][i]), int(pop['b'][i]),
                ))
            hall_fame.sort(key=lambda x: -x[0])
            hall_fame = hall_fame[:5]

        # ── draw (skip if headless) ──────────────────────────────────────────
        if SPEED_STEPS[sim_speed_idx] == 0 and tick % 300 != 0:
            clock.tick()
            continue

        surf.fill((10, 10, 18))
        draw_food(surf, food)

        if sel_idx is not None and sel_idx < len(pop['x']):
            draw_rays(surf, pop['x'][sel_idx], pop['y'][sel_idx],
                      pop['fov'][sel_idx], pop['angle'][sel_idx], pop['ray_len'][sel_idx])

        for i in range(len(pop['x'])):
            draw_organism(surf, pop['x'][i], pop['y'][i], pop['angle'][i],
                          pop['size'][i], int(pop['r'][i]), int(pop['g'][i]), int(pop['b'][i]))

        if sel_idx is not None and sel_idx < len(pop['x']):
            pygame.draw.circle(surf, (255, 255, 0),
                               (int(pop['x'][sel_idx]), int(pop['y'][sel_idx])),
                               int(pop['size'][sel_idx]) + 3, 1)

        draw_panel(surf, font, font_sm, font_lg, tick, pop, sel_idx,
                   history, hall_fame, SPEED_STEPS[sim_speed_idx])

        pygame.display.flip()
        clock.tick(FPS)


def _draw_extinction(surf, font, font_lg, tick, clock, fps):
    cx, cy = surf.get_width() // 2, surf.get_height() // 2
    surf.fill((10, 10, 18))
    label = font_lg.render("EXTINCTION", True, (200, 60, 60))
    surf.blit(label, label.get_rect(center=(cx, cy - 40)))
    sub = font.render(f"survived {tick:,} ticks", True, (160, 160, 180))
    surf.blit(sub, sub.get_rect(center=(cx, cy + 10)))
    btn_rect = pygame.Rect(cx - 80, cy + 50, 160, 40)
    pygame.draw.rect(surf, (50, 120, 50), btn_rect, border_radius=6)
    btn_lbl = font.render("R  restart", True, (220, 255, 220))
    surf.blit(btn_lbl, btn_lbl.get_rect(center=btn_rect.center))
    pygame.display.flip()
    clock.tick(fps)
