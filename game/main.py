"""Pygame event loop — the only place that knows about the screen."""
import sys
import time
import numpy as np
import pygame

import sim
from sim import new_world, tick as sim_tick, init_ane, DRAIN_SCALE
from sim import phylo
from sim.stats import StatsCollector, SAMPLE_EVERY
from game.renderer import draw_organism, draw_rays, draw_food
from game.panel import draw_panel, PANEL_W
from game.panel.hud import _anc_color
from game.snapshot import save_snapshot, load_snapshot
from report import generate as generate_report

FPS         = 60
SPEED_STEPS = [1, 5, 20, 100]   # ticks per frame
TOTAL_W     = sim.WIDTH + PANEL_W
HIST_MAX    = 300


def _pca_proj(W_body):
    """(N, 18) → (N, 2): project onto top 2 PCs."""
    if len(W_body) == 0:
        return None
    if len(W_body) == 1:
        return np.zeros((1, 2), dtype=np.float32)
    W = W_body.astype(np.float32)
    W -= W.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(W, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    return (W @ Vt[:2].T).astype(np.float32)


# ── main ──────────────────────────────────────────────────────────────────────

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

    world, tick, history, hall_fame = load_snapshot(rng)
    if world is None:
        world     = new_world()
        tick      = 0
        history   = []
        hall_fame = []

    stats         = StatsCollector()
    next_sample   = SAMPLE_EVERY
    lineage_hist  = []
    t_start       = time.time()
    sim_speed_idx = 0
    sel_idx       = None
    extinction_reported = False
    
    last_pca_tick = -999
    cached_pca_proj = None

    while True:
        speed = SPEED_STEPS[sim_speed_idx]
        pop   = world['pop']

        # ── events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_snapshot(world, tick, history, hall_fame)
                _exit_with_report(stats, tick, world, t_start, extinct=False)
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                save_snapshot(world, tick, history, hall_fame)
                _exit_with_report(stats, tick, world, t_start, extinct=False)
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                result = load_snapshot(rng)
                if result[0] is not None:
                    world, tick, history, hall_fame = result
                    stats = StatsCollector(); next_sample = SAMPLE_EVERY
                    lineage_hist = []; t_start = time.time(); sel_idx = None
                    
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if mx < sim.WIDTH and len(pop['x']) > 0:
                    dists   = np.hypot(pop['x'] - mx, pop['y'] - my)
                    idx     = int(dists.argmin())
                    sel_idx = idx if dists[idx] < 40 else None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                sim_speed_idx = (sim_speed_idx + 1) % len(SPEED_STEPS)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                world = new_world(); tick = 0; history = []; hall_fame = []
                stats = StatsCollector(); next_sample = SAMPLE_EVERY
                lineage_hist = []; t_start = time.time()
                sel_idx = None; extinction_reported = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                path = f"screenshot_s{world.get('seed', 0)}_{tick:07d}.png"
                pygame.image.save(surf, path)
                print(f"screenshot → {path}")

        # ── sim ticks ─────────────────────────────────────────────────────────
        if len(pop['x']) > 0:
            for _ in range(speed):
                world = sim_tick(world, rng)
                if len(world['pop']['x']) == 0:
                    break
            pop   = world['pop']
            tick += speed

            # stats
            if tick >= next_sample and len(pop['x']) > 0:
                stats.record(tick, pop, world['phylo'])
                next_sample += SAMPLE_EVERY

            # history
            if tick % 30 < speed and len(pop['x']) > 0:
                history.append((
                    float(tick), float(len(pop['x'])),
                    float(pop['generation'].max()),
                    float(pop['speed'].mean()), float(pop['fov'].mean()),
                    float(pop['size'].mean()), float(pop['mutation_rate'].mean()),
                ))
                depth = max(4, int(pop['generation'].max()) // 3)
                anc   = phylo.ancestor_at(pop['individual_id'], depth, world['phylo'])
                u, c  = np.unique(anc, return_counts=True)
                lineage_hist.append(dict(zip(u.tolist(), c.tolist())))
                if len(lineage_hist) > HIST_MAX: lineage_hist.pop(0)
                if len(history)      > HIST_MAX: history.pop(0)

            # hall of fame
            if len(pop['x']) > 0:
                for i in np.where(pop['eaten'] > 0)[0]:
                    hall_fame.append((
                        int(pop['eaten'][i]), int(pop['generation'][i]),
                        int(pop['age'][i]),
                        float(pop['speed'][i]), float(pop['fov'][i]),
                        float(pop['size'][i]), float(DRAIN_SCALE * pop['size'][i]**0.75),
                        int(pop['r'][i]), int(pop['g'][i]), int(pop['b'][i]),
                    ))
                hall_fame.sort(key=lambda x: -x[0])
                hall_fame = hall_fame[:5]

        # ── extinction ────────────────────────────────────────────────────────
        if len(world['pop']['x']) == 0:
            if not extinction_reported:
                _exit_with_report(stats, tick, world, t_start, extinct=True)
                extinction_reported = True
            _draw_extinction_overlay(surf, font, font_lg, tick)
            pygame.display.flip()
            clock.tick(FPS)
            continue

        if sel_idx is not None and sel_idx >= len(pop['x']):
            sel_idx = None

        # ── draw ──────────────────────────────────────────────────────────────
        surf.fill((10, 14, 20))  # deep sea color

        if getattr(sim, 'COASTLINE_X', None) is not None:
            # Draw the sunny land biome
            land_rect = pygame.Rect(sim.COASTLINE_X, 0, sim.WIDTH - sim.COASTLINE_X, sim.HEIGHT)
            pygame.draw.rect(surf, (30, 30, 20), land_rect)
            
            # Draw a subtle shore edge
            pygame.draw.line(surf, (40, 45, 30), (sim.COASTLINE_X, 0), (sim.COASTLINE_X, sim.HEIGHT), 2)

        draw_food(surf, world['food'], world['vents'])

        # Compute common anc_ids to share between functions
        anc_ids = None
        if len(pop['x']) > 0:
            depth = max(4, int(pop['generation'].max()) // 3)
            anc_ids = phylo.ancestor_at(pop['individual_id'], depth, world['phylo'])
            
        _draw_organisms(surf, pop, world['phylo'], sel_idx, anc_ids=anc_ids)

        if len(pop['x']) > 0:
            if tick - last_pca_tick >= 15 or cached_pca_proj is None or len(cached_pca_proj) != len(pop['x']):
                cached_pca_proj = _pca_proj(pop['W_body'])
                last_pca_tick = tick
            pca_proj = cached_pca_proj
        else:
            pca_proj = None

        sel_wb   = pop['W_body'][sel_idx].copy() if sel_idx is not None and sel_idx < len(pop['x']) else None

        draw_panel(surf, font, font_sm, font_lg, tick, pop, sel_idx,
                   history, lineage_hist, hall_fame, speed,
                   vents=world['vents'], phylo_state=world['phylo'], seed=world.get('seed'),
                   pca_proj=pca_proj, sel_W_body=sel_wb, anc_ids=anc_ids)

        pygame.display.flip()
        clock.tick(FPS)


def _draw_organisms(surf, pop, phylo_state, sel_idx, anc_ids=None):
    if sel_idx is not None and sel_idx < len(pop['x']):
        draw_rays(surf, pop['x'][sel_idx], pop['y'][sel_idx],
                  pop['fov'][sel_idx], pop['angle'][sel_idx], pop['ray_len'][sel_idx])
    if len(pop['x']) > 0:
        if anc_ids is None:
            depth   = max(4, int(pop['generation'].max()) // 3)
            anc_ids = phylo.ancestor_at(pop['individual_id'], depth, phylo_state)
        
        halo_colors = [_anc_color(int(a), phylo_state) for a in anc_ids]
        for i in range(len(pop['x'])):
            draw_organism(surf, pop['x'][i], pop['y'][i], pop['angle'][i],
                          pop['size'][i], int(pop['r'][i]), int(pop['g'][i]),
                          int(pop['b'][i]), halo_colors[i])
    if sel_idx is not None and sel_idx < len(pop['x']):
        pygame.draw.circle(surf, (255, 255, 0),
                           (int(pop['x'][sel_idx]), int(pop['y'][sel_idx])),
                           int(pop['size'][sel_idx]) + 3, 1)


def _exit_with_report(stats, tick, world, t_start, extinct):
    elapsed = time.time() - t_start
    pop = world['pop'] if not extinct else None
    stats.finalize(tick, elapsed, pop=pop, phylo_state=world['phylo'], extinct=extinct,
                   seed=world.get('seed'))
    generate_report(stats)


def _draw_extinction_overlay(surf, font, font_lg, tick):
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
if __name__ == '__main__': main()
