"""Main stats panel drawn on the right side of the screen."""
import numpy as np
import pygame
from sim.config import SPEED_MAX, SIZE_MAX, MUTATION_RATE_MAX, DRAIN_SCALE, N_START
from sim.population.genome import N_BODY
from game.panel.sparkline import draw_sparkline

PANEL_W = 320

_LINEAGE_COLORS = [
    (255, 100, 100), (100, 200, 255), (100, 255, 140), (255, 200,  60),
    (200, 100, 255), (255, 150,  50), ( 80, 220, 200), (220, 220, 100),
    (180,  80, 180), (100, 180, 100), (255, 120, 180), (140, 180, 255),
]


def _draw_stacked_area(surf, lineage_history, rect, n_lineages):
    """Stacked area chart: each lineage a coloured band, time on x-axis."""
    if len(lineage_history) < 2:
        return
    rx, ry, rw, rh = rect
    pygame.draw.rect(surf, (10, 10, 20), rect)

    T   = len(lineage_history)
    arr = np.array(lineage_history)          # (T, N_START)
    totals = arr.sum(axis=1).clip(min=1)

    for t in range(T):
        x   = rx + int(t * rw / T)
        x1  = rx + int((t + 1) * rw / T)
        bot = ry + rh
        for lid in range(n_lineages):
            if arr[t, lid] == 0:
                continue
            h   = max(1, int(arr[t, lid] / totals[t] * rh))
            top = bot - h
            color = _LINEAGE_COLORS[lid % len(_LINEAGE_COLORS)]
            pygame.draw.rect(surf, color, (x, top, max(1, x1 - x), h))
            bot = top

    pygame.draw.rect(surf, (50, 50, 80), rect, 1)


def draw_panel(surf, font, font_sm, font_lg, tick, pop, sel_idx,
               history, lineage_history, hall_fame, sim_speed=1):
    px = surf.get_width() - PANEL_W
    pygame.draw.rect(surf, (16, 16, 28), (px, 0, PANEL_W, surf.get_height()))
    pygame.draw.line(surf, (50, 50, 80), (px, 0), (px, surf.get_height()), 1)

    y = 10

    def txt(s, f=None, color=(180, 180, 200)):
        nonlocal y
        surf.blit((f or font).render(s, True, color), (px + 10, y))
        y += (f or font).get_height() + 2

    def sep():
        nonlocal y
        pygame.draw.line(surf, (40, 40, 60), (px + 8, y + 2), (px + PANEL_W - 8, y + 2), 1)
        y += 8

    speed_label = "HEADLESS" if sim_speed == 0 else f"{sim_speed}x"
    speed_color = (255, 180, 50) if sim_speed != 1 else (120, 120, 150)
    txt("BEETLE-BRAIN", font_lg, (220, 220, 255))
    txt(f"tick {tick:,}   [{speed_label}]  SPACE=cycle", font_sm, speed_color)
    sep()

    N = len(pop['x'])
    if N > 0:
        counts    = np.bincount(pop['lineage_id'], minlength=N_START)
        surviving = int((counts > 0).sum())

        txt("POPULATION", font, (160, 200, 160))
        txt(f"  count   {N:4d}    lineages  {surviving}/{N_START}", font_sm)
        txt(f"  max gen {int(pop['generation'].max()):4d}", font_sm)
        txt(f"  max age {int(pop['age'].max()):6d}", font_sm)
        txt(f"  max ate {int(pop['eaten'].max()):4d}", font_sm)
        sep()

        # ── stacked area: lineage populations over time ───────────────────────
        txt("LINEAGES over time", font, (200, 180, 140))
        chart_h = 90
        _draw_stacked_area(surf, lineage_history,
                           (px + 8, y, PANEL_W - 16, chart_h), N_START)
        y += chart_h + 4
        # legend: surviving lineages as small coloured dots + count
        lx = px + 10
        for lid in range(N_START):
            if counts[lid] == 0:
                continue
            color = _LINEAGE_COLORS[lid % len(_LINEAGE_COLORS)]
            pygame.draw.circle(surf, color, (lx + 4, y + 5), 4)
            surf.blit(font_sm.render(f"{counts[lid]}", True, color), (lx + 11, y))
            lx += 38
            if lx > px + PANEL_W - 38:
                lx = px + 10
                y += 14
        y += 16
        sep()

        # ── trait sparklines ─────────────────────────────────────────────────
        txt("TRAITS  (pop avg)", font, (160, 180, 220))
        if len(history) > 1:
            for label, col_idx, color, lo, hi in [
                ("speed",    3, (100, 200, 255), 0, SPEED_MAX),
                ("size",     5, (255, 180, 100), 0, SIZE_MAX),
                ("mut rate", 6, (255, 100, 100), 0, MUTATION_RATE_MAX),
            ]:
                data = [h[col_idx] for h in history]
                lbl_surf = font_sm.render(f"  {label}", True, color)
                surf.blit(lbl_surf, (px + 10, y))
                draw_sparkline(surf, data, (px + 90, y, PANEL_W - 100, 14), color, lo, hi)
                y += 18
        sep()

    if hall_fame:
        txt("HALL OF FAME", font, (255, 210, 80))
        for i, (eaten, gen, age, spd, fov, sz, drn, r, g, b) in enumerate(hall_fame):
            pygame.draw.circle(surf, (r, g, b), (px + 18, y + 6), 5)
            txt(f"  #{i+1} ate:{eaten:3d} g{gen} spd{spd:.1f} sz{sz:.1f}", font_sm, (220, 200, 140))
        sep()

    if sel_idx is not None and sel_idx < N:
        lid  = int(pop['lineage_id'][sel_idx])
        lcol = _LINEAGE_COLORS[lid % len(_LINEAGE_COLORS)]
        txt("SELECTED", font, (255, 255, 100))
        sr, sg, sb = int(pop['r'][sel_idx]), int(pop['g'][sel_idx]), int(pop['b'][sel_idx])
        pygame.draw.circle(surf, lcol,    (px + 18, y + 6), max(2, int(pop['size'][sel_idx])) + 2, 1)
        pygame.draw.circle(surf, (sr, sg, sb), (px + 18, y + 6), max(1, int(pop['size'][sel_idx])))
        y += 4
        for row in [
            f"  lineage {lid}",
            f"  gen    {int(pop['generation'][sel_idx])}",
            f"  age    {int(pop['age'][sel_idx]):,}",
            f"  eaten  {int(pop['eaten'][sel_idx])}",
            f"  energy {pop['energy'][sel_idx]:.0f}",
            f"  speed  {pop['speed'][sel_idx]:.2f}",
            f"  fov    {np.degrees(pop['fov'][sel_idx]):.0f}°",
            f"  size   {pop['size'][sel_idx]:.1f}",
            f"  drain  {DRAIN_SCALE * pop['size'][sel_idx]**0.75:.3f}",
            f"  mut    {pop['mutation_rate'][sel_idx]:.2f}",
        ]:
            txt(row, font_sm, (200, 200, 120))
        sep()

        txt("GENOME  W_body", font_sm, (140, 160, 180))
        bw    = (PANEL_W - 20) // N_BODY
        bar_x = px + 10
        bar_y = y
        for j, (lbl, w_val) in enumerate(zip(
            ["spd", "fov", "ray", "sz", "r", "g", "b", "trn", "brd"],
            pop['W_body'][sel_idx],
        )):
            h_bar     = int(abs(float(w_val)) * 12)
            color_bar = (100, 200, 100) if w_val > 0 else (200, 80, 80)
            pygame.draw.rect(surf, color_bar,
                             (bar_x + j * bw + 1, bar_y + 20 - h_bar, bw - 2, h_bar))
            surf.blit(font_sm.render(lbl, True, (120, 120, 140)),
                      (bar_x + j * bw, bar_y + 22))
        y = bar_y + 38

    y = surf.get_height() - 30
    txt("S save  L load  R restart  click inspect", font_sm, (80, 80, 100))
