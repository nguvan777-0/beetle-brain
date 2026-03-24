"""Main stats panel drawn on the right side of the screen."""
import numpy as np
import pygame
from sim.config import SPEED_MAX, SIZE_MAX, MUTATION_RATE_MAX, DRAIN_SCALE
from sim.population.genome import N_BODY
from game.panel.sparkline import draw_sparkline

PANEL_W = 320


def draw_panel(surf, font, font_sm, font_lg, tick, pop, sel_idx,
               history, hall_fame, sim_speed=1):
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
        txt("POPULATION", font, (160, 200, 160))
        txt(f"  count   {N:4d}", font_sm)
        txt(f"  max gen {int(pop['generation'].max()):4d}", font_sm)
        txt(f"  max age {int(pop['age'].max()):6d}", font_sm)
        txt(f"  max ate {int(pop['eaten'].max()):4d}", font_sm)
        sep()

        txt("TRAIT TRENDS  (pop avg)", font, (160, 180, 220))
        if len(history) > 1:
            for label, col_idx, color, lo, hi in [
                ("speed",    3, (100, 200, 255), 0,    SPEED_MAX),
                ("fov °",    4, (200, 160, 255), 0,    180),
                ("size",     5, (255, 180, 100), 0,    SIZE_MAX),
                ("mut rate", 6, (255, 100, 100), 0,    MUTATION_RATE_MAX),
            ]:
                data = [h[col_idx] for h in history]
                if label == "fov °":
                    data = [np.degrees(v) for v in data]
                txt(f"  {label}", font_sm, color)
                draw_sparkline(surf, data, (px + 16, y, PANEL_W - 60, 22), color, lo, hi)
                y += 28
        sep()

    if hall_fame:
        txt("HALL OF FAME", font, (255, 210, 80))
        for i, (eaten, gen, age, spd, fov, sz, drn, r, g, b) in enumerate(hall_fame):
            pygame.draw.circle(surf, (r, g, b), (px + 18, y + 6), 5)
            txt(f"  #{i+1} ate:{eaten:3d} g{gen} spd{spd:.1f} sz{sz:.1f}", font_sm, (220, 200, 140))
        sep()

    if sel_idx is not None and sel_idx < N:
        txt("SELECTED", font, (255, 255, 100))
        sr, sg, sb = int(pop['r'][sel_idx]), int(pop['g'][sel_idx]), int(pop['b'][sel_idx])
        pygame.draw.circle(surf, (sr, sg, sb), (px + 18, y + 6), max(1, int(pop['size'][sel_idx])))
        y += 4
        for row in [
            f"  gen    {int(pop['generation'][sel_idx])}",
            f"  age    {int(pop['age'][sel_idx]):,}",
            f"  eaten  {int(pop['eaten'][sel_idx])}",
            f"  energy {pop['energy'][sel_idx]:.0f}",
            f"  speed  {pop['speed'][sel_idx]:.2f}",
            f"  fov    {np.degrees(pop['fov'][sel_idx]):.0f}°",
            f"  ray    {pop['ray_len'][sel_idx]:.0f}",
            f"  size   {pop['size'][sel_idx]:.1f}",
            f"  drain  {DRAIN_SCALE * pop['size'][sel_idx]**0.75:.3f}  (size-derived)",
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
    txt("S save  L load  click inspect", font_sm, (80, 80, 100))
