"""Main stats panel drawn on the right side of the screen."""
import colorsys
import numpy as np
import pygame
from sim import phylo

_PHYLO_DEPTH = 15   # generations to walk back when coloring PCA dots
from sim.config import (
    SPEED_MIN, SPEED_MAX, FOV_MIN, FOV_MAX, RAY_MIN, RAY_MAX,
    SIZE_MIN, SIZE_MAX, MUTATION_RATE_MIN, MUTATION_RATE_MAX,
    MUTATION_SCALE_MIN, MUTATION_SCALE_MAX, EPIGENETIC_MIN, EPIGENETIC_MAX,
    WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX, MOUTH_MIN, MOUTH_MAX,
    ENERGY_MAX_MIN, ENERGY_MAX_MAX, PRED_RATIO_MIN, PRED_RATIO_MAX,
    SPEED_SCALE_MIN, SPEED_SCALE_MAX, BREED_AT_MIN, BREED_AT_MAX,
    CLONE_WITH_MIN, CLONE_WITH_MAX, DRAIN_SCALE, N_START, WIDTH, HEIGHT, VENT_RADIUS,
)
from sim.population.genome import N_BODY
from game.panel.sparkline import draw_sparkline

PANEL_W = 320

_LINEAGE_COLORS = [
    (255, 100, 100), (100, 200, 255), (100, 255, 140), (255, 200,  60),
    (200, 100, 255), (255, 150,  50), ( 80, 220, 200), (220, 220, 100),
    (180,  80, 180), (100, 180, 100), (255, 120, 180), (140, 180, 255),
]


def _draw_vent_map(surf, pop, vents, rect):
    """Minimap: world outline, vents coloured by dominant lineage."""
    rx, ry, rw, rh = rect
    pygame.draw.rect(surf, (10, 10, 20), rect)
    pygame.draw.rect(surf, (40, 40, 60), rect, 1)

    sx = rw / WIDTH
    sy = rh / HEIGHT

    # draw each vent circle, coloured by dominant lineage
    for vx, vy in vents:
        mx = rx + int(vx * sx)
        my = ry + int(vy * sy)
        vr = max(3, int(VENT_RADIUS * sx))

        if len(pop['x']) > 0:
            dx = pop['x'] - vx
            dy = pop['y'] - vy
            nearby = np.where(dx*dx + dy*dy < VENT_RADIUS*VENT_RADIUS)[0]
            if len(nearby) > 0:
                lids   = pop['lineage_id'][nearby]
                winner = int(np.bincount(lids, minlength=N_START).argmax())
                color  = _LINEAGE_COLORS[winner % len(_LINEAGE_COLORS)]
            else:
                color = (40, 40, 60)
        else:
            color = (40, 40, 60)

        pygame.draw.circle(surf, color, (mx, my), vr, 2)
        pygame.draw.circle(surf, (*[c//3 for c in color], 80), (mx, my), vr)

    # draw all wights as 1px dots
    for i in range(len(pop['x'])):
        lid   = int(pop['lineage_id'][i])
        color = _LINEAGE_COLORS[lid % len(_LINEAGE_COLORS)]
        wx    = rx + int(pop['x'][i] * sx)
        wy    = ry + int(pop['y'][i] * sy)
        surf.set_at((wx, wy), color)


def _lerp_color(t, lo=(60, 100, 200), mid=(60, 200, 120), hi=(220, 80, 60)):
    """Blue→green→red gradient for 0→1."""
    if t < 0.5:
        s = t * 2
        return tuple(int(lo[i] + (mid[i] - lo[i]) * s) for i in range(3))
    s = (t - 0.5) * 2
    return tuple(int(mid[i] + (hi[i] - mid[i]) * s) for i in range(3))


def _draw_trait_heatmap(surf, pop, rect, font_sm):
    """One row per trait: p10–p90 band + median tick, colored by median position."""
    if len(pop['x']) < 2:
        return
    rx, ry, rw, rh = rect

    traits = [
        ('speed',  pop['speed'],                   SPEED_MIN,        SPEED_MAX),
        ('fov',    np.degrees(pop['fov']),          np.degrees(FOV_MIN), np.degrees(FOV_MAX)),
        ('ray',    pop['ray_len'],                  RAY_MIN,          RAY_MAX),
        ('size',   pop['size'],                     SIZE_MIN,         SIZE_MAX),
        ('turn',   pop['turn_s'],                   0.05,             0.30),
        ('breed@', pop['breed_at'],                 BREED_AT_MIN,     BREED_AT_MAX),
        ('clone',  pop['clone_with'],               CLONE_WITH_MIN,   CLONE_WITH_MAX),
        ('mut.r',  pop['mutation_rate'],            MUTATION_RATE_MIN, MUTATION_RATE_MAX),
        ('mut.s',  pop['mutation_scale'],           MUTATION_SCALE_MIN, MUTATION_SCALE_MAX),
        ('epig',   pop['epigenetic'],               EPIGENETIC_MIN,   EPIGENETIC_MAX),
        ('decay',  pop['weight_decay'],             WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX),
        ('mouth',  pop['mouth'],                    MOUTH_MIN,        MOUTH_MAX),
        ('emax',   pop['energy_max'],               ENERGY_MAX_MIN,   ENERGY_MAX_MAX),
        ('pred×',  pop['pred_ratio'],               PRED_RATIO_MIN,   PRED_RATIO_MAX),
        ('spscl',  pop['speed_scale'],              SPEED_SCALE_MIN,  SPEED_SCALE_MAX),
    ]

    lbl_w  = 38
    bar_x  = rx + lbl_w
    bar_w  = rw - lbl_w - 4
    row_h  = max(6, rh // len(traits))

    for i, (name, vals, lo, hi) in enumerate(traits):
        ty   = ry + i * row_h
        span = hi - lo if hi != lo else 1.0

        p10    = float(np.percentile(vals, 10))
        median = float(np.percentile(vals, 50))
        p90    = float(np.percentile(vals, 90))

        n_p10    = (p10    - lo) / span
        n_med    = (median - lo) / span
        n_p90    = (p90    - lo) / span

        color = _lerp_color(np.clip(n_med, 0, 1))
        dim   = tuple(c // 4 for c in color)

        # background track
        pygame.draw.rect(surf, (20, 20, 32), (bar_x, ty + 1, bar_w, row_h - 2))
        # p10–p90 band
        bx = bar_x + int(np.clip(n_p10, 0, 1) * bar_w)
        bw = max(1, int((np.clip(n_p90, 0, 1) - np.clip(n_p10, 0, 1)) * bar_w))
        pygame.draw.rect(surf, dim, (bx, ty + 1, bw, row_h - 2))
        # median tick
        mx = bar_x + int(np.clip(n_med, 0, 1) * bar_w)
        pygame.draw.rect(surf, color, (mx - 1, ty, 3, row_h))
        # label
        surf.blit(font_sm.render(name, True, (120, 130, 150)), (rx, ty))


def _anc_to_color(ancestor_id: int) -> tuple:
    """Golden-ratio hue hash → RGB.  Deterministic, well-distributed."""
    hue = (int(ancestor_id) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.9)
    return (int(r * 255), int(g * 255), int(b * 255))


def _draw_pca_scatter(surf, pop, rect):
    """Project W_body onto top 2 PCs; colour each dot by phylogenetic sub-lineage.

    Uses phylo.ancestor_at(individual_id, _PHYLO_DEPTH) so wights that share
    a recent common ancestor get the same hue — diverging sub-populations
    naturally get different colours without any threshold decision.
    """
    if len(pop['x']) < 3:
        return
    rx, ry, rw, rh = rect
    pygame.draw.rect(surf, (10, 10, 20), rect)
    pygame.draw.rect(surf, (40, 40, 60), rect, 1)

    W = pop['W_body'].astype(np.float32)
    W -= W.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(W, full_matrices=False)
    except np.linalg.LinAlgError:
        return
    proj = W @ Vt[:2].T          # (N, 2)

    lo  = proj.min(axis=0)
    hi  = proj.max(axis=0)
    span = (hi - lo).clip(min=1e-6)
    xs  = rx + 4 + ((proj[:, 0] - lo[0]) / span[0] * (rw - 8)).astype(int)
    ys  = ry + 4 + ((proj[:, 1] - lo[1]) / span[1] * (rh - 8)).astype(int)

    # colour by ancestor _PHYLO_DEPTH generations back — build color lookup once
    ancestors              = phylo.ancestor_at(pop['individual_id'], _PHYLO_DEPTH)
    unique_anc, inv        = np.unique(ancestors, return_inverse=True)
    color_map              = [_anc_to_color(int(a)) for a in unique_anc]

    for i in range(len(xs)):
        pygame.draw.circle(surf, color_map[inv[i]], (int(xs[i]), int(ys[i])), 2)


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
               history, lineage_history, hall_fame, sim_speed=1, vents=None):
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

        # ── trait heatmap ────────────────────────────────────────────────────
        txt("TRAITS  (median  |  p10–p90 band)", font, (160, 180, 220))
        n_traits = 15
        hmap_h   = n_traits * 9
        _draw_trait_heatmap(surf, pop, (px + 8, y, PANEL_W - 16, hmap_h), font_sm)
        y += hmap_h + 4
        sep()

        # ── PCA scatter ──────────────────────────────────────────────────────
        txt("STRATEGY SPACE  (W_body PCA)", font, (160, 180, 220))
        _draw_pca_scatter(surf, pop, (px + 8, y, PANEL_W - 16, 130))
        y += 134
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
