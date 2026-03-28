"""Main stats panel drawn on the right side of the screen."""
import colorsys
import numpy as np
import pygame
from sim import phylo

_PHYLO_DEPTH = 50   # generations to walk back for sub-lineage coloring
from sim.config import (
    SPEED_MIN, SPEED_MAX, FOV_MIN, FOV_MAX, RAY_MIN, RAY_MAX,
    SIZE_MIN, SIZE_MAX, MUTATION_RATE_MIN, MUTATION_RATE_MAX,
    MUTATION_SCALE_MIN, MUTATION_SCALE_MAX, EPIGENETIC_MIN, EPIGENETIC_MAX,
    WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX, MOUTH_MIN, MOUTH_MAX,
    PRED_RATIO_MIN, PRED_RATIO_MAX, ENERGY_MAX_SCALE,
    BREED_AT_MIN, BREED_AT_MAX,
    CLONE_WITH_MIN, CLONE_WITH_MAX, DRAIN_SCALE, N_START, WIDTH, HEIGHT, VENT_RADIUS,
    HGT_EAT_MIN, HGT_EAT_MAX, HGT_CONTACT_MIN, HGT_CONTACT_MAX,
)
from sim.population.genome import N_BODY
from game.panel.sparkline import draw_sparkline

PANEL_W = 320

_font_title: object = None   # lazily initialised; pygame not ready at import time

def _title_font():
    global _font_title
    if _font_title is None:
        _font_title = pygame.font.SysFont("monospace", 16, bold=True)
    return _font_title

# Rainbow colors for the legend keys: sp · 1x · 2x · 3x · 4x · 5x · s · r
# Red (sp) → violet (r:rst), precomputed once at import time.
_LEGEND_N = 9
_LEGEND_COLORS = [
    (
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / (_LEGEND_N - 1) * 0.75, 0.75, 0.95)),
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / (_LEGEND_N - 1) * 0.75, 0.60, 0.58)),
    )
    for i in range(_LEGEND_N)
]

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


from functools import lru_cache

@lru_cache(maxsize=1024)
def _render_text(text: str, font, color: tuple):
    return font.render(str(text), True, color)

def _lerp_color(t, lo=(60, 100, 200), mid=(60, 200, 120), hi=(220, 80, 60)):
    """Blue→green→red gradient for 0→1."""
    if t < 0.5:
        s = t * 2
        return tuple(int(lo[i] + (mid[i] - lo[i]) * s) for i in range(3))
    s = (t - 0.5) * 2
    return tuple(int(mid[i] + (hi[i] - mid[i]) * s) for i in range(3))


_traits_surf_cache = None

def _draw_trait_heatmap(surf, pop, rect, font_sm):
    """One row per trait: p10–p90 band + median tick, colored by median position."""
    global _traits_surf_cache
    rx, ry, rw, rh = rect

    if len(pop['x']) < 2:
        if _traits_surf_cache is not None:
            surf.blit(_traits_surf_cache, (rx, ry))
        return

    # Create or reuse cache surface
    if _traits_surf_cache is None or _traits_surf_cache.get_size() != (rw, rh):
        _traits_surf_cache = pygame.Surface((rw, rh), pygame.SRCALPHA)
    
    tsurf = _traits_surf_cache
    tsurf.fill((0, 0, 0, 0))  # Clear with transparent

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
        ('emax',   ENERGY_MAX_SCALE * pop['size']**2, 0, ENERGY_MAX_SCALE * SIZE_MAX**2),
        ('pred×',  pop['pred_ratio'],               PRED_RATIO_MIN,   PRED_RATIO_MAX),
        ('hgt-eat',pop['hgt_eat_rate'],             HGT_EAT_MIN,      HGT_EAT_MAX),
        ('hgt-con', pop['hgt_contact_rate'],        HGT_CONTACT_MIN,  HGT_CONTACT_MAX),
    ]

    lbl_w  = 38
    bar_x  = lbl_w
    bar_w  = rw - lbl_w - 4
    row_h  = max(6, rh // len(traits))

    for i, (name, vals, lo, hi) in enumerate(traits):
        ty   = i * row_h
        span = hi - lo if hi != lo else 1.0

        p10, median, p90 = np.percentile(vals, [10, 50, 90])
        p10, median, p90 = float(p10), float(median), float(p90)

        n_p10    = (p10    - lo) / span
        n_med    = (median - lo) / span
        n_p90    = (p90    - lo) / span

        color = _lerp_color(np.clip(n_med, 0, 1))
        dim   = tuple(c // 4 for c in color)

        # background track
        pygame.draw.rect(tsurf, (20, 20, 32), (bar_x, ty + 1, bar_w, row_h - 2))
        # p10–p90 band
        bx = bar_x + int(np.clip(n_p10, 0, 1) * bar_w)
        bw = max(1, int((np.clip(n_p90, 0, 1) - np.clip(n_p10, 0, 1)) * bar_w))
        pygame.draw.rect(tsurf, dim, (bx, ty + 1, bw, row_h - 2))
        # median tick
        mx = bar_x + int(np.clip(n_med, 0, 1) * bar_w)
        pygame.draw.rect(tsurf, color, (mx - 1, ty, 3, row_h))
        # label
        tsurf.blit(_render_text(name, font_sm, (120, 130, 150)), (0, ty))

    surf.blit(tsurf, (rx, ry))


def _hue_to_rgb(hue: float) -> tuple:
    """Phylo hue [0,1] → RGB."""
    r, g, b = colorsys.hsv_to_rgb(float(hue) % 1.0, 0.85, 0.9)
    return (int(r * 255), int(g * 255), int(b * 255))


_anc_color_cache = {}

def _anc_color(ancestor_id: int, phylo_state) -> tuple:
    """Look up lineage color from phylo hue array."""
    from sim.phylo import M
    aid = int(ancestor_id) % M
    hue = phylo_state['hue'][aid]
    
    # Use a cache key based on the actual hue value
    cache_key = float(hue)
    if cache_key not in _anc_color_cache:
        _anc_color_cache[cache_key] = _hue_to_rgb(hue)
    return _anc_color_cache[cache_key]


_pca_surf_cache = None

def _draw_pca_scatter(surf, pop, rect, phylo_state, pca_proj, anc_ids=None):
    """Draw pre-computed PCA projection; colour by phylogenetic sub-lineage."""
    global _pca_surf_cache
    rx, ry, rw, rh = rect

    if pca_proj is None or len(pca_proj) == 0:
        if _pca_surf_cache is not None:
            surf.blit(_pca_surf_cache, (rx, ry))
        return

    # Create or reuse cache surface
    if _pca_surf_cache is None or _pca_surf_cache.get_size() != (rw, rh):
        _pca_surf_cache = pygame.Surface((rw, rh))
    
    psurf = _pca_surf_cache
    pygame.draw.rect(psurf, (10, 10, 20), (0, 0, rw, rh))
    pygame.draw.rect(psurf, (40, 40, 60), (0, 0, rw, rh), 1)

    proj = pca_proj

    lo  = proj.min(axis=0)
    hi  = proj.max(axis=0)
    span = (hi - lo)
    
    # Center points if there's no variance
    xs = np.zeros(len(proj), dtype=int)
    ys = np.zeros(len(proj), dtype=int)
    
    if span[0] < 1e-6:
        xs[:] = rw // 2
    else:
        xs[:] = 4 + ((proj[:, 0] - lo[0]) / span[0] * (rw - 8)).astype(int)
        
    if span[1] < 1e-6:
        ys[:] = rh // 2
    else:
        ys[:] = 4 + ((proj[:, 1] - lo[1]) / span[1] * (rh - 8)).astype(int)

    if anc_ids is None:
        depth                  = max(4, int(pop['generation'].max()) // 3)
        ancestors              = phylo.ancestor_at(pop['individual_id'], depth, phylo_state)
    else:
        ancestors = anc_ids
        
    unique_anc, inv        = np.unique(ancestors, return_inverse=True)
    color_map              = [_anc_color(int(a), phylo_state) for a in unique_anc]

    for i in range(len(xs)):
        pygame.draw.circle(psurf, color_map[inv[i]], (int(xs[i]), int(ys[i])), 2)

    surf.blit(psurf, (rx, ry))


def _draw_stacked_area(surf, lineage_history, rect, phylo_state):
    """Stacked area chart: each phylo sub-lineage a coloured band, time on x-axis."""
    if len(lineage_history) < 2:
        return
    rx, ry, rw, rh = rect
    pygame.draw.rect(surf, (10, 10, 20), rect)

    all_ids = sorted({aid for frame in lineage_history for aid in frame})
    if not all_ids:
        return
        
    color_map = {aid: _anc_color(aid, phylo_state) for aid in all_ids}

    T = len(lineage_history)
    for t in range(T):
        frame  = lineage_history[t]
        total  = max(1, sum(frame.values()))
        x      = rx + int(t * rw / T)
        x1     = rx + int((t + 1) * rw / T)
        bot    = ry + rh
        for aid in all_ids:
            count = frame.get(aid, 0)
            if count == 0:
                continue
            h   = int(count / total * rh)
            if h <= 0:
                continue
            top = bot - h
            pygame.draw.rect(surf, color_map[aid], (x, top, max(1, x1 - x), h))
            bot = top

    pygame.draw.rect(surf, (50, 50, 80), rect, 1)


def draw_panel(surf, font, font_sm, font_lg, tick, pop, sel_idx,
               history, lineage_history, hall_fame, sim_speed=1, vents=None, phylo_state=None,
               seed=None, pca_proj=None, sel_W_body=None, anc_ids=None,
               paused=False, sim_speed_idx=0, snap_active=False, rst_active=False, fps=0):
    px = surf.get_width() - PANEL_W
    pygame.draw.rect(surf, (16, 16, 28), (px, 0, PANEL_W, surf.get_height()))
    pygame.draw.line(surf, (50, 50, 80), (px, 0), (px, surf.get_height()), 1)

    y = 4

    def txt(s, f=None, color=(180, 180, 200)):
        nonlocal y
        surf.blit(_render_text(s, f or font, color), (px + 6, y))
        y += (f or font).get_height() + 2

    def sep():
        nonlocal y
        pygame.draw.line(surf, (40, 40, 60), (px + 6, y + 2), (px + PANEL_W - 6, y + 2), 1)
        y += 8

    tf         = _title_font()
    row_h      = tf.get_height()
    sub_y      = y + (row_h - font_lg.get_height()) // 2   # vertical centre for smaller items

    title_surf = _render_text("BEETLE-BRAIN", tf, (220, 220, 255))
    surf.blit(title_surf, (px + 6, y))

    if seed is not None:
        seed_surf = _render_text(str(seed), font_lg, (80, 95, 130))
        seed_x    = max(px + 6 + title_surf.get_width() + 8,
                        px + (PANEL_W - seed_surf.get_width()) // 2)
        surf.blit(seed_surf, (seed_x, sub_y))

    fps_surf = _render_text(f"{fps:.0f}fps", font_lg, (80, 95, 130))
    surf.blit(fps_surf, (px + PANEL_W - 6 - fps_surf.get_width(), sub_y))

    y += row_h + 4

    # ── key legend — pill buttons, rainbow-colored ───────────────────────
    # key order: 0=sp  1=1x  2=5x  3=20x  4=100x  5=MAX  6=s:snap  7=r:rst
    def _kc(i, active=True):
        return _LEGEND_COLORS[i][0 if active else 1]

    PX2, PY2 = 6, 3   # pill padding x / y
    MARGIN   = 6       # left/right margin inside panel

    def _pill(lx, text_y, label, i, active, f=None):
        """text_y = text top (matches panel y convention). Rect expands PY2 above/below."""
        f = f or font
        hot, dim = _LEGEND_COLORS[i]
        fg = hot if active else dim
        ts = _render_text(label, f, fg)
        tw, th = ts.get_size()
        w = tw + PX2 * 2
        if active:
            fill = tuple(max(0, c - 200) + 30 for c in hot)
            pygame.draw.rect(surf, fill,
                             pygame.Rect(lx, text_y - PY2, w, th + PY2 * 2),
                             border_radius=4)
        surf.blit(ts, (lx + PX2, text_y))
        return w

    def _pill_row(items, text_y):
        """items: [(label, color_idx, active, font_or_None), ...]
        Distributes pills evenly across the full panel width."""
        widths = []
        for label, i, active, f in items:
            f = f or font
            widths.append(_render_text(label, f, (255, 255, 255)).get_width() + PX2 * 2)
        total_w   = sum(widths)
        available = PANEL_W - MARGIN * 2
        gap       = (available - total_w) / max(len(items) - 1, 1)
        lx = px + MARGIN
        for (label, i, active, f), w in zip(items, widths):
            _pill(lx, text_y, label, i, active, f)
            lx += w + gap

    state_label = "sp:PAUSE" if paused else "sp:PLAY "

    _pill_row([
        ("0:.5x",  1, sim_speed_idx == 0, None),
        ("1:1x",   2, sim_speed_idx == 1, None),
        ("2:5x",   3, sim_speed_idx == 2, None),
        ("3:20x",  4, sim_speed_idx == 3, None),
        ("4:100x", 5, sim_speed_idx == 4, None),
        ("5:MAX",  6, sim_speed_idx == 5, None),
    ], y)
    y += font.get_height() + 6

    _pill_row([
        (state_label,    0, paused,       None),
        ("s:screenshot", 7, snap_active,  font_lg),
        ("r:restart",    8, rst_active,   font_lg),
    ], y)
    y += font_lg.get_height() + 6
    sep()

    N = len(pop['x'])
    if N > 0:
        txt(f"tick {tick:,}  pop {N}  gen {int(pop['generation'].max())}  age {int(pop['age'].max())}  hunts {int(pop['hunts'].max())}", font_sm, (160, 200, 160))
        sep()

    # ── stacked area: phylo sub-lineage populations over time ────────────
    if lineage_history:
        txt("LINEAGES over time", font, (200, 180, 140))
        chart_h = 90
        _draw_stacked_area(surf, lineage_history, (px + 8, y, PANEL_W - 16, chart_h), phylo_state)
        y += chart_h + 4
        lx = px + 10
        for aid, cnt in sorted(lineage_history[-1].items(), key=lambda kv: -kv[1]):
            color = _anc_color(aid, phylo_state)
            pygame.draw.circle(surf, color, (lx + 4, y + 5), 4)
            surf.blit(_render_text(f"{cnt}", font_sm, color), (lx + 11, y))
            lx += 38
            if lx > px + PANEL_W - 38:
                lx = px + 10
                y += 14
        y += 16
        sep()

    # ── trait heatmap ────────────────────────────────────────────────────
    if N > 0 or len(history) > 0 or _traits_surf_cache is not None:
        txt("TRAITS  (median  |  p10–p90 band)", font, (160, 180, 220))
        n_traits = 16
        hmap_h   = n_traits * 9
        _draw_trait_heatmap(surf, pop, (px + 8, y, PANEL_W - 16, hmap_h), font_sm)
        y += hmap_h + 4
        sep()

    # ── PCA scatter ──────────────────────────────────────────────────────
    if pca_proj is not None or _pca_surf_cache is not None:
        txt("STRATEGY SPACE  (W_body PCA)", font, (160, 180, 220))
        _draw_pca_scatter(surf, pop, (px + 8, y, PANEL_W - 16, 130), phylo_state, pca_proj, anc_ids=anc_ids)
        y += 134
        sep()

    if hall_fame:
        txt("HALL OF FAME", font, (255, 210, 80))
        for i, (hunts, gen, age, spd, fov, sz, drn, r, g, b) in enumerate(hall_fame):
            pygame.draw.circle(surf, (int(r), int(g), int(b)), (px + 18, y + 6), 5)
            txt(f"  #{i+1} hunts:{hunts:3d} g{gen} spd{spd:.1f} sz{sz:.1f}", font_sm, (220, 200, 140))
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
            f"  hunts  {int(pop['hunts'][sel_idx])}",
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
        genome = sel_W_body if sel_W_body is not None else []
        for j, (lbl, w_val) in enumerate(zip(
            ["spd", "fov", "ray", "sz", "r", "g", "b", "trn", "brd"],
            genome,
        )):
            h_bar     = int(abs(float(w_val)) * 12)
            color_bar = (100, 200, 100) if w_val > 0 else (200, 80, 80)
            pygame.draw.rect(surf, color_bar,
                             (bar_x + j * bw + 1, bar_y + 20 - h_bar, bw - 2, h_bar))
            surf.blit(_render_text(lbl, font_sm, (120, 120, 140)),
                      (bar_x + j * bw, bar_y + 22))
        y = bar_y + 38

