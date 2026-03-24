"""
world.py — beetle-brain: evolving seeing organisms
===================================================
Run:
    uv run --with numpy --with pygame --with coremltools world.py

Simulation engine: sim.py (fully vectorized numpy + CoreML ANE)
The organism IS its weights. Nothing else.
  W_body → speed, fov, ray_length, size, energy_drain, color
  W1/W2  → brain (sense → act)
"""

import numpy as np
import pygame
import sys
import os

import sim
from sim import new_world, tick as sim_tick, init_ane

# ── WORLD CONSTANTS ────────────────────────────────────────────────────────────
PANEL_W         = 320
WIDTH, HEIGHT   = sim.WIDTH, sim.HEIGHT
TOTAL_W         = WIDTH + PANEL_W
FPS             = 60
SPEED_STEPS     = [1, 5, 20, 0]   # 0 = headless (no render)
N_FOOD          = sim.N_FOOD
N_RAYS          = sim.N_RAYS

SNAPSHOT_PATH   = "snapshot.npz"


# ── DRAWING ────────────────────────────────────────────────────────────────────
def draw_organism(surf, x, y, angle, size, r, g, b):
    xi, yi = int(x), int(y)
    pygame.draw.circle(surf, (r, g, b), (xi, yi), max(1, int(size)))
    ex = xi + int(np.cos(angle) * (size + 4))
    ey = yi + int(np.sin(angle) * (size + 4))
    pygame.draw.line(surf, (255, 255, 255), (xi, yi), (ex, ey), 1)

def draw_rays(surf, x, y, fov, angle, ray_len):
    offsets = np.linspace(-1, 1, N_RAYS) * fov * 0.5
    for off in offsets:
        a = angle + off
        ex = x + np.cos(a) * ray_len
        ey = y + np.sin(a) * ray_len
        pygame.draw.line(surf, (50, 50, 80), (int(x), int(y)), (int(ex), int(ey)), 1)


# ── PANEL ──────────────────────────────────────────────────────────────────────
def draw_sparkline(surf, data, rect, color, mn=None, mx=None):
    if len(data) < 2:
        return
    x0, y0, w, h = rect
    mn = mn if mn is not None else min(data)
    mx = mx if mx is not None else max(data)
    rng = mx - mn if mx != mn else 1
    pts = []
    for i, v in enumerate(data):
        px = x0 + int(i / (len(data)-1) * w)
        py = y0 + h - int((v - mn) / rng * h)
        pts.append((px, py))
    if len(pts) > 1:
        pygame.draw.lines(surf, color, False, pts, 1)
    font_tiny = pygame.font.SysFont("monospace", 10)
    lbl = font_tiny.render(f"{data[-1]:.2f}", True, color)
    surf.blit(lbl, (x0 + w + 3, y0 + h//2 - 5))

def draw_panel(surf, font, font_sm, font_lg, tick, pop, sel_idx,
               history, hall_fame, sim_speed=1):
    px = WIDTH
    pygame.draw.rect(surf, (16, 16, 28), (px, 0, PANEL_W, HEIGHT))
    pygame.draw.line(surf, (50, 50, 80), (px, 0), (px, HEIGHT), 1)

    y = 10
    def txt(s, f=None, color=(180,180,200)):
        nonlocal y
        surf.blit((f or font).render(s, True, color), (px+10, y))
        y += (f or font).get_height() + 2

    def sep():
        nonlocal y
        pygame.draw.line(surf, (40,40,60), (px+8, y+2), (px+PANEL_W-8, y+2), 1)
        y += 8

    speed_label = "HEADLESS" if sim_speed == 0 else f"{sim_speed}x"
    speed_color = (255,180,50) if sim_speed != 1 else (120,120,150)
    txt("BEETLE-BRAIN", font_lg, (220,220,255))
    txt(f"tick {tick:,}   [{speed_label}]  SPACE=cycle", font_sm, speed_color)
    sep()

    N = len(pop['x'])
    if N > 0:
        txt("POPULATION", font, (160,200,160))
        txt(f"  count   {N:4d}", font_sm)
        txt(f"  max gen {int(pop['generation'].max()):4d}", font_sm)
        txt(f"  max age {int(pop['age'].max()):6d}", font_sm)
        txt(f"  max ate {int(pop['eaten'].max()):4d}", font_sm)
        sep()

        # ── trait trends ────────────────────────────────────────────────────
        txt("TRAIT TRENDS  (pop avg)", font, (160,180,220))
        if len(history) > 1:
            labels = [
                ("speed",  3, (100,200,255), 0, sim.SPEED_MAX),
                ("fov °",  4, (200,160,255), 0, 180),
                ("size",   5, (255,180,100), 0, sim.SIZE_MAX),
                ("drain",  6, (255,100,100), 0, sim.DRAIN_MAX),
            ]
            for label, col_idx, color, lo, hi in labels:
                data = [h[col_idx] for h in history]
                if label == "fov °":
                    data = [np.degrees(v) for v in data]
                txt(f"  {label}", font_sm, color)
                draw_sparkline(surf, data, (px+16, y, PANEL_W-60, 22), color, lo, hi)
                y += 28
        sep()

    # ── hall of fame ────────────────────────────────────────────────────────
    if hall_fame:
        txt("HALL OF FAME", font, (255,210,80))
        for i, (eaten, gen, age, spd, fov, sz, drn, r, g, b) in enumerate(hall_fame):
            pygame.draw.circle(surf, (r, g, b), (px+18, y+6), 5)
            line = f"  #{i+1} ate:{eaten:3d} g{gen} spd{spd:.1f} sz{sz:.1f}"
            txt(line, font_sm, (220,200,140))
        sep()

    # ── selected organism ────────────────────────────────────────────────────
    if sel_idx is not None and sel_idx < N:
        txt("SELECTED", font, (255,255,100))
        sr, sg, sb = int(pop['r'][sel_idx]), int(pop['g'][sel_idx]), int(pop['b'][sel_idx])
        pygame.draw.circle(surf, (sr, sg, sb), (px+18, y+6), max(1, int(pop['size'][sel_idx])))
        y += 4
        rows = [
            f"  gen    {int(pop['generation'][sel_idx])}",
            f"  age    {int(pop['age'][sel_idx]):,}",
            f"  eaten  {int(pop['eaten'][sel_idx])}",
            f"  energy {pop['energy'][sel_idx]:.0f}",
            f"  speed  {pop['speed'][sel_idx]:.2f}",
            f"  fov    {np.degrees(pop['fov'][sel_idx]):.0f}°",
            f"  ray    {pop['ray_len'][sel_idx]:.0f}",
            f"  size   {pop['size'][sel_idx]:.1f}",
            f"  drain  {pop['drain'][sel_idx]:.3f}",
        ]
        for r_str in rows:
            txt(r_str, font_sm, (200,200,120))
        sep()

        # W_body bar chart
        txt("GENOME  W_body", font_sm, (140,160,180))
        labels_b = ["spd","fov","ray","sz","drn","r","g","b","trn"]
        bar_x = px + 10
        bar_y = y
        bw = (PANEL_W - 20) // sim.N_BODY
        for j, (lbl, w_val) in enumerate(zip(labels_b, pop['W_body'][sel_idx])):
            h_bar = int(abs(float(w_val)) * 12)
            color_bar = (100,200,100) if w_val > 0 else (200,80,80)
            pygame.draw.rect(surf, color_bar,
                             (bar_x + j*bw + 1, bar_y + 20 - h_bar, bw-2, h_bar))
            surf.blit(font_sm.render(lbl, True, (120,120,140)),
                      (bar_x + j*bw, bar_y + 22))
        y = bar_y + 38

    # ── help ────────────────────────────────────────────────────────────────
    y = HEIGHT - 30
    txt("S save  L load  click inspect", font_sm, (80,80,100))


# ── SAVE / LOAD ────────────────────────────────────────────────────────────────
def save_snapshot(pop, food, tick, history, hall_fame):
    hist_arr = np.array(history, dtype=np.float32) if history else np.empty((0,7), dtype=np.float32)
    np.savez_compressed(SNAPSHOT_PATH,
        x=pop['x'], y=pop['y'], angle=pop['angle'], energy=pop['energy'],
        W_body=pop['W_body'], W1=pop['W1'], W2=pop['W2'],
        h_state=pop['h_state'],
        generation=pop['generation'], age=pop['age'], eaten=pop['eaten'],
        food=food, tick=np.array([tick], dtype=np.int32),
        hist=hist_arr)
    print(f"[saved] {len(pop['x'])} organisms → {SNAPSHOT_PATH}  (tick {tick})")

def load_snapshot(rng):
    if not os.path.exists(SNAPSHOT_PATH):
        return None, None, 0, [], []
    d = np.load(SNAPSHOT_PATH, allow_pickle=True)
    # rebuild decoded traits from W_body
    W_body = d['W_body']
    speed, fov, ray, size, drain, turn_s, r, g, b = sim._decode(W_body)
    pop = {
        'x':         d['x'].astype(np.float32),
        'y':         d['y'].astype(np.float32),
        'angle':     d['angle'].astype(np.float32),
        'energy':    d['energy'].astype(np.float32),
        'W_body':    W_body.astype(np.float32),
        'W1':        d['W1'].astype(np.float32),
        'W2':        d['W2'].astype(np.float32),
        'speed':     speed.astype(np.float32),
        'fov':       fov.astype(np.float32),
        'ray_len':   ray.astype(np.float32),
        'size':      size.astype(np.float32),
        'drain':     drain.astype(np.float32),
        'turn_s':    turn_s.astype(np.float32),
        'r': r, 'g': g, 'b': b,
        'generation': d['generation'].astype(np.int32),
        'age':        d['age'].astype(np.int32),
        'eaten':      d['eaten'].astype(np.int32),
        'h_state':    d['h_state'].astype(np.float32) if 'h_state' in d else np.zeros((len(d['x']), sim.N_HIDDEN), dtype=np.float32),
    }
    food    = d['food']
    tick    = int(d['tick'][0])
    history = [tuple(row) for row in d['hist']] if d['hist'].ndim == 2 and len(d['hist']) else []
    print(f"[loaded] {len(pop['x'])} organisms ← {SNAPSHOT_PATH}  (tick {tick})")
    return pop, food, tick, history, []


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    # init ANE / CoreML brain
    ane_ok = init_ane()
    if ane_ok:
        print("[ANE] CoreML brain ready")
    else:
        print("[ANE] Using numpy fallback")

    pygame.init()
    surf    = pygame.display.set_mode((TOTAL_W, HEIGHT))
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
    HIST_MAX      = 300
    sim_speed_idx = 0

    while True:
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
                if mx < WIDTH and len(pop['x']) > 0:
                    dists = np.hypot(pop['x'] - mx, pop['y'] - my)
                    idx = int(dists.argmin())
                    sel_idx = idx if dists[idx] < 40 else None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                sim_speed_idx = (sim_speed_idx + 1) % len(SPEED_STEPS)

        steps_this_frame = SPEED_STEPS[sim_speed_idx] or 80

        for _ in range(steps_this_frame):
            pop, food = sim_tick(pop, food, rng)
            tick += 1

        # clamp selected index if population shrank
        if sel_idx is not None and sel_idx >= len(pop['x']):
            sel_idx = None

        # ── history sample every 30 ticks ───────────────────────────────────
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
            # update hall of fame
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

        for fx, fy in food:
            pygame.draw.circle(surf, (50, 190, 70), (int(fx), int(fy)), 3)

        if sel_idx is not None and sel_idx < len(pop['x']):
            draw_rays(surf, pop['x'][sel_idx], pop['y'][sel_idx],
                      pop['fov'][sel_idx], pop['angle'][sel_idx], pop['ray_len'][sel_idx])

        N = len(pop['x'])
        for i in range(N):
            draw_organism(surf, pop['x'][i], pop['y'][i], pop['angle'][i],
                          pop['size'][i], int(pop['r'][i]), int(pop['g'][i]), int(pop['b'][i]))

        if sel_idx is not None and sel_idx < N:
            pygame.draw.circle(surf, (255,255,0),
                               (int(pop['x'][sel_idx]), int(pop['y'][sel_idx])),
                               int(pop['size'][sel_idx])+3, 1)

        draw_panel(surf, font, font_sm, font_lg, tick, pop, sel_idx,
                   history, hall_fame, SPEED_STEPS[sim_speed_idx])

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
