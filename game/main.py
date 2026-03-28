"""Pygame event loop — the only place that knows about the screen."""
import os
import sys
import time
import threading
import queue as _queue
from dataclasses import dataclass
from typing import Any

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
from scripts.report import generate as generate_report

FPS         = 60
FRAME_S     = 1.0 / FPS                  # frame budget in seconds
SPEED_STEPS = [0.5, 1, 5, 20, 100, None]  # 0.5 = 30 t/s; None = hardware MAX
TOTAL_W     = sim.WIDTH + PANEL_W
HIST_MAX    = 300


# ── shared render state ────────────────────────────────────────────────────────

@dataclass
class RenderState:
    pop:           dict        # copied numpy arrays — safe to read on render thread
    food:          Any         # np.ndarray copy
    vents:         list
    tick:          int
    history:       list
    lineage_hist:  list
    hall_fame:     list
    pca_proj:      Any         # np.ndarray | None
    anc_ids:       Any         # np.ndarray | None
    paused:        bool
    sim_speed_idx: int
    tps:           float       # measured ticks/second
    seed:          Any
    phylo_state:   dict
    is_extinct:    bool
    day:           bool        # True = day (sunlight on); False = night


def _copy_pop(pop):
    """Copy all numpy arrays in pop dict for safe cross-thread reads."""
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in pop.items()}


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


# ── simulation thread ──────────────────────────────────────────────────────────

class SimRunner(threading.Thread):
    """Runs the simulation loop on a background thread.

    The render thread calls get_state() each frame to read the latest snapshot.
    Commands (pause, speed, reset, quit) are enqueued via send().
    """

    def __init__(self, world, tick, rng, stats, history, hall_fame):
        super().__init__(daemon=True, name='sim')
        self._world      = world
        self._tick       = tick
        self._rng        = rng
        self._stats      = stats
        self._history    = history
        self._hall_fame  = hall_fame

        self._cmds       = _queue.SimpleQueue()
        self._lock       = threading.Lock()
        self._state: RenderState | None = None

        self._paused         = False
        self._speed_idx      = 1        # default 1×
        self._next_sample    = SAMPLE_EVERY
        self._lineage_hist: list = []
        self._cached_pca     = None
        self._is_extinct     = False
        self._tps            = 0.0
        self._t_start        = 0.0
        self._quit_done      = threading.Event()

    # ── public interface ───────────────────────────────────────────────────────

    def send(self, cmd: tuple):
        self._cmds.put(cmd)

    def get_state(self) -> 'RenderState | None':
        with self._lock:
            return self._state

    def wait_quit(self):
        """Block until sim thread has finished its shutdown sequence."""
        self._quit_done.wait()

    # ── private ───────────────────────────────────────────────────────────────

    def _drain_cmds(self) -> bool:
        """Drain pending commands. Returns False if a quit was received."""
        while True:
            try:
                cmd = self._cmds.get_nowait()
            except _queue.Empty:
                break
            tag = cmd[0]
            if tag == 'quit':
                self._do_quit()
                return False
            elif tag == 'pause':
                self._paused = not self._paused
                if self._state is not None:
                    self._publish()
            elif tag == 'toggle_day':
                self._world['day'] = not self._world.get('day', True)
                if self._state is not None:
                    self._publish()
            elif tag == 'speed':
                self._speed_idx = cmd[1]
            elif tag == 'reset':
                seed = cmd[1] if len(cmd) > 1 else None
                self._world          = new_world(seed=seed)
                self._tick           = 0
                self._history        = []
                self._hall_fame      = []
                self._stats          = StatsCollector()
                self._next_sample    = SAMPLE_EVERY
                self._lineage_hist   = []
                self._cached_pca     = None
                self._is_extinct     = False
                self._t_start        = time.time()
        return True

    def _do_quit(self):
        world      = self._world
        tick       = self._tick
        is_extinct = self._is_extinct
        elapsed    = time.time() - self._t_start
        if not is_extinct:
            save_snapshot(world, tick, self._history, self._hall_fame, self._stats)
        self._stats.finalize(tick, elapsed,
                             pop=world['pop'] if not is_extinct else None,
                             phylo_state=world['phylo'], extinct=is_extinct,
                             seed=world.get('seed'))
        generate_report(self._stats, world=world if not is_extinct else None, tick=tick)
        self._quit_done.set()

    def _on_extinction(self):
        elapsed = time.time() - self._t_start
        self._stats.finalize(self._tick, elapsed, pop=None,
                             phylo_state=self._world['phylo'], extinct=True,
                             seed=self._world.get('seed'))
        generate_report(self._stats, world=None, tick=self._tick)
        self._is_extinct = True

    def _publish(self):
        """Copy world state into a RenderState and hand it to the render thread."""
        world = self._world
        pop   = world['pop']
        tick  = self._tick

        # PCA: recompute every 15 ticks or on pop-size change
        if len(pop['x']) != (len(self._cached_pca) if self._cached_pca is not None else -1) or tick >= self._next_pca_tick:
            self._cached_pca   = _pca_proj(pop['W_body'])
            self._next_pca_tick = tick + 15

        anc_ids = None
        if len(pop['x']) > 0:
            depth   = max(4, int(pop['generation'].max()) // 3)
            anc_ids = phylo.ancestor_at(pop['individual_id'], depth, world['phylo'])

        state = RenderState(
            pop           = _copy_pop(pop),
            food          = world['food'].copy(),
            vents         = list(world['vents']),
            tick          = tick,
            history       = list(self._history),
            lineage_hist  = list(self._lineage_hist),
            hall_fame     = list(self._hall_fame),
            pca_proj      = self._cached_pca,
            anc_ids       = anc_ids,
            paused        = self._paused,
            sim_speed_idx = self._speed_idx,
            tps           = self._tps,
            seed          = world.get('seed'),
            phylo_state   = world['phylo'],
            is_extinct    = self._is_extinct,
            day           = world.get('day', True),
        )
        with self._lock:
            self._state = state

    def _update_derived(self, ticks_done: int):
        """Update history, stats, and hall-of-fame after a tick batch."""
        world = self._world
        pop   = world['pop']
        tick  = self._tick

        if tick >= self._next_sample and len(pop['x']) > 0:
            self._stats.record(tick, pop, world['phylo'])
            self._next_sample += SAMPLE_EVERY

        if tick % 30 < ticks_done and len(pop['x']) > 0:
            self._history.append((
                float(tick), float(len(pop['x'])),
                float(pop['generation'].max()),
                float(pop['speed'].mean()), float(pop['fov'].mean()),
                float(pop['size'].mean()), float(pop['mutation_rate'].mean()),
            ))
            depth = max(4, int(pop['generation'].max()) // 3)
            anc   = phylo.ancestor_at(pop['individual_id'], depth, world['phylo'])
            u, c  = np.unique(anc, return_counts=True)
            self._lineage_hist.append(dict(zip(u.tolist(), c.tolist())))
            if len(self._lineage_hist) > HIST_MAX: self._lineage_hist.pop(0)
            if len(self._history)      > HIST_MAX: self._history.pop(0)

        if len(pop['x']) > 0:
            for i in np.where(pop['hunts'] > 0)[0]:
                self._hall_fame.append((
                    int(pop['hunts'][i]), int(pop['generation'][i]),
                    int(pop['age'][i]),
                    float(pop['speed'][i]), float(pop['fov'][i]),
                    float(pop['size'][i]), float(DRAIN_SCALE * pop['size'][i]**0.75),
                    int(pop['r'][i]), int(pop['g'][i]), int(pop['b'][i]),
                ))
            self._hall_fame.sort(key=lambda x: -x[0])
            self._hall_fame = self._hall_fame[:5]

    # ── thread entry ──────────────────────────────────────────────────────────

    def run(self):
        init_ane()   # must be called on the thread that will use CoreML
        self._t_start = time.time()

        tps_window_t = time.time()
        tps_window_n = 0
        TPS_WINDOW   = 0.5   # seconds

        while True:
            if not self._drain_cmds():
                break

            if self._paused or self._is_extinct:
                time.sleep(0.001)
                continue

            pop = self._world['pop']
            if len(pop['x']) == 0:
                self._on_extinction()
                self._publish()
                continue

            speed = SPEED_STEPS[self._speed_idx]
            t0    = time.time()

            if speed is None:
                # MAX: run as many ticks as fit in one frame budget
                ticks_done = 0
                while time.time() - t0 < FRAME_S:
                    self._world = sim_tick(self._world, self._rng)
                    ticks_done += 1
                    if len(self._world['pop']['x']) == 0:
                        break
            elif speed == 0.5:
                self._world = sim_tick(self._world, self._rng)
                ticks_done  = 1
            else:
                ticks_done = 0
                for _ in range(speed):
                    self._world = sim_tick(self._world, self._rng)
                    ticks_done += 1
                    if len(self._world['pop']['x']) == 0:
                        break

            self._tick += ticks_done
            self._update_derived(ticks_done)

            # TPS measurement
            tps_window_n += ticks_done
            now = time.time()
            if now - tps_window_t >= TPS_WINDOW:
                self._tps        = tps_window_n / (now - tps_window_t)
                tps_window_n     = 0
                tps_window_t     = now

            self._publish()

            # Throttle / yield
            elapsed = time.time() - t0
            if speed is None:
                time.sleep(0)            # yield to GIL so render thread runs
            elif speed == 0.5:
                remaining = (1.0 / 30) - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            else:
                remaining = FRAME_S - elapsed
                if remaining > 0:
                    time.sleep(remaining)


# ── render helpers ─────────────────────────────────────────────────────────────

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


# ── main ──────────────────────────────────────────────────────────────────────

def main(new=False, seed=None, fork=None, compute_units='CPU_AND_GPU'):
    if compute_units != 'numpy':
        os.environ['BEETLE_COMPUTE_UNITS'] = compute_units

    pygame.init()
    surf    = pygame.display.set_mode((TOTAL_W, sim.HEIGHT))
    pygame.display.set_caption("beetle-brain  |  wight")
    clock   = pygame.time.Clock()
    font    = pygame.font.SysFont("monospace", 12)
    font_sm = pygame.font.SysFont("monospace", 10)
    font_lg = pygame.font.SysFont("monospace", 14)

    rng = np.random.default_rng(fork)
    world, tick, history, hall_fame, _saved_stats = load_snapshot(rng)
    if new or seed is not None or world is None or len(world['pop']['x']) == 0:
        world     = new_world(seed=seed)
        tick      = 0
        history   = []
        hall_fame = []

    stats = _saved_stats if _saved_stats is not None else StatsCollector()

    runner = SimRunner(world, tick, rng, stats, history, hall_fame)
    runner.start()

    sel_idx     = None
    last_snap_t = -999.0
    last_rst_t  = -999.0

    while True:
        snap = runner.get_state()

        # ── events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                runner.send(('quit',))
                runner.wait_quit()
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and snap is not None:
                mx, my = pygame.mouse.get_pos()
                pop = snap.pop
                if mx < sim.WIDTH and len(pop['x']) > 0:
                    dists   = np.hypot(pop['x'] - mx, pop['y'] - my)
                    idx     = int(dists.argmin())
                    sel_idx = idx if dists[idx] < 40 else None

            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                runner.send(('toggle_day',))

            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                runner.send(('pause',))

            if event.type == pygame.KEYDOWN and event.key in (
                pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5,
            ):
                runner.send(('speed', event.key - pygame.K_0))

            if event.type == pygame.KEYDOWN and event.key == pygame.K_KP0:
                runner.send(('speed', 0))

            if event.type == pygame.KEYDOWN and event.key in (
                pygame.K_KP1, pygame.K_KP2, pygame.K_KP3, pygame.K_KP4, pygame.K_KP5,
            ):
                runner.send(('speed', event.key - pygame.K_KP1 + 1))

            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                last_rst_t = time.time()
                sel_idx    = None
                runner.send(('reset', None))

            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                import subprocess
                try:
                    commit = subprocess.check_output(
                        ['git', 'rev-parse', '--short', 'HEAD'],
                        stderr=subprocess.DEVNULL).decode().strip()
                except Exception:
                    commit = 'unknown'
                os.makedirs("screenshots", exist_ok=True)
                seed_val = snap.seed if snap is not None else 0
                tick_val = snap.tick if snap is not None else 0
                path = f"screenshots/screenshot_{commit}_{seed_val}_{tick_val:07d}.png"
                pygame.image.save(surf, path)
                print(f"screenshot → {path}")
                last_snap_t = time.time()

        # ── wait for first state ───────────────────────────────────────────────
        if snap is None:
            clock.tick(FPS)
            continue

        pop = snap.pop
        if sel_idx is not None and sel_idx >= len(pop['x']):
            sel_idx = None

        # ── draw ──────────────────────────────────────────────────────────────
        surf.fill((10, 14, 20))

        if getattr(sim, 'COASTLINE_X', None) is not None:
            land_rect  = pygame.Rect(sim.COASTLINE_X, 0, sim.WIDTH - sim.COASTLINE_X, sim.HEIGHT)
            land_color = (30, 30, 20) if snap.day else (12, 10, 22)
            shore_color = (40, 45, 30) if snap.day else (28, 24, 45)
            pygame.draw.rect(surf, land_color, land_rect)
            pygame.draw.line(surf, shore_color,
                             (sim.COASTLINE_X, 0), (sim.COASTLINE_X, sim.HEIGHT), 2)

        draw_food(surf, snap.food, snap.vents)
        _draw_organisms(surf, pop, snap.phylo_state, sel_idx, anc_ids=snap.anc_ids)

        sel_wb = (pop['W_body'][sel_idx].copy()
                  if sel_idx is not None and sel_idx < len(pop['x']) else None)

        now = time.time()
        draw_panel(surf, font, font_sm, font_lg, snap.tick, pop, sel_idx,
                   snap.history, snap.lineage_hist, snap.hall_fame,
                   vents=snap.vents, phylo_state=snap.phylo_state, seed=snap.seed,
                   pca_proj=snap.pca_proj, sel_W_body=sel_wb, anc_ids=snap.anc_ids,
                   paused=snap.paused, sim_speed_idx=snap.sim_speed_idx,
                   snap_active=(now - last_snap_t) < 0.2,
                   rst_active=(now - last_rst_t) < 0.2,
                   fps=clock.get_fps(), day=snap.day)

        if snap.is_extinct:
            _draw_extinction_overlay(surf, font, font_lg, snap.tick)

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == '__main__':
    main()
