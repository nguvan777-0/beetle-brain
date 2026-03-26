# beetle-brain

Neuroevolution sim where the organism is its weights, accelerated via CoreML on Apple Silicon (numpy fallback included). We're evolving weights called wights. Each wight's brain is a Recurrent Neural Network (RNN) — hidden state carries across ticks, enabling memory to evolve.

- **Encoded** (genome, evolves): speed, fov, size, color, mouth, pred_ratio, mutation rates, HGT rates, epigenetic carry-over, active neurons, n_rays — 20 floats decoded via sigmoid.
- **Derived** (our rules, does not evolve): `energy_max = 10 × size²` (storage ∝ volume), `drain = 0.010 × size^0.75` (Kleiber's law).
- **Emergent**: predator/prey dimorphism, camouflage arms races, lineage divergence, and spatial memory — nothing encodes these, they appear.

![beetle-brain](https://github.com/nguvan777-0/beetle-brain/releases/download/screenshots/screenshot_s753463_0007937.png)

## Run with the sun

Requires Python 3.11+ and numpy. `coremltools`, `pygame`, and `plotly` are optional — drop any of them and the sim adapts.

```bash
uv run --with numpy --with coremltools --with pygame --with plotly python world.py
```

To run headless at max speed, pass a duration in seconds:

```bash
uv run --with numpy --with coremltools --with plotly python world.py 60
```

Without `plotly` the exit report is skipped. Without `coremltools` the brain runs on numpy. The first run compiles two CoreML models and caches them to `build/` — takes a couple of seconds, then it's fast.

**Keys:** `SPACE` cycle speed (1×/5×/20×/100×) · `L` load · `R` restart · `click` inspect wight · `ESC` quit (auto-saves, generates report)

## the wight

Each wight is ~2,294 floats: 20 body weights (W_body), 1,152 for the first brain layer (36 inputs × 32 max hidden), 64 for the second (32 hidden × 2 outputs), 1,024 for the hidden-to-hidden recurrent layer (32 × 32), and 34 bias weights (b1: 32, b2: 2). All decoded from the same array via sigmoid. Starts with 12 wights (a primordial soup). Everything else emerges.

Each ray returns 5 channels: food distance, organism distance, and the organism's r/g/b color. Up to 7 rays per wight, giving 35 ray inputs + 1 energy = 36 total inputs. The number of active rays (0–7) is an evolvable gene — a wight with no vision costs nothing on sensing.

Wights ray-cast through a rasterized world grid — O(N) total regardless of population size. Sensing and predation both use the same grid: sensing ray-marches through it, predation reads a fixed patch around each wight. Food spawns near hydrothermal vents with 1/r² density — dense at the vent centre, sparse at the edge. Sensing and brain run fused in a single GPU dispatch (O(1) wall-clock regardless of population size) via a CoreML program that ray-marches and runs the Elman RNN in one kernel.

## Genome: 20 evolving traits

All traits are decoded from `W_body` (20 floats) via sigmoid into their ranges. `energy_max` and `drain` are derived from `size` (not genes).

| Gene | Trait | Role |
|------|-------|------|
| 0 | `speed` | base movement speed |
| 1 | `fov` | field of view width |
| 2 | `ray_len` | sensing range |
| 3 | `size` | body radius — drives `energy_max = scale × size²` and `drain = scale × size^0.75` |
| 4–6 | `r`, `g`, `b` | body color (camouflage axis) |
| 7 | `turn_s` | turning speed |
| 8 | `breed_at` | energy threshold to reproduce (r/K axis) |
| 9 | `clone_with` | energy given to offspring |
| 10 | `mutation_rate` | per-weight mutation probability |
| 11 | `mutation_scale` | mutation step size |
| 12 | `epigenetic` | fraction of parent RNN state inherited |
| 13 | `weight_decay` | vestigial — no longer applied |
| 14 | `mouth` | food reach radius |
| 15 | `pred_ratio` | size multiplier required to predate (1.05–2.0×) |
| 16 | `hgt_eat_rate` | probability of incorporating prey DNA on a kill |
| 17 | `hgt_contact_rate` | probability of gene exchange on proximity contact |
| 18 | `active_neurons` | number of live hidden neurons (0–32) — brain capacity |
| 19 | `n_rays` | number of active rays (0–7) — 0 = no vision |

## The endgame: a wightcat

The goal is to scale the environment, metabolic systems, and cognitive capacity until the sim evolves a **wightcat**: a complex apex predator with spatial reasoning and pursuit.

Life thrives at the boundary between states—rivers, coastlines, and thermal vents. By blasting energy into localized regions of the world, we create the harsh ecological gradients that generate selection pressure for complex behavioral adaptations.

## Layout

`world.py` — entry point, pygame if available, headless otherwise  
`sim/` — pure simulation logic: tick, sensing, predation, HGT, evolution, vents  
`game/` — pygame loop, renderer, HUD — no sim logic crosses in  
`brain/` — CoreML Elman RNN and fused sensing+brain kernel  
`report.py` — offline HTML report

## Tuning the world

Everything is in `config.toml`. Edit it, restart the sim. Sections: `[world]`, `[metabolism]`, `[hgt]`, `[evolution]`, `[aging]`, `[camouflage]`.

## run report

On exit (ESC, quit, or end of headless run) two files are written: `report_{commit}_{seed}_{tick}.html` and `report_{commit}_{seed}_{tick}.txt`.

The HTML opens in any browser — fully offline. Charts: lineage tree, genome heatmap (20 genes × time), phase scatter, drain breakdown, hall of fame.

The `.txt` covers the same run — trait means at exit, sparkline trajectories, drain by component, lineage dominance, strategy spread, key moments:

```
============================================================
  beetle-brain · run report
  9,822 ticks  ·  seed 869659  ·  pop 4096
============================================================

  max gen        44
  max age     3,141
  max eaten     969

────────────────────────────────────────────────────────────
  hall of fame
────────────────────────────────────────────────────────────
  longest survivor
    age 3,141  ·  ate 151  ·  gen 26
    size 8.74  speed 0.87  fov 72.7°  pred 1.23

────────────────────────────────────────────────────────────
  trajectory  (tick 0 → 3,905 → 8,405)
────────────────────────────────────────────────────────────
  pop             12.0 →  224.0 → 4096.0            ▁▁▁▁▁▂▄▇█
  max_gen          2.0 →    8.0 →   44.0       ▁▁▁▁▁▁▂▃▄▄▅▅▆█
  size             6.5 →    6.9 →    8.6    ▁▁▁▁▁▁▁▁▁▁▃▅▅▆▇▇█
  speed            2.1 →    1.9 →    1.1  ▇▇██▇▇▇▆▆▆▅▅▅▆▅▅▄▁
  fov°            66.3 →   59.4 →   68.5  ▇▇▄▁▂▁▂▂▂▃▃▄▁ ▂▁▃▆█
  pred_ratio       1.3 →    1.3 →    1.3  ██▄▂▃▂▂▃▃▄▄▅▂ ▂▁▁▂
  mut_rate         0.2 →    0.2 →    0.3  ▂▂▂ ▁ ▁▂▂▃▄▅▃▂▄▄▅█▇
  n_rays           4.8 →    5.4 →    4.9    ▁▄▅▅▆▇████▇▅▄▄▅▅▁
  active_neur     12.8 →   13.2 →   12.6  ▂▂▄ ▂ ▁▄▅▄▆█▄▂▃▃▆▅▁

  drain / tick (mean at end)
    kleiber      0.0504    ▁▁▁▁▁▁▁▁▁▁▃▅▅▆▇▇█
    speed        0.0072  ▇▇█▇▆▆▆▆▅▅▄▄▅▅▄▄▃▁
    size         0.0225    ▁▁▁▁▁▁▁▁▁▁▃▄▅▆▇▇█
    sensing      0.0021  ██▇▂▂▁▂▂▃▄▄▄▃▁▂  ▁▂
    brain        0.0093  ▂▂▄▁▂ ▁▄▅▄▆█▄▂▃▃▆▅▂

────────────────────────────────────────────────────────────
  lineage dominance  (147 total)
────────────────────────────────────────────────────────────
  6           born tick      0  share    7%  final    91  ▁▁▂▃▄▆▇█▅▄█▅▅
  393         born tick  5,905  share    5%  final   274  ▁▃▇█
  299         born tick  5,905  share    5%  final   268  ▁▁▃▇█
  161         born tick  4,905  share    4%  final   247  ▁▂▃▅█
  1408        born tick  6,905  share    4%  final   247  ▂▇█

────────────────────────────────────────────────────────────
  strategy spread at final snapshot  — spread suggests multiple strategies
────────────────────────────────────────────────────────────
  size        mean  8.65  std 0.19  range 8.07–8.98
  pred_ratio  mean  1.26  std 0.13  range 1.06–1.77
  speed       mean  1.09

  key moments
    tick  6,405  size crossed 80% of range
    tick  8,405  size crossed 90% — ceiling locked
    tick  4,905  fastest size change (6.85→7.35)
```

## License

BSD 3-Clause
