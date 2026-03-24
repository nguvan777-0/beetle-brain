# beetle-brain

Neuroevolution sim where the organism is its weights, accelerated via CoreML on Apple Silicon (numpy fallback included). We're evolving weights called wights.

![beetle-brain](https://github.com/nguvan777-0/beetle-brain/releases/download/screenshots/screenshot-v2.png)

## What it is

The organism is called a **wight**. Each wight is ~213 floats: 18 body weights, 180 for the first brain layer, 24 for the second. Size, speed, color, field of view, mouth, predation threshold, breeding strategy, mutation style, aging rate, HGT rates, and how it thinks — all decoded from the same array via sigmoid. Mutate the array, you get a child. Copy it, you get a clone. Free it, it dies.

Starts with 12 wights (a primordial soup). Everything else emerges.

Wights ray-cast through a rasterized world grid — O(N) total regardless of population size. Sensing and predation both use the same grid: sensing ray-marches through it, predation reads a fixed patch around each wight. All brains run in a single batched CoreML call (weights passed as runtime inputs, routed to GPU or ANE by the OS).

## Genome: 18 evolved traits

All traits are decoded from `W_body` (18 floats) via sigmoid into their ranges. `energy_max` and `drain` are derived from `size` (not genes).

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

## How evolution works

- eat food → gain energy (split among all eaters touching the same food)
- touch something smaller (`pred_ratio × your size`) → eat it, gain 30% of its energy
- on a kill, roll against `hgt_eat_rate` → single-point crossover your genome with the prey's
- on proximity contact, roll against `hgt_contact_rate` → same crossover with a neighbor
- hit `breed_at` energy → clone + mutate all weights (W_body, W1, W2), reset to `clone_with`
- hit 0 energy → die
- metabolic drain: `DRAIN_SCALE × size^0.75 + speed² × SPEED_TAX + size² × SIZE_TAX` per tick
- population capped at 4096 (keeps youngest generations on overflow)
- extinction → game over screen with restart

## Package architecture

```
world.py             entry point (3 lines)
sim/                 pure numpy — tick(world, rng) → world, no globals
  config.py          all constants from config.toml
  tick.py            one simulation step
  sensing.py         ray-march through grid
  predation.py       O(N) patch-based kill detection
  hgt.py             horizontal gene transfer (eat + contact crossover)
  evolution.py       clone_batch with per-wight mutation
  phylo.py           ring-buffer ancestry + hue inheritance
  grid/
    constants.py     grid geometry
    painter.py       rasterize world to grid + idx_grid
  population/
    factory.py       spawn initial wight population
    genome.py        decode W_body → trait dict (N_BODY=18)
    ops.py           filter_pop, concat_pop
game/                pure pygame — no sim logic
  main.py            event loop, speed modes, extinction screen
  snapshot.py        save/load world state to .npz
  renderer/
    organisms.py     draw wights and rays
    food.py          draw food
  panel/
    hud.py           stats overlay, trait heatmap, lineage chart, PCA scatter
    sparkline.py     population history graph
brain/
  coreml_brain.py    batched Elman RNN via CoreML (GPU/ANE)
```

## Run

```bash
uv run --with numpy --with pygame --with coremltools python world.py
```

First run compiles the CoreML brain model (~1s), cached to `build/brain.mlpackage`.

**Headless** (no pygame needed):

```bash
uv run --with numpy --with coremltools python run_headless.py 30
```

Runs for 30 seconds at max speed, printing population stats every 500 ticks.

**Keys:** `SPACE` cycle speed (1×/5×/20×/headless) · `L` load · `R` restart · `click` inspect wight · `ESC` quit (auto-saves)

## Tuning the world

Everything is in `config.toml`. Edit it, restart the sim.

```toml
[world]
n_start     = 12      # primordial soup
max_pop     = 4096    # hard ceiling
food_count  = 100     # scarce food → real selection pressure

[hgt]
eat_rate_min     = 0.005   # HGT floor on predation
eat_rate_max     = 0.15    # HGT ceiling on predation
contact_rate_min = 0.0005
contact_rate_max = 0.02

[metabolism]
size_tax    = 0.0003  # quadratic size cost
speed_tax   = 0.004   # quadratic speed cost
age_tax     = 0.0005  # per-tick entropy drain

[aging]
weight_decay_min = 0.000005
weight_decay_max = 0.0001

[evolution]
mutation_rate_min  = 0.01
mutation_rate_max  = 0.50
mutation_scale_min = 0.02
mutation_scale_max = 0.50
epigenetic_min     = 0.0
epigenetic_max     = 1.0

[camouflage]
enabled      = true
detect_bonus = 9.0    # how much further bright prey can be detected
```

## Requirements

- Python 3.11+, numpy, pygame
- coremltools (optional — numpy fallback included)
- Apple Silicon recommended (CoreML routes to ANE/GPU automatically)

## License

BSD 3-Clause
