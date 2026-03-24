# beetle-brain

Neuroevolution sim where the organism is its weights, accelerated via CoreML on Apple Silicon (numpy fallback included). We're evolving weights called wights.

![beetle-brain](https://github.com/nguvan777-0/beetle-brain/releases/download/screenshots/screenshot-v1.png)

## What it is

The organism is called a **wight**. Each wight is ~213 floats: 19 body weights, 180 for the first brain layer, 24 for the second. Size, speed, color, field of view, energy drain, mouth, max energy storage, predation threshold, speed scaling, breeding strategy, mutation style, aging rate, and how it thinks — all decoded from the same array via sigmoid. Mutate the array, you get a child. Copy it, you get a clone. Free it, it dies.

Starts with 12 wights (a primordial soup). Everything else emerges.

Wights ray-cast through a rasterized world grid — O(N) total regardless of population size. Sensing and predation both use the same grid: sensing ray-marches through it, predation reads a fixed 21×21 patch around each wight. All brains run in a single batched CoreML call (weights passed as runtime inputs, routed to GPU or ANE by the OS).

## Genome: 19 evolved traits

All traits are decoded from `W_body` (19 floats) via sigmoid into their ranges:

| Gene | Trait | Role |
|------|-------|------|
| 0 | `speed` | base movement speed |
| 1 | `fov` | field of view width |
| 2 | `ray_len` | sensing range |
| 3 | `size` | body radius (affects eating, predation, metabolic cost) |
| 4 | `drain` | base metabolic drain per tick |
| 5–7 | `r`, `g`, `b` | body color (camouflage axis) |
| 8 | `turn_s` | turning speed |
| 9 | `breed_at` | energy threshold to reproduce (r/K axis) |
| 10 | `clone_with` | energy retained after cloning |
| 11 | `mutation_rate` | per-weight mutation probability |
| 12 | `mutation_scale` | mutation step size |
| 13 | `epigenetic` | fraction of parent RNN state inherited |
| 14 | `weight_decay` | per-tick weight erosion (aging rate) |
| 15 | `mouth` | food reach radius |
| 16 | `energy_max` | max energy storage capacity |
| 17 | `pred_ratio` | size multiplier required to predate (1.05–2.0×) |
| 18 | `speed_scale` | modulates brain speed output (0.2–1.0×) |

## How evolution works

- eat food → gain energy (split among all eaters touching the same food)
- touch something smaller (pred_ratio × your size) → eat it, gain 70% of its energy (split among all killers)
- hit `breed_at` energy → clone + mutate, reset to `clone_with`
- hit 0 energy → die
- metabolic drain: `drain + speed² × SPEED_TAX + size² × SIZE_TAX` per tick, plus entropy `age_tax` per tick
- aging: `weight_decay` erodes W_body/W1/W2 each tick, slowly shifting traits
- population capped at 4096 (keeps youngest generations on overflow)
- extinction → game over screen with restart

## Package architecture

```
world.py             entry point (3 lines)
sim/                 pure numpy — no pygame
  config.py          all constants from config.toml
  tick.py            one simulation step
  sensing.py         ray-march through grid
  predation.py       O(N) patch-based kill detection
  evolution.py       clone_batch with per-wight mutation
  grid/
    constants.py     grid geometry
    painter.py       rasterize world to (2,GH,GW) grid + idx_grid
  population/
    factory.py       spawn initial wight population
    genome.py        decode W_body → trait dict (N_BODY=19)
    ops.py           filter_pop, concat_pop
game/                pure pygame — no sim logic
  main.py            event loop, speed modes, extinction screen
  snapshot.py        save/load world state to .npz
  renderer/
    organisms.py     draw wights and rays
    food.py          draw food
  panel/
    hud.py           stats overlay
    sparkline.py     population history graph
brain/
  coreml_brain.py    batched Elman RNN via CoreML (GPU/ANE)
```

## Run

```bash
uv run --with numpy --with pygame --with coremltools python world.py
```

First run compiles the CoreML brain model (~1s), cached to `build/lookup.mlpackage`.

**Keys:** `SPACE` cycle speed (1×/5×/20×/headless) · `S` save · `L` load · `click` inspect wight · `ESC` quit

## Tuning the world

Everything is in `config.toml`. Edit it, restart the sim.

```toml
[world]
n_start     = 12      # primordial soup
max_pop     = 4096    # hard ceiling
food_count  = 12      # scarce food → real selection pressure

[metabolism]
size_tax    = 0.0003  # quadratic size cost
speed_tax   = 0.004   # quadratic speed cost
age_tax     = 0.001   # per-tick entropy drain

[aging]
enabled          = true
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

- Python 3.11+
- numpy
- pygame
- coremltools (optional — numpy fallback included)

## License

BSD 3-Clause
