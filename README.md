# beetle-brain

Neuroevolution sim where the organism is its weights, accelerated via CoreML on Apple Silicon (numpy fallback included). We're evolving weights called wights 

![beetle-brain](https://github.com/nguvan777-0/beetle-brain/releases/download/screenshots/screenshot-v1.png)

## What it is

The organism is called a **wight**. 213 floats per wight. 9 body weights, 180 for the first brain layer, 24 for the second. That's it. Size, speed, color, field of view, energy drain, how it turns, how it thinks — all decoded from the same array. Mutate the array, you get a child. Copy it, you get a clone. Free it, it dies.

Starts with a single wight. Everything else emerges.

Organisms ray-cast through a rasterized world grid — O(1) per organism regardless of population size. All 300 brains run in a single batched CoreML call (weights passed as runtime inputs, routed to GPU or ANE by the OS).


## How evolution works

Inspired by the eyes of the Cambrian.

Every wight has a field of view encoded in its genome. It evolves alongside body size, speed, and brain weights — all in the same mutation pass.

- eat food → gain energy
- touch something smaller (>1.25× your size) → eat it, gain 70% of its energy
- hit 160 energy → clone + mutate, reset to 80
- hit 0 energy → die
- food scarcity is the only population limit — Malthusian pressure
- size, speed, fov, ray length, drain, color, and the entire brain — all evolve together

## Tuning the world

Everything is in `config.toml`. Edit it, restart the sim.

```toml
[world]
max_pop     = 300     # more pressure = faster evolution
food_count  = 200     # scarce food selects harder

[aging]
enabled      = true
weight_decay = 0.00002  # higher = faster aging, more turnover

[evolution]
mutation_rate  = 0.12   # how often a weight mutates
mutation_scale = 0.15   # how much
epigenetic     = 0.25   # fraction of parent memory inherited at birth

[camouflage]
enabled      = true
detect_bonus = 9.0    # how much further bright prey can be hunted from
```

## Run

```bash
uv run --with numpy --with pygame --with coremltools python world.py
```

First run compiles the CoreML brain (~5s), cached to `build/brain.mlpackage`.

**Keys:** `SPACE` cycle speed (1×/5×/20×/headless) · `S` save · `L` load · `click` inspect · `ESC` quit

## Architecture

```
world.py             UI (pygame), save/load
sim.py               simulation engine
  _paint_grid()      rasterize world into (2, 450, 450) uint8 each tick
  _sense()           ray-march all organisms through grid, O(1)/organism
  tick()             one step, pure numpy
brain/
  coreml_brain.py    batched CoreML forward pass, weights as runtime inputs
```

## Requirements

- Python 3.11+
- numpy
- pygame (UI only)
- coremltools (optional, numpy fallback included)

## License

BSD 3-Clause
