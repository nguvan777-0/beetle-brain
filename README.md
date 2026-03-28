# beetle-brain

Neuroevolution sim where the organism is its weights, accelerated via CoreML on Apple Silicon (numpy fallback included). We're evolving weights called wights. Each wight's brain is a Recurrent Neural Network (RNN) — hidden state carries across ticks, enabling memory to evolve.

- **Encoded** (genome, evolves): speed, fov, size, color, mouth, pred_ratio, mutation rates, HGT rates, epigenetic carry-over, active neurons, n_rays — 20 floats decoded via sigmoid.
- **Derived** (our rules, does not evolve): `energy_max = 10 × size²` (storage ∝ volume), `drain = 0.010 × size^0.75` (Kleiber's law).
- **Emergent**: predator/prey dimorphism, camouflage arms races, lineage divergence, and spatial memory — nothing encodes these, they appear.

![beetle-brain](https://github.com/nguvan777-0/beetle-brain/releases/download/screenshots/screenshot_f8fa4f5_225919_0001894.png)

## Run with the sun

On Apple Silicon:
```bash
uv run --with coremltools --with pygame python world.py
```

Everywhere else:
```bash
uv run --with numpy --with pygame python world.py
```

```
$ uv run --with numpy python world.py --help
beetle-brain  —  evolution on weights

Usage:
  uv run --with coremltools --with pygame python world.py
  uv run --with coremltools python world.py --new

Libraries:
  numpy          required — or use coremltools which includes it
  coremltools    CoreML acceleration on Apple Silicon (includes numpy)
  pygame         visual display
  plotly         HTML report generation

Runs forever by default. Omit pygame to run headless. Snapshot and reports written on exit.

Options:
  -h, --help           show this help message and exit
  --duration [N]       stop after N seconds  (default: run forever)
  --backend [BACKEND]  backend  (default: gpu)
                         gpu, ane, cpu  — CoreML with that compute unit
                         all            — CoreML with ANE + GPU + CPU together
                         numpy          — no CoreML
  --seed [N]           start fresh with seed N  (ignores snapshot)
  --new                start fresh with a random seed  (ignores snapshot)
  --fork [N]           load snapshot, run forward with a different RNG seed
  --no-report          skip printing the report on exit
```

**Keys:** `SPACE` play/pause · `0–5` speed (0.5×/1×/5×/20×/100×/MAX) · `S` screenshot · `R` restart · `click` inspect wight · `ESC` quit (auto-saves, generates report)

## the wight

Each wight is ~2,294 floats: 20 body weights (W_body), 1,152 for the first brain layer (36 inputs × 32 max hidden), 64 for the second (32 hidden × 2 outputs), 1,024 for the hidden-to-hidden recurrent layer (32 × 32), and 34 bias weights (b1: 32, b2: 2). W_body is decoded via sigmoid into bounded physical traits; the brain weights are unconstrained floats evolved in weight space. Starts with 12 wights (a primordial soup). Everything else emerges.

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
`sim/` — pure simulation logic: tick, sensing, predation, HGT, evolution, vents, phylo, stats  
`game/` — pygame loop, renderer, HUD, snapshot — no sim logic crosses in  
`brain/` — CoreML Elman RNN and fused sensing+brain kernel  
`scripts/` — report generation, bench, dev hooks

## Tuning the world

Everything is in `config.toml` — edit it, restart the sim. Sections: `[world]`, `[metabolism]`, `[hgt]`, `[evolution]`, `[aging]`, `[camouflage]`

## Performance

Measured on Apple Silicon (Mac mini M4), headless

| `--backend` | hardware | compile | start (pop~14) | grown (pop~38) | max-pop (4096) |
|-------------|----------|---------|----------------|----------------|-------------|
| `ane`       | CoreML → ANE         | ~33s  | ~1,250 t/s | ~714 t/s | failed |
| `gpu` ✓     | CoreML → GPU         | ~1.6s | ~1,250 t/s | ~714 t/s | ~17 t/s |
| `cpu`       | CoreML → CPU         | ~1.5s | ~1,250 t/s | ~714 t/s | ~11 t/s |
| `all`       | CoreML → CPU+GPU+ANE | ~17s  | ~1,000 t/s | ~714 t/s | ~16 t/s |
| `numpy`     | numpy (no CoreML)    | ~1.6s | ~1,250 t/s | ~714 t/s | ~17 t/s |

At small populations throughput is bounded by CoreML dispatch latency (~2 dispatches per tick), not brain computation — all backends converge around 700–1,250 t/s. Backend differences only emerge at max-pop where brain computation dominates: `gpu` and `numpy` tie at ~17 t/s, `cpu` falls to ~11 t/s, `all` sits at ~16 t/s.

`gpu` is the default — fast compile, consistent across all population sizes. `ane` compiles the 4096-batch model but exceeds ANE on-chip memory at inference; it is fast at mid-range populations but the max-pop case is unsupported. `all` routes across all hardware simultaneously but scheduling overhead makes it slower than `gpu` alone.


## License

BSD 3-Clause
