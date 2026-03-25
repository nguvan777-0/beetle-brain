# beetle-brain

Neuroevolution sim where the organism is its weights, accelerated via CoreML on Apple Silicon (numpy fallback included). We're evolving weights called wights. Each wight's brain is a Recurrent Neural Network (RNN) — hidden state carries across ticks, enabling memory to evolve.

- **Encoded** (genome, evolves): speed, fov, size, color, mouth, pred_ratio, mutation rates, HGT rates, epigenetic carry-over, active neurons, n_rays — 20 floats decoded via sigmoid.
- **Derived** (our rules, does not evolve): `energy_max = 10 × size²` (storage ∝ volume), `drain = 0.010 × size^0.75` (Kleiber's law).
- **Emergent**: predator/prey dimorphism, camouflage arms races, lineage divergence, and spatial memory — nothing encodes these, they appear.

![beetle-brain](https://github.com/nguvan777-0/beetle-brain/releases/download/screenshots/screenshot_s753463_0007937.png)

## Run with the sun

Requires Python 3.11+ and numpy. `coremltools` and `pygame` are optional — drop either and the sim adapts.

```bash
uv run --with numpy --with coremltools --with pygame --with plotly python world.py
```

To run headless (default 30s; pass a duration in seconds to override):

```bash
uv run --with numpy --with coremltools --with plotly python world.py 60
```

Without `coremltools` the brain runs on numpy. Without `plotly` the HTML report is skipped. The first run compiles CoreML models and caches them to `build/`.

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

Generates `report.html` on exit (ESC, quit, or extinction). Open in any browser — fully offline.

- **lineage tree** — forks over time, node size = dominance, color = phylo hue
- **genome heatmap** — all 20 genes × time, normalized within each gene's range
- **phase scatter** — size vs pred_ratio at final snapshot, colored by lineage
- **drain breakdown** — Kleiber + speed² + size² + sensing cost, stacked
- **hall of fame** — longest-lived, most ate, highest generation

## License

BSD 3-Clause
