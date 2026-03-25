# Agent Instructions

## Project
`beetle-brain` is a neuroevolution sim where the organism is its weights. The wight is the organism: a genome-decoded body plus recurrent brain weights, all carried as arrays and filtered by survival pressure rather than a hand-written fitness function.

- `world.py` is the main entry point.
- `config.toml` is the main tuning surface for world, metabolism, HGT, and evolution constants.
- `report.py` generates the offline run report.
- `snapshot.npz` is the current saved world state.

## Package management
Always use `uv` for running Python scripts and installing packages. Never use `pip` directly.

- Run scripts with the dependencies they actually need.
- Install into project: edit `requirements.txt`

## Common commands

Run the sim with pygame, CoreML, and report generation:
```
uv run --with numpy --with pygame --with coremltools --with plotly python world.py
```

Run headless for N seconds:
```
uv run --with numpy --with coremltools --with plotly python world.py 60
```

Inspect the current snapshot:
```
uv run --with numpy python test_parse.py
```

`plotly` is optional for report generation. `coremltools` is optional and the sim falls back to numpy.

## After cloning
Run once to install the pre-commit smoke tests (headless + pygame):
```
bash scripts/install-hooks.sh
```

## Code layout
- `sim/`: pure simulation logic and population ops
- `game/`: pygame loop, renderer, HUD, snapshot handling
- `brain/`: CoreML brain and fused sensing/brain wrappers
- `scripts/`: repo setup and hook scripts

## When unsure
- Use `git log` and related history to recover the project's direction, tone, and prior design intent before making product or documentation calls.
- Check recent commits on the file or subsystem you are touching when the current code leaves room for interpretation.

## Commit messages
Sign commits with model and version, no email:
```
Co-Authored-By: <Model Name> <Version>
```

## Notes
- Prefer constant-time, batched, branch-light code in hot paths. Treat extra per-wight work as a real cost and justify it.
- First CoreML run compiles cached models into `build/brain.mlpackage` and `build/sense_brain.mlpackage`.