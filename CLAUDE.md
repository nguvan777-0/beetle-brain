# Agent Instructions

## Package management
Always use `uv` for running Python scripts and installing packages. Never use `pip` directly.

## Common commands
Run the sim:
```
uv run --with numpy --with coremltools python world.py
```

## Commit messages
Sign commits with model and version, no email:
```
Co-Authored-By: <Model Name> <Version>
```
