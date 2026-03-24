"""All constants loaded from config.toml."""
import tomllib
from pathlib import Path

_cfg_path = Path(__file__).resolve().parent.parent / "config.toml"
with open(_cfg_path, "rb") as _f:
    _cfg = tomllib.load(_f)

WIDTH, HEIGHT   = _cfg["world"]["width"],  _cfg["world"]["height"]
N_FOOD          = _cfg["world"]["food_count"]
N_START         = _cfg["world"]["n_start"]
MAX_POP         = _cfg["world"]["max_pop"]

SPEED_MIN       = _cfg["traits"]["speed_min"]; SPEED_MAX  = _cfg["traits"]["speed_max"]
FOV_MIN         = _cfg["traits"]["fov_min"];   FOV_MAX    = _cfg["traits"]["fov_max"]
RAY_MIN         = _cfg["traits"]["ray_min"];   RAY_MAX    = _cfg["traits"]["ray_max"]
SIZE_MIN        = _cfg["traits"]["size_min"];  SIZE_MAX   = _cfg["traits"]["size_max"]
DRAIN_MIN       = _cfg["traits"]["drain_min"]; DRAIN_MAX  = _cfg["traits"]["drain_max"]

ENERGY_START      = _cfg["energy"]["start"]
ENERGY_MAX        = _cfg["energy"]["max"]
ENERGY_FOOD       = _cfg["energy"]["food"]
BREED_AT_MIN      = _cfg["energy"]["breed_at_min"]
BREED_AT_MAX      = _cfg["energy"]["breed_at_max"]
CLONE_WITH_MIN    = _cfg["energy"]["clone_with_min"]
CLONE_WITH_MAX    = _cfg["energy"]["clone_with_max"]

N_RAYS          = _cfg["brain"]["n_rays"]
N_HIDDEN        = _cfg["brain"]["n_hidden"]
N_OUTPUTS       = _cfg["brain"]["n_outputs"]
N_INPUTS        = N_RAYS * 2 + 1

MUTATION_RATE_MIN  = _cfg["evolution"]["mutation_rate_min"]
MUTATION_RATE_MAX  = _cfg["evolution"]["mutation_rate_max"]
MUTATION_SCALE_MIN = _cfg["evolution"]["mutation_scale_min"]
MUTATION_SCALE_MAX = _cfg["evolution"]["mutation_scale_max"]
EPIGENETIC_MIN  = _cfg["evolution"]["epigenetic_min"]
EPIGENETIC_MAX  = _cfg["evolution"]["epigenetic_max"]

AGING_ENABLED     = _cfg["aging"]["enabled"]
WEIGHT_DECAY_MIN  = _cfg["aging"]["weight_decay_min"]
WEIGHT_DECAY_MAX  = _cfg["aging"]["weight_decay_max"]

SIZE_TAX        = _cfg["metabolism"]["size_tax"]
SPEED_TAX       = _cfg["metabolism"]["speed_tax"]
AGE_TAX         = _cfg["metabolism"]["age_tax"]

CAMO_ENABLED    = _cfg["camouflage"]["enabled"]
CAMO_BONUS      = _cfg["camouflage"]["detect_bonus"]
