"""All constants loaded from config.toml."""
import tomllib
from pathlib import Path

_cfg_path = Path(__file__).resolve().parent.parent / "config.toml"
with open(_cfg_path, "rb") as _f:
    _cfg = tomllib.load(_f)

WIDTH, HEIGHT   = _cfg["world"]["width"],  _cfg["world"]["height"]
COASTLINE_X     = _cfg["world"].get("coastline_x", WIDTH // 2)
N_FOOD          = _cfg["world"]["food_count"]
N_START         = _cfg["world"]["n_start"]
MAX_POP         = _cfg["world"]["max_pop"]
WORLD_SEED      = _cfg["world"]["seed"]
VENT_COUNT_MIN  = _cfg["world"]["vent_count_min"]
VENT_COUNT_MAX  = _cfg["world"]["vent_count_max"]
VENT_RADIUS     = _cfg["world"]["vent_radius"]

SPEED_MIN       = _cfg["traits"]["speed_min"]; SPEED_MAX  = _cfg["traits"]["speed_max"]
FOV_MIN         = _cfg["traits"]["fov_min"];   FOV_MAX    = _cfg["traits"]["fov_max"]
RAY_MIN         = _cfg["traits"]["ray_min"];   RAY_MAX    = _cfg["traits"]["ray_max"]
SIZE_MIN        = _cfg["traits"]["size_min"];  SIZE_MAX   = _cfg["traits"]["size_max"]

ENERGY_START      = _cfg["energy"]["start"]
ENERGY_MAX        = _cfg["energy"]["max"]
ENERGY_FOOD       = _cfg["energy"]["food"]
ENERGY_SUNLIGHT   = _cfg["energy"].get("sunlight", 0.05)
MOUTH_MIN         = _cfg["energy"]["mouth_min"]
MOUTH_MAX         = _cfg["energy"]["mouth_max"]
ENERGY_MAX_MIN    = _cfg["energy"]["energy_max_min"]
ENERGY_MAX_MAX    = _cfg["energy"]["energy_max_max"]
PRED_RATIO_MIN    = _cfg["energy"]["pred_ratio_min"]
PRED_RATIO_MAX    = _cfg["energy"]["pred_ratio_max"]
SPEED_SCALE_MIN   = _cfg["energy"]["speed_scale_min"]
SPEED_SCALE_MAX   = _cfg["energy"]["speed_scale_max"]
BREED_AT_MIN      = _cfg["energy"]["breed_at_min"]
BREED_AT_MAX      = _cfg["energy"]["breed_at_max"]
CLONE_WITH_MIN    = _cfg["energy"]["clone_with_min"]
CLONE_WITH_MAX    = _cfg["energy"]["clone_with_max"]

N_RAYS          = _cfg["brain"]["n_rays"]
N_HIDDEN        = _cfg["brain"]["n_hidden"]
N_OUTPUTS       = _cfg["brain"]["n_outputs"]
N_INPUTS        = N_RAYS * 5 + 1

BRAIN_TAX       = _cfg["metabolism"].get("brain_tax", 0.0002)

MUTATION_RATE_MIN  = _cfg["evolution"]["mutation_rate_min"]
MUTATION_RATE_MAX  = _cfg["evolution"]["mutation_rate_max"]
MUTATION_SCALE_MIN = _cfg["evolution"]["mutation_scale_min"]
MUTATION_SCALE_MAX = _cfg["evolution"]["mutation_scale_max"]
EPIGENETIC_MIN  = _cfg["evolution"]["epigenetic_min"]
EPIGENETIC_MAX  = _cfg["evolution"]["epigenetic_max"]

AGING_ENABLED     = _cfg["aging"]["enabled"]
WEIGHT_DECAY_MIN  = _cfg["aging"]["weight_decay_min"]
WEIGHT_DECAY_MAX  = _cfg["aging"]["weight_decay_max"]

ENERGY_MAX_SCALE = _cfg["metabolism"]["energy_max_scale"]
DRAIN_SCALE     = _cfg["metabolism"]["drain_scale"]
SIZE_TAX        = _cfg["metabolism"]["size_tax"]
SPEED_TAX       = _cfg["metabolism"]["speed_tax"]
TURN_TAX        = _cfg["metabolism"].get("turn_tax", 0.01)
AGE_TAX         = _cfg["metabolism"]["age_tax"]
SENSING_TAX     = _cfg["metabolism"]["sensing_tax"]

CAMO_ENABLED    = _cfg["camouflage"]["enabled"]
CAMO_BONUS      = _cfg["camouflage"]["detect_bonus"]

HGT_EAT_MIN     = _cfg["hgt"]["eat_rate_min"]
HGT_EAT_MAX     = _cfg["hgt"]["eat_rate_max"]
HGT_CONTACT_MIN = _cfg["hgt"]["contact_rate_min"]
HGT_CONTACT_MAX = _cfg["hgt"]["contact_rate_max"]
