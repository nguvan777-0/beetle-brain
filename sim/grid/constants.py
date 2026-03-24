"""Grid geometry constants derived from world config."""
import numpy as np
from sim.config import WIDTH, HEIGHT, N_RAYS, RAY_MAX, SIZE_MAX, CAMO_BONUS

GRID_SCALE   = 0.5
GW           = int(WIDTH  * GRID_SCALE)
GH           = int(HEIGHT * GRID_SCALE)
MAX_STEPS    = int(RAY_MAX * GRID_SCALE)

_STEPS       = np.arange(1, MAX_STEPS + 1, dtype=np.float32)
_RAY_OFFSETS = np.linspace(-1, 1, N_RAYS, dtype=np.float32)

PRED_R_PIX   = int(np.ceil((SIZE_MAX + CAMO_BONUS) * GRID_SCALE)) + 1
_PR_OFF      = np.arange(-PRED_R_PIX, PRED_R_PIX + 1, dtype=np.int32)
