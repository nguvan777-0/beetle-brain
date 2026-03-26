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

# Precomputed disk pixel offsets for each possible organism radius (1..SIZE_MAX*GRID_SCALE).
# Used by painter to rasterize organisms proportional to size.
_DISK_OFFSETS = {}
for _r in range(1, int(np.ceil(SIZE_MAX * GRID_SCALE)) + 1):
    _dy, _dx = np.mgrid[-_r:_r + 1, -_r:_r + 1]
    _m = (_dy ** 2 + _dx ** 2) <= _r ** 2
    _DISK_OFFSETS[_r] = (_dy[_m].astype(np.int32), _dx[_m].astype(np.int32))

DILATION_R_PIX = int(np.ceil(SIZE_MAX * GRID_SCALE))   # largest organism disk radius in grid pixels
DILATION_K     = 2 * DILATION_R_PIX + 1               # max-pool kernel size for disk dilation
