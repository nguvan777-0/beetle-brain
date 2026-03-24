"""Draw individual wights and their vision rays."""
import numpy as np
import pygame
from sim.config import N_RAYS

_LINEAGE_COLORS = [
    (255, 100, 100), (100, 200, 255), (100, 255, 140), (255, 200,  60),
    (200, 100, 255), (255, 150,  50), ( 80, 220, 200), (220, 220, 100),
    (180,  80, 180), (100, 180, 100), (255, 120, 180), (140, 180, 255),
]


def draw_organism(surf, x, y, angle, size, r, g, b, lineage_id=0):
    xi, yi = int(x), int(y)
    halo = _LINEAGE_COLORS[lineage_id % len(_LINEAGE_COLORS)]
    pygame.draw.circle(surf, halo, (xi, yi), max(2, int(size) + 2), 1)
    pygame.draw.circle(surf, (r, g, b), (xi, yi), max(1, int(size)))
    ex = xi + int(np.cos(angle) * (size + 4))
    ey = yi + int(np.sin(angle) * (size + 4))
    pygame.draw.line(surf, (255, 255, 255), (xi, yi), (ex, ey), 1)


def draw_rays(surf, x, y, fov, angle, ray_len):
    offsets = np.linspace(-1, 1, N_RAYS) * fov * 0.5
    for off in offsets:
        a  = angle + off
        ex = x + np.cos(a) * ray_len
        ey = y + np.sin(a) * ray_len
        pygame.draw.line(surf, (50, 50, 80), (int(x), int(y)), (int(ex), int(ey)), 1)
