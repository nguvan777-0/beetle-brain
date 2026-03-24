"""Draw individual wights and their vision rays."""
import numpy as np
import pygame
from sim.config import N_RAYS


def draw_organism(surf, x, y, angle, size, r, g, b):
    xi, yi = int(x), int(y)
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
