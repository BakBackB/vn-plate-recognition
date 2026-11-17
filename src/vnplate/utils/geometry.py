"""geometry.py — pure geometry helpers shared by several stages.

Only `polygon_area` exists so far (needed by Stage 1). `order_corners` and
`quad_size` are added at build step 3.
"""
from __future__ import annotations

import numpy as np


def polygon_area(pts: np.ndarray) -> float:
    """Area of a simple polygon via the shoelace formula.

    Why: Stage 1 uses it to drop plates too small to read (min_area_px).
    Works for any vertex order (clockwise or counter-clockwise); returns >= 0.

    Args:
        pts: (N, 2) array-like, N >= 3, vertices in order around the polygon.
    """
    p = np.asarray(pts, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 3:
        raise ValueError(f"pts must be (N>=3, 2), got {p.shape}")
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))