"""geometry.py — pure geometry helpers shared by several stages.

polygon_area : Stage 1 (drops plates below min_area_px).
order_corners, quad_size : Stage 2 (perspective warp and layout rule).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _as_quad(pts: np.ndarray) -> np.ndarray:
    """Validate a (4, 2) array of finite numbers and return it as float64."""
    p = np.asarray(pts, dtype=np.float64)
    if p.shape != (4, 2):
        raise ValueError(f"pts must be (4, 2), got {p.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("pts contains NaN or Inf")
    return p


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


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 corners (any input order) as [TL, TR, BR, BL], float32.

    Why: cv2.getPerspectiveTransform pairs source and destination points by
    index, and Stage 1 returns corners in unspecified order.
    Rule (PROJECT_CONTEXT §8): TL = argmin(x+y), BR = argmax(x+y),
    TR = argmin(y-x), BL = argmax(y-x). Image coordinates: y grows downward.
    Assumes tilt below ~45 deg; at larger tilts two roles can pick the same
    point, which is detected and reported instead of returning a wrong order.
    """
    p = _as_quad(pts)
    s = p[:, 0] + p[:, 1]
    d = p[:, 1] - p[:, 0]
    idx = [int(np.argmin(s)), int(np.argmin(d)), int(np.argmax(s)), int(np.argmax(d))]
    if len(set(idx)) != 4:
        raise ValueError("cannot order corners unambiguously (tilt >= ~45 deg "
                         "or degenerate quadrilateral)")
    return p[idx].astype(np.float32)


def quad_size(pts: np.ndarray) -> Tuple[float, float]:
    """(width, height) of an ORDERED quadrilateral [TL, TR, BR, BL].

    Why: the layout rule (D2) compares W/H against tau. Each side is the mean
    of two opposite edges, which is robust to perspective skew.
    """
    tl, tr, br, bl = _as_quad(pts)
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    return float(width), float(height)