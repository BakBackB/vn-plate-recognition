"""preprocess.py — Stage 2: rectify the detected plate and normalise its look.

Input : the full image and the 4 corners found by Stage 1 (any order).
Output: PlateImage(rect, gray, layout) for Stage 3.

Why this stage exists for Vietnamese plates (PROJECT_CONTEXT section 7.2):
  D2  one-row (~4.3:1) and two-row (~1.4:1) plates need different canvases, so
      the layout is decided *before* the warp.
  D3  white-on-blue/red plates are inverted so the text is always dark on light.
Edge-preserving filters are used so the thin '.' and '-' survive.

Python 3.9 compatible.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import PreprocessConfig
from .schemas import LAYOUT_ONE_ROW, LAYOUT_TWO_ROW, PlateImage
from .utils.geometry import order_corners, quad_size


# ---------------------------------------------------------------- validation
def _as_bgr(image: np.ndarray) -> np.ndarray:
    """Return a 3-channel uint8 view/copy of `image`, or raise ValueError.

    Accepts gray (H, W), BGR (H, W, 3) and BGRA (H, W, 4). The input is never
    modified.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {image.dtype}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"unsupported image shape {image.shape}")


def _check_gray(gray: np.ndarray) -> None:
    if (not isinstance(gray, np.ndarray) or gray.ndim != 2
            or gray.size == 0 or gray.dtype != np.uint8):
        raise ValueError("gray must be a non-empty 2-D uint8 array")


# ---------------------------------------------------------------- layout (D2)
def classify_layout(pts: np.ndarray, tau: float) -> str:
    """Pick the layout from the aspect ratio of an ORDERED quad (rule 8).

    `W / H >= tau` -> one_row, else two_row. A quad with zero (or NaN) width or
    height cannot be rectified, so it raises ValueError instead of dividing by 0.
    """
    width, height = quad_size(pts)
    if not (width > 0.0 and height > 0.0):
        raise ValueError(f"degenerate quad: width={width}, height={height}")
    return LAYOUT_ONE_ROW if width / height >= tau else LAYOUT_TWO_ROW


# ---------------------------------------------------------------- geometry
def warp_plate(image: np.ndarray, pts: np.ndarray,
               size: Tuple[int, int]) -> np.ndarray:
    """Perspective-warp the ordered quad `pts` onto a (W, H) canvas.

    Kept separate from preprocess_plate so the geometric step (ablation A1) can
    be tested and measured on its own.
    """
    width, height = size
    dst = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np.asarray(pts, dtype=np.float32), dst)
    return cv2.warpPerspective(image, matrix, (width, height),
                               flags=cv2.INTER_CUBIC)


# ---------------------------------------------------------------- photometry
def ensure_dark_on_light(gray: np.ndarray) -> np.ndarray:
    """Return `gray` with dark text on a light background (D3, polarity rule).

    Otsu splits the plate into two classes; rho is the share of the bright
    class. The text is the minority, so rho < 0.5 means bright text -> invert.
    A constant image has no text; it is returned unchanged. The input is not
    modified (an inversion returns a new array).
    """
    _check_gray(gray)
    if gray.min() == gray.max():
        return gray
    threshold, _ = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rho = float(np.count_nonzero(gray > threshold)) / gray.size
    return cv2.bitwise_not(gray) if rho < 0.5 else gray


def unsharp_mask(gray: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    """Sharpen with I + amount * (I - G_sigma * I), saturated to uint8."""
    _check_gray(gray)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    # addWeighted(I, 1 + a, blur, -a) == I + a * (I - blur), with uint8 clipping
    return cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0.0)


def enhance_gray(rect: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Photometric chain on a rectified BGR plate -> uint8 gray, dark on light.

    Order follows section 7.2: gray -> bilateral -> CLAHE -> polarity -> unsharp.
    Polarity comes after CLAHE (as specified) and before sharpening.
    """
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, cfg.bilateral_d,
                               cfg.bilateral_sigma_color,
                               cfg.bilateral_sigma_space)
    tile = tuple(int(v) for v in cfg.clahe_tile_grid)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=tile)
    gray = clahe.apply(gray)
    gray = ensure_dark_on_light(gray)
    return unsharp_mask(gray, cfg.unsharp_sigma, cfg.unsharp_amount)


# ---------------------------------------------------------------- public API
def preprocess_plate(image: np.ndarray, corners: np.ndarray,
                     cfg: PreprocessConfig) -> PlateImage:
    """Stage 2 entry point: image + 4 corners -> PlateImage.

    Raises ValueError for an invalid image, invalid corners (see
    order_corners) or a degenerate quad. `cfg` is `Config.preprocess`.
    """
    bgr = _as_bgr(image)
    pts = order_corners(corners)                      # TL, TR, BR, BL
    layout = classify_layout(pts, cfg.layout_ratio_threshold)
    rect = warp_plate(bgr, pts, cfg.canvas[layout])
    gray = enhance_gray(rect, cfg)
    return PlateImage(rect=rect, gray=gray, layout=layout)