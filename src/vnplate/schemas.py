"""schemas.py — dataclasses exchanged between pipeline stages.

Keeping this contract fixed makes every stage replaceable and testable alone.
Python 3.9 compatible (no `X | None` syntax).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

# Layout names are shared by preprocess / recognizer / postprocess / config.
LAYOUT_ONE_ROW = "one_row"
LAYOUT_TWO_ROW = "two_row"
LAYOUTS = (LAYOUT_ONE_ROW, LAYOUT_TWO_ROW)


@dataclass
class Detection:
    """Stage 1 output: one plate found in the image."""
    corners: np.ndarray          # (4, 2) float32, ORIGINAL image pixels, any order
    conf: float                  # detector confidence in [0, 1]

    def __post_init__(self) -> None:
        self.corners = np.asarray(self.corners, dtype=np.float32)
        if self.corners.shape != (4, 2):
            raise ValueError(f"corners must be (4, 2), got {self.corners.shape}")
        self.conf = float(self.conf)


@dataclass
class PlateImage:
    """Stage 2 output: rectified, enhanced plate."""
    rect: np.ndarray             # BGR, canonical canvas size of the layout
    gray: np.ndarray             # uint8 gray, enhanced, dark text on light bg
    layout: str                  # LAYOUT_ONE_ROW | LAYOUT_TWO_ROW

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"unknown layout: {self.layout!r}")


@dataclass
class RawText:
    """Stage 3 output: OCR text per row (top row first), before grammar."""
    rows: List[str]              # e.g. ["59-X1", "123.45"] or ["30A-123.45"]
    row_conf: List[float]        # one confidence per row

    def __post_init__(self) -> None:
        if len(self.rows) != len(self.row_conf):
            raise ValueError("rows and row_conf must have the same length")


@dataclass
class PlateResult:
    """Stage 4 output (+ fields filled in by pipeline.recognize)."""
    text: str                    # formatted, e.g. "59X1-123.45" ("" if unreadable)
    valid: bool                  # passed grammar + province-table check
    province: str = ""           # "59"
    series: str = ""             # "X1"
    number: str = ""             # "123.45"
    conf: float = 0.0            # mean OCR row confidence
    reason: str = ""             # why invalid ("" if valid)
    # filled by pipeline.recognize():
    det_conf: float = 0.0
    score: float = 0.0           # s = det_conf * conf
    corners: Optional[np.ndarray] = None   # (4, 2), kept for drawing


@dataclass
class Models:
    """Loaded heavy models, created once and passed to recognize()."""
    detector: Any                # ultralytics.YOLO
    ocr: Any                     # paddleocr.PaddleOCR (recognition-only use)