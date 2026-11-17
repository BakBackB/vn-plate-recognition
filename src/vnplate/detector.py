"""detector.py — Stage 1: find license plates as 4-corner quadrilaterals (D1)."""
from __future__ import annotations

from typing import Any, List

import numpy as np

from .config import DetectorConfig, select_device
from .schemas import Detection
from .utils.geometry import polygon_area


def load_detector(cfg: DetectorConfig) -> Any:
    """Load the trained YOLOv8-OBB weights once; the result goes in `Models.detector`.

    ultralytics is imported lazily so importing this module (and unit-testing
    `detect_plates` with a fake model) does not require the heavy dependency.
    """
    if not cfg.weights.is_file():
        raise FileNotFoundError(
            f"detector weights not found: {cfg.weights} "
            "(run scripts/train_detector.py first)")
    from ultralytics import YOLO
    return YOLO(str(cfg.weights))


def _to_numpy(x: Any) -> np.ndarray:
    """Accept a torch tensor (any device) or an array-like and return ndarray."""
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _predict(model: Any, image: np.ndarray, cfg: DetectorConfig, device: str) -> Any:
    return model.predict(image, imgsz=cfg.imgsz, conf=cfg.conf_threshold,
                         device=device, agnostic_nms=cfg.agnostic_nms,
                         verbose=False)[0]


def _extract_quads(result: Any):
    """Return (quads (N,4,2), confs (N,)) as numpy; empty arrays if nothing found."""
    obb = getattr(result, "obb", None)
    if obb is None:
        return np.zeros((0, 4, 2), np.float32), np.zeros((0,), np.float32)
    quads = _to_numpy(obb.xyxyxyxy)
    confs = _to_numpy(obb.conf)
    if quads.size == 0:
        return np.zeros((0, 4, 2), np.float32), np.zeros((0,), np.float32)
    return quads.reshape(-1, 4, 2), confs.reshape(-1)


def detect_plates(image: np.ndarray, model: Any, cfg: DetectorConfig) -> List[Detection]:
    """Detect plates in a BGR image.

    Returns Detections with corners in ORIGINAL image pixels (order unspecified;
    Stage 2 orders them), sorted by confidence descending. Plates whose
    quadrilateral area is below `cfg.min_area_px` are skipped as unreadable.
    """
    if (not isinstance(image, np.ndarray) or image.size == 0
            or image.ndim not in (2, 3)):
        raise ValueError("image must be a non-empty 2-D or 3-D numpy array")

    device = select_device(cfg.device)
    try:
        result = _predict(model, image, cfg, device)
    except (NotImplementedError, RuntimeError):
        if device != "mps":
            raise
        result = _predict(model, image, cfg, "cpu")   # §11: mps -> cpu fallback

    quads, confs = _extract_quads(result)
    detections: List[Detection] = []
    for quad, score in zip(quads, confs):
        if not np.all(np.isfinite(quad)):
            continue
        if polygon_area(quad) < cfg.min_area_px:
            continue
        detections.append(Detection(corners=quad, conf=float(score)))
    detections.sort(key=lambda d: d.conf, reverse=True)
    return detections