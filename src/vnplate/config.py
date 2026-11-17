"""config.py — typed loading of configs/config.yaml and the province table."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple, Union

import yaml

from .schemas import LAYOUTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # vn-plate-recognition/
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


# ---------------------------------------------------------------- sections
@dataclass(frozen=True)
class DetectorTrainConfig:
    """Hyperparameters for scripts/train_detector.py (YAML: detector.train)."""
    pretrained: str = "yolov8n-obb.pt"
    data: Path = PROJECT_ROOT / "configs" / "plate_data.yaml"
    epochs: int = 100
    batch: int = 16
    degrees: float = 10.0
    perspective: float = 0.0005
    fliplr: float = 0.0                      # mirrored plates never occur
    hsv_v: float = 0.5
    mosaic: float = 1.0


@dataclass(frozen=True)
class DetectorConfig:
    weights: Path
    imgsz: int = 640
    conf_threshold: float = 0.25
    min_area_px: float = 400.0
    device: str = "auto"                     # "auto" | "mps" | "cpu"
    train: DetectorTrainConfig = field(default_factory=DetectorTrainConfig)
    agnostic_nms: bool = True                # one plate = one detection across BSD/BSV

@dataclass(frozen=True)
class PreprocessConfig:
    layout_ratio_threshold: float = 2.5      # tau
    canvas: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {"one_row": (940, 220), "two_row": (560, 400)})
    bilateral_d: int = 7
    bilateral_sigma_color: float = 40.0
    bilateral_sigma_space: float = 40.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (4, 4)
    unsharp_sigma: float = 1.0
    unsharp_amount: float = 0.6


@dataclass(frozen=True)
class RecognizerConfig:
    alphabet: str = "0123456789ABCDEFGHKLMNPRSTUVXYZ-."
    row_height: int = 48
    border_pad: int = 6


@dataclass(frozen=True)
class PostprocessConfig:
    series_letters: str = "ABCDEFGHKLMNPRSTUVXYZ"
    province_codes: FrozenSet[str] = frozenset()   # loaded from JSON
    min_confidence: float = 0.50


@dataclass(frozen=True)
class Config:
    detector: DetectorConfig
    preprocess: PreprocessConfig
    recognizer: RecognizerConfig
    postprocess: PostprocessConfig
    root: Path = PROJECT_ROOT


# ---------------------------------------------------------------- helpers
def _resolve(p: Union[str, Path], root: Path) -> Path:
    """Relative paths in the YAML are relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else (root / p)


def _load_provinces(path: Path) -> FrozenSet[str]:
    with open(path, "r", encoding="utf-8") as f:
        codes = json.load(f)
    return frozenset(str(c).zfill(2) for c in codes)


def _validate_train(t: DetectorTrainConfig) -> None:
    if t.epochs < 1 or t.batch < 1:
        raise ValueError("detector.train.epochs and batch must be >= 1")
    for name in ("fliplr", "hsv_v", "mosaic"):
        if not 0.0 <= getattr(t, name) <= 1.0:
            raise ValueError(f"detector.train.{name} must be in [0, 1]")
    if not 0.0 <= t.degrees <= 180.0:
        raise ValueError("detector.train.degrees must be in [0, 180]")
    if not 0.0 <= t.perspective <= 0.001:
        raise ValueError("detector.train.perspective must be in [0, 0.001]")


def _validate(cfg: Config) -> None:
    pre = cfg.preprocess
    for name in LAYOUTS:
        if name not in pre.canvas:
            raise ValueError(f"preprocess.canvas missing layout '{name}'")
    if pre.layout_ratio_threshold <= 1.0:
        raise ValueError("layout_ratio_threshold must be > 1")
    if not 0.0 <= cfg.detector.conf_threshold <= 1.0:
        raise ValueError("detector.conf_threshold must be in [0, 1]")
    _validate_train(cfg.detector.train)
    if not 0.0 <= cfg.postprocess.min_confidence <= 1.0:
        raise ValueError("postprocess.min_confidence must be in [0, 1]")
    if not cfg.postprocess.province_codes:
        raise ValueError("province code table is empty")


# ---------------------------------------------------------------- public API
def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Read the YAML file and return a validated, typed Config."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    root = PROJECT_ROOT
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    d = raw["detector"]
    t = d.get("train") or {}
    train = DetectorTrainConfig(
        pretrained=str(t.get("pretrained", "yolov8n-obb.pt")),
        data=_resolve(t.get("data", "configs/plate_data.yaml"), root),
        epochs=int(t.get("epochs", 100)),
        batch=int(t.get("batch", 16)),
        degrees=float(t.get("degrees", 10.0)),
        perspective=float(t.get("perspective", 0.0005)),
        fliplr=float(t.get("fliplr", 0.0)),
        hsv_v=float(t.get("hsv_v", 0.5)),
        mosaic=float(t.get("mosaic", 1.0)),
    )
    detector = DetectorConfig(
        weights=_resolve(d["weights"], root),
        imgsz=int(d.get("imgsz", 640)),
        conf_threshold=float(d.get("conf_threshold", 0.25)),
        min_area_px=float(d.get("min_area_px", 400)),
        device=str(d.get("device", "auto")),
        train=train,
        agnostic_nms=bool(d.get("agnostic_nms", True)),
    )

    p = raw["preprocess"]
    canvas = {k: (int(v[0]), int(v[1])) for k, v in p["canvas"].items()}
    bil, cla, uns = p.get("bilateral", {}), p.get("clahe", {}), p.get("unsharp", {})
    preprocess = PreprocessConfig(
        layout_ratio_threshold=float(p.get("layout_ratio_threshold", 2.5)),
        canvas=canvas,
        bilateral_d=int(bil.get("d", 7)),
        bilateral_sigma_color=float(bil.get("sigma_color", 40)),
        bilateral_sigma_space=float(bil.get("sigma_space", 40)),
        clahe_clip_limit=float(cla.get("clip_limit", 2.0)),
        clahe_tile_grid=tuple(cla.get("tile_grid", (4, 4))),
        unsharp_sigma=float(uns.get("sigma", 1.0)),
        unsharp_amount=float(uns.get("amount", 0.6)),
    )

    r = raw["recognizer"]
    recognizer = RecognizerConfig(
        alphabet=str(r["alphabet"]),
        row_height=int(r.get("row_height", 48)),
        border_pad=int(r.get("border_pad", 6)),
    )

    q = raw["postprocess"]
    postprocess = PostprocessConfig(
        series_letters=str(q["series_letters"]),
        province_codes=_load_provinces(_resolve(q["province_codes"], root)),
        min_confidence=float(q.get("min_confidence", 0.50)),
    )

    cfg = Config(detector, preprocess, recognizer, postprocess, root)
    _validate(cfg)
    return cfg


def select_device(preference: str = "auto") -> str:
    """Return 'mps' (Apple GPU) if available, else 'cpu'. Used by detector only;
    PaddlePaddle always runs on CPU on macOS."""
    if preference != "auto":
        return preference
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"