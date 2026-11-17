"""train_detector.py — Stage 1a: fine-tune YOLOv8n-OBB on the plate dataset.

All hyperparameters come from configs/config.yaml (detector.train).
    python scripts/train_detector.py
    python scripts/train_detector.py --epochs 1 --batch 2     # smoke test
Needs data/dataset (run scripts/convert_labels.py first) and network access the
first time (to download the pretrained weights).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vnplate.config import (DetectorConfig, DetectorTrainConfig,  # noqa: E402
                            load_config, select_device)

RUN_NAME = "plate_obb"
RESOLVED_YAML = "plate_data.resolved.yaml"


def write_resolved_data_yaml(data_yaml: Path, root: Path, out_path: Path) -> Path:
    """Copy the dataset YAML with an ABSOLUTE `path`.

    Why: Ultralytics may resolve a relative `path` against its own datasets
    directory instead of the project, so we pin it. Also fails early, with a
    clear message, if the train/val folders are missing.
    """
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    ds = Path(str(data["path"]))
    if not ds.is_absolute():
        ds = root / ds
    for split in ("train", "val"):
        folder = ds / str(data[split])
        if not folder.is_dir():
            raise FileNotFoundError(
                f"dataset folder not found: {folder} "
                "(run scripts/convert_labels.py first)")
    data["path"] = str(ds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return out_path


def build_train_args(train: DetectorTrainConfig, imgsz: int,
                     data: Path, project: Path) -> Dict[str, Any]:
    """Map config to Ultralytics train() kwargs (device is passed separately)."""
    return dict(data=str(data), imgsz=imgsz, epochs=train.epochs, batch=train.batch,
                degrees=train.degrees, perspective=train.perspective,
                fliplr=train.fliplr, hsv_v=train.hsv_v, mosaic=train.mosaic,
                project=str(project), name=RUN_NAME)


def _fit(model: Any, device: str, args: Dict[str, Any]) -> Any:
    return model.train(device=device, **args)


def train_detector(cfg: DetectorConfig, root: Path = PROJECT_ROOT) -> Path:
    """Train, copy best.pt to cfg.weights, return that path."""
    runs = root / "runs"
    data = write_resolved_data_yaml(cfg.train.data, root, runs / RESOLVED_YAML)
    args = build_train_args(cfg.train, cfg.imgsz, data, runs)

    from ultralytics import YOLO            # lazy: heavy import

    model = YOLO(cfg.train.pretrained)
    device = select_device(cfg.device)
    try:
        _fit(model, device, args)
    except (NotImplementedError, RuntimeError) as exc:
        if device != "mps":
            raise
        print(f"[warn] mps training failed ({exc}); retrying on cpu")
        model = YOLO(cfg.train.pretrained)  # fresh model: failed run left state
        _fit(model, "cpu", args)

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"training finished but {best} is missing")
    cfg.weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, cfg.weights)
    return cfg.weights


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override detector.train.epochs")
    ap.add_argument("--batch", type=int, default=None, help="override detector.train.batch")
    a = ap.parse_args()

    cfg = load_config(a.config)
    overrides = {k: v for k, v in (("epochs", a.epochs), ("batch", a.batch)) if v is not None}
    det = replace(cfg.detector, train=replace(cfg.detector.train, **overrides))
    print(f"weights saved to {train_detector(det)}")


if __name__ == "__main__":
    main()