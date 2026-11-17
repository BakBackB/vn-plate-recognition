import json
from pathlib import Path

import pytest
import yaml

from vnplate.config import PROJECT_ROOT, DetectorConfig, load_config


def _write(tmp_path, train=None):
    prov = tmp_path / "prov.json"
    prov.write_text(json.dumps(["30", "59"]))
    det = {"weights": str(tmp_path / "w.pt"), "imgsz": 640, "conf_threshold": 0.25,
           "min_area_px": 400, "device": "cpu"}
    if train is not None:
        det["train"] = train
    raw = {"detector": det,
           "preprocess": {"layout_ratio_threshold": 2.5,
                          "canvas": {"one_row": [940, 220], "two_row": [560, 400]}},
           "recognizer": {"alphabet": "0123456789ABC-."},
           "postprocess": {"series_letters": "ABC", "province_codes": str(prov)}}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_missing_train_section_uses_defaults(tmp_path):
    t = load_config(_write(tmp_path)).detector.train
    assert (t.epochs, t.batch, t.fliplr, t.pretrained) == (100, 16, 0.0, "yolov8n-obb.pt")


def test_overrides_are_loaded(tmp_path):
    cfg = load_config(_write(tmp_path, {"epochs": 3, "batch": 2, "degrees": 5,
                                        "mosaic": 0.5, "pretrained": "x.pt"}))
    t = cfg.detector.train
    assert (t.epochs, t.batch, t.degrees, t.mosaic, t.pretrained) == (3, 2, 5.0, 0.5, "x.pt")


def test_relative_data_path_resolves_to_project_root(tmp_path):
    t = load_config(_write(tmp_path, {"data": "configs/plate_data.yaml"})).detector.train
    assert t.data == PROJECT_ROOT / "configs" / "plate_data.yaml"


def test_detector_config_still_constructible_without_train():
    assert DetectorConfig(weights=Path("w.pt")).train.epochs == 100


@pytest.mark.parametrize("bad", [{"epochs": 0}, {"batch": 0}, {"fliplr": 1.5},
                                 {"mosaic": -0.1}, {"degrees": 200},
                                 {"perspective": 0.01}])
def test_invalid_train_values_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, bad))