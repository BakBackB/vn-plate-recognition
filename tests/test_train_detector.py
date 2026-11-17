import sys
import types
from pathlib import Path

import pytest
import yaml

import train_detector as td
from vnplate.config import DetectorConfig, DetectorTrainConfig


def _dataset(root):
    for s in ("train", "val"):
        (root / "data" / "dataset" / "images" / s).mkdir(parents=True)
    y = root / "plate_data.yaml"
    y.write_text(yaml.safe_dump({"path": "data/dataset", "train": "images/train",
                                 "val": "images/val", "names": {0: "plate"}}))
    return y


def test_resolved_yaml_has_absolute_path(tmp_path):
    src = _dataset(tmp_path)
    out = td.write_resolved_data_yaml(src, tmp_path, tmp_path / "runs" / "r.yaml")
    data = yaml.safe_load(out.read_text())
    assert Path(data["path"]) == tmp_path / "data" / "dataset"
    assert data["names"] == {0: "plate"}


def test_missing_split_folder_raises(tmp_path):
    src = _dataset(tmp_path)
    (tmp_path / "data" / "dataset" / "images" / "val").rmdir()
    with pytest.raises(FileNotFoundError):
        td.write_resolved_data_yaml(src, tmp_path, tmp_path / "r.yaml")


def test_build_train_args_maps_config(tmp_path):
    t = DetectorTrainConfig(epochs=7, batch=3, degrees=4.0, fliplr=0.0)
    a = td.build_train_args(t, 512, tmp_path / "d.yaml", tmp_path / "runs")
    assert (a["epochs"], a["batch"], a["imgsz"]) == (7, 3, 512)
    assert a["degrees"] == 4.0 and a["fliplr"] == 0.0
    assert a["data"] == str(tmp_path / "d.yaml")


class FakeYOLO:
    calls = []

    def __init__(self, name):
        self.name, self.trainer = name, None

    def train(self, device, **kw):
        FakeYOLO.calls.append((device, kw))
        if device == "mps":
            raise RuntimeError("unsupported op")
        save = Path(kw["project"]) / kw["name"]
        (save / "weights").mkdir(parents=True, exist_ok=True)
        (save / "weights" / "best.pt").write_bytes(b"w")
        self.trainer = types.SimpleNamespace(save_dir=save)


@pytest.fixture
def fake_ultralytics(monkeypatch):
    FakeYOLO.calls = []
    mod = types.ModuleType("ultralytics")
    mod.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    return FakeYOLO


def _cfg(tmp_path, device):
    return DetectorConfig(weights=tmp_path / "weights" / "plate_obb.pt", device=device,
                          train=DetectorTrainConfig(data=_dataset(tmp_path), epochs=1))


def test_best_weights_are_copied(tmp_path, fake_ultralytics):
    out = td.train_detector(_cfg(tmp_path, "cpu"), root=tmp_path)
    assert out.read_bytes() == b"w"
    assert [d for d, _ in fake_ultralytics.calls] == ["cpu"]
    assert fake_ultralytics.calls[0][1]["data"].endswith(td.RESOLVED_YAML)


def test_mps_failure_falls_back_to_cpu(tmp_path, fake_ultralytics):
    out = td.train_detector(_cfg(tmp_path, "mps"), root=tmp_path)
    assert out.is_file()
    assert [d for d, _ in fake_ultralytics.calls] == ["mps", "cpu"]