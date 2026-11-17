from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vnplate.config import DetectorConfig
from vnplate.detector import detect_plates, load_detector


def _cfg(**kw):
    base = dict(weights=Path("w.pt"), imgsz=640, conf_threshold=0.25,
                min_area_px=400.0, device="cpu")
    base.update(kw)
    return DetectorConfig(**base)


def _rect(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


class FakeModel:
    def __init__(self, quads, confs, fail_on=None):
        self.quads, self.confs, self.fail_on = quads, confs, fail_on
        self.calls = []

    def predict(self, image, **kw):
        self.calls.append(kw)
        if kw["device"] == self.fail_on:
            raise RuntimeError("unsupported op")
        obb = SimpleNamespace(
            xyxyxyxy=np.array(self.quads, dtype=np.float32).reshape(-1, 4, 2),
            conf=np.array(self.confs, dtype=np.float32))
        return [SimpleNamespace(obb=obb)]


IMG = np.zeros((480, 640, 3), np.uint8)


def test_sorted_by_conf_desc_and_types():
    m = FakeModel([_rect(0, 0, 40, 20), _rect(100, 100, 50, 25)], [0.4, 0.9])
    dets = detect_plates(IMG, m, _cfg())
    assert [round(d.conf, 2) for d in dets] == [0.9, 0.4]
    assert dets[0].corners.shape == (4, 2)
    assert dets[0].corners.dtype == np.float32


def test_small_plates_filtered():
    m = FakeModel([_rect(0, 0, 10, 10), _rect(0, 0, 40, 20)], [0.9, 0.8])  # 100 vs 800
    dets = detect_plates(IMG, m, _cfg())
    assert len(dets) == 1 and dets[0].conf == pytest.approx(0.8)


def test_area_threshold_comes_from_config():
    m = FakeModel([_rect(0, 0, 40, 20)], [0.9])
    assert detect_plates(IMG, m, _cfg(min_area_px=1000.0)) == []


def test_no_detections():
    assert detect_plates(IMG, FakeModel([], []), _cfg()) == []


def test_obb_none():
    class NoObb:
        def predict(self, image, **kw):
            return [SimpleNamespace(obb=None)]
    assert detect_plates(IMG, NoObb(), _cfg()) == []


def test_nonfinite_quad_skipped():
    bad = _rect(0, 0, 40, 20)
    bad[0][0] = float("nan")
    m = FakeModel([bad, _rect(0, 0, 40, 20)], [0.9, 0.5])
    dets = detect_plates(IMG, m, _cfg())
    assert len(dets) == 1 and dets[0].conf == pytest.approx(0.5)


def test_predict_params_forwarded():
    m = FakeModel([], [])
    detect_plates(IMG, m, _cfg(imgsz=320, conf_threshold=0.4))
    kw = m.calls[0]
    assert kw["imgsz"] == 320 and kw["conf"] == 0.4
    assert kw["device"] == "cpu" and kw["verbose"] is False


def test_mps_falls_back_to_cpu():
    m = FakeModel([_rect(0, 0, 40, 20)], [0.9], fail_on="mps")
    dets = detect_plates(IMG, m, _cfg(device="mps"))
    assert len(dets) == 1
    assert [c["device"] for c in m.calls] == ["mps", "cpu"]


def test_cpu_error_is_not_swallowed():
    m = FakeModel([], [], fail_on="cpu")
    with pytest.raises(RuntimeError):
        detect_plates(IMG, m, _cfg(device="cpu"))


@pytest.mark.parametrize("bad", [None, np.zeros((0, 0, 3), np.uint8),
                                 np.zeros(5, np.uint8), "img.jpg"])
def test_invalid_image(bad):
    with pytest.raises(ValueError):
        detect_plates(bad, FakeModel([], []), _cfg())


def test_torch_tensor_input():
    torch = pytest.importorskip("torch")

    class TensorModel:
        def predict(self, image, **kw):
            obb = SimpleNamespace(
                xyxyxyxy=torch.tensor([_rect(0, 0, 40, 20)], dtype=torch.float32),
                conf=torch.tensor([0.7]))
            return [SimpleNamespace(obb=obb)]

    dets = detect_plates(IMG, TensorModel(), _cfg())
    assert len(dets) == 1 and dets[0].conf == pytest.approx(0.7)


def test_load_detector_missing_weights(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_detector(_cfg(weights=tmp_path / "nope.pt"))

def test_agnostic_nms_default_true_is_forwarded():
    m = FakeModel([], [])
    detect_plates(IMG, m, _cfg())
    assert m.calls[0]["agnostic_nms"] is True

def test_agnostic_nms_can_be_disabled_from_config():
    m = FakeModel([], [])
    detect_plates(IMG, m, _cfg(agnostic_nms=False))
    assert m.calls[0]["agnostic_nms"] is False