import cv2
import numpy as np

import audit_dataset as ad


def _img(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(p), np.full((50, 100, 3), 255, np.uint8))


def _lbl(p, line):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(line + "\n")


def _build(tmp_path):
    im, lb = tmp_path / "images", tmp_path / "labels"
    quad = "0 10 10 50 10 50 30 10 30"          # 9 tokens
    box = "0 0.5 0.5 0.2 0.1"                   # 5 tokens
    for split, name, line in [("train", "carlong_0001", quad),
                              ("val", "carlong_0002", quad),
                              ("val", "dieu_0001", box)]:
        _img(im / split / f"{name}.png")
        _lbl(lb / split / f"{name}.txt", line)
    return im, lb


def test_shared_groups_and_counts(tmp_path):
    im, lb = _build(tmp_path)
    r = ad.audit(im, lb)
    assert r.shared_groups == {"carlong"}
    assert r.per_split["train"] == {"carlong": 1}
    assert r.per_split["val"] == {"carlong": 1, "dieu": 1}


def test_token_histogram_and_warning(tmp_path):
    im, lb = _build(tmp_path)
    r = ad.audit(im, lb)
    assert r.token_hist == {9: 2, 5: 1}
    text = ad.format_report(r)
    assert "LEAKAGE RISK" in text and "axis-aligned" in text


def test_clean_split_reports_no_leakage(tmp_path):
    im, lb = tmp_path / "images", tmp_path / "labels"
    for split, name in [("train", "hung_0001"), ("val", "tgmt_0001")]:
        _img(im / split / f"{name}.png")
        _lbl(lb / split / f"{name}.txt", "0 10 10 50 10 50 30 10 30")
    r = ad.audit(im, lb)
    assert r.shared_groups == set()
    assert "none" in ad.format_report(r)