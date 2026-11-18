import random

import cv2
import numpy as np
import pytest

from scripts import resplit_dataset as rs


def _img(p, w=40, h=20):
    p.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(p), np.full((h, w, 3), 255, np.uint8))


def _txt(p, line):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(line + "\n")


# ------------------------------------------------------------------ names
def test_parse_name():
    assert rs.parse_name("Dieu_0024") == ("dieu", 24)
    assert rs.parse_name("a_b_0001") == ("a_b", 1)
    with pytest.raises(ValueError):
        rs.parse_name("noNumber")


# ----------------------------------------------------------------- blocks
def test_make_blocks_merges_short_tail():
    assert rs.make_blocks(130, 50) == [(0, 50), (50, 100), (100, 130)]
    assert rs.make_blocks(120, 50) == [(0, 50), (50, 120)]
    assert rs.make_blocks(30, 50) == [(0, 30)]


def test_pick_val_blocks_one_per_stratum():
    picked = rs.pick_val_blocks(10, 0.7, random.Random(1))
    assert len(picked) == 3
    for lo, hi in [(0, 3), (3, 7), (7, 10)]:
        assert sum(lo <= b < hi for b in picked) == 1


def test_split_source_blocks_are_contiguous():
    labels, blocks = rs.split_source(200, 50, 0.5, 0, random.Random(0))
    for b in set(blocks):
        assert len({lab for lab, bb in zip(labels, blocks) if bb == b}) == 1
    assert {"train", "val"} <= set(labels)


def test_split_source_gap_separates_train_from_val():
    labels, _ = rs.split_source(200, 50, 0.5, 3, random.Random(0))
    n_drop = labels.count(rs.DROP)
    assert n_drop > 0 and n_drop % 3 == 0
    for a, b in zip(labels, labels[1:]):
        assert {a, b} != {"train", "val"}


def test_split_source_is_deterministic():
    a = rs.split_source(200, 50, 0.7, 2, random.Random("0:x"))
    b = rs.split_source(200, 50, 0.7, 2, random.Random("0:x"))
    assert a == b


def test_single_block_source_rejected():
    with pytest.raises(ValueError):
        rs.split_source(30, 50, 0.7, 0, random.Random(0))


# ----------------------------------------------------------------- labels
def test_duplicate_vertex_is_removed_keeping_the_quad():
    pts = np.array([(.1, .2), (.5, .2), (.5, .6), (.1, .6), (.1, .6)])
    quad = rs.polygon_to_quad(pts, 200, 100)
    assert np.allclose(quad, pts[:4])


def test_hexagon_becomes_min_area_rectangle():
    pts = np.array([(.2, .2), (.4, .2), (.6, .2), (.6, .6), (.4, .6), (.2, .6)])
    quad = rs.polygon_to_quad(pts, 200, 100)
    assert quad.shape == (4, 2)
    assert rs.polygon_area(quad) == pytest.approx(0.16, abs=1e-3)
    assert quad[:, 0].min() == pytest.approx(0.2, abs=1e-3)
    assert quad[:, 0].max() == pytest.approx(0.6, abs=1e-3)


def test_degenerate_polygons_return_none():
    assert rs.polygon_to_quad(np.array([(.1, .1), (.2, .2), (.3, .1)]), 100, 100) is None
    assert rs.polygon_to_quad(np.full((5, 2), 0.5), 100, 100) is None


def test_convert_keeps_class_id():
    res = rs.convert_label_text("1 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6\n", 200, 100)
    assert res.lines == ["1 0.100000 0.200000 0.500000 0.200000 "
                         "0.500000 0.600000 0.100000 0.600000"]
    assert res.classes == [1] and res.n_polygon == 0


def test_convert_eleven_token_line_is_counted_as_polygon():
    res = rs.convert_label_text("1 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6 0.1 0.6", 200, 100)
    assert len(res.lines) == 1 and len(res.lines[0].split()) == 9
    assert res.n_polygon == 1


@pytest.mark.parametrize("bad", ["5 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6",       # class out of range
                                 "0 0.1 0.2 0.5 0.2 0.5 0.6 0.1",            # 8 tokens
                                 "0 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6 0.3",    # even count
                                 "0 nan 0.2 0.5 0.2 0.5 0.6 0.1 0.6"])
def test_convert_rejects_bad_lines(bad):
    res = rs.convert_label_text(bad, 200, 100, num_classes=2)
    assert res.lines == [] and res.n_bad == 1


def test_blank_lines_ignored():
    res = rs.convert_label_text("\n  \n", 200, 100)
    assert res.lines == [] and res.n_bad == 0


# ------------------------------------------------------------- end to end
def _build(tmp_path, per_source=100):
    im, lb = tmp_path / "images", tmp_path / "labels"
    for si, src in enumerate(("Aaa", "bbb")):
        for n in range(1, per_source + 1):
            split = "val" if n % 4 == 0 else "train"      # per-frame split, like Kaggle
            line = f"{si} 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6"
            if src == "bbb" and n == 5:
                line += " 0.1 0.6"                        # 11 tokens, repeated vertex
            _img(im / split / f"{src}_{n:04d}.png")
            _txt(lb / split / f"{src}_{n:04d}.txt", line)
    return im, lb


def _run(tmp_path, **kw):
    im, lb = _build(tmp_path)
    out = tmp_path / "out"
    args = dict(train_ratio=0.7, block_size=20, gap=2, seed=0)
    args.update(kw)
    return im, lb, out, rs.resplit(im, lb, out, **args)


def test_end_to_end(tmp_path):
    im, lb, out, s = _run(tmp_path)
    assert len(list(im.rglob("*.png"))) == 200               # source untouched
    assert s.n_images["train"] > 0 and s.n_images["val"] > 0 and s.n_dropped > 0
    assert sum(s.n_images.values()) + s.n_dropped == 200
    assert s.n_polygon == 1 and s.n_bad_lines == 0
    for f in (out / "labels").rglob("*.txt"):
        parts = f.read_text().split()
        assert len(parts) == 9
        assert parts[0] == ("0" if f.name.startswith("Aaa") else "1")
    with open(out / rs.MANIFEST) as fh:
        assert len(fh.readlines()) == 201


def test_no_adjacent_frames_across_splits(tmp_path):
    _, _, out, _ = _run(tmp_path)
    kept = {}
    for split in rs.SPLITS:
        for p in (out / "images" / split).iterdir():
            src, num = rs.parse_name(p.stem)
            kept[(src, num)] = split
    for (src, num), split in kept.items():
        nxt = kept.get((src, num + 1))
        assert nxt is None or nxt == split


def test_output_must_not_overlap_input(tmp_path):
    im, lb = _build(tmp_path)
    for bad in (im, tmp_path):
        with pytest.raises(ValueError):
            rs.resplit(im, lb, bad)


def test_existing_output_requires_force(tmp_path):
    im, lb, out, _ = _run(tmp_path)
    with pytest.raises(FileExistsError):
        rs.resplit(im, lb, out, block_size=20, gap=2)
    rs.resplit(im, lb, out, block_size=20, gap=2, force=True)


def test_unlabeled_image_is_skipped(tmp_path):
    im, lb = _build(tmp_path)
    _img(im / "train" / "Aaa_9999.png")
    s = rs.resplit(im, lb, tmp_path / "out", block_size=20, gap=2)
    assert s.n_unlabeled == 1


def test_source_too_short_names_the_source(tmp_path):
    im, lb = tmp_path / "images", tmp_path / "labels"
    for n in range(1, 31):
        _img(im / "train" / f"zzz_{n:04d}.png")
        _txt(lb / "train" / f"zzz_{n:04d}.txt", "0 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6")
    with pytest.raises(ValueError, match="zzz"):
        rs.resplit(im, lb, tmp_path / "out", block_size=50)