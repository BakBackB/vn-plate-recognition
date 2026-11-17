"""convert_labels.py — raw images + TXT quads -> YOLO-OBB dataset (train/val).

Input : <images>/**/<name>.<png|jpg|...>   and   <annotations>/**/<name>.txt
        (one plate per line: [class] x1 y1 x2 y2 x3 y3 x4 y4, pixel or normalised)
Output: data/dataset/{images,labels}/{train,val}/  with lines
        "0 x1 y1 x2 y2 x3 y3 x4 y4" normalised to [0, 1] (single class: plate).

The split is by GROUP (source / scene / video), never by frame, to prevent
leakage (PROJECT_CONTEXT §7.6). Ratio is 70/30 as decided by the dataset owner.
"""
from __future__ import annotations

import argparse
import logging
import math
import random
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Pattern

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vnplate.utils.geometry import polygon_area  # noqa: E402

log = logging.getLogger("convert_labels")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val")
COORD_MODES = ("auto", "pixel", "normalized")
GROUP_MODES = ("image", "folder", "regex")
MIN_NORM_AREA = 1e-6            # quads smaller than this (normalised) are degenerate
EXHAUSTIVE_MAX_GROUPS = 16      # 2**16 subsets is still instant
_TOKEN_SPLIT = re.compile(r"[,\s]+")


# ------------------------------------------------------------ label parsing
@dataclass
class LabelResult:
    lines: List[str]
    n_bad: int = 0
    n_clipped: int = 0


def parse_label_line(line: str) -> Optional[List[float]]:
    """Return 8 floats (x1 y1 ... x4 y4), None for a blank line.

    A 9-token line is treated as "<class> + 8 coords"; the class id is dropped
    because the detector has a single class. Anything else raises ValueError.
    """
    tokens = [t for t in _TOKEN_SPLIT.split(line.strip()) if t]
    if not tokens:
        return None
    try:
        vals = [float(t) for t in tokens]
    except ValueError:
        raise ValueError("non-numeric token")
    if len(vals) == 9:
        vals = vals[1:]
    elif len(vals) != 8:
        raise ValueError(f"expected 8 or 9 numbers, got {len(vals)}")
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("non-finite coordinate")
    return vals


def convert_label_file(path: Path, width: int, height: int,
                       coords: str = "auto") -> LabelResult:
    """Convert one TXT file to normalised YOLO-OBB lines."""
    result = LabelResult(lines=[])
    quads: List[List[float]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        try:
            q = parse_label_line(raw)
        except ValueError as exc:
            log.warning("%s:%d skipped (%s)", path.name, lineno, exc)
            result.n_bad += 1
            continue
        if q is not None:
            quads.append(q)

    if coords == "auto":
        normalized = all(max(q) <= 1.0 for q in quads)
    else:
        normalized = coords == "normalized"

    scale = np.array([width, height], dtype=np.float64)
    for q in quads:
        pts = np.asarray(q, dtype=np.float64).reshape(4, 2)
        if not normalized:
            pts = pts / scale
        clipped = np.clip(pts, 0.0, 1.0)
        if not np.array_equal(clipped, pts):
            result.n_clipped += 1
        if polygon_area(clipped) < MIN_NORM_AREA:
            log.warning("%s: degenerate quad skipped", path.name)
            result.n_bad += 1
            continue
        result.lines.append("0 " + " ".join(f"{v:.6f}" for v in clipped.reshape(-1)))
    return result


# ---------------------------------------------------------------- sampling
@dataclass(frozen=True)
class Sample:
    image: Path
    label: Optional[Path]
    group: str


def group_key(image: Path, image_dir: Path, mode: str,
              regex: Optional[Pattern[str]] = None) -> str:
    """Identify the scene/video/source an image belongs to (the unit of splitting)."""
    if mode == "image":
        return image.stem
    if mode == "folder":
        return str(image.parent.relative_to(image_dir))
    if mode == "regex":
        if regex is None:
            raise ValueError("group_by='regex' needs a pattern")
        m = regex.search(image.stem)
        if m is None:
            return image.stem                # no match -> its own group
        return m.group(1) if m.groups() else m.group(0)
    raise ValueError(f"unknown group_by: {mode!r}")


def _label_path(image: Path, image_dir: Path, ann_dir: Path) -> Optional[Path]:
    """Annotation mirrors the image sub-path, or sits flat in ann_dir."""
    rel = image.relative_to(image_dir).with_suffix(".txt")
    for cand in (ann_dir / rel, ann_dir / (image.stem + ".txt")):
        if cand.is_file():
            return cand
    return None


def find_samples(image_dir: Path, ann_dir: Path, group_by: str = "image",
                 group_regex: Optional[Pattern[str]] = None) -> List[Sample]:
    images = sorted(p for p in image_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    dups = [n for n, c in Counter(p.name for p in images).items() if c > 1]
    if dups:
        raise ValueError(f"duplicate image filenames (output is flat): {dups[:5]}")
    return [Sample(p, _label_path(p, image_dir, ann_dir),
                   group_key(p, image_dir, group_by, group_regex)) for p in images]


# --------------------------------------------------------------- splitting
def split_groups(sizes: Dict[str, int], train_ratio: float, seed: int) -> Dict[str, str]:
    """Assign every group to 'train' or 'val' (deterministic for a given seed).

    A group is never divided. With few groups (e.g. 5 sources) a greedy fill can
    miss the ratio badly, so up to EXHAUSTIVE_MAX_GROUPS groups we try every
    subset and keep the one whose train share is closest to train_ratio
    (ties -> the first found, in seed-shuffled order). With more groups we fall
    back to shuffle-and-fill.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if len(sizes) < 2:
        raise ValueError("need at least 2 groups to make a train/val split "
                         "(check --group-by)")
    keys = sorted(sizes)
    random.Random(seed).shuffle(keys)
    target = train_ratio * sum(sizes.values())

    if len(keys) <= EXHAUSTIVE_MAX_GROUPS:
        best_mask, best_dev = 1, float("inf")
        for mask in range(1, 2 ** len(keys) - 1):        # both sides non-empty
            n_train = sum(sizes[k] for i, k in enumerate(keys) if mask >> i & 1)
            dev = abs(n_train - target)
            if dev < best_dev:
                best_mask, best_dev = mask, dev
        return {k: "train" if best_mask >> i & 1 else "val" for i, k in enumerate(keys)}

    assignment: Dict[str, str] = {}
    n_train = 0
    for k in keys:
        if n_train < target:
            assignment[k] = "train"
            n_train += sizes[k]
        else:
            assignment[k] = "val"
    if "val" not in assignment.values():     # one huge group swallowed everything
        assignment[keys[-1]] = "val"
    return assignment


# ------------------------------------------------------------------ driver
@dataclass
class Summary:
    n_images: Dict[str, int] = field(default_factory=lambda: {s: 0 for s in SPLITS})
    n_plates: Dict[str, int] = field(default_factory=lambda: {s: 0 for s in SPLITS})
    n_unlabeled: int = 0
    n_unreadable: int = 0
    n_bad_lines: int = 0
    n_clipped: int = 0


def _read_image(path: Path) -> Optional[np.ndarray]:
    """imdecode (not imread) so non-ASCII paths work on every OS."""
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:                         # noqa: BLE001 - corrupt file
        return None


def _prepare_output(out_dir: Path, force: bool) -> None:
    roots = [out_dir / "images", out_dir / "labels"]
    if any(r.exists() and next(r.rglob("*"), None) is not None for r in roots):
        if not force:
            raise FileExistsError(
                f"{out_dir} already has images/labels; pass --force to rebuild "
                "(old and new splits must never be mixed: leakage)")
        for r in roots:
            shutil.rmtree(r, ignore_errors=True)
    for kind in ("images", "labels"):
        for split in SPLITS:
            (out_dir / kind / split).mkdir(parents=True, exist_ok=True)


def convert_labels(image_dir: Path, ann_dir: Path, out_dir: Path,
                   train_ratio: float = 0.7, seed: int = 0,
                   group_by: str = "image",
                   group_regex: Optional[Pattern[str]] = None,
                   coords: str = "auto", force: bool = False) -> Summary:
    """Build data/dataset from raw images and TXT annotations."""
    if coords not in COORD_MODES:
        raise ValueError(f"coords must be one of {COORD_MODES}")
    samples = find_samples(image_dir, ann_dir, group_by, group_regex)
    summary = Summary()
    labeled = [s for s in samples if s.label is not None]
    for s in samples:
        if s.label is None:
            log.warning("no annotation for %s, skipped", s.image.name)
    summary.n_unlabeled = len(samples) - len(labeled)
    if not labeled:
        raise ValueError("no labelled images found")

    if group_by == "image":
        log.warning("splitting per image: consecutive frames of one source "
                    "may land in both train and val (leakage). Prefer "
                    "--group-by folder or --group-by regex.")
    assignment = split_groups(dict(Counter(s.group for s in labeled)), train_ratio, seed)
    _prepare_output(out_dir, force)

    for s in labeled:
        img = _read_image(s.image)
        if img is None:
            log.warning("unreadable image %s, skipped", s.image.name)
            summary.n_unreadable += 1
            continue
        h, w = img.shape[:2]
        res = convert_label_file(s.label, w, h, coords)
        split = assignment[s.group]
        shutil.copy2(s.image, out_dir / "images" / split / s.image.name)
        (out_dir / "labels" / split / (s.image.stem + ".txt")).write_text(
            "".join(line + "\n" for line in res.lines), encoding="utf-8")
        summary.n_images[split] += 1
        summary.n_plates[split] += len(res.lines)
        summary.n_bad_lines += res.n_bad
        summary.n_clipped += res.n_clipped
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=PROJECT_ROOT / "data" / "raw")
    ap.add_argument("--annotations", type=Path, default=PROJECT_ROOT / "data" / "annotations")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "dataset")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-by", choices=GROUP_MODES, default="image")
    ap.add_argument("--group-regex", default=None,
                    help="applied to the image stem; group = first capture group")
    ap.add_argument("--coords", choices=COORD_MODES, default="auto")
    ap.add_argument("--force", action="store_true", help="rebuild a non-empty output")
    a = ap.parse_args()
    if a.group_by == "regex" and not a.group_regex:
        ap.error("--group-by regex requires --group-regex")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = convert_labels(a.images, a.annotations, a.out, a.train_ratio, a.seed,
                       a.group_by, re.compile(a.group_regex) if a.group_regex else None,
                       a.coords, a.force)
    print(f"images : train {s.n_images['train']}  val {s.n_images['val']}")
    print(f"plates : train {s.n_plates['train']}  val {s.n_plates['val']}")
    print(f"skipped: {s.n_unlabeled} unlabeled, {s.n_unreadable} unreadable; "
          f"{s.n_bad_lines} bad lines, {s.n_clipped} clipped quads")


if __name__ == "__main__":
    main()