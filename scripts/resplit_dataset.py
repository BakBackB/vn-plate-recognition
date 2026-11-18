"""resplit_dataset.py — leakage-aware re-split of the downloaded dataset (option C).

Frames are named <source>_<number>. Neighbouring numbers of one source show the
same vehicle, so a per-image split leaks. Here every source is cut into blocks of
consecutive frames; whole blocks go to val, the rest to train, and a few train
frames next to each val block are dropped (--gap). Class ids are preserved;
polygons with more than 4 vertices are reduced to 4 corners.

Reads  <images>/{train,val}/*.png and <labels>/{train,val}/*.txt (never modifies them).
Writes <out>/{images,labels}/{train,val}/ and <out>/split_manifest.csv.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.convert_labels import _prepare_output, _read_image, find_samples  # noqa: E402
from vnplate.utils.geometry import polygon_area  # noqa: E402

log = logging.getLogger("resplit_dataset")

NAME_RE = re.compile(r"^(?P<src>.+)_(?P<num>\d+)$")
SPLITS = ("train", "val")
DROP = "drop"
MANIFEST = "split_manifest.csv"
DUP_TOL = 1e-9
MIN_NORM_AREA = 1e-6


# ------------------------------------------------------------------ names
@dataclass(frozen=True)
class Frame:
    image: Path
    label: Path
    source: str          # casefolded, so Dieu_ and dieu_ are one source
    number: int


def parse_name(stem: str) -> Tuple[str, int]:
    m = NAME_RE.match(stem)
    if m is None:
        raise ValueError(f"cannot parse '<source>_<number>' from {stem!r}")
    return m.group("src").casefold(), int(m.group("num"))


def collect_frames(image_dir: Path, label_dir: Path) -> Tuple[Dict[str, List[Frame]], int]:
    """Group labelled images by source, each list sorted by frame number."""
    by_source: Dict[str, List[Frame]] = {}
    seen: Dict[Tuple[str, int], Path] = {}
    n_unlabeled = 0
    for s in find_samples(image_dir, label_dir, "image"):
        if s.label is None:
            log.warning("no label for %s, skipped", s.image.name)
            n_unlabeled += 1
            continue
        src, num = parse_name(s.image.stem)
        if (src, num) in seen:
            raise ValueError(f"{s.image.name} and {seen[(src, num)].name} are the "
                             "same frame (names differ only by case)")
        seen[(src, num)] = s.image
        by_source.setdefault(src, []).append(Frame(s.image, s.label, src, num))
    for frames in by_source.values():
        frames.sort(key=lambda f: (f.number, f.image.name))
    return by_source, n_unlabeled


# --------------------------------------------------------------- planning
def make_blocks(n: int, block_size: int) -> List[Tuple[int, int]]:
    """Half-open index ranges of consecutive frames; a short tail merges backwards."""
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    blocks = [(lo, min(lo + block_size, n)) for lo in range(0, n, block_size)]
    if len(blocks) > 1 and blocks[-1][1] - blocks[-1][0] < block_size / 2:
        blocks[-2:] = [(blocks[-2][0], n)]
    return blocks


def pick_val_blocks(n_blocks: int, train_ratio: float, rng: random.Random) -> Set[int]:
    """Choose val blocks, one per equal stratum so they spread along the sequence."""
    n_val = min(n_blocks - 1, max(1, round(n_blocks * (1.0 - train_ratio))))
    edges = [round(i * n_blocks / n_val) for i in range(n_val + 1)]
    return {rng.randrange(edges[i], edges[i + 1]) for i in range(n_val)}


def split_source(n: int, block_size: int, train_ratio: float, gap: int,
                 rng: random.Random) -> Tuple[List[str], List[int]]:
    """Per-frame label ('train' | 'val' | 'drop') and block id for one source."""
    blocks = make_blocks(n, block_size)
    if len(blocks) < 2:
        raise ValueError(f"only {n} frames = {len(blocks)} block(s); need >= 2 "
                         "(lower --block-size)")
    val_blocks = pick_val_blocks(len(blocks), train_ratio, rng)
    labels: List[str] = []
    block_of: List[int] = []
    for b, (lo, hi) in enumerate(blocks):
        labels += ["val" if b in val_blocks else "train"] * (hi - lo)
        block_of += [b] * (hi - lo)

    drop: Set[int] = set()
    for i in range(n - 1):
        if labels[i] == labels[i + 1]:
            continue
        near = (range(max(0, i - gap + 1), i + 1) if labels[i] == "train"
                else range(i + 1, min(n, i + 1 + gap)))
        drop.update(j for j in near if labels[j] == "train")   # val is never dropped
    return [DROP if i in drop else lab for i, lab in enumerate(labels)], block_of


def plan_splits(by_source: Dict[str, List[Frame]], train_ratio: float, block_size: int,
                gap: int, seed: int) -> Dict[Path, Tuple[str, int]]:
    plan: Dict[Path, Tuple[str, int]] = {}
    for src, frames in sorted(by_source.items()):
        rng = random.Random(f"{seed}:{src}")          # deterministic per source
        try:
            labels, blocks = split_source(len(frames), block_size, train_ratio, gap, rng)
        except ValueError as exc:
            raise ValueError(f"source {src!r}: {exc}") from exc
        for f, lab, b in zip(frames, labels, blocks):
            plan[f.image] = (lab, b)
    return plan


# ----------------------------------------------------------------- labels
@dataclass
class LabelResult:
    lines: List[str] = field(default_factory=list)
    classes: List[int] = field(default_factory=list)
    n_polygon: int = 0      # lines that had != 4 vertices
    n_bad: int = 0
    n_clipped: int = 0


def _parse_line(line: str, num_classes: int) -> Tuple[int, np.ndarray]:
    """'<cls> x1 y1 ... xk yk' (k >= 4) -> (cls, (k, 2) array)."""
    tokens = line.split()
    if len(tokens) < 9 or len(tokens) % 2 == 0:
        raise ValueError(f"expected class + >= 4 points, got {len(tokens)} tokens")
    try:
        vals = [float(t) for t in tokens]
    except ValueError:
        raise ValueError("non-numeric token")
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("non-finite coordinate")
    cls = int(vals[0])
    if cls != vals[0] or not 0 <= cls < num_classes:
        raise ValueError(f"class id {vals[0]} outside 0..{num_classes - 1}")
    return cls, np.asarray(vals[1:], dtype=np.float64).reshape(-1, 2)


def _dedupe(pts: np.ndarray) -> np.ndarray:
    """Drop consecutive repeated vertices (also last == first)."""
    out = [pts[0]]
    for p in pts[1:]:
        if np.max(np.abs(p - out[-1])) > DUP_TOL:
            out.append(p)
    if len(out) > 1 and np.max(np.abs(out[0] - out[-1])) <= DUP_TOL:
        out.pop()
    return np.asarray(out)


def polygon_to_quad(pts: np.ndarray, width: int, height: int) -> Optional[np.ndarray]:
    """Normalised polygon (k >= 3 vertices) -> normalised (4, 2) quad, or None.

    4 distinct vertices are kept as labelled. More than 4 use the minimum-area
    rectangle, computed in PIXELS (a rectangle in normalised space is not one
    in the image unless width == height).
    """
    pts = _dedupe(pts)
    if len(pts) < 4:
        return None
    if len(pts) == 4:
        quad = pts
    else:
        scale = np.array([width, height], dtype=np.float64)
        rect = cv2.minAreaRect((pts * scale).astype(np.float32))
        quad = cv2.boxPoints(rect).astype(np.float64) / scale
    quad = np.clip(quad, 0.0, 1.0)
    return None if polygon_area(quad) < MIN_NORM_AREA else quad


def convert_label_text(text: str, width: int, height: int, num_classes: int = 2,
                       name: str = "") -> LabelResult:
    """Rewrite one label file as '<cls> x1 y1 x2 y2 x3 y3 x4 y4' lines."""
    res = LabelResult()
    for lineno, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            cls, pts = _parse_line(raw, num_classes)
        except ValueError as exc:
            log.warning("%s:%d skipped (%s)", name, lineno, exc)
            res.n_bad += 1
            continue
        clipped = np.clip(pts, 0.0, 1.0)
        if not np.array_equal(clipped, pts):
            res.n_clipped += 1
        quad = polygon_to_quad(clipped, width, height)
        if quad is None:
            log.warning("%s:%d degenerate polygon skipped", name, lineno)
            res.n_bad += 1
            continue
        if len(pts) != 4:
            res.n_polygon += 1
        res.lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in quad.reshape(-1)))
        res.classes.append(cls)
    return res


# ----------------------------------------------------------------- driver
@dataclass
class Summary:
    n_images: Dict[str, int] = field(default_factory=lambda: {s: 0 for s in SPLITS})
    classes: Dict[str, Counter] = field(default_factory=lambda: {s: Counter() for s in SPLITS})
    per_source: Dict[str, Counter] = field(default_factory=dict)
    n_dropped: int = 0
    n_unlabeled: int = 0
    n_unreadable: int = 0
    n_polygon: int = 0
    n_bad_lines: int = 0
    n_clipped: int = 0


def _overlaps(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or a in b.parents or b in a.parents


def resplit(image_dir: Path, label_dir: Path, out_dir: Path, train_ratio: float = 0.7,
            block_size: int = 50, gap: int = 5, seed: int = 0, num_classes: int = 2,
            force: bool = False) -> Summary:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if block_size < 2 or gap < 0 or num_classes < 1:
        raise ValueError("need block_size >= 2, gap >= 0, num_classes >= 1")
    for src_dir in (image_dir, label_dir):
        if _overlaps(out_dir, src_dir):
            raise ValueError(f"output {out_dir} overlaps input {src_dir}; "
                             "the source dataset is never modified")

    by_source, n_unlabeled = collect_frames(image_dir, label_dir)
    if not by_source:
        raise ValueError("no labelled images found")
    plan = plan_splits(by_source, train_ratio, block_size, gap, seed)

    _prepare_output(out_dir, force)
    summary = Summary(n_unlabeled=n_unlabeled)
    manifest: List[Tuple[str, str, int, int, str]] = []
    for src, frames in sorted(by_source.items()):
        counts = summary.per_source.setdefault(src, Counter())
        for f in frames:
            split, block = plan[f.image]
            manifest.append((f.image.name, src, f.number, block, split))
            counts[split] += 1
            if split == DROP:
                summary.n_dropped += 1
                continue
            img = _read_image(f.image)
            if img is None:
                log.warning("unreadable image %s, skipped", f.image.name)
                summary.n_unreadable += 1
                continue
            h, w = img.shape[:2]
            res = convert_label_text(f.label.read_text(encoding="utf-8-sig"),
                                     w, h, num_classes, f.label.name)
            shutil.copy2(f.image, out_dir / "images" / split / f.image.name)
            (out_dir / "labels" / split / (f.image.stem + ".txt")).write_text(
                "".join(line + "\n" for line in res.lines), encoding="utf-8")
            summary.n_images[split] += 1
            summary.classes[split].update(res.classes)
            summary.n_polygon += res.n_polygon
            summary.n_bad_lines += res.n_bad
            summary.n_clipped += res.n_clipped

    with open(out_dir / MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "source", "number", "block", "split"])
        w.writerows(manifest)
    return summary


def format_summary(s: Summary) -> str:
    total = sum(s.n_images.values())
    val_share = s.n_images["val"] / total if total else 0.0
    out = [f"images : train {s.n_images['train']}  val {s.n_images['val']} "
           f"(val share {val_share:.1%})  dropped {s.n_dropped}"]
    for split in SPLITS:
        out.append(f"plates {split:<5}: {dict(sorted(s.classes[split].items()))}")
    for src, c in sorted(s.per_source.items()):
        out.append(f"  {src:<12} train {c['train']:>5}  val {c['val']:>5}  drop {c[DROP]:>4}")
    out.append(f"polygons reduced to 4 corners: {s.n_polygon}; bad lines: {s.n_bad_lines}; "
               f"clipped: {s.n_clipped}; unlabeled: {s.n_unlabeled}; unreadable: {s.n_unreadable}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=PROJECT_ROOT / "data" / "dataset" / "images")
    ap.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data" / "dataset" / "labels")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "dataset_resplit")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--block-size", type=int, default=50, help="consecutive frames per block")
    ap.add_argument("--gap", type=int, default=5,
                    help="train frames dropped next to each val block (0 = keep all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="rebuild a non-empty --out")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(format_summary(resplit(a.images, a.labels, a.out, a.train_ratio, a.block_size,
                                 a.gap, a.seed, a.num_classes, a.force)))


if __name__ == "__main__":
    main()