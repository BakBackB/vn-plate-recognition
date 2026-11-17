"""audit_dataset.py — read-only check of a downloaded dataset before converting.

Answers two questions that decide the next step:
  1. Do any sources (groups) appear in both train and val?  -> leakage
  2. How many numbers are on each label line?  8/9 = OBB quad, 5 = axis-aligned box
Usage:
  python scripts/audit_dataset.py --images data/raw/images --labels data/raw/labels
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from convert_labels import _TOKEN_SPLIT, find_samples  # noqa: E402

DEFAULT_GROUP_REGEX = r"^(.+)_\d+$"


@dataclass
class AuditReport:
    per_split: Dict[str, Counter] = field(default_factory=dict)   # split -> group -> n images
    token_hist: Counter = field(default_factory=Counter)          # tokens/line -> n lines
    n_unlabeled: int = 0

    @property
    def shared_groups(self) -> Set[str]:
        """Groups present in more than one split (leakage)."""
        seen: Counter = Counter()
        for groups in self.per_split.values():
            seen.update(set(groups))
        return {g for g, n in seen.items() if n > 1}


def audit(image_dir: Path, label_dir: Path, group_regex: str = DEFAULT_GROUP_REGEX) -> AuditReport:
    report = AuditReport()
    samples = find_samples(image_dir, label_dir, "regex", re.compile(group_regex))
    for s in samples:
        parts = s.image.relative_to(image_dir).parts
        split = parts[0] if len(parts) > 1 else "(none)"
        report.per_split.setdefault(split, Counter())[s.group] += 1
        if s.label is None:
            report.n_unlabeled += 1
            continue
        for line in s.label.read_text(encoding="utf-8-sig").splitlines():
            tokens = [t for t in _TOKEN_SPLIT.split(line.strip()) if t]
            if tokens:
                report.token_hist[len(tokens)] += 1
    return report


def format_report(r: AuditReport) -> str:
    out = []
    for split, groups in sorted(r.per_split.items()):
        out.append(f"[{split}] {sum(groups.values())} images")
        out += [f"    {g:<14}{n}" for g, n in sorted(groups.items())]
    shared = sorted(r.shared_groups)
    out.append(f"groups in more than one split: {shared if shared else 'none'}"
               + ("   <-- LEAKAGE RISK" if shared else ""))
    out.append(f"label tokens per line: {dict(sorted(r.token_hist.items()))}")
    if r.token_hist.get(5):
        out.append("WARNING: 5-token lines are axis-aligned boxes, not OBB quads.")
    if set(r.token_hist) - {5, 8, 9}:
        out.append("WARNING: unexpected token counts; inspect the files.")
    out.append(f"images without a label file: {r.n_unlabeled}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=PROJECT_ROOT / "data" / "raw" / "images")
    ap.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data" / "raw" / "labels")
    ap.add_argument("--group-regex", default=DEFAULT_GROUP_REGEX)
    a = ap.parse_args()
    print(format_report(audit(a.images, a.labels, a.group_regex)))


if __name__ == "__main__":
    main()