# PROJECT_CONTEXT.md — Vietnam License Plate Detection

> Single source of truth for the project. Derived from `Project_Proposal.pdf`.
> Use this file as context when implementing or reviewing any module.
> Course: Artificial Intelligence · Student: Trần Minh Phúc
> Environment: macOS, VS Code, Python 3.9, venv

---

## 1. Project in one paragraph

A four-stage ALPR pipeline designed for Vietnamese plates:

1. **Detect** — YOLOv8-OBB finds each plate as a 4-corner quadrilateral.
2. **Preprocess** — OpenCV rectifies (perspective warp), classifies layout (one-row / two-row), normalises contrast and polarity.
3. **Read** — PaddleOCR (recognition-only, CRNN/SVTR + CTC) reads each row separately.
4. **Post-process** — regex grammar + position-aware character correction → validated plate such as `59X1-123.45`.

```
image/frame → [1 detect_plates] → Detection(corners, conf)
            → [2 preprocess_plate] → PlateImage(rect, gray, layout)
            → [3 read_plate] → RawText(rows, row_conf)
            → [4 postprocess] → PlateResult(text, valid, province, series, number, conf)
```

### Design consequences (referenced everywhere as D1–D6)

| ID | Fact about Vietnamese plates | Consequence | Stage |
|----|------------------------------|-------------|-------|
| D1 | Small, tilted motorbike plates in dense traffic | Detect oriented boxes (4 corners), not axis-aligned | 1 |
| D2 | Two physical layouts (one-row ≈ 4.3:1, two-row ≈ 1.4:1) | Rectify to layout-specific canvas, classify layout, read row by row | 2, 3 |
| D3 | White-on-blue/red vs black-on-white/yellow | Normalise polarity to dark text on light background | 2 |
| D4 | Tiny alphabet, thin separators `-` and `.` | Recognition-only OCR, alphabet filter, border padding | 3 |
| D5 | Strict grammar (province + series + 4–5 digits) | Regex + province-code table validates | 4 |
| D6 | Look-alikes 0/D, 8/B, 5/S, 2/Z … | Position-aware correction (digits vs letters by position) | 4 |

**Scope:** civilian plates (white, blue, yellow), front/rear view. Military (red) and diplomatic plates are future work. Every rule that depends on regulations (province codes, series letters, number length) lives in config files — never hard-coded.

---

## 2. Environment setup

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "numpy<2"                      # pin numpy first
pip install torch torchvision
pip install "ultralytics>=8.1,<9" opencv-python pyyaml pytest
pip install paddlepaddle==2.6.2 paddleocr==2.7.3
pip freeze > requirements.txt
```

Notes:
- VS Code: *Python: Select Interpreter* → `.venv/bin/python`.
- PaddlePaddle runs on CPU on macOS; PyTorch can train on Apple GPU with `device="mps"` (fall back to `cpu`).
- If pip reports an OpenCV conflict: `pip install paddleocr==2.7.3 --no-deps`, then add missing deps by hand. Test in a fresh venv before freezing.
- Python 3.9: use `Optional[X]`, `List[X]` (or `from __future__ import annotations`). `X | None` needs 3.10.

---

## 3. Project tree

```
vn-plate-recognition/
|-- .venv/
|-- .vscode/{settings.json, launch.json}
|-- configs/
|   |-- config.yaml            # thresholds, canvas sizes, paths
|   |-- plate_data.yaml        # Ultralytics dataset description
|   `-- province_codes.json    # valid 2-digit province codes
|-- data/
|   |-- raw/  annotations/
|   |-- dataset/{images,labels}/{train,val,test}/   # YOLO-OBB format
|   `-- ocr_eval/              # rectified crops + ground-truth (csv)
|-- weights/plate_obb.pt
|-- src/vnplate/
|   |-- __init__.py
|   |-- schemas.py             # dataclasses
|   |-- config.py              # load_config()
|   |-- detector.py            # Stage 1
|   |-- preprocess.py          # Stage 2
|   |-- recognizer.py          # Stage 3
|   |-- postprocess.py         # Stage 4
|   |-- pipeline.py            # recognize()
|   `-- utils/{geometry.py, metrics.py, visualize.py}
|-- scripts/{convert_labels.py, train_detector.py, run_image.py, run_video.py, evaluate.py}
|-- tests/{test_geometry.py, test_preprocess.py, test_recognizer.py, test_postprocess.py, test_pipeline.py}
|-- outputs/
|-- requirements.txt
`-- README.md
```

### Recommended build order

1. Environment + tree; write `schemas.py`, `config.py`.
2. `utils/geometry.py` + tests (no model needed).
3. `postprocess.py` + tests (pure Python, fastest feedback).
4. `preprocess.py`; test on hand-picked photos with known corners.
5. Collect/annotate data → `convert_labels.py` → train detector → `detector.py`.
6. `recognizer.py`; evaluate on rectified crops.
7. `pipeline.py`, `run_image.py`, `evaluate.py`.
8. Ablation study, error analysis, tune thresholds.

---

## 4. Configuration files

### 4.1 `configs/config.yaml`

```yaml
detector:
  weights: weights/plate_obb.pt
  imgsz: 640
  conf_threshold: 0.25
  min_area_px: 400            # ignore plates too small to read
  device: auto                # (addition) auto -> mps if available else cpu

preprocess:
  layout_ratio_threshold: 2.5 # tau in the layout rule
  canvas:                     # (width, height) px, ~2 px per mm
    one_row: [940, 220]
    two_row: [560, 400]
  # (addition) enhancement parameters, previously hard-coded in pseudocode
  bilateral: {d: 7, sigma_color: 40, sigma_space: 40}
  clahe: {clip_limit: 2.0, tile_grid: [4, 4]}
  unsharp: {sigma: 1.0, amount: 0.6}

recognizer:
  alphabet: "0123456789ABCDEFGHKLMNPRSTUVXYZ-."
  row_height: 48
  border_pad: 6

postprocess:
  series_letters: "ABCDEFGHKLMNPRSTUVXYZ"
  province_codes: configs/province_codes.json
  min_confidence: 0.50
```

> Items marked *(addition)* are not in the proposal's YAML; they move values that the proposal hard-coded into the config so they can be tuned in the ablation study.

### 4.2 `configs/province_codes.json`

A JSON list of valid 2-digit strings, e.g. `["11", "12", "14", "15", "..."]`. **Fill from the current Ministry of Public Security list; keep it updatable.**

### 4.3 `configs/plate_data.yaml`

```yaml
path: data/dataset
train: images/train
val: images/val
test: images/test
names:
  0: plate
```

---

## 5. `src/vnplate/schemas.py`

Data contract between stages (all images are OpenCV BGR/gray `uint8` arrays).

| Stage | Function | Input → Output |
|-------|----------|----------------|
| 1 | `detect_plates` | image → `List[Detection]` |
| 2 | `preprocess_plate` | image, corners → `PlateImage` |
| 3 | `read_plate` | `PlateImage` → `RawText` |
| 4 | `postprocess` | `RawText`, layout → `PlateResult` |
| all | `recognize` | image → `List[PlateResult]` |

```python
"""schemas.py — dataclasses exchanged between pipeline stages.

Keeping this contract fixed makes every stage replaceable and testable alone.
Python 3.9 compatible (no `X | None` syntax).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

# Layout names are shared by preprocess / recognizer / postprocess / config.
LAYOUT_ONE_ROW = "one_row"
LAYOUT_TWO_ROW = "two_row"
LAYOUTS = (LAYOUT_ONE_ROW, LAYOUT_TWO_ROW)


@dataclass
class Detection:
    """Stage 1 output: one plate found in the image."""
    corners: np.ndarray          # (4, 2) float32, ORIGINAL image pixels, any order
    conf: float                  # detector confidence in [0, 1]

    def __post_init__(self) -> None:
        self.corners = np.asarray(self.corners, dtype=np.float32)
        if self.corners.shape != (4, 2):
            raise ValueError(f"corners must be (4, 2), got {self.corners.shape}")
        self.conf = float(self.conf)


@dataclass
class PlateImage:
    """Stage 2 output: rectified, enhanced plate."""
    rect: np.ndarray             # BGR, canonical canvas size of the layout
    gray: np.ndarray             # uint8 gray, enhanced, dark text on light bg
    layout: str                  # LAYOUT_ONE_ROW | LAYOUT_TWO_ROW

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"unknown layout: {self.layout!r}")


@dataclass
class RawText:
    """Stage 3 output: OCR text per row (top row first), before grammar."""
    rows: List[str]              # e.g. ["59-X1", "123.45"] or ["30A-123.45"]
    row_conf: List[float]        # one confidence per row

    def __post_init__(self) -> None:
        if len(self.rows) != len(self.row_conf):
            raise ValueError("rows and row_conf must have the same length")


@dataclass
class PlateResult:
    """Stage 4 output (+ fields filled in by pipeline.recognize)."""
    text: str                    # formatted, e.g. "59X1-123.45" ("" if unreadable)
    valid: bool                  # passed grammar + province-table check
    province: str = ""           # "59"
    series: str = ""             # "X1"
    number: str = ""             # "123.45"
    conf: float = 0.0            # mean OCR row confidence
    reason: str = ""             # why invalid ("" if valid)
    # filled by pipeline.recognize():
    det_conf: float = 0.0
    score: float = 0.0           # s = det_conf * conf
    corners: Optional[np.ndarray] = None   # (4, 2), kept for drawing


@dataclass
class Models:
    """Loaded heavy models, created once and passed to recognize()."""
    detector: Any                # ultralytics.YOLO
    ocr: Any                     # paddleocr.PaddleOCR (recognition-only use)
```

---

## 6. `src/vnplate/config.py`

```python
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
class DetectorConfig:
    weights: Path
    imgsz: int = 640
    conf_threshold: float = 0.25
    min_area_px: float = 400.0
    device: str = "auto"                     # "auto" | "mps" | "cpu"


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


def _validate(cfg: Config) -> None:
    pre = cfg.preprocess
    for name in LAYOUTS:
        if name not in pre.canvas:
            raise ValueError(f"preprocess.canvas missing layout '{name}'")
    if pre.layout_ratio_threshold <= 1.0:
        raise ValueError("layout_ratio_threshold must be > 1")
    if not 0.0 <= cfg.detector.conf_threshold <= 1.0:
        raise ValueError("detector.conf_threshold must be in [0, 1]")
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
    detector = DetectorConfig(
        weights=_resolve(d["weights"], root),
        imgsz=int(d.get("imgsz", 640)),
        conf_threshold=float(d.get("conf_threshold", 0.25)),
        min_area_px=float(d.get("min_area_px", 400)),
        device=str(d.get("device", "auto")),
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
```

---

## 7. Pseudocode of all stages

### 7.1 Stage 1 — Detect license plate (`detector.py`, `scripts/train_detector.py`)

*Why Vietnam-specific (D1):* motorbike plates are small, rotated and adjacent to other plates; an axis-aligned box would include background and neighbours. OBB returns the four true corners, which are exactly the input of the perspective transform. Augmentation: moderate rotation/perspective, strong brightness changes, **no horizontal flip** (mirrored plates never occur).

```text
# ---------- Stage 1a: train the detector (run once) ----------
FUNCTION train_detector(cfg):
    model = YOLO("yolov8n-obb.pt")                 # pretrained OBB -> transfer learning
    model.train(
        data = "configs/plate_data.yaml",          # 1 class "plate"; label = 4 corner points
        imgsz = 640, epochs = 100, batch = 16,
        device = "mps",                            # Apple GPU, fall back to "cpu"
        degrees = 10, perspective = 0.0005,        # tilted plates are normal
        fliplr = 0.0,                              # mirrored plates never occur
        hsv_v = 0.5, mosaic = 1.0)                 # night / glare robustness
    copy runs/.../best.pt TO weights/plate_obb.pt

# ---------- Stage 1b: inference ----------
FUNCTION detect_plates(image, model, cfg):
    result = model.predict(image, imgsz=cfg.imgsz, conf=cfg.conf_threshold, verbose=False)[0]
    detections = []
    FOR EACH (quad, score) IN zip(result.obb.xyxyxyxy, result.obb.conf):
        # quad: 4x2 array, corner points in ORIGINAL image pixels
        IF polygon_area(quad) < cfg.min_area_px:
            CONTINUE                                # far-away plate: unreadable, skip
        detections.append(Detection(corners=quad, conf=score))
    RETURN sort detections by conf, descending
```

### 7.2 Stage 2 — Preprocessing (`preprocess.py`)

*Why Vietnam-specific (D2, D3):* two layouts with very different aspect ratios → decide layout first, then use the matching canvas (a two-row plate squashed to a one-row canvas is unreadable). Blue/red plates have white text; the recogniser expects dark-on-light, so polarity is normalised. Edge-preserving filters keep the thin `.` and `-`.

```text
FUNCTION preprocess_plate(image, corners, cfg):
    pts    = order_corners(corners)                # TL, TR, BR, BL
    layout = classify_layout(pts, cfg.tau)         # "one_row" | "two_row"
    (W, H) = cfg.canvas[layout]

    # --- geometric correction ---
    dst  = [(0,0), (W-1,0), (W-1,H-1), (0,H-1)]
    M    = cv2.getPerspectiveTransform(pts, dst)   # 8x8 linear system
    rect = cv2.warpPerspective(image, M, (W, H), flags=INTER_CUBIC)

    # --- photometric correction ---
    gray = cv2.cvtColor(rect, COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=40)   # denoise, keep edges
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)).apply(gray) # glare / shadow
    gray = ensure_dark_on_light(gray)              # D3
    gray = unsharp_mask(gray, sigma=1.0, amount=0.6)
    RETURN PlateImage(rect=rect, gray=gray, layout=layout)

FUNCTION classify_layout(pts, tau):                # D2, rule (8)
    (W, H) = quad_size(pts)
    RETURN "one_row" IF W / H >= tau ELSE "two_row"   # tau ~ 2.5 (tune on validation set)

FUNCTION ensure_dark_on_light(gray):               # D3, polarity rule
    (t, _) = cv2.threshold(gray, 0, 255, THRESH_BINARY + THRESH_OTSU)
    rho    = fraction of pixels with gray > t
    IF rho < 0.5:                                  # bright = minority = text
        gray = 255 - gray
    RETURN gray

FUNCTION unsharp_mask(gray, sigma, amount):        # I + lambda*(I - G*I)
    blur = cv2.GaussianBlur(gray, (0,0), sigma)
    RETURN clip(gray + amount * (gray - blur), 0, 255).astype(uint8)
```

### 7.3 Stage 3 — Read text (`recognizer.py`)

*Why Vietnam-specific (D2, D4):* a generic OCR reads one line; on a two-row plate it would merge or misorder rows. Split rows ourselves, read top then bottom. Recognition-only (`det=False`) because the text region is known. Anything outside the plate alphabet is an error and is removed.

```text
FUNCTION read_plate(plate, ocr, cfg):
    rows = split_rows(plate.gray, plate.layout)    # one_row -> [whole]; two_row -> [top, bottom]
    texts, confs = [], []
    FOR EACH row IN rows:
        row = resize_keep_ratio(row, height=cfg.row_height)   # PP-OCR expects height 48
        row = pad_border(row, cfg.border_pad, value=255)      # avoid clipping edge chars
        row = gray_to_bgr(row)                                # (note) Paddle expects 3 channels
        (txt, conf) = ocr.ocr(row, det=False, cls=False)[0][0]
        txt = upper(txt)
        txt = keep_only(txt, cfg.alphabet)                    # D4
        texts.append(txt); confs.append(conf)
    RETURN RawText(rows=texts, row_conf=confs)

# Examples: motorbike -> ["59-X1", "123.45"]   car (1 row) -> ["30A-123.45"]

FUNCTION split_rows(gray, layout):
    IF layout == "one_row": RETURN [gray]
    H   = height(gray)
    bin = otsu_binarize(gray) with text = 1
    P   = sum(bin, axis=1)                         # horizontal projection P(y)
    y   = argmin(P[0.4*H : 0.6*H]) + 0.4*H         # emptiest gap near the middle
    RETURN [gray[0:y, :], gray[y:H, :]]            # (default fallback: y = H/2)

FUNCTION resize_keep_ratio(img, height):
    scale = height / img.height
    RETURN cv2.resize(img, (round(img.width*scale), height), INTER_CUBIC)

FUNCTION pad_border(img, p, value):
    RETURN cv2.copyMakeBorder(img, p, p, p, p, BORDER_CONSTANT, value)

FUNCTION keep_only(s, alphabet):
    RETURN "".join(c FOR c IN s IF c IN alphabet)
```

### 7.4 Stage 4 — Post-processing (`postprocess.py`)

*Why Vietnam-specific (D5, D6):* strict structure → grammar + province table validates and repairs OCR output. Order matters: **split head (province+series) from number by layout first, then parse** — `30A12345` is ambiguous otherwise.

```text
# Grammar
HEAD_RE   = r"^(?P<prov>\d{2})(?P<series>[<SERIES_LETTERS>][<SERIES_LETTERS>0-9]?)$"
NUMBER_RE = r"^\d{4,5}$"
TO_DIGIT  = {"O":"0","Q":"0","D":"0","I":"1","L":"1","Z":"2","S":"5","G":"6","B":"8"}
TO_LETTER = {"0":"D","8":"B","5":"S","2":"Z","6":"G","4":"A"}

FUNCTION postprocess(raw, layout, cfg):
    (head, number) = split_head_number(raw, layout)
    head   = remove(head,   "-. ")
    number = remove(number, "-. ")

    # D6: position-aware correction
    head[0], head[1] = to_digit(head[0]), to_digit(head[1])   # province: digits only
    head[2]          = to_letter(head[2])                     # series letter: letters only
    # head[3] (motorbike 2nd series char) may be digit or letter -> leave as read
    number = map_each_char(number, to_digit)                  # number: digits only
    # (guard every index access: head may be shorter than 3 chars)

    # D5: validate
    m      = re.match(HEAD_RE, head)
    ok     = m AND number matches NUMBER_RE AND m.prov IN cfg.province_codes
    reason = "" IF ok ELSE which check failed

    # format
    IF len(number) == 5: number = number[0:3] + "." + number[3:5]   # 123.45
    text = head + "-" + number                                      # 59X1-123.45
    conf = mean(raw.row_conf)
    RETURN PlateResult(text, valid=ok, province, series, number, conf, reason)

FUNCTION split_head_number(raw, layout):
    IF layout == "two_row" AND len(raw.rows) == 2: RETURN (raw.rows[0], raw.rows[1])
    s = raw.rows[0]
    IF "-" IN s: RETURN s.split("-", 1)            # "30A-123.45" -> ("30A", "123.45")
    c = remove(s, ".")                             # dash lost: number = last digits
    FOR n IN (5, 4):
        IF last n chars of c are digits AND the rest matches HEAD_RE:
            RETURN (c[:-n], c[-n:])
    RETURN (c, "")                                 # cannot split -> flagged invalid

FUNCTION to_digit(c):  RETURN TO_DIGIT.get(c, c)
FUNCTION to_letter(c): RETURN TO_LETTER.get(c, c)
```

### 7.5 Pipeline (`pipeline.py`)

```text
FUNCTION recognize(image, models, cfg):
    results = []
    FOR EACH det IN detect_plates(image, models.detector, cfg.detector):   # Stage 1
        plate = preprocess_plate(image, det.corners, cfg.preprocess)       # Stage 2
        raw   = read_plate(plate, models.ocr, cfg.recognizer)              # Stage 3
        res   = postprocess(raw, plate.layout, cfg.postprocess)            # Stage 4
        res.det_conf = det.conf
        res.score    = det.conf * res.conf                                 # s = c_det * c_ocr
        res.corners  = det.corners
        results.append(res)
    RETURN results
```

### 7.6 Utilities

```text
# ---------- utils/geometry.py ----------
FUNCTION order_corners(pts):                       # 4x2, any order
    s = pts[:,0] + pts[:,1];  d = pts[:,1] - pts[:,0]       # x+y and y-x
    TL = pts[argmin(s)]; BR = pts[argmax(s)]
    TR = pts[argmin(d)]; BL = pts[argmax(d)]
    RETURN float32 [TL, TR, BR, BL]                # assumes tilt < ~45 deg

FUNCTION quad_size(pts):                           # pts ordered
    W = (dist(TR,TL) + dist(BR,BL)) / 2
    H = (dist(BL,TL) + dist(BR,TR)) / 2
    RETURN (W, H)

FUNCTION polygon_area(pts):                        # shoelace
    RETURN 0.5 * abs( sum( x[i]*y[i+1] - x[i+1]*y[i] ) )

# ---------- utils/metrics.py ----------
FUNCTION levenshtein(a, b):                        # DP, O(|a|*|b|)
    D[i][0] = i;  D[0][j] = j
    D[i][j] = min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1] + (a[i] != b[j]))
    RETURN D[|a|][|b|]

FUNCTION cer(pred, truth):  RETURN levenshtein(pred, truth) / len(truth)
FUNCTION plate_accuracy(preds, truths): RETURN count(pred == truth) / len(truths)

# ---------- utils/visualize.py ----------
FUNCTION draw_result(image, result):
    cv2.polylines(image, [result.corners], isClosed=TRUE, color=green_if_valid_else_red)
    cv2.putText(image, result.text + f" {result.score:.2f}", near top-left corner)
    RETURN image

# ---------- scripts/convert_labels.py ----------
FUNCTION convert_labels(annotation_file, image_dir, out_dir):
    FOR EACH image, quads IN annotation_file:
        (Wi, Hi) = image size
        write "0 x1/Wi y1/Hi x2/Wi y2/Hi x3/Wi y3/Hi x4/Wi y4/Hi" per quad   # YOLO-OBB, normalised
    split by SCENE / video (not by image) into train 70 / val 20 / test 10    # prevents leakage
```

---

## 8. Mathematical rules (quick reference)

- **Oriented box → corners:** `p_k = (xc,yc) + R(θ)·(±w/2, ±h/2)`. Training loss uses Gaussian/ProbIoU (Bhattacharyya distance), handled by Ultralytics.
- **Detection metrics:** Precision, Recall, AP, mAP50, mAP50-95.
- **Homography:** `[u',v',w'] = H·[x,y,1]`, `h33 = 1`, 4 corner pairs → 8×8 system (`cv2.getPerspectiveTransform`).
- **⋆ Corner ordering:** TL = argmin(x+y), BR = argmax(x+y), TR = argmin(y−x), BL = argmax(y−x).
- **⋆ Layout rule:** `W/H ≥ τ → one_row`, else `two_row`; real ratios ≈ 4.3 and ≈ 1.4, geometric mean ≈ 2.4, start τ = 2.5.
- **Filtering:** gray `I = 0.299R + 0.587G + 0.114B`; bilateral; CLAHE; **⋆ unsharp** `I + λ(I − G_σ∗I)`.
- **⋆ Polarity rule:** Otsu `t`; `ρ = |{I>t}|/(HW)`; if `ρ < 0.5` invert.
- **CTC:** `p(l|x) = Σ_{π∈B⁻¹(l)} p(π|x)`, `L = −ln p(l|x)`; greedy decoding = argmax + collapse.
- **⋆ Confidence:** `s = c_det · mean(c_rows)`.
- **⋆ Row split:** cut at `y* = argmin_{0.4H ≤ y ≤ 0.6H} P(y)`.
- **Evaluation:** `CER = d(ŷ,y)/|y|`, `Acc_plate = #{ŷ=y}/#plates`.

(⋆ = implemented by us, not by a library.)

---

## 9. What libraries do vs. what we write

| Stage | Libraries already do | We must write |
|-------|----------------------|---------------|
| 1 | Network, training loop, NMS, OBB decoding | Dataset + OBB annotation, `plate_data.yaml`, training script, `detect_plates()` with area filter, label conversion |
| 2 | Homography solver, warp, filters, CLAHE | `order_corners`, `classify_layout`, canvas sizes, `ensure_dark_on_light`, enhancement chain |
| 3 | Recognition network, CTC decoding | `split_rows`, padding/resizing, alphabet filter, row merging, confidence, (optional) fine-tuning data |
| 4 | Regex engine | Grammar, province table, position-aware correction, formatting, validity flag |
| All | — | `pipeline.py`, config loading, visualisation, metrics, evaluation, unit tests |

---

## 10. Verification plan

**Principles:** (1) test each stage in isolation with *oracle* input from the previous stage; (2) split data by scene/video, never by frame; (3) cover Vietnamese conditions: 1-row cars, 2-row cars, motorbikes; white/blue/yellow; day/night/rain/glare; tilted/occluded; far/near.

| Stage | What we do | Measure | Target |
|-------|-----------|---------|--------|
| 1 | `yolo obb val ... split=test`; overlay 50 random images; inspect false negatives by condition | mAP50, mAP50-95, precision, recall | recall ≥ 0.95, mAP50 ≥ 0.90 |
| 2 | Unit tests (`order_corners`, `classify_layout`, `ensure_dark_on_light`); hand-labelled warps; visual check of 30 plates per layout/colour in `outputs/stage2/` | pass rate, layout acc., polarity acc. | all tests pass; layout acc. ≥ 0.98 |
| 3 | Feed ground-truth rectified crops to `read_plate`; compare to `data/ocr_eval/labels.csv`; confusion list (e.g. `8->B`, `0->D`) | CER, row accuracy | CER ≤ 0.05 |
| 4 | Table-driven unit tests (below) | pass rate | 100% |
| End-to-end | `scripts/evaluate.py` on test set | exact-match acc., valid-rate, latency/frame | acc ≥ 0.85 |

The Stage 3 confusion list is the data source for tuning `TO_DIGIT` / `TO_LETTER`. Extra robustness check: Torchvision blur/brightness perturbations, plot CER vs. strength.

### Stage 2 unit tests (pseudocode)

```text
test_order_corners_any_permutation:
    square = [(10,10),(110,10),(110,40),(10,40)]
    FOR perm IN all_permutations(square): ASSERT order_corners(perm) == [TL,TR,BR,BL]
test_identity_warp:
    out = preprocess_plate(flat_image, corners_of_whole_image)
    ASSERT mean_abs_diff(out.rect, resize(flat_image)) < small_threshold
test_layout:
    ASSERT classify_layout(quad_ratio_4_3) == "one_row"
    ASSERT classify_layout(quad_ratio_1_4) == "two_row"
test_polarity:
    ASSERT ensure_dark_on_light(white_text_on_blue) has dark text
    ASSERT ensure_dark_on_light(black_text_on_white) is unchanged
```

### Stage 4 table-driven cases

| Rows read | Layout | Expected | Purpose |
|-----------|--------|----------|---------|
| `["30A-123.45"]` | one row | `30A-123.45` | clean car plate |
| `["59-X1","123.45"]` | two rows | `59X1-123.45` | clean motorbike plate |
| `["3OA-123.45"]` | one row | `30A-123.45` | O→0 in province code |
| `["59-X1","12B.45"]` | two rows | `59X1-128.45` | B→8 in number |
| `["51F","123.45"]` | two rows | `51F-123.45` | square car plate |
| `["30A12345"]` | one row | `30A-123.45` | dash lost, last-5-digits rule |
| `["99-X1","123.45"]` | two rows | invalid | province not in table |
| `["59-X1","12.3"]` | two rows | invalid | number too short |
| `[""]` | one row | invalid | empty OCR output |

### Ablation study (fill with measured results)

| Run | Configuration | Exact-match acc. | CER |
|-----|---------------|------------------|-----|
| A0 | Axis-aligned crop, no warp, no filtering, no post-processing | | |
| A1 | A0 + OBB and perspective warp | | |
| A2 | A1 + filtering, CLAHE, polarity normalisation | | |
| A3 | A2 + row splitting (D2) | | |
| A4 | A3 + regex grammar + position-aware correction (full) | | |

Expected: accuracy rises A0 → A4; a step that does not help signals a bug or badly tuned parameter. Also report latency per stage (ms) and an error analysis grouped by condition and by causing stage.

### Traceability

| Decision | Verified by |
|----------|-------------|
| D1 OBB gives tight corners | Stage 1 overlays; ablation A0→A1 |
| D2 Layout + row-wise reading | Layout unit tests; ablation A2→A3 |
| D3 Polarity normalisation | Polarity tests; CER blue vs. white subset |
| D4 Alphabet filter + padding | Stage 3 confusion list |
| D5 Grammar rejects impossible plates | Stage 4 invalid-case tests |
| D6 Position-aware correction | Stage 4 correction tests; ablation A3→A4 |

---

## 11. Implementation notes & pitfalls

- `cv2.threshold(..., THRESH_OTSU)` returns `(t, binary)`; unpack both.
- PaddleOCR 2.7.3 recognition-only call returns `[[(text, conf)]]`; verify the exact shape on your install and handle empty results (return `("", 0.0)`).
- Feed PaddleOCR a 3-channel image (`cv2.cvtColor(gray, COLOR_GRAY2BGR)`) to avoid shape errors.
- `postprocess` must guard against short strings (index errors on `head[2]`) and return `valid=False` with a `reason` instead of raising.
- Keep `order_corners` assumption in mind: tilt < ~45°.
- Ultralytics OBB: `result.obb.xyxyxyxy` is a tensor — convert with `.cpu().numpy()` before building `Detection`.
- Fall back from `mps` to `cpu` if an operator is unsupported.
- Never hard-code province codes, series letters, or number length.

---

## 12. Future work (after baseline is measured)

| Extension | Builds on | Idea |
|-----------|-----------|------|
| Fine-tune recogniser | Stage 3 | Train PaddleOCR on rectified VN crops with a plate-only dictionary |
| Two-class detector | Stage 1 | `plate_1row` / `plate_2row` classes; compare with aspect-ratio rule |
| Plate colour / vehicle type | Stage 2 | HSV classification: white / blue / yellow / red |
| Province lookup | Stage 4 | Map 2-digit code → province name (updatable table) |
| Other grammars | Stage 4 | Military (red), diplomatic, special series |
| Video tracking + voting | Pipeline | Ultralytics tracking; most frequent / highest-confidence reading across frames |
| Low-light / low-res | Stage 2 | Super-resolution / learned enhancement |
| Speed-up / export | Stage 1 | ONNX / CoreML export; benchmark CPU vs. Apple GPU |
| Demo app | Pipeline | Streamlit or FastAPI front end |
| Privacy & ethics | Whole | Blur plates in stored data, respect data licences and local privacy law |
