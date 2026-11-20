"""Tests for Stage 2 (vnplate.preprocess). No model or dataset needed."""
from __future__ import annotations

import itertools

import cv2
import numpy as np
import pytest

from vnplate.config import PreprocessConfig
from vnplate.preprocess import (
    classify_layout, ensure_dark_on_light, enhance_gray, preprocess_plate,
    unsharp_mask, warp_plate,
)
from vnplate.schemas import LAYOUT_ONE_ROW, LAYOUT_TWO_ROW

CFG = PreprocessConfig()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (160, 60, 20)          # BGR, a dark-ish plate blue


# ---------------------------------------------------------------- helpers
def _quad(w: float, h: float):
    """Ordered axis-aligned quad [TL, TR, BR, BL] of size w x h."""
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)


def _texture(layout: str, plate_bgr, text_bgr) -> np.ndarray:
    """Flat plate with thick 'characters' (bars) on the layout's canvas."""
    w, h = CFG.canvas[layout]
    tex = np.full((h, w, 3), plate_bgr, dtype=np.uint8)
    if layout == LAYOUT_ONE_ROW:
        for i in range(6):
            x = 80 + i * 130
            cv2.rectangle(tex, (x, 50), (x + 60, 170), text_bgr, -1)
    else:
        for row_y in (60, 230):
            for i in range(4):
                x = 70 + i * 110
                cv2.rectangle(tex, (x, row_y), (x + 50, row_y + 100),
                              text_bgr, -1)
    return tex


def _place(tex: np.ndarray, quad, scene_hw=(480, 640), bg=(90, 110, 90)):
    """Paste `tex` into a scene so that it fills `quad` (perspective)."""
    h, w = tex.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                   dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, np.asarray(quad, dtype=np.float32))
    sh, sw = scene_hw
    warped = cv2.warpPerspective(tex, m, (sw, sh), flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, (sw, sh)) > 250
    scene = np.full((sh, sw, 3), bg, dtype=np.uint8)
    scene[mask] = warped[mask]
    return scene


def _inner(img: np.ndarray, margin: int = 8) -> np.ndarray:
    return img[margin:-margin, margin:-margin]


# ---------------------------------------------------------------- classify_layout
@pytest.mark.parametrize("w,h,expected", [
    (430, 100, LAYOUT_ONE_ROW),    # real one-row ratio ~4.3
    (140, 100, LAYOUT_TWO_ROW),    # real two-row ratio ~1.4
    (250, 100, LAYOUT_ONE_ROW),    # exactly tau -> one_row (rule is >=)
    (249, 100, LAYOUT_TWO_ROW),
])
def test_classify_layout(w, h, expected):
    assert classify_layout(_quad(w, h), 2.5) == expected


def test_classify_layout_uses_given_tau():
    assert classify_layout(_quad(200, 100), 1.5) == LAYOUT_ONE_ROW
    assert classify_layout(_quad(200, 100), 3.0) == LAYOUT_TWO_ROW


def test_classify_layout_degenerate_raises():
    flat = np.array([[0, 0], [10, 0], [10, 0], [0, 0]], dtype=np.float32)
    with pytest.raises(ValueError):
        classify_layout(flat, 2.5)


# ---------------------------------------------------------------- polarity (D3)
def test_polarity_white_text_on_blue_is_inverted():
    img = _texture(LAYOUT_ONE_ROW, BLUE, WHITE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = ensure_dark_on_light(gray)
    text = out[gray > 200]
    background = out[gray < 120]
    assert text.mean() < 60 and background.mean() > 180


def test_polarity_black_text_on_white_is_unchanged():
    gray = cv2.cvtColor(_texture(LAYOUT_ONE_ROW, WHITE, BLACK),
                        cv2.COLOR_BGR2GRAY)
    assert np.array_equal(ensure_dark_on_light(gray), gray)


def test_polarity_constant_image_unchanged():
    gray = np.full((20, 40), 77, dtype=np.uint8)
    assert np.array_equal(ensure_dark_on_light(gray), gray)


def test_polarity_does_not_modify_input_and_keeps_dtype():
    gray = cv2.cvtColor(_texture(LAYOUT_ONE_ROW, BLUE, WHITE),
                        cv2.COLOR_BGR2GRAY)
    before = gray.copy()
    out = ensure_dark_on_light(gray)
    assert np.array_equal(gray, before)
    assert out.dtype == np.uint8 and out.shape == gray.shape


@pytest.mark.parametrize("bad", [
    np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), np.float32),
    np.zeros((0, 5), np.uint8),
])
def test_polarity_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        ensure_dark_on_light(bad)


# ---------------------------------------------------------------- unsharp
def test_unsharp_flat_image_unchanged():
    flat = np.full((30, 30), 120, dtype=np.uint8)
    assert np.array_equal(unsharp_mask(flat, 1.0, 0.6), flat)


def test_unsharp_zero_amount_is_identity():
    rng = np.random.RandomState(0)
    img = rng.randint(0, 256, (30, 30)).astype(np.uint8)
    assert np.array_equal(unsharp_mask(img, 1.0, 0.0), img)


def test_unsharp_increases_edge_contrast():
    img = np.full((40, 40), 100, dtype=np.uint8)
    img[:, 20:] = 160
    out = unsharp_mask(img, 1.0, 0.6)
    assert out.dtype == np.uint8
    assert out[20, 19] < img[20, 19] and out[20, 20] > img[20, 20]


def test_unsharp_bad_sigma_raises():
    with pytest.raises(ValueError):
        unsharp_mask(np.zeros((5, 5), np.uint8), 0.0, 0.6)


# ---------------------------------------------------------------- warp
def test_identity_warp():
    # smooth gradient, corners = whole image, canvas size = image size
    w, h = CFG.canvas[LAYOUT_ONE_ROW]
    grad = np.tile(np.linspace(0, 255, w, dtype=np.float32), (h, 1))
    flat = cv2.merge([grad, grad / 2, 255 - grad]).astype(np.uint8)
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    out = preprocess_plate(flat, corners, CFG)
    assert out.layout == LAYOUT_ONE_ROW
    assert np.abs(out.rect.astype(int) - flat.astype(int)).mean() < 0.5


def test_warp_plate_output_size():
    img = np.zeros((100, 200, 3), np.uint8)
    out = warp_plate(img, _quad(150, 50), (300, 100))
    assert out.shape == (100, 300, 3)


@pytest.mark.parametrize("layout,quad", [
    (LAYOUT_ONE_ROW, [[120, 150], [560, 120], [570, 250], [130, 280]]),  # ~4.2:1, tilted
    (LAYOUT_TWO_ROW, [[200, 100], [420, 130], [410, 300], [190, 270]]),  # ~1.3:1, tilted
])
def test_rectification_recovers_flat_plate(layout, quad):
    tex = _texture(layout, WHITE, BLACK)
    scene = _place(tex, quad)
    out = preprocess_plate(scene, np.array(quad, np.float32), CFG)
    assert out.layout == layout
    rect_gray = cv2.cvtColor(out.rect, cv2.COLOR_BGR2GRAY)
    tex_gray = cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY)
    diff = np.abs(_inner(rect_gray).astype(int) - _inner(tex_gray).astype(int))
    assert diff.mean() < 25


def test_corner_order_does_not_matter():
    quad = np.array([[120, 150], [560, 120], [570, 250], [130, 280]],
                    dtype=np.float32)
    scene = _place(_texture(LAYOUT_ONE_ROW, WHITE, BLACK), quad)
    ref = preprocess_plate(scene, quad, CFG)
    for perm in itertools.permutations(range(4)):
        out = preprocess_plate(scene, quad[list(perm)], CFG)
        assert out.layout == ref.layout
        assert np.array_equal(out.rect, ref.rect)
        assert np.array_equal(out.gray, ref.gray)


# ---------------------------------------------------------------- preprocess_plate
@pytest.mark.parametrize("layout,quad", [
    (LAYOUT_ONE_ROW, [[100, 100], [540, 100], [540, 200], [100, 200]]),
    (LAYOUT_TWO_ROW, [[100, 100], [320, 100], [320, 270], [100, 270]]),
])
def test_output_contract(layout, quad):
    scene = _place(_texture(layout, WHITE, BLACK), quad)
    out = preprocess_plate(scene, np.array(quad, np.float32), CFG)
    w, h = CFG.canvas[layout]
    assert out.layout == layout
    assert out.rect.shape == (h, w, 3) and out.rect.dtype == np.uint8
    assert out.gray.shape == (h, w) and out.gray.dtype == np.uint8


def test_canvas_comes_from_config():
    cfg = PreprocessConfig(canvas={"one_row": (470, 110), "two_row": (280, 200)})
    quad = [[100, 100], [540, 100], [540, 200], [100, 200]]
    scene = _place(_texture(LAYOUT_ONE_ROW, WHITE, BLACK), quad)
    out = preprocess_plate(scene, np.array(quad, np.float32), cfg)
    assert out.rect.shape == (110, 470, 3)


def test_tau_comes_from_config():
    quad = [[100, 100], [540, 100], [540, 200], [100, 200]]   # ratio 4.4
    scene = _place(_texture(LAYOUT_ONE_ROW, WHITE, BLACK), quad)
    cfg = PreprocessConfig(layout_ratio_threshold=5.0)
    out = preprocess_plate(scene, np.array(quad, np.float32), cfg)
    assert out.layout == LAYOUT_TWO_ROW


@pytest.mark.parametrize("plate,text", [(WHITE, BLACK), (BLUE, WHITE),
                                        ((0, 215, 255), BLACK)])  # white/blue/yellow
def test_gray_is_dark_text_on_light_for_every_plate_colour(plate, text):
    quad = [[100, 100], [540, 110], [545, 210], [105, 200]]
    scene = _place(_texture(LAYOUT_ONE_ROW, plate, text), quad)
    out = preprocess_plate(scene, np.array(quad, np.float32), CFG)
    inner = _inner(out.gray, 12)
    assert np.median(inner) > 128              # background is light
    assert (inner < 100).mean() > 0.05         # characters are dark
    assert (inner < 100).mean() < 0.5          # ...and are the minority


def test_input_image_is_not_modified():
    quad = [[100, 100], [540, 100], [540, 200], [100, 200]]
    scene = _place(_texture(LAYOUT_ONE_ROW, BLUE, WHITE), quad)
    before = scene.copy()
    preprocess_plate(scene, np.array(quad, np.float32), CFG)
    assert np.array_equal(scene, before)


def test_gray_and_bgra_inputs_are_accepted():
    quad = [[100, 100], [540, 100], [540, 200], [100, 200]]
    scene = _place(_texture(LAYOUT_ONE_ROW, WHITE, BLACK), quad)
    corners = np.array(quad, np.float32)
    gray_in = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    bgra_in = cv2.cvtColor(scene, cv2.COLOR_BGR2BGRA)
    assert preprocess_plate(gray_in, corners, CFG).rect.shape[2] == 3
    assert np.array_equal(preprocess_plate(bgra_in, corners, CFG).rect,
                          preprocess_plate(scene, corners, CFG).rect)


@pytest.mark.parametrize("bad_image", [
    None, np.zeros((0, 10, 3), np.uint8), np.zeros((10, 10, 3), np.float32),
    np.zeros((10, 10, 2), np.uint8), np.zeros((2, 10, 10, 3), np.uint8),
])
def test_invalid_image_raises(bad_image):
    with pytest.raises(ValueError):
        preprocess_plate(bad_image, _quad(50, 20) + 10, CFG)


def test_invalid_corners_raise():
    img = np.zeros((100, 100, 3), np.uint8)
    with pytest.raises(ValueError):
        preprocess_plate(img, np.zeros((3, 2)), CFG)                 # wrong shape
    bad = _quad(50, 20) + 10
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        preprocess_plate(img, bad, CFG)                              # NaN
    collinear = np.array([[0, 0], [10, 10], [20, 20], [30, 30]], np.float32)
    with pytest.raises(ValueError):
        preprocess_plate(img, collinear, CFG)                        # degenerate


def test_enhance_gray_shape_dtype():
    rect = _texture(LAYOUT_TWO_ROW, BLUE, WHITE)
    gray = enhance_gray(rect, CFG)
    assert gray.shape == rect.shape[:2] and gray.dtype == np.uint8