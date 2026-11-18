import numpy as np
import pytest

from vnplate.utils.geometry import polygon_area, order_corners, quad_size
import itertools


def test_rectangle_area():
    assert polygon_area([(0, 0), (10, 0), (10, 4), (0, 4)]) == pytest.approx(40.0)


def test_orientation_invariant():
    pts = np.array([(0, 0), (10, 0), (10, 4), (0, 4)], dtype=np.float32)
    assert polygon_area(pts) == pytest.approx(polygon_area(pts[::-1]))


def test_tilted_square():
    # diamond with diagonals 2 and 2 -> area 2
    assert polygon_area([(1, 0), (2, 1), (1, 2), (0, 1)]) == pytest.approx(2.0)


def test_triangle():
    assert polygon_area([(0, 0), (4, 0), (0, 3)]) == pytest.approx(6.0)


def test_degenerate_collinear_is_zero():
    assert polygon_area([(0, 0), (1, 1), (2, 2), (3, 3)]) == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [[(0, 0), (1, 1)], np.zeros((4, 3)), np.zeros(8)])
def test_bad_shape_raises(bad):
    with pytest.raises(ValueError):
        polygon_area(bad)

RECT = [(10, 10), (110, 10), (110, 40), (10, 40)]       # TL, TR, BR, BL


def test_order_corners_any_permutation():
    expected = np.array(RECT, dtype=np.float32)
    for perm in itertools.permutations(RECT):
        out = order_corners(np.array(perm))
        assert np.array_equal(out, expected)


def test_order_corners_output_dtype_and_shape():
    out = order_corners(RECT)
    assert out.shape == (4, 2) and out.dtype == np.float32


def test_order_corners_tilted_plate():
    # rectangle rotated about 20 degrees, image coordinates (y down)
    tl, tr, br, bl = (30, 20), (128, 55), (118, 82), (20, 47)
    for perm in itertools.permutations([tl, tr, br, bl]):
        out = order_corners(np.array(perm, dtype=np.float32))
        assert np.allclose(out, [tl, tr, br, bl])


def test_order_corners_does_not_modify_input():
    pts = np.array(RECT[::-1], dtype=np.float32)
    before = pts.copy()
    order_corners(pts)
    assert np.array_equal(pts, before)


@pytest.mark.parametrize("bad", [np.zeros((3, 2)), np.zeros((4, 3)), np.zeros(8)])
def test_order_corners_bad_shape(bad):
    with pytest.raises(ValueError):
        order_corners(bad)


def test_order_corners_nan_raises():
    pts = np.array(RECT, dtype=np.float64)
    pts[0, 0] = np.nan
    with pytest.raises(ValueError):
        order_corners(pts)


def test_order_corners_degenerate_raises():
    with pytest.raises(ValueError):
        order_corners(np.zeros((4, 2)))


def test_order_corners_ambiguous_45deg_raises():
    # diamond: x+y and y-x extremes land on the same vertices twice
    with pytest.raises(ValueError):
        order_corners(np.array([(1, 0), (2, 1), (1, 2), (0, 1)], dtype=np.float32))


def test_quad_size_rectangle():
    assert quad_size(np.array(RECT, dtype=np.float32)) == pytest.approx((100.0, 30.0))


def test_quad_size_averages_opposite_edges():
    # trapezoid: top width 80, bottom width 120, both heights 30
    pts = np.array([(20, 0), (100, 0), (120, 30), (0, 30)], dtype=np.float32)
    w, h = quad_size(pts)
    assert w == pytest.approx(100.0)
    assert h == pytest.approx(np.hypot(20, 30))


def test_quad_size_ratio_of_one_row_plate():
    w, h = quad_size(np.array([(0, 0), (430, 0), (430, 100), (0, 100)]))
    assert w / h == pytest.approx(4.3)