import numpy as np
import pytest

from vnplate.utils.geometry import polygon_area


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