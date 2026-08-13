from __future__ import annotations

import pytest

from gait_stability.pose_contracts import normalized_to_pixel


@pytest.mark.parametrize(
    ("value", "extent", "expected"),
    [
        (0.0, 101, 0),
        (0.5, 101, 50),
        (1.0, 101, 100),
        (-0.1, 101, -10),
        (1.1, 101, 110),
    ],
)
def test_normalized_to_pixel_preserves_boundaries_and_out_of_range_estimates(
    value: float, extent: int, expected: int
) -> None:
    assert normalized_to_pixel(value, extent) == expected
