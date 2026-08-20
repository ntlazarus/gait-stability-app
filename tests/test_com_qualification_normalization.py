"""Focused deterministic tests for recompute_normalized_availability.

Uses synthetic data only — no real participant videos, network access, or
large downloaded models.

Coverage:
  - exact eligible: canonical ts matches eligible frame -> exact_match_usable
  - equality threshold: mass_coverage == threshold passes eligibility
  - exact ineligible: canonical ts matches ineligible frame -> unavailable
  - adjacent consecutive eligible identical segment tuple permits linear
  - changed tuple blocks linear interpolation
  - one endpoint below threshold blocks endpoint availability
  - nonconsecutive frame indices block (and report if production fails)
  - start/end availability from stride_window bounds
  - zero COM / mass_coverage==0 makes frame ineligible
  - results change across thresholds independently of FrameComResult.usable
"""

from __future__ import annotations

import pytest

from gait_stability.com_estimation import (
    FrameComResult,
    Point2D,
    SegmentComResult,
    SegmentProvenance,
)
from gait_stability.com_qualification import (
    ReviewedStrideWindow,
    recompute_normalized_availability,
)

# ---------------------------------------------------------------------------
# Minimal SegmentProvenance instance (shared across all test segments)
# ---------------------------------------------------------------------------

_minimal_provenance = SegmentProvenance(
    segment_name="upper_arm",
    proximal_landmark="left_shoulder",
    distal_landmark="left_elbow",
    contributors=("left_shoulder", "left_elbow"),
    usable=True,
    mass_fraction=0.02,
    r=0.1,
    all_raw_observed=True,
    any_x_interpolated=False,
    any_y_interpolated=False,
    any_x_smoothing_changed=False,
    any_y_smoothing_changed=False,
    any_x_smoothing_support_interpolation=False,
    any_y_smoothing_support_interpolation=False,
    other_qc_limited=False,
)


# ---------------------------------------------------------------------------
# Minimal SegmentComResult instance (shared across all test segments)
# ---------------------------------------------------------------------------

_minimal_segment_result = SegmentComResult(
    segment_name="upper_arm",
    com=Point2D(0.0, 0.0),
    mass_fraction=0.02,
    usable=True,
    provenance=_minimal_provenance,
)


# ---------------------------------------------------------------------------
# FrameComResult builder with proper SegmentComResult instances and explicit com
# ---------------------------------------------------------------------------


def _make_frame(
    *,
    frame_index: int = 0,
    timestamp_seconds: float = 0.0,
    frame_status: str = "decoded_pose",
    com: Point2D | None = None,
    mass_coverage: float = 1.0,
    segment_names: tuple[str, ...] = ("upper_arm",),
    model_total_mass: float = 1.0,
) -> FrameComResult:
    """Create a valid FrameComResult with proper SegmentComResult instances.

    Note: callers requiring an eligible frame must pass com=Point2D(0.5, 0.5)
    (or another non-None com) so that frame_eligible_at_threshold returns True.
    """

    seg_results = tuple(
        SegmentComResult(
            segment_name=name,
            com=Point2D(0.0, 0.0),
            mass_fraction=0.02,
            usable=True,
            provenance=_minimal_provenance,
        )
        for name in segment_names
    )

    # Respect FrameComResult invariants
    usable = True if (mass_coverage != 0.0 and com is not None) else False

    return FrameComResult(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        frame_status=frame_status,
        com=com,
        mass_coverage=mass_coverage,
        usable=usable,
        segment_results=seg_results,
        model_total_mass=model_total_mass,
    )


# ---------------------------------------------------------------------------
# Minimal ReviewedStrideWindow builder
# ---------------------------------------------------------------------------


def _make_stride_window(
    *,
    stride_id: str = "stride_01",
    start_frame: int = 0,
    end_frame: int = 9,
    start_timestamp: float = 0.0,
    end_timestamp: float = 1.0,
) -> ReviewedStrideWindow:
    return ReviewedStrideWindow(
        stride_id=stride_id,
        side="left",
        start_frame=start_frame,
        end_frame=end_frame,
        start_timestamp_seconds=start_timestamp,
        end_timestamp_seconds=end_timestamp,
        duration_seconds=end_timestamp - start_timestamp,
        automatic_stride_id=stride_id,
        review_intent="",
        review_changes="",
        provenance_notes="",
    )


# ---------------------------------------------------------------------------
# Canonical timestamps helper
# ---------------------------------------------------------------------------


def _canonical_timestamps(
    start: float = 0.0,
    step: float = 0.2,
    count: int = 5,
) -> list[float]:
    return [start + i * step for i in range(count)]


# ===========================================================================
# Tests
# ===========================================================================


class TestRecomputeNormalizedAvailabilityExactEligible:
    """Canonical timestamp exactly matches an eligible frame."""

    def test_exact_match_eligible_frame(self) -> None:
        """Exact timestamp match with eligible frame -> exact_match_usable."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.2,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
                segment_names=("upper_arm",),
            )
        ]
        window = _make_stride_window(start_frame=0, end_frame=0)
        result = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=window,
            canonical_timestamps=[0.2],
            threshold=0.80,
        )
        assert result.exact_match_usable == 1
        assert result.linear_interpolation_usable == 0
        assert result.unavailable == 0
        assert result.start_endpoint_available is True
        assert result.end_endpoint_available is True


class TestRecomputeNormalizedAvailabilityEqualityThreshold:
    """mass_coverage == threshold should be eligible."""

    def test_equality_passes_eligibility(self) -> None:
        """mass_coverage == threshold -> frame eligible."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.2,
                mass_coverage=0.80,
                com=Point2D(0.5, 0.5),
            )
        ]
        window = _make_stride_window(start_frame=0, end_frame=0)
        result = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=window,
            canonical_timestamps=[0.2],
            threshold=0.80,
        )
        # frame_eligible_at_threshold uses >= so equality passes
        assert result.start_endpoint_available is True


class TestRecomputeNormalizedAvailabilityExactIneligible:
    """Canonical timestamp exactly matches an ineligible frame."""

    def test_exact_match_ineligible_frame(self) -> None:
        """Exact timestamp match with ineligible frame -> unavailable."""
        # Ineligible because mass_coverage=0.5 < threshold 0.80
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.2,
                mass_coverage=0.5,
                com=Point2D(0.5, 0.5),
            )
        ]
        window = _make_stride_window(start_frame=0, end_frame=0)
        result = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=window,
            canonical_timestamps=[0.2],
            threshold=0.80,
        )
        # Frame is ineligible at 0.80 threshold (0.5 < 0.80)
        assert result.exact_match_usable == 0
        assert result.unavailable == 1
        assert result.start_endpoint_available is False
        assert result.end_endpoint_available is False


class TestRecomputeNormalizedAvailabilityAdjacentConsecutiveEligibleIdentical:
    """Eligible frames with identical segments allow linear interpolation."""

    def test_identical_segment_tuples_permit_linear_interpolation(self) -> None:
        """Two eligible frames with identical usable_segments() allow linear."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
                segment_names=("upper_arm",),
            ),
            _make_frame(
                frame_index=1,
                timestamp_seconds=1.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
                segment_names=("upper_arm",),
            ),
        ]
        window = _make_stride_window(start_frame=0, end_frame=1)
        result = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=window,
            canonical_timestamps=[0.5],  # bracketed by 0.0 and 1.0
            threshold=0.80,
        )
        # Both frames eligible and have identical segment tuples -> linear
        assert result.linear_interpolation_usable == 1
        assert result.exact_match_usable == 0
        assert result.unavailable == 0


class TestRecomputeNormalizedAvailabilityChangedTupleBlocks:
    """Changed segment tuple between adjacent eligible blocks linear."""

    def test_different_segment_tuples_block_linear(self) -> None:
        """Adjacent eligible frames with different segment tuples -> unavailable."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
                segment_names=("upper_arm",),
            ),
            _make_frame(
                frame_index=1,
                timestamp_seconds=1.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
                segment_names=("forearm",),
            ),
        ]
        window = _make_stride_window(start_frame=0, end_frame=1)
        result = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=window,
            canonical_timestamps=[0.5],
            threshold=0.80,
        )
        # Different segment tuples should block linear interpolation
        assert result.linear_interpolation_usable == 0
        assert result.unavailable == 1


class TestRecomputeNormalizedAvailabilityOneEndpointBelowThreshold:
    """One endpoint below threshold blocks endpoint availability."""

    def test_start_frame_ineligible_blocks_start_endpoint(self) -> None:
        """If start_frame is ineligible, start_endpoint_available should be False."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.0,
                mass_coverage=0.5,
                com=Point2D(0.5, 0.5),
            ),
            _make_frame(
                frame_index=1,
                timestamp_seconds=1.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
        ]
        window = _make_stride_window(start_frame=0, end_frame=1)
        result = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=window,
            canonical_timestamps=[0.5],
            threshold=0.80,
        )
        assert result.start_endpoint_available is False
        # end frame is eligible
        assert result.end_endpoint_available is True

    def test_end_frame_ineligible_blocks_end_endpoint(self) -> None:
        """If end_frame is ineligible, end_endpoint_available should be False."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
            _make_frame(
                frame_index=1,
                timestamp_seconds=1.0,
                mass_coverage=0.5,
                com=Point2D(0.5, 0.5),
            ),
        ]
        window = _make_stride_window(start_frame=0, end_frame=1)
        result = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=window,
            canonical_timestamps=[0.5],
            threshold=0.80,
        )
        assert result.start_endpoint_available is True
        assert result.end_endpoint_available is False


class TestRecomputeNormalizedAvailabilityNonconsecutiveFrameIndices:
    """Nonconsecutive frame indices bracket behavior."""

    def test_canonical_ts_bracketed_by_consecutive_frames(self) -> None:
        """Canonical timestamps bracketed by consecutive frames work correctly.

        The function checks adjacent frames in the stride_frames list for
        bracketing, not frame indices. Frames at 0.0 and 1.0 bracket 0.3 and 0.7.
        """
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
            _make_frame(
                frame_index=1,
                timestamp_seconds=1.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
        ]
        window = _make_stride_window(start_frame=0, end_frame=1)
        result = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=window,
            canonical_timestamps=[0.3, 0.7],
            threshold=0.80,
        )
        # Both 0.3 and 0.7 are bracketed by (0.0, 1.0) with identical segments
        assert result.linear_interpolation_usable == 2
        assert result.unavailable == 0

    def test_canonical_ts_outside_frame_range_blocks(self) -> None:
        """Canonical timestamps outside the range of stride_frames timestamps
        should be unavailable (no bracket found)."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
            _make_frame(
                frame_index=1,
                timestamp_seconds=1.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
        ]
        window = _make_stride_window(start_frame=0, end_frame=1)
        result = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=window,
            canonical_timestamps=[-0.1, 1.2],  # outside [0.0, 1.0]
            threshold=0.80,
        )
        # -0.1 and 1.2 are outside the timestamp range -> unavailable
        assert result.unavailable == 2
        assert result.linear_interpolation_usable == 0


class TestRecomputeNormalizedAvailabilityStartEndAvailability:
    """start/end availability from stride_window bounds."""

    def test_start_end_available_both_eligible(self) -> None:
        """Both start and end frames eligible -> both endpoints available."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
            _make_frame(
                frame_index=5,
                timestamp_seconds=1.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
        ]
        window = _make_stride_window(start_frame=0, end_frame=5)
        result = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=window,
            canonical_timestamps=[0.5],
            threshold=0.80,
        )
        assert result.start_endpoint_available is True
        assert result.end_endpoint_available is True

    def test_start_ineligible_end_eligible(self) -> None:
        """Start frame ineligible -> start_endpoint_available False."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.0,
                mass_coverage=0.5,
                com=Point2D(0.5, 0.5),
            ),
            _make_frame(
                frame_index=5,
                timestamp_seconds=1.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
        ]
        window = _make_stride_window(start_frame=0, end_frame=5)
        result = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=window,
            canonical_timestamps=[0.5],
            threshold=0.80,
        )
        assert result.start_endpoint_available is False
        assert result.end_endpoint_available is True

    def test_start_eligible_end_ineligible(self) -> None:
        """End frame ineligible -> end_endpoint_available False."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
            _make_frame(
                frame_index=5,
                timestamp_seconds=1.0,
                mass_coverage=0.5,
                com=Point2D(0.5, 0.5),
            ),
        ]
        window = _make_stride_window(start_frame=0, end_frame=5)
        result = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=window,
            canonical_timestamps=[0.5],
            threshold=0.80,
        )
        assert result.start_endpoint_available is True
        assert result.end_endpoint_available is False


class TestRecomputeNormalizedAvailabilityZeroCOM:
    """Zero COM / mass_coverage==0 makes frame ineligible."""

    def test_zero_mass_coverage_frame_ineligible(self) -> None:
        """Frame with mass_coverage==0 is ineligible at any threshold."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.0,
                mass_coverage=0.0,
                com=None,
                segment_names=(),
            ),
        ]
        window = _make_stride_window(start_frame=0, end_frame=0)
        result = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=window,
            canonical_timestamps=[0.0],
            threshold=0.80,
        )
        # mass_coverage==0 => frame_eligible_at_threshold returns False
        assert result.exact_match_usable == 0
        assert result.unavailable == 1
        assert result.start_endpoint_available is False
        assert result.end_endpoint_available is False


class TestRecomputeNormalizedAvailabilityResultsChangeAcrossThresholds:
    """Results change across thresholds independently of FrameComResult.usable."""

    def test_different_thresholds_different_eligibility(self) -> None:
        """Changing threshold changes eligibility even when FrameComResult.usable
        is unchanged (set at primary threshold)."""
        # Frame with mass_coverage=0.95, usable=True at primary threshold 0.80
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.0,
                mass_coverage=0.95,
                com=Point2D(0.5, 0.5),
            ),
        ]

        # At threshold 0.80: eligible
        result_80 = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=_make_stride_window(start_frame=0, end_frame=0),
            canonical_timestamps=[0.0],
            threshold=0.80,
        )
        assert result_80.start_endpoint_available is True
        assert result_80.exact_match_usable == 1

        # At threshold 0.90: still eligible (0.95 >= 0.90)
        result_90 = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=_make_stride_window(start_frame=0, end_frame=0),
            canonical_timestamps=[0.0],
            threshold=0.90,
        )
        assert result_90.start_endpoint_available is True
        assert result_90.exact_match_usable == 1

        # At threshold 0.95: equality passes
        result_95 = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=_make_stride_window(start_frame=0, end_frame=0),
            canonical_timestamps=[0.0],
            threshold=0.95,
        )
        assert result_95.start_endpoint_available is True
        assert result_95.exact_match_usable == 1

        # At threshold 0.96: 0.95 < 0.96 -> ineligible
        result_96 = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=_make_stride_window(start_frame=0, end_frame=0),
            canonical_timestamps=[0.0],
            threshold=0.96,
        )
        assert result_96.start_endpoint_available is False
        assert result_96.exact_match_usable == 0
        assert result_96.unavailable == 1

        # FrameComResult.usable is True (set at primary threshold 0.80),
        # but normalized availability changes across thresholds
        assert frames[0].usable is True
        # Results differ across thresholds
        assert result_80.unavailable == 0
        assert result_96.unavailable == 1

    def test_usable_fraction_changes_across_thresholds(self) -> None:
        """normalized_usable_fraction should change across thresholds even
        though FrameComResult.usable is fixed."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
            _make_frame(
                frame_index=1,
                timestamp_seconds=1.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
        ]

        result_80 = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=_make_stride_window(start_frame=0, end_frame=1),
            canonical_timestamps=[0.5],
            threshold=0.80,
        )
        # Both frames eligible at 0.80
        assert result_80.usable_fraction == 1.0

        result_90 = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=_make_stride_window(start_frame=0, end_frame=1),
            canonical_timestamps=[0.5],
            threshold=0.90,
        )
        # 0.85 < 0.90, both ineligible
        assert result_90.usable_fraction == 0.0

        # FrameComResult.usable is True (set at primary threshold 0.80)
        assert frames[0].usable is True
        assert frames[1].usable is True
        # But normalized availability changed
        assert result_80.usable_fraction != result_90.usable_fraction


class TestRecomputeNormalizedAvailabilityEmptyStrideFrames:
    """Empty stride_frames returns all unavailable."""

    def test_empty_stride_frames(self) -> None:
        """No stride frames -> all canonical timestamps unavailable."""
        frames: list[FrameComResult] = []
        window = _make_stride_window(start_frame=0, end_frame=9)
        result = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=window,
            canonical_timestamps=[0.0, 0.2, 0.4, 0.6, 0.8],
            threshold=0.80,
        )
        assert result.exact_match_usable == 0
        assert result.linear_interpolation_usable == 0
        assert result.unavailable == 5
        assert result.total_normalized == 5
        assert result.start_endpoint_available is False
        assert result.end_endpoint_available is False


class TestRecomputeNormalizedAvailabilityNoGapRule:
    """Enforce no-gap rule: right.frame_index == left.frame_index + 1 for linear."""

    def test_nonconsecutive_frame_indices_block_linear(self) -> None:
        """Frames with nonconsecutive indices (0 and 2) cannot bracket for linear."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
            _make_frame(
                frame_index=2,
                timestamp_seconds=1.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
        ]
        window = _make_stride_window(start_frame=0, end_frame=2)
        # Canonical timestamp 0.5 is bracketed by 0.0 and 1.0, but frame indices
        # are 0 and 2 (gap of 1) -> should be unavailable
        result = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=window,
            canonical_timestamps=[0.5],
            threshold=0.80,
        )
        assert result.linear_interpolation_usable == 0
        assert result.unavailable == 1

    def test_consecutive_frame_indices_allow_linear(self) -> None:
        """Frames with consecutive indices (0 and 1) can bracket for linear."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
            _make_frame(
                frame_index=1,
                timestamp_seconds=1.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
        ]
        window = _make_stride_window(start_frame=0, end_frame=1)
        result = recompute_normalized_availability(
            stride_frames=frames,
            stride_window=window,
            canonical_timestamps=[0.5],
            threshold=0.80,
        )
        assert result.linear_interpolation_usable == 1
        assert result.unavailable == 0


class TestRecomputeNormalizedAvailabilityTimestampValidation:
    """Validate canonical timestamps and stride frame timestamps."""

    def test_nonfinite_canonical_timestamp_raises(self) -> None:
        """Non-finite canonical timestamp should raise ValueError."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            )
        ]
        window = _make_stride_window(start_frame=0, end_frame=0)
        with pytest.raises(ValueError, match="canonical_timestamps must be finite"):
            recompute_normalized_availability(
                stride_frames=frames,
                stride_window=window,
                canonical_timestamps=[0.0, float("nan")],
                threshold=0.80,
            )

    def test_non_increasing_canonical_timestamps_raises(self) -> None:
        """Non-strictly-increasing canonical timestamps should raise ValueError."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=0.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            )
        ]
        window = _make_stride_window(start_frame=0, end_frame=0)
        with pytest.raises(
            ValueError, match="canonical_timestamps must be strictly increasing"
        ):
            recompute_normalized_availability(
                stride_frames=frames,
                stride_window=window,
                canonical_timestamps=[0.5, 0.5],  # duplicate
                threshold=0.80,
            )

    def test_non_increasing_stride_frame_indices_raises(self) -> None:
        """Non-increasing stride frame indices should raise ValueError."""
        frames = [
            _make_frame(
                frame_index=1,
                timestamp_seconds=0.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
            _make_frame(
                frame_index=0,
                timestamp_seconds=1.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
        ]
        window = _make_stride_window(start_frame=0, end_frame=1)
        with pytest.raises(
            ValueError, match="stride frame indices must be strictly increasing"
        ):
            recompute_normalized_availability(
                stride_frames=frames,
                stride_window=window,
                canonical_timestamps=[0.5],
                threshold=0.80,
            )

    def test_non_increasing_stride_frame_timestamps_raises(self) -> None:
        """Non-increasing stride frame timestamps should raise ValueError."""
        frames = [
            _make_frame(
                frame_index=0,
                timestamp_seconds=1.0,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
            _make_frame(
                frame_index=1,
                timestamp_seconds=0.5,
                mass_coverage=0.85,
                com=Point2D(0.5, 0.5),
            ),
        ]
        window = _make_stride_window(start_frame=0, end_frame=1)
        with pytest.raises(
            ValueError, match="stride frame timestamps must be strictly increasing"
        ):
            recompute_normalized_availability(
                stride_frames=frames,
                stride_window=window,
                canonical_timestamps=[0.75],
                threshold=0.80,
            )
