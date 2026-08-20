"""Unit tests for COM qualification module (Step 5b).

Tests use synthetic data only — no real participant videos, network access,
or large downloaded models.

Coverage:
  - ComQualificationConfig default/valid/custom and all rejection modes
  - theoretical_supported_mass_fraction (male/female/invalid sex)
  - supported_mass_coverage (zero/full/tiny excess, material excess,
    negative, nonfinite, bad theoretical),
  - longest_contiguous_run and longest_contiguous_run_duration
    (empty/single/multiple/nonuniform timestamps)
  - frame_eligible_at_threshold (equality/zero/no COM)
  - Minimum synthetic FrameComResult helper
"""

from __future__ import annotations

import pytest

from gait_stability.com_estimation import FrameComResult, Point2D
from gait_stability.com_qualification import (
    ComQualificationConfig,
    frame_eligible_at_threshold,
    longest_contiguous_run,
    longest_contiguous_run_duration,
    supported_mass_coverage,
    theoretical_supported_mass_fraction,
)

# ---------------------------------------------------------------------------
# Synthetic FrameComResult helper (minimum)
# ---------------------------------------------------------------------------


# Factory that creates FrameComResult instances with correct invariants.
# Invariant: if mass_coverage == 0.0 then usable must be False;
# if com is None then usable must be False.  The factory enforces this
# so callers get a valid object without having to remember the consistency
# rules every time.
def _frame_com_result_factory(**kw):
    """Create a valid FrameComResult, auto-setting usable for invariants."""
    frame_index = kw.get("frame_index", 0)
    timestamp_seconds = kw.get("timestamp_seconds", 0.0)
    frame_status = kw.get("frame_status", "decoded_pose")
    com = kw.get("com", Point2D(0.5, 0.5))
    mass_coverage = kw.get("mass_coverage", 1.0)
    segment_results = kw.get("segment_results", ())
    model_total_mass = kw.get("model_total_mass", 1.0)

    # Respect FrameComResult invariants:
    # - if mass_coverage is 0, com must be None
    # - if usable is True, com must not be None
    usable = True if (mass_coverage != 0.0 and com is not None) else False

    return FrameComResult(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        frame_status=frame_status,
        com=com,
        mass_coverage=mass_coverage,
        usable=usable,
        segment_results=segment_results,
        model_total_mass=model_total_mass,
    )


def _make_frame(**kw):
    """Synthetic FrameComResult with sane defaults and invariant-safe usability."""
    return _frame_com_result_factory(**kw)


def _make_frame(**kw):
    """Synthetic FrameComResult with sane defaults and invariant-safe usability.

    The factory automatically sets ``usable`` to ``False`` when
    ``mass_coverage`` is zero or ``com`` is ``None``, because
    ``FrameComResult````s __post_init__ rejects those combinations
    with ``usable=True``.
    """
    # Extract values with defaults
    mc = kw.get("mass_coverage", 1.0)
    com = kw.get("com", Point2D(0.5, 0.5))

    # Determine usable: false if coverage is zero or com is None,
    # otherwise true (matching FrameComResult invariants).
    usable = True if (mc != 0.0 and com is not None) else False

    return FrameComResult(
        frame_index=kw.get("frame_index", 0),
        timestamp_seconds=kw.get("timestamp_seconds", 0.0),
        frame_status=kw.get("frame_status", "decoded_pose"),
        com=com,
        mass_coverage=mc,
        usable=usable,
        segment_results=kw.get("segment_results", ()),
        model_total_mass=kw.get("model_total_mass", 1.0),
    )


# ===========================================================================
# ComQualificationConfig tests
# ===========================================================================


class TestComQualificationConfigDefault:
    """Default configuration is valid."""

    def test_default_is_valid(self) -> None:
        cfg = ComQualificationConfig()
        assert cfg.coverage_thresholds == (0.80, 0.82, 0.84, 0.86, 0.88, 0.90)


class TestComQualificationConfigCustom:
    """Custom valid configurations."""

    def test_custom_valid_tuple(self) -> None:
        cfg = ComQualificationConfig(coverage_thresholds=(0.5, 0.75))
        assert cfg.coverage_thresholds == (0.5, 0.75)

    def test_custom_single_threshold(self) -> None:
        cfg = ComQualificationConfig(coverage_thresholds=(0.9,))
        assert cfg.coverage_thresholds == (0.9,)


class TestComQualificationConfigRejections:
    """All configuration rejection modes."""

    def test_rejects_empty_tuple(self) -> None:
        with pytest.raises(ValueError, match="coverage_thresholds must not be empty"):
            ComQualificationConfig(coverage_thresholds=())

    def test_rejects_non_tuple(self) -> None:
        with pytest.raises(TypeError, match="coverage_thresholds must be a tuple"):
            ComQualificationConfig(coverage_thresholds=[0.5, 0.6])

    def test_rejects_duplicate_values(self) -> None:
        with pytest.raises(ValueError, match="duplicate value"):
            ComQualificationConfig(coverage_thresholds=(0.5, 0.5, 0.6))

    def test_rejects_non_increasing(self) -> None:
        with pytest.raises(ValueError, match="must be strictly increasing"):
            ComQualificationConfig(coverage_thresholds=(0.9, 0.5))

    def test_rejects_nonfinite(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            ComQualificationConfig(coverage_thresholds=(float("nan"), 0.5))

    def test_rejects_out_of_range_upper(self) -> None:
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            ComQualificationConfig(coverage_thresholds=(1.5,))

    def test_rejects_out_of_range_lower(self) -> None:
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            ComQualificationConfig(coverage_thresholds=(-0.1,))

    def test_rejects_bool_values(self) -> None:
        with pytest.raises(TypeError, match="must be a number"):
            ComQualificationConfig(coverage_thresholds=(True, False))

    def test_bool_is_rejected_even_if_0_or_1(self) -> None:
        # Explicitly confirm bool subclass of int is rejected
        with pytest.raises(TypeError):
            ComQualificationConfig(coverage_thresholds=(True,))

    def test_accepts_zero_and_one(self) -> None:
        # 0 and 1 are valid boundaries
        cfg = ComQualificationConfig(coverage_thresholds=(0.0, 1.0))
        assert cfg.coverage_thresholds == (0.0, 1.0)

    def test_accepts_boundary_0(self) -> None:
        cfg = ComQualificationConfig(coverage_thresholds=(0,))
        assert cfg.coverage_thresholds == (0,)

    def test_accepts_boundary_1(self) -> None:
        cfg = ComQualificationConfig(coverage_thresholds=(1,))
        assert cfg.coverage_thresholds == (1,)


# ===========================================================================
# theoretical_supported_mass_fraction tests
# ===========================================================================


class TestTheoreticalSupportedMassFraction:
    """Theoretical supported mass fraction by sex."""

    def test_male_returns_0_9306(self) -> None:
        result = theoretical_supported_mass_fraction("male")
        assert result == 0.9306

    def test_female_returns_0_9331(self) -> None:
        result = theoretical_supported_mass_fraction("female")
        assert result == 0.9331

    def test_invalid_sex_raises(self) -> None:
        with pytest.raises(ValueError, match="sex must be"):
            theoretical_supported_mass_fraction("unknown")


# ===========================================================================
# supported_mass_coverage tests
# ===========================================================================


class TestSupportedMassCoverage:
    """supported_mass_coverage ratio computation."""

    def test_zero_mass_coverage(self) -> None:
        result = supported_mass_coverage(0.0, 0.9306)
        assert result == 0.0

    def test_full_mass_coverage(self) -> None:
        result = supported_mass_coverage(0.9306, 0.9306)
        assert result == 1.0

    def test_tiny_excess_clamped_to_one(self) -> None:
        """slightly above theoretical should clamp to 1.0 within tolerance."""
        # Use excess well within the 1e-12 tolerance so ratio - 1 <= _FLOAT_TOLERANCE
        result = supported_mass_coverage(0.9306 + 1e-13, 0.9306)
        assert result == 1.0

    def test_material_excess_raises(self) -> None:
        """substantially above theoretical should raise."""
        with pytest.raises(ValueError, match="exceeds 1.0 materially"):
            supported_mass_coverage(1.0, 0.9306)

    def test_negative_mass_coverage_raises(self) -> None:
        with pytest.raises(ValueError, match="below 0"):
            supported_mass_coverage(-0.1, 0.9306)

    def test_nonfinite_mass_coverage_raises(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            supported_mass_coverage(float("nan"), 0.9306)

    def test_nonfinite_theoretical_raises(self) -> None:
        # Function checks theoretical finiteness before positive-check,
        # so the matching error message is "must be finite".
        with pytest.raises(ValueError, match="must be finite"):
            supported_mass_coverage(0.5, float("nan"))

    def test_theoretical_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            supported_mass_coverage(0.5, 0.0)

    def test_theoretical_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            supported_mass_coverage(0.5, -0.5)


# ===========================================================================
# longest_contiguous_run tests
# ===========================================================================


class TestLongestContiguousRun:
    """Longest contiguous run of target values."""

    def test_empty_list(self) -> None:
        assert longest_contiguous_run([]) == 0

    def test_all_true(self) -> None:
        assert longest_contiguous_run([True, True, True]) == 3

    def test_all_false(self) -> None:
        assert longest_contiguous_run([False, False, False]) == 0

    def test_single_true(self) -> None:
        assert longest_contiguous_run([False, True, False]) == 1

    def test_single_false_run(self) -> None:
        # longest run of True in [True, False, True] is 1 (two isolated True values)
        assert longest_contiguous_run([True, False, True]) == 1

    def test_multi_run_max_wins(self) -> None:
        # Two runs of 2 and 3, max should be 3
        assert longest_contiguous_run([True, True, False, True, True, True]) == 3

    def test_target_false(self) -> None:
        assert longest_contiguous_run([False, False, True, True], target=False) == 2


# ===========================================================================
# longest_contiguous_run_duration tests
# ===========================================================================


class TestLongestContiguousRunDuration:
    """Longest contiguous run with duration in seconds."""

    def test_empty(self) -> None:
        assert longest_contiguous_run_duration([], []) == (0, 0.0)

    def test_single_frame(self) -> None:
        # Single frame, duration should be 0.0
        assert longest_contiguous_run_duration([True], [1.0]) == (1, 0.0)

    def test_single_false(self) -> None:
        assert longest_contiguous_run_duration([False], [1.0]) == (0, 0.0)

    def test_two_consecutive_true(self) -> None:
        # Timestamps 0.0 and 0.1 -> duration 0.1
        result = longest_contiguous_run_duration([True, True], [0.0, 0.1])
        assert result == (2, 0.1)

    def test_nonuniform_timestamps(self) -> None:
        # Irregular spacing should still compute correct duration
        result = longest_contiguous_run_duration([True, True, True], [0.0, 0.5, 1.0])
        assert result == (3, 1.0)

    def test_run_at_end(self) -> None:
        # Run at the end of the sequence
        result = longest_contiguous_run_duration([False, True, True], [0.0, 0.5, 1.0])
        assert result == (2, 0.5)  # timestamps[2] - timestamps[1] = 1.0 - 0.5


# ===========================================================================
# frame_eligible_at_threshold tests
# ===========================================================================


class TestFrameEligibleAtThreshold:
    """Frame eligibility at a given mass_coverage threshold."""

    def test_equality_passes(self) -> None:
        """mass_coverage == threshold should be eligible."""
        frame = _make_frame(mass_coverage=0.80, com=Point2D(0.5, 0.5))
        assert frame_eligible_at_threshold(frame, 0.80) is True

    def test_below_threshold_fails(self) -> None:
        """mass_coverage < threshold should not be eligible."""
        frame = _make_frame(mass_coverage=0.79, com=Point2D(0.5, 0.5))
        assert frame_eligible_at_threshold(frame, 0.80) is False

    def test_zero_mass_coverage_fails(self) -> None:
        """mass_coverage == 0.0 should not be eligible."""
        # FrameComResult enforces that mass_coverage==0 => com is None,
        # so we test the zero-coverage path via com is None.
        frame = _make_frame(mass_coverage=0.0, com=None)
        assert frame_eligible_at_threshold(frame, 0.80) is False

    def test_no_com_fails(self) -> None:
        """com is None should not be eligible regardless of mass_coverage."""
        frame = _make_frame(mass_coverage=1.0, com=None)
        assert frame_eligible_at_threshold(frame, 0.80) is False

    def test_above_threshold_passes(self) -> None:
        """mass_coverage > threshold should be eligible."""
        frame = _make_frame(mass_coverage=0.85, com=Point2D(0.5, 0.5))
        assert frame_eligible_at_threshold(frame, 0.80) is True

    def test_zero_mass_no_com_fails(self) -> None:
        """mass_coverage == 0 and com is None should be ineligible."""
        frame = _make_frame(mass_coverage=0.0, com=None)
        assert frame_eligible_at_threshold(frame, 0.01) is False


# ===========================================================================
# Integration-style boundary tests
# ===========================================================================


class TestIntegrationBoundaries:
    """Boundary-spanning integration-style tests."""

    def test_config_theoretical_consistency(self) -> None:
        """Config thresholds should be compatible with theoretical values."""
        cfg = ComQualificationConfig()
        for th in cfg.coverage_thresholds:
            assert 0.0 <= th <= 1.0

    def test_supported_coverage_with_theoretical_male(self) -> None:
        """supported_mass_coverage uses male theoretical by default expectation."""
        result = supported_mass_coverage(0.4653, 0.9306)  # half of male theoretical
        assert result == 0.5

    def test_frame_eligible_with_zero_mass_and_com(self) -> None:
        """Frame with mass_coverage=0 is ineligible (com is None)."""
        from gait_stability.com_qualification import frame_eligible_at_threshold

        frame = _make_frame(mass_coverage=0.0, com=None)
        assert frame_eligible_at_threshold(frame, 0.5) is False

    def test_longest_run_duration_nonuniform_respects_timestamps(self) -> None:
        """Duration should respect actual timestamp spacing, not just count."""
        # Three frames at 0.0, 0.2, 0.5 -> run of 3 spans 0.5s
        result = longest_contiguous_run_duration([True, True, True], [0.0, 0.2, 0.5])
        assert result == (3, 0.5)

    def test_custom_config_all_rejection_cases(self) -> None:
        """Exhaustively verify all ComQualificationConfig rejection modes."""
        # non-tuple
        with pytest.raises(TypeError):
            ComQualificationConfig(coverage_thresholds="0.5,0.6")  # type: ignore[arg-type]
        # empty
        with pytest.raises(ValueError):
            ComQualificationConfig(coverage_thresholds=())
        # duplicate
        with pytest.raises(ValueError):
            ComQualificationConfig(coverage_thresholds=(0.5, 0.5))
        # non-increasing
        with pytest.raises(ValueError):
            ComQualificationConfig(coverage_thresholds=(0.6, 0.5))
        # nonfinite nan
        with pytest.raises(ValueError):
            ComQualificationConfig(coverage_thresholds=(float("nan"),))
        # nonfinite inf
        with pytest.raises(ValueError):
            ComQualificationConfig(coverage_thresholds=(float("inf"),))
        # out of range > 1
        with pytest.raises(ValueError):
            ComQualificationConfig(coverage_thresholds=(2.0,))
        # out of range < 0
        with pytest.raises(ValueError):
            ComQualificationConfig(coverage_thresholds=(-1.0,))
        # bool
        with pytest.raises(TypeError):
            ComQualificationConfig(coverage_thresholds=(True,))
