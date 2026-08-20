"""Deterministic pure tests for Step 5b COM qualification summaries."""

from __future__ import annotations

from typing import Literal

import pytest

from gait_stability.com_estimation import (
    SEGMENT_NAMES,
    FrameComResult,
    Point2D,
    SegmentComResult,
    SegmentProvenance,
    StrideComSample,
)
from gait_stability.com_qualification import (
    DEFAULT_COVERAGE_THRESHOLDS,
    SEGMENT_MASS_FRACTIONS,
    ComQualificationConfig,
    ParsedReviewedStride,
    ProcessedLandmarkQCRow,
    ReviewedStrideWindow,
    compute_asymmetry_summaries,
    compute_landmark_coverage,
    compute_qualification,
    compute_segment_coverage,
    compute_stride_coverage,
)


def _segment(
    name: str,
    *,
    usable: bool = True,
    mass: float = 0.1,
    raw: bool = True,
) -> SegmentComResult:
    provenance = SegmentProvenance(
        segment_name=name,
        proximal_landmark=f"{name}_proximal",
        distal_landmark=f"{name}_distal",
        contributors=(f"{name}_proximal", f"{name}_distal"),
        usable=usable,
        mass_fraction=mass,
        r=0.5,
        all_raw_observed=raw,
        any_x_interpolated=False,
        any_y_interpolated=False,
        any_x_smoothing_changed=False,
        any_y_smoothing_changed=False,
        any_x_smoothing_support_interpolation=False,
        any_y_smoothing_support_interpolation=False,
        other_qc_limited=False,
    )
    return SegmentComResult(
        segment_name=name,
        com=Point2D(0.5, 0.5) if usable else None,
        mass_fraction=mass,
        usable=usable,
        provenance=provenance,
    )


def _frame(
    frame_index: int,
    *,
    coverage: float = 0.85,
    segments: tuple[SegmentComResult, ...] = (),
) -> FrameComResult:
    return FrameComResult(
        frame_index=frame_index,
        timestamp_seconds=float(frame_index),
        frame_status="decoded_pose",
        com=Point2D(0.5, 0.5) if coverage else None,
        mass_coverage=coverage,
        usable=coverage > 0.0,
        segment_results=segments,
        model_total_mass=1.0,
    )


def _landmark_row(
    frame_index: int,
    landmark_name: str,
    *,
    final: bool = True,
    raw: bool = True,
    x_interpolated: bool = False,
    y_interpolated: bool = False,
    x_smoothed: bool = False,
    y_smoothed: bool = False,
    x_smoothing_interpolation: bool = False,
    y_smoothing_interpolation: bool = False,
) -> ProcessedLandmarkQCRow:
    return ProcessedLandmarkQCRow(
        frame_index=frame_index,
        landmark_name=landmark_name,
        final_available=final,
        raw_observed_usable=raw,
        x_interpolated=x_interpolated,
        y_interpolated=y_interpolated,
        x_smoothing_changed=x_smoothed,
        y_smoothing_changed=y_smoothed,
        x_smoothing_support_interpolation=x_smoothing_interpolation,
        y_smoothing_support_interpolation=y_smoothing_interpolation,
    )


def _window(
    stride_id: str = "stride-a",
    *,
    side: str = "left",
    start: int = 0,
    end: int = 1,
) -> ReviewedStrideWindow:
    return ReviewedStrideWindow(
        stride_id=stride_id,
        side=side,
        start_frame=start,
        end_frame=end,
        start_timestamp_seconds=float(start),
        end_timestamp_seconds=float(end),
        duration_seconds=float(end - start),
        automatic_stride_id=f"automatic-{stride_id}",
        review_intent="accept",
        review_changes="none",
        provenance_notes="synthetic",
    )


def _reviewed_stride(
    stride_id: str,
    *,
    side: str,
    start: int,
    end: int,
) -> ParsedReviewedStride:
    return ParsedReviewedStride(
        stride_id=stride_id,
        side=side,
        start_event_id=f"{stride_id}-start",
        end_event_id=f"{stride_id}-end",
        start_frame=start,
        end_frame=end,
        start_timestamp_seconds=float(start),
        end_timestamp_seconds=float(end),
        duration_seconds=float(end - start),
        quality="reviewed",
        contralateral_event_id=None,
        contralateral_event_count=1,
        sequence_notes=(),
        source="synthetic",
        review_status="accepted",
        automatic_stride_id=f"automatic-{stride_id}",
        review_intent="accept",
        review_changes="none",
        provenance_notes="synthetic",
    )


def _normalized_sample(index: int, timestamp: float) -> StrideComSample:
    return StrideComSample(
        progression=index * 50.0,
        com=Point2D(0.5, 0.5),
        mass_coverage=0.85,
        usable=True,
        sample_kind="normalized",
        normalized_index=index,
        method="exact",
        source_frame_index=None,
        source_timestamp_seconds=None,
        target_timestamp_seconds=timestamp,
        left_source_frame_index=None,
        left_source_timestamp_seconds=None,
        right_source_frame_index=None,
        right_source_timestamp_seconds=None,
        min_endpoint_coverage=0.85,
        contributors=(),
        qc_flags=(),
    )


def test_structural_head_is_separate_from_transient_supported_segment_loss() -> None:
    frames = tuple(
        _frame(
            index,
            segments=(
                _segment("head", usable=False, mass=0.0694, raw=False),
                _segment("trunk", usable=index not in (1, 2), mass=0.4346),
            ),
        )
        for index in range(4)
    )

    head = compute_segment_coverage("head", frames, "male")
    trunk = compute_segment_coverage("trunk", frames, "male")

    assert head.is_structurally_unsupported is True
    assert head.is_supported is False
    # Structurally unsupported: missing frames/fraction/run are 0/not counted
    assert head.missing_frames == 0
    assert head.missing_fraction == 0.0
    assert head.longest_contiguous_missing_run == 0
    assert head.missingness_pattern == "structurally_unsupported"
    assert head.lost_representable_mass == 0.0
    assert trunk.is_structurally_unsupported is False
    assert trunk.missing_frames == 2
    assert trunk.missing_fraction == 0.5
    assert trunk.longest_contiguous_missing_run == 2
    assert trunk.missingness_pattern == "intermittent"
    assert trunk.lost_representable_mass == pytest.approx(
        SEGMENT_MASS_FRACTIONS["male"]["trunk"] * 0.5
    )


def test_landmark_summary_uses_direct_processed_qc_flags() -> None:
    frames = tuple(_frame(index) for index in range(4))
    rows = (
        _landmark_row(0, "left_wrist"),
        _landmark_row(
            1,
            "left_wrist",
            raw=False,
            x_interpolated=True,
            y_smoothed=True,
            y_smoothing_interpolation=True,
        ),
        _landmark_row(
            2,
            "left_wrist",
            final=False,
            raw=False,
            y_interpolated=True,
            x_smoothed=True,
            x_smoothing_interpolation=True,
        ),
        _landmark_row(3, "left_wrist", final=False, raw=False),
    )

    summary = compute_landmark_coverage(
        "left_wrist", rows, frames, "male", {"left_wrist": ()}
    )

    assert summary.raw_observed_usable_frames == 1
    assert summary.raw_observed_usable_fraction == 0.25
    assert summary.x_interpolated_frames == 1
    assert summary.x_interpolated_fraction == 0.25
    assert summary.y_interpolated_frames == 1
    assert summary.y_interpolated_fraction == 0.25
    assert summary.x_smoothing_changed_frames == 1
    assert summary.x_smoothing_changed_fraction == 0.25
    assert summary.y_smoothing_changed_frames == 1
    assert summary.y_smoothing_changed_fraction == 0.25
    assert summary.x_smoothing_support_interpolation_frames == 1
    assert summary.x_smoothing_support_interpolation_fraction == 0.25
    assert summary.y_smoothing_support_interpolation_frames == 1
    assert summary.y_smoothing_support_interpolation_fraction == 0.25
    assert summary.final_missing_frames == 2
    assert summary.final_missing_fraction == 0.5
    assert summary.longest_contiguous_missing_run == 2


def test_landmark_lost_mass_is_nonexclusive_across_dependent_segments() -> None:
    frames = (
        _frame(
            0,
            segments=(
                _segment("left_upper_arm", usable=False, mass=0.03),
                _segment("trunk", usable=False, mass=0.40),
                _segment("head", usable=False, mass=0.07),
            ),
        ),
        _frame(
            1,
            segments=(
                _segment("left_upper_arm", mass=0.03),
                _segment("trunk", mass=0.40),
                _segment("head", usable=False, mass=0.07),
            ),
        ),
    )
    rows = (
        _landmark_row(0, "left_shoulder", final=False, raw=False),
        _landmark_row(1, "left_shoulder"),
    )
    dependency_map: dict[str, tuple[str, ...]] = {
        "left_shoulder": ("head", "left_upper_arm", "trunk"),
    }

    summary = compute_landmark_coverage(
        "left_shoulder", rows, frames, "male", dependency_map
    )

    assert summary.nonexclusive_affected_mass_fraction == pytest.approx(
        (0.03 + 0.40) / 2
    )


def test_segment_and_direct_landmark_asymmetry_preserve_signed_differences() -> None:
    frames = tuple(
        _frame(
            index,
            segments=(
                _segment(
                    "left_upper_arm",
                    usable=index != 3,
                    raw=index in (0, 1),
                ),
                _segment(
                    "right_upper_arm",
                    usable=index in (0, 1),
                    raw=index == 0,
                ),
            ),
        )
        for index in range(4)
    )
    landmark_rows = tuple(
        [
            _landmark_row(
                index,
                "left_wrist",
                final=index != 3,
                raw=index in (0, 1),
            )
            for index in range(4)
        ]
        + [
            _landmark_row(
                index,
                "right_wrist",
                final=index in (0, 1),
                raw=index == 0,
            )
            for index in range(4)
        ]
    )
    segments = (
        compute_segment_coverage("left_upper_arm", frames, "male"),
        compute_segment_coverage("right_upper_arm", frames, "male"),
    )
    landmarks = tuple(
        compute_landmark_coverage(name, landmark_rows, frames, "male", {name: ()})
        for name in ("left_wrist", "right_wrist")
    )

    summaries = compute_asymmetry_summaries(segments, landmarks, "male")
    by_name = {summary.pair_name: summary for summary in summaries}

    upper_arm = by_name["upper_arm"]
    assert upper_arm.usability_difference == 0.25
    assert upper_arm.raw_observed_difference == 0.25
    assert upper_arm.missing_difference == -0.25
    wrist = by_name["wrist"]
    assert wrist.mass_fraction_per_side == 0.0
    assert wrist.usability_difference == 0.25
    assert wrist.raw_observed_difference == 0.25
    assert wrist.missing_difference == -0.25


def test_primary_stride_policy_complete_and_explicit_failure_reasons() -> None:
    stable_frames = (
        _frame(0, segments=(_segment("trunk"),)),
        _frame(1, segments=(_segment("trunk"),)),
    )
    strict = compute_stride_coverage(
        "stride-a",
        (),
        stable_frames,
        _window(),
        [0.0, 0.5, 1.0],
        0.80,
        "male",
        (0.80,),
    )
    assert strict.qualification_category == "policy_complete_at_threshold"
    assert strict.failure_reasons == ()
    # Explicit stride booleans for policy-complete decomposition
    assert strict.all_original_frames_policy_eligible is True
    assert strict.represented_segment_set_invariant is True
    assert strict.normalized_grid_complete is True
    assert strict.endpoints_policy_eligible is True

    insufficient_frames = (
        stable_frames[0],
        _frame(1, coverage=0.79, segments=(_segment("trunk"),)),
    )
    insufficient = compute_stride_coverage(
        "stride-a",
        (),
        insufficient_frames,
        _window(),
        [0.0, 0.5, 1.0],
        0.80,
        "male",
        (0.80,),
    )
    assert insufficient.qualification_category == "insufficient_coverage"
    assert insufficient.failure_reasons == (
        "usable fraction 0.500 < 1.0 at primary threshold",
    )
    assert insufficient.all_original_frames_policy_eligible is False

    changed_segments = (
        stable_frames[0],
        _frame(1, segments=(_segment("left_thigh"),)),
    )
    failing = compute_stride_coverage(
        "stride-a",
        (),
        changed_segments,
        _window(),
        [0.0, 0.5, 1.0, 1.5],
        0.80,
        "male",
        (0.80,),
    )
    assert failing.qualification_category == "usable_samples_only"
    assert failing.failure_reasons == (
        "represented segment set not invariant",
        "not all normalized samples available",
    )
    assert failing.all_original_frames_policy_eligible is True
    assert failing.represented_segment_set_invariant is False
    assert failing.normalized_grid_complete is False


def test_supported_segment_missing_count_differs_from_segment_frame_burden() -> None:
    frames = (
        _frame(
            0,
            segments=(
                _segment("left_forearm", usable=False),
                _segment("right_forearm"),
            ),
        ),
        _frame(
            1,
            segments=(
                _segment("left_forearm", usable=False),
                _segment("right_forearm", usable=False),
            ),
        ),
        _frame(
            2,
            segments=(_segment("left_forearm"), _segment("right_forearm")),
        ),
    )

    summary = compute_stride_coverage(
        "stride-a",
        (),
        frames,
        _window(end=2),
        [0.0, 1.0, 2.0],
        0.80,
        "male",
        (0.80,),
    )

    assert summary.supported_segment_missing_count == 2
    assert summary.supported_segment_missing_frames == 3
    assert summary.supported_segment_missing_max_consecutive == 2


def test_aggregate_default_and_custom_threshold_sensitivity_is_deterministic() -> None:
    frames = (
        _frame(0, coverage=0.82, segments=(_segment("trunk"),)),
        _frame(1, coverage=0.82, segments=(_segment("trunk"),)),
        _frame(2, coverage=0.90, segments=(_segment("trunk"),)),
    )
    reviewed = [
        _reviewed_stride("stride-a", side="left", start=0, end=1),
        _reviewed_stride("stride-b", side="right", start=1, end=2),
    ]
    stride_results = {
        "stride-a": tuple(
            _normalized_sample(index, timestamp)
            for index, timestamp in enumerate((0.0, 0.5, 1.0))
        ),
        "stride-b": tuple(
            _normalized_sample(index, timestamp)
            for index, timestamp in enumerate((1.0, 1.5, 2.0))
        ),
    }

    default = compute_qualification(
        frames,
        stride_results,
        reviewed,
        ComQualificationConfig(),
        primary_threshold=0.82,
        anthropometry_sex="male",
        landmark_rows=(),
    )
    custom = compute_qualification(
        frames,
        stride_results,
        reviewed,
        ComQualificationConfig(coverage_thresholds=(0.82, 0.83, 0.90)),
        primary_threshold=0.82,
        anthropometry_sex="male",
        landmark_rows=(),
    )

    assert tuple(item.threshold for item in default.threshold_sensitivity) == (
        DEFAULT_COVERAGE_THRESHOLDS
    )
    assert tuple(item.threshold for item in custom.threshold_sensitivity) == (
        0.82,
        0.83,
        0.90,
    )
    assert (
        tuple(item.segment_name for item in custom.segment_summaries) == SEGMENT_NAMES
    )
    assert tuple(item.landmark_name for item in custom.landmark_summaries) == tuple(
        sorted(item.landmark_name for item in custom.landmark_summaries)
    )
    assert tuple(item.stride_id for item in custom.stride_summaries) == (
        "stride-a",
        "stride-b",
    )

    at_equality, above_equality, at_high_equality = custom.threshold_sensitivity
    default_by_threshold = {
        item.threshold: item for item in default.threshold_sensitivity
    }
    assert default_by_threshold[0.82] == at_equality
    assert default_by_threshold[0.84].stride_usable_original_frames == {
        "stride-a": 0,
        "stride-b": 1,
    }
    assert default_by_threshold[0.84].policy_complete_strides == 0
    assert default_by_threshold[0.84].normalized_exact_match_usable == 1
    assert default_by_threshold[0.84].normalized_linear_interpolation_usable == 0
    assert default_by_threshold[0.84].normalized_total_samples == 6
    assert at_equality.usable_frames == 3
    assert at_equality.stride_usable_original_frames == {
        "stride-a": 2,
        "stride-b": 2,
    }
    assert at_equality.policy_complete_strides == 2
    assert at_equality.normalized_total_samples == 6
    assert at_equality.normalized_exact_match_usable == 4
    assert at_equality.normalized_linear_interpolation_usable == 2
    assert at_equality.normalized_usable_samples == 6
    assert at_equality.normalized_usable_fraction == 1.0

    for result in (above_equality, at_high_equality):
        assert result.usable_frames == 1
        assert result.stride_usable_original_frames == {
            "stride-a": 0,
            "stride-b": 1,
        }
        assert result.policy_complete_strides == 0
        assert result.normalized_total_samples == 6
        assert result.normalized_exact_match_usable == 1
        assert result.normalized_linear_interpolation_usable == 0
        assert result.normalized_usable_samples == 1
        assert result.normalized_usable_fraction == pytest.approx(1 / 6)


def test_female_supported_max_and_summed_instance_masses() -> None:
    """Test female theoretical supported max and instance masses."""
    from gait_stability.com_estimation import (
        REPRESENTED_MASS_MAX_FEMALE,
        SEGMENT_NAMES,
        UNSUPPORTED_SEGMENTS,
    )
    from gait_stability.com_qualification import (
        THEORETICAL_SUPPORTED_MASS,
        _get_segment_mass,
        theoretical_supported_mass_fraction,
    )

    # Theoretical supported mass for female
    assert theoretical_supported_mass_fraction("female") == REPRESENTED_MASS_MAX_FEMALE
    assert THEORETICAL_SUPPORTED_MASS["female"] == REPRESENTED_MASS_MAX_FEMALE

    # Sum of all supported segment masses (instance masses, bilateral) for female
    supported_segments = [s for s in SEGMENT_NAMES if s not in UNSUPPORTED_SEGMENTS]
    summed_instance_mass = sum(
        _get_segment_mass("female", s) for s in supported_segments
    )
    # Should equal theoretical max (0.9331 for female)
    assert summed_instance_mass == pytest.approx(REPRESENTED_MASS_MAX_FEMALE, rel=1e-10)

    # Male for comparison
    from gait_stability.com_estimation import REPRESENTED_MASS_MAX_MALE
    from gait_stability.com_qualification import THEORETICAL_SUPPORTED_MASS as TSM

    assert theoretical_supported_mass_fraction("male") == REPRESENTED_MASS_MAX_MALE
    assert TSM["male"] == REPRESENTED_MASS_MAX_MALE
    summed_instance_mass_male = sum(
        _get_segment_mass("male", s) for s in supported_segments
    )
    assert summed_instance_mass_male == pytest.approx(
        REPRESENTED_MASS_MAX_MALE, rel=1e-10
    )


def test_aggregate_lost_mass_vs_theoretical_minus_mean_coverage() -> None:
    """Aggregate lost representable mass should equal theoretical max minus mean
    absolute supported mass coverage (within tolerance).

    This is a sanity check: sum over all supported segments of
    (mass_fraction * missing_fraction) should equal
    theoretical_supported_mass - mean(supported_mass_coverage *
    theoretical_supported_mass).
    """
    from gait_stability.com_estimation import (
        DE_LEVA_FEMALE,
        SEGMENT_NAMES,
        UNSUPPORTED_SEGMENTS,
        FrameComResult,
        Point2D,
        SegmentComResult,
        SegmentProvenance,
    )
    from gait_stability.com_qualification import (
        _get_segment_mass,
        compute_segment_coverage,
        theoretical_supported_mass_fraction,
    )

    sex: Literal["male", "female"] = "female"
    theoretical = theoretical_supported_mass_fraction(sex)

    # Get all supported segment instance masses
    supported_segments = [s for s in SEGMENT_NAMES if s not in UNSUPPORTED_SEGMENTS]
    segment_masses = {seg: _get_segment_mass(sex, seg) for seg in supported_segments}

    # Minimal provenance for segments
    def _prov(seg_name: str, usable: bool) -> SegmentProvenance:
        return SegmentProvenance(
            segment_name=seg_name,
            proximal_landmark=f"{seg_name}_prox",
            distal_landmark=f"{seg_name}_dist",
            contributors=(f"{seg_name}_prox", f"{seg_name}_dist"),
            usable=usable,
            mass_fraction=segment_masses.get(seg_name, 0.0),
            r=0.5,
            all_raw_observed=usable,
            any_x_interpolated=False,
            any_y_interpolated=False,
            any_x_smoothing_changed=False,
            any_y_smoothing_changed=False,
            any_x_smoothing_support_interpolation=False,
            any_y_smoothing_support_interpolation=False,
            other_qc_limited=False,
        )

    # Create frames with known missing pattern using ALL supported segments
    frame_list: list[FrameComResult] = []
    for index in range(12):
        seg_results = []
        frame_mass = 0.0
        for seg in supported_segments:
            mass = segment_masses[seg]
            if seg == "trunk":
                usable = True
            elif "thigh" in seg:
                usable = index % 2 == 0
            elif "shank" in seg:
                usable = index % 3 != 0
            elif "upper_arm" in seg:
                usable = index % 4 != 0
            elif "forearm" in seg:
                usable = index % 5 != 0
            elif "hand" in seg:
                usable = index % 6 != 0
            elif "foot" in seg:
                usable = index % 7 != 0
            else:
                usable = True
            if usable:
                frame_mass += mass
            seg_results.append(
                SegmentComResult(
                    segment_name=seg,
                    com=Point2D(0.5, 0.5) if usable else None,
                    mass_fraction=mass,
                    usable=usable,
                    provenance=_prov(seg, usable),
                )
            )
        # Add head (structurally unsupported, never usable)
        head_mass = DE_LEVA_FEMALE["head"]["mass"]
        seg_results.append(
            SegmentComResult(
                segment_name="head",
                com=None,
                mass_fraction=head_mass,
                usable=False,
                provenance=_prov("head", False),
            )
        )
        frame_list.append(
            FrameComResult(
                frame_index=index,
                timestamp_seconds=float(index),
                frame_status="decoded_pose",
                com=Point2D(0.5, 0.5) if frame_mass > 0 else None,
                mass_coverage=frame_mass,
                usable=frame_mass > 0.0,
                segment_results=tuple(seg_results),
                model_total_mass=1.0,
            )
        )
    frames = tuple(frame_list)

    # Compute segment summaries for all segments
    segment_summaries = tuple(
        compute_segment_coverage(seg, frames, sex) for seg in SEGMENT_NAMES
    )

    # Sum of lost representable mass across supported segments
    total_lost_representable_mass = sum(
        s.lost_representable_mass for s in segment_summaries if s.is_supported
    )

    # Supported mass coverage per frame = mass_coverage / theoretical
    coverages = [fr.mass_coverage for fr in frames]
    supported_coverages = [c / theoretical for c in coverages]
    mean_supported_coverage = sum(supported_coverages) / len(supported_coverages)

    # Theoretical max minus mean absolute coverage = theoretical *
    # (1 - mean_supported_coverage)
    expected_lost = theoretical - (mean_supported_coverage * theoretical)

    assert total_lost_representable_mass == pytest.approx(expected_lost, rel=1e-6)

    # Also test: sum of segment lost mass = theoretical * mean missing fraction
    # where missing fraction is 1 - supported_coverage for each frame
    frame_missing_fractions = [1.0 - sc for sc in supported_coverages]
    mean_missing_fraction = sum(frame_missing_fractions) / len(frame_missing_fractions)
    assert total_lost_representable_mass == pytest.approx(
        theoretical * mean_missing_fraction, rel=1e-6
    )
