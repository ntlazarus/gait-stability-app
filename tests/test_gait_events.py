from __future__ import annotations

from dataclasses import fields, replace
from typing import Literal

import pytest

from gait_stability.gait_events import (
    GAIT_EVENT_FIELDS,
    STRIDE_FIELDS,
    GaitEvent,
    GaitEventConfig,
    SignalSample,
    Stride,
    construct_strides,
    detect_candidate_events,
)

Side = Literal["left", "right"]


def _trajectory(
    peaks: tuple[int, ...] = (),
    *,
    direction: str = "image_right",
    end_frame: int = 120,
    timestamp_step: float = 0.05,
    amplitudes: dict[int, float] | None = None,
    plateaus: dict[int, int] | None = None,
    missing_processed: set[int] | None = None,
    include_raw: bool = True,
    include_ankle: bool = True,
    interpolated: set[int] | None = None,
    smoothing_interpolated: set[int] | None = None,
    unusable: set[int] | None = None,
    ankle_interpolated: set[int] | None = None,
    ankle_smoothing_interpolated: set[int] | None = None,
    ankle_unusable: set[int] | None = None,
    ankle_peaks: tuple[int, ...] | None = None,
    hip_interpolated: set[int] | None = None,
    hip_smoothing_interpolated: set[int] | None = None,
    hip_unusable: set[int] | None = None,
) -> tuple[SignalSample, ...]:
    amplitudes = amplitudes or {}
    plateaus = plateaus or {}
    missing_processed = missing_processed or set()
    interpolated = interpolated or set()
    smoothing_interpolated = smoothing_interpolated or set()
    unusable = unusable or set()
    ankle_interpolated = ankle_interpolated or set()
    ankle_smoothing_interpolated = ankle_smoothing_interpolated or set()
    ankle_unusable = ankle_unusable or set()
    ankle_peaks = peaks if ankle_peaks is None else ankle_peaks
    hip_interpolated = hip_interpolated or set()
    hip_smoothing_interpolated = hip_smoothing_interpolated or set()
    hip_unusable = hip_unusable or set()
    sign = 1.0 if direction == "image_right" else -1.0
    samples: list[SignalSample] = []
    for frame in range(end_frame + 1):
        signal = 0.05
        for peak in peaks:
            amplitude = amplitudes.get(peak, 0.20)
            plateau_length = plateaus.get(peak, 1)
            if peak <= frame < peak + plateau_length:
                contribution = amplitude
            else:
                distance = min(
                    abs(frame - peak), abs(frame - (peak + plateau_length - 1))
                )
                contribution = max(0.0, amplitude * (1.0 - distance / 4.0))
            signal = max(signal, 0.05 + contribution)
        heel = 0.5 + sign * signal
        ankle_signal = 0.05
        for peak in ankle_peaks:
            distance = abs(frame - peak)
            contribution = max(0.0, 0.18 * (1.0 - distance / 4.0))
            ankle_signal = max(ankle_signal, 0.05 + contribution)
        ankle = 0.5 + sign * ankle_signal
        processed_present = frame not in missing_processed
        samples.append(
            SignalSample(
                frame_index=frame,
                nominal_timestamp_seconds=frame * timestamp_step,
                processed_heel_x=heel if processed_present else None,
                processed_hip_left_x=0.45 if processed_present else None,
                processed_hip_right_x=0.55 if processed_present else None,
                raw_heel_x=heel if include_raw else None,
                raw_hip_left_x=0.45 if include_raw else None,
                raw_hip_right_x=0.55 if include_raw else None,
                processed_ankle_x=ankle if include_ankle else None,
                primary_observed_usable=frame not in unusable,
                primary_interpolated=frame in interpolated,
                primary_smoothing_support_contains_interpolation=(
                    frame in smoothing_interpolated
                ),
                ankle_observed_usable=(include_ankle and frame not in ankle_unusable),
                ankle_interpolated=frame in ankle_interpolated,
                ankle_smoothing_support_contains_interpolation=(
                    frame in ankle_smoothing_interpolated
                ),
                hip_observed_usable=(
                    frame not in hip_unusable and frame not in hip_interpolated
                ),
                hip_interpolated=frame in hip_interpolated,
                hip_smoothing_support_contains_interpolation=(
                    frame in hip_smoothing_interpolated
                ),
            )
        )
    return tuple(samples)


def _samples(
    left: tuple[int, ...],
    right: tuple[int, ...],
    **kwargs: object,
) -> dict[str, tuple[SignalSample, ...]]:
    return {
        "left": _trajectory(left, **kwargs),
        "right": _trajectory(right, **kwargs),
    }


def _accepted(events: tuple[GaitEvent, ...]) -> tuple[GaitEvent, ...]:
    return tuple(event for event in events if event.detection_status == "accepted")


def test_regular_alternating_trajectories_and_stable_ids() -> None:
    events = detect_candidate_events(
        _samples((20, 60, 100), (40, 80)), GaitEventConfig(), 0, 120
    )

    assert [(event.event_id, event.side, event.frame_index) for event in events] == [
        ("E0001", "left", 20),
        ("E0002", "right", 40),
        ("E0003", "left", 60),
        ("E0004", "right", 80),
        ("E0005", "left", 100),
    ]
    assert all(event.detection_status == "accepted" for event in events)
    assert all(event.confidence_or_quality == "high" for event in events)
    assert all(event.raw_peak_offset_frames == 0 for event in events)
    assert all(event.ankle_peak_offset_frames == 0 for event in events)
    assert all(not event.sequence_context_notes for event in events)


def test_direction_mirror_detects_identical_events() -> None:
    rightward = detect_candidate_events(
        _samples((30,), (60,)), GaitEventConfig(), 0, 100
    )
    leftward = detect_candidate_events(
        _samples((30,), (60,), direction="image_left"),
        GaitEventConfig(direction="image_left"),
        0,
        100,
    )

    assert [(event.side, event.frame_index) for event in rightward] == [
        (event.side, event.frame_index) for event in leftward
    ]
    assert [event.peak_value for event in rightward] == pytest.approx(
        [event.peak_value for event in leftward]
    )


def test_peak_prominence_reversal_and_forward_gates() -> None:
    config = GaitEventConfig(
        min_prominence=0.10,
        derivative_deadband=0.5,
        min_forward_relative_x=0.20,
    )
    samples = {
        "left": _trajectory((30, 70), amplitudes={30: 0.20, 70: 0.04}),
        "right": _trajectory(),
    }
    events = detect_candidate_events(samples, config, 0, 120)

    assert [(event.frame_index, event.detection_status) for event in events] == [
        (30, "accepted"),
        (70, "rejected_candidate"),
    ]
    accepted, rejected = events
    assert accepted.prominence == pytest.approx(0.20)
    assert accepted.pre_velocity is not None and accepted.pre_velocity > 0.5
    assert accepted.post_velocity is not None and accepted.post_velocity < -0.5
    assert "prominence_below_threshold" in rejected.rejection_reasons
    assert "peak_below_min_forward_relative_x" in rejected.rejection_reasons
    assert rejected.confidence_or_quality == "low"


def test_exact_even_plateau_is_one_candidate_at_earlier_midpoint() -> None:
    events = detect_candidate_events(
        {"left": _trajectory((30,), plateaus={30: 2}), "right": _trajectory()},
        GaitEventConfig(),
        0,
        100,
    )

    assert len(events) == 1
    assert events[0].frame_index == 30
    assert events[0].plateau_start_frame == 30
    assert events[0].plateau_end_frame == 31
    assert events[0].detection_status == "accepted"


def test_same_side_conflict_rejects_later_tie_and_equality_passes() -> None:
    conflict = detect_candidate_events(
        _samples((30, 39), ()), GaitEventConfig(), 0, 100
    )
    equality = detect_candidate_events(
        _samples((30, 40), ()), GaitEventConfig(), 0, 100
    )

    assert [event.detection_status for event in conflict] == [
        "accepted",
        "rejected_candidate",
    ]
    assert conflict[1].rejection_reasons == ("same_side_interval_below_minimum",)
    assert len(_accepted(equality)) == 2
    assert equality[1].sequence_context_notes == ("nonalternating_adjacent_event",)


def test_same_side_conflict_prefers_higher_forward_peak_over_prominence() -> None:
    left = tuple(
        replace(sample, processed_heel_x=0.76)
        if 20 <= sample.frame_index < 30
        else sample
        for sample in _trajectory((30, 39), amplitudes={30: 0.25, 39: 0.20})
    )

    events = detect_candidate_events(
        {"left": left, "right": _trajectory()}, GaitEventConfig(), 0, 100
    )

    assert [(event.frame_index, event.detection_status) for event in events] == [
        (30, "accepted"),
        (39, "rejected_candidate"),
    ]
    assert events[0].peak_value > events[1].peak_value
    assert events[0].prominence is not None
    assert events[1].prominence is not None
    assert events[0].prominence < events[1].prominence
    assert events[1].rejection_reasons == ("same_side_interval_below_minimum",)


def test_same_side_conflict_does_not_count_unclean_ankle_cue() -> None:
    events = detect_candidate_events(
        {
            "left": _trajectory((30, 39), ankle_unusable={29, 30, 31}),
            "right": _trajectory(),
        },
        GaitEventConfig(),
        0,
        100,
    )

    assert [(event.frame_index, event.detection_status) for event in events] == [
        (30, "rejected_candidate"),
        (39, "accepted"),
    ]
    assert events[0].ankle_peak_frame == 30
    assert not events[0].ankle_support_observed_usable
    assert events[0].rejection_reasons == ("same_side_interval_below_minimum",)


def test_opposite_side_conflict_and_exact_equality() -> None:
    config = GaitEventConfig(opposite_side_min_interval_seconds=0.25)
    conflict = detect_candidate_events(
        _samples((30,), (30,), timestamp_step=0.25), config, 0, 100
    )
    equality = detect_candidate_events(
        _samples((30,), (31,), timestamp_step=0.25), config, 0, 100
    )

    assert [event.detection_status for event in conflict] == [
        "accepted",
        "rejected_candidate",
    ]
    assert conflict[1].rejection_reasons == ("opposite_side_interval_below_minimum",)
    assert len(_accepted(equality)) == 2


def test_missing_primary_data_splits_trajectory_without_event() -> None:
    events = detect_candidate_events(
        {
            "left": _trajectory((30,), missing_processed={30}),
            "right": _trajectory(),
        },
        GaitEventConfig(),
        0,
        100,
    )
    assert events == ()


def test_bout_boundaries_are_inclusive_and_other_candidates_are_omitted() -> None:
    events = detect_candidate_events(_samples((30, 60), ()), GaitEventConfig(), 30, 30)
    assert [(event.frame_index, event.detection_status) for event in events] == [
        (30, "accepted")
    ]


@pytest.mark.parametrize(
    ("kwargs", "expected_note"),
    [
        ({"interpolated": {38}}, "primary_support_contains_interpolation"),
        (
            {"smoothing_interpolated": {28}},
            "smoothing_support_contains_interpolation",
        ),
        ({"unusable": {31}}, "primary_support_not_all_observed_usable"),
    ],
)
def test_primary_support_provenance_downgrades_to_review(
    kwargs: dict[str, set[int]], expected_note: str
) -> None:
    events = detect_candidate_events(
        {"left": _trajectory((30,), **kwargs), "right": _trajectory()},
        GaitEventConfig(),
        0,
        100,
    )
    assert events[0].detection_status == "accepted"
    assert events[0].confidence_or_quality == "review"
    assert expected_note in events[0].signal_support_notes


def test_clean_ankle_peak_supports_high_quality() -> None:
    event = detect_candidate_events(_samples((30,), ()), GaitEventConfig(), 0, 100)[0]

    assert event.confidence_or_quality == "high"
    assert event.ankle_peak_frame == 30
    assert event.ankle_support_observed_usable
    assert not event.ankle_support_interpolated
    assert not event.ankle_support_smoothing_contains_interpolation


@pytest.mark.parametrize(
    ("kwargs", "field", "expected_note"),
    [
        (
            {"ankle_interpolated": {31}},
            "ankle_support_interpolated",
            "ankle_support_contains_interpolation",
        ),
        (
            {"ankle_smoothing_interpolated": {29}},
            "ankle_support_smoothing_contains_interpolation",
            "ankle_support_smoothing_contains_interpolation",
        ),
        (
            {"ankle_unusable": {30}},
            "ankle_support_observed_usable",
            "ankle_support_not_all_observed_usable",
        ),
    ],
)
def test_unclean_ankle_peak_is_retained_and_downgraded(
    kwargs: dict[str, set[int]], field: str, expected_note: str
) -> None:
    event = detect_candidate_events(
        {"left": _trajectory((30,), **kwargs), "right": _trajectory()},
        GaitEventConfig(),
        0,
        100,
    )[0]

    assert event.detection_status == "accepted"
    assert event.ankle_peak_frame == 30
    assert event.confidence_or_quality == "review"
    assert getattr(event, field) is (field != "ankle_support_observed_usable")
    assert expected_note in event.signal_support_notes


def test_ankle_cue_outside_primary_support_includes_hip_provenance() -> None:
    event = detect_candidate_events(
        {
            "left": _trajectory(
                (30,),
                ankle_peaks=(45,),
                hip_interpolated={45},
            ),
            "right": _trajectory(),
        },
        GaitEventConfig(ankle_agreement_window_frames=20),
        0,
        100,
    )[0]

    assert event.frame_index == 30
    assert event.ankle_peak_frame == 45
    assert event.detection_status == "accepted"
    assert event.ankle_support_interpolated
    assert not event.ankle_support_observed_usable
    assert event.confidence_or_quality == "review"
    assert "ankle_support_contains_interpolation" in event.signal_support_notes


@pytest.mark.parametrize("missing_cue", ["raw", "ankle"])
def test_missing_agreement_cue_is_review_not_rejection(missing_cue: str) -> None:
    events = detect_candidate_events(
        {
            "left": _trajectory(
                (30,),
                include_raw=missing_cue != "raw",
                include_ankle=missing_cue != "ankle",
            ),
            "right": _trajectory(),
        },
        GaitEventConfig(),
        0,
        100,
    )
    assert events[0].detection_status == "accepted"
    assert events[0].confidence_or_quality == "review"


def test_max_gap_is_warning_only_for_events() -> None:
    events = detect_candidate_events(_samples((20, 100), ()), GaitEventConfig(), 0, 120)
    assert all(event.detection_status == "accepted" for event in events)
    assert all(event.confidence_or_quality == "high" for event in events)
    assert events[1].sequence_context_notes == (
        "nonalternating_adjacent_event",
        "same_side_train_discontinuity",
    )


def test_construct_strides_exact_pairs_durations_and_contralateral_ids() -> None:
    events = detect_candidate_events(
        _samples((20, 60, 100), (40, 80)), GaitEventConfig(), 0, 120
    )
    strides = construct_strides(events, GaitEventConfig())

    assert [
        (
            stride.stride_id,
            stride.side,
            stride.start_event_id,
            stride.end_event_id,
            stride.duration_seconds,
            stride.contralateral_event_id,
            stride.quality,
        )
        for stride in strides
    ] == [
        ("S0001", "left", "E0001", "E0003", 2.0, "E0002", "high"),
        ("S0002", "right", "E0002", "E0004", 2.0, "E0003", "high"),
        ("S0003", "left", "E0003", "E0005", 2.0, "E0004", "high"),
    ]


def test_missing_event_is_not_fabricated_and_long_interval_is_preserved() -> None:
    events = detect_candidate_events(
        _samples((20, 100), (40, 80)), GaitEventConfig(), 0, 120
    )
    left_strides = [
        stride
        for stride in construct_strides(events, GaitEventConfig())
        if stride.side == "left"
    ]

    assert len(left_strides) == 1
    assert (left_strides[0].start_frame, left_strides[0].end_frame) == (20, 100)
    assert left_strides[0].duration_seconds == pytest.approx(4.0)
    assert left_strides[0].contralateral_event_count == 2
    assert left_strides[0].quality == "review"
    assert left_strides[0].sequence_notes == (
        "multiple_contralateral_events",
        "same_side_interval_above_warning",
    )


def test_schema_order_matches_dataclasses_exactly() -> None:
    assert GAIT_EVENT_FIELDS == tuple(field.name for field in fields(GaitEvent))
    assert STRIDE_FIELDS == tuple(field.name for field in fields(Stride))
    assert GAIT_EVENT_FIELDS == (
        "event_id",
        "frame_index",
        "timestamp_seconds",
        "side",
        "event_type",
        "detection_method",
        "detection_status",
        "included_in_stride_construction",
        "confidence_or_quality",
        "peak_value",
        "prominence",
        "pre_velocity",
        "post_velocity",
        "plateau_start_frame",
        "plateau_end_frame",
        "raw_peak_frame",
        "raw_peak_offset_frames",
        "ankle_peak_frame",
        "ankle_peak_offset_frames",
        "ankle_support_observed_usable",
        "ankle_support_interpolated",
        "ankle_support_smoothing_contains_interpolation",
        "primary_support_observed_usable",
        "primary_support_interpolated",
        "primary_support_smoothing_contains_interpolation",
        "signal_support_notes",
        "sequence_context_notes",
        "source",
        "review_status",
        "rejection_reasons",
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GaitEventConfig(direction=True),
        lambda: GaitEventConfig(peak_radius_frames=True),
        lambda: GaitEventConfig(min_prominence=False),
        lambda: GaitEventConfig(peak_radius_frames=0),
        lambda: GaitEventConfig(prominence_window_frames=0),
        lambda: GaitEventConfig(reversal_half_window_frames=0),
        lambda: GaitEventConfig(min_prominence=float("nan")),
        lambda: SignalSample(frame_index=True, nominal_timestamp_seconds=0.0),
        lambda: SignalSample(frame_index=0, nominal_timestamp_seconds=float("inf")),
        lambda: SignalSample(
            frame_index=0,
            nominal_timestamp_seconds=0.0,
            processed_heel_x=True,
        ),
        lambda: SignalSample(
            frame_index=0,
            nominal_timestamp_seconds=0.0,
            ankle_observed_usable=1,
        ),
        lambda: SignalSample(
            frame_index=0,
            nominal_timestamp_seconds=0.0,
            hip_interpolated=1,
        ),
        lambda: SignalSample(
            frame_index=0,
            nominal_timestamp_seconds=0.0,
            hip_observed_usable=1,
        ),
        lambda: SignalSample(
            frame_index=0,
            nominal_timestamp_seconds=0.0,
            hip_smoothing_support_contains_interpolation=1,
        ),
    ],
)
def test_strict_config_and_sample_validation(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        assert callable(factory)
        factory()


def test_mapping_order_and_timestamp_validation() -> None:
    config = GaitEventConfig()
    with pytest.raises(ValueError, match="exactly"):
        detect_candidate_events({"left": ()}, config, 0, 1)
    duplicate = (
        SignalSample(frame_index=1, nominal_timestamp_seconds=0.0),
        SignalSample(frame_index=1, nominal_timestamp_seconds=0.1),
    )
    with pytest.raises(ValueError, match="ordered and unique"):
        detect_candidate_events({"left": duplicate, "right": ()}, config, 0, 1)
    reversed_time = (
        SignalSample(frame_index=1, nominal_timestamp_seconds=0.1),
        SignalSample(frame_index=2, nominal_timestamp_seconds=0.1),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        detect_candidate_events({"left": reversed_time, "right": ()}, config, 0, 2)
