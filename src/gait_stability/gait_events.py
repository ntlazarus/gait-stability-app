"""Pure gait-event detection from pelvis-relative image-plane trajectories.

The detector reports candidate initial contacts, not validated clinical events.
It uses normalized image x coordinates and the bilateral hip midpoint as a
pelvis proxy; no metric scale or laboratory coordinate system is implied.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Literal

Side = Literal["left", "right"]
Quality = Literal["high", "review", "low"]
Status = Literal["accepted", "rejected_candidate"]


def _number(name: str, value: object, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class GaitEventConfig:
    direction: str = "image_right"
    peak_radius_frames: int = 2
    prominence_window_frames: int = 10
    min_prominence: float = 0.02
    reversal_half_window_frames: int = 2
    derivative_deadband: float = 0.0
    min_forward_relative_x: float = 0.0
    raw_agreement_window_frames: int = 2
    ankle_agreement_window_frames: int = 2
    same_side_min_interval_seconds: float = 0.5
    opposite_side_min_interval_seconds: float = 0.15
    same_side_max_interval_warning_seconds: float = 3.0

    def __post_init__(self) -> None:
        if not isinstance(self.direction, str):
            raise TypeError("direction must be a string")
        if self.direction not in {"image_right", "image_left"}:
            raise ValueError("direction must be 'image_right' or 'image_left'")
        for name in (
            "peak_radius_frames",
            "prominence_window_frames",
            "reversal_half_window_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, not bool")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("raw_agreement_window_frames", "ankle_agreement_window_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, not bool")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "min_prominence",
            "derivative_deadband",
            "same_side_min_interval_seconds",
            "opposite_side_min_interval_seconds",
        ):
            _number(name, getattr(self, name), nonnegative=True)
        _number("min_forward_relative_x", self.min_forward_relative_x)
        maximum = _number(
            "same_side_max_interval_warning_seconds",
            self.same_side_max_interval_warning_seconds,
        )
        if maximum <= 0.0:
            raise ValueError("same_side_max_interval_warning_seconds must be positive")


@dataclass(frozen=True, slots=True)
class SignalSample:
    """One side's observations and provenance flags for one video frame."""

    frame_index: int
    nominal_timestamp_seconds: float
    processed_heel_x: float | None = None
    processed_hip_left_x: float | None = None
    processed_hip_right_x: float | None = None
    raw_heel_x: float | None = None
    raw_hip_left_x: float | None = None
    raw_hip_right_x: float | None = None
    processed_ankle_x: float | None = None
    raw_ankle_x: float | None = None
    primary_observed_usable: bool = False
    primary_interpolated: bool = False
    primary_smoothing_support_contains_interpolation: bool = False
    ankle_observed_usable: bool = False
    ankle_interpolated: bool = False
    ankle_smoothing_support_contains_interpolation: bool = False
    hip_observed_usable: bool = False
    hip_interpolated: bool = False
    hip_smoothing_support_contains_interpolation: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer, not bool")
        _number("nominal_timestamp_seconds", self.nominal_timestamp_seconds)
        for name in (
            "processed_heel_x",
            "processed_hip_left_x",
            "processed_hip_right_x",
            "raw_heel_x",
            "raw_hip_left_x",
            "raw_hip_right_x",
            "processed_ankle_x",
            "raw_ankle_x",
        ):
            value = getattr(self, name)
            if value is not None:
                _number(name, value)
        for name in (
            "primary_observed_usable",
            "primary_interpolated",
            "primary_smoothing_support_contains_interpolation",
            "ankle_observed_usable",
            "ankle_interpolated",
            "ankle_smoothing_support_contains_interpolation",
            "hip_observed_usable",
            "hip_interpolated",
            "hip_smoothing_support_contains_interpolation",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")


@dataclass(frozen=True, slots=True, kw_only=True)
class GaitEvent:
    event_id: str
    frame_index: int
    timestamp_seconds: float
    side: Side
    event_type: str = "candidate_initial_contact"
    detection_method: str
    detection_status: Status
    included_in_stride_construction: bool
    confidence_or_quality: Quality
    peak_value: float
    prominence: float | None
    pre_velocity: float | None
    post_velocity: float | None
    plateau_start_frame: int
    plateau_end_frame: int
    raw_peak_frame: int | None
    raw_peak_offset_frames: int | None
    ankle_peak_frame: int | None
    ankle_peak_offset_frames: int | None
    ankle_support_observed_usable: bool
    ankle_support_interpolated: bool
    ankle_support_smoothing_contains_interpolation: bool
    primary_support_observed_usable: bool
    primary_support_interpolated: bool
    primary_support_smoothing_contains_interpolation: bool
    signal_support_notes: tuple[str, ...]
    sequence_context_notes: tuple[str, ...]
    source: str = "automatic"
    review_status: str = "unreviewed"
    rejection_reasons: tuple[str, ...]


GAIT_EVENT_FIELDS: tuple[str, ...] = tuple(field.name for field in fields(GaitEvent))


@dataclass(frozen=True, slots=True, kw_only=True)
class Stride:
    stride_id: str
    side: Side
    start_event_id: str
    end_event_id: str
    start_frame: int
    end_frame: int
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    duration_seconds: float
    quality: Quality
    contralateral_event_id: str | None
    contralateral_event_count: int
    sequence_notes: tuple[str, ...]
    source: str = "automatic"
    review_status: str = "unreviewed"


STRIDE_FIELDS: tuple[str, ...] = tuple(field.name for field in fields(Stride))


@dataclass(frozen=True, slots=True)
class _Point:
    sample: SignalSample
    value: float


@dataclass(frozen=True, slots=True)
class _CuePeak:
    frame: int
    support: tuple[_Point, ...]


def _relative(value: float, left_hip: float, right_hip: float, sign: float) -> float:
    return sign * (value - (left_hip + right_hip) / 2.0)


def _processed(sample: SignalSample, sign: float) -> float | None:
    if (
        sample.processed_heel_x is None
        or sample.processed_hip_left_x is None
        or sample.processed_hip_right_x is None
    ):
        return None
    return _relative(
        sample.processed_heel_x,
        sample.processed_hip_left_x,
        sample.processed_hip_right_x,
        sign,
    )


def _raw(sample: SignalSample, sign: float) -> float | None:
    if (
        sample.raw_heel_x is None
        or sample.raw_hip_left_x is None
        or sample.raw_hip_right_x is None
    ):
        return None
    return _relative(
        sample.raw_heel_x,
        sample.raw_hip_left_x,
        sample.raw_hip_right_x,
        sign,
    )


def _ankle(sample: SignalSample, sign: float) -> float | None:
    if (
        sample.processed_ankle_x is None
        or sample.processed_hip_left_x is None
        or sample.processed_hip_right_x is None
    ):
        return None
    return _relative(
        sample.processed_ankle_x,
        sample.processed_hip_left_x,
        sample.processed_hip_right_x,
        sign,
    )


def _segments(
    samples: Sequence[SignalSample],
    getter: Callable[[SignalSample], float | None],
) -> list[list[_Point]]:
    result: list[list[_Point]] = []
    current: list[_Point] = []
    previous_frame: int | None = None
    for sample in samples:
        value = getter(sample)
        if value is None:
            if current:
                result.append(current)
                current = []
            previous_frame = None
            continue
        if previous_frame is not None and sample.frame_index != previous_frame + 1:
            result.append(current)
            current = []
        current.append(_Point(sample, value))
        previous_frame = sample.frame_index
    if current:
        result.append(current)
    return result


def _plateau_bounds(points: Sequence[_Point], index: int) -> tuple[int, int]:
    start = index
    end = index
    while start > 0 and points[start - 1].value == points[index].value:
        start -= 1
    while end + 1 < len(points) and points[end + 1].value == points[index].value:
        end += 1
    return start, end


def _local_peak_plateaus(
    points: Sequence[_Point], radius: int
) -> list[tuple[int, int, int]]:
    peaks: list[tuple[int, int, int]] = []
    index = 0
    while index < len(points):
        start, end = _plateau_bounds(points, index)
        if start >= radius and end + radius < len(points):
            neighbors = [point.value for point in points[start - radius : start]] + [
                point.value for point in points[end + 1 : end + radius + 1]
            ]
            if neighbors and points[index].value > max(neighbors):
                midpoint = start + (end - start) // 2
                peaks.append((midpoint, start, end))
        index = end + 1
    return peaks


def _cue_peak(
    segments: Sequence[Sequence[_Point]], frame: int, window: int
) -> _CuePeak | None:
    candidates: list[_CuePeak] = []
    for segment in segments:
        for index, plateau_start, plateau_end in _local_peak_plateaus(segment, 1):
            peak_frame = segment[index].sample.frame_index
            if abs(peak_frame - frame) <= window:
                candidates.append(
                    _CuePeak(
                        frame=peak_frame,
                        support=tuple(segment[plateau_start - 1 : plateau_end + 2]),
                    )
                )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (abs(candidate.frame - frame), candidate.frame),
    )


def _validate_inputs(
    samples_by_side: Mapping[str, Sequence[SignalSample]],
    bout_start_frame: int,
    bout_end_frame: int,
) -> None:
    if set(samples_by_side) != {"left", "right"}:
        raise ValueError("samples_by_side must contain exactly 'left' and 'right'")
    for name, value in (
        ("bout_start_frame", bout_start_frame),
        ("bout_end_frame", bout_end_frame),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer, not bool")
    if bout_end_frame < bout_start_frame:
        raise ValueError("bout_end_frame must be at least bout_start_frame")
    for side in ("left", "right"):
        previous_frame: int | None = None
        previous_time: float | None = None
        for sample in samples_by_side[side]:
            if not isinstance(sample, SignalSample):
                raise TypeError(f"{side} samples must be SignalSample instances")
            if previous_frame is not None and sample.frame_index <= previous_frame:
                raise ValueError(f"{side} frame indices must be ordered and unique")
            if (
                previous_time is not None
                and sample.nominal_timestamp_seconds <= previous_time
            ):
                raise ValueError(
                    f"{side} nominal timestamps must be strictly increasing"
                )
            previous_frame = sample.frame_index
            previous_time = sample.nominal_timestamp_seconds


def _candidate_events_for_side(
    side: Side,
    samples: Sequence[SignalSample],
    config: GaitEventConfig,
    bout_start_frame: int,
    bout_end_frame: int,
) -> list[GaitEvent]:
    sign = 1.0 if config.direction == "image_right" else -1.0
    processed_segments = _segments(samples, lambda sample: _processed(sample, sign))
    raw_segments = _segments(samples, lambda sample: _raw(sample, sign))
    ankle_segments = _segments(samples, lambda sample: _ankle(sample, sign))
    events: list[GaitEvent] = []
    for segment in processed_segments:
        for index, plateau_start, plateau_end in _local_peak_plateaus(
            segment, config.peak_radius_frames
        ):
            point = segment[index]
            frame = point.sample.frame_index
            if not bout_start_frame <= frame <= bout_end_frame:
                continue
            reasons: list[str] = []
            prominence: float | None = None
            window = config.prominence_window_frames
            if index < window or index + window >= len(segment):
                reasons.append("insufficient_prominence_support")
            else:
                left_min = min(p.value for p in segment[index - window : index])
                right_min = min(
                    p.value for p in segment[index + 1 : index + window + 1]
                )
                prominence = point.value - max(left_min, right_min)
                if prominence < config.min_prominence:
                    reasons.append("prominence_below_threshold")
            half = config.reversal_half_window_frames
            pre_velocity: float | None = None
            post_velocity: float | None = None
            if index < half or index + half >= len(segment):
                reasons.append("insufficient_reversal_support")
            else:
                before = segment[index - half]
                after = segment[index + half]
                pre_velocity = (point.value - before.value) / (
                    point.sample.nominal_timestamp_seconds
                    - before.sample.nominal_timestamp_seconds
                )
                post_velocity = (after.value - point.value) / (
                    after.sample.nominal_timestamp_seconds
                    - point.sample.nominal_timestamp_seconds
                )
                if pre_velocity <= config.derivative_deadband:
                    reasons.append("pre_velocity_not_above_deadband")
                if post_velocity >= -config.derivative_deadband:
                    reasons.append("post_velocity_not_below_negative_deadband")
            if point.value < config.min_forward_relative_x:
                reasons.append("peak_below_min_forward_relative_x")
            support_half = max(
                config.peak_radius_frames,
                half,
                config.prominence_window_frames,
            )
            full_support = index >= support_half and index + support_half < len(segment)
            support = (
                segment[index - support_half : index + support_half + 1]
                if full_support
                else ()
            )
            observed = full_support and all(
                p.sample.primary_observed_usable for p in support
            )
            interpolated = any(p.sample.primary_interpolated for p in support)
            smoothing_interpolated = any(
                p.sample.primary_smoothing_support_contains_interpolation
                for p in support
            )
            raw_cue = _cue_peak(raw_segments, frame, config.raw_agreement_window_frames)
            ankle_cue = _cue_peak(
                ankle_segments, frame, config.ankle_agreement_window_frames
            )
            raw_peak = None if raw_cue is None else raw_cue.frame
            ankle_peak = None if ankle_cue is None else ankle_cue.frame
            ankle_support = () if ankle_cue is None else ankle_cue.support
            ankle_observed = bool(ankle_support) and all(
                p.sample.ankle_observed_usable and p.sample.hip_observed_usable
                for p in ankle_support
            )
            ankle_interpolated = any(
                p.sample.ankle_interpolated or p.sample.hip_interpolated
                for p in ankle_support
            )
            ankle_smoothing_interpolated = any(
                p.sample.ankle_smoothing_support_contains_interpolation
                or p.sample.hip_smoothing_support_contains_interpolation
                for p in ankle_support
            )
            ankle_clean = (
                ankle_observed
                and not ankle_interpolated
                and not ankle_smoothing_interpolated
            )
            notes: list[str] = []
            if raw_peak is None:
                notes.append("raw_peak_agreement_missing")
            else:
                notes.append("raw_peak_agreement")
            if ankle_peak is None:
                notes.append("ankle_peak_agreement_missing")
            else:
                notes.append("ankle_peak_agreement")
                if not ankle_observed:
                    notes.append("ankle_support_not_all_observed_usable")
                if ankle_interpolated:
                    notes.append("ankle_support_contains_interpolation")
                if ankle_smoothing_interpolated:
                    notes.append("ankle_support_smoothing_contains_interpolation")
            if interpolated:
                notes.append("primary_support_contains_interpolation")
            if smoothing_interpolated:
                notes.append("smoothing_support_contains_interpolation")
            if not observed:
                notes.append("primary_support_not_all_observed_usable")
            accepted = not reasons
            clean = observed and not interpolated and not smoothing_interpolated
            quality: Quality = (
                "high"
                if accepted and clean and raw_peak is not None and ankle_clean
                else "review"
                if accepted
                else "low"
            )
            events.append(
                GaitEvent(
                    event_id="",
                    frame_index=frame,
                    timestamp_seconds=point.sample.nominal_timestamp_seconds,
                    side=side,
                    detection_method="processed_pelvis_relative_heel_local_maximum",
                    detection_status="accepted" if accepted else "rejected_candidate",
                    included_in_stride_construction=accepted,
                    confidence_or_quality=quality,
                    peak_value=point.value,
                    prominence=prominence,
                    pre_velocity=pre_velocity,
                    post_velocity=post_velocity,
                    plateau_start_frame=segment[plateau_start].sample.frame_index,
                    plateau_end_frame=segment[plateau_end].sample.frame_index,
                    raw_peak_frame=raw_peak,
                    raw_peak_offset_frames=None
                    if raw_peak is None
                    else raw_peak - frame,
                    ankle_peak_frame=ankle_peak,
                    ankle_peak_offset_frames=None
                    if ankle_peak is None
                    else ankle_peak - frame,
                    ankle_support_observed_usable=ankle_observed,
                    ankle_support_interpolated=ankle_interpolated,
                    ankle_support_smoothing_contains_interpolation=(
                        ankle_smoothing_interpolated
                    ),
                    primary_support_observed_usable=observed,
                    primary_support_interpolated=interpolated,
                    primary_support_smoothing_contains_interpolation=smoothing_interpolated,
                    signal_support_notes=tuple(notes),
                    sequence_context_notes=(),
                    rejection_reasons=tuple(reasons),
                )
            )
    return events


def _support_rank(event: GaitEvent) -> tuple[int, int, float, float, int]:
    clean = int(
        event.primary_support_observed_usable
        and not event.primary_support_interpolated
        and not event.primary_support_smoothing_contains_interpolation
    )
    clean_ankle_cue = (
        event.ankle_peak_frame is not None
        and event.ankle_support_observed_usable
        and not event.ankle_support_interpolated
        and not event.ankle_support_smoothing_contains_interpolation
    )
    cues = int(event.raw_peak_frame is not None) + int(clean_ankle_cue)
    prominence = event.prominence if event.prominence is not None else -math.inf
    return clean, cues, event.peak_value, prominence, -event.frame_index


def _hard_conflict(
    events: Sequence[GaitEvent], config: GaitEventConfig
) -> tuple[int, int, str] | None:
    accepted = [
        index
        for index, event in enumerate(events)
        if event.detection_status == "accepted"
    ]
    conflicts: list[tuple[int, int, str]] = []
    for side in ("left", "right"):
        same_side = [index for index in accepted if events[index].side == side]
        for first, second in zip(same_side, same_side[1:], strict=False):
            if (
                events[second].timestamp_seconds - events[first].timestamp_seconds
                < config.same_side_min_interval_seconds
            ):
                conflicts.append((first, second, "same_side_interval_below_minimum"))
    for first, second in zip(accepted, accepted[1:], strict=False):
        if (
            events[first].side != events[second].side
            and events[second].timestamp_seconds - events[first].timestamp_seconds
            < config.opposite_side_min_interval_seconds
        ):
            conflicts.append((first, second, "opposite_side_interval_below_minimum"))
    if not conflicts:
        return None
    return min(
        conflicts,
        key=lambda item: (events[item[1]].timestamp_seconds, item[0], item[1]),
    )


def detect_candidate_events(
    samples_by_side: Mapping[str, Sequence[SignalSample]],
    config: GaitEventConfig,
    bout_start_frame: int,
    bout_end_frame: int,
) -> tuple[GaitEvent, ...]:
    """Return all formed local-peak candidates in chronological order."""

    _validate_inputs(samples_by_side, bout_start_frame, bout_end_frame)
    events = _candidate_events_for_side(
        "left", samples_by_side["left"], config, bout_start_frame, bout_end_frame
    ) + _candidate_events_for_side(
        "right", samples_by_side["right"], config, bout_start_frame, bout_end_frame
    )
    events.sort(
        key=lambda event: (event.timestamp_seconds, event.frame_index, event.side)
    )
    while (conflict := _hard_conflict(events, config)) is not None:
        first, second, reason = conflict
        loser = (
            first
            if _support_rank(events[first]) < _support_rank(events[second])
            else second
        )
        event = events[loser]
        events[loser] = replace(
            event,
            detection_status="rejected_candidate",
            included_in_stride_construction=False,
            confidence_or_quality="low",
            rejection_reasons=event.rejection_reasons + (reason,),
        )
    notes: list[list[str]] = [[] for _ in events]
    accepted = [
        index
        for index, event in enumerate(events)
        if event.detection_status == "accepted"
    ]
    for first, second in zip(accepted, accepted[1:], strict=False):
        if events[first].side == events[second].side:
            notes[second].append("nonalternating_adjacent_event")
    for side in ("left", "right"):
        same_side = [index for index in accepted if events[index].side == side]
        for first, second in zip(same_side, same_side[1:], strict=False):
            if (
                events[second].timestamp_seconds - events[first].timestamp_seconds
                > config.same_side_max_interval_warning_seconds
            ):
                notes[second].append("same_side_train_discontinuity")
    return tuple(
        replace(
            event,
            event_id=f"E{index:04d}",
            sequence_context_notes=tuple(notes[index - 1]),
        )
        for index, event in enumerate(events, start=1)
    )


def construct_strides(
    events: Sequence[GaitEvent], config: GaitEventConfig
) -> tuple[Stride, ...]:
    """Construct overlapping same-side strides without fabricating events."""

    accepted = sorted(
        (
            event
            for event in events
            if event.detection_status == "accepted"
            and event.included_in_stride_construction
            and event.event_type == "candidate_initial_contact"
        ),
        key=lambda event: (event.timestamp_seconds, event.frame_index, event.side),
    )
    pairs: list[tuple[GaitEvent, GaitEvent]] = []
    for side in ("left", "right"):
        side_events = [event for event in accepted if event.side == side]
        pairs.extend(zip(side_events, side_events[1:], strict=False))
    pairs.sort(
        key=lambda pair: (
            pair[0].timestamp_seconds,
            pair[1].timestamp_seconds,
            pair[0].side,
        )
    )
    strides: list[Stride] = []
    for index, (start, end) in enumerate(pairs, start=1):
        duration = end.timestamp_seconds - start.timestamp_seconds
        contralateral = [
            event
            for event in accepted
            if event.side != start.side
            and start.timestamp_seconds
            < event.timestamp_seconds
            < end.timestamp_seconds
        ]
        notes: list[str] = []
        if not contralateral:
            notes.append("missing_contralateral_event")
        elif len(contralateral) > 1:
            notes.append("multiple_contralateral_events")
        if duration <= 0.0:
            notes.append("nonpositive_duration")
        if duration > config.same_side_max_interval_warning_seconds:
            notes.append("same_side_interval_above_warning")
        high = (
            start.confidence_or_quality == "high"
            and end.confidence_or_quality == "high"
            and len(contralateral) == 1
            and 0.0 < duration <= config.same_side_max_interval_warning_seconds
        )
        strides.append(
            Stride(
                stride_id=f"S{index:04d}",
                side=start.side,
                start_event_id=start.event_id,
                end_event_id=end.event_id,
                start_frame=start.frame_index,
                end_frame=end.frame_index,
                start_timestamp_seconds=start.timestamp_seconds,
                end_timestamp_seconds=end.timestamp_seconds,
                duration_seconds=duration,
                quality="high" if high else "review",
                contralateral_event_id=contralateral[0].event_id
                if len(contralateral) == 1
                else None,
                contralateral_event_count=len(contralateral),
                sequence_notes=tuple(notes),
            )
        )
    return tuple(strides)
