"""Step 4 artifact workflow for video-derived candidate gait events."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import cv2

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
from gait_stability.pose_contracts import (
    CANONICAL_POSE_CONNECTIONS,
    MEDIAPIPE_LANDMARK_NAMES,
    normalized_to_pixel,
)
from gait_stability.pose_preprocessing import (
    PREPROCESSING_ALGORITHM_VERSION,
    PREPROCESSING_SCHEMA_VERSION,
    PROCESSED_FIELDS,
)
from gait_stability.video_ingestion import ArtifactPublishError, sha256_file

GAIT_EVENT_SCHEMA_VERSION = 1
GAIT_EVENT_ALGORITHM_VERSION = "step4-mvp-1"
STEP4_ARTIFACT_NAMES = (
    "walking_bout.json",
    "gait_events.csv",
    "strides.csv",
    "gait_event_diagnostic.png",
    "annotated_gait_events.mp4",
    "gait_event_metadata.json",
)
_INPUT_NAMES = (
    "processed_landmarks.csv",
    "pose_quality.json",
    "preprocessing_metadata.json",
)
_SIGNAL_LANDMARKS = (
    "left_hip",
    "right_hip",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
)
SelectionMethod = Literal["automatic", "manual", "full_recording_fallback"]


class GaitEventPipelineError(Exception):
    """Expected Step 4 input, processing, rendering, or artifact error."""


class GaitEventArtifactValidationError(GaitEventPipelineError):
    """Raised when canonical Step 3 artifacts violate their contracts."""


class GaitEventRenderError(GaitEventPipelineError):
    """Raised when a required diagnostic or annotated video cannot be rendered."""


@dataclass(frozen=True, slots=True)
class GaitEventPipelineConfig:
    """Validated Step 4 integration and pure-detector configuration."""

    event_config: GaitEventConfig
    manual_start_frame: int | None = None
    manual_end_frame: int | None = None
    automatic_minimum_bout_duration_seconds: float = 3.0
    automatic_minimum_accepted_events_per_side: int = 2
    event_flash_radius_frames: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.event_config, GaitEventConfig):
            raise TypeError("event_config must be a GaitEventConfig")
        manual_values = (self.manual_start_frame, self.manual_end_frame)
        if (manual_values[0] is None) != (manual_values[1] is None):
            raise ValueError(
                "manual_start_frame and manual_end_frame are both required"
            )
        for name, value in (
            ("manual_start_frame", self.manual_start_frame),
            ("manual_end_frame", self.manual_end_frame),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"{name} must be an integer or None")
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if (
            self.manual_start_frame is not None
            and self.manual_end_frame is not None
            and self.manual_end_frame < self.manual_start_frame
        ):
            raise ValueError("manual_end_frame must be at least manual_start_frame")
        duration = self.automatic_minimum_bout_duration_seconds
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise TypeError("automatic_minimum_bout_duration_seconds must be a number")
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(
                "automatic_minimum_bout_duration_seconds must be finite and positive"
            )
        for name in (
            "automatic_minimum_accepted_events_per_side",
            "event_flash_radius_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            minimum = 1 if name.startswith("automatic") else 0
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")


@dataclass(frozen=True, slots=True)
class GaitEventArtifacts:
    """Published Step 4 artifact paths."""

    artifact_directory: Path
    walking_bout_path: Path
    gait_events_path: Path
    strides_path: Path
    diagnostic_path: Path
    annotated_video_path: Path
    metadata_path: Path


@dataclass(frozen=True, slots=True)
class _LandmarkRow:
    frame_index: int
    timestamp: float
    landmark_id: int
    landmark_name: str
    raw_x: float | None
    raw_y: float | None
    processed_x: float | None
    processed_y: float | None
    x_observed_usable: bool
    x_interpolated: bool
    x_smoothing_support_contains_interpolation: bool


@dataclass(frozen=True, slots=True)
class _Bout:
    start_frame: int
    end_frame: int
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    selection_method: SelectionMethod
    selection_reason: str
    quality: str
    limitations: tuple[str, ...]
    candidates: tuple[dict[str, Any], ...]


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise GaitEventArtifactValidationError(
            f"Required Step 3 artifact is missing: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GaitEventArtifactValidationError(
            f"Could not read valid {label}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise GaitEventArtifactValidationError(f"{label} root must be an object")
    return value


def _parse_int(text: str, field: str, row_number: int) -> int:
    try:
        return int(text)
    except ValueError as exc:
        raise GaitEventArtifactValidationError(
            f"processed_landmarks.csv row {row_number}: {field} must be an integer"
        ) from exc


def _parse_float(
    text: str,
    field: str,
    row_number: int,
    *,
    nullable: bool = False,
    allow_nonfinite: bool = False,
) -> float | None:
    if nullable and text == "":
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise GaitEventArtifactValidationError(
            f"processed_landmarks.csv row {row_number}: {field} must be numeric"
        ) from exc
    if not allow_nonfinite and not math.isfinite(value):
        raise GaitEventArtifactValidationError(
            f"processed_landmarks.csv row {row_number}: {field} must be finite"
        )
    return value


def _parse_bool(text: str, field: str, row_number: int) -> bool:
    if text not in {"true", "false"}:
        raise GaitEventArtifactValidationError(
            f"processed_landmarks.csv row {row_number}: {field} must be true or false"
        )
    return text == "true"


def _read_processed(
    path: Path,
) -> tuple[dict[int, dict[str, _LandmarkRow]], tuple[float, ...]]:
    if not path.is_file():
        raise GaitEventArtifactValidationError(
            f"Required Step 3 artifact is missing: {path}"
        )
    boolean_fields = {
        "raw_row_present",
        "x_observed_usable",
        "y_observed_usable",
        "observed_usable",
        "rejected_low_confidence",
        "missing_or_nonfinite_enabled_score",
        "nonfinite_x_coordinate",
        "nonfinite_y_coordinate",
        "out_of_image_x",
        "out_of_image_y",
        "x_interpolated",
        "y_interpolated",
        "x_smoothing_changed",
        "y_smoothing_changed",
        "x_smoothing_support_contains_interpolation",
        "y_smoothing_support_contains_interpolation",
        "x_final_missing",
        "y_final_missing",
        "final_missing",
    }
    nullable_numeric_fields = {
        "raw_x_normalized",
        "raw_y_normalized",
        "raw_z_backend_relative",
        "visibility",
        "presence",
        "confidence",
        "pre_smoothed_x_normalized",
        "pre_smoothed_y_normalized",
        "processed_x_normalized",
        "processed_y_normalized",
    }
    grid: dict[int, dict[str, _LandmarkRow]] = {}
    timestamps: list[float] = []
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if tuple(reader.fieldnames or ()) != PROCESSED_FIELDS:
                raise GaitEventArtifactValidationError(
                    "processed_landmarks.csv schema must exactly equal PROCESSED_FIELDS"
                )
            expected_position = 0
            canonical_count = len(MEDIAPIPE_LANDMARK_NAMES)
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(row[field] is None for field in PROCESSED_FIELDS):
                    raise GaitEventArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: malformed columns"
                    )
                frame_index = _parse_int(row["frame_index"], "frame_index", row_number)
                landmark_id = _parse_int(row["landmark_id"], "landmark_id", row_number)
                expected_frame, expected_id = divmod(expected_position, canonical_count)
                if frame_index != expected_frame or landmark_id != expected_id:
                    raise GaitEventArtifactValidationError(
                        "processed_landmarks.csv must be an ordered complete "
                        "nominal-frame "
                        "x canonical-landmark grid starting at frame 0"
                    )
                expected_name = MEDIAPIPE_LANDMARK_NAMES[landmark_id]
                if row["landmark_name"] != expected_name:
                    raise GaitEventArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: landmark "
                        "ID/name mismatch"
                    )
                timestamp = _parse_float(
                    row["nominal_timestamp_seconds"],
                    "nominal_timestamp_seconds",
                    row_number,
                )
                assert timestamp is not None
                if timestamp < 0.0:
                    raise GaitEventArtifactValidationError(
                        "nominal timestamps must be nonnegative"
                    )
                if expected_id == 0:
                    if timestamps and timestamp <= timestamps[-1]:
                        raise GaitEventArtifactValidationError(
                            "per-frame nominal timestamps must be strictly increasing"
                        )
                    timestamps.append(timestamp)
                elif timestamp != timestamps[-1]:
                    raise GaitEventArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: timestamp "
                        "differs within frame"
                    )
                parsed_booleans = {
                    field: _parse_bool(row[field], field, row_number)
                    for field in boolean_fields
                }
                parsed_numeric = {
                    field: _parse_float(
                        row[field],
                        field,
                        row_number,
                        nullable=True,
                        allow_nonfinite=field
                        in {
                            "raw_x_normalized",
                            "raw_y_normalized",
                            "raw_z_backend_relative",
                            "visibility",
                            "presence",
                            "confidence",
                        },
                    )
                    for field in nullable_numeric_fields
                }
                processed_x = parsed_numeric["processed_x_normalized"]
                raw_x = parsed_numeric["raw_x_normalized"]
                raw_y = parsed_numeric["raw_y_normalized"]
                x_observed_usable = parsed_booleans["x_observed_usable"]
                y_observed_usable = parsed_booleans["y_observed_usable"]
                observed_usable = parsed_booleans["observed_usable"]
                if x_observed_usable and (raw_x is None or not math.isfinite(raw_x)):
                    raise GaitEventArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: observed usable x "
                        "requires a finite raw x value"
                    )
                if y_observed_usable and (raw_y is None or not math.isfinite(raw_y)):
                    raise GaitEventArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: observed usable y "
                        "requires a finite raw y value"
                    )
                if observed_usable != (x_observed_usable and y_observed_usable):
                    raise GaitEventArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: observed_usable "
                        "conflicts with axis observed usability"
                    )
                x_final_missing = parsed_booleans["x_final_missing"]
                y_final_missing = parsed_booleans["y_final_missing"]
                final_missing = parsed_booleans["final_missing"]
                processed_y = parsed_numeric["processed_y_normalized"]
                if x_final_missing != (processed_x is None):
                    raise GaitEventArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        "x_final_missing conflicts"
                    )
                if y_final_missing != (processed_y is None):
                    raise GaitEventArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        "y_final_missing conflicts"
                    )
                if final_missing != (x_final_missing or y_final_missing):
                    raise GaitEventArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: final_missing "
                        "conflicts"
                    )
                grid.setdefault(frame_index, {})[expected_name] = _LandmarkRow(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    landmark_id=landmark_id,
                    landmark_name=expected_name,
                    raw_x=raw_x,
                    raw_y=raw_y,
                    processed_x=processed_x,
                    processed_y=processed_y,
                    x_observed_usable=x_observed_usable,
                    x_interpolated=parsed_booleans["x_interpolated"],
                    x_smoothing_support_contains_interpolation=parsed_booleans[
                        "x_smoothing_support_contains_interpolation"
                    ],
                )
                expected_position += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        raise GaitEventArtifactValidationError(f"Could not read {path}: {exc}") from exc
    if not grid or expected_position % len(MEDIAPIPE_LANDMARK_NAMES):
        raise GaitEventArtifactValidationError(
            "processed_landmarks.csv must contain at least one complete canonical frame"
        )
    for name in _SIGNAL_LANDMARKS:
        if any(name not in frame for frame in grid.values()):
            raise GaitEventArtifactValidationError(
                f"Required signal landmark is missing: {name}"
            )
    return grid, tuple(timestamps)


def _validate_metadata(
    metadata: dict[str, Any],
    quality: dict[str, Any],
    processed_hash: str,
    frame_count: int,
) -> dict[str, Any]:
    if quality.get("schema_version") != PREPROCESSING_SCHEMA_VERSION:
        raise GaitEventArtifactValidationError(
            "pose_quality.json schema_version must be exactly 1"
        )
    if metadata.get("schema_version") != PREPROCESSING_SCHEMA_VERSION:
        raise GaitEventArtifactValidationError(
            "preprocessing_metadata.json schema_version must be exactly 1"
        )
    if metadata.get("algorithm_version") != PREPROCESSING_ALGORITHM_VERSION:
        raise GaitEventArtifactValidationError(
            "preprocessing_metadata.json must identify the Step 3 algorithm"
        )
    total_frames = quality.get("total_frames")
    if (
        isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or total_frames != frame_count
    ):
        raise GaitEventArtifactValidationError(
            "pose_quality.json total_frames must match the processed grid"
        )
    required = quality.get("required_gait_landmarks")
    if not isinstance(required, list) or not all(
        isinstance(name, str) for name in required
    ):
        raise GaitEventArtifactValidationError(
            "pose_quality.json required_gait_landmarks must be a string array"
        )
    per_landmark = quality.get("per_landmark")
    if not isinstance(per_landmark, dict):
        raise GaitEventArtifactValidationError(
            "pose_quality.json per_landmark must be an object"
        )
    outputs = metadata.get("outputs")
    if not isinstance(outputs, dict):
        raise GaitEventArtifactValidationError(
            "preprocessing metadata outputs must be an object"
        )
    processed = outputs.get("processed_landmarks.csv")
    if not isinstance(processed, dict) or processed.get("sha256") != processed_hash:
        raise GaitEventArtifactValidationError(
            "preprocessing metadata processed_landmarks.csv hash does not match "
            "current file"
        )
    path = processed.get("path")
    if not isinstance(path, str) or Path(path).name != "processed_landmarks.csv":
        raise GaitEventArtifactValidationError(
            "preprocessing metadata does not identify canonical processed_landmarks.csv"
        )
    inherited = metadata.get("inherited_provenance")
    source = inherited.get("source") if isinstance(inherited, dict) else None
    if not isinstance(source, dict):
        raise GaitEventArtifactValidationError(
            "Step 3 inherited source provenance is missing"
        )
    inherited_hash = source.get("sha256")
    if not isinstance(inherited_hash, str) or not inherited_hash:
        raise GaitEventArtifactValidationError(
            "Step 3 inherited source provenance sha256 hash must be a nonempty string"
        )
    return source


def _build_samples(
    grid: Mapping[int, Mapping[str, _LandmarkRow]],
) -> dict[str, tuple[SignalSample, ...]]:
    result: dict[str, tuple[SignalSample, ...]] = {}
    for side in ("left", "right"):
        samples: list[SignalSample] = []
        for frame_index, frame in grid.items():
            heel = frame[f"{side}_heel"]
            left_hip = frame["left_hip"]
            right_hip = frame["right_hip"]
            ankle = frame[f"{side}_ankle"]
            primary = (heel, left_hip, right_hip)
            hips = (left_hip, right_hip)
            samples.append(
                SignalSample(
                    frame_index=frame_index,
                    nominal_timestamp_seconds=heel.timestamp,
                    processed_heel_x=heel.processed_x,
                    processed_hip_left_x=left_hip.processed_x,
                    processed_hip_right_x=right_hip.processed_x,
                    raw_heel_x=heel.raw_x
                    if all(point.x_observed_usable for point in primary)
                    else None,
                    raw_hip_left_x=left_hip.raw_x
                    if all(point.x_observed_usable for point in primary)
                    else None,
                    raw_hip_right_x=right_hip.raw_x
                    if all(point.x_observed_usable for point in primary)
                    else None,
                    processed_ankle_x=ankle.processed_x,
                    raw_ankle_x=ankle.raw_x if ankle.x_observed_usable else None,
                    primary_observed_usable=all(
                        point.x_observed_usable for point in primary
                    ),
                    primary_interpolated=any(point.x_interpolated for point in primary),
                    primary_smoothing_support_contains_interpolation=any(
                        point.x_smoothing_support_contains_interpolation
                        for point in primary
                    ),
                    ankle_observed_usable=ankle.x_observed_usable,
                    ankle_interpolated=ankle.x_interpolated,
                    ankle_smoothing_support_contains_interpolation=(
                        ankle.x_smoothing_support_contains_interpolation
                    ),
                    hip_observed_usable=all(point.x_observed_usable for point in hips),
                    hip_interpolated=any(point.x_interpolated for point in hips),
                    hip_smoothing_support_contains_interpolation=any(
                        point.x_smoothing_support_contains_interpolation
                        for point in hips
                    ),
                )
            )
        result[side] = tuple(samples)
    return result


def _processed_primary_complete(
    samples: Mapping[str, Sequence[SignalSample]], index: int
) -> bool:
    for side in ("left", "right"):
        sample = samples[side][index]
        if (
            sample.processed_heel_x is None
            or sample.processed_hip_left_x is None
            or sample.processed_hip_right_x is None
        ):
            return False
    return True


def _samples_interval(
    samples: Mapping[str, Sequence[SignalSample]], start: int, end: int
) -> dict[str, tuple[SignalSample, ...]]:
    return {
        side: tuple(
            sample for sample in samples[side] if start <= sample.frame_index <= end
        )
        for side in ("left", "right")
    }


def _select_bout(
    samples: Mapping[str, Sequence[SignalSample]],
    timestamps: Sequence[float],
    config: GaitEventPipelineConfig,
) -> _Bout:
    last_frame = len(timestamps) - 1
    if config.manual_start_frame is not None and config.manual_end_frame is not None:
        if config.manual_end_frame > last_frame:
            raise GaitEventArtifactValidationError(
                f"manual interval exceeds final nominal frame {last_frame}"
            )
        return _Bout(
            config.manual_start_frame,
            config.manual_end_frame,
            timestamps[config.manual_start_frame],
            timestamps[config.manual_end_frame],
            "manual",
            "Inclusive analysis interval explicitly selected by the user.",
            "user_selected_not_objectively_classified_as_walking",
            (
                "Manual boundaries do not prove walking or steady-state gait; manual "
                "confirmation is required before downstream interpretation.",
            ),
            (),
        )
    candidates: list[dict[str, Any]] = []
    index = 0
    while index <= last_frame:
        if not _processed_primary_complete(samples, index):
            index += 1
            continue
        start = index
        while index <= last_frame and _processed_primary_complete(samples, index):
            index += 1
        end = index - 1
        duration = timestamps[end] - timestamps[start]
        if duration < config.automatic_minimum_bout_duration_seconds:
            continue
        preliminary = detect_candidate_events(
            _samples_interval(samples, start, end), config.event_config, start, end
        )
        accepted = Counter(
            event.side for event in preliminary if event.detection_status == "accepted"
        )
        if all(
            accepted[side] >= config.automatic_minimum_accepted_events_per_side
            for side in ("left", "right")
        ):
            candidates.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "start_timestamp_seconds": timestamps[start],
                    "end_timestamp_seconds": timestamps[end],
                    "duration_seconds": duration,
                    "accepted_preliminary_events": {
                        "left": accepted["left"],
                        "right": accepted["right"],
                        "total": accepted["left"] + accepted["right"],
                    },
                    "boundary_semantics": (
                        "maximal contiguous interval with complete bilateral primary "
                        "processed signals and candidate-count evidence only; manual "
                        "confirmation is required before downstream interpretation"
                    ),
                }
            )
    if candidates:
        selected = min(
            candidates,
            key=lambda candidate: (
                -int(candidate["accepted_preliminary_events"]["total"]),
                -float(candidate["duration_seconds"]),
                int(candidate["start_frame"]),
            ),
        )
        return _Bout(
            int(selected["start_frame"]),
            int(selected["end_frame"]),
            float(selected["start_timestamp_seconds"]),
            float(selected["end_timestamp_seconds"]),
            "automatic",
            (
                "Selected qualifying complete-primary-signal candidate by accepted "
                "preliminary candidate count, then duration, then earlier start; "
                "manual confirmation is required before downstream interpretation."
            ),
            "complete_primary_signal_interval_with_minimum_candidate_count",
            (
                "Boundaries reflect complete primary signals and minimum candidate-"
                "count evidence only, not periodicity, physical walking onset, offset, "
                "or steady-state classification; manual confirmation is required "
                "before downstream interpretation.",
            ),
            tuple(candidates),
        )
    return _Bout(
        0,
        last_frame,
        timestamps[0],
        timestamps[last_frame],
        "full_recording_fallback",
        (
            "No maximal complete-signal run met duration and bilateral preliminary "
            "event-count requirements; the full stored nominal range is retained."
        ),
        "walking_boundaries_not_inferred_or_objectively_classified",
        (
            "The full interval is not evidence that every frame contains walking.",
            "Manual confirmation is required before downstream interpretation.",
            "No terminal sample or boundary timestamp was synthesized.",
        ),
        (),
    )


def _bout_payload(bout: _Bout) -> dict[str, Any]:
    return {
        "schema_version": GAIT_EVENT_SCHEMA_VERSION,
        "start_frame": bout.start_frame,
        "end_frame": bout.end_frame,
        "start_timestamp_seconds": bout.start_timestamp_seconds,
        "end_timestamp_seconds": bout.end_timestamp_seconds,
        "selection_method": bout.selection_method,
        "selection_reason": bout.selection_reason,
        "quality": bout.quality,
        "limitations": list(bout.limitations),
        "candidate_bouts": list(bout.candidates),
        "boundary_inclusivity": "start and end frames/timestamps are inclusive",
    }


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return "|".join(str(item) for item in value)
    return value


def _write_csv(
    path: Path, rows: Sequence[GaitEvent] | Sequence[Stride], fields: tuple[str, ...]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _csv_value(getattr(row, field)) for field in fields}
            )


def _relative_signal(
    sample: SignalSample, direction: str, *, raw: bool = False
) -> float:
    heel = sample.raw_heel_x if raw else sample.processed_heel_x
    left = sample.raw_hip_left_x if raw else sample.processed_hip_left_x
    right = sample.raw_hip_right_x if raw else sample.processed_hip_right_x
    if heel is None or left is None or right is None:
        return math.nan
    sign = 1.0 if direction == "image_right" else -1.0
    return sign * (heel - (left + right) / 2.0)


def _write_diagnostic(
    path: Path,
    samples: Mapping[str, Sequence[SignalSample]],
    events: Sequence[GaitEvent],
    bout: _Bout,
    direction: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise GaitEventRenderError("Step 4 diagnostic requires matplotlib") from exc
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    try:
        for axis, side, color in zip(
            axes, ("left", "right"), ("tab:blue", "tab:orange"), strict=True
        ):
            side_samples = samples[side]
            times = [sample.nominal_timestamp_seconds for sample in side_samples]
            processed = [_relative_signal(sample, direction) for sample in side_samples]
            raw = [
                _relative_signal(sample, direction, raw=True) for sample in side_samples
            ]
            axis.plot(
                times, processed, color=color, linewidth=1.5, label=f"{side} processed"
            )
            if any(math.isfinite(value) for value in raw):
                axis.plot(
                    times, raw, color="0.55", linewidth=0.8, alpha=0.7, label="raw cue"
                )
            affected = [
                sample.primary_interpolated
                or sample.primary_smoothing_support_contains_interpolation
                for sample in side_samples
            ]
            axis.scatter(
                [time for time, flag in zip(times, affected, strict=True) if flag],
                [
                    value
                    for value, flag in zip(processed, affected, strict=True)
                    if flag
                ],
                color="magenta",
                marker=".",
                s=18,
                label="primary interpolation affected",
                zorder=4,
            )
            for event in (event for event in events if event.side == side):
                marker = "x" if event.detection_status == "rejected_candidate" else "o"
                event_color = (
                    "green"
                    if event.confidence_or_quality == "high"
                    else "goldenrod"
                    if event.detection_status == "accepted"
                    else "red"
                )
                axis.scatter(
                    [event.timestamp_seconds],
                    [event.peak_value],
                    marker=marker,
                    color=event_color,
                    s=55,
                    zorder=5,
                )
            axis.axvline(
                bout.start_timestamp_seconds,
                color="black",
                linestyle="--",
                label="bout boundary",
            )
            axis.axvline(bout.end_timestamp_seconds, color="black", linestyle="--")
            axis.set_ylabel("pelvis-relative heel x\n(normalized image width)")
            axis.set_title(
                f"{side.title()} video-derived candidate initial contact signal"
            )
            axis.grid(alpha=0.2)
            axis.legend(loc="upper right", fontsize="small")
        axes[-1].set_xlabel("nominal timestamp (s)")
        figure.suptitle("Direction-normalized pelvis-proxy-relative heel trajectories")
        figure.tight_layout()
        figure.savefig(path, dpi=140)
    except OSError as exc:
        raise GaitEventRenderError(f"Could not write diagnostic PNG: {exc}") from exc
    finally:
        plt.close(figure)


def _source_video(
    source_provenance: Mapping[str, Any], override: str | Path | None
) -> tuple[Path, str]:
    if override is None:
        value = source_provenance.get("path_identifier")
        if not isinstance(value, str) or not value:
            raise GaitEventArtifactValidationError(
                "Inherited source video path is missing"
            )
        return Path(value).expanduser().resolve(), "inherited"
    return Path(override).expanduser().resolve(), "override"


def _video_properties(source: Mapping[str, Any]) -> tuple[int, int, float]:
    width = source.get("width_pixels")
    height = source.get("height_pixels")
    fps = source.get("nominal_fps")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0.0
    ):
        raise GaitEventArtifactValidationError(
            "Inherited source width, height, and nominal_fps must be valid"
        )
    return width, height, float(fps)


def _draw_pose(
    frame: Any, rows: Mapping[str, _LandmarkRow], width: int, height: int
) -> None:
    points: dict[int, tuple[int, int]] = {}
    for row in rows.values():
        if row.processed_x is not None and row.processed_y is not None:
            points[row.landmark_id] = (
                normalized_to_pixel(row.processed_x, width),
                normalized_to_pixel(row.processed_y, height),
            )
    for first, second in CANONICAL_POSE_CONNECTIONS:
        if first in points and second in points:
            cv2.line(
                frame, points[first], points[second], (80, 220, 80), 2, cv2.LINE_AA
            )
    for point in points.values():
        cv2.circle(frame, point, 3, (255, 255, 255), -1, cv2.LINE_AA)


def _current_stride_ids(strides: Sequence[Stride], frame_index: int) -> dict[str, str]:
    return {
        stride.side: stride.stride_id
        for stride in strides
        if stride.start_frame <= frame_index <= stride.end_frame
    }


def _write_annotated_video(
    path: Path,
    video_path: Path,
    grid: Mapping[int, Mapping[str, _LandmarkRow]],
    timestamps: Sequence[float],
    events: Sequence[GaitEvent],
    strides: Sequence[Stride],
    bout: _Bout,
    width: int,
    height: int,
    fps: float,
    flash_radius: int,
) -> None:
    if not video_path.is_file():
        raise GaitEventRenderError(f"Source video is not a readable file: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise GaitEventRenderError(f"Could not open source video: {video_path}")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise GaitEventRenderError(f"Could not open annotated video writer: {path}")
    try:
        for frame_index, timestamp in enumerate(timestamps):
            ok, decoded = capture.read()
            if not ok or decoded is None:
                frame = __import__("numpy").zeros((height, width, 3), dtype="uint8")
                cv2.putText(
                    frame,
                    "SOURCE DECODE FAILURE PLACEHOLDER",
                    (20, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                frame = (
                    cv2.resize(decoded, (width, height))
                    if decoded.shape[1::-1] != (width, height)
                    else decoded
                )
            _draw_pose(frame, grid[frame_index], width, height)
            in_bout = bout.start_frame <= frame_index <= bout.end_frame
            lines = [
                "POSE-MODEL ESTIMATE",
                f"frame {frame_index} | nominal time {timestamp:.3f} s",
                "bout: "
                + ("IN SELECTED INTERVAL" if in_bout else "OUTSIDE SELECTED INTERVAL"),
            ]
            stride_ids = _current_stride_ids(strides, frame_index)
            lines.append(
                f"candidate strides: L {stride_ids.get('left', '-')} | "
                f"R {stride_ids.get('right', '-')}"
            )
            flashes = [
                event
                for event in events
                if abs(event.frame_index - frame_index) <= flash_radius
            ]
            for event in flashes:
                lines.append(
                    f"{event.side.upper()} CANDIDATE INITIAL CONTACT | "
                    f"{event.confidence_or_quality} | {event.detection_status}"
                )
                heel = grid[frame_index][f"{event.side}_heel"]
                if heel.processed_x is not None and heel.processed_y is not None:
                    cv2.circle(
                        frame,
                        (
                            normalized_to_pixel(heel.processed_x, width),
                            normalized_to_pixel(heel.processed_y, height),
                        ),
                        12,
                        (0, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )
            panel_height = 30 + 28 * len(lines)
            cv2.rectangle(frame, (0, 0), (width, panel_height), (20, 20, 20), -1)
            for line_index, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (12, 27 + line_index * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
    except cv2.error as exc:
        raise GaitEventRenderError(f"Annotated video rendering failed: {exc}") from exc
    finally:
        capture.release()
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise GaitEventRenderError(
            "Annotated video writer produced no readable artifact"
        )


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("gait-stability", "opencv-contrib-python-headless", "matplotlib"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _nested_counts(events: Sequence[GaitEvent]) -> dict[str, Any]:
    counts: Counter[tuple[str, str, str, str]] = Counter(
        (
            event.side,
            event.event_type,
            event.detection_status,
            event.confidence_or_quality,
        )
        for event in events
    )
    result: dict[str, Any] = {}
    for (side, event_type, status, quality), count in sorted(counts.items()):
        result.setdefault(side, {}).setdefault(event_type, {}).setdefault(status, {})[
            quality
        ] = count
    return result


def _metadata_payload(
    directory: Path,
    staging: Path,
    config: GaitEventPipelineConfig,
    input_hashes: Mapping[Path, str],
    video_path: Path,
    video_hash: str,
    video_selection: str,
    source: Mapping[str, Any],
    bout: _Bout,
    events: Sequence[GaitEvent],
    strides: Sequence[Stride],
    run_id: str,
    created_at: str,
) -> dict[str, Any]:
    stride_counts = Counter((stride.side, stride.quality) for stride in strides)
    outputs = {
        name: {
            "path": str(directory / name),
            "sha256": None
            if name == "gait_event_metadata.json"
            else sha256_file(staging / name),
        }
        for name in STEP4_ARTIFACT_NAMES
    }
    outputs["gait_event_metadata.json"]["sha256_semantics"] = (
        "null because embedding a file's own digest changes that digest"
    )
    return {
        "schema_version": GAIT_EVENT_SCHEMA_VERSION,
        "algorithm_version": GAIT_EVENT_ALGORITHM_VERSION,
        "run_id": run_id,
        "created_at_utc": created_at,
        "scope": (
            "walking interval selection, candidate initial contact detection, "
            "stride construction, and review artifacts only"
        ),
        "inputs": {
            path.name: {"path": str(path), "sha256": digest}
            for path, digest in input_hashes.items()
        },
        "source_video": {
            "path": str(video_path),
            "sha256": video_hash,
            "selection": video_selection,
            "inherited_provenance": dict(source),
        },
        "walking_bout": _bout_payload(bout),
        "method": {
            "event_definition": (
                "video-derived candidate initial contact at a direction-normalized "
                "local maximum of ipsilateral heel x relative to the bilateral hip "
                "midpoint"
            ),
            "formula": (
                "s_side(t) = d * (heel_side_x(t) - (left_hip_x(t) + "
                "right_hip_x(t)) / 2), where d=+1 for image_right and -1 for "
                "image_left"
            ),
            "pelvis_proxy": "bilateral hip midpoint; it is not center of mass",
            "direction": "user-declared and not inferred",
            "coordinates": "normalized image-plane x/y pose-model estimates",
            "units": (
                "dimensionless normalized image width/height; signal and prominence "
                "use normalized image-width units; velocities use normalized image "
                "width per nominal second"
            ),
            "required_capture_context": (
                "near-sagittal view, static camera, and treadmill or compatible "
                "walking context; not automatically verified"
            ),
            "primary_landmarks": ["ipsilateral heel", "left hip", "right hip"],
            "support_landmark": (
                "ipsilateral ankle x relative to the bilateral hip midpoint local "
                "maximum; the ankle and both hips across its selected local-peak "
                "support must be raw-observed usable and free of interpolation "
                "influence to support high quality"
            ),
            "raw_cue": (
                "raw heel and both raw hips only when all three x coordinates are "
                "raw-observed usable; raw and ankle cues are correlated sensitivity "
                "support, not independent corroboration"
            ),
            "detector_smoothing": (
                "none; reuses Step 3 processed trajectories and raw sensitivity cue"
            ),
            "peak_plateau_rule": (
                "strict local peak over configured radius; equal plateau forms one "
                "candidate at its earlier midpoint"
            ),
            "prominence_rule": (
                "peak minus the larger adjacent minimum over the configured bilateral "
                "window; the default minimum 0.02 is 2% of image width, is capture- "
                "and framing-dependent, and is not transferable without evaluation"
            ),
            "reversal_rule": (
                "pre velocity above deadband and post velocity below negative deadband "
                "over configured half-window"
            ),
            "missing_rule": (
                "missing processed primary x splits signal segments; no event or "
                "terminal sample is fabricated"
            ),
            "bout_rule": (
                "manual inclusive range, otherwise qualifying maximal complete-primary-"
                "signal runs with minimum candidate-count evidence, otherwise full "
                "stored nominal range; automatic selection does not test periodicity "
                "and requires manual confirmation before downstream interpretation"
            ),
            "temporal_conflict_rule": (
                "iteratively rank conflicting candidates by clean support, "
                "clean cue count, greater forward peak value, prominence, then earlier "
                "frame, and reject the lower-ranked candidate; equality at interval "
                "thresholds passes; this ranking may bias nearby-peak selection"
            ),
            "opposite_side_gate": (
                "the default 0.15-second minimum is an ordinary-walking engineering "
                "assumption and can reject near-simultaneous contacts"
            ),
            "quality_rule": (
                "high requires accepted, all primary support raw-observed usable, no "
                "primary interpolation influence across the full peak, reversal, and "
                "prominence support, raw peak agreement, and clean ankle-plus-"
                "bilateral-hip local-peak cue support; accepted otherwise review; "
                "rejected low"
            ),
            "sequence_regularity": (
                "nonalternation and long same-side intervals are QC context, not "
                "signal "
                "confidence or calibrated accuracy"
            ),
            "stride_rule": (
                "consecutive accepted same-side candidate initial contacts only; no "
                "partial or fabricated stride rows; high means bounded by two accepted "
                "same-side candidates plus current QC, not proof that no cycle was "
                "missed"
            ),
            "toe_off": "omitted",
            "phase_metrics": "no stance, swing, or double-support metrics are produced",
            "event_id_stability": (
                "deterministic only for the same inputs, configuration, and algorithm; "
                "not stable across parameter or algorithm changes"
            ),
        },
        "config": {
            "pipeline": {
                "manual_start_frame": config.manual_start_frame,
                "manual_end_frame": config.manual_end_frame,
                "automatic_minimum_bout_duration_seconds": (
                    config.automatic_minimum_bout_duration_seconds
                ),
                "automatic_minimum_accepted_events_per_side": (
                    config.automatic_minimum_accepted_events_per_side
                ),
                "event_flash_radius_frames": config.event_flash_radius_frames,
            },
            "detector": asdict(config.event_config),
        },
        "counts": {
            "events_by_side_type_status_quality": _nested_counts(events),
            "complete_strides_by_side_quality": {
                side: {
                    quality: stride_counts[(side, quality)]
                    for quality in ("high", "review", "low")
                }
                for side in ("left", "right")
            },
        },
        "timing": {
            "semantics": (
                "exact stored nominal timestamps from Step 3; not verified "
                "presentation timestamps"
            ),
            "bout_boundaries": "inclusive; no terminal sample synthesized",
        },
        "schemas": {
            "walking_bout.json": list(_bout_payload(bout)),
            "gait_events.csv": list(GAIT_EVENT_FIELDS),
            "strides.csv": list(STRIDE_FIELDS),
            "csv_tuple_serialization": "vertical-bar joined in stable tuple order",
            "csv_boolean_serialization": "lowercase true/false",
            "csv_null_serialization": "blank",
        },
        "renderer": {
            "video": (
                "every nominal frame slot 0 through final Step 3 frame; sequential "
                "source decode failure becomes a black labeled placeholder; nominal "
                "inherited FPS/dimensions; no audio"
            ),
            "pose": (
                "processed x/y canonical pose-model skeleton; coordinates are not "
                "clipped"
            ),
            "event_flash_radius_frames": config.event_flash_radius_frames,
        },
        "outputs": outputs,
        "runtime": {
            "python_version": sys.version,
            "dependency_versions": _dependency_versions(),
            "git": _git_provenance(),
            "randomness": "none",
        },
        "limitations": [
            "Candidate initial contacts are not force-plate-confirmed contacts and "
            "have not been validated for this pipeline or recording.",
            "MediaPipe landmarks are pose-model estimates, not anatomical joint "
            "centers or ground-contact markers.",
            "Camera view, static placement, treadmill context, direction, and "
            "mirroring are not automatically verified.",
            "Step 3 interpolation and smoothing can alter extrema and event timing.",
            "Monocular normalized coordinates have no physical scale or laboratory "
            "frame.",
            "High is deterministic algorithmic support, not probability, accuracy, "
            "or validation.",
            "Raw and ankle cues are correlated sensitivity support, not independent "
            "corroboration.",
            "No toe-off, stance, swing, double-support, COM, stability metric, "
            "clinical output, diagnosis, or fall-risk conclusion is produced.",
        ],
        "validation_status": (
            "software workflow only; no manual-label, marker, force, reference-system, "
            "participant-level, or clinical validation"
        ),
    }


def _hash_inputs(paths: Sequence[Path]) -> dict[Path, str]:
    try:
        return {path: sha256_file(path) for path in paths}
    except OSError as exc:
        raise GaitEventArtifactValidationError(
            f"Could not hash Step 3 input: {exc}"
        ) from exc


def _verify_hashes(expected: Mapping[Path, str]) -> None:
    current = _hash_inputs(tuple(expected))
    changed = sorted(
        path.name for path, digest in expected.items() if current[path] != digest
    )
    if changed:
        raise GaitEventArtifactValidationError(
            "Step 3 input artifact changed during Step 4 processing: "
            + ", ".join(changed)
        )


def _publish(staging: Path, destination: Path) -> None:
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for name in STEP4_ARTIFACT_NAMES:
            target = destination / name
            if target.exists():
                backup = destination / f"{name}.backup-{uuid.uuid4().hex}"
                target.replace(backup)
                backups[target] = backup
        for name in STEP4_ARTIFACT_NAMES:
            staged = staging / name
            if not staged.is_file():
                raise OSError(f"staged Step 4 artifact is missing: {staged}")
            target = destination / name
            staged.replace(target)
            published.append(target)
    except OSError as exc:
        failures: list[str] = []
        for target in published:
            with suppress(OSError):
                target.unlink()
        for target, backup in backups.items():
            try:
                backup.replace(target)
            except OSError as restore_exc:
                failures.append(
                    f"could not restore {target} from {backup}: {restore_exc}"
                )
        detail = "; ".join(failures)
        raise ArtifactPublishError(
            "Could not publish complete Step 4 artifact set"
            + (f"; rollback may be incomplete: {detail}" if detail else "")
        ) from exc
    for backup in backups.values():
        with suppress(OSError):
            backup.unlink()


def detect_gait_events(
    artifact_directory: str | Path,
    config: GaitEventPipelineConfig,
    *,
    video_path: str | Path | None = None,
) -> GaitEventArtifacts:
    """Validate Step 3, create Step 4 artifacts, and publish the complete set."""

    if not isinstance(config, GaitEventPipelineConfig):
        raise TypeError("config must be a GaitEventPipelineConfig")
    directory = Path(artifact_directory).expanduser().resolve()
    if not directory.is_dir():
        raise GaitEventArtifactValidationError(
            f"Artifact directory does not exist: {directory}"
        )
    input_paths = tuple(directory / name for name in _INPUT_NAMES)
    input_hashes = _hash_inputs(input_paths)
    quality = _load_json(input_paths[1], "pose_quality.json")
    preprocessing_metadata = _load_json(input_paths[2], "preprocessing_metadata.json")
    grid, timestamps = _read_processed(input_paths[0])
    source = _validate_metadata(
        preprocessing_metadata,
        quality,
        input_hashes[input_paths[0]],
        len(timestamps),
    )
    samples = _build_samples(grid)
    bout = _select_bout(samples, timestamps, config)
    events = detect_candidate_events(
        _samples_interval(samples, bout.start_frame, bout.end_frame),
        config.event_config,
        bout.start_frame,
        bout.end_frame,
    )
    strides = construct_strides(events, config.event_config)
    selected_video, video_selection = _source_video(source, video_path)
    width, height, fps = _video_properties(source)
    try:
        video_hash = sha256_file(selected_video)
    except OSError as exc:
        raise GaitEventRenderError(
            f"Could not read/hash source video {selected_video}: {exc}"
        ) from exc
    inherited_hash = source.get("sha256")
    if inherited_hash != video_hash:
        raise GaitEventArtifactValidationError(
            f"{video_selection.title()} source video hash does not match inherited "
            "Step 3 provenance hash"
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f"{directory.name}.gait-events-staging-", dir=directory.parent
        )
    )
    run_id = uuid.uuid4().hex
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        (staging / "walking_bout.json").write_text(
            json.dumps(_bout_payload(bout), indent=2) + "\n", encoding="utf-8"
        )
        _write_csv(staging / "gait_events.csv", events, GAIT_EVENT_FIELDS)
        _write_csv(staging / "strides.csv", strides, STRIDE_FIELDS)
        _write_diagnostic(
            staging / "gait_event_diagnostic.png",
            samples,
            events,
            bout,
            config.event_config.direction,
        )
        _write_annotated_video(
            staging / "annotated_gait_events.mp4",
            selected_video,
            grid,
            timestamps,
            events,
            strides,
            bout,
            width,
            height,
            fps,
            config.event_flash_radius_frames,
        )
        metadata = _metadata_payload(
            directory,
            staging,
            config,
            input_hashes,
            selected_video,
            video_hash,
            video_selection,
            source,
            bout,
            events,
            strides,
            run_id,
            created_at,
        )
        (staging / "gait_event_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        _verify_hashes(input_hashes)
        try:
            if sha256_file(selected_video) != video_hash:
                raise GaitEventArtifactValidationError(
                    "Source video changed during Step 4 rendering"
                )
        except OSError as exc:
            raise GaitEventRenderError(
                f"Could not recheck source video before publication: {exc}"
            ) from exc
        _publish(staging, directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return GaitEventArtifacts(
        artifact_directory=directory,
        walking_bout_path=directory / "walking_bout.json",
        gait_events_path=directory / "gait_events.csv",
        strides_path=directory / "strides.csv",
        diagnostic_path=directory / "gait_event_diagnostic.png",
        annotated_video_path=directory / "annotated_gait_events.mp4",
        metadata_path=directory / "gait_event_metadata.json",
    )
