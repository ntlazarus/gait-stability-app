"""Quality assessment and minimal preprocessing for canonical raw pose CSVs."""

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
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gait_stability.pose_contracts import MEDIAPIPE_LANDMARK_NAMES, PoseFrameStatus
from gait_stability.pose_pipeline import FRAME_FIELDS, LANDMARK_FIELDS
from gait_stability.video_ingestion import ArtifactPublishError, sha256_file

PREPROCESSING_SCHEMA_VERSION = 1
PREPROCESSING_ALGORITHM_VERSION = "step3-mvp-1"
DIAGNOSTIC_NAME = "pose_trajectory_diagnostic.png"
STEP3_ARTIFACT_NAMES = (
    "processed_landmarks.csv",
    "pose_quality.json",
    "preprocessing_metadata.json",
    DIAGNOSTIC_NAME,
)
REQUIRED_GAIT_LANDMARKS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)
DEFAULT_DIAGNOSTIC_LANDMARKS = (
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_hip",
    "right_hip",
)
PROCESSED_FIELDS = (
    "frame_index",
    "nominal_timestamp_seconds",
    "frame_status",
    "landmark_id",
    "landmark_name",
    "raw_row_present",
    "raw_x_normalized",
    "raw_y_normalized",
    "raw_z_backend_relative",
    "visibility",
    "presence",
    "confidence",
    "x_observed_usable",
    "y_observed_usable",
    "observed_usable",
    "rejected_low_confidence",
    "missing_or_nonfinite_enabled_score",
    "nonfinite_x_coordinate",
    "nonfinite_y_coordinate",
    "out_of_image_x",
    "out_of_image_y",
    "pre_smoothed_x_normalized",
    "pre_smoothed_y_normalized",
    "processed_x_normalized",
    "processed_y_normalized",
    "x_interpolated",
    "y_interpolated",
    "x_smoothing_changed",
    "y_smoothing_changed",
    "x_smoothing_support_contains_interpolation",
    "y_smoothing_support_contains_interpolation",
    "x_final_missing",
    "y_final_missing",
    "final_missing",
)


class PosePreprocessingError(Exception):
    """Expected invalid-input, preprocessing, or artifact error."""


class PoseArtifactValidationError(PosePreprocessingError):
    """Raised when Step 2 artifacts violate their canonical contracts."""


@dataclass(frozen=True, slots=True)
class PosePreprocessingConfig:
    """Validated Step 3 quality-control and trajectory-processing settings."""

    visibility_threshold: float = 0.5
    presence_threshold: float = 0.5
    confidence_threshold: float = 0.5
    use_visibility: bool = True
    use_presence: bool = True
    use_confidence: bool = False
    max_gap_frames: int = 3
    smoothing_window_frames: int = 3
    diagnostic_landmarks: tuple[str, ...] = DEFAULT_DIAGNOSTIC_LANDMARKS
    write_diagnostic: bool = True

    def __post_init__(self) -> None:
        for name in (
            "use_visibility",
            "use_presence",
            "use_confidence",
            "write_diagnostic",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for name in (
            "visibility_threshold",
            "presence_threshold",
            "confidence_threshold",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if isinstance(self.max_gap_frames, bool) or not isinstance(
            self.max_gap_frames, int
        ):
            raise TypeError("max_gap_frames must be an integer")
        if self.max_gap_frames < 0:
            raise ValueError("max_gap_frames must be nonnegative")
        if isinstance(self.smoothing_window_frames, bool) or not isinstance(
            self.smoothing_window_frames, int
        ):
            raise TypeError("smoothing_window_frames must be an integer")
        if self.smoothing_window_frames < 1 or self.smoothing_window_frames % 2 == 0:
            raise ValueError("smoothing_window_frames must be a positive odd integer")
        if not isinstance(self.diagnostic_landmarks, tuple) or not all(
            isinstance(name, str) for name in self.diagnostic_landmarks
        ):
            raise TypeError("diagnostic_landmarks must be a tuple of strings")
        if not self.diagnostic_landmarks:
            raise ValueError("diagnostic_landmarks must not be empty")
        unknown = sorted(set(self.diagnostic_landmarks) - set(MEDIAPIPE_LANDMARK_NAMES))
        if unknown:
            raise ValueError(f"Unknown diagnostic landmark(s): {', '.join(unknown)}")
        if len(set(self.diagnostic_landmarks)) != len(self.diagnostic_landmarks):
            raise ValueError("diagnostic_landmarks must not contain duplicates")


@dataclass(frozen=True, slots=True)
class RawPoseArtifacts:
    """Paths comprising one Step 2 raw-pose artifact set."""

    artifact_directory: Path
    raw_landmarks_path: Path
    pose_frames_path: Path
    pose_metadata_path: Path

    @classmethod
    def from_directory(cls, directory: str | Path) -> RawPoseArtifacts:
        artifact_directory = Path(directory).expanduser().resolve()
        return cls(
            artifact_directory=artifact_directory,
            raw_landmarks_path=artifact_directory / "raw_landmarks.csv",
            pose_frames_path=artifact_directory / "pose_frames.csv",
            pose_metadata_path=artifact_directory / "pose_metadata.json",
        )


@dataclass(frozen=True, slots=True)
class PosePreprocessingArtifacts:
    """Published Step 3 artifact paths."""

    artifact_directory: Path
    processed_landmarks_path: Path
    pose_quality_path: Path
    preprocessing_metadata_path: Path
    diagnostic_path: Path | None


@dataclass(slots=True)
class _Frame:
    frame_index: int
    timestamp: float
    status: str
    landmark_count: int


@dataclass(slots=True)
class _RawLandmark:
    frame_index: int
    timestamp: float
    landmark_id: int
    landmark_name: str
    x: float
    y: float
    z: float | None
    visibility: float | None
    presence: float | None
    confidence: float | None


@dataclass(slots=True)
class _Point:
    frame: _Frame
    landmark_id: int
    landmark_name: str
    raw: _RawLandmark | None
    x_observed_usable: bool = False
    y_observed_usable: bool = False
    observed_usable: bool = False
    rejected_low_confidence: bool = False
    missing_or_nonfinite_enabled_score: bool = False
    nonfinite_x: bool = False
    nonfinite_y: bool = False
    out_of_image_x: bool = False
    out_of_image_y: bool = False
    pre_x: float | None = None
    pre_y: float | None = None
    processed_x: float | None = None
    processed_y: float | None = None
    x_interpolated: bool = False
    y_interpolated: bool = False
    x_smoothing_changed: bool = False
    y_smoothing_changed: bool = False
    x_support_interpolated: bool = False
    y_support_interpolated: bool = False


def _parse_int(text: str, field_name: str, path: Path, row_number: int) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise PoseArtifactValidationError(
            f"{path.name} row {row_number}: {field_name} must be an integer"
        ) from exc
    return value


def _parse_float(
    text: str, field_name: str, path: Path, row_number: int, *, nullable: bool = False
) -> float | None:
    if nullable and text == "":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise PoseArtifactValidationError(
            f"{path.name} row {row_number}: {field_name} must be numeric"
        ) from exc


def _read_csv(path: Path, expected_fields: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise PoseArtifactValidationError(
            f"Required Step 2 artifact is missing: {path}"
        )
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if tuple(reader.fieldnames or ()) != expected_fields:
                raise PoseArtifactValidationError(
                    f"{path.name} schema mismatch: expected {list(expected_fields)!r}, "
                    f"found {reader.fieldnames!r}"
                )
            rows: list[dict[str, str]] = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise PoseArtifactValidationError(
                        f"{path.name} row {row_number}: extra columns"
                    )
                missing_fields = [
                    field for field in expected_fields if row[field] is None
                ]
                if missing_fields:
                    raise PoseArtifactValidationError(
                        f"{path.name} row {row_number}: missing value for "
                        + ", ".join(missing_fields)
                    )
                rows.append(row)
            return rows
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PoseArtifactValidationError(f"Could not read {path}: {exc}") from exc


def _load_inputs(
    artifacts: RawPoseArtifacts,
) -> tuple[list[_Frame], dict[tuple[int, int], _RawLandmark], dict[str, Any]]:
    frame_rows = _read_csv(artifacts.pose_frames_path, FRAME_FIELDS)
    raw_rows = _read_csv(artifacts.raw_landmarks_path, LANDMARK_FIELDS)
    if not artifacts.pose_metadata_path.is_file():
        raise PoseArtifactValidationError(
            f"Required Step 2 artifact is missing: {artifacts.pose_metadata_path}"
        )
    try:
        metadata = json.loads(artifacts.pose_metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PoseArtifactValidationError(
            f"Could not read valid JSON from {artifacts.pose_metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise PoseArtifactValidationError("pose_metadata.json root must be an object")
    schema_version = metadata.get("schema_version")
    if schema_version != 2:
        raise PoseArtifactValidationError(
            "pose_metadata.json schema_version must be exactly 2"
        )
    outputs = metadata.get("outputs")
    if (
        not isinstance(outputs, dict)
        or outputs.get("raw_landmarks") != "raw_landmarks.csv"
    ):
        raise PoseArtifactValidationError(
            "pose_metadata.json does not identify canonical raw_landmarks.csv"
        )
    if outputs.get("frame_manifest") != "pose_frames.csv":
        raise PoseArtifactValidationError(
            "pose_metadata.json does not identify canonical pose_frames.csv"
        )
    schemas = metadata.get("schemas")
    if schemas is not None:
        if not isinstance(schemas, dict):
            raise PoseArtifactValidationError(
                "pose_metadata.json schemas must be an object when provided"
            )
        expected_schemas = {
            "raw_landmarks.csv": LANDMARK_FIELDS,
            "pose_frames.csv": FRAME_FIELDS,
        }
        for artifact_name, expected_columns in expected_schemas.items():
            embedded_schema = schemas.get(artifact_name)
            if embedded_schema is None:
                continue
            if not isinstance(embedded_schema, dict):
                raise PoseArtifactValidationError(
                    f"pose_metadata.json schema for {artifact_name} must be an object"
                )
            columns = embedded_schema.get("columns")
            if columns is not None and columns != list(expected_columns):
                raise PoseArtifactValidationError(
                    f"pose_metadata.json embedded columns conflict for {artifact_name}"
                )

    frames: list[_Frame] = []
    seen_indices: set[int] = set()
    statuses = {status.value for status in PoseFrameStatus}
    for row_number, row in enumerate(frame_rows, start=2):
        frame_index = _parse_int(
            row["frame_index"], "frame_index", artifacts.pose_frames_path, row_number
        )
        timestamp_value = _parse_float(
            row["nominal_timestamp_seconds"],
            "nominal_timestamp_seconds",
            artifacts.pose_frames_path,
            row_number,
        )
        assert timestamp_value is not None
        landmark_count = _parse_int(
            row["landmark_count"],
            "landmark_count",
            artifacts.pose_frames_path,
            row_number,
        )
        if frame_index < 0 or frame_index in seen_indices:
            raise PoseArtifactValidationError(
                f"pose_frames.csv row {row_number}: duplicate or negative frame_index"
            )
        if not math.isfinite(timestamp_value) or timestamp_value < 0:
            raise PoseArtifactValidationError(
                f"pose_frames.csv row {row_number}: timestamp must be finite and "
                "nonnegative"
            )
        if row["status"] not in statuses:
            raise PoseArtifactValidationError(
                f"pose_frames.csv row {row_number}: invalid status {row['status']!r}"
            )
        if landmark_count < 0:
            raise PoseArtifactValidationError(
                f"pose_frames.csv row {row_number}: landmark_count must be nonnegative"
            )
        seen_indices.add(frame_index)
        frames.append(
            _Frame(frame_index, timestamp_value, row["status"], landmark_count)
        )
    if not frames:
        raise PoseArtifactValidationError(
            "pose_frames.csv must contain at least one frame"
        )
    for position, frame in enumerate(frames):
        if frame.frame_index != position:
            raise PoseArtifactValidationError(
                "pose_frames.csv frame_index values must be ordered and contiguous "
                "from zero"
            )
        if position and frame.timestamp <= frames[position - 1].timestamp:
            raise PoseArtifactValidationError(
                "pose_frames.csv timestamps must be finite and strictly increasing"
            )

    frame_by_index = {frame.frame_index: frame for frame in frames}
    raw: dict[tuple[int, int], _RawLandmark] = {}
    raw_counts = {frame.frame_index: 0 for frame in frames}
    for row_number, row in enumerate(raw_rows, start=2):
        path = artifacts.raw_landmarks_path
        frame_index = _parse_int(row["frame_index"], "frame_index", path, row_number)
        landmark_id = _parse_int(row["landmark_id"], "landmark_id", path, row_number)
        timestamp_value = _parse_float(
            row["nominal_timestamp_seconds"],
            "nominal_timestamp_seconds",
            path,
            row_number,
        )
        assert timestamp_value is not None
        if frame_index not in frame_by_index:
            raise PoseArtifactValidationError(
                f"raw_landmarks.csv row {row_number}: frame_index has no manifest row"
            )
        if not 0 <= landmark_id < len(MEDIAPIPE_LANDMARK_NAMES):
            raise PoseArtifactValidationError(
                f"raw_landmarks.csv row {row_number}: landmark_id is not canonical"
            )
        expected_name = MEDIAPIPE_LANDMARK_NAMES[landmark_id]
        if row["landmark_name"] != expected_name:
            raise PoseArtifactValidationError(
                f"raw_landmarks.csv row {row_number}: landmark ID/name mismatch"
            )
        key = (frame_index, landmark_id)
        if key in raw:
            raise PoseArtifactValidationError(
                f"raw_landmarks.csv row {row_number}: duplicate frame/landmark row"
            )
        frame = frame_by_index[frame_index]
        if frame.status != PoseFrameStatus.DECODED_POSE:
            raise PoseArtifactValidationError(
                f"raw_landmarks.csv row {row_number}: raw row belongs to "
                f"{frame.status} frame"
            )
        if not math.isfinite(timestamp_value) or timestamp_value != frame.timestamp:
            raise PoseArtifactValidationError(
                f"raw_landmarks.csv row {row_number}: timestamp does not match "
                "frame manifest"
            )
        x = _parse_float(row["x_normalized"], "x_normalized", path, row_number)
        y = _parse_float(row["y_normalized"], "y_normalized", path, row_number)
        assert x is not None and y is not None
        raw[key] = _RawLandmark(
            frame_index,
            timestamp_value,
            landmark_id,
            expected_name,
            x,
            y,
            _parse_float(
                row["z_backend_relative"],
                "z_backend_relative",
                path,
                row_number,
                nullable=True,
            ),
            _parse_float(
                row["visibility"], "visibility", path, row_number, nullable=True
            ),
            _parse_float(row["presence"], "presence", path, row_number, nullable=True),
            _parse_float(
                row["confidence"], "confidence", path, row_number, nullable=True
            ),
        )
        raw_counts[frame_index] += 1
    for frame in frames:
        if raw_counts[frame.frame_index] != frame.landmark_count:
            raise PoseArtifactValidationError(
                f"Frame {frame.frame_index} landmark_count does not match raw rows"
            )
        if frame.status == PoseFrameStatus.DECODED_POSE and frame.landmark_count == 0:
            raise PoseArtifactValidationError(
                f"Frame {frame.frame_index} is decoded_pose but has no landmark rows"
            )
        if frame.status != PoseFrameStatus.DECODED_POSE and frame.landmark_count != 0:
            raise PoseArtifactValidationError(
                f"Frame {frame.frame_index} has landmarks but status is {frame.status}"
            )
    return frames, raw, metadata


def _enabled_scores(config: PosePreprocessingConfig) -> tuple[tuple[str, float], ...]:
    fields: list[tuple[str, float]] = []
    if config.use_visibility:
        fields.append(("visibility", config.visibility_threshold))
    if config.use_presence:
        fields.append(("presence", config.presence_threshold))
    if config.use_confidence:
        fields.append(("confidence", config.confidence_threshold))
    return tuple(fields)


def _build_grid(
    frames: list[_Frame],
    raw: dict[tuple[int, int], _RawLandmark],
    config: PosePreprocessingConfig,
) -> dict[str, list[_Point]]:
    grid: dict[str, list[_Point]] = {name: [] for name in MEDIAPIPE_LANDMARK_NAMES}
    enabled_scores = _enabled_scores(config)
    for landmark_id, landmark_name in enumerate(MEDIAPIPE_LANDMARK_NAMES):
        for frame in frames:
            raw_point = raw.get((frame.frame_index, landmark_id))
            point = _Point(frame, landmark_id, landmark_name, raw_point)
            if raw_point is not None:
                point.nonfinite_x = not math.isfinite(raw_point.x)
                point.nonfinite_y = not math.isfinite(raw_point.y)
                point.out_of_image_x = (
                    not point.nonfinite_x and not 0 <= raw_point.x <= 1
                )
                point.out_of_image_y = (
                    not point.nonfinite_y and not 0 <= raw_point.y <= 1
                )
                for score_name, threshold in enabled_scores:
                    score = getattr(raw_point, score_name)
                    if score is None or not math.isfinite(score):
                        point.missing_or_nonfinite_enabled_score = True
                    elif score < threshold:
                        point.rejected_low_confidence = True
                scores_usable = not (
                    point.missing_or_nonfinite_enabled_score
                    or point.rejected_low_confidence
                )
                point.x_observed_usable = scores_usable and not point.nonfinite_x
                point.y_observed_usable = scores_usable and not point.nonfinite_y
                point.observed_usable = (
                    point.x_observed_usable and point.y_observed_usable
                )
                if point.x_observed_usable:
                    point.pre_x = raw_point.x
                if point.y_observed_usable:
                    point.pre_y = raw_point.y
            grid[landmark_name].append(point)
    return grid


def _interpolate_coordinate(
    points: list[_Point],
    attribute: str,
    raw_usable_attribute: str,
    flag_attribute: str,
    max_gap: int,
) -> None:
    index = 0
    while index < len(points):
        if getattr(points[index], attribute) is not None:
            index += 1
            continue
        start = index
        while index < len(points) and getattr(points[index], attribute) is None:
            index += 1
        end = index
        if start == 0 or end == len(points) or end - start > max_gap:
            continue
        left = points[start - 1]
        right = points[end]
        if not getattr(left, raw_usable_attribute) or not getattr(
            right, raw_usable_attribute
        ):
            continue
        left_value = getattr(left, attribute)
        right_value = getattr(right, attribute)
        assert left_value is not None and right_value is not None
        duration = right.frame.timestamp - left.frame.timestamp
        for gap_index in range(start, end):
            fraction = (
                points[gap_index].frame.timestamp - left.frame.timestamp
            ) / duration
            setattr(
                points[gap_index],
                attribute,
                left_value + fraction * (right_value - left_value),
            )
            setattr(points[gap_index], flag_attribute, True)


def _smooth_coordinate(
    points: list[_Point],
    source_attribute: str,
    destination_attribute: str,
    interpolation_flag: str,
    changed_flag: str,
    support_flag: str,
    window: int,
) -> None:
    for point in points:
        setattr(point, destination_attribute, getattr(point, source_attribute))
    if window == 1:
        return
    index = 0
    radius_limit = window // 2
    while index < len(points):
        if getattr(points[index], source_attribute) is None:
            index += 1
            continue
        start = index
        while (
            index < len(points) and getattr(points[index], source_attribute) is not None
        ):
            index += 1
        end = index
        for center in range(start, end):
            radius = min(radius_limit, center - start, end - center - 1)
            support = points[center - radius : center + radius + 1]
            values = [getattr(point, source_attribute) for point in support]
            assert all(value is not None for value in values)
            smoothed = math.fsum(value for value in values if value is not None) / len(
                values
            )
            source = getattr(points[center], source_attribute)
            setattr(points[center], destination_attribute, smoothed)
            setattr(points[center], changed_flag, smoothed != source)
            setattr(
                points[center],
                support_flag,
                any(getattr(point, interpolation_flag) for point in support),
            )


def _process_grid(
    grid: dict[str, list[_Point]], config: PosePreprocessingConfig
) -> None:
    for points in grid.values():
        _interpolate_coordinate(
            points,
            "pre_x",
            "x_observed_usable",
            "x_interpolated",
            config.max_gap_frames,
        )
        _interpolate_coordinate(
            points,
            "pre_y",
            "y_observed_usable",
            "y_interpolated",
            config.max_gap_frames,
        )
        _smooth_coordinate(
            points,
            "pre_x",
            "processed_x",
            "x_interpolated",
            "x_smoothing_changed",
            "x_support_interpolated",
            config.smoothing_window_frames,
        )
        _smooth_coordinate(
            points,
            "pre_y",
            "processed_y",
            "y_interpolated",
            "y_smoothing_changed",
            "y_support_interpolated",
            config.smoothing_window_frames,
        )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _point_row(point: _Point) -> dict[str, Any]:
    raw = point.raw
    return {
        "frame_index": point.frame.frame_index,
        "nominal_timestamp_seconds": point.frame.timestamp,
        "frame_status": point.frame.status,
        "landmark_id": point.landmark_id,
        "landmark_name": point.landmark_name,
        "raw_row_present": raw is not None,
        "raw_x_normalized": None if raw is None else raw.x,
        "raw_y_normalized": None if raw is None else raw.y,
        "raw_z_backend_relative": None if raw is None else raw.z,
        "visibility": None if raw is None else raw.visibility,
        "presence": None if raw is None else raw.presence,
        "confidence": None if raw is None else raw.confidence,
        "x_observed_usable": point.x_observed_usable,
        "y_observed_usable": point.y_observed_usable,
        "observed_usable": point.observed_usable,
        "rejected_low_confidence": point.rejected_low_confidence,
        "missing_or_nonfinite_enabled_score": point.missing_or_nonfinite_enabled_score,
        "nonfinite_x_coordinate": point.nonfinite_x,
        "nonfinite_y_coordinate": point.nonfinite_y,
        "out_of_image_x": point.out_of_image_x,
        "out_of_image_y": point.out_of_image_y,
        "pre_smoothed_x_normalized": point.pre_x,
        "pre_smoothed_y_normalized": point.pre_y,
        "processed_x_normalized": point.processed_x,
        "processed_y_normalized": point.processed_y,
        "x_interpolated": point.x_interpolated,
        "y_interpolated": point.y_interpolated,
        "x_smoothing_changed": point.x_smoothing_changed,
        "y_smoothing_changed": point.y_smoothing_changed,
        "x_smoothing_support_contains_interpolation": point.x_support_interpolated,
        "y_smoothing_support_contains_interpolation": point.y_support_interpolated,
        "x_final_missing": point.processed_x is None,
        "y_final_missing": point.processed_y is None,
        "final_missing": point.processed_x is None or point.processed_y is None,
    }


def _runs(points: list[_Point], selected: Iterable[bool]) -> list[dict[str, Any]]:
    selected_values = list(selected)
    runs: list[dict[str, Any]] = []
    index = 0
    while index < len(points):
        if not selected_values[index]:
            index += 1
            continue
        start = index
        while index < len(points) and selected_values[index]:
            index += 1
        end = index - 1
        bracketing_duration: float | None = None
        if start > 0 and end + 1 < len(points):
            bracketing_duration = (
                points[end + 1].frame.timestamp - points[start - 1].frame.timestamp
            )
        runs.append(
            {
                "start_frame_index": points[start].frame.frame_index,
                "end_frame_index": points[end].frame.frame_index,
                "start_timestamp_seconds": points[start].frame.timestamp,
                "end_timestamp_seconds": points[end].frame.timestamp,
                "missing_sample_count": end - start + 1,
                "nominal_sample_span_seconds": (
                    points[end].frame.timestamp - points[start].frame.timestamp
                ),
                "bracketing_duration_seconds": bracketing_duration,
            }
        )
    return runs


def _summarize_values(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "available_count": len(values),
        "finite_count": len(finite),
        "nonfinite_count": len(values) - len(finite),
        "minimum": min(finite) if finite else None,
        "maximum": max(finite) if finite else None,
        "mean": math.fsum(finite) / len(finite) if finite else None,
    }


def _gap_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    nominal_spans = [run["nominal_sample_span_seconds"] for run in runs]
    bracketing_durations = [
        duration
        for run in runs
        if (duration := run["bracketing_duration_seconds"]) is not None
    ]
    return {
        "count": len(runs),
        "longest_missing_sample_count": max(
            (run["missing_sample_count"] for run in runs), default=0
        ),
        "longest_nominal_sample_span_seconds": max(nominal_spans, default=0.0),
        "longest_bracketing_duration_seconds": max(bracketing_durations, default=None),
    }


def _quality_payload(
    frames: list[_Frame], grid: dict[str, list[_Point]]
) -> dict[str, Any]:
    total_frames = len(frames)
    total_grid_points = total_frames * len(MEDIAPIPE_LANDMARK_NAMES)
    required_observed_usable = sum(
        all(grid[name][index].observed_usable for name in REQUIRED_GAIT_LANDMARKS)
        for index in range(total_frames)
    )
    required_processed_complete = sum(
        all(
            grid[name][index].processed_x is not None
            and grid[name][index].processed_y is not None
            for name in REQUIRED_GAIT_LANDMARKS
        )
        for index in range(total_frames)
    )
    per_landmark: dict[str, Any] = {}
    all_points = [point for points in grid.values() for point in points]
    for name, points in grid.items():
        raw_count = sum(point.raw is not None for point in points)
        usable_count = sum(point.observed_usable for point in points)
        low_count = sum(point.rejected_low_confidence for point in points)
        score_missing_count = sum(
            point.missing_or_nonfinite_enabled_score for point in points
        )
        interpolated_count = sum(
            point.x_interpolated or point.y_interpolated for point in points
        )
        point_initial_missing_mask = [not point.observed_usable for point in points]
        point_remaining_missing_mask = [
            point.processed_x is None or point.processed_y is None for point in points
        ]
        point_interpolated_mask = [
            point.x_interpolated or point.y_interpolated for point in points
        ]
        point_interpolated_runs = _runs(points, point_interpolated_mask)
        point_initial_runs = _runs(points, point_initial_missing_mask)
        point_remaining_runs = _runs(points, point_remaining_missing_mask)
        x_initial_missing_mask = [not point.x_observed_usable for point in points]
        y_initial_missing_mask = [not point.y_observed_usable for point in points]
        x_interpolated_mask = [point.x_interpolated for point in points]
        y_interpolated_mask = [point.y_interpolated for point in points]
        x_remaining_missing_mask = [point.processed_x is None for point in points]
        y_remaining_missing_mask = [point.processed_y is None for point in points]

        def coordinate_gap_payload(
            coordinate_points: list[_Point],
            initial_mask: list[bool],
            interpolated_mask: list[bool],
            remaining_mask: list[bool],
        ) -> dict[str, Any]:
            initial_runs = _runs(coordinate_points, initial_mask)
            interpolated_runs = _runs(coordinate_points, interpolated_mask)
            remaining_runs = _runs(coordinate_points, remaining_mask)
            return {
                "initial_missing_gaps": initial_runs,
                "initial_missing_gap_summary": _gap_summary(initial_runs),
                "interpolated_gaps": interpolated_runs,
                "interpolated_gap_summary": _gap_summary(interpolated_runs),
                "remaining_missing_gaps": remaining_runs,
                "remaining_missing_gap_summary": _gap_summary(remaining_runs),
            }

        point_gap_payload = {
            "semantics": "union across x and y; not a scalar interpolation bound",
            "initial_missing_gaps": point_initial_runs,
            "initial_missing_gap_summary": _gap_summary(point_initial_runs),
            "interpolated_gaps": point_interpolated_runs,
            "interpolated_gap_summary": _gap_summary(point_interpolated_runs),
            "remaining_missing_gaps": point_remaining_runs,
            "remaining_missing_gap_summary": _gap_summary(point_remaining_runs),
        }
        x_gap_payload = coordinate_gap_payload(
            points,
            x_initial_missing_mask,
            x_interpolated_mask,
            x_remaining_missing_mask,
        )
        y_gap_payload = coordinate_gap_payload(
            points,
            y_initial_missing_mask,
            y_interpolated_mask,
            y_remaining_missing_mask,
        )
        per_landmark[name] = {
            "raw_row_count": raw_count,
            "raw_row_coverage": raw_count / total_frames,
            "usable_observation_count": usable_count,
            "usable_observation_coverage": usable_count / total_frames,
            "low_confidence_rejection_count": low_count,
            "low_confidence_rejection_coverage": low_count / total_frames,
            "low_confidence_rejection_fraction_of_raw_rows": (
                low_count / raw_count if raw_count else None
            ),
            "missing_or_nonfinite_enabled_score_count": score_missing_count,
            "missing_or_nonfinite_enabled_score_coverage": (
                score_missing_count / total_frames
            ),
            "missing_or_nonfinite_enabled_score_fraction_of_raw_rows": (
                score_missing_count / raw_count if raw_count else None
            ),
            "initial_missing_count": sum(point_initial_missing_mask),
            "initial_missing_coverage": (
                sum(point_initial_missing_mask) / total_frames
            ),
            "interpolated_count": interpolated_count,
            "interpolation_coverage": interpolated_count / total_frames,
            "remaining_missing_count": sum(point_remaining_missing_mask),
            "remaining_missing_coverage": (
                sum(point_remaining_missing_mask) / total_frames
            ),
            "point_union_gaps": point_gap_payload,
            "x_coordinate_gaps": x_gap_payload,
            "y_coordinate_gaps": y_gap_payload,
            # Aliases retained for the initial Step 3 schema.
            "initial_missing_gaps": point_initial_runs,
            "interpolated_gaps": point_interpolated_runs,
            "remaining_missing_gaps": point_remaining_runs,
            "initial_missing_gap_summary": _gap_summary(point_initial_runs),
            "interpolated_gap_summary": _gap_summary(point_interpolated_runs),
            "remaining_missing_gap_summary": _gap_summary(point_remaining_runs),
        }
    confidence_summaries: dict[str, Any] = {}
    for score_name in ("visibility", "presence", "confidence"):
        values = [
            value
            for point in all_points
            if point.raw is not None
            and (value := getattr(point.raw, score_name)) is not None
        ]
        confidence_summaries[score_name] = _summarize_values(values)
    interpolated_points = sum(
        point.x_interpolated or point.y_interpolated for point in all_points
    )
    remaining_missing_points = sum(
        point.processed_x is None or point.processed_y is None for point in all_points
    )
    interpolated_coordinates = sum(point.x_interpolated for point in all_points) + sum(
        point.y_interpolated for point in all_points
    )
    remaining_missing_coordinates = sum(
        point.processed_x is None for point in all_points
    ) + sum(point.processed_y is None for point in all_points)
    total_grid_coordinates = total_grid_points * 2
    return {
        "schema_version": PREPROCESSING_SCHEMA_VERSION,
        "semantics": {
            "observed_usable": (
                "heuristic planar usability based only on enabled raw backend score "
                "gating and finite raw x/y coordinates; it is not positional or "
                "anatomical accuracy and does not establish ground truth"
            ),
            "processed_completeness": (
                "both processed planar coordinates are present after Step 3; this may "
                "include coordinates produced by bounded interpolation and does not "
                "mean the landmark was observed or accurate"
            ),
            "simultaneous_all_12": (
                "all 12 named required gait landmarks satisfy the stated condition in "
                "the same nominal frame; this is not any 12 landmarks or coverage "
                "accumulated across frames"
            ),
            "legacy_aliases": {
                "fields": [
                    "frames_with_all_12_required_gait_landmarks",
                    "required_landmark_coverage",
                ],
                "ambiguity": (
                    "legacy names do not state observed versus processed semantics; "
                    "both aliases mean simultaneous-all-12 observed_usable and must "
                    "not be interpreted as processed completeness or accuracy"
                ),
            },
        },
        "total_frames": total_frames,
        "pose_detected_frames": sum(
            frame.status == PoseFrameStatus.DECODED_POSE for frame in frames
        ),
        "frames_with_all_required_landmarks_observed_usable": (
            required_observed_usable
        ),
        "required_landmarks_observed_usable_fraction": (
            required_observed_usable / total_frames
        ),
        "frames_with_all_required_landmarks_processed_complete": (
            required_processed_complete
        ),
        "required_landmarks_processed_complete_fraction": (
            required_processed_complete / total_frames
        ),
        # Aliases retained for the initial Step 3 schema; both mean raw planar usable.
        "frames_with_all_12_required_gait_landmarks": required_observed_usable,
        "required_landmark_coverage": required_observed_usable / total_frames,
        "required_gait_landmarks": list(REQUIRED_GAIT_LANDMARKS),
        "per_landmark": per_landmark,
        "available_confidence_summaries": confidence_summaries,
        "interpolated_fraction": interpolated_points / total_grid_points,
        "remaining_missing_fraction": remaining_missing_points / total_grid_points,
        "interpolated_coordinate_fraction": (
            interpolated_coordinates / total_grid_coordinates
        ),
        "remaining_missing_coordinate_fraction": (
            remaining_missing_coordinates / total_grid_coordinates
        ),
        "denominators": {
            "frame_fractions": "all pose_frames.csv rows",
            "per_landmark_coverage": "all nominal frames for that landmark",
            "fraction_of_raw_rows": (
                "raw returned rows for that landmark; null when raw_row_count is zero"
            ),
            "overall_point_fractions": "all nominal frames x 33 canonical landmarks",
            "overall_coordinate_fractions": (
                "all nominal frames x 33 canonical landmarks x 2 planar coordinates"
            ),
            "interpolated_point": "either planar coordinate was interpolated",
            "remaining_missing_point": "either processed planar coordinate is missing",
            "interpolated_coordinate": "one processed x or y scalar was interpolated",
            "remaining_missing_coordinate": (
                "one processed x or y scalar remains missing"
            ),
            "gap_length": "count of missing nominal samples; not elapsed duration",
            "nominal_sample_span_seconds": (
                "last missing sample timestamp minus first missing sample timestamp; "
                "zero for a one-sample gap"
            ),
            "bracketing_duration_seconds": (
                "right usable bracket timestamp minus left usable bracket timestamp; "
                "null for boundary gaps"
            ),
        },
    }


def _git_provenance(directory: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in (
        "gait-stability",
        "opencv-contrib-python-headless",
        "matplotlib",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _metadata_payload(
    artifacts: RawPoseArtifacts,
    staging: Path,
    config: PosePreprocessingConfig,
    source_metadata: dict[str, Any],
    input_hashes: dict[Path, str],
    run_id: str,
    created_at: str,
) -> dict[str, Any]:
    output_names = [
        name
        for name in STEP3_ARTIFACT_NAMES
        if name != "preprocessing_metadata.json" and (staging / name).exists()
    ]
    input_paths = (
        artifacts.raw_landmarks_path,
        artifacts.pose_frames_path,
        artifacts.pose_metadata_path,
    )
    return {
        "schema_version": PREPROCESSING_SCHEMA_VERSION,
        "algorithm_version": PREPROCESSING_ALGORITHM_VERSION,
        "run_id": run_id,
        "created_at_utc": created_at,
        "scope": (
            "quality assessment, bounded interpolation, and centered moving-average "
            "smoothing of normalized image-plane x/y pose estimates only"
        ),
        "inputs": {
            path.name: {"path": str(path), "sha256": input_hashes[path]}
            for path in input_paths
        },
        "outputs": {
            name: {
                "path": str(artifacts.artifact_directory / name),
                "sha256": sha256_file(staging / name),
            }
            for name in output_names
        }
        | {
            "preprocessing_metadata.json": {
                "path": str(
                    artifacts.artifact_directory / "preprocessing_metadata.json"
                ),
                "sha256": None,
                "sha256_semantics": (
                    "not self-recorded because embedding a file's own digest changes "
                    "that digest"
                ),
            }
        },
        "inherited_provenance": {
            "source": source_metadata.get("source"),
            "backend": source_metadata.get("backend"),
            "capture_assumptions": source_metadata.get("capture_assumptions"),
        },
        "runtime": {
            "python_version": sys.version,
            "dependency_versions": _dependency_versions(),
            "git": _git_provenance(Path.cwd()),
            "randomness": "none",
        },
        "config": asdict(config),
        "confidence_semantics": {
            "fields": {
                "visibility": (
                    "raw backend model score associated with landmark visibility; "
                    "not calibrated accuracy, probability, or ground truth"
                ),
                "presence": (
                    "raw backend model score associated with landmark presence; not "
                    "calibrated accuracy, probability, or ground truth"
                ),
                "confidence": (
                    "nullable generic raw backend field when supplied; it is distinct "
                    "from visibility and presence"
                ),
            },
            "enabled_fields": [name for name, _threshold in _enabled_scores(config)],
            "acceptance_rule": (
                "all enabled scores must be finite, present, and greater than or equal "
                "to their configured threshold; equality passes"
            ),
            "missing_nonfinite_rule": (
                "rejected distinctly from finite below-threshold scores"
            ),
            "threshold_status": (
                "configurable engineering heuristics, not calibrated probabilities or "
                "validated positional-accuracy cutoffs"
            ),
            "applicability": (
                "configured at schema/field level for every raw landmark row"
            ),
        },
        "interpolation": {
            "method": (
                "linear in nominal_timestamp_seconds per landmark and scalar coordinate"
            ),
            "maximum_missing_samples": config.max_gap_frames,
            "eligibility": (
                "interior scalar-coordinate gaps only, bounded by raw-observed usable "
                "endpoints for that coordinate; no recursive use of interpolated "
                "endpoints"
            ),
            "extrapolation": "none",
            "confidence_interpolation": "none",
            "gait_event_crossing": (
                "not assessed because gait events are not available at Step 3"
            ),
            "eligible_landmarks": (
                "all 33 canonical landmarks listed in eligible_landmark_names"
            ),
            "eligible_landmark_names": list(MEDIAPIPE_LANDMARK_NAMES),
        },
        "smoothing": {
            "method": "centered unweighted moving average (boxcar)",
            "configured_window_frames": config.smoothing_window_frames,
            "phase": (
                "centered/noncausal and index-symmetric, with no fixed group delay "
                "under uniform sampling; this does not claim preservation of extrema, "
                "threshold crossings, derivatives, or gait-event timing; not "
                "time-weighted for irregular timestamps"
            ),
            "sampling_rate_assumption": (
                "window is frame-count based; timestamps are not resampled"
            ),
            "segments": (
                "each contiguous nonmissing scalar coordinate segment independently"
            ),
            "edge_behavior": (
                "largest symmetric odd support not exceeding the configured window or "
                "available samples; segment endpoints therefore use one sample"
            ),
            "default_window_context": (
                "the default 3-frame window at 30 fps spans 3 samples (~0.1 s of "
                "samples) and ~0.067 s between support endpoints; this is an "
                "unvalidated engineering choice"
            ),
            "confidence_fields": "not smoothed",
        },
        "coordinates": {
            "processed": ["x_normalized", "y_normalized"],
            "x_normalized": (
                "dimensionless model estimate; image left to right; finite values "
                "outside "
                "[0, 1] remain usable but are flagged"
            ),
            "y_normalized": (
                "dimensionless model estimate; image top to bottom; finite values "
                "outside "
                "[0, 1] remain usable but are flagged"
            ),
            "z_backend_relative": "preserved raw and never interpolated or smoothed",
            "physical_conversion": "none",
        },
        "output_schema": {
            "processed_landmarks.csv": {
                "relationship": (
                    "derived complete nominal frame x 33 canonical-landmark audit grid"
                ),
                "columns": list(PROCESSED_FIELDS),
                "boolean_serialization": "lowercase true/false",
                "missing_serialization": "blank CSV field",
                "observed_usable_semantics": (
                    "planar raw observation usability: x_observed_usable AND "
                    "y_observed_usable after enabled-score gating"
                ),
                "coordinate_observed_usable_semantics": (
                    "enabled scores pass and that raw scalar coordinate is finite"
                ),
                "out_of_image_semantics": (
                    "finite normalized x/y outside [0, 1] are flagged but count as "
                    "usable"
                ),
            },
            "pose_quality.json": "descriptive pose-data quality only; no quality label",
        },
        "limitations": [
            (
                "Interpolation can hide phase-dependent missingness and may cross an "
                "unknown gait event."
            ),
            (
                "No detection of left/right swaps, spikes, tracking discontinuities, "
                "or camera motion is performed."
            ),
            (
                "Centered boxcar smoothing attenuates trajectory amplitude and can "
                "suppress real variation."
            ),
            (
                "Monocular normalized coordinates are nonphysical pose-model "
                "estimates, not measured anatomy."
            ),
            (
                "No camera calibration, physical scale, gait events, COM, gait "
                "metrics, stability metrics, or clinical interpretation is produced."
            ),
            (
                "Nominal timestamps derive from Step 2 and are not verified "
                "presentation timestamps."
            ),
        ],
    }


def _write_processed_csv(staging: Path, grid: dict[str, list[_Point]]) -> None:
    rows_by_frame: dict[int, list[_Point]] = {}
    for points in grid.values():
        for point in points:
            rows_by_frame.setdefault(point.frame.frame_index, []).append(point)
    with (staging / "processed_landmarks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=PROCESSED_FIELDS)
        writer.writeheader()
        for frame_index in sorted(rows_by_frame):
            for point in sorted(
                rows_by_frame[frame_index], key=lambda item: item.landmark_id
            ):
                writer.writerow(
                    {key: _csv_value(value) for key, value in _point_row(point).items()}
                )


def _write_diagnostic(
    staging: Path,
    grid: dict[str, list[_Point]],
    landmarks: tuple[str, ...],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PosePreprocessingError(
            "Diagnostic plotting requires optional matplotlib; use --no-diagnostic "
            "or install it"
        ) from exc
    figure, axes = plt.subplots(
        len(landmarks), 2, figsize=(12, 2.3 * len(landmarks)), squeeze=False
    )
    try:
        for row_index, name in enumerate(landmarks):
            points = grid[name]
            time = [point.frame.timestamp for point in points]
            for column, coordinate in enumerate(("x", "y")):
                axis = axes[row_index][column]
                raw_values = [
                    math.nan if point.raw is None else getattr(point.raw, coordinate)
                    for point in points
                ]
                processed_values = [
                    math.nan
                    if (value := getattr(point, f"processed_{coordinate}")) is None
                    else value
                    for point in points
                ]
                axis.plot(time, raw_values, color="0.65", linewidth=1, label="raw")
                axis.plot(
                    time,
                    processed_values,
                    color="tab:blue",
                    linewidth=1.2,
                    label="processed",
                )
                axis.set_title(f"{name} {coordinate}_normalized")
                axis.set_xlabel("nominal time (s)")
                axis.set_ylabel("normalized image coordinate")
                axis.grid(alpha=0.2)
                if row_index == 0 and column == 0:
                    axis.legend()
        figure.suptitle("Raw vs processed pose-model trajectories")
        figure.tight_layout()
        figure.savefig(staging / DIAGNOSTIC_NAME, dpi=120)
    finally:
        plt.close(figure)


def _publish_step3_artifacts(staging: Path, destination: Path) -> None:
    """Replace the complete Step 3 set and restore it on a rename failure."""
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for name in STEP3_ARTIFACT_NAMES:
            target = destination / name
            if target.exists():
                backup = destination / f"{name}.backup-{uuid.uuid4().hex}"
                target.replace(backup)
                backups[target] = backup
        for name in STEP3_ARTIFACT_NAMES:
            staged = staging / name
            if not staged.exists():
                continue
            target = destination / name
            staged.replace(target)
            published.append(target)
    except OSError as exc:
        restore_failures: list[str] = []
        for target in published:
            try:
                target.unlink()
            except OSError as restore_exc:
                restore_failures.append(f"could not remove {target}: {restore_exc}")
        for target, backup in backups.items():
            try:
                backup.replace(target)
            except OSError as restore_exc:
                restore_failures.append(
                    f"could not restore {target} from backup {backup}: {restore_exc}"
                )
        if restore_failures:
            raise ArtifactPublishError(
                "Could not publish pose preprocessing artifacts and restoration may "
                "be incomplete. Remaining backup locations and failures: "
                + "; ".join(restore_failures)
            ) from exc
        raise ArtifactPublishError(
            "Could not atomically publish pose preprocessing artifacts"
        ) from exc
    for backup in backups.values():
        with suppress(OSError):
            backup.unlink()


def _resolved_artifacts(artifacts: RawPoseArtifacts) -> RawPoseArtifacts:
    return RawPoseArtifacts(
        artifact_directory=artifacts.artifact_directory.expanduser().resolve(),
        raw_landmarks_path=artifacts.raw_landmarks_path.expanduser().resolve(),
        pose_frames_path=artifacts.pose_frames_path.expanduser().resolve(),
        pose_metadata_path=artifacts.pose_metadata_path.expanduser().resolve(),
    )


def _validate_artifact_path_isolation(artifacts: RawPoseArtifacts) -> None:
    input_paths = {
        "raw_landmarks.csv input": artifacts.raw_landmarks_path,
        "pose_frames.csv input": artifacts.pose_frames_path,
        "pose_metadata.json input": artifacts.pose_metadata_path,
    }
    output_paths = {
        f"{name} output": artifacts.artifact_directory / name
        for name in STEP3_ARTIFACT_NAMES
    }
    seen: dict[Path, str] = {}
    for label, path in input_paths.items():
        previous = seen.get(path)
        if previous is not None:
            raise PoseArtifactValidationError(
                f"Artifact paths overlap: {label} and {previous} both resolve to {path}"
            )
        seen[path] = label
    for label, path in output_paths.items():
        resolved_output = path.resolve()
        previous = seen.get(resolved_output)
        if previous is not None:
            raise PoseArtifactValidationError(
                f"Artifact paths overlap: {label} and {previous} both resolve to "
                f"{resolved_output}"
            )
        seen[resolved_output] = label


def _input_hashes(artifacts: RawPoseArtifacts) -> dict[Path, str]:
    paths = (
        artifacts.raw_landmarks_path,
        artifacts.pose_frames_path,
        artifacts.pose_metadata_path,
    )
    try:
        return {path: sha256_file(path) for path in paths}
    except OSError as exc:
        raise PoseArtifactValidationError(
            f"Could not hash required Step 2 artifact: {exc}"
        ) from exc


def _verify_input_hashes(
    artifacts: RawPoseArtifacts, expected_hashes: dict[Path, str]
) -> None:
    current_hashes = _input_hashes(artifacts)
    changed = sorted(
        path.name
        for path, expected_hash in expected_hashes.items()
        if current_hashes.get(path) != expected_hash
    )
    if changed:
        raise PoseArtifactValidationError(
            "Step 2 input artifact changed during preprocessing: " + ", ".join(changed)
        )


def preprocess_pose(
    raw_pose: str | Path | RawPoseArtifacts,
    config: PosePreprocessingConfig | None = None,
) -> PosePreprocessingArtifacts:
    """Validate Step 2 artifacts, preprocess planar trajectories, and publish Step 3."""
    if config is None:
        config = PosePreprocessingConfig()
    artifacts = _resolved_artifacts(
        raw_pose
        if isinstance(raw_pose, RawPoseArtifacts)
        else RawPoseArtifacts.from_directory(raw_pose)
    )
    _validate_artifact_path_isolation(artifacts)
    if not artifacts.artifact_directory.is_dir():
        raise PoseArtifactValidationError(
            f"Artifact directory does not exist: {artifacts.artifact_directory}"
        )
    input_hashes = _input_hashes(artifacts)
    frames, raw, source_metadata = _load_inputs(artifacts)
    grid = _build_grid(frames, raw, config)
    _process_grid(grid, config)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f"{artifacts.artifact_directory.name}.preprocessing-staging-",
            dir=artifacts.artifact_directory.parent,
        )
    )
    run_id = uuid.uuid4().hex
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        _write_processed_csv(staging, grid)
        quality = _quality_payload(frames, grid)
        (staging / "pose_quality.json").write_text(
            json.dumps(quality, indent=2) + "\n", encoding="utf-8"
        )
        if config.write_diagnostic:
            _write_diagnostic(staging, grid, config.diagnostic_landmarks)
        metadata = _metadata_payload(
            artifacts,
            staging,
            config,
            source_metadata,
            input_hashes,
            run_id,
            created_at,
        )
        (staging / "preprocessing_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        _verify_input_hashes(artifacts, input_hashes)
        _publish_step3_artifacts(staging, artifacts.artifact_directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    diagnostic_path = artifacts.artifact_directory / DIAGNOSTIC_NAME
    return PosePreprocessingArtifacts(
        artifact_directory=artifacts.artifact_directory,
        processed_landmarks_path=artifacts.artifact_directory
        / "processed_landmarks.csv",
        pose_quality_path=artifacts.artifact_directory / "pose_quality.json",
        preprocessing_metadata_path=artifacts.artifact_directory
        / "preprocessing_metadata.json",
        diagnostic_path=diagnostic_path if config.write_diagnostic else None,
    )
