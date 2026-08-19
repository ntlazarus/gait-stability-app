"""Step 5: COM proxy estimation from processed pose landmarks.

Loads six required inputs (processed_landmarks.csv, preprocessing_metadata.json,
pose_frames.csv, reviewed_gait_events.csv, reviewed_strides.csv,
review_resolution_metadata.json), validates all contracts and hashes, computes
segment-weighted anthropometric COM proxy for every frame, normalizes per stride,
and publishes com_proxy.csv, stride_com.csv, com_diagnostic.png, and
com_metadata.json as a transactional set.

All six inputs are resolved to absolute paths and their SHA-256 hashes are
snapshotted once before any parsing begins.  Output paths are checked for alias
collisions against inputs and each other.  Hashes are rechecked immediately
before publication.
"""

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
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gait_stability.com_estimation import (
    BILATERAL_SEGMENTS,
    COM_PROXY_FIELDS,
    DE_LEVA_FEMALE,
    DE_LEVA_MALE,
    MODEL_MASS_TOTAL_FEMALE,
    MODEL_MASS_TOTAL_MALE,
    REPRESENTED_MASS_MAX_FEMALE,
    REPRESENTED_MASS_MAX_MALE,
    SEGMENT_ENDPOINTS,
    SEGMENT_NAMES,
    STRIDE_COM_FIELDS,
    UNSUPPORTED_SEGMENTS,
    ComEstimationConfig,
    FrameComResult,
    LandmarkProvenance,
    Point2D,
    StrideComSample,
    create_landmark_provenance_from_processed,
    estimate_frame_com,
    normalize_stride_com,
)
from gait_stability.gait_events import STRIDE_FIELDS
from gait_stability.pose_contracts import MEDIAPIPE_LANDMARK_NAMES, PoseFrameStatus
from gait_stability.pose_pipeline import FRAME_FIELDS
from gait_stability.pose_preprocessing import (
    PREPROCESSING_ALGORITHM_VERSION,
    PREPROCESSING_SCHEMA_VERSION,
    PROCESSED_FIELDS,
)
from gait_stability.review_resolution import (
    REVIEW_RESOLUTION_ALGORITHM_VERSION,
    REVIEW_RESOLUTION_SCHEMA_VERSION,
    REVIEWED_GAIT_EVENT_FIELDS,
)
from gait_stability.video_ingestion import ArtifactPublishError, sha256_file

# ---------------------------------------------------------------------------
# Schema/version constants
# ---------------------------------------------------------------------------

COM_SCHEMA_VERSION: int = 1
COM_ALGORITHM_VERSION: str = "step5-com-proxy-1"

# Backward-compatible aliases for tests that import from com_pipeline
COM_PROXY_FIELD_NAMES: tuple[str, ...] = COM_PROXY_FIELDS
STRIDE_COM_FIELD_NAMES: tuple[str, ...] = STRIDE_COM_FIELDS

COM_OUTPUT_ARTIFACT_NAMES: tuple[str, ...] = (
    "com_proxy.csv",
    "stride_com.csv",
    "com_diagnostic.png",
    "com_metadata.json",
)

COM_PROXY_SCHEMA_VERSION: int = 1

# Canonical reviewed_strides.csv header: STRIDE_FIELDS + four review extras
REVIEWED_STRIDE_FIELDS: tuple[str, ...] = STRIDE_FIELDS + (
    "automatic_stride_id",
    "review_intent",
    "review_changes",
    "provenance_notes",
)

# The six required input basenames
_INPUT_BASENAMES: tuple[str, ...] = (
    "processed_landmarks.csv",
    "preprocessing_metadata.json",
    "pose_frames.csv",
    "reviewed_gait_events.csv",
    "reviewed_strides.csv",
    "review_resolution_metadata.json",
)

# Output basenames for alias checking
_OUTPUT_BASENAMES: tuple[str, ...] = (
    "com_proxy.csv",
    "stride_com.csv",
    "com_diagnostic.png",
    "com_metadata.json",
)

# ---------------------------------------------------------------------------
# Public error hierarchy
# ---------------------------------------------------------------------------


class ComPipelineError(Exception):
    """Expected Step 5 input, processing, or artifact error."""


class ComArtifactValidationError(ComPipelineError):
    """Raised when canonical artifacts violate their contracts."""


# ---------------------------------------------------------------------------
# Public return type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComArtifacts:
    """Published Step 5 artifact paths."""

    artifact_directory: Path
    com_proxy_path: Path
    stride_com_path: Path
    diagnostic_path: Path  # Required, not optional
    com_metadata_path: Path


# ---------------------------------------------------------------------------
# Internal: typed structures for reviewed artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ReviewedEvent:
    """Fully parsed and validated row from reviewed_gait_events.csv."""

    event_id: str
    automatic_event_id: str
    side: str
    event_type: str
    automatic_frame_index: int
    automatic_timestamp_seconds: float
    automatic_disposition: str
    automatic_quality: str
    automatic_peak_value: float
    automatic_prominence: float | None
    automatic_rejection_reasons: tuple[str, ...]
    manual_event_review_status: str
    stride_review_provenance: str
    reviewed_frame_index: int
    reviewed_timestamp_seconds: float
    reviewed_accepted: bool
    reviewed_rejected: bool
    reviewed_included_in_stride: bool
    reviewed_quality: str
    resolution_disposition: str
    replaces_event_id: str | None
    replaced_by_event_id: str | None
    source: str
    review_notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ReviewedStride:
    """Fully parsed and validated row from reviewed_strides.csv.

    Carries the four extra review-provenance fields that must be written
    to every stride_com.csv row.
    """

    stride_id: str
    side: str
    start_event_id: str
    end_event_id: str
    start_frame: int
    end_frame: int
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    duration_seconds: float
    quality: str
    contralateral_event_id: str | None
    contralateral_event_count: int
    sequence_notes: tuple[str, ...]
    source: str
    review_status: str
    # Review-provenance extras (written to stride_com.csv)
    automatic_stride_id: str
    review_intent: str
    review_changes: str
    provenance_notes: str


@dataclass(frozen=True, slots=True)
class _InputPaths:
    """Resolved absolute paths for the six required inputs."""

    processed_landmarks: Path
    preprocessing_metadata: Path
    pose_frames: Path
    reviewed_gait_events: Path
    reviewed_strides: Path
    review_resolution_metadata: Path

    def as_mapping(self) -> dict[str, Path]:
        return {
            "processed_landmarks.csv": self.processed_landmarks,
            "preprocessing_metadata.json": self.preprocessing_metadata,
            "pose_frames.csv": self.pose_frames,
            "reviewed_gait_events.csv": self.reviewed_gait_events,
            "reviewed_strides.csv": self.reviewed_strides,
            "review_resolution_metadata.json": self.review_resolution_metadata,
        }


# ---------------------------------------------------------------------------
# Internal: CSV helpers
# ---------------------------------------------------------------------------


def _csv_bool_lower(value: bool) -> str:
    return "true" if value else "false"


def _csv_tuple_pipe(value: tuple[str, ...]) -> str:
    return "|".join(value)


def _parse_int(text: str, field: str, row_number: int, label: str) -> int:
    try:
        return int(text)
    except ValueError as exc:
        raise ComArtifactValidationError(
            f"{label} row {row_number}: {field} must be an integer"
        ) from exc


def _parse_float(
    text: str,
    field: str,
    row_number: int,
    label: str,
    *,
    nullable: bool = False,
) -> float | None:
    if nullable and text == "":
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ComArtifactValidationError(
            f"{label} row {row_number}: {field} must be numeric"
        ) from exc
    if not math.isfinite(value):
        raise ComArtifactValidationError(
            f"{label} row {row_number}: {field} must be finite"
        )
    return value


def _parse_bool_lower(text: str, field: str, row_number: int, label: str) -> bool:
    if text not in {"true", "false"}:
        raise ComArtifactValidationError(
            f"{label} row {row_number}: {field} must be true or false (lowercase)"
        )
    return text == "true"


def _parse_tuple_pipe(text: str) -> tuple[str, ...]:
    if text == "":
        return ()
    return tuple(text.split("|"))


def _parse_opt_str(text: str) -> str | None:
    return text if text != "" else None


# ---------------------------------------------------------------------------
# Internal: input resolution and path-alias checking
# ---------------------------------------------------------------------------


def _resolve_inputs(directory: Path) -> _InputPaths:
    """Resolve all six input basenames to absolute paths.

    Raises ComArtifactValidationError if any input is missing.
    """
    paths: dict[str, Path] = {}
    for basename in _INPUT_BASENAMES:
        p = directory / basename
        if not p.is_file():
            raise ComArtifactValidationError(f"Required Step 5 input is missing: {p}")
        paths[basename] = p
    return _InputPaths(
        processed_landmarks=paths["processed_landmarks.csv"],
        preprocessing_metadata=paths["preprocessing_metadata.json"],
        pose_frames=paths["pose_frames.csv"],
        reviewed_gait_events=paths["reviewed_gait_events.csv"],
        reviewed_strides=paths["reviewed_strides.csv"],
        review_resolution_metadata=paths["review_resolution_metadata.json"],
    )


def _snapshot_input_hashes(inputs: _InputPaths) -> dict[str, str]:
    """Snapshot SHA-256 hashes for all six inputs, keyed by canonical basename."""
    hashes: dict[str, str] = {}
    for basename, path in inputs.as_mapping().items():
        hashes[basename] = sha256_file(path)
    return hashes


def _verify_output_no_alias(
    directory: Path,
    inputs: _InputPaths,
) -> None:
    """Verify that no output path resolves to or aliases any input path."""
    input_resolved: set[Path] = {p.resolve() for p in inputs.as_mapping().values()}
    for out_name in _OUTPUT_BASENAMES:
        out_resolved = (directory / out_name).resolve()
        if out_resolved in input_resolved:
            raise ComArtifactValidationError(
                f"Output path {out_name} resolves to an input path"
            )
    # Check outputs don't alias each other
    out_resolved_list: list[Path] = []
    for out_name in _OUTPUT_BASENAMES:
        r = (directory / out_name).resolve()
        if r in out_resolved_list:
            raise ComArtifactValidationError(
                f"Output path {out_name} aliases another output path"
            )
        out_resolved_list.append(r)


def _recheck_input_hashes(
    snapshot_hashes: dict[str, str],
    inputs: _InputPaths,
) -> None:
    """Recheck all input hashes match the original snapshot."""
    mapping = inputs.as_mapping()
    for basename, expected in snapshot_hashes.items():
        current = sha256_file(mapping[basename])
        if current != expected:
            raise ComArtifactValidationError(
                f"Input artifact changed during COM estimation: {basename}"
            )


# ---------------------------------------------------------------------------
# Internal: strict input readers
# ---------------------------------------------------------------------------


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ComArtifactValidationError(f"Required artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComArtifactValidationError(
            f"Could not read valid {label}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ComArtifactValidationError(f"{label} root must be an object")
    return value


def _read_pose_frames_strict(
    path: Path,
) -> dict[int, tuple[float, str, int, int | None]]:
    """Read and validate pose_frames.csv strictly.

    Returns {frame_index: (timestamp, status, landmark_count, backend_ts_ms)}.

    Enforces:
    - exact FRAME_FIELDS header
    - nonempty
    - file row order: frame_index == row_number-2 (ordered contiguous from
      zero; not merely sorted afterward)
    - strictly increasing finite nonnegative nominal timestamps
    - valid PoseFrameStatus
    - nonnegative landmark_count
    - backend_timestamp_milliseconds parsed (blank allowed for decode failures)
      and nonnegative when present
    """
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != FRAME_FIELDS:
                raise ComArtifactValidationError(
                    "pose_frames.csv header must exactly match FRAME_FIELDS"
                )
            result: dict[int, tuple[float, str, int, int | None]] = {}
            prev_timestamp: float | None = None
            valid_statuses = {s.value for s in PoseFrameStatus}
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(row[field] is None for field in FRAME_FIELDS):
                    raise ComArtifactValidationError(
                        f"pose_frames.csv row {row_number}: malformed columns"
                    )
                fi = _parse_int(
                    row["frame_index"], "frame_index", row_number, "pose_frames.csv"
                )
                # Enforce file row order: frame_index must equal row_number-2
                expected_frame = row_number - 2
                if fi != expected_frame:
                    raise ComArtifactValidationError(
                        f"pose_frames.csv row {row_number}: frame_index {fi} "
                        f"does not match expected {expected_frame} from file "
                        "row order (duplicate or out-of-order)"
                    )
                ts = _parse_float(
                    row["nominal_timestamp_seconds"],
                    "nominal_timestamp_seconds",
                    row_number,
                    "pose_frames.csv",
                )
                if ts is None:
                    raise ComArtifactValidationError(
                        f"pose_frames.csv row {row_number}: timestamp must be finite"
                    )
                if ts < 0.0:
                    raise ComArtifactValidationError(
                        f"pose_frames.csv row {row_number}: "
                        "timestamp must be nonnegative"
                    )
                if prev_timestamp is not None and ts <= prev_timestamp:
                    raise ComArtifactValidationError(
                        f"pose_frames.csv row {row_number}: "
                        "timestamps must be strictly increasing"
                    )
                prev_timestamp = ts

                status = row["status"]
                if status not in valid_statuses:
                    raise ComArtifactValidationError(
                        f"pose_frames.csv row {row_number}: invalid status {status!r}"
                    )

                landmark_count = _parse_int(
                    row["landmark_count"],
                    "landmark_count",
                    row_number,
                    "pose_frames.csv",
                )
                if landmark_count < 0:
                    raise ComArtifactValidationError(
                        f"pose_frames.csv row {row_number}: "
                        "landmark_count must be nonnegative"
                    )

                # Parse backend_timestamp_milliseconds (blank allowed)
                bt_ms_text = row["backend_timestamp_milliseconds"]
                if bt_ms_text == "":
                    bt_ms: int | None = None
                else:
                    bt_ms = _parse_int(
                        bt_ms_text,
                        "backend_timestamp_milliseconds",
                        row_number,
                        "pose_frames.csv",
                    )
                    if bt_ms < 0:
                        raise ComArtifactValidationError(
                            f"pose_frames.csv row {row_number}: "
                            "backend_timestamp_milliseconds must be nonnegative"
                        )

                result[fi] = (ts, status, landmark_count, bt_ms)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ComArtifactValidationError(f"Could not read {path}: {exc}") from exc

    if not result:
        raise ComArtifactValidationError(
            "pose_frames.csv must contain at least one row"
        )

    return result


def _read_processed_landmarks_strict(
    path: Path,
    pose_manifest: dict[int, tuple[float, str, int, int | None]],
) -> dict[int, dict[int, dict[str, str]]]:
    """Read and validate processed_landmarks.csv strictly.

    Returns {frame_index: {landmark_id: row_dict}}.

    Enforces:
    - exact PROCESSED_FIELDS header
    - file row order: frame_index == position // 33 and landmark_id == position
      % 33 (ordered contiguous, not merely complete set after reading)
    - canonical ID/name (MEDIAPIPE_LANDMARK_NAMES[landmark_id] == name)
    - same exact timestamp/status within frame and against pose manifest
    - every lowercase boolean parsed and validated
    - all nullable numeric fields numeric
    - raw values may be nonfinite only as Step3 permits
    - processed/pre-smoothed finite-or-blank
    - x/y observed_usable requires finite raw values and observed=AND
    - x/y final_missing exactly matches blank processed and final_missing=OR
    """
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != PROCESSED_FIELDS:
                raise ComArtifactValidationError(
                    "processed_landmarks.csv header must exactly match PROCESSED_FIELDS"
                )
            result: dict[int, dict[int, dict[str, str]]] = {}
            expected_position = 0
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(row[field] is None for field in PROCESSED_FIELDS):
                    raise ComArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: malformed"
                    )
                fi = _parse_int(
                    row["frame_index"],
                    "frame_index",
                    row_number,
                    "processed_landmarks.csv",
                )
                lid = _parse_int(
                    row["landmark_id"],
                    "landmark_id",
                    row_number,
                    "processed_landmarks.csv",
                )

                # Enforce file row order: position-based expected frame/landmark
                num_landmarks = len(MEDIAPIPE_LANDMARK_NAMES)
                expected_frame = expected_position // num_landmarks
                expected_lid = expected_position % num_landmarks
                if fi != expected_frame or lid != expected_lid:
                    raise ComArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        f"frame_index {fi}/landmark_id {lid} does not match "
                        f"expected ({expected_frame}/{expected_lid}) from file "
                        f"row order (expected_position={expected_position})"
                    )
                expected_position += 1

                # Validate canonical ID/name
                if not 0 <= lid < len(MEDIAPIPE_LANDMARK_NAMES):
                    raise ComArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        f"landmark_id {lid} is not canonical"
                    )
                expected_name = MEDIAPIPE_LANDMARK_NAMES[lid]
                if row["landmark_name"] != expected_name:
                    raise ComArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        f"landmark ID {lid} name mismatch: "
                        f"expected {expected_name!r}, got {row['landmark_name']!r}"
                    )

                # Validate against pose manifest
                if fi not in pose_manifest:
                    raise ComArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        f"frame_index {fi} not in pose_frames.csv manifest"
                    )
                manifest_ts, manifest_status, _, _ = pose_manifest[fi]

                row_ts = _parse_float(
                    row["nominal_timestamp_seconds"],
                    "nominal_timestamp_seconds",
                    row_number,
                    "processed_landmarks.csv",
                )
                if row_ts is None:
                    raise ComArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        "timestamp must be finite"
                    )
                if abs(row_ts - manifest_ts) > 1e-9:
                    raise ComArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        f"timestamp {row_ts} does not match pose manifest "
                        f"{manifest_ts} for frame {fi}"
                    )

                row_status = row["frame_status"]
                if row_status != manifest_status:
                    raise ComArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        f"frame_status {row_status!r} does not match pose "
                        f"manifest {manifest_status!r} for frame {fi}"
                    )

                # Validate duplicate
                if fi not in result:
                    result[fi] = {}
                if lid in result[fi]:
                    raise ComArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        f"duplicate frame {fi} / landmark_id {lid}"
                    )
                result[fi][lid] = dict(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ComArtifactValidationError(f"Could not read {path}: {exc}") from exc

    if not result:
        raise ComArtifactValidationError(
            "processed_landmarks.csv must contain at least one frame"
        )

    # Validate field-level contracts for every row
    _validate_processed_field_contracts(result, pose_manifest)

    return result


def _validate_processed_field_contracts(
    data: dict[int, dict[int, dict[str, str]]],
    pose_manifest: dict[int, tuple[float, str, int, int | None]],
) -> None:
    """Validate field-level contracts for every processed_landmarks row."""
    nullable_numeric_fields = (
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
    )
    boolean_fields = (
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
    )

    for fi in sorted(data.keys()):
        _, manifest_status, _, _ = pose_manifest[fi]
        for lid in sorted(data[fi].keys()):
            row = data[fi][lid]

            # Validate nullable numeric fields are numeric when non-blank
            for nfield in nullable_numeric_fields:
                val = row.get(nfield, "")
                if val != "":
                    try:
                        fval = float(val)
                    except ValueError as exc:
                        raise ComArtifactValidationError(
                            f"processed_landmarks.csv frame {fi} landmark "
                            f"{lid}: {nfield} must be numeric when present"
                        ) from exc

            # Validate boolean fields are lowercase true/false
            for bfield in boolean_fields:
                val = row.get(bfield, "")
                if val not in {"true", "false"}:
                    raise ComArtifactValidationError(
                        f"processed_landmarks.csv frame {fi} landmark "
                        f"{lid}: {bfield} must be true or false (lowercase)"
                    )

            # Validate processed/pre-smoothed finite-or-blank
            for coord_field in (
                "pre_smoothed_x_normalized",
                "pre_smoothed_y_normalized",
                "processed_x_normalized",
                "processed_y_normalized",
            ):
                val = row.get(coord_field, "")
                if val != "":
                    try:
                        fval = float(val)
                    except ValueError as exc:
                        raise ComArtifactValidationError(
                            f"processed_landmarks.csv frame {fi} landmark "
                            f"{lid}: {coord_field} must be numeric"
                        ) from exc
                    if not math.isfinite(fval):
                        raise ComArtifactValidationError(
                            f"processed_landmarks.csv frame {fi} landmark "
                            f"{lid}: {coord_field} must be finite when present"
                        )

            # Validate x/y observed_usable requires finite raw values
            raw_x_text = row.get("raw_x_normalized", "")
            raw_y_text = row.get("raw_y_normalized", "")
            raw_x_finite = raw_x_text != "" and _is_finite(raw_x_text)
            raw_y_finite = raw_y_text != "" and _is_finite(raw_y_text)

            x_observed = row["x_observed_usable"] == "true"
            y_observed = row["y_observed_usable"] == "true"
            observed = row["observed_usable"] == "true"

            # x_observed_usable => raw_x must be finite
            if x_observed and not raw_x_finite:
                raise ComArtifactValidationError(
                    f"processed_landmarks.csv frame {fi} landmark {lid}: "
                    "x_observed_usable is true but raw_x_normalized is "
                    "not finite"
                )
            # y_observed_usable => raw_y must be finite
            if y_observed and not raw_y_finite:
                raise ComArtifactValidationError(
                    f"processed_landmarks.csv frame {fi} landmark {lid}: "
                    "y_observed_usable is true but raw_y_normalized is "
                    "not finite"
                )
            # observed_usable == x_observed AND y_observed
            if observed != (x_observed and y_observed):
                raise ComArtifactValidationError(
                    f"processed_landmarks.csv frame {fi} landmark {lid}: "
                    "observed_usable must equal "
                    "x_observed_usable AND y_observed_usable"
                )

            # Validate final_missing consistency
            proc_x_text = row.get("processed_x_normalized", "")
            proc_y_text = row.get("processed_y_normalized", "")
            x_final_missing = row["x_final_missing"] == "true"
            y_final_missing = row["y_final_missing"] == "true"
            final_missing = row["final_missing"] == "true"

            if x_final_missing != (proc_x_text == ""):
                raise ComArtifactValidationError(
                    f"processed_landmarks.csv frame {fi} landmark {lid}: "
                    "x_final_missing does not match blank processed_x"
                )
            if y_final_missing != (proc_y_text == ""):
                raise ComArtifactValidationError(
                    f"processed_landmarks.csv frame {fi} landmark {lid}: "
                    "y_final_missing does not match blank processed_y"
                )
            if final_missing != (x_final_missing or y_final_missing):
                raise ComArtifactValidationError(
                    f"processed_landmarks.csv frame {fi} landmark {lid}: "
                    "final_missing must equal x_final_missing OR y_final_missing"
                )


def _is_finite(text: str) -> bool:
    """Check if a numeric text string represents a finite value."""
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def _read_reviewed_events_strict(
    path: Path,
    pose_timestamps: dict[int, tuple[float, str, int, int | None]],
) -> tuple[_ReviewedEvent, ...]:
    """Read and validate reviewed_gait_events.csv strictly.

    Enforces:
    - exact REVIEWED_GAIT_EVENT_FIELDS header
    - unique nonempty event_ids
    - valid side (left/right)
    - event_type must be candidate_initial_contact
    - all booleans lowercase
    - automatic_frame_index nonnegative and exists in pose manifest
    - automatic_timestamp_seconds finite nonnegative and exact manifest lookup
    - reviewed_frame_index nonnegative and exists in pose manifest
    - reviewed_timestamp_seconds finite nonnegative and exact manifest lookup
    - exactly one of reviewed_accepted/reviewed_rejected is true
    - manual_event_review_status in {unreviewed, retain_rejection, promote_to_candidate}
    - resolution_disposition in {accepted_unchanged, corrected,
      promoted_from_rejected_candidate, rejected, replaced}
    - nonempty source
    """
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != REVIEWED_GAIT_EVENT_FIELDS:
                raise ComArtifactValidationError(
                    "reviewed_gait_events.csv header must exactly match "
                    "REVIEWED_GAIT_EVENT_FIELDS"
                )
            events: list[_ReviewedEvent] = []
            seen_ids: set[str] = set()
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(
                    row[field] is None for field in REVIEWED_GAIT_EVENT_FIELDS
                ):
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: malformed columns"
                    )

                event_id = row["event_id"]
                if not event_id:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        "event_id must be nonempty"
                    )
                if event_id in seen_ids:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        f"duplicate event_id {event_id}"
                    )
                seen_ids.add(event_id)

                automatic_event_id = row["automatic_event_id"]
                if not automatic_event_id:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        "automatic_event_id must be nonempty"
                    )

                side = row["side"]
                if side not in {"left", "right"}:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        f"invalid side {side!r}"
                    )

                event_type = row["event_type"]
                if event_type != "candidate_initial_contact":
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        f"event_type must be candidate_initial_contact, "
                        f"got {event_type!r}"
                    )

                auto_frame = _parse_int(
                    row["automatic_frame_index"],
                    "automatic_frame_index",
                    row_number,
                    "reviewed_gait_events.csv",
                )
                if auto_frame < 0:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        "automatic_frame_index must be nonnegative"
                    )
                if auto_frame not in pose_timestamps:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        f"automatic_frame_index {auto_frame} not in "
                        "pose_frames.csv manifest"
                    )
                auto_ts = _parse_float(
                    row["automatic_timestamp_seconds"],
                    "automatic_timestamp_seconds",
                    row_number,
                    "reviewed_gait_events.csv",
                )
                if auto_ts is None:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        "automatic_timestamp_seconds must be finite"
                    )
                if auto_ts < 0.0:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        "automatic_timestamp_seconds must be nonnegative"
                    )
                # Exact manifest lookup for automatic timestamp
                expected_auto_ts = pose_timestamps[auto_frame][0]
                if abs(auto_ts - expected_auto_ts) > 1e-9:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        f"automatic_timestamp_seconds {auto_ts} does not "
                        f"match pose_frames timestamp {expected_auto_ts} "
                        f"for frame {auto_frame}"
                    )

                auto_disposition = row["automatic_disposition"]
                auto_quality = row["automatic_quality"]
                if auto_quality not in {"high", "review", "low"}:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        f"invalid automatic_quality {auto_quality!r}"
                    )

                auto_peak = _parse_float(
                    row["automatic_peak_value"],
                    "automatic_peak_value",
                    row_number,
                    "reviewed_gait_events.csv",
                )
                if auto_peak is None:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        "automatic_peak_value must be finite"
                    )

                auto_prominence = _parse_float(
                    row["automatic_prominence"],
                    "automatic_prominence",
                    row_number,
                    "reviewed_gait_events.csv",
                    nullable=True,
                )

                auto_rejection = _parse_tuple_pipe(row["automatic_rejection_reasons"])

                manual_review_status = row["manual_event_review_status"]
                if manual_review_status not in {
                    "unreviewed",
                    "retain_rejection",
                    "promote_to_candidate",
                }:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        f"invalid manual_event_review_status "
                        f"{manual_review_status!r}"
                    )
                stride_review_prov = row["stride_review_provenance"]

                rev_frame = _parse_int(
                    row["reviewed_frame_index"],
                    "reviewed_frame_index",
                    row_number,
                    "reviewed_gait_events.csv",
                )
                if rev_frame < 0:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        "reviewed_frame_index must be nonnegative"
                    )

                rev_ts = _parse_float(
                    row["reviewed_timestamp_seconds"],
                    "reviewed_timestamp_seconds",
                    row_number,
                    "reviewed_gait_events.csv",
                )
                if rev_ts is None:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        "reviewed_timestamp_seconds must be finite"
                    )
                if rev_ts < 0.0:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        "reviewed_timestamp_seconds must be nonnegative"
                    )

                # Reviewed timestamp exact pose lookup
                if rev_frame not in pose_timestamps:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        f"reviewed_frame_index {rev_frame} not in "
                        "pose_frames.csv"
                    )
                expected_ts = pose_timestamps[rev_frame][0]
                if abs(rev_ts - expected_ts) > 1e-9:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        f"reviewed_timestamp_seconds {rev_ts} does not "
                        f"match pose_frames timestamp {expected_ts} "
                        f"for frame {rev_frame}"
                    )

                rev_accepted = _parse_bool_lower(
                    row["reviewed_accepted"],
                    "reviewed_accepted",
                    row_number,
                    "reviewed_gait_events.csv",
                )
                rev_rejected = _parse_bool_lower(
                    row["reviewed_rejected"],
                    "reviewed_rejected",
                    row_number,
                    "reviewed_gait_events.csv",
                )
                rev_included = _parse_bool_lower(
                    row["reviewed_included_in_stride"],
                    "reviewed_included_in_stride",
                    row_number,
                    "reviewed_gait_events.csv",
                )

                # Consistency: exactly one of accepted/rejected must be true
                if not (rev_accepted ^ rev_rejected):
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        "exactly one of reviewed_accepted/reviewed_rejected "
                        "must be true"
                    )
                # Included implies accepted
                if rev_included and not rev_accepted:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        "reviewed_included_in_stride is true but "
                        "reviewed_accepted is false"
                    )

                rev_quality = row["reviewed_quality"]
                if rev_quality not in {"high", "review", "low"}:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        f"invalid reviewed_quality {rev_quality!r}"
                    )

                resolution_disp = row["resolution_disposition"]
                _VALID_RESOLUTION_DISPOSITIONS = frozenset(
                    {
                        "accepted_unchanged",
                        "corrected",
                        "promoted_from_rejected_candidate",
                        "rejected",
                        "replaced",
                    }
                )
                if resolution_disp not in _VALID_RESOLUTION_DISPOSITIONS:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        f"invalid resolution_disposition {resolution_disp!r}"
                    )
                replaces_id = _parse_opt_str(row["replaces_event_id"])
                replaced_by_id = _parse_opt_str(row["replaced_by_event_id"])
                source = row["source"]
                if not source:
                    raise ComArtifactValidationError(
                        f"reviewed_gait_events.csv row {row_number}: "
                        "source must be nonempty"
                    )
                review_notes = _parse_tuple_pipe(row["review_notes"])

                events.append(
                    _ReviewedEvent(
                        event_id=event_id,
                        automatic_event_id=automatic_event_id,
                        side=side,
                        event_type=event_type,
                        automatic_frame_index=auto_frame,
                        automatic_timestamp_seconds=auto_ts,
                        automatic_disposition=auto_disposition,
                        automatic_quality=auto_quality,
                        automatic_peak_value=auto_peak,
                        automatic_prominence=auto_prominence,
                        automatic_rejection_reasons=auto_rejection,
                        manual_event_review_status=manual_review_status,
                        stride_review_provenance=stride_review_prov,
                        reviewed_frame_index=rev_frame,
                        reviewed_timestamp_seconds=rev_ts,
                        reviewed_accepted=rev_accepted,
                        reviewed_rejected=rev_rejected,
                        reviewed_included_in_stride=rev_included,
                        reviewed_quality=rev_quality,
                        resolution_disposition=resolution_disp,
                        replaces_event_id=replaces_id,
                        replaced_by_event_id=replaced_by_id,
                        source=source,
                        review_notes=review_notes,
                    )
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ComArtifactValidationError(
            f"Could not read reviewed_gait_events.csv: {exc}"
        ) from exc

    if not events:
        raise ComArtifactValidationError(
            "reviewed_gait_events.csv must contain at least one row"
        )
    return tuple(events)


def _read_reviewed_strides_strict(
    path: Path,
    reviewed_events: tuple[_ReviewedEvent, ...],
) -> tuple[_ReviewedStride, ...]:
    """Read and validate reviewed_strides.csv strictly.

    Enforces:
    - exact REVIEWED_STRIDE_FIELDS header in that order
    - every field parsed including tuple pipes
    - unique nonempty stride_ids
    - valid side/quality
    - start_frame < end_frame
    - start_timestamp < end_timestamp
    - positive duration equals difference
    - nonnegative contralateral_event_count
    - nonempty source and automatic_stride_id
    - review_intent in {accept, correct}
    - endpoints resolve to reviewed accepted+included candidate IC events
    - event IDs/frames/times/sides match
    """
    events_by_id: dict[str, _ReviewedEvent] = {e.event_id: e for e in reviewed_events}

    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != REVIEWED_STRIDE_FIELDS:
                raise ComArtifactValidationError(
                    "reviewed_strides.csv header must exactly match "
                    "STRIDE_FIELDS + (automatic_stride_id, review_intent, "
                    "review_changes, provenance_notes)"
                )
            strides: list[_ReviewedStride] = []
            seen_ids: set[str] = set()
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(
                    row[field] is None for field in REVIEWED_STRIDE_FIELDS
                ):
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: malformed columns"
                    )

                stride_id = row["stride_id"]
                if not stride_id:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        "stride_id must be nonempty"
                    )
                if stride_id in seen_ids:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"duplicate stride_id {stride_id}"
                    )
                seen_ids.add(stride_id)

                side = row["side"]
                if side not in {"left", "right"}:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: invalid side {side!r}"
                    )

                start_event_id = row["start_event_id"]
                end_event_id = row["end_event_id"]

                start_frame = _parse_int(
                    row["start_frame"],
                    "start_frame",
                    row_number,
                    "reviewed_strides.csv",
                )
                end_frame = _parse_int(
                    row["end_frame"],
                    "end_frame",
                    row_number,
                    "reviewed_strides.csv",
                )
                if start_frame >= end_frame:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"start_frame {start_frame} must be < "
                        f"end_frame {end_frame}"
                    )

                start_ts = _parse_float(
                    row["start_timestamp_seconds"],
                    "start_timestamp_seconds",
                    row_number,
                    "reviewed_strides.csv",
                )
                end_ts = _parse_float(
                    row["end_timestamp_seconds"],
                    "end_timestamp_seconds",
                    row_number,
                    "reviewed_strides.csv",
                )
                if start_ts is None or end_ts is None:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        "start/end timestamps must be finite"
                    )
                if start_ts >= end_ts:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"start_timestamp {start_ts} must be < "
                        f"end_timestamp {end_ts}"
                    )

                duration = _parse_float(
                    row["duration_seconds"],
                    "duration_seconds",
                    row_number,
                    "reviewed_strides.csv",
                )
                if duration is None:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        "duration_seconds must be finite"
                    )
                if duration <= 0.0:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"duration_seconds must be positive, got {duration}"
                    )
                expected_duration = end_ts - start_ts
                if abs(duration - expected_duration) > 1e-9:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"duration_seconds {duration} does not equal "
                        f"end-start difference {expected_duration}"
                    )

                quality = row["quality"]
                if quality not in {"high", "review", "low"}:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"invalid quality {quality!r}"
                    )

                contralateral_id = row["contralateral_event_id"]
                if contralateral_id == "":
                    contralateral_id = None
                contralateral_count = _parse_int(
                    row["contralateral_event_count"],
                    "contralateral_event_count",
                    row_number,
                    "reviewed_strides.csv",
                )
                if contralateral_count < 0:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        "contralateral_event_count must be nonnegative"
                    )
                sequence_notes = _parse_tuple_pipe(row["sequence_notes"])
                source = row["source"]
                if not source:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        "source must be nonempty"
                    )
                review_status = row["review_status"]

                # Parse the four review-provenance extras
                automatic_stride_id = row["automatic_stride_id"]
                if not automatic_stride_id:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        "automatic_stride_id must be nonempty"
                    )
                review_intent = row["review_intent"]
                if review_intent not in {"accept", "correct"}:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"invalid review_intent {review_intent!r}"
                    )
                review_changes = row["review_changes"]
                provenance_notes = row["provenance_notes"]

                # Validate endpoints against reviewed events
                start_event = events_by_id.get(start_event_id)
                end_event = events_by_id.get(end_event_id)
                if start_event is None:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"start_event_id {start_event_id} not found in "
                        "reviewed_gait_events.csv"
                    )
                if end_event is None:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"end_event_id {end_event_id} not found in "
                        "reviewed_gait_events.csv"
                    )
                # Endpoints must be accepted + included + candidate IC
                if not start_event.reviewed_accepted:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"start_event {start_event_id} is not reviewed "
                        "accepted"
                    )
                if not start_event.reviewed_included_in_stride:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"start_event {start_event_id} is not included "
                        "in stride"
                    )
                if start_event.event_type != "candidate_initial_contact":
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"start_event {start_event_id} event_type is "
                        f"{start_event.event_type!r}, expected "
                        "candidate_initial_contact"
                    )
                if not end_event.reviewed_accepted:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"end_event {end_event_id} is not reviewed "
                        "accepted"
                    )
                if not end_event.reviewed_included_in_stride:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"end_event {end_event_id} is not included "
                        "in stride"
                    )
                if end_event.event_type != "candidate_initial_contact":
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"end_event {end_event_id} event_type is "
                        f"{end_event.event_type!r}, expected "
                        "candidate_initial_contact"
                    )
                # Side must match start event side
                if side != start_event.side:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"stride side {side!r} does not match start "
                        f"event side {start_event.side!r}"
                    )
                if side != end_event.side:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"stride side {side!r} does not match end "
                        f"event side {end_event.side!r}"
                    )
                # Frames must match events
                if start_frame != start_event.reviewed_frame_index:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"start_frame {start_frame} does not match "
                        f"start event reviewed_frame_index "
                        f"{start_event.reviewed_frame_index}"
                    )
                if end_frame != end_event.reviewed_frame_index:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"end_frame {end_frame} does not match "
                        f"end event reviewed_frame_index "
                        f"{end_event.reviewed_frame_index}"
                    )
                # Timestamps must match events
                if abs(start_ts - start_event.reviewed_timestamp_seconds) > 1e-9:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"start_timestamp {start_ts} does not match "
                        "start event reviewed timestamp"
                    )
                if abs(end_ts - end_event.reviewed_timestamp_seconds) > 1e-9:
                    raise ComArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"end_timestamp {end_ts} does not match "
                        "end event reviewed timestamp"
                    )

                strides.append(
                    _ReviewedStride(
                        stride_id=stride_id,
                        side=side,
                        start_event_id=start_event_id,
                        end_event_id=end_event_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        start_timestamp_seconds=start_ts,
                        end_timestamp_seconds=end_ts,
                        duration_seconds=duration,
                        quality=quality,
                        contralateral_event_id=contralateral_id,
                        contralateral_event_count=contralateral_count,
                        sequence_notes=sequence_notes,
                        source=source,
                        review_status=review_status,
                        automatic_stride_id=automatic_stride_id,
                        review_intent=review_intent,
                        review_changes=review_changes,
                        provenance_notes=provenance_notes,
                    )
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ComArtifactValidationError(
            f"Could not read reviewed_strides.csv: {exc}"
        ) from exc

    if not strides:
        raise ComArtifactValidationError(
            "reviewed_strides.csv must contain at least one row"
        )
    return tuple(strides)


# ---------------------------------------------------------------------------
# Internal: metadata validation
# ---------------------------------------------------------------------------


def _validate_preprocessing_metadata(
    directory: Path,
    pose_frames_path: Path,
    pose_hash: str,
    processed_path: Path,
    processed_hash: str,
) -> dict[str, Any]:
    """Validate preprocessing_metadata.json strict contracts.

    Enforces:
    - schema_version == PREPROCESSING_SCHEMA_VERSION
    - algorithm_version == PREPROCESSING_ALGORITHM_VERSION
    - inputs.pose_frames.csv: path is nonempty string whose
      Path(...).name equals "pose_frames.csv"; hash matches actual
    - outputs.processed_landmarks.csv: path is nonempty string whose
      Path(...).name equals "processed_landmarks.csv"; hash matches actual

    Returns the metadata object for downstream use.
    """
    meta = _load_json(
        directory / "preprocessing_metadata.json",
        "preprocessing_metadata.json",
    )

    sv = meta.get("schema_version")
    if sv != PREPROCESSING_SCHEMA_VERSION:
        raise ComArtifactValidationError(
            f"preprocessing_metadata.json schema_version must be "
            f"{PREPROCESSING_SCHEMA_VERSION}, got {sv}"
        )

    av = meta.get("algorithm_version")
    if av != PREPROCESSING_ALGORITHM_VERSION:
        raise ComArtifactValidationError(
            f"preprocessing_metadata.json algorithm_version must be "
            f"{PREPROCESSING_ALGORITHM_VERSION!r}, got {av!r}"
        )

    # Validate inputs.pose_frames.csv
    inputs_section = meta.get("inputs", {})
    if not isinstance(inputs_section, dict):
        raise ComArtifactValidationError(
            "preprocessing_metadata.json inputs must be an object"
        )
    pf_info = inputs_section.get("pose_frames.csv")
    if not isinstance(pf_info, dict):
        raise ComArtifactValidationError(
            "preprocessing_metadata.json must contain "
            "inputs.pose_frames.csv as an object"
        )
    # Validate path is nonempty and basename matches
    pf_path_str = pf_info.get("path")
    if not isinstance(pf_path_str, str) or not pf_path_str:
        raise ComArtifactValidationError(
            "preprocessing_metadata.json inputs.pose_frames.csv "
            "must have a nonempty path string"
        )
    if Path(pf_path_str).name != "pose_frames.csv":
        raise ComArtifactValidationError(
            "preprocessing_metadata.json inputs.pose_frames.csv "
            f"path basename {Path(pf_path_str).name!r} does not match "
            "canonical 'pose_frames.csv'"
        )
    stored_pf_hash = pf_info.get("sha256")
    if not isinstance(stored_pf_hash, str) or not stored_pf_hash:
        raise ComArtifactValidationError(
            "preprocessing_metadata.json inputs.pose_frames.csv "
            "must have a sha256 string"
        )
    if stored_pf_hash != pose_hash:
        raise ComArtifactValidationError(
            "pose_frames.csv hash does not match "
            "preprocessing_metadata.json inputs.pose_frames.csv.sha256"
        )

    # Validate outputs.processed_landmarks.csv
    outputs_section = meta.get("outputs", {})
    if not isinstance(outputs_section, dict):
        raise ComArtifactValidationError(
            "preprocessing_metadata.json outputs must be an object"
        )
    pl_info = outputs_section.get("processed_landmarks.csv")
    if not isinstance(pl_info, dict):
        raise ComArtifactValidationError(
            "preprocessing_metadata.json must contain "
            "outputs.processed_landmarks.csv as an object"
        )
    # Validate path is nonempty and basename matches
    pl_path_str = pl_info.get("path")
    if not isinstance(pl_path_str, str) or not pl_path_str:
        raise ComArtifactValidationError(
            "preprocessing_metadata.json outputs.processed_landmarks.csv "
            "must have a nonempty path string"
        )
    if Path(pl_path_str).name != "processed_landmarks.csv":
        raise ComArtifactValidationError(
            "preprocessing_metadata.json outputs.processed_landmarks.csv "
            f"path basename {Path(pl_path_str).name!r} does not match "
            "canonical 'processed_landmarks.csv'"
        )
    stored_pl_hash = pl_info.get("sha256")
    if not isinstance(stored_pl_hash, str) or not stored_pl_hash:
        raise ComArtifactValidationError(
            "preprocessing_metadata.json outputs.processed_landmarks.csv "
            "must have a sha256 string"
        )
    if stored_pl_hash != processed_hash:
        raise ComArtifactValidationError(
            "processed_landmarks.csv hash does not match "
            "preprocessing_metadata.json "
            "outputs.processed_landmarks.csv.sha256"
        )

    return meta


def _validate_review_resolution_metadata(
    directory: Path,
    rge_path: Path,
    rge_hash: str,
    rs_path: Path,
    rs_hash: str,
    preproc_meta_path: Path,
    preproc_meta_hash: str,
    pose_frames_path: Path,
    pose_hash: str,
    expected_stride_count: int,
) -> dict[str, Any]:
    """Validate review_resolution_metadata.json strict contracts.

    Enforces:
    - schema_version == REVIEW_RESOLUTION_SCHEMA_VERSION
    - algorithm_version == REVIEW_RESOLUTION_ALGORITHM_VERSION
    - outputs reviewed events/strides: path is nonempty string whose
      Path(...).name equals canonical basename; hash matches actual
    - inputs.preprocessing_metadata.json: path is nonempty string whose
      Path(...).name equals "preprocessing_metadata.json"; hash matches actual
    - timestamp_source: path is nonempty string whose Path(...).name
      equals "pose_frames.csv"; hash matches actual
    - blocking_unresolved must be a list and empty
    - scientific_unresolved must be a list of strings
    - counts.reviewed_strides equals actual stride count

    Returns the metadata object for downstream use.
    """
    meta = _load_json(
        directory / "review_resolution_metadata.json",
        "review_resolution_metadata.json",
    )

    sv = meta.get("schema_version")
    if sv != REVIEW_RESOLUTION_SCHEMA_VERSION:
        raise ComArtifactValidationError(
            f"review_resolution_metadata.json schema_version must be "
            f"{REVIEW_RESOLUTION_SCHEMA_VERSION}, got {sv}"
        )

    av = meta.get("algorithm_version")
    if av != REVIEW_RESOLUTION_ALGORITHM_VERSION:
        raise ComArtifactValidationError(
            f"review_resolution_metadata.json algorithm_version must be "
            f"{REVIEW_RESOLUTION_ALGORITHM_VERSION!r}, got {av!r}"
        )

    # Validate outputs
    outputs_section = meta.get("outputs", {})
    if not isinstance(outputs_section, dict):
        raise ComArtifactValidationError(
            "review_resolution_metadata.json outputs must be an object"
        )
    for canonical_name, actual_hash, _actual_path in [
        ("reviewed_gait_events.csv", rge_hash, rge_path),
        ("reviewed_strides.csv", rs_hash, rs_path),
    ]:
        info = outputs_section.get(canonical_name)
        if not isinstance(info, dict):
            raise ComArtifactValidationError(
                f"review_resolution_metadata.json must contain "
                f"outputs.{canonical_name} as an object"
            )
        # Validate path is nonempty and basename matches
        out_path_str = info.get("path")
        if not isinstance(out_path_str, str) or not out_path_str:
            raise ComArtifactValidationError(
                f"review_resolution_metadata.json "
                f"outputs.{canonical_name} must have a nonempty path string"
            )
        if Path(out_path_str).name != canonical_name:
            raise ComArtifactValidationError(
                f"review_resolution_metadata.json "
                f"outputs.{canonical_name} path basename "
                f"{Path(out_path_str).name!r} does not match "
                f"canonical {canonical_name!r}"
            )
        stored_hash = info.get("sha256")
        if not isinstance(stored_hash, str) or not stored_hash:
            raise ComArtifactValidationError(
                f"review_resolution_metadata.json "
                f"outputs.{canonical_name} must have a sha256 string"
            )
        if stored_hash != actual_hash:
            raise ComArtifactValidationError(
                f"{canonical_name} hash does not match "
                f"review_resolution_metadata.json "
                f"outputs.{canonical_name}.sha256"
            )

    # Validate inputs.preprocessing_metadata.json
    inputs_section = meta.get("inputs", {})
    if not isinstance(inputs_section, dict):
        raise ComArtifactValidationError(
            "review_resolution_metadata.json inputs must be an object"
        )
    pm_info = inputs_section.get("preprocessing_metadata.json")
    if not isinstance(pm_info, dict):
        raise ComArtifactValidationError(
            "review_resolution_metadata.json must contain "
            "inputs.preprocessing_metadata.json as an object"
        )
    # Validate path is nonempty and basename matches
    pm_path_str = pm_info.get("path")
    if not isinstance(pm_path_str, str) or not pm_path_str:
        raise ComArtifactValidationError(
            "review_resolution_metadata.json "
            "inputs.preprocessing_metadata.json must have a nonempty path string"
        )
    if Path(pm_path_str).name != "preprocessing_metadata.json":
        raise ComArtifactValidationError(
            "review_resolution_metadata.json "
            "inputs.preprocessing_metadata.json path basename "
            f"{Path(pm_path_str).name!r} does not match "
            "canonical 'preprocessing_metadata.json'"
        )
    stored_pm_hash = pm_info.get("sha256")
    if not isinstance(stored_pm_hash, str) or not stored_pm_hash:
        raise ComArtifactValidationError(
            "review_resolution_metadata.json "
            "inputs.preprocessing_metadata.json must have a sha256 string"
        )
    if stored_pm_hash != preproc_meta_hash:
        raise ComArtifactValidationError(
            "preprocessing_metadata.json hash does not match "
            "review_resolution_metadata.json "
            "inputs.preprocessing_metadata.json.sha256"
        )

    # Validate timestamp_source
    ts_section = meta.get("timestamp_source")
    if not isinstance(ts_section, dict):
        raise ComArtifactValidationError(
            "review_resolution_metadata.json must contain timestamp_source as an object"
        )
    # Validate path is nonempty and basename matches
    ts_path_str = ts_section.get("path")
    if not isinstance(ts_path_str, str) or not ts_path_str:
        raise ComArtifactValidationError(
            "review_resolution_metadata.json timestamp_source "
            "must have a nonempty path string"
        )
    if Path(ts_path_str).name != "pose_frames.csv":
        raise ComArtifactValidationError(
            "review_resolution_metadata.json timestamp_source "
            f"path basename {Path(ts_path_str).name!r} does not match "
            "canonical 'pose_frames.csv'"
        )
    ts_hash = ts_section.get("sha256")
    if not isinstance(ts_hash, str) or not ts_hash:
        raise ComArtifactValidationError(
            "review_resolution_metadata.json timestamp_source must have a sha256 string"
        )
    if ts_hash != pose_hash:
        raise ComArtifactValidationError(
            "pose_frames.csv hash does not match "
            "review_resolution_metadata.json timestamp_source.sha256"
        )

    # Validate blocking_unresolved: must be a list and empty
    blocking = meta.get("blocking_unresolved")
    if not isinstance(blocking, list):
        raise ComArtifactValidationError(
            "review_resolution_metadata.json blocking_unresolved must be a list"
        )
    if len(blocking) != 0:
        raise ComArtifactValidationError(
            f"review_resolution_metadata.json blocking_unresolved "
            f"must be empty, got {len(blocking)} items"
        )

    # Validate scientific_unresolved: must be a list of strings
    sci = meta.get("scientific_unresolved")
    if not isinstance(sci, list):
        raise ComArtifactValidationError(
            "review_resolution_metadata.json scientific_unresolved must be a list"
        )
    for item in sci:
        if not isinstance(item, str):
            raise ComArtifactValidationError(
                "review_resolution_metadata.json scientific_unresolved "
                "entries must be strings"
            )

    # Validate counts.reviewed_strides if present
    counts_section = meta.get("counts", {})
    if isinstance(counts_section, dict):
        stored_stride_count = counts_section.get("reviewed_strides")
        if (
            stored_stride_count is not None
            and stored_stride_count != expected_stride_count
        ):
            raise ComArtifactValidationError(
                f"review_resolution_metadata.json counts.reviewed_strides "
                f"is {stored_stride_count} but actual stride count is "
                f"{expected_stride_count}"
            )

    return meta


# ---------------------------------------------------------------------------
# Internal: input validation orchestrator
# ---------------------------------------------------------------------------


def _validate_all_inputs(
    directory: Path,
) -> tuple[
    _InputPaths,
    dict[str, str],
    dict[int, tuple[float, str, int, int | None]],
    dict[int, dict[int, dict[str, str]]],
    tuple[_ReviewedEvent, ...],
    tuple[_ReviewedStride, ...],
    dict[str, Any],
    dict[str, Any],
]:
    """Resolve paths, snapshot hashes, parse and validate all six inputs.

    Returns all parsed structures needed for computation plus the two
    metadata objects for downstream propagation.
    """
    inputs = _resolve_inputs(directory)
    input_hashes = _snapshot_input_hashes(inputs)
    _verify_output_no_alias(directory, inputs)

    # Parse pose frames
    pose_manifest = _read_pose_frames_strict(inputs.pose_frames)

    # Parse processed landmarks
    processed_data = _read_processed_landmarks_strict(
        inputs.processed_landmarks, pose_manifest
    )

    # Parse reviewed events
    reviewed_events = _read_reviewed_events_strict(
        inputs.reviewed_gait_events, pose_manifest
    )

    # Parse reviewed strides
    reviewed_strides = _read_reviewed_strides_strict(
        inputs.reviewed_strides, reviewed_events
    )

    # Validate preprocessing_metadata
    preproc_meta = _validate_preprocessing_metadata(
        directory,
        inputs.pose_frames,
        input_hashes["pose_frames.csv"],
        inputs.processed_landmarks,
        input_hashes["processed_landmarks.csv"],
    )

    # Validate review_resolution_metadata
    rr_meta = _validate_review_resolution_metadata(
        directory,
        inputs.reviewed_gait_events,
        input_hashes["reviewed_gait_events.csv"],
        inputs.reviewed_strides,
        input_hashes["reviewed_strides.csv"],
        inputs.preprocessing_metadata,
        input_hashes["preprocessing_metadata.json"],
        inputs.pose_frames,
        input_hashes["pose_frames.csv"],
        expected_stride_count=len(reviewed_strides),
    )

    return (
        inputs,
        input_hashes,
        pose_manifest,
        processed_data,
        reviewed_events,
        reviewed_strides,
        preproc_meta,
        rr_meta,
    )


# ---------------------------------------------------------------------------
# Internal: COM computation for all frames
# ---------------------------------------------------------------------------


def _compute_frame_coms(
    processed_data: dict[int, dict[int, dict[str, str]]],
    pose_timestamps: dict[int, tuple[float, str, int, int | None]],
    config: ComEstimationConfig,
) -> tuple[FrameComResult, ...]:
    """Compute COM for every frame in pose_frames.csv order.

    Nonfinite processed x/y values are treated as validation errors and
    raise ComArtifactValidationError rather than being silently skipped.
    """
    results: list[FrameComResult] = []
    for fi in sorted(pose_timestamps.keys()):
        ts, status, _lc, _bt = pose_timestamps[fi]
        landmark_rows = processed_data.get(fi, {})

        # Build Point2D and provenance for each landmark
        landmarks: dict[str, Point2D] = {}
        provenance: dict[str, LandmarkProvenance] = {}

        for lid in sorted(landmark_rows.keys()):
            row = landmark_rows[lid]
            lm_name = row["landmark_name"]
            proc_x_str = row.get("processed_x_normalized", "")
            proc_y_str = row.get("processed_y_normalized", "")

            prov = create_landmark_provenance_from_processed(row)

            if proc_x_str == "" or proc_y_str == "":
                # Missing processed coordinates: record provenance but
                # do not contribute to COM
                provenance[lm_name] = prov
                continue

            proc_x = float(proc_x_str)
            proc_y = float(proc_y_str)

            # Nonfinite processed coordinates are validation errors
            if not math.isfinite(proc_x):
                raise ComArtifactValidationError(
                    f"Frame {fi} landmark {lm_name}: "
                    f"processed_x_normalized is nonfinite ({proc_x_str})"
                )
            if not math.isfinite(proc_y):
                raise ComArtifactValidationError(
                    f"Frame {fi} landmark {lm_name}: "
                    f"processed_y_normalized is nonfinite ({proc_y_str})"
                )

            landmarks[lm_name] = Point2D(proc_x, proc_y)
            provenance[lm_name] = prov

        frame_result = estimate_frame_com(
            processed_landmarks=landmarks,
            landmark_provenance=provenance,
            config=config,
            frame_index=fi,
            timestamp_seconds=ts,
            frame_status=status,
        )
        results.append(frame_result)

    return tuple(results)


# ---------------------------------------------------------------------------
# Internal: CSV output writers
# ---------------------------------------------------------------------------


def _write_com_proxy(path: Path, frame_results: tuple[FrameComResult, ...]) -> None:
    """Write com_proxy.csv with per-frame COM and per-segment details."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(COM_PROXY_FIELDS))
        writer.writeheader()
        for fr in frame_results:
            # Base fields
            prov_masses = fr.provenance_mass_totals()
            row: dict[str, Any] = {
                "frame_index": fr.frame_index,
                "timestamp_seconds": fr.timestamp_seconds,
                "frame_status": fr.frame_status,
                "com_x": "" if fr.com is None else fr.com.x,
                "com_y": "" if fr.com is None else fr.com.y,
                "mass_coverage": fr.mass_coverage,
                "model_total_mass": fr.model_total_mass,
                "usable": _csv_bool_lower(fr.usable),
                "usable_segments": "|".join(fr.usable_segments()),
                "missing_segments": "|".join(fr.missing_segments()),
                "contributors_raw_observed": "",
                "contributors_x_interpolated": "",
                "contributors_y_interpolated": "",
                "contributors_x_smoothing_changed": "",
                "contributors_y_smoothing_changed": "",
                "contributors_x_smoothing_support_interpolation": "",
                "contributors_y_smoothing_support_interpolation": "",
                "contributors_other_qc_limited": "",
                "mass_x_interpolated": prov_masses.get("x_interpolated", 0.0),
                "mass_y_interpolated": prov_masses.get("y_interpolated", 0.0),
                "mass_x_smoothing_changed": prov_masses.get("x_smoothing_changed", 0.0),
                "mass_y_smoothing_changed": prov_masses.get("y_smoothing_changed", 0.0),
                "mass_x_smoothing_support_interpolation": prov_masses.get(
                    "x_smoothing_support_interpolation", 0.0
                ),
                "mass_y_smoothing_support_interpolation": prov_masses.get(
                    "y_smoothing_support_interpolation", 0.0
                ),
                "mass_other_qc_limited": prov_masses.get("other_qc_limited", 0.0),
                "mass_missing": prov_masses.get("missing", 0.0),
            }
            # Aggregate contributors across all usable segments
            all_contrib_raw: set[str] = set()
            all_contrib_xi: set[str] = set()
            all_contrib_yi: set[str] = set()
            all_contrib_xsc: set[str] = set()
            all_contrib_ysc: set[str] = set()
            all_contrib_xssi: set[str] = set()
            all_contrib_yssi: set[str] = set()
            all_contrib_other: set[str] = set()
            for sr in fr.segment_results:
                if not sr.usable:
                    continue
                prov = sr.provenance
                if prov.all_raw_observed:
                    all_contrib_raw.update(prov.contributors)
                if prov.any_x_interpolated:
                    all_contrib_xi.update(prov.contributors)
                if prov.any_y_interpolated:
                    all_contrib_yi.update(prov.contributors)
                if prov.any_x_smoothing_changed:
                    all_contrib_xsc.update(prov.contributors)
                if prov.any_y_smoothing_changed:
                    all_contrib_ysc.update(prov.contributors)
                if prov.any_x_smoothing_support_interpolation:
                    all_contrib_xssi.update(prov.contributors)
                if prov.any_y_smoothing_support_interpolation:
                    all_contrib_yssi.update(prov.contributors)
                if prov.other_qc_limited:
                    all_contrib_other.update(prov.contributors)
            row["contributors_raw_observed"] = "|".join(sorted(all_contrib_raw))
            row["contributors_x_interpolated"] = "|".join(sorted(all_contrib_xi))
            row["contributors_y_interpolated"] = "|".join(sorted(all_contrib_yi))
            row["contributors_x_smoothing_changed"] = "|".join(sorted(all_contrib_xsc))
            row["contributors_y_smoothing_changed"] = "|".join(sorted(all_contrib_ysc))
            row["contributors_x_smoothing_support_interpolation"] = "|".join(
                sorted(all_contrib_xssi)
            )
            row["contributors_y_smoothing_support_interpolation"] = "|".join(
                sorted(all_contrib_yssi)
            )
            row["contributors_other_qc_limited"] = "|".join(sorted(all_contrib_other))

            # Per-segment fields
            seg_by_name = {s.segment_name: s for s in fr.segment_results}
            for seg_name in SEGMENT_NAMES:
                seg_result = seg_by_name.get(seg_name)
                if seg_result is not None and seg_result.com is not None:
                    row[f"seg_{seg_name}_com_x"] = seg_result.com.x
                    row[f"seg_{seg_name}_com_y"] = seg_result.com.y
                else:
                    row[f"seg_{seg_name}_com_x"] = ""
                    row[f"seg_{seg_name}_com_y"] = ""
                if seg_result is not None:
                    row[f"seg_{seg_name}_usable"] = _csv_bool_lower(seg_result.usable)
                    row[f"seg_{seg_name}_mass_fraction"] = seg_result.mass_fraction
                    if seg_name in UNSUPPORTED_SEGMENTS:
                        # Unsupported segments: no real contributors, flag unsupported
                        row[f"seg_{seg_name}_contributors"] = ""
                        row[f"seg_{seg_name}_qc_flags"] = "unsupported"
                    else:
                        row[f"seg_{seg_name}_contributors"] = "|".join(
                            seg_result.provenance.contributors
                        )
                        prov = seg_result.provenance
                        row[f"seg_{seg_name}_qc_flags"] = "|".join(
                            sorted(
                                flag
                                for flag in (
                                    "raw_observed" if prov.all_raw_observed else "",
                                    "x_interpolated" if prov.any_x_interpolated else "",
                                    "y_interpolated" if prov.any_y_interpolated else "",
                                    "x_smoothing_changed"
                                    if prov.any_x_smoothing_changed
                                    else "",
                                    "y_smoothing_changed"
                                    if prov.any_y_smoothing_changed
                                    else "",
                                    "x_smoothing_support_interpolation"
                                    if prov.any_x_smoothing_support_interpolation
                                    else "",
                                    "y_smoothing_support_interpolation"
                                    if prov.any_y_smoothing_support_interpolation
                                    else "",
                                    "other_qc_limited" if prov.other_qc_limited else "",
                                )
                                if flag
                            )
                        )
                else:
                    row[f"seg_{seg_name}_usable"] = "false"
                    row[f"seg_{seg_name}_mass_fraction"] = ""
                    row[f"seg_{seg_name}_contributors"] = ""
                    row[f"seg_{seg_name}_qc_flags"] = ""
            writer.writerow(row)


def _write_stride_com(
    path: Path,
    stride_samples: dict[str, tuple[StrideComSample, ...]],
    reviewed_strides: tuple[_ReviewedStride, ...],
) -> None:
    """Write stride_com.csv with original+normalized samples per stride.

    Receives typed reviewed strides and populates
    automatic_stride_id/review_intent/review_changes/provenance_notes
    from the _ReviewedStride records.
    """
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(STRIDE_COM_FIELDS))
        writer.writeheader()
        for stride in reviewed_strides:
            samples = stride_samples.get(stride.stride_id, ())
            for sample in samples:
                row: dict[str, Any] = {
                    "stride_id": stride.stride_id,
                    "side": stride.side,
                    "start_event_id": stride.start_event_id,
                    "end_event_id": stride.end_event_id,
                    "start_frame": stride.start_frame,
                    "end_frame": stride.end_frame,
                    "start_timestamp_seconds": stride.start_timestamp_seconds,
                    "end_timestamp_seconds": stride.end_timestamp_seconds,
                    "duration_seconds": stride.duration_seconds,
                    "quality": stride.quality,
                    "contralateral_event_id": stride.contralateral_event_id or "",
                    "contralateral_event_count": stride.contralateral_event_count,
                    "source": stride.source,
                    "review_status": stride.review_status,
                    "automatic_stride_id": stride.automatic_stride_id,
                    "review_intent": stride.review_intent,
                    "review_changes": stride.review_changes,
                    "provenance_notes": stride.provenance_notes,
                    "sample_kind": sample.sample_kind,
                    "normalized_index": ""
                    if sample.normalized_index is None
                    else sample.normalized_index,
                    "progression": sample.progression,
                    "method": sample.method,
                    "source_frame_index": ""
                    if sample.source_frame_index is None
                    else sample.source_frame_index,
                    "source_timestamp_seconds": ""
                    if sample.source_timestamp_seconds is None
                    else sample.source_timestamp_seconds,
                    "target_timestamp_seconds": ""
                    if sample.target_timestamp_seconds is None
                    else sample.target_timestamp_seconds,
                    "left_source_frame_index": ""
                    if sample.left_source_frame_index is None
                    else sample.left_source_frame_index,
                    "left_source_timestamp_seconds": ""
                    if sample.left_source_timestamp_seconds is None
                    else sample.left_source_timestamp_seconds,
                    "right_source_frame_index": ""
                    if sample.right_source_frame_index is None
                    else sample.right_source_frame_index,
                    "right_source_timestamp_seconds": ""
                    if sample.right_source_timestamp_seconds is None
                    else sample.right_source_timestamp_seconds,
                    "com_x": "" if sample.com is None else sample.com.x,
                    "com_y": "" if sample.com is None else sample.com.y,
                    "usable": _csv_bool_lower(sample.usable),
                    "mass_coverage": sample.mass_coverage,
                    "min_endpoint_coverage": sample.min_endpoint_coverage,
                    "contributors": "|".join(sample.contributors),
                    "qc_flags": "|".join(sample.qc_flags),
                }
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Internal: diagnostic plot (lazy matplotlib, mandatory)
# ---------------------------------------------------------------------------


def _write_diagnostic(
    path: Path,
    frame_results: tuple[FrameComResult, ...],
    reviewed_strides: tuple[_ReviewedStride, ...],
    config: ComEstimationConfig,
) -> None:
    """Write com_diagnostic.png: COM x/y, coverage, and stride context.

    Plots:
    - COM x with below-coverage markers and missing-COM vertical markers
    - COM y with below-coverage markers and missing-COM vertical markers
    - Mass coverage with threshold line and missing-COM coverage-axis markers
    - Normalized stride x context (stride boundary spans)

    Title indicates 2D normalized image-plane COM proxy and research-only.

    Raises ComPipelineError if matplotlib is unavailable or save fails.
    """
    fig = None
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:
        raise ComPipelineError(
            "matplotlib is required for com_diagnostic.png but is not installed"
        ) from exc

    try:
        timestamps = [fr.timestamp_seconds for fr in frame_results]
        com_x = [
            fr.com.x if fr.com is not None and fr.usable else float("nan")
            for fr in frame_results
        ]
        com_y = [
            fr.com.y if fr.com is not None and fr.usable else float("nan")
            for fr in frame_results
        ]
        coverage = [fr.mass_coverage for fr in frame_results]

        # Classify frames for markers
        missing_com = [fr.com is None for fr in frame_results]
        missing_times = [t for t, m in zip(timestamps, missing_com, strict=True) if m]

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        # COM x (usable frames only as line)
        axes[0].plot(timestamps, com_x, "b-", linewidth=0.5, label="COM x (usable)")
        # Below-threshold finite markers
        bc_times = [
            t
            for t, fr in zip(timestamps, frame_results, strict=True)
            if fr.usable is False and fr.com is not None
        ]
        bc_x = [
            fr.com.x
            for fr in frame_results
            if fr.usable is False and fr.com is not None
        ]
        if bc_times:
            axes[0].plot(bc_times, bc_x, "rv", markersize=3, label="below threshold")
        # Missing-COM vertical line markers
        for mt in missing_times:
            axes[0].axvline(mt, color="red", linewidth=0.3, alpha=0.4)
        if missing_times:
            axes[0].plot([], [], "r-", linewidth=0.5, alpha=0.4, label="missing COM")
        # Stride boundary spans and IDs
        for s in reviewed_strides:
            axes[0].axvspan(
                s.start_timestamp_seconds,
                s.end_timestamp_seconds,
                alpha=0.1,
                color="green",
            )
            mid_t = (s.start_timestamp_seconds + s.end_timestamp_seconds) / 2.0
            axes[0].text(
                mid_t,
                axes[0].get_ylim()[0],
                s.stride_id,
                fontsize=4,
                ha="center",
                va="bottom",
                alpha=0.5,
            )
        axes[0].set_ylabel("COM x (norm)")
        axes[0].set_title(
            f"Represented-segment mass-weighted 2D centroid proxy - "
            f"{config.anthropometry_sex} - "
            f"coverage threshold {config.minimum_mass_coverage:.0%} - research-only"
        )
        axes[0].legend(loc="upper right")

        # COM y (usable frames only as line)
        axes[1].plot(timestamps, com_y, "r-", linewidth=0.5, label="COM y (usable)")
        bc_y = [
            fr.com.y
            for fr in frame_results
            if fr.usable is False and fr.com is not None
        ]
        if bc_times:
            axes[1].plot(bc_times, bc_y, "rv", markersize=3, label="below threshold")
        for mt in missing_times:
            axes[1].axvline(mt, color="red", linewidth=0.3, alpha=0.4)
        if missing_times:
            axes[1].plot([], [], "r-", linewidth=0.5, alpha=0.4, label="missing COM")
        for s in reviewed_strides:
            axes[1].axvspan(
                s.start_timestamp_seconds,
                s.end_timestamp_seconds,
                alpha=0.1,
                color="green",
            )
            mid_t = (s.start_timestamp_seconds + s.end_timestamp_seconds) / 2.0
            axes[1].text(
                mid_t,
                axes[1].get_ylim()[0],
                s.stride_id,
                fontsize=4,
                ha="center",
                va="bottom",
                alpha=0.5,
            )
        axes[1].set_ylabel("COM y (norm)")
        axes[1].legend(loc="upper right")

        # Mass coverage
        axes[2].plot(timestamps, coverage, "k-", linewidth=0.5, label="coverage")
        axes[2].axhline(
            config.minimum_mass_coverage,
            color="red",
            linestyle="--",
            linewidth=0.8,
            label=f"threshold={config.minimum_mass_coverage:.0%}",
        )
        axes[2].fill_between(
            timestamps,
            coverage,
            alpha=0.2,
            color="blue",
        )
        # Mark below-coverage frames
        bc_covs = [
            fr.mass_coverage
            for fr in frame_results
            if fr.usable is False and fr.com is not None
        ]
        if bc_times:
            axes[2].plot(
                bc_times,
                bc_covs,
                "rv",
                markersize=4,
                label="below threshold",
            )
        # Coverage-axis markers for missing COM
        if missing_times:
            axes[2].plot(
                missing_times,
                [1.02] * len(missing_times),
                "r|",
                markersize=6,
                label="missing COM",
            )
        for mt in missing_times:
            axes[2].axvline(mt, color="red", linewidth=0.3, alpha=0.4)
        axes[2].set_ylabel("Mass coverage")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend(loc="upper right")
        axes[2].set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
    except Exception as exc:
        raise ComPipelineError(f"Failed to write com_diagnostic.png: {exc}") from exc
    finally:
        if fig is not None:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Internal: metadata
# ---------------------------------------------------------------------------


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
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
    for name in ("gait-stability", "matplotlib"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _build_metadata(
    directory: Path,
    staging: Path,
    config: ComEstimationConfig,
    input_hashes: dict[str, str],
    frame_results: tuple[FrameComResult, ...],
    reviewed_strides: tuple[_ReviewedStride, ...],
    stride_sample_counts: dict[str, int],
    stride_original_counts: dict[str, int],
    preproc_meta: dict[str, Any],
    rr_meta: dict[str, Any],
    scientific_unresolved: list[str],
) -> dict[str, Any]:
    usable_count = sum(1 for fr in frame_results if fr.usable)
    total_count = len(frame_results)

    # Select the correct coefficient table and model total
    if config.anthropometry_sex == "male":
        selected_coefficients = DE_LEVA_MALE
        model_total = MODEL_MASS_TOTAL_MALE
    else:
        selected_coefficients = DE_LEVA_FEMALE
        model_total = MODEL_MASS_TOTAL_FEMALE

    # Build per-stride statistics with separated counts
    stride_stats: list[dict[str, Any]] = []
    for s in reviewed_strides:
        orig_count = stride_original_counts.get(s.stride_id, 0)
        stride_stats.append(
            {
                "stride_id": s.stride_id,
                "side": s.side,
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "start_timestamp_seconds": s.start_timestamp_seconds,
                "end_timestamp_seconds": s.end_timestamp_seconds,
                "duration_seconds": s.duration_seconds,
                "quality": s.quality,
                "normalized_sample_count": config.normalized_stride_samples,
                "original_sample_count": orig_count,
            }
        )

    output_hashes: dict[str, str | None] = {}
    for name in COM_OUTPUT_ARTIFACT_NAMES:
        if name == "com_metadata.json":
            output_hashes[name] = None  # self-null semantics
        else:
            try:
                output_hashes[name] = sha256_file(staging / name)
            except OSError:
                output_hashes[name] = None

    # Build segment endpoint definitions with proxy status
    segment_endpoint_definitions: dict[str, dict[str, Any]] = {}
    for seg_name in SEGMENT_NAMES:
        prox, dist = SEGMENT_ENDPOINTS[seg_name]

        # Classify each endpoint as direct MediaPipe landmark or derived
        def _endpoint_status(name: str) -> str:
            if name in ("_unsupported_head_proximal", "_unsupported_head_distal"):
                return "unsupported_no_source_compatible_vertex_and_neck_joint_center"
            if name in MEDIAPIPE_LANDMARK_NAMES:
                return "direct_media_pipe_landmark"
            if name == "shoulder_midpoint":
                return "derived_midpoint(left_shoulder, right_shoulder)"
            if name == "hip_midpoint":
                return "derived_midpoint(left_hip, right_hip)"
            if name == "left_index_pinky_midpoint_proxy":
                return "derived_midpoint_proxy(left_index, left_pinky)"
            if name == "right_index_pinky_midpoint_proxy":
                return "derived_midpoint_proxy(right_index, right_pinky)"
            return "unknown_proxy"

        # Approximation status by segment class
        _SEGMENT_APPROXIMATION: dict[str, str] = {
            "head": "unsupported",
            "trunk": "joint_centre_proxy",
            "left_upper_arm": "joint_centre_proxy",
            "right_upper_arm": "joint_centre_proxy",
            "left_forearm": "joint_centre_proxy",
            "right_forearm": "joint_centre_proxy",
            "left_hand": "approximate",
            "right_hand": "approximate",
            "left_thigh": "joint_centre_proxy",
            "right_thigh": "joint_centre_proxy",
            "left_shank": "joint_centre_proxy",
            "right_shank": "joint_centre_proxy",
            "left_foot": "approximate",
            "right_foot": "approximate",
        }

        segment_endpoint_definitions[seg_name] = {
            "proximal": {
                "name": prox,
                "proxy_status": _endpoint_status(prox),
            },
            "distal": {
                "name": dist,
                "proxy_status": _endpoint_status(dist),
            },
            "approximation_status": _SEGMENT_APPROXIMATION.get(seg_name, "unknown"),
        }

        # Published reference endpoints by segment class (anatomical
        # descriptions from de Leva 1996, not proxy landmarks).
        # For supported segments, these are the published model endpoints
        # that the proxy landmarks approximate.
        # For head: the exact source figure/reference must be consulted;
        # implementation omits head; do not invent or call nose vertex.
        published_reference_endpoints: dict[str, dict[str, str]] = {
            "head": {
                "proximal": (
                    "UNSUPPORTED - exact source figure/reference must be "
                    "consulted; nose is NOT the vertex; implementation "
                    "omits head"
                ),
                "distal": (
                    "UNSUPPORTED - neck joint-centre; not available from "
                    "standard MediaPipe Pose"
                ),
                "status": (
                    "unsupported_no_source_compatible_vertex_and_neck_joint_center"
                ),
            },
            "trunk": {
                "proximal": "bilateral shoulder joint-centre midpoint",
                "distal": "bilateral hip joint-centre midpoint",
            },
            "upper_arm": {
                "proximal": "shoulder joint-centre",
                "distal": "elbow joint-centre",
            },
            "forearm": {
                "proximal": "elbow joint-centre",
                "distal": "wrist joint-centre",
            },
            "hand": {
                "proximal": "wrist joint-centre",
                "distal": "third metacarpal head (source model endpoint); "
                "proxy uses midpoint of MediaPipe index and pinky points, "
                "an unvalidated distal hand endpoint surrogate",
            },
            "thigh": {
                "proximal": "hip joint-centre",
                "distal": "knee joint-centre",
            },
            "shank": {
                "proximal": "knee joint-centre",
                "distal": "ankle joint-centre",
            },
            "foot": {
                "proximal": "ankle joint-centre",
                "distal": "toe",
            },
        }

    # Build warnings list
    warnings_list: list[str] = []
    if config.anthropometry_sex == "female" and model_total < 1.0:
        warnings_list.append(
            f"Female model total is {model_total}; coverage threshold "
            f"1.0 can never pass"
        )
    for fr in frame_results:
        if fr.mass_coverage > 0.0 and fr.mass_coverage < config.minimum_mass_coverage:
            warnings_list.append(
                f"Frame {fr.frame_index} coverage {fr.mass_coverage:.4f} "
                f"below threshold {config.minimum_mass_coverage:.4f}"
            )

    return {
        "schema_version": COM_SCHEMA_VERSION,
        "algorithm_version": COM_ALGORITHM_VERSION,
        "scope": (
            "segment-weighted anthropometric COM proxy from processed pose "
            "landmarks; not a stability metric, clinical result, or "
            "force-plate-validated measurement"
        ),
        "run_id": uuid.uuid4().hex,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "inputs": {
            basename: {
                "path": str(directory / basename),
                "sha256": digest,
            }
            for basename, digest in input_hashes.items()
        },
        "config": {
            "anthropometry_sex": config.anthropometry_sex,
            "minimum_mass_coverage": config.minimum_mass_coverage,
            "normalized_stride_samples": config.normalized_stride_samples,
        },
        "frame_counts": {
            "total": total_count,
            "usable": usable_count,
            "unusable": total_count - usable_count,
            "usable_fraction": usable_count / total_count if total_count else 0.0,
        },
        "stride_statistics": stride_stats,
        "algorithm": {
            "model": "de Leva 1996 adjustments to Zatsiorsky-Seluyanov",
            "citation": (
                "Paolo de Leva, \"Adjustments to Zatsiorsky-Seluyanov's "
                'segment inertia parameters", Journal of Biomechanics '
                "29(9), 1223-1230 (1996), "
                "DOI 10.1016/0021-9290(95)00178-6"
            ),
            "citation_author": "Paolo de Leva",
            "citation_title": (
                "Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters"
            ),
            "citation_journal": "Journal of Biomechanics",
            "citation_volume": 29,
            "citation_issue": 9,
            "citation_pages": "1223-1230",
            "citation_year": 1996,
            "doi": "10.1016/0021-9290(95)00178-6",
            "selected_sex": config.anthropometry_sex,
            "coefficient_table": {
                segment: {
                    "mass_fraction": vals["mass"],
                    "centroid_ratio_r": vals["r"],
                    "centroid_formula": (
                        "centroid = proximal + r * (distal - proximal)"
                    ),
                }
                for segment, vals in selected_coefficients.items()
            },
            "model_total_mass": model_total,
            "model_total_mass_note": (
                "Sum of all segment mass fractions for the selected sex; "
                "male=1.0000, female=0.9999"
            ),
            "unsupported_segments": list(UNSUPPORTED_SEGMENTS),
            "unsupported_segments_note": (
                "Head is always unavailable because standard MediaPipe "
                "Pose lacks a defensible source-compatible vertex/neck "
                "joint-centre line. Nose is NOT the vertex. Head coefficient "
                "and mass are retained for provenance but never participate "
                "in frame COM calculations."
            ),
            "represented_mass_max": (
                REPRESENTED_MASS_MAX_MALE
                if config.anthropometry_sex == "male"
                else REPRESENTED_MASS_MAX_FEMALE
            ),
            "represented_mass_max_note": (
                "Theoretical maximum represented body-mass fraction when all "
                "supported segments are usable (female max .9331 after omitted "
                "head). Full model coverage cannot occur because head is "
                "unsupported. Empirical maximum from frame results is "
                f"{max((fr.mass_coverage for fr in frame_results), default=0.0):.4f}."
            ),
            "empirical_max_mass_coverage": max(
                (fr.mass_coverage for fr in frame_results), default=0.0
            ),
            "segments": len(SEGMENT_NAMES),
            "segment_names": list(SEGMENT_NAMES),
            "bilateral_segments": list(BILATERAL_SEGMENTS),
            "segment_endpoint_definitions": segment_endpoint_definitions,
            "published_reference_endpoints": published_reference_endpoints,
            "centroid_formula": (
                "COM = sum(mass_fraction_i * centroid_i) / "
                "sum(mass_fraction_i for usable segments); the denominator "
                "is the sum of represented usable mass fractions only, "
                "not normalized by total model mass or body mass. "
                "This is a represented-segment centroid, renormalized "
                "by represented usable mass."
            ),
            "mass_coverage_formula": (
                "mass_coverage = represented_body_mass_fraction = "
                "sum(mass_fraction_i for usable segments); this is "
                "unrenormalized and does NOT include unsupported segments "
                "(head). Female max after omitted head is .9331."
            ),
            "normalization": (
                "canonical timestamps for stride progression, not "
                "frame-number fractions; normalized grid uses "
                "config.normalized_stride_samples exact progressions "
                "from 0% to 100%"
            ),
            "no_gap_rule": (
                "normalized stride interpolation requires adjacent usable "
                "frames with consecutive frame_index values; non-consecutive "
                "usable frames do not bridge gaps"
            ),
            "identical_segment_set_rule": (
                "stride interpolation requires identical usable_segments() "
                "tuples at adjacent bracket frames before linear interpolation. "
                "If different, method=none, com=None, union contributors, and "
                "qc_flags includes 'represented_segment_set_changed'; centroids "
                "with different denominators are never blended."
            ),
            "coordinates": (
                "normalized image-plane coordinates (x right, y down); "
                "values are not constrained to [0, 1]; no z-coordinate "
                "or physical scale is used"
            ),
            "camera_view": ("single 2D side-view; depth is not reconstructed"),
            "projection_assumptions": (
                "requires static, near-sagittal, low-distortion, "
                "weak-perspective/equivalent endpoint-depth capture with "
                "minimal out-of-plane motion; user-established and not "
                "machine-verified. Affine segment-fraction projection is "
                "not generally valid under perspective depth differences."
            ),
        },
        "assumptions": {
            "anthropometric_coefficients": (
                "population averages from de Leva 1996, not individual measurements"
            ),
            "endpoints_are_not_joint_centers": (
                "MediaPipe 2D landmarks are used as segment endpoint "
                "proxies; they are not anatomical joint centers"
            ),
            "no_scale": (
                "no physical scale, camera calibration, or 3D reconstruction "
                "is performed"
            ),
            "progression_is_not_gait_cycle": (
                "normalized stride progression is time-based, not validated "
                "gait-cycle percentage"
            ),
        },
        "coverage": {
            "rationale": (
                "A frame is marked usable when its mass_coverage "
                "(represented_body_mass_fraction, sum of usable segment "
                "mass fractions) meets or exceeds the configured "
                "minimum_mass_coverage threshold. The default 0.90 "
                "threshold is an unvalidated engineering QC policy that "
                "bounds omitted published body-mass fraction to "
                "approximately/at most 10 percentage points under "
                "rounded coefficients; it is not a positional accuracy "
                "bound. When a custom threshold is used, the default "
                "0.90 rationale is still recorded as the reference QC "
                "policy. A zero-coverage frame is never usable regardless "
                "of threshold. Frames below threshold retain finite COM "
                "values but usable=False."
            ),
            "threshold": config.minimum_mass_coverage,
            "unrenormalized": (
                "mass_coverage is the raw represented_body_mass_fraction "
                "sum of usable mass fractions without rescaling to 1.0; "
                "this preserves the actual fraction of model mass that "
                "contributed to the represented-segment centroid. "
                "Unsupported segments (head) are excluded; female max "
                "after omitted head is .9331."
            ),
        },
        "qc_propagation": (
            "Per-segment QC flags use nonexclusive semantics: each flag "
            "(raw_observed, x_interpolated, y_interpolated, "
            "x_smoothing_changed, y_smoothing_changed, "
            "x_smoothing_support_interpolation, "
            "y_smoothing_support_interpolation, other_qc_limited, missing) "
            "is independently tracked. A single segment may have multiple "
            "QC flags simultaneously. Mass totals in com_proxy.csv are "
            "nonexclusive (sums per independent flag, not exclusive "
            "categories). Processed smoothed coordinates are never called "
            "raw. The exclusive provenance_category() is retained for "
            "display only."
        ),
        "frame_vs_stride_dependency": (
            "com_proxy.csv contains per-frame COM estimates independent "
            "of strides. stride_com.csv contains both original frame "
            "samples and normalized stride samples; normalized samples "
            "may include linear interpolation between bracket frames."
        ),
        "reviewed_strides_are_canonical_qc_windows": (
            "Reviewed strides are canonical Step4b segmentation/QC windows "
            "using nominal pose-frame timestamps. They are not "
            "force-plate-confirmed contacts, validated stride durations, "
            "or ground truth."
        ),
        "schemas": {
            "com_proxy.csv": {
                "columns": list(COM_PROXY_FIELDS),
                "boolean_serialization": "lowercase true/false",
                "missing_serialization": "blank CSV field",
            },
            "stride_com.csv": {
                "columns": list(STRIDE_COM_FIELDS),
                "boolean_serialization": "lowercase true/false",
                "missing_serialization": "blank CSV field",
            },
            "com_metadata.json": "this object",
        },
        "outputs": {
            name: {
                "path": str(directory / name),
                "sha256": digest,
            }
            for name, digest in output_hashes.items()
            if name != "com_metadata.json"
        }
        | {
            "com_metadata.json": {
                "path": str(directory / "com_metadata.json"),
                "sha256": None,
                "sha256_semantics": (
                    "not self-recorded because embedding a file's own "
                    "digest changes that digest"
                ),
            }
        },
        "upstream_preprocessing": {
            "schema_version": preproc_meta.get("schema_version"),
            "algorithm_version": preproc_meta.get("algorithm_version"),
            "config": preproc_meta.get("config"),
            "inherited_provenance": preproc_meta.get("inherited_provenance"),
        },
        "carried_scientific_unresolved": scientific_unresolved,
        "runtime": {
            "python_version": sys.version,
            "dependency_versions": _dependency_versions(),
            "git": _git_provenance(),
            "randomness": "none",
        },
        "counts": {
            "total_frames": total_count,
            "usable_frames": usable_count,
            "reviewed_strides": len(reviewed_strides),
        },
        "warnings": warnings_list if warnings_list else [],
        "validation_status": (
            "This COM proxy is a prerequisite metric derived from "
            "video-based pose estimates. It has not been validated "
            "against force plates, motion capture, or clinical "
            "measurements. It is NOT a stability metric, fall-risk "
            "assessment, clinical result, or ground-truth measurement."
        ),
        "limitations": [
            ("COM is a 2D image-plane proxy, not a 3D or laboratory measurement."),
            ("Anthropometric coefficients are population averages, not individual."),
            (
                "Coverage threshold is an engineering QC gate, not a "
                "validated accuracy bound."
            ),
            (
                "No physical scale, camera calibration, or gait-stability "
                "metric is produced."
            ),
            (
                "Stride normalization uses canonical timestamps; "
                "progression is not validated gait-cycle percentage."
            ),
            (
                "No camera calibration or physical scale; coordinates "
                "are not constrained to [0, 1]."
            ),
            (
                "Derived endpoints (midpoints) are arithmetic means of "
                "component landmarks, not anatomical joint centers."
            ),
            (
                "Step 3 interpolation and smoothing affect landmark "
                "positions and propagate through COM calculations."
            ),
            (
                "Head segment is unsupported; full model coverage "
                "cannot occur. Represented body mass fraction is at "
                "most .9306 (male) or .9331 (female)."
            ),
            (
                "Hand distal endpoint proxy uses midpoint of MediaPipe "
                "index and pinky points, an unvalidated distal hand "
                "endpoint surrogate, not anatomical hand midpoint or "
                "third metacarpal head."
            ),
            (
                "The COM proxy is a represented-segment centroid, not "
                "a whole-body center of mass."
            ),
            (
                "This output is not a fall-risk indicator, diagnostic "
                "tool, or ground-truth reference."
            ),
            (
                "Projection assumes static, near-sagittal, low-distortion, "
                "weak-perspective capture; not machine-verified."
            ),
        ],
    }


# ---------------------------------------------------------------------------
# Internal: transactional publish
# ---------------------------------------------------------------------------


def _publish(staging: Path, destination: Path) -> None:
    """Transactional publish with UUID backups and rollback."""
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for name in COM_OUTPUT_ARTIFACT_NAMES:
            target = destination / name
            if target.exists():
                backup = destination / f"{name}.backup-{uuid.uuid4().hex}"
                target.replace(backup)
                backups[target] = backup
        for name in COM_OUTPUT_ARTIFACT_NAMES:
            staged = staging / name
            if not staged.is_file():
                raise OSError(f"staged Step 5 artifact is missing: {staged}")
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
            "Could not publish complete Step 5 artifact set"
            + (f"; rollback may be incomplete: {detail}" if detail else "")
        ) from exc
    for backup in backups.values():
        with suppress(OSError):
            backup.unlink()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_com(
    artifact_directory: str | Path,
    config: ComEstimationConfig,
) -> ComArtifacts:
    """Estimate COM proxy for all frames and normalize per stride.

    Requires canonical Step 3 and Step 4b artifacts. Validates all contracts
    and hashes before processing. Publishes four output files as a
    transactional set.
    """
    # Strict config type check
    if not isinstance(config, ComEstimationConfig):
        raise TypeError(
            f"config must be ComEstimationConfig, got {type(config).__name__}"
        )

    directory = Path(artifact_directory).expanduser().resolve()
    if not directory.is_dir():
        raise ComArtifactValidationError(
            f"Artifact directory does not exist: {directory}"
        )

    # Phase 1: resolve paths, snapshot hashes, parse all inputs
    (
        inputs,
        input_hashes,
        pose_manifest,
        processed_data,
        reviewed_events,
        reviewed_strides,
        preproc_meta,
        rr_meta,
    ) = _validate_all_inputs(directory)

    # Extract carried scientific_unresolved from review_resolution_metadata
    scientific_unresolved: list[str] = list(rr_meta.get("scientific_unresolved", []))

    # Build pose_timestamps dict for compute (ts, status, lc, bt)
    pose_timestamps = pose_manifest

    # Phase 2: compute COM for all frames
    frame_results = _compute_frame_coms(processed_data, pose_timestamps, config)

    # Phase 3: normalize per stride
    stride_com_samples: dict[str, tuple[StrideComSample, ...]] = {}
    stride_sample_counts: dict[str, int] = {}
    stride_original_counts: dict[str, int] = {}
    for stride in reviewed_strides:
        samples = normalize_stride_com(
            frame_results=frame_results,
            stride_start_frame=stride.start_frame,
            stride_end_frame=stride.end_frame,
            stride_start_timestamp=stride.start_timestamp_seconds,
            stride_end_timestamp=stride.end_timestamp_seconds,
            config=config,
        )
        stride_com_samples[stride.stride_id] = samples
        # normalized_sample_count = config N (the normalized grid count)
        # original_sample_count = number of original (frame) samples
        norm_count = sum(1 for s in samples if s.sample_kind == "normalized")
        orig_count = sum(1 for s in samples if s.sample_kind == "original")
        stride_sample_counts[stride.stride_id] = norm_count
        stride_original_counts[stride.stride_id] = orig_count

    # Phase 4: stage and write outputs
    staging = Path(
        tempfile.mkdtemp(
            prefix=f"{directory.name}.com-staging-",
            dir=directory.parent,
        )
    )
    try:
        _write_com_proxy(staging / "com_proxy.csv", frame_results)
        _write_stride_com(
            staging / "stride_com.csv", stride_com_samples, reviewed_strides
        )
        _write_diagnostic(
            staging / "com_diagnostic.png",
            frame_results,
            reviewed_strides,
            config,
        )

        metadata = _build_metadata(
            directory=directory,
            staging=staging,
            config=config,
            input_hashes=input_hashes,
            frame_results=frame_results,
            reviewed_strides=reviewed_strides,
            stride_sample_counts=stride_sample_counts,
            stride_original_counts=stride_original_counts,
            preproc_meta=preproc_meta,
            rr_meta=rr_meta,
            scientific_unresolved=scientific_unresolved,
        )
        (staging / "com_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        # Re-verify all input hashes (transactional integrity)
        _recheck_input_hashes(input_hashes, inputs)

        # Publish
        _publish(staging, directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    diag = directory / "com_diagnostic.png"
    if not diag.is_file():
        raise ComPipelineError("com_diagnostic.png was not published successfully")

    return ComArtifacts(
        artifact_directory=directory,
        com_proxy_path=directory / "com_proxy.csv",
        stride_com_path=directory / "stride_com.csv",
        diagnostic_path=diag,
        com_metadata_path=directory / "com_metadata.json",
    )
