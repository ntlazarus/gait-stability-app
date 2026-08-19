"""Step 4b: Review resolution for automatic gait-event and stride detections.

Resolves manual review and correction of automatic gait-event and stride
detections, producing reviewed_gait_events.csv, reviewed_strides.csv, and
review_resolution_metadata.json as a transactionally published set.

``reviewed_quality`` is a Step 4b resolution/QC category, not a validated
measurement quality label.  ``"review"`` marks corrected or promoted accepted
boundaries; ``"high"`` / ``"low"`` carry the detector's ``automatic_quality``
forward for unchanged events; replaced events are assigned ``"low"`` as a
Step 4b exclusion/QC flag (not a detector quality assessment).
Reviewed stride quality derives from reviewed endpoints and is not
validated measurement quality.

Counts ``rejected`` and ``replaced`` may overlap: an event that was replaced
also counts as rejected (reviewed_rejected is true).

Reviewed intervals are candidate temporal segmentation / QC windows using
nominal pose-frame timestamps.  They are **not** force-plate-confirmed
contacts, validated stride durations, or measured toe-off / stance / swing /
spatial metrics.  No COM, step-5, or stability metrics are produced.

Inputs: gait_events.csv, strides.csv, gait_event_reviews.csv,
strides_reviews.csv, gait_event_metadata.json, pose_frames.csv,
preprocessing_metadata.json, and an external plain-text assumption document.
"""

from __future__ import annotations

import argparse
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
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from gait_stability.gait_events import (
    GAIT_EVENT_FIELDS,
    STRIDE_FIELDS,
    GaitEvent,
    GaitEventConfig,
    Stride,
    construct_strides,
)
from gait_stability.video_ingestion import ArtifactPublishError, sha256_file

# ---------------------------------------------------------------------------
# Public error hierarchy
# ---------------------------------------------------------------------------


class ReviewResolutionError(Exception):
    """Expected Step 4b input, validation, or resolution error."""


class ReviewResolutionArtifactValidationError(ReviewResolutionError):
    """Raised when canonical Step 4/4b artifacts violate their contracts."""


# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

ReviewStatus = Literal["unreviewed", "retain_rejection", "promote_to_candidate"]
"""Review status for individual gait events."""

StrideReviewStatus = Literal["accept", "correct"]
"""Review status for individual strides."""

ResolutionDisposition = Literal[
    "accepted_unchanged",
    "corrected",
    "promoted_from_rejected_candidate",
    "rejected",
    "replaced",
]
"""Resolution disposition for an event."""


# ---------------------------------------------------------------------------
# Public return type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReviewResolutionArtifacts:
    """Published Step 4b artifact paths."""

    artifact_directory: Path
    reviewed_gait_events_path: Path
    reviewed_strides_path: Path
    review_resolution_metadata_path: Path


# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

REVIEW_RESOLUTION_SCHEMA_VERSION = 1
REVIEW_RESOLUTION_ALGORITHM_VERSION = "step4b-review-resolution-1"

INPUT_ARTIFACT_NAMES = (
    "gait_events.csv",
    "strides.csv",
    "gait_event_reviews.csv",
    "strides_reviews.csv",
    "gait_event_metadata.json",
    "pose_frames.csv",
    "preprocessing_metadata.json",
)

OUTPUT_ARTIFACT_NAMES = (
    "reviewed_gait_events.csv",
    "reviewed_strides.csv",
    "review_resolution_metadata.json",
)

EVENT_REVIEW_FIELDS = (
    "event_id",
    "frame_index",
    "timestamp_seconds",
    "side",
    "detection_status",
    "review_status",
)

STRIDE_REVIEW_STATUS_VALUES = frozenset({"accept", "correct"})
EVENT_REVIEW_STATUS_VALUES = frozenset(
    {"unreviewed", "retain_rejection", "promote_to_candidate"}
)

REVIEWED_GAIT_EVENT_FIELDS = (
    "event_id",
    "automatic_event_id",
    "side",
    "event_type",
    "automatic_frame_index",
    "automatic_timestamp_seconds",
    "automatic_disposition",
    "automatic_quality",
    "automatic_peak_value",
    "automatic_prominence",
    "automatic_rejection_reasons",
    "manual_event_review_status",
    "stride_review_provenance",
    "reviewed_frame_index",
    "reviewed_timestamp_seconds",
    "reviewed_accepted",
    "reviewed_rejected",
    "reviewed_included_in_stride",
    "reviewed_quality",
    "resolution_disposition",
    "replaces_event_id",
    "replaced_by_event_id",
    "source",
    "review_notes",
)


# ---------------------------------------------------------------------------
# Internal typed structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _EventReview:
    """Parsed and validated row from gait_event_reviews.csv."""

    event_id: str
    frame_index: int
    timestamp_seconds: float
    side: str
    detection_status: str
    review_status: str


@dataclass(frozen=True, slots=True)
class _StrideReview:
    """Parsed and validated row from strides_reviews.csv."""

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


@dataclass(frozen=True, slots=True)
class _ResolvedEvent:
    """Reviewed event with disposition, linking to automatic and replacement IDs."""

    reviewed_event_id: str
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


# ---------------------------------------------------------------------------
# Internal helpers: CSV parsing
# ---------------------------------------------------------------------------


def _parse_int(text: str, field: str, row_number: int, label: str) -> int:
    try:
        return int(text)
    except ValueError as exc:
        raise ReviewResolutionArtifactValidationError(
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
        raise ReviewResolutionArtifactValidationError(
            f"{label} row {row_number}: {field} must be numeric"
        ) from exc
    if not math.isfinite(value):
        raise ReviewResolutionArtifactValidationError(
            f"{label} row {row_number}: {field} must be finite"
        )
    return value


def _parse_bool_lower(text: str, field: str, row_number: int, label: str) -> bool:
    if text not in {"true", "false"}:
        raise ReviewResolutionArtifactValidationError(
            f"{label} row {row_number}: {field} must be true or false (lowercase)"
        )
    return text == "true"


def _parse_tuple_pipe(text: str) -> tuple[str, ...]:
    if text == "":
        return ()
    return tuple(text.split("|"))


def _csv_bool_lower(value: bool) -> str:
    return "true" if value else "false"


def _csv_tuple_pipe(value: tuple[str, ...]) -> str:
    return "|".join(value)


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return _csv_bool_lower(value)
    if isinstance(value, tuple):
        return _csv_tuple_pipe(value)
    if isinstance(value, float):
        return f"{value}"
    return str(value)


# ---------------------------------------------------------------------------
# Internal helpers: file loading
# ---------------------------------------------------------------------------


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ReviewResolutionArtifactValidationError(
            f"Required artifact is missing: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewResolutionArtifactValidationError(
            f"Could not read valid {label}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ReviewResolutionArtifactValidationError(f"{label} root must be an object")
    return value


def _read_gait_events(path: Path) -> tuple[GaitEvent, ...]:
    """Read gait_events.csv into GaitEvent tuples, validating exact fields."""
    if not path.is_file():
        raise ReviewResolutionArtifactValidationError(
            f"Required artifact is missing: {path}"
        )
    events: list[GaitEvent] = []
    seen_ids: set[str] = set()
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != GAIT_EVENT_FIELDS:
                raise ReviewResolutionArtifactValidationError(
                    "gait_events.csv header must exactly match GAIT_EVENT_FIELDS"
                )
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(
                    row[field] is None for field in GAIT_EVENT_FIELDS
                ):
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_events.csv row {row_number}: malformed columns"
                    )
                event_id = row["event_id"]
                if not event_id:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_events.csv row {row_number}: event_id must be nonempty"
                    )
                if event_id in seen_ids:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_events.csv row {row_number}: "
                        f"duplicate event_id {event_id}"
                    )
                seen_ids.add(event_id)
                frame_index = _parse_int(
                    row["frame_index"], "frame_index", row_number, "gait_events.csv"
                )
                timestamp = _parse_float(
                    row["timestamp_seconds"],
                    "timestamp_seconds",
                    row_number,
                    "gait_events.csv",
                )
                if timestamp is None:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_events.csv row {row_number}: "
                        "timestamp_seconds must be finite"
                    )
                side = row["side"]
                if side not in {"left", "right"}:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_events.csv row {row_number}: side must be left or right"
                    )
                event_type = row["event_type"]
                detection_method = row["detection_method"]
                detection_status = row["detection_status"]
                if detection_status not in {"accepted", "rejected_candidate"}:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_events.csv row {row_number}: invalid detection_status"
                    )
                included = _parse_bool_lower(
                    row["included_in_stride_construction"],
                    "included_in_stride_construction",
                    row_number,
                    "gait_events.csv",
                )
                quality = row["confidence_or_quality"]
                if quality not in {"high", "review", "low"}:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_events.csv row {row_number}: invalid quality"
                    )
                peak_value = _parse_float(
                    row["peak_value"], "peak_value", row_number, "gait_events.csv"
                )
                if peak_value is None:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_events.csv row {row_number}: peak_value must be finite"
                    )
                prominence = _parse_float(
                    row["prominence"],
                    "prominence",
                    row_number,
                    "gait_events.csv",
                    nullable=True,
                )
                pre_velocity = _parse_float(
                    row["pre_velocity"],
                    "pre_velocity",
                    row_number,
                    "gait_events.csv",
                    nullable=True,
                )
                post_velocity = _parse_float(
                    row["post_velocity"],
                    "post_velocity",
                    row_number,
                    "gait_events.csv",
                    nullable=True,
                )
                plateau_start_frame = _parse_int(
                    row["plateau_start_frame"],
                    "plateau_start_frame",
                    row_number,
                    "gait_events.csv",
                )
                plateau_end_frame = _parse_int(
                    row["plateau_end_frame"],
                    "plateau_end_frame",
                    row_number,
                    "gait_events.csv",
                )
                raw_peak_frame = (
                    _parse_int(
                        row["raw_peak_frame"],
                        "raw_peak_frame",
                        row_number,
                        "gait_events.csv",
                    )
                    if row["raw_peak_frame"] != ""
                    else None
                )
                raw_peak_offset_frames = (
                    _parse_int(
                        row["raw_peak_offset_frames"],
                        "raw_peak_offset_frames",
                        row_number,
                        "gait_events.csv",
                    )
                    if row["raw_peak_offset_frames"] != ""
                    else None
                )
                ankle_peak_frame = (
                    _parse_int(
                        row["ankle_peak_frame"],
                        "ankle_peak_frame",
                        row_number,
                        "gait_events.csv",
                    )
                    if row["ankle_peak_frame"] != ""
                    else None
                )
                ankle_peak_offset_frames = (
                    _parse_int(
                        row["ankle_peak_offset_frames"],
                        "ankle_peak_offset_frames",
                        row_number,
                        "gait_events.csv",
                    )
                    if row["ankle_peak_offset_frames"] != ""
                    else None
                )
                ankle_support_observed = _parse_bool_lower(
                    row["ankle_support_observed_usable"],
                    "ankle_support_observed_usable",
                    row_number,
                    "gait_events.csv",
                )
                ankle_support_interp = _parse_bool_lower(
                    row["ankle_support_interpolated"],
                    "ankle_support_interpolated",
                    row_number,
                    "gait_events.csv",
                )
                ankle_support_smooth = _parse_bool_lower(
                    row["ankle_support_smoothing_contains_interpolation"],
                    "ankle_support_smoothing_contains_interpolation",
                    row_number,
                    "gait_events.csv",
                )
                primary_support_observed = _parse_bool_lower(
                    row["primary_support_observed_usable"],
                    "primary_support_observed_usable",
                    row_number,
                    "gait_events.csv",
                )
                primary_support_interp = _parse_bool_lower(
                    row["primary_support_interpolated"],
                    "primary_support_interpolated",
                    row_number,
                    "gait_events.csv",
                )
                primary_support_smooth = _parse_bool_lower(
                    row["primary_support_smoothing_contains_interpolation"],
                    "primary_support_smoothing_contains_interpolation",
                    row_number,
                    "gait_events.csv",
                )
                signal_support_notes = _parse_tuple_pipe(row["signal_support_notes"])
                sequence_context_notes = _parse_tuple_pipe(
                    row["sequence_context_notes"]
                )
                source = row["source"]
                review_status = row["review_status"]
                rejection_reasons = _parse_tuple_pipe(row["rejection_reasons"])
                events.append(
                    GaitEvent(
                        event_id=event_id,
                        frame_index=frame_index,
                        timestamp_seconds=timestamp,
                        side=side,  # type: ignore[arg-type]
                        event_type=event_type,
                        detection_method=detection_method,
                        detection_status=detection_status,  # type: ignore[arg-type]
                        included_in_stride_construction=included,
                        confidence_or_quality=quality,  # type: ignore[arg-type]
                        peak_value=peak_value,
                        prominence=prominence,
                        pre_velocity=pre_velocity,
                        post_velocity=post_velocity,
                        plateau_start_frame=plateau_start_frame,
                        plateau_end_frame=plateau_end_frame,
                        raw_peak_frame=raw_peak_frame,
                        raw_peak_offset_frames=raw_peak_offset_frames,
                        ankle_peak_frame=ankle_peak_frame,
                        ankle_peak_offset_frames=ankle_peak_offset_frames,
                        ankle_support_observed_usable=ankle_support_observed,
                        ankle_support_interpolated=ankle_support_interp,
                        ankle_support_smoothing_contains_interpolation=ankle_support_smooth,
                        primary_support_observed_usable=primary_support_observed,
                        primary_support_interpolated=primary_support_interp,
                        primary_support_smoothing_contains_interpolation=primary_support_smooth,
                        signal_support_notes=signal_support_notes,
                        sequence_context_notes=sequence_context_notes,
                        source=source,
                        review_status=review_status,
                        rejection_reasons=rejection_reasons,
                    )
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewResolutionArtifactValidationError(
            f"Could not read {path}: {exc}"
        ) from exc
    return tuple(events)


def _read_strides(path: Path) -> tuple[Stride, ...]:
    """Read strides.csv into Stride tuples, validating exact fields."""
    if not path.is_file():
        raise ReviewResolutionArtifactValidationError(
            f"Required artifact is missing: {path}"
        )
    strides: list[Stride] = []
    seen_ids: set[str] = set()
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != STRIDE_FIELDS:
                raise ReviewResolutionArtifactValidationError(
                    "strides.csv header must exactly match STRIDE_FIELDS"
                )
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(row[field] is None for field in STRIDE_FIELDS):
                    raise ReviewResolutionArtifactValidationError(
                        f"strides.csv row {row_number}: malformed columns"
                    )
                stride_id = row["stride_id"]
                if not stride_id:
                    raise ReviewResolutionArtifactValidationError(
                        f"strides.csv row {row_number}: stride_id must be nonempty"
                    )
                if stride_id in seen_ids:
                    raise ReviewResolutionArtifactValidationError(
                        f"strides.csv row {row_number}: duplicate stride_id {stride_id}"
                    )
                seen_ids.add(stride_id)
                side = row["side"]
                if side not in {"left", "right"}:
                    raise ReviewResolutionArtifactValidationError(
                        f"strides.csv row {row_number}: side must be left or right"
                    )
                start_event_id = row["start_event_id"]
                end_event_id = row["end_event_id"]
                start_frame = _parse_int(
                    row["start_frame"], "start_frame", row_number, "strides.csv"
                )
                end_frame = _parse_int(
                    row["end_frame"], "end_frame", row_number, "strides.csv"
                )
                start_ts = _parse_float(
                    row["start_timestamp_seconds"],
                    "start_timestamp_seconds",
                    row_number,
                    "strides.csv",
                )
                end_ts = _parse_float(
                    row["end_timestamp_seconds"],
                    "end_timestamp_seconds",
                    row_number,
                    "strides.csv",
                )
                duration = _parse_float(
                    row["duration_seconds"],
                    "duration_seconds",
                    row_number,
                    "strides.csv",
                )
                quality = row["quality"]
                if quality not in {"high", "review", "low"}:
                    raise ReviewResolutionArtifactValidationError(
                        f"strides.csv row {row_number}: invalid quality"
                    )
                contralateral_id = row["contralateral_event_id"]
                if contralateral_id == "":
                    contralateral_id = None
                contralateral_count = _parse_int(
                    row["contralateral_event_count"],
                    "contralateral_event_count",
                    row_number,
                    "strides.csv",
                )
                sequence_notes = _parse_tuple_pipe(row["sequence_notes"])
                source = row["source"]
                review_status = row["review_status"]
                strides.append(
                    Stride(
                        stride_id=stride_id,
                        side=side,  # type: ignore[arg-type]
                        start_event_id=start_event_id,
                        end_event_id=end_event_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        start_timestamp_seconds=start_ts
                        if start_ts is not None
                        else 0.0,
                        end_timestamp_seconds=end_ts if end_ts is not None else 0.0,
                        duration_seconds=duration if duration is not None else 0.0,
                        quality=quality,  # type: ignore[arg-type]
                        contralateral_event_id=contralateral_id,
                        contralateral_event_count=contralateral_count,
                        sequence_notes=sequence_notes,
                        source=source,
                        review_status=review_status,
                    )
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewResolutionArtifactValidationError(
            f"Could not read {path}: {exc}"
        ) from exc
    return tuple(strides)


def _read_event_reviews(
    path: Path, automatic_events: Mapping[str, GaitEvent]
) -> tuple[_EventReview, ...]:
    """Read and validate gait_event_reviews.csv: one row per automatic event."""
    if not path.is_file():
        raise ReviewResolutionArtifactValidationError(
            f"Required artifact is missing: {path}"
        )
    reviews: list[_EventReview] = []
    seen_ids: set[str] = set()
    auto_ids_seen: set[str] = set()
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != EVENT_REVIEW_FIELDS:
                raise ReviewResolutionArtifactValidationError(
                    "gait_event_reviews.csv header must exactly match "
                    + str(EVENT_REVIEW_FIELDS)
                )
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(
                    row[field] is None for field in EVENT_REVIEW_FIELDS
                ):
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_event_reviews.csv row {row_number}: malformed columns"
                    )
                event_id = row["event_id"]
                if not event_id:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_event_reviews.csv row {row_number}: event_id empty"
                    )
                if event_id in seen_ids:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_event_reviews.csv row {row_number}: "
                        f"duplicate event_id {event_id}"
                    )
                seen_ids.add(event_id)
                if event_id not in automatic_events:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_event_reviews.csv row {row_number}: "
                        f"event_id {event_id} not in automatic events"
                    )
                auto_events_list = [
                    e for e in automatic_events.values() if e.event_id == event_id
                ]
                auto_event = auto_events_list[0]
                frame_index = _parse_int(
                    row["frame_index"],
                    "frame_index",
                    row_number,
                    "gait_event_reviews.csv",
                )
                timestamp = _parse_float(
                    row["timestamp_seconds"],
                    "timestamp_seconds",
                    row_number,
                    "gait_event_reviews.csv",
                )
                if timestamp is None:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_event_reviews.csv row {row_number}: "
                        "timestamp_seconds must be finite"
                    )
                side = row["side"]
                if side not in {"left", "right"}:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_event_reviews.csv row {row_number}: invalid side"
                    )
                # Identity cross-validation: review row must match automatic event
                if frame_index != auto_event.frame_index:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_event_reviews.csv row {row_number}: "
                        f"frame_index {frame_index} does not match "
                        f"automatic {auto_event.frame_index}"
                    )
                if abs(timestamp - auto_event.timestamp_seconds) > 1e-9:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_event_reviews.csv row {row_number}: timestamp "
                        f"does not match automatic event"
                    )
                if side != auto_event.side:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_event_reviews.csv row {row_number}: side "
                        f"{side} does not match automatic {auto_event.side}"
                    )
                detection_status = row["detection_status"]
                if detection_status != auto_event.detection_status:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_event_reviews.csv row {row_number}: detection_status "
                        f"{detection_status} does not match automatic "
                        f"{auto_event.detection_status}"
                    )
                review_status = row["review_status"]
                if review_status not in EVENT_REVIEW_STATUS_VALUES:
                    raise ReviewResolutionArtifactValidationError(
                        f"gait_event_reviews.csv row {row_number}: "
                        f"invalid review_status {review_status}"
                    )
                auto_ids_seen.add(event_id)
                reviews.append(
                    _EventReview(
                        event_id=event_id,
                        frame_index=frame_index,
                        timestamp_seconds=timestamp,
                        side=side,
                        detection_status=detection_status,
                        review_status=review_status,
                    )
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewResolutionArtifactValidationError(
            f"Could not read {path}: {exc}"
        ) from exc
    # Complete set: every automatic event must have exactly one review row
    if auto_ids_seen != set(automatic_events.keys()):
        missing = sorted(set(automatic_events.keys()) - auto_ids_seen)
        raise ReviewResolutionArtifactValidationError(
            f"gait_event_reviews.csv is missing rows for event IDs: {missing}"
        )
    return tuple(reviews)


def _read_stride_reviews(
    path: Path,
    automatic_strides: Mapping[str, Stride],
) -> tuple[_StrideReview, ...]:
    """Read and validate strides_reviews.csv: one row per automatic stride."""
    if not path.is_file():
        raise ReviewResolutionArtifactValidationError(
            f"Required artifact is missing: {path}"
        )
    reviews: list[_StrideReview] = []
    seen_ids: set[str] = set()
    auto_ids_seen: set[str] = set()
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != STRIDE_FIELDS:
                raise ReviewResolutionArtifactValidationError(
                    "strides_reviews.csv header must exactly match STRIDE_FIELDS"
                )
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(row[field] is None for field in STRIDE_FIELDS):
                    raise ReviewResolutionArtifactValidationError(
                        f"strides_reviews.csv row {row_number}: malformed columns"
                    )
                stride_id = row["stride_id"]
                if not stride_id:
                    raise ReviewResolutionArtifactValidationError(
                        f"strides_reviews.csv row {row_number}: stride_id empty"
                    )
                if stride_id in seen_ids:
                    raise ReviewResolutionArtifactValidationError(
                        f"strides_reviews.csv row {row_number}: "
                        f"duplicate stride_id {stride_id}"
                    )
                seen_ids.add(stride_id)
                if stride_id not in automatic_strides:
                    raise ReviewResolutionArtifactValidationError(
                        f"strides_reviews.csv row {row_number}: "
                        f"stride_id {stride_id} not in automatic strides"
                    )
                auto_stride = automatic_strides[stride_id]
                side = row["side"]
                if side != auto_stride.side:
                    raise ReviewResolutionArtifactValidationError(
                        f"strides_reviews.csv row {row_number}: side mismatch"
                    )
                start_event_id = row["start_event_id"]
                end_event_id = row["end_event_id"]
                if start_event_id != auto_stride.start_event_id:
                    raise ReviewResolutionArtifactValidationError(
                        f"strides_reviews.csv row {row_number}: start_event_id mismatch"
                    )
                if end_event_id != auto_stride.end_event_id:
                    raise ReviewResolutionArtifactValidationError(
                        f"strides_reviews.csv row {row_number}: end_event_id mismatch"
                    )
                start_frame = _parse_int(
                    row["start_frame"],
                    "start_frame",
                    row_number,
                    "strides_reviews.csv",
                )
                end_frame = _parse_int(
                    row["end_frame"],
                    "end_frame",
                    row_number,
                    "strides_reviews.csv",
                )
                start_ts = _parse_float(
                    row["start_timestamp_seconds"],
                    "start_timestamp_seconds",
                    row_number,
                    "strides_reviews.csv",
                )
                end_ts = _parse_float(
                    row["end_timestamp_seconds"],
                    "end_timestamp_seconds",
                    row_number,
                    "strides_reviews.csv",
                )
                duration = _parse_float(
                    row["duration_seconds"],
                    "duration_seconds",
                    row_number,
                    "strides_reviews.csv",
                )
                quality = row["quality"]
                contralateral_id = row["contralateral_event_id"]
                if contralateral_id == "":
                    contralateral_id = None
                contralateral_count = _parse_int(
                    row["contralateral_event_count"],
                    "contralateral_event_count",
                    row_number,
                    "strides_reviews.csv",
                )
                sequence_notes = _parse_tuple_pipe(row["sequence_notes"])
                source = row["source"]
                review_status = row["review_status"]
                if review_status not in STRIDE_REVIEW_STATUS_VALUES:
                    raise ReviewResolutionArtifactValidationError(
                        f"strides_reviews.csv row {row_number}: "
                        f"invalid review_status {review_status}"
                    )
                # Validate unchanged columns against automatic stride
                if (
                    start_frame != auto_stride.start_frame
                    or end_frame != auto_stride.end_frame
                ):
                    if review_status != "correct":
                        raise ReviewResolutionArtifactValidationError(
                            f"strides_reviews.csv row {row_number}: frame edits "
                            "require review_status correct"
                        )
                else:
                    if review_status != "accept":
                        raise ReviewResolutionArtifactValidationError(
                            f"strides_reviews.csv row {row_number}: "
                            "no frame edits require review_status accept"
                        )
                # Stale timestamps/durations must match automatic stride to
                # prevent silent data corruption from stale copies.
                if (
                    start_ts is not None
                    and abs(start_ts - auto_stride.start_timestamp_seconds) > 1e-9
                ):
                    raise ReviewResolutionArtifactValidationError(
                        f"strides_reviews.csv row {row_number}: "
                        "start_timestamp_seconds does not match automatic"
                    )
                if (
                    end_ts is not None
                    and abs(end_ts - auto_stride.end_timestamp_seconds) > 1e-9
                ):
                    raise ReviewResolutionArtifactValidationError(
                        f"strides_reviews.csv row {row_number}: "
                        "end_timestamp_seconds does not match automatic"
                    )
                if (
                    duration is not None
                    and abs(duration - auto_stride.duration_seconds) > 1e-9
                ):
                    raise ReviewResolutionArtifactValidationError(
                        f"strides_reviews.csv row {row_number}: "
                        "duration_seconds does not match automatic"
                    )
                if quality != auto_stride.quality:
                    raise ReviewResolutionArtifactValidationError(
                        f"strides_reviews.csv row {row_number}: quality "
                        f"{quality!r} does not match automatic "
                        f"{auto_stride.quality!r}"
                    )
                if contralateral_id != auto_stride.contralateral_event_id:
                    raise ReviewResolutionArtifactValidationError(
                        f"strides_reviews.csv row {row_number}: "
                        f"contralateral_event_id does not match automatic"
                    )
                if contralateral_count != auto_stride.contralateral_event_count:
                    raise ReviewResolutionArtifactValidationError(
                        f"strides_reviews.csv row {row_number}: "
                        f"contralateral_event_count does not match automatic"
                    )
                if sequence_notes != auto_stride.sequence_notes:
                    raise ReviewResolutionArtifactValidationError(
                        f"strides_reviews.csv row {row_number}: "
                        f"sequence_notes does not match automatic"
                    )
                if source != auto_stride.source:
                    raise ReviewResolutionArtifactValidationError(
                        f"strides_reviews.csv row {row_number}: source "
                        f"{source!r} does not match automatic "
                        f"{auto_stride.source!r}"
                    )
                auto_ids_seen.add(stride_id)
                reviews.append(
                    _StrideReview(
                        stride_id=stride_id,
                        side=side,
                        start_event_id=start_event_id,
                        end_event_id=end_event_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        start_timestamp_seconds=start_ts
                        if start_ts is not None
                        else 0.0,
                        end_timestamp_seconds=end_ts if end_ts is not None else 0.0,
                        duration_seconds=duration if duration is not None else 0.0,
                        quality=quality,
                        contralateral_event_id=contralateral_id,
                        contralateral_event_count=contralateral_count,
                        sequence_notes=sequence_notes,
                        source=source,
                        review_status=review_status,
                    )
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewResolutionArtifactValidationError(
            f"Could not read {path}: {exc}"
        ) from exc
    # Complete set
    if auto_ids_seen != set(automatic_strides.keys()):
        missing = sorted(set(automatic_strides.keys()) - auto_ids_seen)
        raise ReviewResolutionArtifactValidationError(
            f"strides_reviews.csv is missing rows for stride IDs: {missing}"
        )
    return tuple(reviews)


def _read_pose_frames(
    path: Path,
) -> dict[int, float]:
    """Read pose_frames.csv and return {frame_index: timestamp_seconds}."""
    if not path.is_file():
        raise ReviewResolutionArtifactValidationError(
            f"Required artifact is missing: {path}"
        )
    expected_header = (
        "frame_index",
        "nominal_timestamp_seconds",
        "backend_timestamp_milliseconds",
        "status",
        "landmark_count",
        "detail",
    )
    result: dict[int, float] = {}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != expected_header:
                raise ReviewResolutionArtifactValidationError(
                    "pose_frames.csv header must exactly match expected schema"
                )
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(row[field] is None for field in expected_header):
                    raise ReviewResolutionArtifactValidationError(
                        f"pose_frames.csv row {row_number}: malformed columns"
                    )
                fi = _parse_int(
                    row["frame_index"],
                    "frame_index",
                    row_number,
                    "pose_frames.csv",
                )
                ts = _parse_float(
                    row["nominal_timestamp_seconds"],
                    "nominal_timestamp_seconds",
                    row_number,
                    "pose_frames.csv",
                )
                if ts is None:
                    raise ReviewResolutionArtifactValidationError(
                        f"pose_frames.csv row {row_number}: "
                        "nominal_timestamp_seconds must be finite"
                    )
                if fi in result:
                    raise ReviewResolutionArtifactValidationError(
                        f"pose_frames.csv row {row_number}: duplicate frame_index {fi}"
                    )
                result[fi] = ts
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewResolutionArtifactValidationError(
            f"Could not read {path}: {exc}"
        ) from exc
    if not result:
        raise ReviewResolutionArtifactValidationError(
            "pose_frames.csv must contain at least one row"
        )
    return result


# ---------------------------------------------------------------------------
# Internal helpers: validation and resolution
# ---------------------------------------------------------------------------


def _validate_event_times_from_pose(
    events: Sequence[GaitEvent],
    pose_timestamps: Mapping[int, float],
    label: str,
) -> None:
    """Validate that every automatic event timestamp exactly matches pose_frames."""
    for event in events:
        if event.frame_index not in pose_timestamps:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: event {event.event_id} frame {event.frame_index} "
                "not in pose_frames.csv"
            )
        expected_ts = pose_timestamps[event.frame_index]
        if abs(event.timestamp_seconds - expected_ts) > 1e-9:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: event {event.event_id} timestamp "
                f"{event.timestamp_seconds} does not match pose_frames "
                f"{expected_ts} for frame {event.frame_index}"
            )


def _validate_strides_against_events(
    strides: Sequence[Stride],
    events_by_id: Mapping[str, GaitEvent],
    label: str,
) -> None:
    """Validate stride endpoints, times, duration, side, and quality.

    Checks that automatic stride endpoints are accepted, included, and
    candidate_initial_contact against automatic events.
    """
    for stride in strides:
        start_event = events_by_id.get(stride.start_event_id)
        end_event = events_by_id.get(stride.end_event_id)
        if start_event is None:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} start_event_id "
                f"{stride.start_event_id} not found"
            )
        if end_event is None:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} end_event_id "
                f"{stride.end_event_id} not found"
            )
        if start_event.side != stride.side:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} side {stride.side} "
                f"mismatch with start event side {start_event.side}"
            )
        if end_event.side != stride.side:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} side {stride.side} "
                f"mismatch with end event side {end_event.side}"
            )
        if start_event.frame_index != stride.start_frame:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} start_frame "
                f"{stride.start_frame} != event frame {start_event.frame_index}"
            )
        if end_event.frame_index != stride.end_frame:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} end_frame "
                f"{stride.end_frame} != event frame {end_event.frame_index}"
            )
        if abs(start_event.timestamp_seconds - stride.start_timestamp_seconds) > 1e-9:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} start_timestamp mismatch"
            )
        if abs(end_event.timestamp_seconds - stride.end_timestamp_seconds) > 1e-9:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} end_timestamp mismatch"
            )
        expected_duration = end_event.timestamp_seconds - start_event.timestamp_seconds
        if abs(stride.duration_seconds - expected_duration) > 1e-9:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} duration mismatch"
            )
        if stride.duration_seconds <= 0.0:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} nonpositive duration"
            )
        # Endpoints must be accepted, included, and candidate_initial_contact
        if start_event.detection_status != "accepted":
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} start_event "
                f"{start_event.event_id} detection_status "
                f"{start_event.detection_status!r} is not accepted"
            )
        if end_event.detection_status != "accepted":
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} end_event "
                f"{end_event.event_id} detection_status "
                f"{end_event.detection_status!r} is not accepted"
            )
        if not start_event.included_in_stride_construction:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} start_event "
                f"{start_event.event_id} included_in_stride_construction "
                f"is false"
            )
        if not end_event.included_in_stride_construction:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} end_event "
                f"{end_event.event_id} included_in_stride_construction "
                f"is false"
            )
        if start_event.event_type != "candidate_initial_contact":
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} start_event "
                f"{start_event.event_id} event_type "
                f"{start_event.event_type!r} is not candidate_initial_contact"
            )
        if end_event.event_type != "candidate_initial_contact":
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} end_event "
                f"{end_event.event_id} event_type "
                f"{end_event.event_type!r} is not candidate_initial_contact"
            )


def _build_event_dict(events: Sequence[GaitEvent]) -> dict[str, GaitEvent]:
    return {e.event_id: e for e in events}


def _build_stride_dict(strides: Sequence[Stride]) -> dict[str, Stride]:
    return {s.stride_id: s for s in strides}


# ---------------------------------------------------------------------------
# Core resolution logic
# ---------------------------------------------------------------------------


def _resolve_corrections_from_stride_reviews(
    stride_reviews: Sequence[_StrideReview],
    automatic_events: Mapping[str, GaitEvent],
) -> dict[str, dict[str, Any]]:
    """Parse stride intent and convert frame edits into event corrections.

    Returns {event_id: {type, new_frame, source_stride, boundary}}.
    Conflicting corrections for the same event raise.
    """
    corrections: dict[str, dict[str, Any]] = {}
    for review in stride_reviews:
        if review.review_status != "correct":
            continue
        start_event = automatic_events.get(review.start_event_id)
        end_event = automatic_events.get(review.end_event_id)
        if start_event is None or end_event is None:
            raise ReviewResolutionArtifactValidationError(
                f"stride review {review.stride_id}: event IDs not found"
            )
        # Check start frame edit
        if review.start_frame != start_event.frame_index:
            event_id = review.start_event_id
            if event_id in corrections:
                raise ReviewResolutionArtifactValidationError(
                    f"conflicting correction for event {event_id} from "
                    f"stride {review.stride_id}"
                )
            corrections[event_id] = {
                "type": "boundary_correction",
                "boundary": "start",
                "new_frame": review.start_frame,
                "source_stride": review.stride_id,
            }
        # Check end frame edit
        if review.end_frame != end_event.frame_index:
            event_id = review.end_event_id
            if event_id in corrections:
                raise ReviewResolutionArtifactValidationError(
                    f"conflicting correction for event {event_id} from "
                    f"stride {review.stride_id}"
                )
            corrections[event_id] = {
                "type": "boundary_correction",
                "boundary": "end",
                "new_frame": review.end_frame,
                "source_stride": review.stride_id,
            }
    return corrections


def _resolve_events(
    automatic_events: Sequence[GaitEvent],
    event_reviews: Sequence[_EventReview],
    corrections: dict[str, dict[str, Any]],
    pose_timestamps: Mapping[int, float],
) -> tuple[_ResolvedEvent, ...]:
    """Resolve all events: baseline + overrides + corrections + promotions."""
    review_map = {r.event_id: r for r in event_reviews}

    # Build automatic event lookup for side validation during replacement matching
    auto_events_by_id = {e.event_id: e for e in automatic_events}

    # Build a frame->events lookup for candidates at a given frame
    frame_events: dict[int, list[GaitEvent]] = {}
    for event in automatic_events:
        frame_events.setdefault(event.frame_index, []).append(event)

    resolved: list[_ResolvedEvent] = []

    for event in automatic_events:
        review = review_map.get(event.event_id)
        review_status = review.review_status if review else "unreviewed"
        manual_status = review_status

        # Default: keep automatic
        reviewed_frame = event.frame_index
        reviewed_ts = event.timestamp_seconds
        reviewed_accepted = event.detection_status == "accepted"
        reviewed_rejected = event.detection_status == "rejected_candidate"
        reviewed_included = event.included_in_stride_construction
        reviewed_quality = event.confidence_or_quality
        disposition = "accepted_unchanged" if reviewed_accepted else "rejected"
        replaces_id: str | None = None
        replaced_by_id: str | None = None
        notes: list[str] = []
        stride_provenance = ""
        source = "automatic"
        new_event_id = event.event_id

        if review_status == "unreviewed":
            notes.append("unreviewed_baseline_preserved")
        elif review_status == "retain_rejection":
            if event.detection_status != "rejected_candidate":
                raise ReviewResolutionArtifactValidationError(
                    f"retain_rejection only valid for rejected candidates: "
                    f"{event.event_id}"
                )
            disposition = "rejected"
            notes.append("retain_rejection_confirmed")
        elif review_status == "promote_to_candidate":
            if event.detection_status != "rejected_candidate":
                raise ReviewResolutionArtifactValidationError(
                    f"promote_to_candidate only valid for rejected candidates: "
                    f"{event.event_id}"
                )
            # Accept the candidate, preserving auto-rejected provenance
            reviewed_accepted = True
            reviewed_rejected = False
            reviewed_included = True
            reviewed_quality = "review"
            disposition = "promoted_from_rejected_candidate"
            source = "automatic|manual_review"
            notes.append("promoted_candidate_retains_automatic_rejection_provenance")
            # For replacement promotions, find the original event being replaced.
            # Must match both frame AND side; fail on ambiguity.
            if replaces_id is None:
                candidates: list[str] = []
                for orig_id, orig_corr in corrections.items():
                    orig_auto = auto_events_by_id.get(orig_id)
                    if (
                        orig_corr["new_frame"] == event.frame_index
                        and orig_auto is not None
                        and orig_auto.side == event.side
                    ):
                        candidates.append(orig_id)
                if len(candidates) == 1:
                    replaces_id = candidates[0]
                elif len(candidates) > 1:
                    raise ReviewResolutionArtifactValidationError(
                        f"ambiguous replacement for promoted candidate "
                        f"{event.event_id} at frame {event.frame_index}: "
                        f"multiple same-side corrections target this frame: "
                        f"{candidates}"
                    )

            # Set provenance: use correction source if a replacement was matched,
            # else standalone
            if replaces_id is not None:
                orig_corr = corrections[replaces_id]
                stride_provenance = (
                    f"boundary_correction_from_stride:"
                    f"{orig_corr['source_stride']}:{orig_corr['boundary']}"
                )
                notes.append(
                    f"promoted_as_replacement_for_{replaces_id}_via_"
                    f"{orig_corr['source_stride']}:{orig_corr['boundary']}"
                )
            elif event.event_id in corrections:
                corr = corrections[event.event_id]
                stride_provenance = (
                    f"boundary_correction_from_stride:"
                    f"{corr['source_stride']}:{corr['boundary']}"
                )
            else:
                # An unlinked promotion has no corresponding stride-level
                # boundary correction and would create an unmatched stride
                # without a review decision.  Reject before publication.
                raise ReviewResolutionArtifactValidationError(
                    f"promote_to_candidate for {event.event_id} at frame "
                    f"{event.frame_index} is not linked as the replacement "
                    f"target of any event-level stride boundary correction; "
                    f"standalone promotions are not supported"
                )

        # Handle boundary corrections
        if event.event_id in corrections:
            corr = corrections[event.event_id]
            new_frame = corr["new_frame"]
            if new_frame not in pose_timestamps:
                raise ReviewResolutionArtifactValidationError(
                    f"corrected frame {new_frame} for event {event.event_id} "
                    "not in pose_frames.csv"
                )

            if event.detection_status != "accepted":
                raise ReviewResolutionArtifactValidationError(
                    f"correction for event {event.event_id} requires "
                    "automatic accepted status"
                )

            # Check for any candidate at corrected frame on same side
            same_side_candidates = [
                e
                for e in frame_events.get(new_frame, [])
                if e.side == event.side and e.event_id != event.event_id
            ]
            wrong_side_candidates = [
                e
                for e in frame_events.get(new_frame, [])
                if e.side != event.side and e.detection_status == "accepted"
            ]

            # Fail if any wrong-side candidate exists at the target frame
            if wrong_side_candidates:
                wrong_ids = [e.event_id for e in wrong_side_candidates]
                raise ReviewResolutionArtifactValidationError(
                    f"corrected frame {new_frame} for event {event.event_id} "
                    f"contains wrong-side candidates: {wrong_ids}"
                )

            # Find matching promoted candidate review among same-side candidates
            promoted_candidate = None
            if same_side_candidates:
                # Must be exactly one same-side candidate, and it must be a
                # promoted rejected candidate
                rejected_candidates = [
                    e
                    for e in same_side_candidates
                    if e.detection_status == "rejected_candidate"
                ]
                accepted_candidates = [
                    e for e in same_side_candidates if e.detection_status == "accepted"
                ]
                if accepted_candidates:
                    raise ReviewResolutionArtifactValidationError(
                        f"corrected frame {new_frame} for event {event.event_id} "
                        f"contains same-side accepted candidates: "
                        f"{[e.event_id for e in accepted_candidates]}"
                    )
                if len(rejected_candidates) != 1:
                    raise ReviewResolutionArtifactValidationError(
                        f"corrected frame {new_frame} for event {event.event_id} "
                        f"must contain exactly one same-side rejected candidate, "
                        f"found {len(rejected_candidates)}"
                    )
                cand = rejected_candidates[0]
                cand_review = review_map.get(cand.event_id)
                if (
                    cand_review is None
                    or cand_review.review_status != "promote_to_candidate"
                ):
                    raise ReviewResolutionArtifactValidationError(
                        f"corrected frame {new_frame} for event {event.event_id}: "
                        f"same-side rejected candidate {cand.event_id} is not "
                        f"explicitly promoted"
                    )
                promoted_candidate = cand

            if promoted_candidate is not None:
                # Original event: replaced
                disposition = "replaced"
                reviewed_accepted = False
                reviewed_rejected = True
                reviewed_included = False
                reviewed_quality = "low"
                replaced_by_id = promoted_candidate.event_id
                source = "automatic|manual_review"
                notes.append(
                    f"replaced_by_promoted_candidate_{promoted_candidate.event_id}"
                )
                stride_provenance = (
                    f"boundary_correction_replaced_by:{promoted_candidate.event_id}"
                )
            else:
                # No candidate: retain original ID, mark corrected
                disposition = "corrected"
                reviewed_frame = new_frame
                reviewed_ts = pose_timestamps[new_frame]
                reviewed_quality = "review"
                source = "automatic|manual_review"
                notes.append(
                    f"boundary_corrected_from_{event.frame_index}_to_{new_frame}"
                )
                stride_provenance = (
                    f"boundary_correction_from_stride:"
                    f"{corr['source_stride']}:{corr['boundary']}"
                )

        resolved.append(
            _ResolvedEvent(
                reviewed_event_id=new_event_id,
                automatic_event_id=event.event_id,
                side=event.side,
                event_type=event.event_type,
                automatic_frame_index=event.frame_index,
                automatic_timestamp_seconds=event.timestamp_seconds,
                automatic_disposition=event.detection_status,
                automatic_quality=event.confidence_or_quality,
                automatic_peak_value=event.peak_value,
                automatic_prominence=event.prominence,
                automatic_rejection_reasons=event.rejection_reasons,
                manual_event_review_status=manual_status,
                stride_review_provenance=stride_provenance,
                reviewed_frame_index=reviewed_frame,
                reviewed_timestamp_seconds=reviewed_ts,
                reviewed_accepted=reviewed_accepted,
                reviewed_rejected=reviewed_rejected,
                reviewed_included_in_stride=reviewed_included,
                reviewed_quality=reviewed_quality,
                resolution_disposition=disposition,
                replaces_event_id=replaces_id,
                replaced_by_event_id=replaced_by_id,
                source=source,
                review_notes=tuple(notes),
            )
        )

    return tuple(resolved)


# ---------------------------------------------------------------------------
# Stride reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_reviewed_strides(
    resolved_events: Sequence[_ResolvedEvent],
    automatic_strides: Sequence[Stride],
    automatic_events: Sequence[GaitEvent],
    event_reviews: Sequence[_EventReview],
    stride_reviews: Sequence[_StrideReview],
    corrections: dict[str, dict[str, Any]],
    gait_event_config: GaitEventConfig,
) -> tuple[Stride, ...]:
    """Reconstruct strides from reviewed events using construct_strides,
    then map back to automatic stride IDs and attach review provenance."""
    # Build GaitEvent copies with reviewed frame/time/status/inclusion/quality
    events_by_id = _build_event_dict(automatic_events)
    resolved_map = {r.automatic_event_id: r for r in resolved_events}

    reviewed_gait_events: list[GaitEvent] = []
    for resolved in resolved_events:
        auto_event = events_by_id[resolved.automatic_event_id]
        reviewed_gait_events.append(
            replace(
                auto_event,
                frame_index=resolved.reviewed_frame_index,
                timestamp_seconds=resolved.reviewed_timestamp_seconds,
                detection_status="accepted"
                if resolved.reviewed_accepted
                else "rejected_candidate",
                included_in_stride_construction=resolved.reviewed_included_in_stride,
                confidence_or_quality=resolved.reviewed_quality,  # type: ignore[arg-type]
                source=resolved.source,
                review_status=resolved.resolution_disposition,
            )
        )

    # Call construct_strides on reviewed events
    generated_strides = construct_strides(reviewed_gait_events, gait_event_config)

    # Build stride review lookup
    stride_review_map = {r.stride_id: r for r in stride_reviews}

    # For each automatic stride, find the generated stride with matching
    # (start_event_id, end_event_id) after replacement
    # Build a replacement mapping: automatic event_id -> reviewed event_id
    # In our case, the event_id stays the same unless replaced
    reviewed_event_ids = {
        r.automatic_event_id: r.automatic_event_id for r in resolved_events
    }
    # If an event is replaced by a promoted candidate, the promoted candidate
    # gets a new ID (deterministic)
    for resolved in resolved_events:
        if resolved.replaced_by_event_id is not None:
            # The promoted candidate's automatic_event_id is used as the new boundary
            reviewed_event_ids[resolved.replaced_by_event_id] = (
                resolved.replaced_by_event_id
            )

    result: list[Stride] = []
    used_generated: set[int] = set()

    for auto_stride in automatic_strides:
        start_resolved = resolved_map[auto_stride.start_event_id]
        end_resolved = resolved_map[auto_stride.end_event_id]

        # Determine the reviewed start/end event IDs
        if start_resolved.replaced_by_event_id is not None:
            new_start_id = start_resolved.replaced_by_event_id
        else:
            new_start_id = auto_stride.start_event_id

        if end_resolved.replaced_by_event_id is not None:
            new_end_id = end_resolved.replaced_by_event_id
        else:
            new_end_id = auto_stride.end_event_id

        # Find the generated stride with matching reviewed event IDs
        matched_idx: int | None = None
        for i, gen_stride in enumerate(generated_strides):
            if i in used_generated:
                continue
            if (
                gen_stride.start_event_id == new_start_id
                and gen_stride.end_event_id == new_end_id
            ):
                matched_idx = i
                break

        if matched_idx is None:
            raise ReviewResolutionArtifactValidationError(
                f"no generated stride matches automatic stride "
                f"{auto_stride.stride_id} after review resolution"
            )

        used_generated.add(matched_idx)
        gen_stride = generated_strides[matched_idx]
        stride_review = stride_review_map.get(auto_stride.stride_id)

        # Carry the generated stride's timing, durations, contralateral info
        # but use automatic stride ID and attach review provenance
        review_intent = stride_review.review_status if stride_review else "unreviewed"
        review_changes: list[str] = []
        if stride_review is not None:
            if stride_review.start_frame != auto_stride.start_frame:
                review_changes.append(
                    f"start_frame:{auto_stride.start_frame}->{stride_review.start_frame}"
                )
            if stride_review.end_frame != auto_stride.end_frame:
                review_changes.append(
                    f"end_frame:{auto_stride.end_frame}->{stride_review.end_frame}"
                )

        provenance = "automatic"
        if auto_stride.source != "automatic":
            provenance = auto_stride.source
        if review_intent != "unreviewed":
            provenance = "automatic|manual_review"

        result.append(
            Stride(
                stride_id=auto_stride.stride_id,
                side=gen_stride.side,
                start_event_id=gen_stride.start_event_id,
                end_event_id=gen_stride.end_event_id,
                start_frame=gen_stride.start_frame,
                end_frame=gen_stride.end_frame,
                start_timestamp_seconds=gen_stride.start_timestamp_seconds,
                end_timestamp_seconds=gen_stride.end_timestamp_seconds,
                duration_seconds=gen_stride.duration_seconds,
                quality=gen_stride.quality,
                contralateral_event_id=gen_stride.contralateral_event_id,
                contralateral_event_count=gen_stride.contralateral_event_count,
                sequence_notes=gen_stride.sequence_notes,
                source=provenance,
                review_status=review_intent,
            )
        )

    return tuple(result)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def _write_reviewed_gait_events(
    path: Path,
    resolved_events: Sequence[_ResolvedEvent],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEWED_GAIT_EVENT_FIELDS)
        writer.writeheader()
        for r in resolved_events:
            writer.writerow(
                {
                    "event_id": r.reviewed_event_id,
                    "automatic_event_id": r.automatic_event_id,
                    "side": r.side,
                    "event_type": r.event_type,
                    "automatic_frame_index": r.automatic_frame_index,
                    "automatic_timestamp_seconds": r.automatic_timestamp_seconds,
                    "automatic_disposition": r.automatic_disposition,
                    "automatic_quality": r.automatic_quality,
                    "automatic_peak_value": r.automatic_peak_value,
                    "automatic_prominence": r.automatic_prominence,
                    "automatic_rejection_reasons": _csv_tuple_pipe(
                        r.automatic_rejection_reasons
                    ),
                    "manual_event_review_status": r.manual_event_review_status,
                    "stride_review_provenance": r.stride_review_provenance,
                    "reviewed_frame_index": r.reviewed_frame_index,
                    "reviewed_timestamp_seconds": r.reviewed_timestamp_seconds,
                    "reviewed_accepted": _csv_bool_lower(r.reviewed_accepted),
                    "reviewed_rejected": _csv_bool_lower(r.reviewed_rejected),
                    "reviewed_included_in_stride": _csv_bool_lower(
                        r.reviewed_included_in_stride
                    ),
                    "reviewed_quality": r.reviewed_quality,
                    "resolution_disposition": r.resolution_disposition,
                    "replaces_event_id": r.replaces_event_id or "",
                    "replaced_by_event_id": r.replaced_by_event_id or "",
                    "source": r.source,
                    "review_notes": _csv_tuple_pipe(r.review_notes),
                }
            )


def _write_reviewed_strides(
    path: Path,
    strides: Sequence[Stride],
    automatic_strides: Mapping[str, Stride],
    stride_reviews: Mapping[str, _StrideReview],
    resolved_events: Sequence[_ResolvedEvent],
) -> None:
    """Write reviewed strides with automatic stride ID and review provenance."""
    # Build field list: STRIDE_FIELDS + extras
    extra_fields = (
        "automatic_stride_id",
        "review_intent",
        "review_changes",
        "provenance_notes",
    )
    all_fields = STRIDE_FIELDS + extra_fields

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for stride in strides:
            auto_stride = automatic_strides.get(stride.stride_id)
            stride_review = stride_reviews.get(stride.stride_id)

            review_intent = stride.review_status
            review_changes_parts: list[str] = []
            provenance_notes: list[str] = []

            if auto_stride is not None and stride_review is not None:
                if stride_review.start_frame != auto_stride.start_frame:
                    review_changes_parts.append(
                        f"start_frame:{auto_stride.start_frame}->{stride_review.start_frame}"
                    )
                if stride_review.end_frame != auto_stride.end_frame:
                    review_changes_parts.append(
                        f"end_frame:{auto_stride.end_frame}->{stride_review.end_frame}"
                    )
            elif auto_stride is None:
                provenance_notes.append("generated_from_standalone_promotion")

            row = {field: _csv_value(getattr(stride, field)) for field in STRIDE_FIELDS}
            row["automatic_stride_id"] = stride.stride_id
            row["review_intent"] = review_intent
            row["review_changes"] = "|".join(review_changes_parts)
            row["provenance_notes"] = "|".join(provenance_notes)
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Final validation
# ---------------------------------------------------------------------------


def _validate_final_events(
    resolved_events: Sequence[_ResolvedEvent],
    pose_timestamps: Mapping[int, float],
    automatic_events: Mapping[str, GaitEvent],
    label: str,
) -> None:
    """Validate canonical event times for all resolved events."""
    for resolved in resolved_events:
        if resolved.reviewed_frame_index not in pose_timestamps:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: event {resolved.reviewed_event_id} reviewed frame "
                f"{resolved.reviewed_frame_index} not in pose_frames"
            )
        expected_ts = pose_timestamps[resolved.reviewed_frame_index]
        if abs(resolved.reviewed_timestamp_seconds - expected_ts) > 1e-9:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: event {resolved.reviewed_event_id} reviewed timestamp "
                f"{resolved.reviewed_timestamp_seconds} != pose_frames {expected_ts}"
            )


def _validate_final_strides(
    reviewed_strides: Sequence[Stride],
    resolved_events: Sequence[_ResolvedEvent],
    automatic_events: Mapping[str, GaitEvent],
    corrections: dict[str, dict[str, Any]],
    label: str,
) -> list[str]:
    """Validate final in-memory stride rows before write.

    Returns a deduplicated sorted list of check names actually performed.
    """
    checks: set[str] = set()
    # Index by reviewed_event_id for promoted candidates
    reviewed_by_id: dict[str, _ResolvedEvent] = {}
    for r in resolved_events:
        reviewed_by_id[r.automatic_event_id] = r
        reviewed_by_id[r.reviewed_event_id] = r

    for stride in reviewed_strides:
        correction_propagated = False
        start_info = reviewed_by_id.get(stride.start_event_id)
        end_info = reviewed_by_id.get(stride.end_event_id)
        if start_info is None:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} start_event_id "
                f"{stride.start_event_id} not in resolved events"
            )
        if end_info is None:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} end_event_id "
                f"{stride.end_event_id} not in resolved events"
            )
        # Verify exact endpoint side
        if start_info.side != stride.side:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} side {stride.side} "
                f"mismatch start event side {start_info.side}"
            )
        if end_info.side != stride.side:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} side {stride.side} "
                f"mismatch end event side {end_info.side}"
            )
        # Verify exact endpoint frames
        if start_info.reviewed_frame_index != stride.start_frame:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} start_frame "
                f"{stride.start_frame} != event reviewed frame "
                f"{start_info.reviewed_frame_index}"
            )
        if end_info.reviewed_frame_index != stride.end_frame:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} end_frame "
                f"{stride.end_frame} != event reviewed frame "
                f"{end_info.reviewed_frame_index}"
            )
        checks.add("endpoint_event_id_side_frame")
        # Verify exact endpoint timestamps
        if (
            abs(start_info.reviewed_timestamp_seconds - stride.start_timestamp_seconds)
            > 1e-9
        ):
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} start_timestamp mismatch"
            )
        if (
            abs(end_info.reviewed_timestamp_seconds - stride.end_timestamp_seconds)
            > 1e-9
        ):
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} end_timestamp mismatch"
            )
        checks.add("endpoint_timestamps")
        # Duration exactly end-start and positive
        expected_duration = (
            end_info.reviewed_timestamp_seconds - start_info.reviewed_timestamp_seconds
        )
        if abs(stride.duration_seconds - expected_duration) > 1e-9:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} duration "
                f"{stride.duration_seconds} != expected {expected_duration}"
            )
        if stride.duration_seconds <= 0.0:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} nonpositive duration"
            )
        checks.add("duration_positive")
        # Accepted endpoints only
        if not start_info.reviewed_accepted:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} uses rejected start event"
            )
        if not end_info.reviewed_accepted:
            raise ReviewResolutionArtifactValidationError(
                f"{label}: stride {stride.stride_id} uses rejected end event"
            )
        checks.add("accepted_endpoints")
        # Verify every correction is propagated into every affected stride.
        # A replaced event's original ID may not appear in stride endpoints;
        # check via replaced_by_event_id as well.
        for event_id, corr in corrections.items():
            resolved_event = reviewed_by_id.get(event_id)
            replacement_id = (
                resolved_event.replaced_by_event_id if resolved_event else None
            )
            if stride.start_event_id in (event_id, replacement_id):
                if stride.start_frame != corr["new_frame"]:
                    raise ReviewResolutionArtifactValidationError(
                        f"{label}: stride {stride.stride_id} start_event "
                        f"{event_id} not corrected to frame {corr['new_frame']}"
                    )
                correction_propagated = True
            if stride.end_event_id in (event_id, replacement_id):
                if stride.end_frame != corr["new_frame"]:
                    raise ReviewResolutionArtifactValidationError(
                        f"{label}: stride {stride.stride_id} end_event "
                        f"{event_id} not corrected to frame {corr['new_frame']}"
                    )
                correction_propagated = True
        if correction_propagated:
            checks.add("correction_propagated")

    # Verify consecutive same-side shared boundary equality
    by_side: dict[str, list[Stride]] = {}
    for s in reviewed_strides:
        by_side.setdefault(s.side, []).append(s)
    for side_strides in by_side.values():
        for a, b in zip(side_strides, side_strides[1:], strict=False):
            if a.end_event_id != b.start_event_id:
                raise ReviewResolutionArtifactValidationError(
                    f"{label}: consecutive {side_strides[0].side} strides "
                    f"{a.stride_id} and {b.stride_id} do not share boundary: "
                    f"{a.end_event_id} != {b.start_event_id}"
                )
            if a.end_frame != b.start_frame:
                raise ReviewResolutionArtifactValidationError(
                    f"{label}: consecutive {side_strides[0].side} strides "
                    f"{a.stride_id} and {b.stride_id} boundary frame mismatch: "
                    f"{a.end_frame} != {b.start_frame}"
                )
    checks.add("consecutive_shared_boundary")

    return sorted(checks)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


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
    for name in ("gait-stability",):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _build_metadata(
    directory: Path,
    staging: Path,
    input_hashes: Mapping[Path, str],
    assumption_path: Path,
    assumption_hash: str,
    gait_event_config: GaitEventConfig,
    automatic_events: Sequence[GaitEvent],
    resolved_events: Sequence[_ResolvedEvent],
    reviewed_strides: Sequence[Stride],
    corrections: dict[str, dict[str, Any]],
    checks_performed: list[str],
    unreviewed_event_ids: list[str],
    unresolved_blocking: list[str],
    timestamp_source_path: Path,
    timestamp_source_hash: str,
    gait_event_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counts = Counter(r.resolution_disposition for r in resolved_events)
    reviewed_accepted = sum(1 for r in resolved_events if r.reviewed_accepted)
    reviewed_rejected = sum(1 for r in resolved_events if r.reviewed_rejected)

    resolved_by_auto_id = {r.automatic_event_id: r for r in resolved_events}

    boundary_changes: list[dict[str, Any]] = []
    for event_id, corr in corrections.items():
        auto_event = next((e for e in automatic_events if e.event_id == event_id), None)
        resolved_event = resolved_by_auto_id.get(event_id)
        # For replacement: reviewed event ID is the promoted candidate
        if resolved_event and resolved_event.replaced_by_event_id:
            reviewed_event_id = resolved_event.replaced_by_event_id
            reviewed_resolved = resolved_by_auto_id.get(
                resolved_event.replaced_by_event_id
            )
            reviewed_ts = (
                reviewed_resolved.reviewed_timestamp_seconds
                if reviewed_resolved
                else None
            )
            promotion_provenance = (
                f"promoted_candidate:{resolved_event.replaced_by_event_id}"
            )
        else:
            reviewed_event_id = event_id
            reviewed_ts = (
                resolved_event.reviewed_timestamp_seconds if resolved_event else None
            )
            promotion_provenance = None
        boundary_changes.append(
            {
                "automatic_event_id": event_id,
                "reviewed_event_id": reviewed_event_id,
                "boundary": corr["boundary"],
                "automatic_frame": auto_event.frame_index if auto_event else None,
                "reviewed_frame": corr["new_frame"],
                "automatic_timestamp": auto_event.timestamp_seconds
                if auto_event
                else None,
                "reviewed_timestamp": reviewed_ts,
                "source_stride": corr["source_stride"],
                "promotion_provenance": promotion_provenance,
                "reason": "manual_boundary_correction",
            }
        )

    output_hashes: dict[str, str | None] = {}
    for name in OUTPUT_ARTIFACT_NAMES:
        if name == "review_resolution_metadata.json":
            output_hashes[name] = None
        else:
            try:
                output_hashes[name] = sha256_file(staging / name)
            except OSError:
                output_hashes[name] = None

    input_records = {
        path.name: {"path": str(path), "sha256": digest}
        for path, digest in input_hashes.items()
    }
    input_records[assumption_path.name] = {
        "path": str(assumption_path),
        "sha256": assumption_hash,
        "note": ("hashed provenance only; content is not machine-evaluated"),
    }

    return {
        "schema_version": REVIEW_RESOLUTION_SCHEMA_VERSION,
        "algorithm_version": REVIEW_RESOLUTION_ALGORITHM_VERSION,
        "scope": (
            "review resolution of automatic gait-event and stride detections; "
            "no COM, step5, or stability metrics"
        ),
        "run_id": uuid.uuid4().hex,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "inputs": input_records,
        "source_step4": {
            "gait_event_config": asdict(gait_event_config),
            "schema_version": (
                gait_event_metadata.get("schema_version")
                if gait_event_metadata
                else None
            ),
            "algorithm_version": (
                gait_event_metadata.get("algorithm_version")
                if gait_event_metadata
                else None
            ),
        },
        "counts": {
            "automatic_events": len(automatic_events),
            "reviewed_accepted": reviewed_accepted,
            "accepted_unchanged": counts.get("accepted_unchanged", 0),
            "corrected": counts.get("corrected", 0),
            "promoted": counts.get("promoted_from_rejected_candidate", 0),
            "rejected": reviewed_rejected,
            "replaced": counts.get("replaced", 0),
            "reviewed_strides": len(reviewed_strides),
        },
        "boundary_changes": boundary_changes,
        "timestamp_source": {
            "path": str(timestamp_source_path),
            "sha256": timestamp_source_hash,
        },
        "regeneration_method": (
            "construct_strides called on reviewed GaitEvent copies with "
            "reviewed frame/time/status/inclusion/quality"
        ),
        "checks_performed": checks_performed,
        "baseline_unreviewed": {
            "count": len(unreviewed_event_ids),
            "event_ids": unreviewed_event_ids,
        },
        "blocking_unresolved": unresolved_blocking,
        "scientific_unresolved": [
            (
                "Assumption document content (orientation, mirroring, capture "
                "setup, calibration) is hashed provenance only and must be "
                "manually interpreted; it has not been machine-evaluated."
            ),
        ],
        "limitations": [
            "No COM, step5, or stability metrics are produced.",
            "Promoted candidates retain automatic rejection provenance.",
            "Manual corrections are not force-plate or reference-validated.",
            "Reviewed timestamps are canonical pose_frames lookups only.",
        ],
        "outputs": {
            name: {
                "path": str(directory / name),
                "sha256": digest,
            }
            for name, digest in output_hashes.items()
        },
        "runtime": {
            "python_version": sys.version,
            "dependency_versions": _dependency_versions(),
            "git": _git_provenance(),
            "randomness": "none",
        },
    }


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------


def _publish(staging: Path, destination: Path) -> None:
    """Transactional 3-file publish with UUID backups and rollback."""
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for name in OUTPUT_ARTIFACT_NAMES:
            target = destination / name
            if target.exists():
                backup = destination / f"{name}.backup-{uuid.uuid4().hex}"
                target.replace(backup)
                backups[target] = backup
        for name in OUTPUT_ARTIFACT_NAMES:
            staged = staging / name
            if not staged.is_file():
                raise OSError(f"staged Step 4b artifact is missing: {staged}")
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
            "Could not publish complete Step 4b artifact set"
            + (f"; rollback may be incomplete: {detail}" if detail else "")
        ) from exc
    for backup in backups.values():
        with suppress(OSError):
            backup.unlink()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_gait_reviews(
    artifact_directory: str | Path,
    assumption_response_document: str | Path,
) -> ReviewResolutionArtifacts:
    """Resolve manual review of automatic gait-event and stride detections.

    Validates all eight inputs, resolves event reviews and stride corrections,
    reconstructs strides from reviewed events, and publishes the three output
    files as a transactional set.
    """
    directory = Path(artifact_directory).expanduser().resolve()
    if not directory.is_dir():
        raise ReviewResolutionArtifactValidationError(
            f"Artifact directory does not exist: {directory}"
        )

    assumption_path = Path(assumption_response_document).expanduser().resolve()
    if not assumption_path.is_file():
        raise ReviewResolutionArtifactValidationError(
            f"Assumption response document is missing: {assumption_path}"
        )

    # Resolve and verify all input files
    input_paths = tuple(directory / name for name in INPUT_ARTIFACT_NAMES)
    for p in input_paths:
        if not p.is_file():
            raise ReviewResolutionArtifactValidationError(
                f"Required artifact is missing: {p}"
            )

    # Snapshot absolute Path -> SHA-256 hashes
    input_hashes: dict[Path, str] = {}
    for p in input_paths:
        input_hashes[p] = sha256_file(p)
    assumption_hash = sha256_file(assumption_path)

    # Load automatic events and strides
    events_path = directory / "gait_events.csv"
    strides_path = directory / "strides.csv"
    automatic_events = _read_gait_events(events_path)
    automatic_strides = _read_strides(strides_path)
    events_by_id = _build_event_dict(automatic_events)
    strides_by_id = _build_stride_dict(automatic_strides)

    # Load pose frames and validate against preprocessing metadata
    pose_frames_path = directory / "pose_frames.csv"
    pose_timestamps = _read_pose_frames(pose_frames_path)
    preprocessing_metadata = _load_json(
        directory / "preprocessing_metadata.json", "preprocessing_metadata.json"
    )
    # Verify pose_frames hash matches preprocessing metadata
    inputs_section = preprocessing_metadata.get("inputs", {})
    pose_frames_info = inputs_section.get("pose_frames.csv", {})
    stored_hash = (
        pose_frames_info.get("sha256") if isinstance(pose_frames_info, dict) else None
    )
    if not isinstance(stored_hash, str) or not stored_hash:
        raise ReviewResolutionArtifactValidationError(
            "preprocessing_metadata.json must contain inputs.pose_frames.csv.sha256"
        )
    if stored_hash != input_hashes[pose_frames_path]:
        raise ReviewResolutionArtifactValidationError(
            "pose_frames.csv hash does not match preprocessing_metadata.json"
        )

    # Load and validate event reviews
    event_reviews_path = directory / "gait_event_reviews.csv"
    event_reviews = _read_event_reviews(event_reviews_path, events_by_id)

    # Validate event times from pose_frames
    _validate_event_times_from_pose(
        automatic_events, pose_timestamps, "automatic_events"
    )

    # Load and validate stride reviews
    stride_reviews_path = directory / "strides_reviews.csv"
    stride_reviews = _read_stride_reviews(stride_reviews_path, strides_by_id)

    # Validate strides against events
    _validate_strides_against_events(
        automatic_strides, events_by_id, "automatic_strides"
    )

    # Load gait_event_metadata for config
    gait_event_metadata = _load_json(
        directory / "gait_event_metadata.json", "gait_event_metadata.json"
    )
    detector_config = gait_event_metadata.get("config", {}).get("detector", {})
    if not isinstance(detector_config, dict):
        raise ReviewResolutionArtifactValidationError(
            "gait_event_metadata.json config.detector must be an object"
        )
    gait_event_config = GaitEventConfig(**detector_config)

    # Parse stride intent and build corrections
    corrections = _resolve_corrections_from_stride_reviews(stride_reviews, events_by_id)

    # Resolve events
    resolved_events = _resolve_events(
        automatic_events, event_reviews, corrections, pose_timestamps
    )

    # Reconstruct reviewed strides
    reviewed_strides = _reconstruct_reviewed_strides(
        resolved_events,
        automatic_strides,
        automatic_events,
        event_reviews,
        stride_reviews,
        corrections,
        gait_event_config,
    )

    # Final validation
    _validate_final_events(
        resolved_events, pose_timestamps, events_by_id, "reviewed_events"
    )
    checks = _validate_final_strides(
        reviewed_strides, resolved_events, events_by_id, corrections, "reviewed_strides"
    )

    # Identify unreviewed events
    unreviewed_ids = sorted(
        r.automatic_event_id
        for r in resolved_events
        if r.manual_event_review_status == "unreviewed"
    )
    unresolved_blocking: list[str] = []  # No blocking unresolved items

    # Stage files
    staging = Path(
        tempfile.mkdtemp(
            prefix=f"{directory.name}.review-resolution-staging-",
            dir=directory.parent,
        )
    )
    try:
        _write_reviewed_gait_events(
            staging / "reviewed_gait_events.csv", resolved_events
        )
        _write_reviewed_strides(
            staging / "reviewed_strides.csv",
            reviewed_strides,
            strides_by_id,
            {r.stride_id: r for r in stride_reviews},
            resolved_events,
        )

        metadata = _build_metadata(
            directory=directory,
            staging=staging,
            input_hashes=input_hashes,
            assumption_path=assumption_path,
            assumption_hash=assumption_hash,
            gait_event_config=gait_event_config,
            automatic_events=automatic_events,
            resolved_events=resolved_events,
            reviewed_strides=reviewed_strides,
            corrections=corrections,
            checks_performed=checks,
            unreviewed_event_ids=unreviewed_ids,
            unresolved_blocking=unresolved_blocking,
            timestamp_source_path=pose_frames_path,
            timestamp_source_hash=input_hashes[pose_frames_path],
            gait_event_metadata=gait_event_metadata,
        )
        (staging / "review_resolution_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        # Re-verify all input hashes
        for p, expected_hash in input_hashes.items():
            current_hash = sha256_file(p)
            if current_hash != expected_hash:
                raise ReviewResolutionArtifactValidationError(
                    f"Input artifact changed during resolution: {p.name}"
                )
        current_assumption_hash = sha256_file(assumption_path)
        if current_assumption_hash != assumption_hash:
            raise ReviewResolutionArtifactValidationError(
                "Assumption response document changed during resolution"
            )

        # Publish
        _publish(staging, directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return ReviewResolutionArtifacts(
        artifact_directory=directory,
        reviewed_gait_events_path=directory / "reviewed_gait_events.csv",
        reviewed_strides_path=directory / "reviewed_strides.csv",
        review_resolution_metadata_path=directory / "review_resolution_metadata.json",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main_cli() -> int:
    """CLI entry point for Step 4b review resolution."""
    parser = argparse.ArgumentParser(
        description="Resolve manual review of automatic gait-event "
        "and stride detections."
    )
    parser.add_argument("artifact_directory", type=Path)
    parser.add_argument("assumption_response_document", type=Path)
    args = parser.parse_args()
    try:
        artifacts = resolve_gait_reviews(
            args.artifact_directory, args.assumption_response_document
        )
    except (ReviewResolutionError, ArtifactPublishError, OSError, ValueError) as exc:
        print(f"Review resolution failed: {exc}", file=sys.stderr)
        return 1
    print(artifacts.review_resolution_metadata_path)
    return 0


def main() -> int:
    """Thin wrapper for main_cli returning exit code."""
    return main_cli()


if __name__ == "__main__":
    raise SystemExit(main())
