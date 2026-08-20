"""Center-of-mass qualification pipeline.

This module defines the qualification pipeline for COM estimation outputs,
including schema/algorithm version constants, artifact naming, exception
hierarchy, and the frozen artifact container dataclass.
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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast, overload

import cv2

from gait_stability.com_estimation import (
    COM_PROXY_FIELDS,
    DE_LEVA_FEMALE,
    DE_LEVA_MALE,
    MODEL_MASS_TOTAL_FEMALE,
    MODEL_MASS_TOTAL_MALE,
    SEGMENT_ENDPOINTS,
    SEGMENT_NAMES,
    STRIDE_COM_FIELDS,
    UNSUPPORTED_SEGMENTS,
    FrameComResult,
    Point2D,
    SegmentComResult,
    SegmentProvenance,
    StrideComSample,
)
from gait_stability.com_pipeline import (
    _INPUT_BASENAMES,
    COM_ALGORITHM_VERSION,
    COM_OUTPUT_ARTIFACT_NAMES,
    COM_SCHEMA_VERSION,
    REVIEWED_STRIDE_FIELDS,
)
from gait_stability.com_pipeline import (
    _validate_all_inputs as com_validate_all_inputs,
)
from gait_stability.com_qualification import (
    AggregateCoverageResult,
    ComQualificationConfig,
    ParsedReviewedStride,
    ProcessedLandmarkQCRow,
    build_segment_dependency_map,
    compute_qualification,
    theoretical_supported_mass_fraction,
)
from gait_stability.pose_contracts import (
    CANONICAL_POSE_CONNECTIONS,
    MEDIAPIPE_LANDMARK_NAMES,
)
from gait_stability.pose_pipeline import FRAME_FIELDS
from gait_stability.pose_preprocessing import (
    PREPROCESSING_ALGORITHM_VERSION,
    PREPROCESSING_SCHEMA_VERSION,
    PROCESSED_FIELDS,
)
from gait_stability.video_ingestion import ArtifactPublishError, sha256_file

COM_QUALIFICATION_SCHEMA_VERSION: int = 1
"""Schema version for COM qualification output artifacts."""

COM_QUALIFICATION_ALGORITHM_VERSION: str = "step5b-com-qualification-1"
"""Algorithm version identifier for COM qualification pipeline."""

COM_QUALIFICATION_OUTPUT_ARTIFACT_NAMES: tuple[str, str, str] = (
    "com_qualification.json",
    "com_stride_qc.csv",
    "annotated_com.mp4",
)
"""Expected output artifact filenames produced by the qualification pipeline."""

# The six canonical Step 5 upstream input basenames (from com_pipeline)
_STEP5_UPSTREAM_BASENAMES: tuple[str, ...] = _INPUT_BASENAMES

# Step 5 output basenames (from com_pipeline)
_STEP5_OUTPUT_BASENAMES: tuple[str, ...] = COM_OUTPUT_ARTIFACT_NAMES

# Qualification output basenames
_QUALIFICATION_OUTPUT_BASENAMES: tuple[str, ...] = (
    COM_QUALIFICATION_OUTPUT_ARTIFACT_NAMES
)


class ComQualificationPipelineError(Exception):
    """Base exception for COM qualification pipeline errors."""


class ComQualificationArtifactValidationError(ComQualificationPipelineError):
    """Raised when qualification output artifacts fail validation."""


class ComQualificationRenderError(ComQualificationPipelineError):
    """Raised when rendering annotated COM video fails."""


@dataclass(frozen=True, slots=True)
class _QualificationInputs:
    """Resolved absolute paths for all qualification pipeline inputs.

    Contains:
    - Four Step 5 output artifacts (com_proxy.csv, stride_com.csv,
      com_diagnostic.png, com_metadata.json)
    - Six Step 5 upstream inputs (processed_landmarks.csv,
      preprocessing_metadata.json, pose_frames.csv, reviewed_gait_events.csv,
      reviewed_strides.csv, review_resolution_metadata.json)
    - Source video file (resolved from inherited_provenance.source.path_identifier)
    """

    # Step 5 outputs
    com_proxy: Path
    stride_com: Path
    com_diagnostic: Path
    com_metadata: Path

    # Step 5 upstream inputs (six canonical basenames)
    processed_landmarks: Path
    preprocessing_metadata: Path
    pose_frames: Path
    reviewed_gait_events: Path
    reviewed_strides: Path
    review_resolution_metadata: Path

    # Source video
    source_video: Path

    def all_paths(self) -> tuple[Path, ...]:
        """Return all paths as a tuple for iteration."""
        return (
            self.com_proxy,
            self.stride_com,
            self.com_diagnostic,
            self.com_metadata,
            self.processed_landmarks,
            self.preprocessing_metadata,
            self.pose_frames,
            self.reviewed_gait_events,
            self.reviewed_strides,
            self.review_resolution_metadata,
            self.source_video,
        )

    def step5_output_paths(self) -> dict[str, Path]:
        """Return Step 5 output paths keyed by canonical basename."""
        return {
            "com_proxy.csv": self.com_proxy,
            "stride_com.csv": self.stride_com,
            "com_diagnostic.png": self.com_diagnostic,
            "com_metadata.json": self.com_metadata,
        }

    def step5_upstream_paths(self) -> dict[str, Path]:
        """Return Step 5 upstream input paths keyed by canonical basename."""
        return {
            "processed_landmarks.csv": self.processed_landmarks,
            "preprocessing_metadata.json": self.preprocessing_metadata,
            "pose_frames.csv": self.pose_frames,
            "reviewed_gait_events.csv": self.reviewed_gait_events,
            "reviewed_strides.csv": self.reviewed_strides,
            "review_resolution_metadata.json": self.review_resolution_metadata,
        }

    def all_input_paths_by_basename(self) -> dict[str, Path]:
        """Return all input paths keyed by canonical basename."""
        result = {}
        result.update(self.step5_output_paths())
        result.update(self.step5_upstream_paths())
        result["source_video"] = self.source_video
        return result


def _load_json(path: Path, label: str) -> dict[str, Any]:
    """Load and parse JSON file, wrapping errors."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ComQualificationArtifactValidationError(
            f"Failed to parse {label} JSON at {path}: {e}"
        ) from e
    except OSError as e:
        raise ComQualificationArtifactValidationError(
            f"Failed to read {label} JSON at {path}: {e}"
        ) from e
    if not isinstance(data, dict):
        raise ComQualificationArtifactValidationError(
            f"{label} JSON at {path} must be an object, got {type(data).__name__}"
        )
    return data


def _parse_int(value: Any, label: str) -> int:
    """Parse value as finite int, raising on failure."""
    if isinstance(value, bool):
        raise ComQualificationArtifactValidationError(
            f"{label} must be an integer, got boolean"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ComQualificationArtifactValidationError(
                f"{label} must be finite, got {value}"
            )
        if int(value) != value:
            raise ComQualificationArtifactValidationError(
                f"{label} must be an integer, got float {value}"
            )
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as e:
            raise ComQualificationArtifactValidationError(
                f"{label} must be an integer, got '{value}'"
            ) from e
    raise ComQualificationArtifactValidationError(
        f"{label} must be an integer, got {type(value).__name__}"
    )


@overload
def _parse_float(value: Any, label: str, nullable: Literal[False]) -> float: ...


@overload
def _parse_float(value: Any, label: str, nullable: Literal[True]) -> float | None: ...


def _parse_float(value: Any, label: str, nullable: bool = False) -> float | None:
    """Parse value as finite float, raising on failure.

    If nullable=True, accepts blank/None and returns None.
    """
    if nullable and (value is None or value == ""):
        return None
    if isinstance(value, bool):
        raise ComQualificationArtifactValidationError(
            f"{label} must be a number, got boolean"
        )
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ComQualificationArtifactValidationError(
                f"{label} must be finite, got {value}"
            )
        return float(value)
    if isinstance(value, str):
        if value == "":
            raise ComQualificationArtifactValidationError(
                f"{label} must be a number, got empty string"
            )
        try:
            result = float(value)
            if not math.isfinite(result):
                raise ComQualificationArtifactValidationError(
                    f"{label} must be finite, got {value}"
                )
            return result
        except ValueError as e:
            raise ComQualificationArtifactValidationError(
                f"{label} must be a number, got '{value}'"
            ) from e
    raise ComQualificationArtifactValidationError(
        f"{label} must be a number, got {type(value).__name__}"
    )


def _parse_bool_lower(value: Any, label: str) -> bool:
    """Parse value as boolean (exact lowercase 'true'/'false' only)."""
    if isinstance(value, bool):
        raise ComQualificationArtifactValidationError(
            f"{label} must be a string 'true' or 'false', got boolean"
        )
    if isinstance(value, str):
        if value == "true":
            return True
        if value == "false":
            return False
    raise ComQualificationArtifactValidationError(
        f"{label} must be 'true' or 'false', got {value!r}"
    )


def _parse_tuple_pipe(value: Any, label: str) -> tuple[str, ...]:
    """Parse 'a|b|c' string as tuple of strings, blank -> empty tuple."""
    if not isinstance(value, str):
        raise ComQualificationArtifactValidationError(
            f"{label} must be a pipe-separated string, got {type(value).__name__}"
        )
    if value == "":
        return ()
    parts = value.split("|")
    return tuple(parts)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    """Require value to be a mapping, raising on failure."""
    if not isinstance(value, dict):
        raise ComQualificationArtifactValidationError(
            f"{label} must be an object, got {type(value).__name__}"
        )
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    """Require value to be a list, raising on failure."""
    if not isinstance(value, list):
        raise ComQualificationArtifactValidationError(
            f"{label} must be an array, got {type(value).__name__}"
        )
    return value


def _require_str(value: Any, label: str) -> str:
    """Require value to be a string, raising on failure."""
    if not isinstance(value, str):
        raise ComQualificationArtifactValidationError(
            f"{label} must be a string, got {type(value).__name__}"
        )
    return value


def _require_number(value: Any, label: str) -> float:
    """Require value to be a finite number (int or float), raising on failure."""
    return _parse_float(value, label, nullable=False)


def _landmark_contributor_union(
    segment_results: list[SegmentComResult], flag_attribute: str
) -> tuple[str, ...]:
    """Union `contributors` for usable segments where the provenance flag is true.

    Args:
        segment_results: List of SegmentComResult objects.
        flag_attribute: Name of the boolean attribute on SegmentProvenance
            (e.g., "all_raw_observed", "any_x_interpolated").

    Returns:
        Sorted tuple of unique landmark names that are contributors for
        usable segments where the flag attribute is True.
    """
    contributors: set[str] = set()
    for sr in segment_results:
        if sr.usable and getattr(sr.provenance, flag_attribute):
            contributors.update(sr.provenance.contributors)
    return tuple(sorted(contributors))


def _read_com_proxy(
    path: Path,
    primary_threshold: float,
    sex: str,
) -> tuple[FrameComResult, ...]:
    """Read COM proxy CSV and reconstruct FrameComResult objects.

    Args:
        path: Path to the COM proxy CSV file.
        primary_threshold: Minimum mass coverage threshold for frame usability.
        sex: Anthropometry sex ('male' or 'female').

    Returns:
        Tuple of FrameComResult objects ordered by frame_index 0..N-1
        with strictly increasing timestamps.

    Raises:
        ComQualificationArtifactValidationError: If CSV structure or content
            is invalid.
    """
    if sex not in ("male", "female"):
        raise ComQualificationArtifactValidationError(
            f"sex must be 'male' or 'female', got {sex!r}"
        )
    if isinstance(primary_threshold, bool) or not isinstance(
        primary_threshold, (int, float)
    ):
        raise ComQualificationArtifactValidationError(
            "primary_threshold must be a number"
        )
    if not math.isfinite(primary_threshold):
        raise ComQualificationArtifactValidationError(
            "primary_threshold must be finite"
        )
    if not 0.0 <= primary_threshold <= 1.0:
        raise ComQualificationArtifactValidationError(
            "primary_threshold must be between 0 and 1"
        )

    model = DE_LEVA_MALE if sex == "male" else DE_LEVA_FEMALE
    model_total = MODEL_MASS_TOTAL_MALE if sex == "male" else MODEL_MASS_TOTAL_FEMALE

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ComQualificationArtifactValidationError(
                f"CSV at {path} has no header"
            )
        fieldnames = tuple(reader.fieldnames)
        if fieldnames != COM_PROXY_FIELDS:
            raise ComQualificationArtifactValidationError(
                f"CSV fields do not match COM_PROXY_FIELDS at {path}: "
                f"expected {COM_PROXY_FIELDS}, got {fieldnames}"
            )

        frames: list[FrameComResult] = []
        prev_ts: float | None = None
        expected_frame_index = 0

        for row in reader:
            # Parse base fields
            frame_index = _parse_int(row["frame_index"], "frame_index")
            if frame_index != expected_frame_index:
                raise ComQualificationArtifactValidationError(
                    "frame_index must be consecutive 0..N-1: "
                    f"expected {expected_frame_index}, got {frame_index}"
                )
            expected_frame_index += 1

            timestamp_seconds = _parse_float(
                row["timestamp_seconds"], "timestamp_seconds", nullable=False
            )
            if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0.0:
                raise ComQualificationArtifactValidationError(
                    "timestamp_seconds must be nonnegative finite, "
                    f"got {timestamp_seconds}"
                )
            if prev_ts is not None and timestamp_seconds <= prev_ts:
                raise ComQualificationArtifactValidationError(
                    "timestamps must be strictly increasing: "
                    f"{prev_ts} >= {timestamp_seconds}"
                )
            prev_ts = timestamp_seconds

            frame_status = _require_str(row["frame_status"], "frame_status")
            if not frame_status:
                raise ComQualificationArtifactValidationError(
                    "frame_status must be nonempty"
                )

            com_x = _parse_float(row["com_x"], "com_x", nullable=True)
            com_y = _parse_float(row["com_y"], "com_y", nullable=True)
            if (com_x is None) != (com_y is None):
                raise ComQualificationArtifactValidationError(
                    "com_x and com_y must be both finite or both blank"
                )
            if com_x is not None:
                assert com_y is not None
                com = Point2D(com_x, com_y)
            else:
                com = None

            mass_coverage = _parse_float(
                row["mass_coverage"], "mass_coverage", nullable=False
            )
            if not math.isfinite(mass_coverage) or mass_coverage < 0.0:
                raise ComQualificationArtifactValidationError(
                    f"mass_coverage must be finite >= 0, got {mass_coverage}"
                )

            model_total_mass_csv = _parse_float(
                row["model_total_mass"], "model_total_mass", nullable=False
            )
            if not math.isfinite(model_total_mass_csv) or model_total_mass_csv <= 0.0:
                raise ComQualificationArtifactValidationError(
                    "model_total_mass must be positive finite, "
                    f"got {model_total_mass_csv}"
                )
            if abs(model_total_mass_csv - model_total) > 1e-9:
                raise ComQualificationArtifactValidationError(
                    "model_total_mass "
                    f"{model_total_mass_csv} does not match selected model "
                    f"{model_total} (sex={sex})"
                )

            usable_csv = _parse_bool_lower(row["usable"], "usable")
            usable_segments_csv = _parse_tuple_pipe(
                row["usable_segments"], "usable_segments"
            )
            missing_segments_csv = _parse_tuple_pipe(
                row["missing_segments"], "missing_segments"
            )

            # Parse per-segment fields and reconstruct SegmentComResult
            segment_results: list[SegmentComResult] = []
            total_usable_mass = 0.0

            for segment_name in SEGMENT_NAMES:
                base_name = (
                    segment_name
                    if segment_name in ("head", "trunk")
                    else segment_name.split("_", 1)[1]
                )
                seg_mass_fraction = model[base_name]["mass"]
                seg_r = model[base_name]["r"]

                # Parse segment fields
                seg_com_x = _parse_float(
                    row[f"seg_{segment_name}_com_x"],
                    f"seg_{segment_name}_com_x",
                    nullable=True,
                )
                seg_com_y = _parse_float(
                    row[f"seg_{segment_name}_com_y"],
                    f"seg_{segment_name}_com_y",
                    nullable=True,
                )
                if (seg_com_x is None) != (seg_com_y is None):
                    raise ComQualificationArtifactValidationError(
                        f"seg_{segment_name}_com_x and seg_{segment_name}_com_y "
                        "must be both finite or both blank"
                    )
                seg_com = (
                    Point2D(seg_com_x, seg_com_y)  # type: ignore[arg-type]
                    if seg_com_x is not None
                    else None
                )

                seg_mass_fraction_csv = _parse_float(
                    row[f"seg_{segment_name}_mass_fraction"],
                    f"seg_{segment_name}_mass_fraction",
                    nullable=False,
                )
                if abs(seg_mass_fraction_csv - seg_mass_fraction) > 1e-9:
                    raise ComQualificationArtifactValidationError(
                        f"seg_{segment_name}_mass_fraction {seg_mass_fraction_csv} "
                        f"does not match model {seg_mass_fraction}"
                    )

                seg_usable_csv = _parse_bool_lower(
                    row[f"seg_{segment_name}_usable"], f"seg_{segment_name}_usable"
                )
                seg_contributors_csv = _parse_tuple_pipe(
                    row[f"seg_{segment_name}_contributors"],
                    f"seg_{segment_name}_contributors",
                )
                seg_qc_flags_csv = _parse_tuple_pipe(
                    row[f"seg_{segment_name}_qc_flags"], f"seg_{segment_name}_qc_flags"
                )

                # Validate unsupported segment (head)
                if segment_name in UNSUPPORTED_SEGMENTS:
                    if seg_com is not None:
                        raise ComQualificationArtifactValidationError(
                            "Unsupported segment "
                            f"{segment_name} must have blank com_x/com_y"
                        )
                    if seg_usable_csv:
                        raise ComQualificationArtifactValidationError(
                            f"Unsupported segment {segment_name} must have usable=false"
                        )
                    if seg_mass_fraction_csv != seg_mass_fraction:
                        raise ComQualificationArtifactValidationError(
                            f"Unsupported segment {segment_name} mass_fraction mismatch"
                        )

                # Derive provenance booleans from QC flag names
                all_raw_observed = "raw_observed" in seg_qc_flags_csv
                any_x_interpolated = "x_interpolated" in seg_qc_flags_csv
                any_y_interpolated = "y_interpolated" in seg_qc_flags_csv
                any_x_smoothing_changed = "x_smoothing_changed" in seg_qc_flags_csv
                any_y_smoothing_changed = "y_smoothing_changed" in seg_qc_flags_csv
                any_x_smoothing_support_interpolation = (
                    "x_smoothing_support_interpolation" in seg_qc_flags_csv
                )
                any_y_smoothing_support_interpolation = (
                    "y_smoothing_support_interpolation" in seg_qc_flags_csv
                )
                other_qc_limited = "other_qc_limited" in seg_qc_flags_csv

                # Segment usable iff centroid pair finite AND CSV usable is true
                centroid_finite = seg_com is not None
                seg_usable = centroid_finite and seg_usable_csv

                if seg_usable:
                    total_usable_mass += seg_mass_fraction
                else:
                    # If not usable, COM must be None
                    if seg_com is not None:
                        raise ComQualificationArtifactValidationError(
                            f"Segment {segment_name} has finite COM but usable=false"
                        )

                # Build provenance
                prox_name, dist_name = SEGMENT_ENDPOINTS[segment_name]
                provenance = SegmentProvenance(
                    segment_name=segment_name,
                    proximal_landmark=prox_name,
                    distal_landmark=dist_name,
                    contributors=seg_contributors_csv,
                    usable=seg_usable,
                    mass_fraction=seg_mass_fraction,
                    r=seg_r,
                    all_raw_observed=all_raw_observed,
                    any_x_interpolated=any_x_interpolated,
                    any_y_interpolated=any_y_interpolated,
                    any_x_smoothing_changed=any_x_smoothing_changed,
                    any_y_smoothing_changed=any_y_smoothing_changed,
                    any_x_smoothing_support_interpolation=(
                        any_x_smoothing_support_interpolation
                    ),
                    any_y_smoothing_support_interpolation=(
                        any_y_smoothing_support_interpolation
                    ),
                    other_qc_limited=other_qc_limited,
                )

                seg_result = SegmentComResult(
                    segment_name=segment_name,
                    com=seg_com if seg_usable else None,
                    mass_fraction=seg_mass_fraction,
                    usable=seg_usable,
                    provenance=provenance,
                )
                segment_results.append(seg_result)

            # Validate row mass_coverage equals sum usable segment
            # masses (tolerance 1e-9)
            if abs(mass_coverage - total_usable_mass) > 1e-9:
                raise ComQualificationArtifactValidationError(
                    "mass_coverage "
                    f"{mass_coverage} != sum usable segment masses "
                    f"{total_usable_mass} (tolerance 1e-9)"
                )

            # Validate row usable: finite nonzero coverage >= primary_threshold
            expected_usable = mass_coverage > 0.0 and mass_coverage >= primary_threshold
            if usable_csv != expected_usable:
                raise ComQualificationArtifactValidationError(
                    "usable "
                    f"{usable_csv} inconsistent: mass_coverage={mass_coverage}, "
                    f"primary_threshold={primary_threshold}"
                )

            # Validate usable_segments and missing_segments tuples
            usable_segments_computed = tuple(
                sr.segment_name for sr in segment_results if sr.usable
            )
            missing_segments_computed = tuple(
                sr.segment_name for sr in segment_results if not sr.usable
            )
            if usable_segments_csv != usable_segments_computed:
                raise ComQualificationArtifactValidationError(
                    "usable_segments mismatch: "
                    f"CSV={usable_segments_csv}, computed={usable_segments_computed}"
                )
            if missing_segments_csv != missing_segments_computed:
                raise ComQualificationArtifactValidationError(
                    "missing_segments mismatch: "
                    f"CSV={missing_segments_csv}, computed={missing_segments_computed}"
                )

            # Validate aggregate contributor/mass fields against FrameComResult methods
            frame_result = FrameComResult(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                frame_status=frame_status,
                com=com,
                mass_coverage=mass_coverage,
                usable=usable_csv,
                segment_results=tuple(segment_results),
                model_total_mass=model_total,
            )

            # Verify contributor fields
            contributors_raw_observed_csv = _parse_tuple_pipe(
                row["contributors_raw_observed"], "contributors_raw_observed"
            )
            contributors_raw_observed_computed = _landmark_contributor_union(
                segment_results, "all_raw_observed"
            )
            if contributors_raw_observed_csv != contributors_raw_observed_computed:
                raise ComQualificationArtifactValidationError(
                    "contributors_raw_observed mismatch: "
                    f"CSV={contributors_raw_observed_csv}, "
                    f"computed={contributors_raw_observed_computed}"
                )

            contributors_x_interpolated_csv = _parse_tuple_pipe(
                row["contributors_x_interpolated"], "contributors_x_interpolated"
            )
            contributors_x_interpolated_computed = _landmark_contributor_union(
                segment_results, "any_x_interpolated"
            )
            if contributors_x_interpolated_csv != contributors_x_interpolated_computed:
                raise ComQualificationArtifactValidationError(
                    "contributors_x_interpolated mismatch: "
                    f"CSV={contributors_x_interpolated_csv}, "
                    f"computed={contributors_x_interpolated_computed}"
                )

            contributors_y_interpolated_csv = _parse_tuple_pipe(
                row["contributors_y_interpolated"], "contributors_y_interpolated"
            )
            contributors_y_interpolated_computed = _landmark_contributor_union(
                segment_results, "any_y_interpolated"
            )
            if contributors_y_interpolated_csv != contributors_y_interpolated_computed:
                raise ComQualificationArtifactValidationError(
                    "contributors_y_interpolated mismatch: "
                    f"CSV={contributors_y_interpolated_csv}, "
                    f"computed={contributors_y_interpolated_computed}"
                )

            contributors_x_smoothing_changed_csv = _parse_tuple_pipe(
                row["contributors_x_smoothing_changed"],
                "contributors_x_smoothing_changed",
            )
            contributors_x_smoothing_changed_computed = _landmark_contributor_union(
                segment_results, "any_x_smoothing_changed"
            )
            if (
                contributors_x_smoothing_changed_csv
                != contributors_x_smoothing_changed_computed
            ):
                raise ComQualificationArtifactValidationError(
                    "contributors_x_smoothing_changed mismatch: "
                    f"CSV={contributors_x_smoothing_changed_csv}, "
                    f"computed={contributors_x_smoothing_changed_computed}"
                )

            contributors_y_smoothing_changed_csv = _parse_tuple_pipe(
                row["contributors_y_smoothing_changed"],
                "contributors_y_smoothing_changed",
            )
            contributors_y_smoothing_changed_computed = _landmark_contributor_union(
                segment_results, "any_y_smoothing_changed"
            )
            if (
                contributors_y_smoothing_changed_csv
                != contributors_y_smoothing_changed_computed
            ):
                raise ComQualificationArtifactValidationError(
                    "contributors_y_smoothing_changed mismatch: "
                    f"CSV={contributors_y_smoothing_changed_csv}, "
                    f"computed={contributors_y_smoothing_changed_computed}"
                )

            contributors_x_smoothing_support_interpolation_csv = _parse_tuple_pipe(
                row["contributors_x_smoothing_support_interpolation"],
                "contributors_x_smoothing_support_interpolation",
            )
            contributors_x_smoothing_support_interpolation_computed = (
                _landmark_contributor_union(
                    segment_results, "any_x_smoothing_support_interpolation"
                )
            )
            if (
                contributors_x_smoothing_support_interpolation_csv
                != contributors_x_smoothing_support_interpolation_computed
            ):
                raise ComQualificationArtifactValidationError(
                    "contributors_x_smoothing_support_interpolation mismatch: "
                    f"CSV={contributors_x_smoothing_support_interpolation_csv}, "
                    "computed="
                    f"{contributors_x_smoothing_support_interpolation_computed}"
                )

            contributors_y_smoothing_support_interpolation_csv = _parse_tuple_pipe(
                row["contributors_y_smoothing_support_interpolation"],
                "contributors_y_smoothing_support_interpolation",
            )
            contributors_y_smoothing_support_interpolation_computed = (
                _landmark_contributor_union(
                    segment_results, "any_y_smoothing_support_interpolation"
                )
            )
            if (
                contributors_y_smoothing_support_interpolation_csv
                != contributors_y_smoothing_support_interpolation_computed
            ):
                raise ComQualificationArtifactValidationError(
                    "contributors_y_smoothing_support_interpolation mismatch: "
                    f"CSV={contributors_y_smoothing_support_interpolation_csv}, "
                    "computed="
                    f"{contributors_y_smoothing_support_interpolation_computed}"
                )

            contributors_other_qc_limited_csv = _parse_tuple_pipe(
                row["contributors_other_qc_limited"], "contributors_other_qc_limited"
            )
            contributors_other_qc_limited_computed = _landmark_contributor_union(
                segment_results, "other_qc_limited"
            )
            if (
                contributors_other_qc_limited_csv
                != contributors_other_qc_limited_computed
            ):
                raise ComQualificationArtifactValidationError(
                    "contributors_other_qc_limited mismatch: "
                    f"CSV={contributors_other_qc_limited_csv}, "
                    f"computed={contributors_other_qc_limited_computed}"
                )

            # Verify mass fields
            mass_totals = frame_result.provenance_mass_totals()

            mass_x_interpolated_csv = _parse_float(
                row["mass_x_interpolated"], "mass_x_interpolated", nullable=False
            )
            if abs(mass_x_interpolated_csv - mass_totals["x_interpolated"]) > 1e-9:
                raise ComQualificationArtifactValidationError(
                    "mass_x_interpolated mismatch: "
                    f"CSV={mass_x_interpolated_csv}, "
                    f"computed={mass_totals['x_interpolated']}"
                )

            mass_y_interpolated_csv = _parse_float(
                row["mass_y_interpolated"], "mass_y_interpolated", nullable=False
            )
            if abs(mass_y_interpolated_csv - mass_totals["y_interpolated"]) > 1e-9:
                raise ComQualificationArtifactValidationError(
                    "mass_y_interpolated mismatch: "
                    f"CSV={mass_y_interpolated_csv}, "
                    f"computed={mass_totals['y_interpolated']}"
                )

            mass_x_smoothing_changed_csv = _parse_float(
                row["mass_x_smoothing_changed"],
                "mass_x_smoothing_changed",
                nullable=False,
            )
            if (
                abs(mass_x_smoothing_changed_csv - mass_totals["x_smoothing_changed"])
                > 1e-9
            ):
                raise ComQualificationArtifactValidationError(
                    "mass_x_smoothing_changed mismatch: "
                    f"CSV={mass_x_smoothing_changed_csv}, "
                    f"computed={mass_totals['x_smoothing_changed']}"
                )

            mass_y_smoothing_changed_csv = _parse_float(
                row["mass_y_smoothing_changed"],
                "mass_y_smoothing_changed",
                nullable=False,
            )
            if (
                abs(mass_y_smoothing_changed_csv - mass_totals["y_smoothing_changed"])
                > 1e-9
            ):
                raise ComQualificationArtifactValidationError(
                    "mass_y_smoothing_changed mismatch: "
                    f"CSV={mass_y_smoothing_changed_csv}, "
                    f"computed={mass_totals['y_smoothing_changed']}"
                )

            mass_x_smoothing_support_interpolation_csv = _parse_float(
                row["mass_x_smoothing_support_interpolation"],
                "mass_x_smoothing_support_interpolation",
                nullable=False,
            )
            if (
                abs(
                    mass_x_smoothing_support_interpolation_csv
                    - mass_totals["x_smoothing_support_interpolation"]
                )
                > 1e-9
            ):
                raise ComQualificationArtifactValidationError(
                    "mass_x_smoothing_support_interpolation mismatch: "
                    f"CSV={mass_x_smoothing_support_interpolation_csv}, "
                    f"computed={mass_totals['x_smoothing_support_interpolation']}"
                )

            mass_y_smoothing_support_interpolation_csv = _parse_float(
                row["mass_y_smoothing_support_interpolation"],
                "mass_y_smoothing_support_interpolation",
                nullable=False,
            )
            if (
                abs(
                    mass_y_smoothing_support_interpolation_csv
                    - mass_totals["y_smoothing_support_interpolation"]
                )
                > 1e-9
            ):
                raise ComQualificationArtifactValidationError(
                    "mass_y_smoothing_support_interpolation mismatch: "
                    f"CSV={mass_y_smoothing_support_interpolation_csv}, "
                    f"computed={mass_totals['y_smoothing_support_interpolation']}"
                )

            mass_other_qc_limited_csv = _parse_float(
                row["mass_other_qc_limited"], "mass_other_qc_limited", nullable=False
            )
            if abs(mass_other_qc_limited_csv - mass_totals["other_qc_limited"]) > 1e-9:
                raise ComQualificationArtifactValidationError(
                    "mass_other_qc_limited mismatch: "
                    f"CSV={mass_other_qc_limited_csv}, "
                    f"computed={mass_totals['other_qc_limited']}"
                )

            mass_missing_csv = _parse_float(
                row["mass_missing"], "mass_missing", nullable=False
            )
            if abs(mass_missing_csv - mass_totals["missing"]) > 1e-9:
                raise ComQualificationArtifactValidationError(
                    "mass_missing mismatch: "
                    f"CSV={mass_missing_csv}, "
                    f"computed={mass_totals['missing']}"
                )

            # Also verify raw_observed mass (though not in CSV base fields explicitly,
            # it's implied). The CSV doesn't have mass_raw_observed field, so we skip.

            frames.append(frame_result)

        if not frames:
            raise ComQualificationArtifactValidationError("CSV contains no data rows")

        return tuple(frames)


# ---------------------------------------------------------------------------
# Internal: strict readers for qualification pipeline inputs
# ---------------------------------------------------------------------------


def _read_pose_frames(
    path: Path,
) -> dict[int, tuple[float, str]]:
    """Read and validate pose_frames.csv strictly.

    Returns {frame_index: (nominal_timestamp_seconds, status)} ordered by frame_index.

    Enforces:
    - exact FRAME_FIELDS header
    - nonempty
    - file row order: frame_index == row_number-2 (ordered contiguous from zero)
    - strictly increasing finite nonnegative nominal timestamps
    - valid status (nonempty string)
    - nonnegative landmark_count
    - backend_timestamp_milliseconds parsed (blank allowed for decode failures)
      and nonnegative when present
    """
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != FRAME_FIELDS:
                raise ComQualificationArtifactValidationError(
                    "pose_frames.csv header must exactly match FRAME_FIELDS"
                )
            result: dict[int, tuple[float, str]] = {}
            prev_timestamp: float | None = None
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(row[field] is None for field in FRAME_FIELDS):
                    raise ComQualificationArtifactValidationError(
                        f"pose_frames.csv row {row_number}: malformed columns"
                    )
                fi = _parse_int(row["frame_index"], "frame_index")
                expected_frame = row_number - 2
                if fi != expected_frame:
                    raise ComQualificationArtifactValidationError(
                        f"pose_frames.csv row {row_number}: frame_index {fi} "
                        f"does not match expected {expected_frame} from file "
                        "row order (duplicate or out-of-order)"
                    )
                ts = _parse_float(
                    row["nominal_timestamp_seconds"],
                    "nominal_timestamp_seconds",
                    nullable=False,
                )
                if ts is None:
                    raise ComQualificationArtifactValidationError(
                        f"pose_frames.csv row {row_number}: timestamp must be finite"
                    )
                if ts < 0.0:
                    raise ComQualificationArtifactValidationError(
                        f"pose_frames.csv row {row_number}: "
                        "timestamp must be nonnegative"
                    )
                if prev_timestamp is not None and ts <= prev_timestamp:
                    raise ComQualificationArtifactValidationError(
                        f"pose_frames.csv row {row_number}: "
                        "timestamps must be strictly increasing"
                    )
                prev_timestamp = ts

                status = row["status"]
                if not status:
                    raise ComQualificationArtifactValidationError(
                        f"pose_frames.csv row {row_number}: status must be nonempty"
                    )

                landmark_count = _parse_int(row["landmark_count"], "landmark_count")
                if landmark_count < 0:
                    raise ComQualificationArtifactValidationError(
                        f"pose_frames.csv row {row_number}: "
                        "landmark_count must be nonnegative"
                    )

                # Parse backend_timestamp_milliseconds (blank allowed)
                bt_ms_text = row["backend_timestamp_milliseconds"]
                if bt_ms_text != "":
                    bt_ms = _parse_int(bt_ms_text, "backend_timestamp_milliseconds")
                    if bt_ms < 0:
                        raise ComQualificationArtifactValidationError(
                            f"pose_frames.csv row {row_number}: "
                            "backend_timestamp_milliseconds must be nonnegative"
                        )

                result[fi] = (ts, status)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ComQualificationArtifactValidationError(
            f"Could not read {path}: {exc}"
        ) from exc

    if not result:
        raise ComQualificationArtifactValidationError(
            "pose_frames.csv must contain at least one row"
        )

    return result


def _read_reviewed_strides(
    path: Path,
    pose_frames: dict[int, tuple[float, str]],
) -> list[ParsedReviewedStride]:
    """Read and validate reviewed_strides.csv strictly.

    Enforces:
    - exact REVIEWED_STRIDE_FIELDS header in that order
    - unique nonempty stride_ids in file order
    - valid side (left/right)
    - start_frame < end_frame
    - start_timestamp_seconds < end_timestamp_seconds
    - positive duration_seconds equals difference (tolerance 1e-9)
    - start_frame and end_frame exist in pose_frames
    - start_timestamp_seconds and end_timestamp_seconds exactly match pose_frames
      timestamps at those frame indices (tolerance 1e-9)
    - quality in {high, review, low}
    - nonnegative contralateral_event_count
    - nonempty source and automatic_stride_id
    - review_intent in {accept, correct}
    - all tuple pipe fields parsed (sequence_notes)
    """
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != REVIEWED_STRIDE_FIELDS:
                raise ComQualificationArtifactValidationError(
                    "reviewed_strides.csv header must exactly match "
                    "REVIEWED_STRIDE_FIELDS"
                )
            strides: list[ParsedReviewedStride] = []
            seen_ids: set[str] = set()
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(
                    row[field] is None for field in REVIEWED_STRIDE_FIELDS
                ):
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: malformed columns"
                    )

                stride_id = row["stride_id"]
                if not stride_id:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        "stride_id must be nonempty"
                    )
                if stride_id in seen_ids:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"duplicate stride_id {stride_id}"
                    )
                seen_ids.add(stride_id)

                side = row["side"]
                if side not in {"left", "right"}:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: invalid side {side!r}"
                    )

                start_event_id = row["start_event_id"]
                end_event_id = row["end_event_id"]

                start_frame = _parse_int(row["start_frame"], "start_frame")
                end_frame = _parse_int(row["end_frame"], "end_frame")
                if start_frame >= end_frame:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"start_frame {start_frame} must be < "
                        f"end_frame {end_frame}"
                    )

                start_ts = _parse_float(
                    row["start_timestamp_seconds"],
                    "start_timestamp_seconds",
                    nullable=False,
                )
                end_ts = _parse_float(
                    row["end_timestamp_seconds"],
                    "end_timestamp_seconds",
                    nullable=False,
                )
                if start_ts is None or end_ts is None:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        "start/end timestamps must be finite"
                    )
                if start_ts >= end_ts:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"start_timestamp {start_ts} must be < "
                        f"end_timestamp {end_ts}"
                    )

                duration = _parse_float(
                    row["duration_seconds"], "duration_seconds", nullable=False
                )
                if duration is None:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        "duration_seconds must be finite"
                    )
                if duration <= 0.0:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"duration_seconds must be positive, got {duration}"
                    )
                expected_duration = end_ts - start_ts
                if abs(duration - expected_duration) > 1e-9:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"duration_seconds {duration} does not equal "
                        f"end-start difference {expected_duration}"
                    )

                # Validate frames exist in pose_frames
                if start_frame not in pose_frames:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"start_frame {start_frame} not in pose_frames.csv"
                    )
                if end_frame not in pose_frames:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"end_frame {end_frame} not in pose_frames.csv"
                    )

                # Validate timestamps exactly match pose_frames
                pose_start_ts = pose_frames[start_frame][0]
                pose_end_ts = pose_frames[end_frame][0]
                if abs(start_ts - pose_start_ts) > 1e-9:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"start_timestamp_seconds {start_ts} does not match "
                        f"pose_frames timestamp {pose_start_ts} for frame {start_frame}"
                    )
                if abs(end_ts - pose_end_ts) > 1e-9:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"end_timestamp_seconds {end_ts} does not match "
                        f"pose_frames timestamp {pose_end_ts} for frame {end_frame}"
                    )

                quality = row["quality"]
                if quality not in {"high", "review", "low"}:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"invalid quality {quality!r}"
                    )

                contralateral_id = row["contralateral_event_id"]
                if contralateral_id == "":
                    contralateral_id = None

                contralateral_count = _parse_int(
                    row["contralateral_event_count"], "contralateral_event_count"
                )
                if contralateral_count < 0:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        "contralateral_event_count must be nonnegative"
                    )

                sequence_notes = _parse_tuple_pipe(
                    row["sequence_notes"], "sequence_notes"
                )

                source = row["source"]
                if not source:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        "source must be nonempty"
                    )

                review_status = row["review_status"]

                automatic_stride_id = row["automatic_stride_id"]
                if not automatic_stride_id:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        "automatic_stride_id must be nonempty"
                    )

                review_intent = row["review_intent"]
                if review_intent not in {"accept", "correct"}:
                    raise ComQualificationArtifactValidationError(
                        f"reviewed_strides.csv row {row_number}: "
                        f"invalid review_intent {review_intent!r}"
                    )

                review_changes = row["review_changes"]
                provenance_notes = row["provenance_notes"]

                strides.append(
                    ParsedReviewedStride(
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
        raise ComQualificationArtifactValidationError(
            f"Could not read reviewed_strides.csv: {exc}"
        ) from exc

    if not strides:
        raise ComQualificationArtifactValidationError(
            "reviewed_strides.csv must contain at least one row"
        )

    return strides


# ---------------------------------------------------------------------------
# Internal: qualification pipeline input resolution and validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SourceInfo:
    """Validated source video information."""

    path: Path
    sha256: str
    width_pixels: int
    height_pixels: int
    nominal_fps: float
    nominal_frame_count: int
    nominal_duration_seconds: float
    project_rotation: str
    project_mirroring: str


def _snapshot_hashes(inputs: _QualificationInputs) -> dict[Path, str]:
    """Snapshot SHA-256 hashes for all qualification input files.

    Args:
        inputs: _QualificationInputs containing all resolved paths.

    Returns:
        Dictionary mapping each input Path to its SHA-256 digest.
    """
    hashes: dict[Path, str] = {}
    for path in inputs.all_paths():
        hashes[path] = sha256_file(path)
    return hashes


def _recheck_hashes(snapshot: dict[Path, str]) -> None:
    """Recheck all input hashes match the original snapshot.

    Args:
        snapshot: Dictionary mapping Path to expected SHA-256 digest.

    Raises:
        ComQualificationArtifactValidationError: If any file's current hash
            differs from the snapshot.
    """
    for path, expected in snapshot.items():
        current = sha256_file(path)
        if current != expected:
            raise ComQualificationArtifactValidationError(
                f"Input artifact changed during qualification: {path}"
            )


def _require_int(value: Any, label: str) -> int:
    """Require value to be an integer, raising on failure."""
    if isinstance(value, bool):
        raise ComQualificationArtifactValidationError(
            f"{label} must be an integer, got boolean"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or int(value) != value:
            raise ComQualificationArtifactValidationError(
                f"{label} must be an integer, got float {value}"
            )
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as e:
            raise ComQualificationArtifactValidationError(
                f"{label} must be an integer, got '{value}'"
            ) from e
    raise ComQualificationArtifactValidationError(
        f"{label} must be an integer, got {type(value).__name__}"
    )


def _require_bool(value: Any, label: str) -> bool:
    """Require value to be a boolean, raising on failure."""
    if not isinstance(value, bool):
        raise ComQualificationArtifactValidationError(
            f"{label} must be a boolean, got {type(value).__name__}"
        )
    return value


def _require_list_str(value: Any, label: str) -> list[str]:
    """Require value to be a list of strings, raising on failure."""
    if not isinstance(value, list):
        raise ComQualificationArtifactValidationError(
            f"{label} must be a list, got {type(value).__name__}"
        )
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise ComQualificationArtifactValidationError(
                f"{label}[{i}] must be a string, got {type(item).__name__}"
            )
    return value


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    """Require value to be a dict, raising on failure."""
    if not isinstance(value, dict):
        raise ComQualificationArtifactValidationError(
            f"{label} must be an object, got {type(value).__name__}"
        )
    return value


def _validate_coefficient_table(
    coeff_table: dict[str, dict[str, float]], sex: Literal["male", "female"]
) -> None:
    """Validate that coefficient table exactly matches DE_LEVA constants."""
    expected = DE_LEVA_MALE if sex == "male" else DE_LEVA_FEMALE
    if set(coeff_table.keys()) != set(expected.keys()):
        raise ComQualificationArtifactValidationError(
            f"Coefficient table keys mismatch for {sex}: "
            f"expected {set(expected.keys())}, got {set(coeff_table.keys())}"
        )
    for seg, vals in coeff_table.items():
        exp = expected[seg]
        if abs(vals.get("mass_fraction", -1) - exp["mass"]) > 1e-9:
            raise ComQualificationArtifactValidationError(
                f"Coefficient table mass mismatch for {seg} ({sex}): "
                f"expected {exp['mass']}, got {vals.get('mass_fraction')}"
            )
        if abs(vals.get("centroid_ratio_r", -1) - exp["r"]) > 1e-9:
            raise ComQualificationArtifactValidationError(
                f"Coefficient table r mismatch for {seg} ({sex}): "
                f"expected {exp['r']}, got {vals.get('centroid_ratio_r')}"
            )


def _resolve_and_validate_metadata(
    directory: Path, video_override: Path | None = None
) -> tuple[
    _QualificationInputs,
    dict[str, Any],
    dict[str, Any],
    Literal["male", "female"],
    float,
    int,
    _SourceInfo,
]:
    """Resolve all input paths and validate metadata contracts.

    Args:
        directory: Artifact directory containing Step 5 outputs and upstream inputs.
        video_override: Optional override for source video path. If provided,
            the source file must still exist and its hash must match the
            inherited provenance source path_identifier hash from
            preprocessing_metadata.

    Returns:
        Tuple of:
        - _QualificationInputs: All resolved absolute input paths
        - com_metadata: Validated com_metadata.json object
        - preprocessing_metadata: Validated preprocessing_metadata.json object
        - sex: Anthropometry sex ('male' or 'female')
        - primary_threshold: Minimum mass coverage threshold (finite, [0, 1])
        - normalized_n: Normalized stride sample count (>= 2)
        - source_info: Validated _SourceInfo with path, hash, and video metadata

    Raises:
        ComQualificationArtifactValidationError: If any file is missing, any
            hash mismatches, or any schema/contract validation fails.
    """
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise ComQualificationArtifactValidationError(
            f"Artifact directory does not exist: {directory}"
        )

    # --- Step 1: Resolve all required file paths ---
    # Step 5 outputs
    com_proxy_path = directory / "com_proxy.csv"
    stride_com_path = directory / "stride_com.csv"
    com_diagnostic_path = directory / "com_diagnostic.png"
    com_metadata_path = directory / "com_metadata.json"

    # Step 5 upstream inputs (six canonical basenames)
    processed_landmarks_path = directory / "processed_landmarks.csv"
    preprocessing_metadata_path = directory / "preprocessing_metadata.json"
    pose_frames_path = directory / "pose_frames.csv"
    reviewed_gait_events_path = directory / "reviewed_gait_events.csv"
    reviewed_strides_path = directory / "reviewed_strides.csv"
    review_resolution_metadata_path = directory / "review_resolution_metadata.json"

    # Verify all required files exist
    required_files = [
        ("com_proxy.csv", com_proxy_path),
        ("stride_com.csv", stride_com_path),
        ("com_diagnostic.png", com_diagnostic_path),
        ("com_metadata.json", com_metadata_path),
        ("processed_landmarks.csv", processed_landmarks_path),
        ("preprocessing_metadata.json", preprocessing_metadata_path),
        ("pose_frames.csv", pose_frames_path),
        ("reviewed_gait_events.csv", reviewed_gait_events_path),
        ("reviewed_strides.csv", reviewed_strides_path),
        ("review_resolution_metadata.json", review_resolution_metadata_path),
    ]
    for name, path in required_files:
        if not path.is_file():
            raise ComQualificationArtifactValidationError(
                f"Required file is missing: {name} at {path}"
            )

    # --- Step 2: Load and validate com_metadata.json ---
    com_metadata = _load_json(com_metadata_path, "com_metadata.json")

    # Validate schema_version
    sv = com_metadata.get("schema_version")
    if sv != COM_SCHEMA_VERSION:
        raise ComQualificationArtifactValidationError(
            f"com_metadata.json schema_version must be {COM_SCHEMA_VERSION}, got {sv}"
        )

    # Validate algorithm_version
    av = com_metadata.get("algorithm_version")
    if av != COM_ALGORITHM_VERSION:
        raise ComQualificationArtifactValidationError(
            f"com_metadata.json algorithm_version must be "
            f"{COM_ALGORITHM_VERSION!r}, got {av!r}"
        )

    # Validate config section
    config_section = _require_dict(com_metadata.get("config"), "config")
    sex_raw = _require_str(
        config_section.get("anthropometry_sex"), "config.anthropometry_sex"
    )
    if sex_raw not in ("male", "female"):
        raise ComQualificationArtifactValidationError(
            f"config.anthropometry_sex must be 'male' or 'female', got {sex_raw!r}"
        )
    sex = cast(Literal["male", "female"], sex_raw)

    primary_threshold = _require_number(
        config_section.get("minimum_mass_coverage"), "config.minimum_mass_coverage"
    )
    if not 0.0 <= primary_threshold <= 1.0:
        raise ComQualificationArtifactValidationError(
            f"config.minimum_mass_coverage must be in [0, 1], got {primary_threshold}"
        )

    normalized_n = _require_int(
        config_section.get("normalized_stride_samples"),
        "config.normalized_stride_samples",
    )
    if normalized_n < 2:
        raise ComQualificationArtifactValidationError(
            f"config.normalized_stride_samples must be >= 2, got {normalized_n}"
        )

    # Validate coverage.threshold equals config.minimum_mass_coverage
    coverage_section = _require_dict(com_metadata.get("coverage"), "coverage")
    cov_threshold = _require_number(
        coverage_section.get("threshold"), "coverage.threshold"
    )
    if abs(cov_threshold - primary_threshold) > 1e-9:
        raise ComQualificationArtifactValidationError(
            f"coverage.threshold ({cov_threshold}) must equal "
            f"config.minimum_mass_coverage ({primary_threshold})"
        )

    # Validate algorithm.selected_sex matches config.anthropometry_sex
    algorithm_section = _require_dict(com_metadata.get("algorithm"), "algorithm")
    selected_sex = _require_str(
        algorithm_section.get("selected_sex"), "algorithm.selected_sex"
    )
    if selected_sex != sex:
        raise ComQualificationArtifactValidationError(
            f"algorithm.selected_sex ({selected_sex}) must match "
            f"config.anthropometry_sex ({sex})"
        )

    # Validate model.total_mass matches DE_LEVA constant
    model_total_expected = (
        MODEL_MASS_TOTAL_MALE if sex == "male" else MODEL_MASS_TOTAL_FEMALE
    )
    model_total_actual = _require_number(
        algorithm_section.get("model_total_mass"), "algorithm.model_total_mass"
    )
    if abs(model_total_actual - model_total_expected) > 1e-9:
        raise ComQualificationArtifactValidationError(
            f"algorithm.model_total_mass ({model_total_actual}) does not match "
            f"expected DE_LEVA total for {sex} ({model_total_expected})"
        )

    # Validate full coefficient table matches DE_LEVA constants
    coeff_table = _require_dict(
        algorithm_section.get("coefficient_table"), "algorithm.coefficient_table"
    )
    _validate_coefficient_table(coeff_table, sex)

    # Validate unsupported_segments exactly ['head']
    unsupported = _require_list_str(
        algorithm_section.get("unsupported_segments"), "algorithm.unsupported_segments"
    )
    if unsupported != ["head"]:
        raise ComQualificationArtifactValidationError(
            f"algorithm.unsupported_segments must be exactly "
            f"['head'], got {unsupported}"
        )

    # Validate represented_mass_max equals theoretical_supported_mass_fraction(sex)
    represented_max_expected = theoretical_supported_mass_fraction(sex)
    represented_max_actual = _require_number(
        algorithm_section.get("represented_mass_max"), "algorithm.represented_mass_max"
    )
    if abs(represented_max_actual - represented_max_expected) > 1e-9:
        raise ComQualificationArtifactValidationError(
            f"algorithm.represented_mass_max ({represented_max_actual}) does not match "
            f"theoretical_supported_mass_fraction({sex}) ({represented_max_expected})"
        )

    # Validate Step 5 metadata input entries for all six canonical basenames
    inputs_section = _require_dict(com_metadata.get("inputs"), "inputs")
    for basename in _STEP5_UPSTREAM_BASENAMES:
        entry = _require_dict(inputs_section.get(basename), f"inputs.{basename}")
        entry_path = _require_str(entry.get("path"), f"inputs.{basename}.path")
        entry_hash = _require_str(entry.get("sha256"), f"inputs.{basename}.sha256")
        # Path basename must match canonical
        if Path(entry_path).name != basename:
            raise ComQualificationArtifactValidationError(
                f"inputs.{basename}.path basename {Path(entry_path).name!r} "
                f"does not match canonical {basename!r}"
            )
        # Hash must match actual file
        actual_path = directory / basename
        actual_hash = sha256_file(actual_path)
        if actual_hash != entry_hash:
            raise ComQualificationArtifactValidationError(
                f"inputs.{basename}.sha256 does not match actual file hash"
            )

    # Validate Step 5 metadata output entries
    outputs_section = _require_dict(com_metadata.get("outputs"), "outputs")
    for basename in _STEP5_OUTPUT_BASENAMES:
        if basename == "com_metadata.json":
            # Self-null semantics: sha256 is null, path basename must match
            entry = _require_dict(outputs_section.get(basename), f"outputs.{basename}")
            entry_path = _require_str(entry.get("path"), f"outputs.{basename}.path")
            output_hash: str | None = entry.get("sha256")
            if output_hash is not None:
                raise ComQualificationArtifactValidationError(
                    "outputs.com_metadata.json.sha256 must be null "
                    "(self-null semantics)"
                )
            if Path(entry_path).name != basename:
                raise ComQualificationArtifactValidationError(
                    f"outputs.com_metadata.json.path basename "
                    f"{Path(entry_path).name!r} does not match "
                    f"canonical {basename!r}"
                )
        else:
            entry = _require_dict(outputs_section.get(basename), f"outputs.{basename}")
            entry_path = _require_str(entry.get("path"), f"outputs.{basename}.path")
            entry_hash = _require_str(entry.get("sha256"), f"outputs.{basename}.sha256")
            if Path(entry_path).name != basename:
                raise ComQualificationArtifactValidationError(
                    f"outputs.{basename}.path basename {Path(entry_path).name!r} "
                    f"does not match canonical {basename!r}"
                )
            actual_path = directory / basename
            actual_hash = sha256_file(actual_path)
            if actual_hash != entry_hash:
                raise ComQualificationArtifactValidationError(
                    f"outputs.{basename}.sha256 does not match actual file hash"
                )

    # --- Step 3: Load and validate preprocessing_metadata.json ---
    preprocessing_metadata = _load_json(
        preprocessing_metadata_path, "preprocessing_metadata.json"
    )

    # Validate schema/algorithm version (Step 3 constants)
    if preprocessing_metadata.get("schema_version") != PREPROCESSING_SCHEMA_VERSION:
        raise ComQualificationArtifactValidationError(
            f"preprocessing_metadata.json schema_version must be "
            f"{PREPROCESSING_SCHEMA_VERSION}, got "
            f"{preprocessing_metadata.get('schema_version')}"
        )
    if preprocessing_metadata.get("algorithm_version") != (
        PREPROCESSING_ALGORITHM_VERSION
    ):
        raise ComQualificationArtifactValidationError(
            f"preprocessing_metadata.json algorithm_version must be "
            f"{PREPROCESSING_ALGORITHM_VERSION!r}, got "
            f"{preprocessing_metadata.get('algorithm_version')!r}"
        )

    # Validate exact path/hash linking
    # inputs.pose_frames.csv
    preproc_inputs = _require_dict(
        preprocessing_metadata.get("inputs"), "preprocessing_metadata.inputs"
    )
    pf_entry = _require_dict(
        preproc_inputs.get("pose_frames.csv"),
        "preprocessing_metadata.inputs.pose_frames.csv",
    )
    pf_path_str = _require_str(
        pf_entry.get("path"), "preprocessing_metadata.inputs.pose_frames.csv.path"
    )
    if Path(pf_path_str).name != "pose_frames.csv":
        raise ComQualificationArtifactValidationError(
            "preprocessing_metadata.inputs.pose_frames.csv path basename mismatch"
        )
    pf_hash_stored = _require_str(
        pf_entry.get("sha256"), "preprocessing_metadata.inputs.pose_frames.csv.sha256"
    )
    pf_hash_actual = sha256_file(pose_frames_path)
    if pf_hash_stored != pf_hash_actual:
        raise ComQualificationArtifactValidationError(
            "pose_frames.csv hash does not match preprocessing_metadata input hash"
        )

    # outputs.processed_landmarks.csv
    preproc_outputs = _require_dict(
        preprocessing_metadata.get("outputs"), "preprocessing_metadata.outputs"
    )
    pl_entry = _require_dict(
        preproc_outputs.get("processed_landmarks.csv"),
        "preprocessing_metadata.outputs.processed_landmarks.csv",
    )
    pl_path_str = _require_str(
        pl_entry.get("path"),
        "preprocessing_metadata.outputs.processed_landmarks.csv.path",
    )
    if Path(pl_path_str).name != "processed_landmarks.csv":
        raise ComQualificationArtifactValidationError(
            "preprocessing_metadata.outputs.processed_landmarks.csv "
            "path basename mismatch"
        )
    pl_hash_stored = _require_str(
        pl_entry.get("sha256"),
        "preprocessing_metadata.outputs.processed_landmarks.csv.sha256",
    )
    pl_hash_actual = sha256_file(processed_landmarks_path)
    if pl_hash_stored != pl_hash_actual:
        raise ComQualificationArtifactValidationError(
            "processed_landmarks.csv hash does not match "
            "preprocessing_metadata output hash"
        )

    # --- Step 4: Resolve source video ---
    # From inherited_provenance.source.path_identifier in preprocessing_metadata
    inherited_prov = _require_dict(
        preprocessing_metadata.get("inherited_provenance"),
        "preprocessing_metadata.inherited_provenance",
    )
    source_section = _require_dict(
        inherited_prov.get("source"),
        "preprocessing_metadata.inherited_provenance.source",
    )
    source_path_identifier = _require_str(
        source_section.get("path_identifier"),
        "preprocessing_metadata.inherited_provenance.source.path_identifier",
    )
    source_hash_expected = _require_str(
        source_section.get("sha256"),
        "preprocessing_metadata.inherited_provenance.source.sha256",
    )

    # Use override if provided, but still require hash match
    if video_override is not None:
        source_path = video_override.expanduser().resolve()
        if not source_path.is_file():
            raise ComQualificationArtifactValidationError(
                f"Override source video does not exist: {source_path}"
            )
        # Hash must still match the inherited source sha256
        source_hash_actual = sha256_file(source_path)
        if source_hash_actual != source_hash_expected:
            raise ComQualificationArtifactValidationError(
                f"Override source video hash ({source_hash_actual}) does not match "
                f"inherited source hash ({source_hash_expected})"
            )
    else:
        source_path = Path(source_path_identifier).expanduser().resolve()
        if not source_path.is_file():
            raise ComQualificationArtifactValidationError(
                f"Source video from inherited_provenance does not exist: {source_path}"
            )
        source_hash_actual = sha256_file(source_path)
        if source_hash_actual != source_hash_expected:
            raise ComQualificationArtifactValidationError(
                f"Source video hash ({source_hash_actual}) does not match "
                f"inherited source hash ({source_hash_expected})"
            )

    # Validate video metadata: positive width/height/fps
    if not isinstance(source_section.get("width_pixels"), int) or (
        source_section["width_pixels"] <= 0
    ):
        raise ComQualificationArtifactValidationError(
            "inherited_provenance.source.width_pixels must be a positive integer"
        )
    if not isinstance(source_section.get("height_pixels"), int) or (
        source_section["height_pixels"] <= 0
    ):
        raise ComQualificationArtifactValidationError(
            "inherited_provenance.source.height_pixels must be a positive integer"
        )
    fps_val = source_section.get("nominal_fps")
    if (
        not isinstance(fps_val, (int, float))
        or not math.isfinite(fps_val)
        or fps_val <= 0
    ):
        raise ComQualificationArtifactValidationError(
            "inherited_provenance.source.nominal_fps must be a positive finite number"
        )

    # Validate project_rotation = 'none' and project_mirroring = 'none'
    project_rotation = _require_str(
        source_section.get("project_rotation"),
        "inherited_provenance.source.project_rotation",
    )
    if project_rotation != "none":
        raise ComQualificationArtifactValidationError(
            f"inherited_provenance.source.project_rotation must be "
            f"'none', got {project_rotation!r}"
        )
    project_mirroring = _require_str(
        source_section.get("project_mirroring"),
        "inherited_provenance.source.project_mirroring",
    )
    if project_mirroring != "none":
        raise ComQualificationArtifactValidationError(
            f"inherited_provenance.source.project_mirroring must be "
            f"'none', got {project_mirroring!r}"
        )

    source_info = _SourceInfo(
        path=source_path,
        sha256=source_hash_actual,
        width_pixels=source_section["width_pixels"],
        height_pixels=source_section["height_pixels"],
        nominal_fps=float(fps_val),
        nominal_frame_count=source_section.get("nominal_frame_count", 0),
        nominal_duration_seconds=source_section.get("nominal_duration_seconds", 0.0),
        project_rotation=project_rotation,
        project_mirroring=project_mirroring,
    )

    # --- Step 5: Reject output path alias with any input/source ---
    # Check that no output path aliases any input path
    input_paths_set = {
        p.resolve()
        for p in [
            com_proxy_path,
            stride_com_path,
            com_diagnostic_path,
            com_metadata_path,
            processed_landmarks_path,
            preprocessing_metadata_path,
            pose_frames_path,
            reviewed_gait_events_path,
            reviewed_strides_path,
            review_resolution_metadata_path,
            source_path,
        ]
    }
    output_paths = [directory / name for name in _QUALIFICATION_OUTPUT_BASENAMES]
    for out_path in output_paths:
        if out_path.resolve() in input_paths_set:
            raise ComQualificationArtifactValidationError(
                f"Output path {out_path.name} resolves to an input/source path"
            )
    # Check outputs don't alias each other
    out_resolved = set()
    for out_path in output_paths:
        resolved = out_path.resolve()
        if resolved in out_resolved:
            raise ComQualificationArtifactValidationError(
                f"Output path {out_path.name} aliases another output path"
            )
        out_resolved.add(resolved)

    # --- Step 6: Build and return _QualificationInputs ---
    inputs = _QualificationInputs(
        com_proxy=com_proxy_path,
        stride_com=stride_com_path,
        com_diagnostic=com_diagnostic_path,
        com_metadata=com_metadata_path,
        processed_landmarks=processed_landmarks_path,
        preprocessing_metadata=preprocessing_metadata_path,
        pose_frames=pose_frames_path,
        reviewed_gait_events=reviewed_gait_events_path,
        reviewed_strides=reviewed_strides_path,
        review_resolution_metadata=review_resolution_metadata_path,
        source_video=source_path,
    )

    return (
        inputs,
        com_metadata,
        preprocessing_metadata,
        sex,
        primary_threshold,
        normalized_n,
        source_info,
    )


def _read_stride_com(
    path: Path,
    reviewed_strides: list[ParsedReviewedStride],
    pose_frames: dict[int, tuple[float, str]],
    normalized_n: int,
) -> dict[str, tuple[StrideComSample, ...]]:
    """Read and validate stride_com.csv strictly.

    Enforces:
    - exact STRIDE_COM_FIELDS header in that order
    - every row has known stride identity/bounds metadata matching
      reviewed_strides
    - sample_kind in {original, normalized}
    - method in {exact, linear, none}
    - usable is lowercase true/false
    - finite/nullable fields validated per schema
    - exactly one ordered original row for every stride frame
      (start_frame .. end_frame inclusive) in frame_index order
    - exactly normalized_n ordered normalized rows with indices 0..N-1,
      canonical progression = i * 100/(N-1), canonical target timestamp
    - stride_id groups are contiguous in file order matching
      reviewed_strides order
    """
    if normalized_n < 2:
        raise ComQualificationArtifactValidationError("normalized_n must be >= 2")

    # Build lookup for reviewed stride metadata
    stride_meta: dict[str, ParsedReviewedStride] = {
        s.stride_id: s for s in reviewed_strides
    }
    reviewed_stride_ids = [s.stride_id for s in reviewed_strides]

    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != STRIDE_COM_FIELDS:
                raise ComQualificationArtifactValidationError(
                    "stride_com.csv header must exactly match STRIDE_COM_FIELDS"
                )

            # Group rows by stride_id preserving file order
            stride_rows: dict[str, list[dict[str, str]]] = {}
            current_stride_id: str | None = None
            seen_stride_ids_in_file: list[str] = []

            for row_number, row in enumerate(reader, start=2):
                if None in row or any(
                    row[field] is None for field in STRIDE_COM_FIELDS
                ):
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv row {row_number}: malformed columns"
                    )

                stride_id = row["stride_id"]
                if stride_id not in stride_meta:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv row {row_number}: "
                        f"stride_id {stride_id!r} not found in reviewed_strides"
                    )

                # Track stride_id order in file
                if stride_id != current_stride_id:
                    if stride_id in seen_stride_ids_in_file:
                        raise ComQualificationArtifactValidationError(
                            f"stride_com.csv row {row_number}: "
                            f"stride_id {stride_id!r} appears non-contiguously"
                        )
                    seen_stride_ids_in_file.append(stride_id)
                    current_stride_id = stride_id
                    stride_rows[stride_id] = []

                stride_rows[stride_id].append(dict(row))

        # Verify file stride order matches reviewed_strides order
        if seen_stride_ids_in_file != reviewed_stride_ids:
            raise ComQualificationArtifactValidationError(
                "stride_com.csv stride_id order does not match reviewed_strides order"
            )

        # Parse and validate each stride's rows
        result: dict[str, tuple[StrideComSample, ...]] = {}

        for stride in reviewed_strides:
            sid = stride.stride_id
            rows = stride_rows.get(sid, [])
            if not rows:
                raise ComQualificationArtifactValidationError(
                    f"stride_com.csv has no rows for stride {sid!r}"
                )

            meta = stride_meta[sid]
            frame_count = meta.end_frame - meta.start_frame + 1
            expected_original_count = frame_count
            expected_normalized_count = normalized_n

            original_samples: list[StrideComSample] = []
            normalized_samples: list[StrideComSample] = []

            for row_idx, row in enumerate(rows):
                # Validate stride identity/bounds metadata matches reviewed stride
                if row["stride_id"] != meta.stride_id:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: stride_id mismatch"
                    )
                if row["side"] != meta.side:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: side mismatch"
                    )
                if row["start_event_id"] != meta.start_event_id:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"start_event_id mismatch"
                    )
                if row["end_event_id"] != meta.end_event_id:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"end_event_id mismatch"
                    )
                if _parse_int(row["start_frame"], "start_frame") != meta.start_frame:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"start_frame mismatch"
                    )
                if _parse_int(row["end_frame"], "end_frame") != meta.end_frame:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: end_frame mismatch"
                    )
                if (
                    abs(
                        _parse_float(
                            row["start_timestamp_seconds"],
                            "start_timestamp_seconds",
                            nullable=False,
                        )
                        - meta.start_timestamp_seconds
                    )
                    > 1e-9
                ):
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"start_timestamp_seconds mismatch"
                    )
                if (
                    abs(
                        _parse_float(
                            row["end_timestamp_seconds"],
                            "end_timestamp_seconds",
                            nullable=False,
                        )
                        - meta.end_timestamp_seconds
                    )
                    > 1e-9
                ):
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"end_timestamp_seconds mismatch"
                    )
                if (
                    abs(
                        _parse_float(
                            row["duration_seconds"],
                            "duration_seconds",
                            nullable=False,
                        )
                        - meta.duration_seconds
                    )
                    > 1e-9
                ):
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"duration_seconds mismatch"
                    )
                if row["quality"] != meta.quality:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: quality mismatch"
                    )
                if (row["contralateral_event_id"] or None) != (
                    meta.contralateral_event_id
                ):
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"contralateral_event_id mismatch"
                    )
                if (
                    _parse_int(
                        row["contralateral_event_count"], "contralateral_event_count"
                    )
                    != meta.contralateral_event_count
                ):
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"contralateral_event_count mismatch"
                    )
                if row["source"] != meta.source:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: source mismatch"
                    )
                if row["review_status"] != meta.review_status:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"review_status mismatch"
                    )
                if row["automatic_stride_id"] != meta.automatic_stride_id:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"automatic_stride_id mismatch"
                    )
                if row["review_intent"] != meta.review_intent:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"review_intent mismatch"
                    )
                if row["review_changes"] != meta.review_changes:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"review_changes mismatch"
                    )
                if row["provenance_notes"] != meta.provenance_notes:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"provenance_notes mismatch"
                    )

                # Parse sample fields
                sample_kind = row["sample_kind"]
                if sample_kind not in {"original", "normalized"}:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"sample_kind must be 'original' or 'normalized', "
                        f"got {sample_kind!r}"
                    )

                normalized_index_text = row["normalized_index"]
                normalized_index: int | None = None
                if sample_kind == "normalized":
                    if normalized_index_text == "":
                        raise ComQualificationArtifactValidationError(
                            f"stride_com.csv stride {sid} row {row_idx}: "
                            f"normalized_index required for normalized samples"
                        )
                    normalized_index = _parse_int(
                        normalized_index_text, "normalized_index"
                    )
                    if normalized_index < 0:
                        raise ComQualificationArtifactValidationError(
                            f"stride_com.csv stride {sid} row {row_idx}: "
                            f"normalized_index must be >= 0"
                        )
                else:
                    if normalized_index_text != "":
                        raise ComQualificationArtifactValidationError(
                            f"stride_com.csv stride {sid} row {row_idx}: "
                            f"normalized_index must be blank for original samples"
                        )

                progression = _parse_float(
                    row["progression"], "progression", nullable=False
                )
                if progression is None:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"progression must be finite"
                    )
                if not (0.0 <= progression <= 100.0):
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"progression must be in [0, 100]"
                    )

                method = row["method"]
                if method not in {"exact", "linear", "none"}:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"method must be 'exact', 'linear', or 'none', got {method!r}"
                    )

                # Parse source frame/timestamp fields (nullable)
                source_frame_index_text = row["source_frame_index"]
                source_frame_index = (
                    _parse_int(source_frame_index_text, "source_frame_index")
                    if source_frame_index_text != ""
                    else None
                )

                source_timestamp_seconds_text = row["source_timestamp_seconds"]
                source_timestamp_seconds = (
                    _parse_float(
                        source_timestamp_seconds_text,
                        "source_timestamp_seconds",
                        nullable=True,
                    )
                    if source_timestamp_seconds_text != ""
                    else None
                )

                target_timestamp_seconds_text = row["target_timestamp_seconds"]
                target_timestamp_seconds = (
                    _parse_float(
                        target_timestamp_seconds_text,
                        "target_timestamp_seconds",
                        nullable=True,
                    )
                    if target_timestamp_seconds_text != ""
                    else None
                )

                left_source_frame_index_text = row["left_source_frame_index"]
                left_source_frame_index = (
                    _parse_int(left_source_frame_index_text, "left_source_frame_index")
                    if left_source_frame_index_text != ""
                    else None
                )

                left_src_ts_text = row["left_source_timestamp_seconds"]
                left_source_timestamp_seconds = (
                    _parse_float(
                        left_src_ts_text,
                        "left_source_timestamp_seconds",
                        nullable=True,
                    )
                    if left_src_ts_text != ""
                    else None
                )

                right_source_frame_index_text = row["right_source_frame_index"]
                right_source_frame_index = (
                    _parse_int(
                        right_source_frame_index_text, "right_source_frame_index"
                    )
                    if right_source_frame_index_text != ""
                    else None
                )

                right_src_ts_text = row["right_source_timestamp_seconds"]
                right_source_timestamp_seconds = (
                    _parse_float(
                        right_src_ts_text,
                        "right_source_timestamp_seconds",
                        nullable=True,
                    )
                    if right_src_ts_text != ""
                    else None
                )

                # Parse COM fields
                com_x_text = row["com_x"]
                com_y_text = row["com_y"]
                com_x = (
                    _parse_float(com_x_text, "com_x", nullable=True)
                    if com_x_text != ""
                    else None
                )
                com_y = (
                    _parse_float(com_y_text, "com_y", nullable=True)
                    if com_y_text != ""
                    else None
                )
                if (com_x is None) != (com_y is None):
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"com_x and com_y must be both finite or both blank"
                    )
                if com_x is not None:
                    assert com_y is not None
                    com = Point2D(com_x, com_y)
                else:
                    com = None

                # Parse usable (lowercase true/false only)
                usable = _parse_bool_lower(row["usable"], "usable")

                mass_coverage = _parse_float(
                    row["mass_coverage"], "mass_coverage", nullable=False
                )
                if mass_coverage is None or mass_coverage < 0.0:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"mass_coverage must be finite >= 0"
                    )

                min_endpoint_coverage = _parse_float(
                    row["min_endpoint_coverage"],
                    "min_endpoint_coverage",
                    nullable=False,
                )
                if min_endpoint_coverage is None or min_endpoint_coverage < 0.0:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"min_endpoint_coverage must be finite >= 0"
                    )

                contributors = _parse_tuple_pipe(row["contributors"], "contributors")
                qc_flags = _parse_tuple_pipe(row["qc_flags"], "qc_flags")

                # Validate usable consistency
                # Step 5a intentionally retains finite below-threshold COM
                # while marking unusable; only reject usable=true with missing COM.
                if usable and com is None:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} row {row_idx}: "
                        f"usable is true but com is None"
                    )

                # Build StrideComSample
                sample = StrideComSample(
                    progression=progression,
                    com=com,
                    mass_coverage=mass_coverage,
                    usable=usable,
                    sample_kind=sample_kind,  # type: ignore[arg-type]
                    normalized_index=normalized_index,
                    method=method,  # type: ignore[arg-type]
                    source_frame_index=source_frame_index,
                    source_timestamp_seconds=source_timestamp_seconds,
                    target_timestamp_seconds=target_timestamp_seconds,
                    left_source_frame_index=left_source_frame_index,
                    left_source_timestamp_seconds=left_source_timestamp_seconds,
                    right_source_frame_index=right_source_frame_index,
                    right_source_timestamp_seconds=right_source_timestamp_seconds,
                    min_endpoint_coverage=min_endpoint_coverage,
                    contributors=contributors,
                    qc_flags=qc_flags,
                )

                if sample_kind == "original":
                    original_samples.append(sample)
                else:
                    normalized_samples.append(sample)

            # Validate original samples: exactly one per frame in order
            if len(original_samples) != expected_original_count:
                raise ComQualificationArtifactValidationError(
                    f"stride_com.csv stride {sid}: "
                    f"expected {expected_original_count} "
                    f"original samples (one per frame "
                    f"{meta.start_frame}..{meta.end_frame}), "
                    f"got {len(original_samples)}"
                )

            # Check they are ordered by frame index and match expected frames
            for i, sample in enumerate(original_samples):
                expected_frame = meta.start_frame + i
                if sample.source_frame_index != expected_frame:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} original sample {i}: "
                        f"source_frame_index {sample.source_frame_index} "
                        f"does not match expected {expected_frame}"
                    )
                # Check progression matches frame timestamp
                expected_ts = pose_frames[expected_frame][0]
                if sample.source_timestamp_seconds is None:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} original sample {i}: "
                        f"source_timestamp_seconds must be present"
                    )
                if abs(sample.source_timestamp_seconds - expected_ts) > 1e-9:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} original sample {i}: "
                        f"source_timestamp_seconds {sample.source_timestamp_seconds} "
                        f"does not match pose_frames {expected_ts}"
                    )
                if sample.normalized_index is not None:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} original sample {i}: "
                        f"normalized_index must be None for original samples"
                    )
                if sample.method != "exact":
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} original sample {i}: "
                        f"method must be 'exact' for original samples"
                    )
                if sample.target_timestamp_seconds is None:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} original sample {i}: "
                        f"target_timestamp_seconds must be present"
                    )
                if (
                    abs(
                        sample.target_timestamp_seconds
                        - sample.source_timestamp_seconds
                    )
                    > 1e-9
                ):
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} original sample {i}: "
                        f"target_timestamp_seconds must equal source_timestamp_seconds"
                    )

            # Validate normalized samples: exactly N ordered indices 0..N-1
            if len(normalized_samples) != expected_normalized_count:
                raise ComQualificationArtifactValidationError(
                    f"stride_com.csv stride {sid}: "
                    f"expected {expected_normalized_count} "
                    f"normalized samples, got {len(normalized_samples)}"
                )

            for i, sample in enumerate(normalized_samples):
                if sample.normalized_index != i:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} normalized sample {i}: "
                        f"normalized_index {sample.normalized_index} must be {i}"
                    )
                # Canonical progression
                expected_progression = i * 100.0 / (normalized_n - 1)
                if abs(sample.progression - expected_progression) > 1e-9:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} normalized sample {i}: "
                        f"progression {sample.progression} must be "
                        f"{expected_progression} (canonical)"
                    )
                # Canonical target timestamp
                expected_target_ts = (
                    meta.start_timestamp_seconds
                    + (expected_progression / 100.0) * meta.duration_seconds
                )
                if sample.target_timestamp_seconds is None:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} normalized sample {i}: "
                        f"target_timestamp_seconds must be present"
                    )
                if abs(sample.target_timestamp_seconds - expected_target_ts) > 1e-9:
                    raise ComQualificationArtifactValidationError(
                        f"stride_com.csv stride {sid} normalized sample {i}: "
                        f"target_timestamp_seconds {sample.target_timestamp_seconds} "
                        f"does not match canonical {expected_target_ts}"
                    )
                if sample.method == "exact":
                    if sample.source_frame_index is None:
                        raise ComQualificationArtifactValidationError(
                            f"stride_com.csv stride {sid} normalized sample {i}: "
                            f"method='exact' requires source_frame_index"
                        )
                    if sample.source_timestamp_seconds is None:
                        raise ComQualificationArtifactValidationError(
                            f"stride_com.csv stride {sid} normalized sample {i}: "
                            f"method='exact' requires source_timestamp_seconds"
                        )
                    if abs(sample.source_timestamp_seconds - expected_target_ts) > 1e-9:
                        raise ComQualificationArtifactValidationError(
                            f"stride_com.csv stride {sid} normalized sample {i}: "
                            f"method='exact' source_timestamp must match target"
                        )
                elif sample.method == "linear":
                    if (
                        sample.left_source_frame_index is None
                        or sample.right_source_frame_index is None
                    ):
                        raise ComQualificationArtifactValidationError(
                            f"stride_com.csv stride {sid} normalized sample {i}: "
                            f"method='linear' requires left/right source frame indices"
                        )
                    if (
                        sample.left_source_timestamp_seconds is None
                        or sample.right_source_timestamp_seconds is None
                    ):
                        raise ComQualificationArtifactValidationError(
                            f"stride_com.csv stride {sid} normalized sample {i}: "
                            f"method='linear' requires left/right source timestamps"
                        )
                    # Verify bracket contains target
                    if not (
                        sample.left_source_timestamp_seconds
                        < sample.target_timestamp_seconds
                        < sample.right_source_timestamp_seconds
                    ):
                        raise ComQualificationArtifactValidationError(
                            f"stride_com.csv stride {sid} normalized sample {i}: "
                            f"method='linear' target not strictly between "
                            f"left/right timestamps"
                        )
                elif sample.method == "none":
                    if sample.source_frame_index is not None:
                        raise ComQualificationArtifactValidationError(
                            f"stride_com.csv stride {sid} normalized sample {i}: "
                            f"method='none' requires source_frame_index to be blank"
                        )
                    if sample.source_timestamp_seconds is not None:
                        raise ComQualificationArtifactValidationError(
                            f"stride_com.csv stride {sid} normalized sample {i}: "
                            f"method='none' requires source_timestamp_seconds "
                            f"to be blank"
                        )
                    if sample.com is not None:
                        raise ComQualificationArtifactValidationError(
                            f"stride_com.csv stride {sid} normalized sample {i}: "
                            f"method='none' requires COM to be blank"
                        )
                    if sample.usable:
                        raise ComQualificationArtifactValidationError(
                            f"stride_com.csv stride {sid} normalized sample {i}: "
                            f"method='none' requires usable=false"
                        )
                    bracket = (
                        sample.left_source_frame_index,
                        sample.left_source_timestamp_seconds,
                        sample.right_source_frame_index,
                        sample.right_source_timestamp_seconds,
                    )
                    if any(value is None for value in bracket):
                        if not all(value is None for value in bracket):
                            raise ComQualificationArtifactValidationError(
                                f"stride_com.csv stride {sid} normalized sample {i}: "
                                f"method='none' requires all four left/right bracket "
                                f"fields or none"
                            )
                    else:
                        left_frame = sample.left_source_frame_index
                        left_timestamp = sample.left_source_timestamp_seconds
                        right_frame = sample.right_source_frame_index
                        right_timestamp = sample.right_source_timestamp_seconds
                        assert left_frame is not None
                        assert left_timestamp is not None
                        assert right_frame is not None
                        assert right_timestamp is not None
                        if right_frame != left_frame + 1:
                            raise ComQualificationArtifactValidationError(
                                f"stride_com.csv stride {sid} normalized sample {i}: "
                                f"method='none' bracket frame indices must be "
                                f"consecutive"
                            )
                        if not (
                            left_timestamp
                            < sample.target_timestamp_seconds
                            < right_timestamp
                        ):
                            raise ComQualificationArtifactValidationError(
                                f"stride_com.csv stride {sid} normalized sample {i}: "
                                f"method='none' target not strictly between "
                                f"left/right timestamps"
                            )

            # Combine: original first (by frame order), then normalized (by index order)
            result[sid] = tuple(original_samples + normalized_samples)

        return result

    except (OSError, UnicodeError, csv.Error) as exc:
        raise ComQualificationArtifactValidationError(
            f"Could not read {path}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Internal: strict reader for Step 3 processed landmarks
# ---------------------------------------------------------------------------


def _read_processed_landmarks(
    path: Path,
    pose_frames: dict[int, tuple[float, str]],
) -> tuple[tuple[ProcessedLandmarkQCRow, ...], dict[int, dict[str, Point2D]]]:
    """Read and validate processed_landmarks.csv strictly.

    Enforces:
    - exact PROCESSED_FIELDS header in that order
    - canonical complete ordered frame x 33 grid using MEDIAPIPE_LANDMARK_NAMES
    - frame timestamps/status match pose_frames
    - exact lowercase booleans for raw/observed/interpolation/
      smoothing/final missing fields
    - processed x/y finite-or-blank and final_available iff both
      present and final_missing false
    - raw_observed_usable from observed_usable

    Returns:
        (required_qc_rows, points_by_frame) where:
        - required_qc_rows: tuple of ProcessedLandmarkQCRow for
          landmarks in build_segment_dependency_map keys (required
          underlying landmarks), ordered by frame_index then landmark
          order in MEDIAPIPE_LANDMARK_NAMES
        - points_by_frame: dict mapping frame_index ->
          {landmark_name: Point2D} only for landmarks where
          final_available is True

    Raises:
        ComQualificationArtifactValidationError: If CSV structure or
            content is invalid.
    """
    dep_map = build_segment_dependency_map()
    required_landmark_names = set(dep_map.keys())

    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != PROCESSED_FIELDS:
                raise ComQualificationArtifactValidationError(
                    "processed_landmarks.csv header must exactly match PROCESSED_FIELDS"
                )

            # Build expected grid: frame_index 0..N-1 x 33 landmarks
            # in MEDIAPIPE order
            expected_frame_count = len(pose_frames)
            if expected_frame_count == 0:
                raise ComQualificationArtifactValidationError(
                    "pose_frames is empty, cannot validate processed_landmarks grid"
                )

            required_rows: list[ProcessedLandmarkQCRow] = []
            points_by_frame: dict[int, dict[str, Point2D]] = {}

            prev_frame_index = -1
            prev_landmark_id = -1
            row_number = 1  # header is row 1

            for row in reader:
                row_number += 1
                if None in row or any(row[field] is None for field in PROCESSED_FIELDS):
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: malformed columns"
                    )

                # Parse base fields
                frame_index = _parse_int(row["frame_index"], "frame_index")
                if frame_index < 0 or frame_index >= expected_frame_count:
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        f"frame_index {frame_index} out of range "
                        f"0..{expected_frame_count - 1}"
                    )

                # Validate canonical grid ordering: frames then landmarks
                landmark_id = _parse_int(row["landmark_id"], "landmark_id")
                if not 0 <= landmark_id < len(MEDIAPIPE_LANDMARK_NAMES):
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        f"landmark_id {landmark_id} not in canonical "
                        "range 0..32"
                    )
                expected_name = MEDIAPIPE_LANDMARK_NAMES[landmark_id]
                if row["landmark_name"] != expected_name:
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        f"landmark_name {row['landmark_name']!r} does not "
                        f"match MEDIAPIPE_LANDMARK_NAMES[{landmark_id}] "
                        f"= {expected_name!r}"
                    )

                # Check grid ordering
                if frame_index < prev_frame_index or (
                    frame_index == prev_frame_index and landmark_id <= prev_landmark_id
                ):
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        "rows must be ordered by frame_index then "
                        f"landmark_id (got frame {frame_index}, "
                        f"landmark {landmark_id} after frame "
                        f"{prev_frame_index}, landmark {prev_landmark_id})"
                    )
                prev_frame_index = frame_index
                prev_landmark_id = landmark_id

                # Validate timestamp and status match pose_frames
                timestamp_seconds = _parse_float(
                    row["nominal_timestamp_seconds"],
                    "nominal_timestamp_seconds",
                    nullable=False,
                )
                if timestamp_seconds is None:
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        "timestamp must be finite"
                    )
                frame_status = row["frame_status"]
                if not frame_status:
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        "frame_status must be nonempty"
                    )

                if frame_index not in pose_frames:
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        f"frame_index {frame_index} not in pose_frames"
                    )
                expected_ts, expected_status = pose_frames[frame_index]
                if abs(timestamp_seconds - expected_ts) > 1e-9:
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        f"timestamp {timestamp_seconds} does not match "
                        f"pose_frames timestamp {expected_ts} for frame "
                        f"{frame_index}"
                    )
                if frame_status != expected_status:
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        f"frame_status {frame_status!r} does not match "
                        f"pose_frames status {expected_status!r} for "
                        f"frame {frame_index}"
                    )

                # Parse boolean fields (exact lowercase 'true'/'false' only)
                _ = _parse_bool_lower(row["raw_row_present"], "raw_row_present")
                x_observed_usable = _parse_bool_lower(
                    row["x_observed_usable"], "x_observed_usable"
                )
                y_observed_usable = _parse_bool_lower(
                    row["y_observed_usable"], "y_observed_usable"
                )
                observed_usable = _parse_bool_lower(
                    row["observed_usable"], "observed_usable"
                )
                _ = _parse_bool_lower(
                    row["rejected_low_confidence"], "rejected_low_confidence"
                )
                _ = _parse_bool_lower(
                    row["missing_or_nonfinite_enabled_score"],
                    "missing_or_nonfinite_enabled_score",
                )
                _ = _parse_bool_lower(
                    row["nonfinite_x_coordinate"], "nonfinite_x_coordinate"
                )
                _ = _parse_bool_lower(
                    row["nonfinite_y_coordinate"], "nonfinite_y_coordinate"
                )
                _ = _parse_bool_lower(row["out_of_image_x"], "out_of_image_x")
                _ = _parse_bool_lower(row["out_of_image_y"], "out_of_image_y")
                x_interpolated = _parse_bool_lower(
                    row["x_interpolated"], "x_interpolated"
                )
                y_interpolated = _parse_bool_lower(
                    row["y_interpolated"], "y_interpolated"
                )
                x_smoothing_changed = _parse_bool_lower(
                    row["x_smoothing_changed"], "x_smoothing_changed"
                )
                y_smoothing_changed = _parse_bool_lower(
                    row["y_smoothing_changed"], "y_smoothing_changed"
                )
                x_smoothing_support_contains_interpolation = _parse_bool_lower(
                    row["x_smoothing_support_contains_interpolation"],
                    "x_smoothing_support_contains_interpolation",
                )
                y_smoothing_support_contains_interpolation = _parse_bool_lower(
                    row["y_smoothing_support_contains_interpolation"],
                    "y_smoothing_support_contains_interpolation",
                )
                _ = _parse_bool_lower(row["x_final_missing"], "x_final_missing")
                _ = _parse_bool_lower(row["y_final_missing"], "y_final_missing")
                final_missing = _parse_bool_lower(row["final_missing"], "final_missing")

                # Validate observed_usable ==
                # (x_observed_usable and y_observed_usable)
                if observed_usable != (x_observed_usable and y_observed_usable):
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        f"observed_usable ({observed_usable}) must equal "
                        f"x_observed_usable ({x_observed_usable}) and "
                        f"y_observed_usable ({y_observed_usable})"
                    )

                # Parse processed coordinates (nullable, blank -> None)
                processed_x_text = row["processed_x_normalized"]
                processed_y_text = row["processed_y_normalized"]
                processed_x = (
                    _parse_float(
                        processed_x_text, "processed_x_normalized", nullable=True
                    )
                    if processed_x_text != ""
                    else None
                )
                processed_y = (
                    _parse_float(
                        processed_y_text, "processed_y_normalized", nullable=True
                    )
                    if processed_y_text != ""
                    else None
                )
                if (processed_x is None) != (processed_y is None):
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        "processed_x_normalized and processed_y_normalized "
                        "must be both finite or both blank"
                    )
                if processed_x is not None and not math.isfinite(processed_x):
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        "processed_x_normalized must be finite"
                    )
                if processed_y is not None and not math.isfinite(processed_y):
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        "processed_y_normalized must be finite"
                    )

                # Validate final_available logic
                final_available = (
                    processed_x is not None
                    and processed_y is not None
                    and not final_missing
                )
                if final_missing and final_available:
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        "final_missing is true but final_available would "
                        "be true"
                    )
                if final_available and final_missing:
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        "final_available true but final_missing is true"
                    )
                if not final_missing and processed_x is None:
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        "final_missing is false but processed coordinates "
                        "are blank"
                    )
                if not final_missing and processed_y is None:
                    raise ComQualificationArtifactValidationError(
                        f"processed_landmarks.csv row {row_number}: "
                        "final_missing is false but processed coordinates "
                        "are blank"
                    )

                # raw_observed_usable comes from observed_usable
                raw_observed_usable = observed_usable

                # Build QC row for required landmarks only
                if expected_name in required_landmark_names:
                    qc_row = ProcessedLandmarkQCRow(
                        frame_index=frame_index,
                        landmark_name=expected_name,
                        final_available=final_available,
                        raw_observed_usable=raw_observed_usable,
                        x_interpolated=x_interpolated,
                        y_interpolated=y_interpolated,
                        x_smoothing_changed=x_smoothing_changed,
                        y_smoothing_changed=y_smoothing_changed,
                        x_smoothing_support_interpolation=(
                            x_smoothing_support_contains_interpolation
                        ),
                        y_smoothing_support_interpolation=(
                            y_smoothing_support_contains_interpolation
                        ),
                    )
                    required_rows.append(qc_row)

                # Build points_by_frame for rendering (only when final_available)
                if (
                    final_available
                    and processed_x is not None
                    and processed_y is not None
                ):
                    if frame_index not in points_by_frame:
                        points_by_frame[frame_index] = {}
                    points_by_frame[frame_index][expected_name] = Point2D(
                        processed_x, processed_y
                    )

            # Validate complete grid: exactly expected_frame_count * 33 rows
            expected_total_rows = expected_frame_count * len(MEDIAPIPE_LANDMARK_NAMES)
            actual_total_rows = row_number - 1
            if actual_total_rows != expected_total_rows:
                raise ComQualificationArtifactValidationError(
                    f"processed_landmarks.csv must have exactly "
                    f"{expected_total_rows} rows "
                    f"({expected_frame_count} frames x 33 landmarks), "
                    f"got {actual_total_rows}"
                )

            # Validate every frame/landmark combination was seen
            seen_combinations: set[tuple[int, int]] = set()
            for qc_row in required_rows:
                landmark_id = MEDIAPIPE_LANDMARK_NAMES.index(qc_row.landmark_name)
                seen_combinations.add((qc_row.frame_index, landmark_id))

            # Check all required combinations for required landmarks
            for fi in range(expected_frame_count):
                for landmark_id, landmark_name in enumerate(MEDIAPIPE_LANDMARK_NAMES):
                    if landmark_name in required_landmark_names:
                        if (fi, landmark_id) not in seen_combinations:
                            raise ComQualificationArtifactValidationError(
                                f"processed_landmarks.csv missing row for "
                                f"frame {fi}, landmark {landmark_name} "
                                f"(id {landmark_id})"
                            )

        # Sort required_rows by frame_index then landmark order
        def _sort_key(r: ProcessedLandmarkQCRow) -> tuple[int, int]:
            return (r.frame_index, MEDIAPIPE_LANDMARK_NAMES.index(r.landmark_name))

        required_rows.sort(key=_sort_key)

        return tuple(required_rows), points_by_frame

    except (OSError, UnicodeError, csv.Error) as exc:
        raise ComQualificationArtifactValidationError(
            f"Could not read {path}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Internal: annotated COM video rendering
# ---------------------------------------------------------------------------


def _write_annotated_com_video(
    output_path: Path,
    source_info: _SourceInfo,
    frame_results: tuple[FrameComResult, ...],
    points_by_frame: dict[int, dict[str, Point2D]],
    reviewed_strides: list[ParsedReviewedStride],
    primary_threshold: float,
    theoretical_supported: float,
) -> None:
    """Render annotated COM qualification video.

    Args:
        output_path: Path to write the annotated MP4 video.
        source_info: Validated source video metadata (width, height, FPS, etc.).
        frame_results: Tuple of FrameComResult ordered by frame_index 0..N-1.
        points_by_frame: Mapping frame_index -> {landmark_name: Point2D} for
            processed landmarks with final_available=True (normalized coords).
        reviewed_strides: Parsed reviewed strides for stride ID overlay.
        primary_threshold: Minimum mass coverage threshold for frame usability.
        theoretical_supported: Theoretical maximum supported mass fraction
            (theoretical_supported_mass_fraction for the sex).

    Raises:
        ComQualificationRenderError: On any open/decode/write/frame-count failure.
    """
    from gait_stability.pose_contracts import PoseFrameStatus

    if not isinstance(primary_threshold, (int, float)) or isinstance(
        primary_threshold, bool
    ):
        raise ComQualificationRenderError("primary_threshold must be a number")
    if not math.isfinite(primary_threshold) or not (0.0 <= primary_threshold <= 1.0):
        raise ComQualificationRenderError("primary_threshold must be finite in [0, 1]")
    if not isinstance(theoretical_supported, (int, float)) or isinstance(
        theoretical_supported, bool
    ):
        raise ComQualificationRenderError("theoretical_supported must be a number")
    if not math.isfinite(theoretical_supported) or not (
        0.0 <= theoretical_supported <= 1.0
    ):
        raise ComQualificationRenderError(
            "theoretical_supported must be finite in [0, 1]"
        )

    # Build stride frame lookup for overlay
    stride_frame_to_ids: dict[int, list[str]] = {}
    for stride in reviewed_strides:
        for fi in range(stride.start_frame, stride.end_frame + 1):
            stride_frame_to_ids.setdefault(fi, []).append(stride.stride_id)

    cap: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None

    try:
        # Open source video
        cap = cv2.VideoCapture(str(source_info.path))
        if not cap.isOpened():
            raise ComQualificationRenderError(
                f"Failed to open source video: {source_info.path}"
            )

        # Verify width/height/FPS against source_info
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        if actual_width != source_info.width_pixels:
            raise ComQualificationRenderError(
                f"Video width mismatch: source_info={source_info.width_pixels}, "
                f"decoded={actual_width}"
            )
        if actual_height != source_info.height_pixels:
            raise ComQualificationRenderError(
                f"Video height mismatch: source_info={source_info.height_pixels}, "
                f"decoded={actual_height}"
            )
        # FPS tolerance: allow small floating-point differences (1e-3 relative)
        fps_tol = max(1e-3, 1e-3 * source_info.nominal_fps)
        if abs(actual_fps - source_info.nominal_fps) > fps_tol:
            raise ComQualificationRenderError(
                f"Video FPS mismatch: source_info={source_info.nominal_fps}, "
                f"decoded={actual_fps}"
            )

        # Prepare video writer (MP4V codec)
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            source_info.nominal_fps,
            (source_info.width_pixels, source_info.height_pixels),
        )
        if not writer.isOpened():
            raise ComQualificationRenderError(
                f"Failed to open video writer: {output_path}"
            )

        # Precompute landmark name to index for CANONICAL_POSE_CONNECTIONS
        # Connections use integer indices into MEDIAPIPE_LANDMARK_NAMES
        # points_by_frame uses landmark names as keys

        # Rendering constants
        skeleton_color = (255, 255, 255)  # white
        skeleton_thickness = 2
        com_radius = 6
        text_color = (255, 255, 255)
        text_bg_color = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 1
        line_height = 20
        margin = 10

        # Supported segments (all except unsupported)
        supported_segments = tuple(
            s for s in SEGMENT_NAMES if s not in UNSUPPORTED_SEGMENTS
        )

        for frame_index, fr in enumerate(frame_results):
            ret, frame_bgr = cap.read()
            source_frame_unavailable = not ret or frame_bgr is None
            if source_frame_unavailable:
                if fr.frame_status != PoseFrameStatus.DECODE_FAILURE.value:
                    raise ComQualificationRenderError(
                        f"Failed to decode source frame {frame_index}, but "
                        f"frame status is {fr.frame_status!r}"
                    )
                frame_bgr = __import__("numpy").zeros(
                    (source_info.height_pixels, source_info.width_pixels, 3),
                    dtype="uint8",
                )
            pts_norm = points_by_frame.get(frame_index, {})

            # Draw pose skeleton using CANONICAL_POSE_CONNECTIONS
            # Convert normalized points to pixel coordinates (no clamping)
            pts_pixel: dict[int, tuple[int, int]] = {}
            for lm_idx, lm_name in enumerate(MEDIAPIPE_LANDMARK_NAMES):
                pt = pts_norm.get(lm_name)
                if pt is not None:
                    px = round(pt.x * (source_info.width_pixels - 1))
                    py = round(pt.y * (source_info.height_pixels - 1))
                    pts_pixel[lm_idx] = (px, py)

            for start_idx, end_idx in CANONICAL_POSE_CONNECTIONS:
                p1 = pts_pixel.get(start_idx)
                p2 = pts_pixel.get(end_idx)
                if p1 is not None and p2 is not None:
                    cv2.line(
                        frame_bgr,
                        p1,
                        p2,
                        skeleton_color,
                        skeleton_thickness,
                        cv2.LINE_AA,
                    )

            # Draw COM marker if finite
            if fr.com is not None:
                com_px = round(fr.com.x * (source_info.width_pixels - 1))
                com_py = round(fr.com.y * (source_info.height_pixels - 1))
                # Color: green if usable (frame_eligible_at_threshold primary),
                # else orange/red
                if fr.usable:
                    com_color = (0, 255, 0)  # green
                else:
                    # orange if mass_coverage > 0 but below threshold,
                    # red if zero coverage
                    if fr.mass_coverage > 0.0:
                        com_color = (0, 165, 255)  # orange (BGR)
                    else:
                        com_color = (0, 0, 255)  # red
                cv2.circle(
                    frame_bgr, (com_px, com_py), com_radius, com_color, -1, cv2.LINE_AA
                )
                cv2.circle(
                    frame_bgr,
                    (com_px, com_py),
                    com_radius + 2,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Build overlay text lines
            lines: list[str] = []

            if source_frame_unavailable:
                lines.append("SOURCE FRAME UNAVAILABLE")

            # Frame/index
            lines.append(f"Frame: {frame_index}  Index: {frame_index}")

            # Represented-segment COM proxy label
            lines.append("Represented-segment COM proxy")

            # Primary coverage gate
            qc_status = "ELIGIBLE" if fr.usable else "INELIGIBLE"
            lines.append(
                f"Primary coverage gate @ {primary_threshold:.3f}: {qc_status}"
            )

            # Absolute mass_coverage (total-model mass fraction, rounded)
            lines.append(f"Mass coverage (total-model): {fr.mass_coverage:.4f}")

            # Supported mass coverage (supported-model completeness)
            if theoretical_supported > 0.0:
                supported_cov = fr.mass_coverage / theoretical_supported
            else:
                supported_cov = 0.0
            lines.append(f"Supported mass coverage: {supported_cov:.4f}")

            # Absent SUPPORTED segments only
            absent_supported = tuple(
                s for s in supported_segments if s not in fr.usable_segments()
            )
            if absent_supported:
                lines.append(
                    f"Absent supported segments: {', '.join(absent_supported)}"
                )
            else:
                lines.append("Absent supported segments: (none)")

            # Reviewed stride IDs containing this frame
            stride_ids = stride_frame_to_ids.get(frame_index, [])
            if stride_ids:
                lines.append(f"Reviewed strides: {', '.join(stride_ids)}")
            else:
                lines.append("Reviewed strides: (none)")

            # Explicit disclaimer
            lines.append("Research-only proxy; NOT validated physical COM")

            # Draw text overlay with background
            y = margin
            for line in lines:
                (text_w, text_h), baseline = cv2.getTextSize(
                    line, font, font_scale, font_thickness
                )
                # Draw background rectangle
                cv2.rectangle(
                    frame_bgr,
                    (margin - 3, y - text_h - 2),
                    (margin + text_w + 3, y + baseline + 2),
                    text_bg_color,
                    -1,
                )
                # Draw text
                cv2.putText(
                    frame_bgr,
                    line,
                    (margin, y),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA,
                )
                y += line_height

            # Write frame
            success = writer.write(frame_bgr)
            if not success:
                raise ComQualificationRenderError(
                    f"Failed to write frame {frame_index}"
                )

        extra_ret, _extra_frame = cap.read()
        if extra_ret:
            raise ComQualificationRenderError(
                "Source video contains an extra decoded frame beyond "
                f"frame_results length ({len(frame_results)})"
            )

    except ComQualificationRenderError:
        raise
    except Exception as exc:
        raise ComQualificationRenderError(
            f"Unexpected error during video rendering: {exc}"
        ) from exc
    finally:
        # Ensure resources are released
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


# ---------------------------------------------------------------------------
# Public: COM stride QC CSV field names (exact order)
# ---------------------------------------------------------------------------

COM_STRIDE_QC_FIELDS: tuple[str, ...] = (
    # Stride identity / bounds
    "stride_id",
    "side",
    "start_event_id",
    "end_event_id",
    "start_frame",
    "end_frame",
    "start_timestamp_seconds",
    "end_timestamp_seconds",
    "duration_seconds",
    "quality",
    "automatic_stride_id",
    "review_intent",
    "review_changes",
    "provenance_notes",
    # Frame counts at primary threshold
    "frame_count",
    "primary_threshold",
    # Finite COM
    "finite_com_frames",
    "finite_com_fraction",
    # Usable at primary threshold
    "usable_frames_primary",
    "usable_fraction_primary",
    # Mass coverage distributions (absolute)
    "mass_coverage_min",
    "mass_coverage_mean",
    "mass_coverage_median",
    "mass_coverage_max",
    # Supported mass coverage distributions
    "supported_mass_coverage_min",
    "supported_mass_coverage_mean",
    "supported_mass_coverage_median",
    "supported_mass_coverage_max",
    # Supported segment missing burden (supported segments only)
    "supported_segment_missing_count",
    "supported_segment_missing_frames",
    "supported_segment_missing_max_consecutive",
    # Longest unusable interval at primary threshold
    "longest_unusable_interval_frames",
    "longest_unusable_interval_seconds",
    # Normalized sample availability at primary threshold
    "normalized_samples_total",
    "normalized_exact_match_usable",
    "normalized_linear_interpolation_usable",
    "normalized_unavailable",
    "normalized_usable",
    "normalized_usable_fraction",
    # Qualification
    "qualification_category",
    "policy_complete_at_primary_threshold",
    "failure_reasons",
    # Explicit stride boolean diagnostics (policy-complete decomposition)
    "all_original_frames_policy_eligible",
    "all_supported_segments_represented",
    "represented_segment_set_invariant",
    "normalized_grid_complete",
    "endpoints_policy_eligible",
    "all_contributing_segments_raw_observed",
)

# ---------------------------------------------------------------------------
# Internal: generic JSON-serializable dataclass converter
# ---------------------------------------------------------------------------


def _jsonable_dataclass(obj: Any) -> Any:
    """Convert a dataclass or nested structure to JSON-serializable form.

    Uses dataclasses.asdict and recursively converts:
    - float dict keys -> stable strings with repr()
    - tuples -> lists
    - sets -> lists
    - Path -> str
    - enum values -> their value
    """
    import dataclasses
    from enum import Enum
    from pathlib import Path

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable_dataclass(asdict(obj))
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Convert float keys to stable string representation
            if isinstance(k, float):
                k = repr(k)
            elif isinstance(k, Path):
                k = str(k)
            elif isinstance(k, Enum):
                k = k.value
            result[k] = _jsonable_dataclass(v)
        return result
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable_dataclass(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None  # JSON doesn't support inf/nan
        return obj
    return obj


# ---------------------------------------------------------------------------
# Internal: git provenance and dependency versions
# ---------------------------------------------------------------------------


def _git_provenance() -> dict[str, Any]:
    """Return git commit hash and dirty status."""
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
    """Return versions of key dependencies."""
    result: dict[str, str | None] = {}
    for name in ("gait-stability", "opencv-python", "numpy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


# ---------------------------------------------------------------------------
# Internal: write stride QC CSV
# ---------------------------------------------------------------------------


def _write_stride_qc(
    path: Path,
    reviewed_strides: list[ParsedReviewedStride],
    aggregate: AggregateCoverageResult,
    primary_threshold: float,
) -> None:
    """Write per-stride QC CSV with one row per canonical stride in order.

    Args:
        path: Output CSV path.
        reviewed_strides: Parsed reviewed strides in canonical order.
        aggregate: AggregateCoverageResult containing stride_summaries.
        primary_threshold: Step 5a inherited primary threshold
            (minimum_mass_coverage).

    Raises:
        ComQualificationArtifactValidationError: On write failure or mismatch.
    """
    # Build stride summary lookup
    stride_summary_by_id = {s.stride_id: s for s in aggregate.stride_summaries}

    # Verify all reviewed strides have summaries
    for rs in reviewed_strides:
        if rs.stride_id not in stride_summary_by_id:
            raise ComQualificationArtifactValidationError(
                f"No StrideCoverageSummary for stride {rs.stride_id!r}"
            )

    try:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COM_STRIDE_QC_FIELDS)

            for rs in reviewed_strides:
                ss = stride_summary_by_id[rs.stride_id]

                # Stride identity / bounds
                row: list[Any] = [
                    rs.stride_id,
                    rs.side,
                    rs.start_event_id,
                    rs.end_event_id,
                    rs.start_frame,
                    rs.end_frame,
                    f"{rs.start_timestamp_seconds:.9f}",
                    f"{rs.end_timestamp_seconds:.9f}",
                    f"{rs.duration_seconds:.9f}",
                    rs.quality,
                    rs.automatic_stride_id,
                    rs.review_intent,
                    rs.review_changes,
                    rs.provenance_notes,
                    # Frame counts at primary threshold
                    ss.frame_count,
                    f"{primary_threshold:.6f}",
                    # Finite COM
                    ss.finite_com_frames,
                    f"{ss.finite_com_fraction:.6f}",
                    # Usable at primary threshold
                    ss.usable_frames_primary,
                    f"{ss.usable_fraction_primary:.6f}",
                    # Mass coverage distributions (absolute)
                    f"{ss.mass_coverage_min:.6f}",
                    f"{ss.mass_coverage_mean:.6f}",
                    f"{ss.mass_coverage_median:.6f}",
                    f"{ss.mass_coverage_max:.6f}",
                    # Supported mass coverage distributions
                    f"{ss.supported_mass_coverage_min:.6f}",
                    f"{ss.supported_mass_coverage_mean:.6f}",
                    f"{ss.supported_mass_coverage_median:.6f}",
                    f"{ss.supported_mass_coverage_max:.6f}",
                    # Supported segment missing burden
                    ss.supported_segment_missing_count,
                    ss.supported_segment_missing_frames,
                    ss.supported_segment_missing_max_consecutive,
                    # Longest unusable interval
                    ss.longest_unusable_interval_frames,
                    f"{ss.longest_unusable_interval_seconds:.6f}",
                    # Normalized sample availability
                    ss.normalized_samples_total,
                    ss.normalized_exact_match_count,
                    ss.normalized_linear_interpolation_count,
                    ss.normalized_samples_total - ss.normalized_samples_usable,
                    ss.normalized_samples_usable,
                    f"{ss.normalized_samples_usable_fraction:.6f}",
                    # Qualification
                    ss.qualification_category,
                    "true"
                    if ss.qualification_category == "policy_complete_at_threshold"
                    else "false",
                    "|".join(ss.failure_reasons) if ss.failure_reasons else "",
                    # Explicit policy-complete component diagnostics
                    "true" if ss.all_original_frames_policy_eligible else "false",
                    "true" if ss.all_supported_segments_represented else "false",
                    "true" if ss.represented_segment_set_invariant else "false",
                    "true" if ss.normalized_grid_complete else "false",
                    "true" if ss.endpoints_policy_eligible else "false",
                    "true" if ss.all_contributing_segments_raw_observed else "false",
                ]
                if len(row) != len(COM_STRIDE_QC_FIELDS):
                    raise ComQualificationArtifactValidationError(
                        "COM stride QC row length does not match its declared schema: "
                        f"{len(row)} values for {len(COM_STRIDE_QC_FIELDS)} fields"
                    )
                writer.writerow(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ComQualificationArtifactValidationError(
            f"Could not write stride QC CSV to {path}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Internal: transactional publish with UUID backups and rollback
# ---------------------------------------------------------------------------


def _publish(staging: Path, destination: Path) -> None:
    """Transactional publish of exactly 3 Step 5b qualification artifacts.

    Matches Step 5a pattern: UUID backups, rollback on failure,
    ArtifactPublishError on any failure.

    Args:
        staging: Directory containing staged artifacts.
        destination: Destination directory for published artifacts.

    Raises:
        ArtifactPublishError: If any artifact fails to publish or
            rollback is incomplete.
    """
    backups: dict[Path, Path] = {}
    published: list[Path] = []

    try:
        # Phase 1: backup existing destination files with UUID suffixes
        for name in COM_QUALIFICATION_OUTPUT_ARTIFACT_NAMES:
            target = destination / name
            if target.exists():
                backup = destination / f"{name}.backup-{uuid.uuid4().hex}"
                target.replace(backup)
                backups[target] = backup

        # Phase 2: move staged artifacts to destination
        for name in COM_QUALIFICATION_OUTPUT_ARTIFACT_NAMES:
            staged = staging / name
            if not staged.is_file():
                raise OSError(f"Staged qualification artifact is missing: {staged}")
            target = destination / name
            staged.replace(target)
            published.append(target)

    except OSError as exc:
        # Rollback: remove any published files, restore backups
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
            "Could not publish complete Step 5b qualification artifact set"
            + (f"; rollback may be incomplete: {detail}" if detail else "")
        ) from exc

    # Phase 3: cleanup backups on success
    for backup in backups.values():
        with suppress(OSError):
            backup.unlink()


# ---------------------------------------------------------------------------
# Public return type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComQualificationArtifacts:
    """Published Step 5b qualification artifact paths."""

    artifact_directory: Path
    qualification_json_path: Path
    stride_qc_csv_path: Path
    annotated_video_path: Path


# ---------------------------------------------------------------------------
# Internal: qualification metadata builder
# ---------------------------------------------------------------------------


def _build_qualification_metadata(
    directory: Path,
    staging: Path,
    config: ComQualificationConfig,
    primary_threshold: float,
    sex: Literal["male", "female"],
    frame_results: tuple[FrameComResult, ...],
    reviewed_strides: list[ParsedReviewedStride],
    stride_results: dict[str, tuple[StrideComSample, ...]],
    aggregate: AggregateCoverageResult,
    input_snapshot: dict[Path, str],
    com_metadata: dict[str, Any],
    preprocessing_metadata: dict[str, Any],
    source_info: _SourceInfo,
    com_proxy_path: Path,
    stride_com_path: Path,
) -> dict[str, Any]:
    """Build the qualification metadata JSON object.

    Args:
        directory: Artifact destination directory.
        staging: Staging directory containing written output artifacts.
        config: Qualification configuration (threshold grid).
        primary_threshold: Step 5a configured minimum_mass_coverage.
        sex: Anthropometry sex used in Step 5a.
        frame_results: All FrameComResult from Step 5a.
        reviewed_strides: Parsed reviewed strides.
        stride_results: StrideComSample tuples by stride_id from Step 5a.
        aggregate: AggregateCoverageResult from compute_qualification.
        input_snapshot: Snapshot of all input file hashes (Path -> sha256).
        com_metadata: Validated com_metadata.json from Step 5a.
        preprocessing_metadata: Validated preprocessing_metadata.json from Step 3.
        source_info: Validated source video info.
        com_proxy_path: Path to com_proxy.csv.
        stride_com_path: Path to stride_com.csv.

    Returns:
        Complete qualification metadata dict ready for JSON serialization.
    """
    theoretical = theoretical_supported_mass_fraction(sex)
    model_total = MODEL_MASS_TOTAL_MALE if sex == "male" else MODEL_MASS_TOTAL_FEMALE
    head_mass_fraction = model_total - theoretical

    # Compute output hashes from staged artifacts
    output_hashes: dict[str, str | None] = {}
    for name in COM_QUALIFICATION_OUTPUT_ARTIFACT_NAMES:
        if name == "com_qualification.json":
            output_hashes[name] = None  # self-null semantics
        else:
            try:
                output_hashes[name] = sha256_file(staging / name)
            except OSError:
                output_hashes[name] = None

    # Build stride statistics
    stride_stats: list[dict[str, Any]] = []
    for s in reviewed_strides:
        stride_summary = next(
            (ss for ss in aggregate.stride_summaries if ss.stride_id == s.stride_id),
            None,
        )
        qual_cat = (
            stride_summary.qualification_category if stride_summary else "unknown"
        )
        policy_complete_at_primary_threshold = (
            stride_summary.qualification_category == "policy_complete_at_threshold"
            if stride_summary
            else False
        )
        norm_total = stride_summary.normalized_samples_total if stride_summary else 0
        norm_frac = (
            stride_summary.normalized_samples_usable_fraction if stride_summary else 0.0
        )
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
                "qualification_category": qual_cat,
                "policy_complete_at_primary_threshold": (
                    policy_complete_at_primary_threshold
                ),
                "normalized_samples_total": norm_total,
                "normalized_usable_fraction": norm_frac,
            }
        )

    # Build warnings (summarized, not per-frame)
    warnings_list: list[str] = []

    # No primary-usable frames
    if aggregate.usable_frames_primary == 0:
        warnings_list.append(
            "No frames meet primary threshold usable criteria "
            f"(usable_frames_primary=0 at threshold={primary_threshold:.3f})"
        )

    # No policy-complete strides
    policy_complete_count = sum(
        1
        for ss in aggregate.stride_summaries
        if ss.qualification_category == "policy_complete_at_threshold"
    )
    if policy_complete_count == 0:
        warnings_list.append("No strides qualify as policy_complete_at_threshold")

    # Structurally unsupported head
    total_model = MODEL_MASS_TOTAL_MALE if sex == "male" else MODEL_MASS_TOTAL_FEMALE
    head_mass_frac = total_model - theoretical
    warnings_list.append(
        f"Head segment structurally unsupported (mass fraction {head_mass_frac:.4f}); "
        "MediaPipe Pose lacks source-compatible vertex/neck joint-centre line; "
        "nose is NOT the vertex; head coefficient retained for provenance but "
        "never participates in frame COM calculations"
    )

    # Supported segments persistently unavailable
    persistently_unavailable_segments: list[str] = []
    for seg_summary in aggregate.segment_summaries:
        if seg_summary.is_supported and seg_summary.usable_frames == 0:
            persistently_unavailable_segments.append(seg_summary.segment_name)
    if persistently_unavailable_segments:
        warnings_list.append(
            "Supported segments persistently unavailable (0 usable frames): "
            f"{', '.join(persistently_unavailable_segments)}"
        )

    # Low usable fraction at primary
    if aggregate.usable_fraction_primary < 1.0 and aggregate.total_frames > 0:
        usable_frac = aggregate.usable_fraction_primary
        usable_frames = aggregate.usable_frames_primary
        total_frames = aggregate.total_frames
        warnings_list.append(
            f"Global usable fraction at primary threshold: "
            f"{usable_frac:.3f} ({usable_frames}/{total_frames} frames)"
        )

    # Step 5a config and metadata
    step5a_config = _require_dict(com_metadata.get("config"), "com_metadata.config")
    step5a_algorithm = _require_dict(
        com_metadata.get("algorithm"), "com_metadata.algorithm"
    )
    step5a_coverage = _require_dict(
        com_metadata.get("coverage"), "com_metadata.coverage"
    )
    step5a_inputs = _require_dict(com_metadata.get("inputs"), "com_metadata.inputs")
    step5a_outputs = _require_dict(com_metadata.get("outputs"), "com_metadata.outputs")

    # Step 5a run identification
    step5a_run_id = com_metadata.get("run_id")
    step5a_created = com_metadata.get("created_at_utc")
    step5a_algorithm_version = com_metadata.get("algorithm_version")
    step5a_schema_version = com_metadata.get("schema_version")

    # Step 5a input hashes (upstream sources)
    step5a_input_hashes: dict[str, str] = {}
    for basename in _STEP5_UPSTREAM_BASENAMES:
        entry = step5a_inputs.get(basename, {})
        step5a_input_hashes[basename] = entry.get("sha256", "")

    # Step 5a output hashes (including self-null for com_metadata.json)
    step5a_output_hashes: dict[str, str | None] = {}
    for basename in _STEP5_OUTPUT_BASENAMES:
        entry = step5a_outputs.get(basename, {})
        step5a_output_hashes[basename] = entry.get("sha256")

    # Build all input/source paths with hashes
    all_inputs_with_hashes: dict[str, dict[str, Any]] = {}

    # Step 5a outputs
    for basename in _STEP5_OUTPUT_BASENAMES:
        path = directory / basename
        all_inputs_with_hashes[f"step5_output.{basename}"] = {
            "path": str(path),
            "sha256": input_snapshot.get(path, ""),
        }

    # Step 5a upstream inputs
    for basename in _STEP5_UPSTREAM_BASENAMES:
        path = directory / basename
        all_inputs_with_hashes[f"step5_upstream.{basename}"] = {
            "path": str(path),
            "sha256": input_snapshot.get(path, ""),
        }

    # Source video
    all_inputs_with_hashes["source_video"] = {
        "path": str(source_info.path),
        "sha256": source_info.sha256,
    }

    preprocessing_inherited = _require_dict(
        preprocessing_metadata.get("inherited_provenance"),
        "preprocessing_metadata.inherited_provenance",
    )
    capture_assumptions = _require_dict(
        preprocessing_inherited.get("capture_assumptions"),
        "preprocessing_metadata.inherited_provenance.capture_assumptions",
    )
    preprocessing_interpolation = _require_dict(
        preprocessing_metadata.get("interpolation"),
        "preprocessing_metadata.interpolation",
    )
    preprocessing_smoothing = _require_dict(
        preprocessing_metadata.get("smoothing"),
        "preprocessing_metadata.smoothing",
    )

    return {
        "schema_version": COM_QUALIFICATION_SCHEMA_VERSION,
        "algorithm_version": COM_QUALIFICATION_ALGORITHM_VERSION,
        "scope": (
            "Engineering/QC qualification of represented-segment COM proxy coverage "
            "and stride-level sample completeness. Not a stability metric, clinical "
            "result, or force-plate-validated measurement."
        ),
        "run_id": uuid.uuid4().hex,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "inputs": all_inputs_with_hashes,
        "config": {
            "primary_threshold": primary_threshold,
            "coverage_thresholds": list(config.coverage_thresholds),
        },
        "step5a": {
            "run_id": step5a_run_id,
            "created_at_utc": step5a_created,
            "algorithm_version": step5a_algorithm_version,
            "schema_version": step5a_schema_version,
            "config": {
                "anthropometry_sex": step5a_config.get("anthropometry_sex"),
                "minimum_mass_coverage": step5a_config.get("minimum_mass_coverage"),
                "normalized_stride_samples": step5a_config.get(
                    "normalized_stride_samples"
                ),
            },
            "algorithm": {
                "selected_sex": step5a_algorithm.get("selected_sex"),
                "model_total_mass": step5a_algorithm.get("model_total_mass"),
                "coefficient_table": step5a_algorithm.get("coefficient_table"),
                "unsupported_segments": step5a_algorithm.get("unsupported_segments"),
                "represented_mass_max": step5a_algorithm.get("represented_mass_max"),
            },
            "coverage": {
                "threshold": step5a_coverage.get("threshold"),
                "rationale": step5a_coverage.get("rationale"),
            },
            "input_hashes": step5a_input_hashes,
            "output_hashes": step5a_output_hashes,
        },
        "sex": sex,
        "anthropometric_model": {
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
            "model_total_mass": (
                MODEL_MASS_TOTAL_MALE if sex == "male" else MODEL_MASS_TOTAL_FEMALE
            ),
            "theoretical_supported_mass_fraction": theoretical,
            "theoretical_supported_mass_fraction_note": (
                "Maximum representable body-mass fraction when all supported "
                "segments are usable. Head (mass fraction "
                f"{head_mass_fraction:.4f}) "
                "is structurally unsupported because standard MediaPipe Pose "
                "lacks a defensible source-compatible vertex/neck joint-centre "
                "line. Nose is NOT the vertex. Head coefficient and mass are "
                "retained for provenance but never participate in frame COM "
                "calculations."
            ),
            "structurally_unsupported_segments": list(UNSUPPORTED_SEGMENTS),
            "structurally_unsupported_reason": (
                "Head: no source-compatible vertex and neck joint-centre available "
                "from standard MediaPipe Pose landmarks. Nose is not the anatomical "
                "vertex. This is a structural limitation of the landmark set, not a "
                "data quality issue."
            ),
        },
        "formulas": {
            "mass_coverage": (
                "mass_coverage = represented_body_mass_fraction = "
                "sum(mass_fraction_i for usable segments); "
                "unrenormalized sum of mass fractions of usable segments; "
                "unsupported segments (head) excluded; "
                f"maximum possible: {theoretical:.4f} ({sex})"
            ),
            "supported_mass_coverage": (
                "supported_mass_coverage = mass_coverage / "
                "theoretical_supported_mass_fraction; "
                "fraction of theoretically representable mass that is actually usable; "
                "dimensionless ratio in [0, 1]; clamped to 1.0 within 1e-12 tolerance"
            ),
            "mass_coverage_interpretation": (
                "Absolute mass_coverage is the raw represented body mass fraction "
                "contributing to the COM proxy. It is NOT renormalized to 1.0. "
                "Supported_mass_coverage rescales by the theoretical maximum. "
                "High coverage is NOT accuracy, confidence, probability, or COM "
                "validity. Pose-model estimates and processed trajectories are NOT "
                "observed/validated measurements."
            ),
        },
        "coordinate_system": (
            "Normalized image-plane coordinates (x right, y down); "
            "values are not constrained to [0, 1]; no z-coordinate or physical scale. "
            "Image axes are NOT laboratory progression/vertical/gravity/ground axes. "
            "No roll or tilt correction has been applied."
        ),
        "camera_view": {
            "required_view": (
                "Static near-sagittal weak-perspective view with minimal out-of-plane "
                "motion; single 2D side-view assumed; depth not reconstructed; "
                "projection assumes low-distortion capture with equivalent endpoint "
                "depth."
            ),
            "artifact_declaration": {
                "capture_assumptions": capture_assumptions,
                "camera_view": capture_assumptions.get("camera_view"),
                "status": "not_declared_or_verified",
                "machine_verified": False,
            },
            "human_review": {
                "required": True,
                "status": "not_recorded",
            },
            "note": (
                "The inherited artifact explicitly reports that camera view was not "
                "required, declared, or verified. The Step 5b required view is stated "
                "separately and has not been established for this artifact. "
                "Image axes are not laboratory progression/vertical/gravity/ground "
                "axes. "
                "No roll or tilt correction has been applied. "
                "Human review of capture suitability is required but not recorded by "
                "this pipeline."
            ),
        },
        "primary_policy_gate": {
            "description": (
                "A frame is marked usable (usable=True) iff: "
                "COM is finite AND mass_coverage > 0 AND "
                "mass_coverage >= primary_threshold (equality passes). "
                "This policy is inherited unchanged from Step 5a."
            ),
            "threshold": primary_threshold,
            "zero_coverage_never_usable": True,
            "equality_passes": True,
        },
        "sensitivity_grid": {
            "coverage_thresholds": list(config.coverage_thresholds),
            "note": (
                "Predeclared project engineering grid for absolute mass_coverage "
                "thresholds. This grid is unvalidated and not selected or tuned on "
                "the current artifact. No automatic threshold selection is performed. "
                "Each threshold is independently evaluated for frame/stride/normalized "
                "availability. Primary threshold (from Step 5a config) governs the "
                "primary qualification category; sensitivity grid provides additional "
                "diagnostic visibility only."
            ),
        },
        "preprocessing_inheritance": {
            "run_id": preprocessing_metadata.get("run_id"),
            "created_at_utc": preprocessing_metadata.get("created_at_utc"),
            "schema_version": preprocessing_metadata.get("schema_version"),
            "algorithm_version": preprocessing_metadata.get("algorithm_version"),
            "config": preprocessing_metadata.get("config", {}),
            "interpolation": {
                "method": preprocessing_interpolation.get("method"),
                "maximum_missing_samples": preprocessing_interpolation.get(
                    "maximum_missing_samples"
                ),
                "gait_event_crossing": preprocessing_interpolation.get(
                    "gait_event_crossing"
                ),
                "note": (
                    "Interpolation method and gap limits are provenance metadata; they "
                    "do not imply accuracy of interpolated values. Interpolation or "
                    "smoothing flags may occur on contributors to an unavailable "
                    "segment."
                ),
            },
            "smoothing": {
                "method": preprocessing_smoothing.get("method"),
                "configured_window_frames": preprocessing_smoothing.get(
                    "configured_window_frames"
                ),
                "phase": preprocessing_smoothing.get("phase"),
                "edge_behavior": preprocessing_smoothing.get("edge_behavior"),
                "note": (
                    "Smoothing method and parameters are provenance metadata; they do "
                    "not imply accuracy of smoothed values. Smoothing flags may occur "
                    "on contributors to an unavailable segment."
                ),
            },
        },
        "stride_completeness": {
            "definition": (
                "A stride qualifies as 'policy_complete_at_threshold' iff ALL of the "
                "following hold at the primary threshold: (1) every original frame "
                "in the stride window is eligible (finite COM, mass_coverage > 0, "
                "mass_coverage >= primary_threshold); (2) the represented segment "
                "set (usable_segments()) is invariant across all frames in the stride; "
                "(3) all normalized stride samples are available via exact timestamp "
                "match or linear interpolation between adjacent eligible frames with "
                "identical usable_segments() tuples; (4) start and end endpoint frames "
                "are eligible. This is an engineering/QC category only; it does NOT "
                "imply biomechanical validity, accuracy, or clinical significance."
            ),
            "category_semantics": {
                "policy_complete_at_threshold": (
                    "All frames eligible, invariant segment set, all normalized "
                    "samples available, endpoints available"
                ),
                "usable_samples_only": (
                    "All frames eligible but policy_complete_at_threshold criteria "
                    "not met (segment set varies or endpoints unavailable)"
                ),
                "insufficient_coverage": "Some frames below primary threshold",
                "no_usable_frames": "No finite COM frames in stride",
                "endpoint_unavailable": "Stride window empty or no frames",
            },
        },
        "stride_statistics": stride_stats,
        "sensitivity_results": _jsonable_dataclass(aggregate),
        "outputs": {
            name: {
                "path": str(directory / name),
                "sha256": digest,
            }
            for name, digest in output_hashes.items()
            if name != "com_qualification.json"
        }
        | {
            "com_qualification.json": {
                "path": str(directory / "com_qualification.json"),
                "sha256": None,
                "sha256_semantics": (
                    "not self-recorded because embedding a file's own "
                    "digest changes that digest"
                ),
            }
        },
        "warnings": warnings_list,
        "runtime": {
            "python_version": sys.version,
            "dependency_versions": _dependency_versions(),
            "git": _git_provenance(),
            "randomness": "none",
        },
        "limitations": [
            "High mass coverage is NOT accuracy, confidence, probability, "
            "or COM validity.",
            "Pose-model estimates and processed trajectories are NOT "
            "observed/validated measurements.",
            "Coordinates are normalized image-plane (x right, y down); no physical "
            "scale or [0,1] constraint.",
            "No stability metrics are produced by this pipeline.",
            "Anthropometric coefficients are population averages from de Leva 1996, "
            "not individual measurements.",
            "MediaPipe landmarks are 2D proxies, not anatomical joint centers.",
            "Head segment is structurally unsupported; maximum representable mass "
            f"fraction is {theoretical:.4f} ({sex}). Head mass fraction "
            f"is {head_mass_fraction:.4f}.",
            "Coverage threshold is an engineering QC gate, not a validated accuracy "
            "bound.",
            "Stride normalization uses canonical timestamps; progression is not "
            "validated gait-cycle percentage.",
            "Projection assumes static, near-sagittal, low-distortion, "
            "weak-perspective capture; not machine-verified.",
            "Step 3 interpolation and smoothing affect landmark positions and "
            "propagate through COM calculations.",
            "Hand distal endpoint proxy uses midpoint of MediaPipe index and pinky "
            "points, an unvalidated distal hand endpoint surrogate.",
            "This output is a qualification diagnostic, not a fall-risk indicator, "
            "diagnostic tool, or ground-truth reference.",
        ],
        "validation_status": (
            "This qualification assesses engineering/QC coverage completeness of a "
            "represented-segment COM proxy derived from video-based pose estimates. "
            "It has not been validated against force plates, motion capture, or "
            "clinical measurements. It is NOT a stability metric, fall-risk "
            "assessment, clinical result, or ground-truth measurement."
        ),
        "clean_capture_evaluation": {
            "status": "required",
            "note": (
                "Required clean-capture evidence is an appropriate capture with the "
                "complete body, both arms, and both feet visible, minimal occlusion, "
                "and a static near-sagittal setup, demonstrating sufficiently complete "
                "proxy trajectories under documented QC. Independent reference-system "
                "COM accuracy validation remains required scientific work, but it is "
                "not the Step5b clean-capture coverage gate."
            ),
            "machine_established": False,
        },
        "readiness_for_step6": {
            "status": "CONDITIONAL",
            "reason": (
                "A suitable capture demonstrating sufficiently complete proxy "
                "trajectories under a documented QC policy is required for Step 6 GO. "
                "Independent reference validation of COM accuracy (force-plate, "
                "motion-capture) remains a scientific limitation; this gate does not "
                "fabricate such validation. Readiness remains CONDITIONAL pending "
                "external clean-capture evidence."
            ),
            "engineering_exploratory_readiness": {
                "status": "CONDITIONAL",
                "reason": "clean capture with predeclared QC still required",
            },
            "scientific_measurement_readiness": {
                "status": "NO-GO",
                "reason": (
                    "Step5b does not establish laboratory-equivalent COM "
                    "position/velocity or downstream stability measurement validity "
                    "and lacks reference validation, calibrated "
                    "scale/geometry/ground-or-gravity alignment, validated gait events "
                    "and downstream inputs"
                ),
            },
        },
    }


# ---------------------------------------------------------------------------
# Public: COM qualification pipeline entry point
# ---------------------------------------------------------------------------


def qualify_com(
    artifact_directory: str | Path,
    config: ComQualificationConfig,
    *,
    video_path: str | Path | None = None,
) -> ComQualificationArtifacts:
    """Run COM qualification pipeline (Step 5b).

    Loads Step 5a outputs and all upstream inputs, validates complete
    Step 3/4b lineage and hashes, computes coverage/QC diagnostics via
    com_qualification.compute_qualification, renders annotated COM video,
    writes stride QC CSV and qualification metadata, and publishes all
    three artifacts transactionally.

    Args:
        artifact_directory: Directory containing Step 5a outputs and
            upstream inputs (six canonical basenames).
        config: ComQualificationConfig with coverage_thresholds grid.
        video_path: Optional override for source video path. If provided,
            the file must exist and its hash must match the inherited
            provenance source hash from preprocessing_metadata.json.

    Returns:
        ComQualificationArtifacts with paths to the three published outputs:
        com_qualification.json, com_stride_qc.csv, annotated_com.mp4.

    Raises:
        ComQualificationArtifactValidationError: If any input is missing,
            hash mismatches, or schema/contract validation fails.
        ComQualificationRenderError: If annotated video rendering fails.
        ArtifactPublishError: If transactional publish fails.
        TypeError: If config is not a ComQualificationConfig instance.
    """
    # Strict config type check
    if not isinstance(config, ComQualificationConfig):
        raise TypeError(
            f"config must be ComQualificationConfig, got {type(config).__name__}"
        )

    directory = Path(artifact_directory).expanduser().resolve()
    if not directory.is_dir():
        raise ComQualificationArtifactValidationError(
            f"Artifact directory does not exist: {directory}"
        )

    # Phase 1: resolve paths, validate metadata, snapshot all input hashes
    (
        inputs,
        com_metadata,
        preprocessing_metadata,
        sex,
        primary_threshold,
        normalized_n,
        source_info,
    ) = _resolve_and_validate_metadata(
        directory, Path(video_path) if video_path is not None else None
    )

    # Snapshot all input hashes BEFORE any reading
    input_snapshot = _snapshot_hashes(inputs)

    # Revalidate complete Step 3/4b lineage/events/review metadata via
    # com_pipeline._validate_all_inputs and compare its input hashes
    # to our snapshot / Step 5 metadata
    com_validation_result = com_validate_all_inputs(directory)
    (
        com_inputs,
        com_input_hashes,
        _com_pose_manifest,
        _com_processed_data,
        _com_reviewed_events,
        _com_reviewed_strides,
        _com_preproc_meta,
        _com_rr_meta,
    ) = com_validation_result

    # Compare com_pipeline's input hashes (six canonical basenames) to our snapshot
    for basename in _STEP5_UPSTREAM_BASENAMES:
        path = directory / basename
        our_hash = input_snapshot.get(path)
        com_hash = com_input_hashes.get(basename)
        if our_hash != com_hash:
            raise ComQualificationArtifactValidationError(
                f"Hash mismatch for {basename}: qualification snapshot={our_hash}, "
                f"com_pipeline validation={com_hash}"
            )

    # Phase 2: read all data using existing strict readers
    # Read pose frames
    pose_frames = _read_pose_frames(inputs.pose_frames)

    # Read reviewed strides
    reviewed_strides = _read_reviewed_strides(inputs.reviewed_strides, pose_frames)

    # Read com_proxy.csv -> FrameComResult tuple
    frame_results = _read_com_proxy(inputs.com_proxy, primary_threshold, sex)

    # Read stride_com.csv -> dict[stride_id, tuple[StrideComSample]]
    stride_results = _read_stride_com(
        inputs.stride_com, reviewed_strides, pose_frames, normalized_n
    )

    # Read processed landmarks -> (landmark_rows, points_by_frame)
    landmark_rows, points_by_frame = _read_processed_landmarks(
        inputs.processed_landmarks, pose_frames
    )

    # Validate counts/timestamps/sex/model consistency
    if len(frame_results) != len(pose_frames):
        raise ComQualificationArtifactValidationError(
            f"Frame count mismatch: com_proxy has {len(frame_results)} frames, "
            f"pose_frames has {len(pose_frames)}"
        )

    # Verify frame timestamps match exactly
    for fi, (fr_ts, fr_status) in pose_frames.items():
        fr = frame_results[fi]
        if abs(fr.timestamp_seconds - fr_ts) > 1e-9:
            raise ComQualificationArtifactValidationError(
                f"Frame {fi} timestamp mismatch: com_proxy={fr.timestamp_seconds}, "
                f"pose_frames={fr_ts}"
            )
        if fr.frame_status != fr_status:
            raise ComQualificationArtifactValidationError(
                f"Frame {fi} status mismatch: com_proxy={fr.frame_status}, "
                f"pose_frames={fr_status}"
            )

    # Verify sex matches
    if sex not in ("male", "female"):
        raise ComQualificationArtifactValidationError(
            f"Invalid sex from metadata: {sex!r}"
        )

    # Phase 3: compute qualification via pure function
    aggregate = compute_qualification(
        frame_results=frame_results,
        stride_results=stride_results,
        reviewed_strides=reviewed_strides,
        config=config,
        primary_threshold=primary_threshold,
        anthropometry_sex=sex,
        landmark_rows=landmark_rows,
    )

    # Phase 4: stage outputs
    staging = Path(
        tempfile.mkdtemp(
            prefix=f"{directory.name}.com-qualification-staging-",
            dir=directory.parent,
        )
    )

    try:
        # Write stride QC CSV
        stride_qc_path = staging / "com_stride_qc.csv"
        _write_stride_qc(stride_qc_path, reviewed_strides, aggregate, primary_threshold)

        # Write annotated COM video
        annotated_video_path = staging / "annotated_com.mp4"
        _write_annotated_com_video(
            output_path=annotated_video_path,
            source_info=source_info,
            frame_results=frame_results,
            points_by_frame=points_by_frame,
            reviewed_strides=reviewed_strides,
            primary_threshold=primary_threshold,
            theoretical_supported=theoretical_supported_mass_fraction(sex),
        )

        # Build and write qualification metadata JSON
        qualification_metadata = _build_qualification_metadata(
            directory=directory,
            staging=staging,
            config=config,
            primary_threshold=primary_threshold,
            sex=sex,
            frame_results=frame_results,
            reviewed_strides=reviewed_strides,
            stride_results=stride_results,
            aggregate=aggregate,
            input_snapshot=input_snapshot,
            com_metadata=com_metadata,
            preprocessing_metadata=preprocessing_metadata,
            source_info=source_info,
            com_proxy_path=inputs.com_proxy,
            stride_com_path=inputs.stride_com,
        )
        qual_json_path = staging / "com_qualification.json"
        qual_json_path.write_text(
            json.dumps(_jsonable_dataclass(qualification_metadata), indent=2) + "\n",
            encoding="utf-8",
        )

        # Recheck ALL input/source hashes before publish
        _recheck_hashes(input_snapshot)

        # Also verify com_proxy and stride_com were not modified
        current_com_proxy_hash = sha256_file(inputs.com_proxy)
        current_stride_com_hash = sha256_file(inputs.stride_com)
        if current_com_proxy_hash != input_snapshot[inputs.com_proxy]:
            raise ComQualificationArtifactValidationError(
                "com_proxy.csv was modified during qualification"
            )
        if current_stride_com_hash != input_snapshot[inputs.stride_com]:
            raise ComQualificationArtifactValidationError(
                "stride_com.csv was modified during qualification"
            )

        # Phase 5: transactional publish
        _publish(staging, directory)

    finally:
        # Cleanup staging directory
        shutil.rmtree(staging, ignore_errors=True)

    # Verify all three outputs exist
    for name in COM_QUALIFICATION_OUTPUT_ARTIFACT_NAMES:
        out_path = directory / name
        if not out_path.is_file():
            raise ComQualificationPipelineError(
                f"Qualification artifact not published: {name}"
            )

    return ComQualificationArtifacts(
        artifact_directory=directory,
        qualification_json_path=directory / "com_qualification.json",
        stride_qc_csv_path=directory / "com_stride_qc.csv",
        annotated_video_path=directory / "annotated_com.mp4",
    )
