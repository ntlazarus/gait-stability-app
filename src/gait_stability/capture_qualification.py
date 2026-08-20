"""Clean-capture qualification from frozen Step 5b evidence (Step 5c).

This module does not recompute pose, COM, gait-event, or Step 5b results. It
validates two Step 5b records and an external capture review, applies one frozen
engineering policy, and publishes an auditable qualification record.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from gait_stability.com_estimation import SEGMENT_NAMES, UNSUPPORTED_SEGMENTS
from gait_stability.com_pipeline import COM_ALGORITHM_VERSION, COM_SCHEMA_VERSION
from gait_stability.com_qualification_pipeline import (
    COM_QUALIFICATION_ALGORITHM_VERSION,
    COM_QUALIFICATION_OUTPUT_ARTIFACT_NAMES,
    COM_QUALIFICATION_SCHEMA_VERSION,
)
from gait_stability.pose_preprocessing import (
    PREPROCESSING_ALGORITHM_VERSION,
    PREPROCESSING_SCHEMA_VERSION,
)
from gait_stability.review_resolution import (
    REVIEW_RESOLUTION_ALGORITHM_VERSION,
    REVIEW_RESOLUTION_SCHEMA_VERSION,
)
from gait_stability.video_ingestion import ArtifactPublishError, sha256_file

CAPTURE_QUALIFICATION_SCHEMA_VERSION = 1
CAPTURE_QUALIFICATION_ALGORITHM_VERSION = "step5c-clean-capture-qualification-1"
CAPTURE_QUALIFICATION_OUTPUT_NAME = "capture_qualification.json"
STEP5B_QUALIFICATION_OUTPUT_NAME = COM_QUALIFICATION_OUTPUT_ARTIFACT_NAMES[0]

PRIMARY_THRESHOLD = 0.90
FROZEN_COVERAGE_GRID = (0.80, 0.82, 0.84, 0.86, 0.88, 0.90)

CAPTURE_ITEM_NAMES = (
    "full_body_in_frame",
    "head_to_feet_framing_adequate",
    "both_arms_visible",
    "both_feet_visible",
    "approximately_sagittal_view",
    "fixed_camera_no_pan_tilt_zoom_or_reframing",
    "camera_motion_absent_or_negligible",
    "minimal_subject_out_of_plane_motion_and_no_turns",
    "no_obvious_severe_lens_or_perspective_distortion",
    "major_self_occlusion_limited",
    "lighting_adequate",
    "subject_background_separation_adequate",
    "several_complete_candidate_gait_cycles_present",
    "anatomical_left_right_established",
    "walking_direction_established",
    "bilateral_com_landmarks_trackable",
    "bilateral_event_landmarks_trackable",
)

ANNOTATED_ITEM_NAMES = (
    "trajectory_continuity_adequate",
    "no_visibly_nonphysiological_tracking_jumps",
    "no_identity_loss_or_whole_pose_reset",
    "no_visible_anatomical_left_right_label_swap",
    "no_unexplained_sustained_supported_segment_dropout",
    "no_obvious_centroid_relocation_with_limb_disappearance",
    "no_visible_tracking_discontinuity_within_0_2_seconds_of_reviewed_event_boundaries",
    "proxy_grossly_follows_tracked_body",
)

ReviewStatus = Literal["confirmed", "not_confirmed", "not_applicable", "uncertain"]
ReviewerType = Literal["human", "automated_assistant"]
ReviewIndependence = Literal["independent", "not_independent", "uncertain"]
EngineeringDecision = Literal["GO", "CONDITIONAL", "NO-GO"]
ExternalHumanReviewState = Literal["confirmed", "pending", "incomplete"]

_STRIDE_QUALIFICATION_CATEGORIES = {
    "policy_complete_at_threshold",
    "usable_samples_only",
    "insufficient_coverage",
    "no_usable_frames",
    "endpoint_unavailable",
}


class CaptureQualificationError(Exception):
    """Base exception for Step 5c failures."""


class CaptureQualificationValidationError(CaptureQualificationError):
    """Raised when an input violates the Step 5c contract."""


@dataclass(frozen=True, slots=True)
class CaptureReviewItem:
    """One required external-review finding."""

    status: ReviewStatus
    note: str


@dataclass(frozen=True, slots=True)
class CaptureReviewer:
    """Identity and independence declaration for the reviewer."""

    reviewer_type: ReviewerType
    identifier: str
    role: str
    independence: ReviewIndependence


@dataclass(frozen=True, slots=True)
class CaptureReview:
    """Strictly parsed external clean-capture review."""

    schema_version: int
    review_id: str
    reviewed_at_utc: str
    reviewer: CaptureReviewer
    whole_source_video_inspected: bool
    whole_annotated_com_video_inspected: bool
    capture_protocol: str
    source_video_sha256: str
    annotated_com_sha256: str
    current_qualification_sha256: str
    walking_direction: Literal["image_left", "image_right"]
    declared_direction_matches_inherited_step4: bool
    orientation_notes: str
    capture_items: tuple[tuple[str, CaptureReviewItem], ...]
    annotated_com_items: tuple[tuple[str, CaptureReviewItem], ...]

    def all_items(self) -> tuple[tuple[str, CaptureReviewItem], ...]:
        """Return capture and annotation findings in canonical order."""
        return self.capture_items + self.annotated_com_items


@dataclass(frozen=True, slots=True)
class Step5cEvidence:
    """Frozen quantitative evidence extracted from one Step 5b record."""

    finite_com_fraction: float
    primary_eligible_fraction: float
    primary_longest_usable_interval_seconds: float
    persistent_supported_segments: tuple[str, ...]
    reviewed_candidate_stride_count: int
    policy_complete_stride_count: int
    policy_complete_stride_fraction: float
    policy_complete_left_count: int
    policy_complete_right_count: int
    normalized_usable_fraction: float
    empirical_max_mass_coverage: float
    theoretical_supported_mass_fraction: float = 0.0
    total_frames: int = 0
    finite_com_frames: int = 0
    primary_eligible_frame_count: int = 0
    primary_longest_usable_interval_frames: int = 0
    normalized_total_samples: int = 0
    normalized_exact_match_usable: int = 0
    normalized_linear_interpolation_usable: int = 0
    normalized_usable_samples: int = 0
    transient_supported_segments: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CriterionResult:
    """Outcome of one frozen Step 5c engineering criterion."""

    name: str
    passed: bool
    actual: Any
    requirement: str
    severity: Literal["go_required", "hard_no_go"]


@dataclass(frozen=True, slots=True)
class EngineeringReadinessEvaluation:
    """Pure frozen-policy evaluation result."""

    decision: EngineeringDecision
    state: Literal["evaluated", "pending_external_human_review"]
    evaluation_state: Literal["evaluated"]
    external_human_review_state: ExternalHumanReviewState
    criteria: tuple[CriterionResult, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CaptureQualificationArtifacts:
    """Published Step 5c artifact path."""

    artifact_directory: Path
    qualification_json_path: Path


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CaptureQualificationValidationError(f"{label} must be an object")
    for key in value:
        if not isinstance(key, str):
            raise CaptureQualificationValidationError(
                f"{label} keys must all be strings"
            )
    return cast(Mapping[str, object], value)


def _exact_keys(
    value: Mapping[str, object], expected: Sequence[str], label: str
) -> None:
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        extra = sorted(actual - expected_set)
        raise CaptureQualificationValidationError(
            f"{label} keys do not match the canonical schema; "
            f"missing={missing}, extra={extra}"
        )


def _string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a nonempty string"
        raise CaptureQualificationValidationError(f"{label} must be {qualifier}")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise CaptureQualificationValidationError(f"{label} must be a boolean")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaptureQualificationValidationError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise CaptureQualificationValidationError(f"{label} must be finite")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaptureQualificationValidationError(f"{label} must be an integer")
    if value < minimum:
        raise CaptureQualificationValidationError(
            f"{label} must be greater than or equal to {minimum}"
        )
    return value


def _exact_version(value: object, expected: int, label: str) -> int:
    version = _integer(value, label)
    if version != expected:
        raise CaptureQualificationValidationError(f"{label} must be exactly {expected}")
    return version


def _fraction(value: object, label: str) -> float:
    result = _number(value, label)
    if not 0.0 <= result <= 1.0:
        raise CaptureQualificationValidationError(f"{label} must be in [0, 1]")
    return result


def _sha256(value: object, label: str) -> str:
    digest = _string(value, label)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise CaptureQualificationValidationError(
            f"{label} must be a lowercase hexadecimal SHA-256 digest"
        )
    return digest


def _utc_timestamp(value: object, label: str) -> str:
    timestamp = _string(value, label)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CaptureQualificationValidationError(
            f"{label} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise CaptureQualificationValidationError(f"{label} must be in UTC")
    return timestamp


def _parse_review_items(
    value: object, expected_names: tuple[str, ...], label: str
) -> tuple[tuple[str, CaptureReviewItem], ...]:
    items = _mapping(value, label)
    _exact_keys(items, expected_names, label)
    parsed: list[tuple[str, CaptureReviewItem]] = []
    valid_statuses = {"confirmed", "not_confirmed", "not_applicable", "uncertain"}
    for name in expected_names:
        item = _mapping(items[name], f"{label}.{name}")
        _exact_keys(item, ("status", "note"), f"{label}.{name}")
        status = _string(item["status"], f"{label}.{name}.status")
        if status not in valid_statuses:
            raise CaptureQualificationValidationError(
                f"{label}.{name}.status must be exactly one of {sorted(valid_statuses)}"
            )
        note = _string(item["note"], f"{label}.{name}.note", allow_empty=True)
        if status != "confirmed" and not note.strip():
            raise CaptureQualificationValidationError(
                f"{label}.{name}.note is required unless status is confirmed"
            )
        parsed.append((name, CaptureReviewItem(cast(ReviewStatus, status), note)))
    return tuple(parsed)


def parse_capture_review(payload: Mapping[str, object]) -> CaptureReview:
    """Strictly parse the canonical external capture-review JSON object.

    Aliased field or item names and unknown keys are rejected rather than
    silently normalized.
    """
    review = _mapping(payload, "capture_review")
    _exact_keys(
        review,
        (
            "schema_version",
            "review_id",
            "reviewed_at_utc",
            "reviewer",
            "whole_source_video_inspected",
            "whole_annotated_com_video_inspected",
            "capture_protocol",
            "artifact_hashes",
            "walking_direction",
            "declared_direction_matches_inherited_step4",
            "orientation_notes",
            "capture_items",
            "annotated_com_items",
        ),
        "capture_review",
    )
    schema_version = _exact_version(
        review["schema_version"],
        CAPTURE_QUALIFICATION_SCHEMA_VERSION,
        "capture_review.schema_version",
    )
    reviewer_data = _mapping(review["reviewer"], "capture_review.reviewer")
    _exact_keys(
        reviewer_data,
        ("reviewer_type", "identifier", "role", "independence"),
        "capture_review.reviewer",
    )
    reviewer_type = _string(
        reviewer_data["reviewer_type"], "capture_review.reviewer.reviewer_type"
    )
    if reviewer_type not in {"human", "automated_assistant"}:
        raise CaptureQualificationValidationError(
            "capture_review.reviewer.reviewer_type must be exactly human or "
            "automated_assistant"
        )
    independence = _string(
        reviewer_data["independence"], "capture_review.reviewer.independence"
    )
    if independence not in {"independent", "not_independent", "uncertain"}:
        raise CaptureQualificationValidationError(
            "capture_review.reviewer.independence must be exactly independent, "
            "not_independent, or uncertain"
        )
    hashes = _mapping(review["artifact_hashes"], "capture_review.artifact_hashes")
    _exact_keys(
        hashes,
        ("source_video", "annotated_com.mp4", STEP5B_QUALIFICATION_OUTPUT_NAME),
        "capture_review.artifact_hashes",
    )
    direction = _string(review["walking_direction"], "capture_review.walking_direction")
    if direction not in {"image_left", "image_right"}:
        raise CaptureQualificationValidationError(
            "capture_review.walking_direction must be exactly image_left or image_right"
        )
    return CaptureReview(
        schema_version=schema_version,
        review_id=_string(review["review_id"], "capture_review.review_id"),
        reviewed_at_utc=_utc_timestamp(
            review["reviewed_at_utc"], "capture_review.reviewed_at_utc"
        ),
        reviewer=CaptureReviewer(
            reviewer_type=cast(ReviewerType, reviewer_type),
            identifier=_string(
                reviewer_data["identifier"], "capture_review.reviewer.identifier"
            ),
            role=_string(reviewer_data["role"], "capture_review.reviewer.role"),
            independence=cast(ReviewIndependence, independence),
        ),
        whole_source_video_inspected=_boolean(
            review["whole_source_video_inspected"],
            "capture_review.whole_source_video_inspected",
        ),
        whole_annotated_com_video_inspected=_boolean(
            review["whole_annotated_com_video_inspected"],
            "capture_review.whole_annotated_com_video_inspected",
        ),
        capture_protocol=_string(
            review["capture_protocol"], "capture_review.capture_protocol"
        ),
        source_video_sha256=_sha256(
            hashes["source_video"], "capture_review.artifact_hashes.source_video"
        ),
        annotated_com_sha256=_sha256(
            hashes["annotated_com.mp4"],
            "capture_review.artifact_hashes.annotated_com.mp4",
        ),
        current_qualification_sha256=_sha256(
            hashes[STEP5B_QUALIFICATION_OUTPUT_NAME],
            f"capture_review.artifact_hashes.{STEP5B_QUALIFICATION_OUTPUT_NAME}",
        ),
        walking_direction=cast(Literal["image_left", "image_right"], direction),
        declared_direction_matches_inherited_step4=_boolean(
            review["declared_direction_matches_inherited_step4"],
            "capture_review.declared_direction_matches_inherited_step4",
        ),
        orientation_notes=_string(
            review["orientation_notes"], "capture_review.orientation_notes"
        ),
        capture_items=_parse_review_items(
            review["capture_items"], CAPTURE_ITEM_NAMES, "capture_review.capture_items"
        ),
        annotated_com_items=_parse_review_items(
            review["annotated_com_items"],
            ANNOTATED_ITEM_NAMES,
            "capture_review.annotated_com_items",
        ),
    )


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CaptureQualificationValidationError(f"{label} must be an array")
    return cast(Sequence[object], value)


def _threshold_entry(
    sensitivity: Mapping[str, object], threshold: float
) -> Mapping[str, object]:
    entries = _sequence(
        sensitivity.get("threshold_sensitivity"),
        "sensitivity_results.threshold_sensitivity",
    )
    matches = []
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"threshold_sensitivity[{index}]")
        if (
            _number(entry.get("threshold"), f"threshold_sensitivity[{index}].threshold")
            == threshold
        ):
            matches.append(entry)
    if len(matches) != 1:
        raise CaptureQualificationValidationError(
            f"sensitivity_results must contain exactly one threshold {threshold:.2f}"
        )
    return matches[0]


def extract_step5c_evidence(qualification: Mapping[str, object]) -> Step5cEvidence:
    """Extract only frozen Step 5c evidence from a Step 5b JSON object."""
    sensitivity = _mapping(
        qualification.get("sensitivity_results"), "sensitivity_results"
    )
    segments = _sequence(
        sensitivity.get("segment_summaries"), "sensitivity_results.segment_summaries"
    )
    persistent: list[str] = []
    transient: list[str] = []
    parsed_segment_names: list[str] = []
    for index, raw_segment in enumerate(segments):
        segment = _mapping(raw_segment, f"segment_summaries[{index}]")
        segment_name = _string(
            segment.get("segment_name"), f"segment_summaries[{index}].segment_name"
        )
        parsed_segment_names.append(segment_name)
        supported = segment.get("is_supported")
        if not isinstance(supported, bool):
            raise CaptureQualificationValidationError(
                f"segment_summaries[{index}].is_supported must be a boolean"
            )
        pattern = _string(
            segment.get("missingness_pattern"),
            f"segment_summaries[{index}].missingness_pattern",
        )
        if pattern not in {
            "persistent",
            "intermittent",
            "none",
            "structurally_unsupported",
        }:
            raise CaptureQualificationValidationError(
                f"segment_summaries[{index}].missingness_pattern is invalid"
            )
        structurally_unsupported = segment_name in UNSUPPORTED_SEGMENTS
        if (
            supported == structurally_unsupported
            or (pattern == "structurally_unsupported") != structurally_unsupported
        ):
            raise CaptureQualificationValidationError(
                f"segment_summaries[{index}] support and structural missingness "
                "semantics are inconsistent"
            )
        if supported and pattern == "persistent":
            persistent.append(segment_name)
        elif supported and pattern == "intermittent":
            transient.append(segment_name)
    if tuple(parsed_segment_names) != SEGMENT_NAMES:
        raise CaptureQualificationValidationError(
            "sensitivity_results.segment_summaries must contain the exact canonical "
            f"segment names in order: {list(SEGMENT_NAMES)}"
        )

    strides = _sequence(
        sensitivity.get("stride_summaries"), "sensitivity_results.stride_summaries"
    )
    complete = 0
    complete_by_side = {"left": 0, "right": 0}
    for index, raw_stride in enumerate(strides):
        stride = _mapping(raw_stride, f"stride_summaries[{index}]")
        side = _string(stride.get("side"), f"stride_summaries[{index}].side")
        if side not in complete_by_side:
            raise CaptureQualificationValidationError(
                f"stride_summaries[{index}].side must be left or right"
            )
        category = _string(
            stride.get("qualification_category"),
            f"stride_summaries[{index}].qualification_category",
        )
        if category not in _STRIDE_QUALIFICATION_CATEGORIES:
            raise CaptureQualificationValidationError(
                f"stride_summaries[{index}].qualification_category must be one of "
                f"{sorted(_STRIDE_QUALIFICATION_CATEGORIES)}"
            )
        if category == "policy_complete_at_threshold":
            complete += 1
            complete_by_side[side] += 1

    threshold_entries = _sequence(
        sensitivity.get("threshold_sensitivity"),
        "sensitivity_results.threshold_sensitivity",
    )
    parsed_thresholds = tuple(
        _number(
            _mapping(entry, f"threshold_sensitivity[{index}]").get("threshold"),
            f"threshold_sensitivity[{index}].threshold",
        )
        for index, entry in enumerate(threshold_entries)
    )
    if parsed_thresholds != FROZEN_COVERAGE_GRID:
        raise CaptureQualificationValidationError(
            "sensitivity_results.threshold_sensitivity must contain the exact "
            f"frozen grid {list(FROZEN_COVERAGE_GRID)} in order"
        )
    for threshold in FROZEN_COVERAGE_GRID:
        entry = _threshold_entry(sensitivity, threshold)
        total = _integer(
            entry.get("total_frames"), f"threshold {threshold}.total_frames"
        )
        usable = _integer(
            entry.get("usable_frames"), f"threshold {threshold}.usable_frames"
        )
        usable_fraction = _fraction(
            entry.get("usable_fraction"), f"threshold {threshold}.usable_fraction"
        )
        longest_frames = _integer(
            entry.get("longest_usable_interval_frames"),
            f"threshold {threshold}.longest_usable_interval_frames",
        )
        duration = _number(
            entry.get("longest_usable_interval_seconds"),
            f"threshold {threshold}.longest_usable_interval_seconds",
        )
        total_strides = _integer(
            entry.get("total_strides"), f"threshold {threshold}.total_strides"
        )
        strides_with_any = _integer(
            entry.get("strides_with_any_usable"),
            f"threshold {threshold}.strides_with_any_usable",
        )
        policy_complete = _integer(
            entry.get("policy_complete_strides"),
            f"threshold {threshold}.policy_complete_strides",
        )
        normalized_total_entry = _integer(
            entry.get("normalized_total_samples"),
            f"threshold {threshold}.normalized_total_samples",
        )
        normalized_exact_entry = _integer(
            entry.get("normalized_exact_match_usable"),
            f"threshold {threshold}.normalized_exact_match_usable",
        )
        normalized_interpolated_entry = _integer(
            entry.get("normalized_linear_interpolation_usable"),
            f"threshold {threshold}.normalized_linear_interpolation_usable",
        )
        normalized_usable_entry = _integer(
            entry.get("normalized_usable_samples"),
            f"threshold {threshold}.normalized_usable_samples",
        )
        normalized_fraction_entry = _fraction(
            entry.get("normalized_usable_fraction"),
            f"threshold {threshold}.normalized_usable_fraction",
        )
        expected_usable_fraction = usable / total if total else 0.0
        expected_normalized_fraction = (
            normalized_usable_entry / normalized_total_entry
            if normalized_total_entry
            else 0.0
        )
        if not (
            usable <= total
            and longest_frames <= usable
            and duration >= 0.0
            and strides_with_any <= total_strides
            and policy_complete <= total_strides
            and normalized_exact_entry + normalized_interpolated_entry
            == normalized_usable_entry
            and normalized_usable_entry <= normalized_total_entry
            and math.isclose(usable_fraction, expected_usable_fraction, abs_tol=1e-12)
            and math.isclose(
                normalized_fraction_entry,
                expected_normalized_fraction,
                abs_tol=1e-12,
            )
        ):
            raise CaptureQualificationValidationError(
                f"threshold {threshold} counts, durations, or fractions are "
                "inconsistent"
            )

    reviewed_count = len(strides)
    primary_entry = _threshold_entry(sensitivity, PRIMARY_THRESHOLD)
    total_frames = _integer(
        sensitivity.get("total_frames"), "sensitivity_results.total_frames"
    )
    finite_com_frames = _integer(
        sensitivity.get("finite_com_frames"),
        "sensitivity_results.finite_com_frames",
    )
    usable_frames_primary = _integer(
        sensitivity.get("usable_frames_primary"),
        "sensitivity_results.usable_frames_primary",
    )
    primary_total_frames = _integer(
        primary_entry.get("total_frames"),
        "threshold_sensitivity[0.90].total_frames",
    )
    primary_usable_frames = _integer(
        primary_entry.get("usable_frames"),
        "threshold_sensitivity[0.90].usable_frames",
    )
    longest_interval_frames = _integer(
        primary_entry.get("longest_usable_interval_frames"),
        "threshold_sensitivity[0.90].longest_usable_interval_frames",
    )
    normalized_total = _integer(
        sensitivity.get("normalized_total_samples"),
        "sensitivity_results.normalized_total_samples",
    )
    normalized_exact = _integer(
        sensitivity.get("normalized_exact_match_usable"),
        "sensitivity_results.normalized_exact_match_usable",
    )
    normalized_interpolated = _integer(
        sensitivity.get("normalized_linear_interpolation_usable"),
        "sensitivity_results.normalized_linear_interpolation_usable",
    )
    normalized_usable = _integer(
        sensitivity.get("normalized_usable_samples"),
        "sensitivity_results.normalized_usable_samples",
    )
    finite_com_fraction = _fraction(
        sensitivity.get("finite_com_fraction"),
        "sensitivity_results.finite_com_fraction",
    )
    primary_eligible_fraction = _fraction(
        sensitivity.get("usable_fraction_primary"),
        "sensitivity_results.usable_fraction_primary",
    )
    normalized_usable_fraction = _fraction(
        sensitivity.get("normalized_usable_fraction"),
        "sensitivity_results.normalized_usable_fraction",
    )
    theoretical_supported = _fraction(
        sensitivity.get("theoretical_supported_mass_fraction"),
        "sensitivity_results.theoretical_supported_mass_fraction",
    )
    if theoretical_supported <= 0.0:
        raise CaptureQualificationValidationError(
            "sensitivity_results.theoretical_supported_mass_fraction must be positive"
        )
    empirical_max = _fraction(
        sensitivity.get("empirical_max_mass_coverage"),
        "sensitivity_results.empirical_max_mass_coverage",
    )
    mass_coverage_max = _fraction(
        sensitivity.get("mass_coverage_max"),
        "sensitivity_results.mass_coverage_max",
    )
    expected_finite_fraction = finite_com_frames / total_frames if total_frames else 0.0
    expected_primary_fraction = (
        usable_frames_primary / total_frames if total_frames else 0.0
    )
    expected_normalized_fraction = (
        normalized_usable / normalized_total if normalized_total else 0.0
    )
    if not (
        finite_com_frames <= total_frames
        and usable_frames_primary <= finite_com_frames
        and primary_total_frames == total_frames
        and primary_usable_frames == usable_frames_primary
        and longest_interval_frames <= usable_frames_primary
        and normalized_exact + normalized_interpolated == normalized_usable
        and normalized_usable <= normalized_total
        and _integer(
            primary_entry.get("total_strides"),
            "threshold_sensitivity[0.90].total_strides",
        )
        == reviewed_count
        and _integer(
            primary_entry.get("policy_complete_strides"),
            "threshold_sensitivity[0.90].policy_complete_strides",
        )
        == complete
        and _integer(
            primary_entry.get("normalized_total_samples"),
            "threshold_sensitivity[0.90].normalized_total_samples",
        )
        == normalized_total
        and _integer(
            primary_entry.get("normalized_exact_match_usable"),
            "threshold_sensitivity[0.90].normalized_exact_match_usable",
        )
        == normalized_exact
        and _integer(
            primary_entry.get("normalized_linear_interpolation_usable"),
            "threshold_sensitivity[0.90].normalized_linear_interpolation_usable",
        )
        == normalized_interpolated
        and _integer(
            primary_entry.get("normalized_usable_samples"),
            "threshold_sensitivity[0.90].normalized_usable_samples",
        )
        == normalized_usable
        and math.isclose(empirical_max, mass_coverage_max, abs_tol=1e-12)
        and math.isclose(finite_com_fraction, expected_finite_fraction, abs_tol=1e-12)
        and math.isclose(
            primary_eligible_fraction, expected_primary_fraction, abs_tol=1e-12
        )
        and math.isclose(
            normalized_usable_fraction, expected_normalized_fraction, abs_tol=1e-12
        )
    ):
        raise CaptureQualificationValidationError(
            "sensitivity_results frame and normalized counts are inconsistent"
        )
    return Step5cEvidence(
        finite_com_fraction=finite_com_fraction,
        primary_eligible_fraction=primary_eligible_fraction,
        primary_longest_usable_interval_seconds=_number(
            primary_entry.get("longest_usable_interval_seconds"),
            "threshold_sensitivity[0.90].longest_usable_interval_seconds",
        ),
        persistent_supported_segments=tuple(sorted(persistent)),
        reviewed_candidate_stride_count=reviewed_count,
        policy_complete_stride_count=complete,
        policy_complete_stride_fraction=(
            complete / reviewed_count if reviewed_count else 0.0
        ),
        policy_complete_left_count=complete_by_side["left"],
        policy_complete_right_count=complete_by_side["right"],
        normalized_usable_fraction=normalized_usable_fraction,
        empirical_max_mass_coverage=empirical_max,
        theoretical_supported_mass_fraction=theoretical_supported,
        total_frames=total_frames,
        finite_com_frames=finite_com_frames,
        primary_eligible_frame_count=usable_frames_primary,
        primary_longest_usable_interval_frames=longest_interval_frames,
        normalized_total_samples=normalized_total,
        normalized_exact_match_usable=normalized_exact,
        normalized_linear_interpolation_usable=normalized_interpolated,
        normalized_usable_samples=normalized_usable,
        transient_supported_segments=tuple(sorted(transient)),
    )


def evaluate_engineering_readiness(
    current: Step5cEvidence,
    prior: Step5cEvidence,
    review: CaptureReview | None,
    *,
    comparable: bool = True,
    provenance_valid: bool = True,
) -> EngineeringReadinessEvaluation:
    """Apply the frozen Step 5c GO/CONDITIONAL/NO-GO policy.

    ``provenance_valid`` is an upstream-validation flag for callers that already
    performed artifact hash and lineage validation; it is not recomputed here.
    """
    if not isinstance(current, Step5cEvidence) or not isinstance(prior, Step5cEvidence):
        raise TypeError("current and prior must be Step5cEvidence instances")
    if review is not None and not isinstance(review, CaptureReview):
        raise TypeError("review must be CaptureReview or None")

    improvement = (
        current.empirical_max_mass_coverage - prior.empirical_max_mass_coverage
    )
    human_complete = bool(
        review is not None
        and review.reviewer.reviewer_type == "human"
        and review.reviewer.independence == "independent"
        and review.whole_source_video_inspected
        and review.whole_annotated_com_video_inspected
    )
    all_confirmed = bool(
        review is not None
        and all(item.status == "confirmed" for _, item in review.all_items())
    )
    if review is None or review.reviewer.reviewer_type != "human":
        external_human_review_state: ExternalHumanReviewState = "pending"
    elif human_complete and all_confirmed:
        external_human_review_state = "confirmed"
    else:
        external_human_review_state = "incomplete"
    criteria = (
        CriterionResult(
            "finite_com_fraction",
            current.finite_com_fraction >= 0.95,
            current.finite_com_fraction,
            ">= 0.95",
            "go_required",
        ),
        CriterionResult(
            "primary_eligible_fraction",
            current.primary_eligible_fraction >= 0.90,
            current.primary_eligible_fraction,
            ">= 0.90 at absolute mass coverage 0.90",
            "go_required",
        ),
        CriterionResult(
            "primary_longest_usable_interval",
            current.primary_longest_usable_interval_seconds >= 3.0,
            current.primary_longest_usable_interval_seconds,
            ">= 3 nominal seconds at absolute mass coverage 0.90",
            "go_required",
        ),
        CriterionResult(
            "no_persistent_supported_segments",
            not current.persistent_supported_segments,
            list(current.persistent_supported_segments),
            "none",
            "hard_no_go",
        ),
        CriterionResult(
            "reviewed_candidate_strides",
            current.reviewed_candidate_stride_count >= 3,
            current.reviewed_candidate_stride_count,
            ">= 3",
            "go_required",
        ),
        CriterionResult(
            "policy_complete_stride_count",
            current.policy_complete_stride_count >= 3,
            current.policy_complete_stride_count,
            ">= 3 (hard minimum 2)",
            "go_required",
        ),
        CriterionResult(
            "policy_complete_stride_fraction",
            current.policy_complete_stride_fraction >= 0.75,
            current.policy_complete_stride_fraction,
            ">= 0.75",
            "go_required",
        ),
        CriterionResult(
            "bilateral_policy_complete_strides",
            current.policy_complete_left_count >= 1
            and current.policy_complete_right_count >= 1,
            {
                "left": current.policy_complete_left_count,
                "right": current.policy_complete_right_count,
            },
            ">= 1 each side",
            "go_required",
        ),
        CriterionResult(
            "normalized_usable_fraction",
            current.normalized_usable_fraction >= 0.90,
            current.normalized_usable_fraction,
            ">= 0.90",
            "go_required",
        ),
        CriterionResult(
            "empirical_max_mass_coverage",
            current.empirical_max_mass_coverage >= 0.90,
            current.empirical_max_mass_coverage,
            ">= 0.90 absolute represented body-mass fraction",
            "go_required",
        ),
        CriterionResult(
            "prior_comparable", comparable, comparable, "true", "go_required"
        ),
        CriterionResult(
            "max_coverage_improvement",
            comparable and improvement >= 0.01,
            improvement,
            ">= 0.01 absolute for a comparable pair",
            "go_required",
        ),
        CriterionResult(
            "independent_human_whole_video_review",
            human_complete,
            human_complete,
            "independent human inspected whole source and annotation",
            "hard_no_go",
        ),
        CriterionResult(
            "all_review_items_confirmed",
            all_confirmed,
            all_confirmed,
            "all required items confirmed",
            "go_required",
        ),
        CriterionResult(
            "declared_direction_matches_inherited_step4",
            bool(
                review is not None and review.declared_direction_matches_inherited_step4
            ),
            review.declared_direction_matches_inherited_step4
            if review is not None
            else None,
            "declared review direction matches inherited Step4 direction",
            "hard_no_go",
        ),
        CriterionResult(
            "upstream_provenance_valid",
            provenance_valid,
            provenance_valid,
            "all hashes and inherited lineage match",
            "hard_no_go",
        ),
    )

    blockers: list[str] = []
    warnings: list[str] = []
    pending = not human_complete
    if review is None or review.reviewer.reviewer_type != "human":
        blockers.append("external independent human whole-video review is missing")
    elif not human_complete:
        blockers.append("human review is incomplete or not independent")

    if review is not None:
        if not review.declared_direction_matches_inherited_step4:
            blockers.append(
                "declared review direction does not match inherited Step4 direction"
            )
        for name, item in review.all_items():
            if item.status in {"not_confirmed", "not_applicable"}:
                blockers.append(f"review item {name} is {item.status}")
            elif item.status == "uncertain":
                warnings.append(f"review item {name} is uncertain: {item.note}")

    if current.persistent_supported_segments:
        blockers.append(
            "persistent supported segments: "
            + ", ".join(current.persistent_supported_segments)
        )
    if current.primary_eligible_fraction < 0.50:
        blockers.append("primary eligible fraction is below hard minimum 0.50")
    if current.policy_complete_stride_count < 2:
        blockers.append("fewer than 2 policy-complete strides")
    if current.normalized_usable_fraction < 0.50:
        blockers.append("normalized availability is below hard minimum 0.50")
    if not provenance_valid:
        blockers.append("provenance mismatch")

    failed_go = [criterion.name for criterion in criteria if not criterion.passed]
    if blockers:
        decision: EngineeringDecision = "NO-GO"
    elif failed_go:
        decision = "CONDITIONAL"
        warnings.extend(f"GO criterion not met: {name}" for name in failed_go)
    else:
        decision = "GO"
    return EngineeringReadinessEvaluation(
        decision=decision,
        state="pending_external_human_review" if pending else "evaluated",
        evaluation_state="evaluated",
        external_human_review_state=external_human_review_state,
        criteria=criteria,
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _load_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureQualificationValidationError(
            f"could not read {label} at {path}: {exc}"
        ) from exc
    return _mapping(raw, label)


def _entry_path_and_hash(
    entry_value: object, label: str, expected_basename: str | None = None
) -> tuple[Path, str]:
    entry = _mapping(entry_value, label)
    path = Path(_string(entry.get("path"), f"{label}.path")).expanduser().resolve()
    digest = _sha256(entry.get("sha256"), f"{label}.sha256")
    if expected_basename is not None and path.name != expected_basename:
        raise CaptureQualificationValidationError(
            f"{label}.path must have canonical basename {expected_basename!r}"
        )
    if not path.is_file():
        raise CaptureQualificationValidationError(
            f"referenced artifact is missing: {path}"
        )
    if sha256_file(path) != digest:
        raise CaptureQualificationValidationError(
            f"{label}.sha256 does not match {path}"
        )
    return path, digest


@dataclass(frozen=True, slots=True)
class _ValidatedQualification:
    path: Path
    digest: str
    payload: Mapping[str, object]
    evidence: Step5cEvidence
    source_path: Path
    source_hash: str
    annotated_path: Path
    annotated_hash: str
    stride_qc_path: Path
    stride_qc_hash: str
    preprocessing_path: Path
    preprocessing_hash: str
    review_resolution_path: Path
    review_resolution_hash: str
    pose_model_hash: str
    preprocessing_config: Mapping[str, object]
    capture_protocol_declaration: Mapping[str, object]
    walking_direction: str


def _validate_qualification(path: Path, label: str) -> _ValidatedQualification:
    payload = _load_json(path, label)
    _exact_version(
        payload.get("schema_version"),
        COM_QUALIFICATION_SCHEMA_VERSION,
        f"{label}.schema_version",
    )
    if payload.get("algorithm_version") != COM_QUALIFICATION_ALGORITHM_VERSION:
        raise CaptureQualificationValidationError(
            f"{label}.algorithm_version must be {COM_QUALIFICATION_ALGORITHM_VERSION!r}"
        )
    config = _mapping(payload.get("config"), f"{label}.config")
    if (
        _number(config.get("primary_threshold"), f"{label}.config.primary_threshold")
        != PRIMARY_THRESHOLD
    ):
        raise CaptureQualificationValidationError(
            f"{label} primary threshold must be exactly {PRIMARY_THRESHOLD}"
        )
    grid = tuple(
        _number(value, f"{label}.config.coverage_thresholds")
        for value in _sequence(
            config.get("coverage_thresholds"), f"{label}.config.coverage_thresholds"
        )
    )
    if grid != FROZEN_COVERAGE_GRID:
        raise CaptureQualificationValidationError(
            f"{label} coverage grid must be exactly {list(FROZEN_COVERAGE_GRID)}"
        )
    step5a = _mapping(payload.get("step5a"), f"{label}.step5a")
    _exact_version(
        step5a.get("schema_version"),
        COM_SCHEMA_VERSION,
        f"{label}.step5a.schema_version",
    )
    if step5a.get("algorithm_version") != COM_ALGORITHM_VERSION:
        raise CaptureQualificationValidationError(
            f"{label}.step5a.algorithm_version must be {COM_ALGORITHM_VERSION!r}"
        )
    sex = _string(payload.get("sex"), f"{label}.sex")
    if sex not in {"male", "female"}:
        raise CaptureQualificationValidationError(f"{label}.sex must be male or female")
    step5a_config = _mapping(step5a.get("config"), f"{label}.step5a.config")
    if step5a_config.get("anthropometry_sex") != sex:
        raise CaptureQualificationValidationError(
            f"{label}.step5a.config.anthropometry_sex must match sex"
        )

    inputs = _mapping(payload.get("inputs"), f"{label}.inputs")
    outputs = _mapping(payload.get("outputs"), f"{label}.outputs")
    source_path, source_hash = _entry_path_and_hash(
        inputs.get("source_video"), f"{label}.inputs.source_video"
    )
    annotated_path, annotated_hash = _entry_path_and_hash(
        outputs.get("annotated_com.mp4"),
        f"{label}.outputs.annotated_com.mp4",
        "annotated_com.mp4",
    )
    stride_qc_path, stride_qc_hash = _entry_path_and_hash(
        outputs.get("com_stride_qc.csv"),
        f"{label}.outputs.com_stride_qc.csv",
        "com_stride_qc.csv",
    )
    preprocessing_path, preprocessing_hash = _entry_path_and_hash(
        inputs.get("step5_upstream.preprocessing_metadata.json"),
        f"{label}.inputs.step5_upstream.preprocessing_metadata.json",
        "preprocessing_metadata.json",
    )
    review_resolution_path, review_resolution_hash = _entry_path_and_hash(
        inputs.get("step5_upstream.review_resolution_metadata.json"),
        f"{label}.inputs.step5_upstream.review_resolution_metadata.json",
        "review_resolution_metadata.json",
    )

    preprocessing = _load_json(preprocessing_path, f"{label} preprocessing metadata")
    try:
        _exact_version(
            preprocessing.get("schema_version"),
            PREPROCESSING_SCHEMA_VERSION,
            f"{label}.preprocessing.schema_version",
        )
    except CaptureQualificationValidationError as exc:
        raise CaptureQualificationValidationError(
            f"{label} referenced preprocessing metadata version is invalid"
        ) from exc
    if preprocessing.get("algorithm_version") != PREPROCESSING_ALGORITHM_VERSION:
        raise CaptureQualificationValidationError(
            f"{label} referenced preprocessing metadata version is invalid"
        )
    inherited = _mapping(
        preprocessing.get("inherited_provenance"),
        f"{label}.preprocessing.inherited_provenance",
    )
    backend = _mapping(
        inherited.get("backend"), f"{label}.preprocessing.inherited_provenance.backend"
    )
    model_hash = _sha256(
        backend.get("model_sha256"),
        f"{label}.preprocessing.inherited_provenance.backend.model_sha256",
    )
    preprocessing_config = _mapping(
        preprocessing.get("config"), f"{label}.preprocessing.config"
    )
    capture_protocol_declaration = _mapping(
        inherited.get("capture_assumptions"),
        f"{label}.preprocessing.inherited_provenance.capture_assumptions",
    )
    inherited_source = _mapping(
        inherited.get("source"),
        f"{label}.preprocessing.inherited_provenance.source",
    )
    if (
        _sha256(
            inherited_source.get("sha256"),
            f"{label}.preprocessing.inherited_provenance.source.sha256",
        )
        != source_hash
    ):
        raise CaptureQualificationValidationError(
            f"{label} source hash does not match preprocessing provenance"
        )

    resolution = _load_json(
        review_resolution_path, f"{label} review-resolution metadata"
    )
    try:
        _exact_version(
            resolution.get("schema_version"),
            REVIEW_RESOLUTION_SCHEMA_VERSION,
            f"{label}.review_resolution.schema_version",
        )
    except CaptureQualificationValidationError as exc:
        raise CaptureQualificationValidationError(
            f"{label} referenced review-resolution metadata version is invalid"
        ) from exc
    if resolution.get("algorithm_version") != REVIEW_RESOLUTION_ALGORITHM_VERSION:
        raise CaptureQualificationValidationError(
            f"{label} referenced review-resolution metadata version is invalid"
        )
    resolution_inputs = _mapping(
        resolution.get("inputs"), f"{label}.review_resolution.inputs"
    )
    resolution_preprocessing = _mapping(
        resolution_inputs.get("preprocessing_metadata.json"),
        f"{label}.review_resolution.inputs.preprocessing_metadata.json",
    )
    if (
        _sha256(
            resolution_preprocessing.get("sha256"),
            f"{label}.review_resolution.inputs.preprocessing_metadata.json.sha256",
        )
        != preprocessing_hash
    ):
        raise CaptureQualificationValidationError(
            f"{label} review-resolution preprocessing hash does not match"
        )
    source_step4 = _mapping(
        resolution.get("source_step4"), f"{label}.review_resolution.source_step4"
    )
    event_config = _mapping(
        source_step4.get("gait_event_config"),
        f"{label}.review_resolution.source_step4.gait_event_config",
    )
    direction = _string(
        event_config.get("direction"),
        f"{label}.review_resolution.source_step4.gait_event_config.direction",
    )
    if direction not in {"image_left", "image_right"}:
        raise CaptureQualificationValidationError(
            f"{label} inherited Step4 direction must be image_left or image_right"
        )
    return _ValidatedQualification(
        path=path,
        digest=sha256_file(path),
        payload=payload,
        evidence=extract_step5c_evidence(payload),
        source_path=source_path,
        source_hash=source_hash,
        annotated_path=annotated_path,
        annotated_hash=annotated_hash,
        stride_qc_path=stride_qc_path,
        stride_qc_hash=stride_qc_hash,
        preprocessing_path=preprocessing_path,
        preprocessing_hash=preprocessing_hash,
        review_resolution_path=review_resolution_path,
        review_resolution_hash=review_resolution_hash,
        pose_model_hash=model_hash,
        preprocessing_config=preprocessing_config,
        capture_protocol_declaration=capture_protocol_declaration,
        walking_direction=direction,
    )


def _comparability(
    current: _ValidatedQualification,
    prior: _ValidatedQualification,
) -> tuple[bool, dict[str, object]]:
    checks = {
        "step5a_schema": True,
        "step5a_algorithm": True,
        "step5b_schema": True,
        "step5b_algorithm": True,
        "sex": current.payload.get("sex") == prior.payload.get("sex"),
        "pose_model_sha256": current.pose_model_hash == prior.pose_model_hash,
        "preprocessing_config": dict(current.preprocessing_config)
        == dict(prior.preprocessing_config),
        "primary_gate": True,
        "sensitivity_grid": True,
        "capture_protocol_declaration": dict(current.capture_protocol_declaration)
        == dict(prior.capture_protocol_declaration),
    }
    comparable = all(value is True for value in checks.values())
    return comparable, {
        "comparable": comparable,
        "checks": checks,
        "current_capture_protocol_declaration": dict(
            current.capture_protocol_declaration
        ),
        "prior_capture_protocol_declaration": dict(prior.capture_protocol_declaration),
    }


def _review_json(review: CaptureReview) -> dict[str, object]:
    return {
        "schema_version": review.schema_version,
        "review_id": review.review_id,
        "reviewed_at_utc": review.reviewed_at_utc,
        "reviewer": asdict(review.reviewer),
        "whole_source_video_inspected": review.whole_source_video_inspected,
        "whole_annotated_com_video_inspected": (
            review.whole_annotated_com_video_inspected
        ),
        "capture_protocol": review.capture_protocol,
        "artifact_hashes": {
            "source_video": review.source_video_sha256,
            "annotated_com.mp4": review.annotated_com_sha256,
            STEP5B_QUALIFICATION_OUTPUT_NAME: review.current_qualification_sha256,
        },
        "walking_direction": review.walking_direction,
        "declared_direction_matches_inherited_step4": (
            review.declared_direction_matches_inherited_step4
        ),
        "orientation_notes": review.orientation_notes,
        "capture_items": {name: asdict(item) for name, item in review.capture_items},
        "annotated_com_items": {
            name: asdict(item) for name, item in review.annotated_com_items
        },
    }


def _runtime() -> dict[str, object]:
    dependencies: dict[str, str] = {}
    for name in ("gait-stability",):
        with suppress(importlib.metadata.PackageNotFoundError):
            dependencies[name] = importlib.metadata.version(name)
    git: dict[str, object] = {"available": False}
    # Git metadata is intentionally best-effort so provenance capture does not
    # make qualification depend on running inside an available Git worktree.
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        )
        git = {"available": True, "revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python_version": sys.version,
        "dependency_versions": dependencies,
        "git": git,
        "randomness": "none",
    }


def _ensure_distinct(paths: Sequence[Path]) -> None:
    resolved: set[Path] = set()
    for path in paths:
        canonical = path.resolve()
        if canonical in resolved:
            raise CaptureQualificationValidationError(
                f"input/output paths must not alias: {path}"
            )
        resolved.add(canonical)


def _recheck(snapshot: Mapping[Path, str]) -> None:
    for path, expected in snapshot.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise CaptureQualificationValidationError(
                f"consumed artifact changed before publication: {path}"
            )


def _publish_one(staged: Path, target: Path) -> None:
    backup: Path | None = None
    published = False
    try:
        if target.exists():
            backup = target.with_name(f"{target.name}.backup-{uuid.uuid4().hex}")
            target.replace(backup)
        staged.replace(target)
        published = True
    except OSError as exc:
        if published:
            with suppress(OSError):
                target.unlink()
        restore_error: OSError | None = None
        if backup is not None:
            try:
                backup.replace(target)
            except OSError as error:
                restore_error = error
        detail = f"; rollback failed: {restore_error}" if restore_error else ""
        raise ArtifactPublishError(
            f"Could not publish Step 5c capture qualification{detail}"
        ) from exc
    if backup is not None:
        with suppress(OSError):
            backup.unlink()


def qualify_clean_capture(
    artifact_directory: str | Path,
    capture_review_path: str | Path,
    prior_qualification_path: str | Path,
) -> CaptureQualificationArtifacts:
    """Validate frozen Step 5b evidence and publish Step 5c qualification."""
    directory = Path(artifact_directory).expanduser().resolve()
    if not directory.is_dir():
        raise CaptureQualificationValidationError(
            f"artifact directory does not exist: {directory}"
        )
    current_path = (directory / STEP5B_QUALIFICATION_OUTPUT_NAME).resolve()
    review_path = Path(capture_review_path).expanduser().resolve()
    prior_path = Path(prior_qualification_path).expanduser().resolve()
    output_path = (directory / CAPTURE_QUALIFICATION_OUTPUT_NAME).resolve()
    for label, path in (
        (f"current canonical {STEP5B_QUALIFICATION_OUTPUT_NAME}", current_path),
        ("prior qualification", prior_path),
    ):
        if not path.is_file():
            raise CaptureQualificationValidationError(f"{label} is missing: {path}")
    _ensure_distinct((current_path, review_path, prior_path, output_path))

    current = _validate_qualification(current_path, "current_qualification")
    prior = _validate_qualification(prior_path, "prior_qualification")
    review: CaptureReview | None = None
    review_hash: str | None = None
    if review_path.is_file():
        review_payload = _load_json(review_path, "capture_review")
        review = parse_capture_review(review_payload)
        review_hash = sha256_file(review_path)

        if review.source_video_sha256 != current.source_hash:
            raise CaptureQualificationValidationError(
                "capture-review source_video hash does not match current Step 5b source"
            )
        if review.annotated_com_sha256 != current.annotated_hash:
            raise CaptureQualificationValidationError(
                "capture-review annotated_com.mp4 hash does not match current Step 5b "
                "output"
            )
        if review.current_qualification_sha256 != current.digest:
            raise CaptureQualificationValidationError(
                "capture-review com_qualification.json hash does not match current "
                "Step 5b"
            )
        if review.walking_direction != current.walking_direction:
            raise CaptureQualificationValidationError(
                "capture-review walking direction does not match inherited Step4 "
                "direction"
            )

    comparable, comparison = _comparability(current, prior)
    evaluation = evaluate_engineering_readiness(
        current.evidence,
        prior.evidence,
        review,
        comparable=comparable,
        provenance_valid=True,
    )
    delta = (
        current.evidence.empirical_max_mass_coverage
        - prior.evidence.empirical_max_mass_coverage
    )
    current_sensitivity = _mapping(
        current.payload["sensitivity_results"],
        "current_qualification.sensitivity_results",
    )
    prior_sensitivity = _mapping(
        prior.payload["sensitivity_results"],
        "prior_qualification.sensitivity_results",
    )

    consumed_paths = [
        current.path,
        current.source_path,
        current.annotated_path,
        current.stride_qc_path,
        current.preprocessing_path,
        current.review_resolution_path,
        prior.path,
        prior.source_path,
        prior.annotated_path,
        prior.stride_qc_path,
        prior.preprocessing_path,
        prior.review_resolution_path,
    ]
    if review is not None:
        consumed_paths.append(review_path)
    _ensure_distinct((*consumed_paths, output_path))
    snapshot = {path: sha256_file(path) for path in consumed_paths}
    payload: dict[str, object] = {
        "schema_version": CAPTURE_QUALIFICATION_SCHEMA_VERSION,
        "algorithm_version": CAPTURE_QUALIFICATION_ALGORITHM_VERSION,
        "scope": (
            "Engineering/QC clean-capture qualification for exploratory pipeline "
            "readiness. Not a stability metric, diagnostic result, clinical "
            "conclusion, or validated COM measurement."
        ),
        "run_id": uuid.uuid4().hex,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "current_com_qualification.json": {
                "path": str(current.path),
                "sha256": current.digest,
            },
            "prior_qualification.json": {
                "path": str(prior.path),
                "sha256": prior.digest,
            },
            "capture_review.json": {
                "path": str(review_path),
                "sha256": review_hash,
                "status": "present" if review is not None else "missing",
            },
            "current_source_video": {
                "path": str(current.source_path),
                "sha256": current.source_hash,
            },
            "current_annotated_com.mp4": {
                "path": str(current.annotated_path),
                "sha256": current.annotated_hash,
            },
            "current_com_stride_qc.csv": {
                "path": str(current.stride_qc_path),
                "sha256": current.stride_qc_hash,
            },
            "current_preprocessing_metadata.json": {
                "path": str(current.preprocessing_path),
                "sha256": current.preprocessing_hash,
            },
            "current_review_resolution_metadata.json": {
                "path": str(current.review_resolution_path),
                "sha256": current.review_resolution_hash,
            },
            "prior_source_video": {
                "path": str(prior.source_path),
                "sha256": prior.source_hash,
            },
            "prior_annotated_com.mp4": {
                "path": str(prior.annotated_path),
                "sha256": prior.annotated_hash,
            },
            "prior_com_stride_qc.csv": {
                "path": str(prior.stride_qc_path),
                "sha256": prior.stride_qc_hash,
            },
            "prior_preprocessing_metadata.json": {
                "path": str(prior.preprocessing_path),
                "sha256": prior.preprocessing_hash,
            },
            "prior_review_resolution_metadata.json": {
                "path": str(prior.review_resolution_path),
                "sha256": prior.review_resolution_hash,
            },
        },
        "capture_review": _review_json(review) if review is not None else None,
        "inherited_step4": {
            "walking_direction": current.walking_direction,
            "declared_direction_matches_inherited_step4": (
                review.declared_direction_matches_inherited_step4
                if review is not None
                else None
            ),
        },
        "frozen_policy": {
            "primary_absolute_mass_coverage_threshold": PRIMARY_THRESHOLD,
            "sensitivity_grid": list(FROZEN_COVERAGE_GRID),
            "go_thresholds": {
                "finite_com_fraction": 0.95,
                "primary_eligible_fraction": 0.90,
                "longest_usable_interval_nominal_seconds": 3.0,
                "reviewed_candidate_strides": 3,
                "policy_complete_strides": 3,
                "policy_complete_stride_fraction": 0.75,
                "policy_complete_each_side": 1,
                "normalized_usable_fraction": 0.90,
                "empirical_max_absolute_mass_coverage": 0.90,
                "empirical_max_improvement_over_prior": 0.01,
            },
            "hard_no_go_thresholds": {
                "primary_eligible_fraction_below": 0.50,
                "policy_complete_strides_below": 2,
                "normalized_usable_fraction_below": 0.50,
                "persistent_supported_segment": True,
                "nonhuman_or_incomplete_review": True,
                "not_confirmed_or_not_applicable_review_item": True,
                "provenance_mismatch": True,
            },
            "conditional_conditions": {
                "prior_qualification_not_comparable": True,
            },
        },
        "definitions": {
            "finite_com_fraction": {
                "definition": "finite COM proxy frames / all recording frames",
                "denominator": "all recording frames",
                "units": "dimensionless fraction",
                "scope": "whole recording",
            },
            "primary_eligible_fraction": {
                "definition": (
                    "finite nonzero COM proxy frames at absolute mass coverage "
                    ">= 0.90 / all recording frames"
                ),
                "denominator": "all recording frames",
                "units": "dimensionless fraction",
                "scope": "whole recording",
            },
            "longest_usable_interval": {
                "definition": (
                    "longest contiguous primary-eligible frame run using nominal "
                    "timestamps"
                ),
                "denominator": "not applicable",
                "units": "nominal seconds",
                "scope": "whole recording",
            },
            "policy_complete_stride_fraction": {
                "definition": (
                    "policy-complete reviewed candidate strides / all reviewed "
                    "candidate strides"
                ),
                "denominator": "all reviewed candidate strides",
                "units": "dimensionless fraction",
                "scope": "all reviewed candidate strides",
            },
            "normalized_usable_fraction": {
                "definition": (
                    "usable normalized COM samples / all canonical normalized samples"
                ),
                "denominator": "all canonical normalized samples",
                "units": "dimensionless fraction",
                "scope": "all reviewed candidate strides",
            },
            "mass_coverage": {
                "definition": (
                    "unrenormalized sum of represented body-mass fractions for "
                    "usable supported segments"
                ),
                "denominator": "total modeled body mass",
                "units": "dimensionless absolute body-mass fraction",
                "scope": "whole recording",
            },
        },
        "current": {
            "evidence": asdict(current.evidence),
            "distributions": {
                key: current_sensitivity.get(key)
                for key in (
                    "mass_coverage_min",
                    "mass_coverage_max",
                    "mass_coverage_mean",
                    "mass_coverage_median",
                    "supported_mass_coverage_min",
                    "supported_mass_coverage_max",
                    "supported_mass_coverage_mean",
                    "supported_mass_coverage_median",
                )
            },
            "missing_segments": current_sensitivity.get("segment_summaries"),
            "asymmetry": current_sensitivity.get("asymmetry_summaries"),
            "stride_results": current_sensitivity.get("stride_summaries"),
            "normalized_results": {
                key: current_sensitivity.get(key)
                for key in (
                    "normalized_total_samples",
                    "normalized_exact_match_usable",
                    "normalized_linear_interpolation_usable",
                    "normalized_usable_samples",
                    "normalized_usable_fraction",
                )
            },
            "full_sensitivity_grid": current_sensitivity.get("threshold_sensitivity"),
        },
        "prior": {
            "evidence": asdict(prior.evidence),
            "distributions": {
                key: prior_sensitivity.get(key)
                for key in (
                    "mass_coverage_min",
                    "mass_coverage_max",
                    "mass_coverage_mean",
                    "mass_coverage_median",
                    "supported_mass_coverage_min",
                    "supported_mass_coverage_max",
                    "supported_mass_coverage_mean",
                    "supported_mass_coverage_median",
                )
            },
            "missing_segments": prior_sensitivity.get("segment_summaries"),
            "asymmetry": prior_sensitivity.get("asymmetry_summaries"),
            "stride_results": prior_sensitivity.get("stride_summaries"),
            "normalized_results": {
                key: prior_sensitivity.get(key)
                for key in (
                    "normalized_total_samples",
                    "normalized_exact_match_usable",
                    "normalized_linear_interpolation_usable",
                    "normalized_usable_samples",
                    "normalized_usable_fraction",
                )
            },
            "full_sensitivity_grid": prior_sensitivity.get("threshold_sensitivity"),
        },
        "comparison": comparison
        | {
            "current_empirical_max_mass_coverage": (
                current.evidence.empirical_max_mass_coverage
            ),
            "prior_empirical_max_mass_coverage": (
                prior.evidence.empirical_max_mass_coverage
            ),
            "empirical_max_mass_coverage_delta": delta,
            "direction": "current_minus_prior",
            "coverage_delta_status": (
                "interpretable_for_go" if comparable else "not_interpretable_for_go"
            ),
        },
        "criterion_results": [asdict(criterion) for criterion in evaluation.criteria],
        "blockers": list(evaluation.blockers),
        "warnings": list(evaluation.warnings),
        "engineering_readiness": {
            "decision": evaluation.decision,
            "state": evaluation.state,
            "evaluation_state": evaluation.evaluation_state,
            "external_human_review_state": (evaluation.external_human_review_state),
            "scope": "exploratory engineering use only",
        },
        "scientific_readiness": {
            "decision": "NO-GO",
            "status": "not_established",
            "reason": (
                "No independent reference-system validation establishes COM position, "
                "velocity, gait-event, or downstream stability-measurement validity."
            ),
        },
        "provenance": {
            "current_pose_model_sha256": current.pose_model_hash,
            "prior_pose_model_sha256": prior.pose_model_hash,
            "coordinate_system": (
                "normalized 2D image plane (x right, y down), no physical scale"
            ),
            "camera_view": {
                "method_required_view": (
                    "static near-sagittal weak-perspective view with minimal "
                    "out-of-plane motion"
                ),
                "external_independent_human_confirmation_established": bool(
                    review is not None
                    and review.reviewer.reviewer_type == "human"
                    and review.reviewer.independence == "independent"
                    and review.whole_source_video_inspected
                    and review.whole_annotated_com_video_inspected
                    and dict(review.capture_items)["approximately_sagittal_view"].status
                    == "confirmed"
                ),
                "machine_verified": False,
            },
            "filtering_and_smoothing": dict(current.preprocessing_config),
            "gait_events": (
                "reviewed candidate events; not reference-system validated contacts"
            ),
            "anthropometric_coefficient_sex": {
                "value": current.payload.get("sex"),
                "selection_source": "user_supplied_to_step5a",
                "inferred": False,
                "model_scope": (
                    "population-average coefficient model, not individual anthropometry"
                ),
            },
        },
        "runtime": _runtime(),
        "limitations": [
            "High coverage is not accuracy, confidence, probability, or COM validity.",
            "Pose-model landmarks and the represented-segment COM proxy are not "
            "laboratory measurements.",
            "The review is external declarative evidence and is not machine-verified "
            "by Step 5c.",
            "Normalized image-plane coordinates have no physical scale, depth, "
            "gravity alignment, or laboratory frame.",
            "Candidate gait events and normalized stride progression are not ground "
            "truth.",
            "This engineering gate is not a stability metric, fall-risk assessment, "
            "diagnostic tool, or clinical conclusion.",
        ],
        "validation_status": (
            "research proxy only; scientific and clinical validity not established"
        ),
        "outputs": {
            CAPTURE_QUALIFICATION_OUTPUT_NAME: {
                "path": str(output_path),
                "sha256": None,
                "sha256_semantics": (
                    "not self-recorded because embedding a file's own digest changes "
                    "that digest"
                ),
            }
        },
    }

    staging = Path(
        tempfile.mkdtemp(
            prefix=f"{directory.name}.capture-qualification-staging-",
            dir=directory.parent,
        )
    )
    try:
        staged = staging / CAPTURE_QUALIFICATION_OUTPUT_NAME
        staged.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        # Recheck source videos as well as derived artifacts to detect mutation
        # anywhere in the consumed lineage before atomic publication.
        _recheck(snapshot)
        _publish_one(staged, output_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if not output_path.is_file():
        raise CaptureQualificationError(
            f"capture qualification was not published: {output_path}"
        )
    return CaptureQualificationArtifacts(directory, output_path)
