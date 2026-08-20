"""Unit tests for frozen Step 5c capture-qualification policy."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from gait_stability.capture_qualification import (
    ANNOTATED_ITEM_NAMES,
    CAPTURE_ITEM_NAMES,
    CaptureQualificationValidationError,
    CaptureReview,
    Step5cEvidence,
    evaluate_engineering_readiness,
    extract_step5c_evidence,
    parse_capture_review,
)
from gait_stability.com_estimation import SEGMENT_NAMES, UNSUPPORTED_SEGMENTS

_DIGEST = "a" * 64


def _review_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "review_id": "synthetic-review-1",
        "reviewed_at_utc": "2026-08-20T12:00:00Z",
        "reviewer": {
            "reviewer_type": "human",
            "identifier": "reviewer-1",
            "role": "research reviewer",
            "independence": "independent",
        },
        "whole_source_video_inspected": True,
        "whole_annotated_com_video_inspected": True,
        "capture_protocol": "synthetic protocol declaration",
        "artifact_hashes": {
            "source_video": _DIGEST,
            "annotated_com.mp4": _DIGEST,
            "com_qualification.json": _DIGEST,
        },
        "walking_direction": "image_right",
        "declared_direction_matches_inherited_step4": True,
        "orientation_notes": "Synthetic rightward walk.",
        "capture_items": {
            name: {"status": "confirmed", "note": ""} for name in CAPTURE_ITEM_NAMES
        },
        "annotated_com_items": {
            name: {"status": "confirmed", "note": ""} for name in ANNOTATED_ITEM_NAMES
        },
    }


def _review(**reviewer_changes: object) -> CaptureReview:
    payload = _review_payload()
    payload["reviewer"].update(reviewer_changes)
    return parse_capture_review(payload)


def _evidence(**changes: object) -> Step5cEvidence:
    baseline = Step5cEvidence(
        finite_com_fraction=0.95,
        primary_eligible_fraction=0.90,
        primary_longest_usable_interval_seconds=3.0,
        persistent_supported_segments=(),
        reviewed_candidate_stride_count=4,
        policy_complete_stride_count=3,
        policy_complete_stride_fraction=0.75,
        policy_complete_left_count=1,
        policy_complete_right_count=2,
        normalized_usable_fraction=0.90,
        empirical_max_mass_coverage=0.90,
    )
    return replace(baseline, **changes)


def _qualification_payload() -> dict[str, Any]:
    grid = (0.80, 0.82, 0.84, 0.86, 0.88, 0.90)
    return {
        "sensitivity_results": {
            "total_frames": 100,
            "finite_com_frames": 96,
            "finite_com_fraction": 0.96,
            "usable_frames_primary": 91,
            "usable_fraction_primary": 0.91,
            "theoretical_supported_mass_fraction": 0.9306,
            "normalized_total_samples": 100,
            "normalized_exact_match_usable": 80,
            "normalized_linear_interpolation_usable": 12,
            "normalized_usable_samples": 92,
            "normalized_usable_fraction": 0.92,
            "empirical_max_mass_coverage": 0.93,
            "mass_coverage_max": 0.93,
            "segment_summaries": [
                {
                    "segment_name": name,
                    "is_supported": name not in UNSUPPORTED_SEGMENTS,
                    "missingness_pattern": (
                        "structurally_unsupported"
                        if name in UNSUPPORTED_SEGMENTS
                        else "persistent"
                        if name == "left_forearm"
                        else "intermittent"
                        if name == "right_forearm"
                        else "none"
                    ),
                }
                for name in SEGMENT_NAMES
            ],
            "stride_summaries": [
                {
                    "side": "left",
                    "qualification_category": "policy_complete_at_threshold",
                },
                {
                    "side": "right",
                    "qualification_category": "policy_complete_at_threshold",
                },
                {"side": "right", "qualification_category": "usable_samples_only"},
                {"side": "left", "qualification_category": "insufficient_coverage"},
            ],
            "threshold_sensitivity": [
                {
                    "threshold": threshold,
                    "total_frames": 100,
                    "usable_frames": 91
                    if threshold == 0.90
                    else round(threshold * 100),
                    "usable_fraction": 0.91 if threshold == 0.90 else threshold,
                    "longest_usable_interval_frames": round(threshold * 100),
                    "longest_usable_interval_seconds": threshold * 10,
                    "strides_with_any_usable": 4,
                    "total_strides": 4,
                    "policy_complete_strides": 2,
                    "normalized_total_samples": 100,
                    "normalized_exact_match_usable": 80
                    if threshold == 0.90
                    else round((threshold - 0.2) * 100),
                    "normalized_linear_interpolation_usable": 12
                    if threshold == 0.90
                    else 10,
                    "normalized_usable_samples": 92
                    if threshold == 0.90
                    else round((threshold - 0.1) * 100),
                    "normalized_usable_fraction": 0.92
                    if threshold == 0.90
                    else round(threshold - 0.1, 2),
                }
                for threshold in grid
            ],
        }
    }


def test_review_schema_version_1_is_required_and_accepted() -> None:
    payload = _review_payload()
    payload["schema_version"] = 1

    review = parse_capture_review(payload)

    assert review.schema_version == 1


def test_review_schema_version_is_not_optional() -> None:
    payload = _review_payload()
    del payload["schema_version"]

    with pytest.raises(CaptureQualificationValidationError, match="schema_version"):
        parse_capture_review(payload)


@pytest.mark.parametrize(
    "section,names",
    [
        ("capture_items", CAPTURE_ITEM_NAMES),
        ("annotated_com_items", ANNOTATED_ITEM_NAMES),
    ],
)
def test_review_requires_exact_canonical_checklist(
    section: str, names: tuple[str, ...]
) -> None:
    for mutation in ("missing", "extra"):
        payload = _review_payload()
        if mutation == "missing":
            del payload[section][names[0]]
        else:
            payload[section]["alias_item"] = {"status": "confirmed", "note": ""}

        with pytest.raises(
            CaptureQualificationValidationError, match="keys do not match"
        ):
            parse_capture_review(payload)


def test_review_accepts_statuses_and_requires_nonconfirmed_notes() -> None:
    payload = _review_payload()
    statuses = ("confirmed", "not_confirmed", "not_applicable", "uncertain")
    for name, status in zip(CAPTURE_ITEM_NAMES[:4], statuses, strict=True):
        payload["capture_items"][name] = {
            "status": status,
            "note": "" if status == "confirmed" else f"synthetic {status}",
        }

    review = parse_capture_review(payload)

    assert [item.status for _, item in review.capture_items[:4]] == list(statuses)

    for status in statuses[1:]:
        invalid = _review_payload()
        invalid["capture_items"][CAPTURE_ITEM_NAMES[0]] = {
            "status": status,
            "note": " ",
        }
        with pytest.raises(
            CaptureQualificationValidationError, match="note is required"
        ):
            parse_capture_review(invalid)


def test_review_rejects_noncanonical_status() -> None:
    payload = _review_payload()
    payload["capture_items"][CAPTURE_ITEM_NAMES[0]]["status"] = "Confirmed"

    with pytest.raises(CaptureQualificationValidationError, match="exactly one of"):
        parse_capture_review(payload)


def test_extracts_exact_frozen_grid_segments_and_stride_categories() -> None:
    evidence = extract_step5c_evidence(_qualification_payload())

    assert evidence == Step5cEvidence(
        finite_com_fraction=0.96,
        primary_eligible_fraction=0.91,
        primary_longest_usable_interval_seconds=9.0,
        persistent_supported_segments=("left_forearm",),
        reviewed_candidate_stride_count=4,
        policy_complete_stride_count=2,
        policy_complete_stride_fraction=0.5,
        policy_complete_left_count=1,
        policy_complete_right_count=1,
        normalized_usable_fraction=0.92,
        empirical_max_mass_coverage=0.93,
        theoretical_supported_mass_fraction=0.9306,
        total_frames=100,
        finite_com_frames=96,
        primary_eligible_frame_count=91,
        primary_longest_usable_interval_frames=90,
        normalized_total_samples=100,
        normalized_exact_match_usable=80,
        normalized_linear_interpolation_usable=12,
        normalized_usable_samples=92,
        transient_supported_segments=("right_forearm",),
    )


def test_evidence_rejects_grid_reordering_and_nonfinite_values() -> None:
    reordered = _qualification_payload()
    entries = reordered["sensitivity_results"]["threshold_sensitivity"]
    entries[0], entries[1] = entries[1], entries[0]
    with pytest.raises(CaptureQualificationValidationError, match="exact frozen grid"):
        extract_step5c_evidence(reordered)

    nonfinite = _qualification_payload()
    nonfinite["sensitivity_results"]["threshold_sensitivity"][5][
        "longest_usable_interval_seconds"
    ] = float("nan")
    with pytest.raises(CaptureQualificationValidationError, match="must be finite"):
        extract_step5c_evidence(nonfinite)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("segment_order", "exact canonical segment names in order"),
        ("segment_semantics", "support and structural missingness"),
        ("stride_category", "qualification_category must be one of"),
        ("negative_duration", "counts, durations, or fractions"),
        ("frame_fraction", "counts, durations, or fractions"),
        ("primary_total_strides", "frame and normalized counts are inconsistent"),
        ("primary_complete_strides", "frame and normalized counts are inconsistent"),
        ("normalized_counts", "counts, durations, or fractions"),
        ("empirical_max", "frame and normalized counts are inconsistent"),
    ],
)
def test_evidence_rejects_malformed_step5b_records(mutation: str, match: str) -> None:
    payload = _qualification_payload()
    sensitivity = payload["sensitivity_results"]
    primary = sensitivity["threshold_sensitivity"][-1]
    if mutation == "segment_order":
        segments = sensitivity["segment_summaries"]
        segments[1], segments[2] = segments[2], segments[1]
    elif mutation == "segment_semantics":
        sensitivity["segment_summaries"][0]["is_supported"] = True
    elif mutation == "stride_category":
        sensitivity["stride_summaries"][0]["qualification_category"] = "unknown"
    elif mutation == "negative_duration":
        primary["longest_usable_interval_seconds"] = -0.1
    elif mutation == "frame_fraction":
        primary["usable_fraction"] = 0.90
    elif mutation == "primary_total_strides":
        primary["total_strides"] = 5
    elif mutation == "primary_complete_strides":
        primary["policy_complete_strides"] = 3
    elif mutation == "normalized_counts":
        primary["normalized_exact_match_usable"] = 79
    elif mutation == "empirical_max":
        sensitivity["empirical_max_mass_coverage"] = 0.92

    with pytest.raises(CaptureQualificationValidationError, match=match):
        extract_step5c_evidence(payload)


def test_exact_go_thresholds_and_prior_improvement_produce_go() -> None:
    result = evaluate_engineering_readiness(
        _evidence(), _evidence(empirical_max_mass_coverage=0.89), _review()
    )

    assert result.decision == "GO"
    assert result.state == "evaluated"
    assert result.evaluation_state == "evaluated"
    assert result.external_human_review_state == "confirmed"
    assert not result.blockers
    assert not result.warnings
    assert all(criterion.passed for criterion in result.criteria)


@pytest.mark.parametrize(
    "change",
    [
        {"finite_com_fraction": 0.95 - 1e-9},
        {"primary_eligible_fraction": 0.90 - 1e-9},
        {"primary_longest_usable_interval_seconds": 3.0 - 1e-9},
        {"reviewed_candidate_stride_count": 2},
        {"policy_complete_stride_count": 2},
        {"policy_complete_stride_fraction": 0.75 - 1e-9},
        {"policy_complete_left_count": 0, "policy_complete_right_count": 3},
        {"normalized_usable_fraction": 0.90 - 1e-9},
        {"empirical_max_mass_coverage": 0.90 - 1e-9},
    ],
)
def test_below_go_thresholds_are_conditional(change: dict[str, object]) -> None:
    result = evaluate_engineering_readiness(
        _evidence(**change), _evidence(empirical_max_mass_coverage=0.88), _review()
    )

    assert result.decision == "CONDITIONAL"
    assert not result.blockers


def test_prior_improvement_and_comparability_are_enforced() -> None:
    insufficient = evaluate_engineering_readiness(
        _evidence(), _evidence(empirical_max_mass_coverage=0.891), _review()
    )
    incomparable = evaluate_engineering_readiness(
        _evidence(),
        _evidence(empirical_max_mass_coverage=0.89),
        _review(),
        comparable=False,
    )

    assert insufficient.decision == "CONDITIONAL"
    assert "GO criterion not met: max_coverage_improvement" in insufficient.warnings
    assert incomparable.decision == "CONDITIONAL"
    assert not incomparable.blockers
    assert "GO criterion not met: prior_comparable" in incomparable.warnings
    assert "GO criterion not met: max_coverage_improvement" in incomparable.warnings


def test_upstream_provenance_flag_false_is_no_go() -> None:
    result = evaluate_engineering_readiness(
        _evidence(),
        _evidence(empirical_max_mass_coverage=0.89),
        _review(),
        provenance_valid=False,
    )

    assert result.decision == "NO-GO"
    assert "provenance mismatch" in result.blockers
    criterion = next(
        item for item in result.criteria if item.name == "upstream_provenance_valid"
    )
    assert not criterion.passed


def test_declared_direction_mismatch_is_a_hard_blocker() -> None:
    payload = _review_payload()
    payload["declared_direction_matches_inherited_step4"] = False

    result = evaluate_engineering_readiness(
        _evidence(),
        _evidence(empirical_max_mass_coverage=0.89),
        parse_capture_review(payload),
    )

    assert result.decision == "NO-GO"
    assert any("does not match inherited Step4" in value for value in result.blockers)
    criterion = next(
        item
        for item in result.criteria
        if item.name == "declared_direction_matches_inherited_step4"
    )
    assert criterion.severity == "hard_no_go"
    assert not criterion.passed


@pytest.mark.parametrize(
    "review,expected_state,blocker",
    [
        (None, "pending_external_human_review", "external independent human"),
        (
            _review(reviewer_type="automated_assistant"),
            "pending_external_human_review",
            "external independent human",
        ),
        (
            _review(independence="not_independent"),
            "pending_external_human_review",
            "incomplete or not independent",
        ),
    ],
)
def test_missing_nonhuman_or_incomplete_review_is_no_go(
    review: CaptureReview | None, expected_state: str, blocker: str
) -> None:
    result = evaluate_engineering_readiness(
        _evidence(), _evidence(empirical_max_mass_coverage=0.89), review
    )

    assert result.decision == "NO-GO"
    assert result.state == expected_state
    assert result.evaluation_state == "evaluated"
    assert result.external_human_review_state in {"pending", "incomplete"}
    assert any(blocker in value for value in result.blockers)


def test_quantitative_hard_blocker_is_reported_while_review_is_pending() -> None:
    result = evaluate_engineering_readiness(
        _evidence(persistent_supported_segments=("left_forearm",)),
        _evidence(empirical_max_mass_coverage=0.89),
        None,
    )

    assert result.decision == "NO-GO"
    assert result.evaluation_state == "evaluated"
    assert result.external_human_review_state == "pending"
    assert any("persistent supported segments" in value for value in result.blockers)


@pytest.mark.parametrize(
    "status,decision",
    [
        ("uncertain", "CONDITIONAL"),
        ("not_confirmed", "NO-GO"),
        ("not_applicable", "NO-GO"),
    ],
)
def test_nonconfirmed_review_status_policy(status: str, decision: str) -> None:
    payload = _review_payload()
    payload["capture_items"][CAPTURE_ITEM_NAMES[0]] = {
        "status": status,
        "note": "synthetic reason",
    }
    result = evaluate_engineering_readiness(
        _evidence(),
        _evidence(empirical_max_mass_coverage=0.89),
        parse_capture_review(payload),
    )

    assert result.decision == decision
    if status == "uncertain":
        assert not result.blockers
        assert any("is uncertain" in value for value in result.warnings)
        assert result.external_human_review_state == "incomplete"
    else:
        assert any(status in value for value in result.blockers)

    criterion = next(
        item for item in result.criteria if item.name == "all_review_items_confirmed"
    )
    assert criterion.severity == "go_required"


@pytest.mark.parametrize(
    "change,blocker",
    [
        (
            {"persistent_supported_segments": ("left_thigh",)},
            "persistent supported segments",
        ),
        ({"primary_eligible_fraction": 0.50 - 1e-9}, "below hard minimum 0.50"),
        ({"policy_complete_stride_count": 1}, "fewer than 2 policy-complete strides"),
        ({"normalized_usable_fraction": 0.50 - 1e-9}, "below hard minimum 0.50"),
    ],
)
def test_hard_blockers_are_no_go(change: dict[str, object], blocker: str) -> None:
    result = evaluate_engineering_readiness(
        _evidence(**change), _evidence(empirical_max_mass_coverage=0.88), _review()
    )

    assert result.decision == "NO-GO"
    assert any(blocker in value for value in result.blockers)


@pytest.mark.parametrize(
    "change",
    [
        {"primary_eligible_fraction": 0.50},
        {"policy_complete_stride_count": 2},
        {"normalized_usable_fraction": 0.50},
    ],
)
def test_hard_minimums_are_exclusive_lower_bounds(
    change: dict[str, object],
) -> None:
    result = evaluate_engineering_readiness(
        _evidence(**change), _evidence(empirical_max_mass_coverage=0.88), _review()
    )

    assert result.decision == "CONDITIONAL"
    assert not result.blockers
