"""Synthetic integration tests for Step 5c and its CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_com_qualification_pipeline import _build_qualification_fixture

from gait_stability.capture_qualification import (
    ANNOTATED_ITEM_NAMES,
    CAPTURE_ITEM_NAMES,
    CAPTURE_QUALIFICATION_ALGORITHM_VERSION,
    CAPTURE_QUALIFICATION_SCHEMA_VERSION,
    CaptureQualificationValidationError,
    Step5cEvidence,
    evaluate_engineering_readiness,
    parse_capture_review,
    qualify_clean_capture,
)
from gait_stability.com_qualification import ComQualificationConfig
from gait_stability.com_qualification_pipeline import qualify_com
from gait_stability.video_ingestion import sha256_file


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _finish_step5c_lineage(artifacts: Path) -> None:
    """Add lineage fields required by Step 5c to the reusable Step 5b fixture."""
    preprocessing_path = artifacts / "preprocessing_metadata.json"
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    preprocessing["inherited_provenance"]["backend"] = {"model_sha256": "b" * 64}
    _write_json(preprocessing_path, preprocessing)

    resolution_path = artifacts / "review_resolution_metadata.json"
    resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
    resolution["inputs"]["preprocessing_metadata.json"]["sha256"] = sha256_file(
        preprocessing_path
    )
    resolution["source_step4"] = {"gait_event_config": {"direction": "image_right"}}
    _write_json(resolution_path, resolution)

    qualification_path = artifacts / "com_qualification.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["inputs"]["step5_upstream.preprocessing_metadata.json"]["sha256"] = (
        sha256_file(preprocessing_path)
    )
    qualification["inputs"]["step5_upstream.review_resolution_metadata.json"][
        "sha256"
    ] = sha256_file(resolution_path)
    _write_json(qualification_path, qualification)


def _build_step5b(tmp_path: Path) -> tuple[Path, Path]:
    artifacts, source = _build_qualification_fixture(tmp_path)
    qualify_com(artifacts, ComQualificationConfig())
    _finish_step5c_lineage(artifacts)
    return artifacts, source


def _review_payload(artifacts: Path, source: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "review_id": "synthetic-pipeline-review",
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
            "source_video": sha256_file(source),
            "annotated_com.mp4": sha256_file(artifacts / "annotated_com.mp4"),
            "com_qualification.json": sha256_file(artifacts / "com_qualification.json"),
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


@pytest.fixture
def step5c_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    current, current_source = _build_step5b(tmp_path / "current")
    prior, _ = _build_step5b(tmp_path / "prior")

    prior_path = prior / "com_qualification.json"
    prior_payload = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_payload["sensitivity_results"]["empirical_max_mass_coverage"] = 0.90
    prior_payload["sensitivity_results"]["mass_coverage_max"] = 0.90
    _write_json(prior_path, prior_payload)

    review_path = tmp_path / "capture_review.json"
    _write_json(review_path, _review_payload(current, current_source))
    return current, current_source, prior_path, review_path


def test_pipeline_publishes_versioned_artifact_with_hash_linkage(
    step5c_inputs: tuple[Path, Path, Path, Path],
) -> None:
    current, source, prior_path, review_path = step5c_inputs

    result = qualify_clean_capture(current, review_path, prior_path)

    assert result.qualification_json_path == current / "capture_qualification.json"
    payload = json.loads(result.qualification_json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CAPTURE_QUALIFICATION_SCHEMA_VERSION
    assert payload["algorithm_version"] == CAPTURE_QUALIFICATION_ALGORITHM_VERSION
    assert payload["inputs"]["current_source_video"]["sha256"] == sha256_file(source)
    assert payload["inputs"]["capture_review.json"]["sha256"] == sha256_file(
        review_path
    )
    assert payload["comparison"]["empirical_max_mass_coverage_delta"] == pytest.approx(
        0.0306
    )
    assert payload["engineering_readiness"] == {
        "decision": "NO-GO",
        "state": "evaluated",
        "evaluation_state": "evaluated",
        "external_human_review_state": "confirmed",
        "scope": "exploratory engineering use only",
    }
    assert (
        payload["inherited_step4"]["declared_direction_matches_inherited_step4"] is True
    )
    assert payload["provenance"]["anthropometric_coefficient_sex"] == {
        "value": "male",
        "selection_source": "user_supplied_to_step5a",
        "inferred": False,
        "model_scope": (
            "population-average coefficient model, not individual anthropometry"
        ),
    }


def test_pipeline_rejects_review_hash_mismatch(
    step5c_inputs: tuple[Path, Path, Path, Path],
) -> None:
    current, _, prior_path, review_path = step5c_inputs
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["artifact_hashes"]["annotated_com.mp4"] = "c" * 64
    _write_json(review_path, review)

    with pytest.raises(
        CaptureQualificationValidationError,
        match=r"annotated_com\.mp4 hash does not match",
    ):
        qualify_clean_capture(current, review_path, prior_path)

    assert not (current / "capture_qualification.json").exists()


def test_missing_review_publishes_pending_no_go_artifact(
    step5c_inputs: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    current, _, prior_path, _ = step5c_inputs
    missing_review = tmp_path / "not_supplied.json"

    result = qualify_clean_capture(current, missing_review, prior_path)

    payload = json.loads(result.qualification_json_path.read_text(encoding="utf-8"))
    assert payload["inputs"]["capture_review.json"]["status"] == "missing"
    assert payload["capture_review"] is None
    assert payload["engineering_readiness"]["decision"] == "NO-GO"
    assert payload["engineering_readiness"]["state"] == (
        "pending_external_human_review"
    )
    assert payload["engineering_readiness"]["evaluation_state"] == "evaluated"
    assert payload["engineering_readiness"]["external_human_review_state"] == (
        "pending"
    )


def _mutate_prior_preprocessing(prior_path: Path, mismatch: str) -> None:
    artifacts = prior_path.parent
    preprocessing_path = artifacts / "preprocessing_metadata.json"
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    if mismatch == "preprocessing_config":
        preprocessing["config"]["synthetic_comparability_change"] = True
    else:
        preprocessing["inherited_provenance"]["backend"]["model_sha256"] = "c" * 64
    _write_json(preprocessing_path, preprocessing)

    resolution_path = artifacts / "review_resolution_metadata.json"
    resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
    resolution["inputs"]["preprocessing_metadata.json"]["sha256"] = sha256_file(
        preprocessing_path
    )
    _write_json(resolution_path, resolution)

    qualification = json.loads(prior_path.read_text(encoding="utf-8"))
    qualification["inputs"]["step5_upstream.preprocessing_metadata.json"]["sha256"] = (
        sha256_file(preprocessing_path)
    )
    qualification["inputs"]["step5_upstream.review_resolution_metadata.json"][
        "sha256"
    ] = sha256_file(resolution_path)
    _write_json(prior_path, qualification)


@pytest.mark.parametrize("mismatch", ["preprocessing_config", "pose_model_sha256"])
def test_actual_comparability_mismatch_is_conditional_without_other_blocker(
    step5c_inputs: tuple[Path, Path, Path, Path], mismatch: str
) -> None:
    current_path, _, prior_path, review_path = step5c_inputs
    _mutate_prior_preprocessing(prior_path, mismatch)

    from gait_stability import capture_qualification as pipeline

    current = pipeline._validate_qualification(
        current_path / "com_qualification.json", "current"
    )
    prior = pipeline._validate_qualification(prior_path, "prior")
    comparable, comparison = pipeline._comparability(current, prior)
    evidence = Step5cEvidence(
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
    prior_evidence = replace(evidence, empirical_max_mass_coverage=0.89)
    review = parse_capture_review(json.loads(review_path.read_text(encoding="utf-8")))
    evaluation = evaluate_engineering_readiness(
        evidence, prior_evidence, review, comparable=comparable
    )

    assert comparable is False
    assert comparison["checks"][mismatch] is False
    assert evaluation.decision == "CONDITIONAL"
    assert not evaluation.blockers

    result = qualify_clean_capture(current_path, review_path, prior_path)
    output = json.loads(result.qualification_json_path.read_text(encoding="utf-8"))
    assert output["comparison"]["coverage_delta_status"] == ("not_interpretable_for_go")
    assert output["comparison"]["current_empirical_max_mass_coverage"] == pytest.approx(
        current.evidence.empirical_max_mass_coverage
    )
    assert output["comparison"]["prior_empirical_max_mass_coverage"] == pytest.approx(
        prior.evidence.empirical_max_mass_coverage
    )


def test_pipeline_rechecks_consumed_hashes_before_publication(
    step5c_inputs: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current, source, prior_path, review_path = step5c_inputs
    from gait_stability import capture_qualification as pipeline

    original_recheck = pipeline._recheck

    def mutate_then_recheck(snapshot: dict[Path, str]) -> None:
        with source.open("ab") as file:
            file.write(b"changed-before-publication")
        original_recheck(snapshot)

    monkeypatch.setattr(pipeline, "_recheck", mutate_then_recheck)

    with pytest.raises(
        CaptureQualificationValidationError,
        match="consumed artifact changed before publication",
    ):
        qualify_clean_capture(current, review_path, prior_path)

    assert not (current / "capture_qualification.json").exists()


def _run_cli(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/qualify_capture.py", *arguments],
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
        check=False,
    )


def test_cli_help() -> None:
    result = _run_cli(["--help"])

    assert result.returncode == 0
    assert "--prior-qualification" in result.stdout
    assert "capture_review" in result.stdout


def test_cli_reports_validation_error(tmp_path: Path) -> None:
    result = _run_cli(
        [
            str(tmp_path / "missing-artifacts"),
            str(tmp_path / "missing-review.json"),
            "--prior-qualification",
            str(tmp_path / "missing-prior.json"),
        ]
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "Capture qualification failed:" in result.stderr
    assert "artifact directory does not exist" in result.stderr


def test_cli_success_prints_published_path(
    step5c_inputs: tuple[Path, Path, Path, Path],
) -> None:
    current, _, prior_path, review_path = step5c_inputs

    result = _run_cli(
        [
            str(current),
            str(review_path),
            "--prior-qualification",
            str(prior_path),
        ]
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.strip() == str(current / "capture_qualification.json")
    assert (current / "capture_qualification.json").is_file()


def test_cli_rejects_malformed_review_hash(
    step5c_inputs: tuple[Path, Path, Path, Path],
) -> None:
    current, _, prior_path, review_path = step5c_inputs
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["artifact_hashes"]["source_video"] = "A" * 64
    _write_json(review_path, review)

    result = _run_cli(
        [
            str(current),
            str(review_path),
            "--prior-qualification",
            str(prior_path),
        ]
    )

    assert result.returncode == 1
    assert "lowercase hexadecimal SHA-256" in result.stderr
