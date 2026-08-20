"""Synthetic end-to-end tests for the Step 5b COM qualification pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from test_com_pipeline import _build_fixture

from gait_stability.com_estimation import ComEstimationConfig
from gait_stability.com_pipeline import estimate_com
from gait_stability.com_qualification import ComQualificationConfig
from gait_stability.com_qualification_pipeline import (
    COM_QUALIFICATION_OUTPUT_ARTIFACT_NAMES,
    COM_STRIDE_QC_FIELDS,
    ComQualificationArtifactValidationError,
    qualify_com,
)
from gait_stability.pose_preprocessing import (
    PREPROCESSING_ALGORITHM_VERSION,
    PREPROCESSING_SCHEMA_VERSION,
)
from gait_stability.video_ingestion import ArtifactPublishError, sha256_file

_WIDTH = 64
_HEIGHT = 48
_FPS = 30.0


def _write_source_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), _FPS, (_WIDTH, _HEIGHT)
    )
    assert writer.isOpened(), "OpenCV MP4V writer is required for this integration test"
    try:
        for frame_index in range(3):
            frame = np.full((_HEIGHT, _WIDTH, 3), frame_index * 30, dtype=np.uint8)
            cv2.circle(frame, (12 + frame_index * 8, 24), 5, (20, 80, 180), -1)
            writer.write(frame)
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == _WIDTH
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == _HEIGHT
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(_FPS)
    finally:
        capture.release()


def _build_qualification_fixture(tmp_path: Path) -> tuple[Path, Path]:
    artifacts = _build_fixture(tmp_path)
    source_video = tmp_path / "source.mp4"
    _write_source_video(source_video)

    preprocessing_path = artifacts / "preprocessing_metadata.json"
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    preprocessing["inherited_provenance"]["source"] = {
        "path_identifier": str(source_video),
        "filename": source_video.name,
        "sha256": sha256_file(source_video),
        "width_pixels": _WIDTH,
        "height_pixels": _HEIGHT,
        "nominal_fps": _FPS,
        "nominal_frame_count": "3",
        "nominal_duration_seconds": "0.1",
        "project_rotation": "none",
        "project_mirroring": "none",
    }
    preprocessing["inherited_provenance"]["capture_assumptions"] = {
        "input": "single monocular RGB video",
        "calibration": "none",
        "camera_view": "not required, declared, or verified",
        "gait_direction": "not required, declared, or verified",
        "camera_placement": "not required, declared, or verified",
        "mirroring": "not required, detected, or verified",
    }
    preprocessing["run_id"] = "step3-test-run"
    preprocessing["interpolation"] = {
        "method": "linear in nominal_timestamp_seconds",
        "maximum_missing_samples": 3,
        "gait_event_crossing": (
            "not assessed because gait events are not available at Step 3"
        ),
    }
    preprocessing["smoothing"] = {
        "method": "centered unweighted moving average (boxcar)",
        "configured_window_frames": 3,
        "phase": "centered/noncausal",
        "edge_behavior": "largest symmetric odd support",
    }
    preprocessing_path.write_text(
        json.dumps(preprocessing, indent=2) + "\n", encoding="utf-8"
    )

    review_path = artifacts / "review_resolution_metadata.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["inputs"]["preprocessing_metadata.json"]["sha256"] = sha256_file(
        preprocessing_path
    )
    review_path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")

    estimate_com(artifacts, ComEstimationConfig(anthropometry_sex="male"))
    return artifacts, source_video


def _qualification_paths(artifacts: Path) -> tuple[Path, ...]:
    return tuple(artifacts / name for name in COM_QUALIFICATION_OUTPUT_ARTIFACT_NAMES)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_qualify_com_publishes_expected_qc_artifacts(tmp_path: Path) -> None:
    artifacts, _ = _build_qualification_fixture(tmp_path)

    result = qualify_com(artifacts, ComQualificationConfig())

    assert result.qualification_json_path == artifacts / "com_qualification.json"
    assert result.stride_qc_csv_path == artifacts / "com_stride_qc.csv"
    assert result.annotated_video_path == artifacts / "annotated_com.mp4"
    assert all(path.is_file() for path in _qualification_paths(artifacts))

    metadata = json.loads(result.qualification_json_path.read_text(encoding="utf-8"))
    assert metadata["sensitivity_grid"]["coverage_thresholds"] == [
        0.80,
        0.82,
        0.84,
        0.86,
        0.88,
        0.90,
    ]
    assert metadata["primary_policy_gate"]["threshold"] == 0.9
    assert (
        metadata["anthropometric_model"]["theoretical_supported_mass_fraction"]
        == 0.9306
    )
    assert metadata["readiness_for_step6"]["status"] == "CONDITIONAL"
    assert (
        metadata["stride_statistics"][0]["policy_complete_at_primary_threshold"] is True
    )
    assert metadata["preprocessing_inheritance"] == {
        "run_id": "step3-test-run",
        "created_at_utc": None,
        "schema_version": PREPROCESSING_SCHEMA_VERSION,
        "algorithm_version": PREPROCESSING_ALGORITHM_VERSION,
        "config": {"dummy": True},
        "interpolation": {
            "method": "linear in nominal_timestamp_seconds",
            "maximum_missing_samples": 3,
            "gait_event_crossing": (
                "not assessed because gait events are not available at Step 3"
            ),
            "note": metadata["preprocessing_inheritance"]["interpolation"]["note"],
        },
        "smoothing": {
            "method": "centered unweighted moving average (boxcar)",
            "configured_window_frames": 3,
            "phase": "centered/noncausal",
            "edge_behavior": "largest symmetric odd support",
            "note": metadata["preprocessing_inheritance"]["smoothing"]["note"],
        },
    }
    assert metadata["camera_view"]["artifact_declaration"]["camera_view"] == (
        "not required, declared, or verified"
    )
    assert metadata["camera_view"]["artifact_declaration"]["status"] == (
        "not_declared_or_verified"
    )
    assert metadata["camera_view"]["human_review"] == {
        "required": True,
        "status": "not_recorded",
    }

    with result.stride_qc_csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == list(COM_STRIDE_QC_FIELDS)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["primary_threshold"] == "0.900000"
    assert rows[0]["qualification_category"] == "policy_complete_at_threshold"
    assert rows[0]["policy_complete_at_primary_threshold"] == "true"
    assert [
        rows[0][field]
        for field in (
            "all_original_frames_policy_eligible",
            "all_supported_segments_represented",
            "represented_segment_set_invariant",
            "normalized_grid_complete",
            "endpoints_policy_eligible",
            "all_contributing_segments_raw_observed",
        )
    ] == ["true"] * 6


def test_qualify_com_renders_all_source_frames(tmp_path: Path) -> None:
    artifacts, source_path = _build_qualification_fixture(tmp_path)
    output_path = qualify_com(artifacts, ComQualificationConfig()).annotated_video_path

    source = cv2.VideoCapture(str(source_path))
    output = cv2.VideoCapture(str(output_path))
    assert source.isOpened()
    assert output.isOpened()
    decoded = 0
    changed = []
    try:
        while True:
            source_ok, source_frame = source.read()
            output_ok, output_frame = output.read()
            assert source_ok == output_ok
            if not output_ok:
                break
            decoded += 1
            assert output_frame.shape == (_HEIGHT, _WIDTH, 3)
            changed.append(np.any(cv2.absdiff(source_frame, output_frame) > 10))
    finally:
        source.release()
        output.release()

    assert decoded == 3
    assert all(changed), "Every rendered frame should contain qualification overlays"


def test_qualify_com_preserves_step5_coordinates_byte_for_byte(tmp_path: Path) -> None:
    artifacts, _ = _build_qualification_fixture(tmp_path)
    proxy_path = artifacts / "com_proxy.csv"
    stride_path = artifacts / "stride_com.csv"
    hashes_before = {path: sha256_file(path) for path in (proxy_path, stride_path)}
    coordinates_before = [
        (row["frame_index"], row["com_x"], row["com_y"])
        for row in _read_csv(proxy_path)
    ]

    qualify_com(artifacts, ComQualificationConfig())

    assert {path: sha256_file(path) for path in hashes_before} == hashes_before
    assert [
        (row["frame_index"], row["com_x"], row["com_y"])
        for row in _read_csv(proxy_path)
    ] == coordinates_before


def test_qualify_com_rejects_changed_coefficients_without_outputs(
    tmp_path: Path,
) -> None:
    artifacts, _ = _build_qualification_fixture(tmp_path)
    metadata_path = artifacts / "com_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["algorithm"]["coefficient_table"]["head"]["mass_fraction"] = 0.9999
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(
        ComQualificationArtifactValidationError,
        match=r"Coefficient table mass mismatch for head",
    ):
        qualify_com(artifacts, ComQualificationConfig())

    assert not any(path.exists() for path in _qualification_paths(artifacts))


def test_qualify_com_render_failure_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, _ = _build_qualification_fixture(tmp_path)

    def fail_render(**_: Any) -> None:
        raise RuntimeError("intentional renderer failure")

    monkeypatch.setattr(
        "gait_stability.com_qualification_pipeline._write_annotated_com_video",
        fail_render,
    )

    with pytest.raises(RuntimeError, match="intentional renderer failure"):
        qualify_com(artifacts, ComQualificationConfig())

    assert not any(path.exists() for path in _qualification_paths(artifacts))


def test_qualify_com_rechecks_source_hash_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, source_path = _build_qualification_fixture(tmp_path)
    from gait_stability import com_qualification_pipeline as pipeline

    original_render = pipeline._write_annotated_com_video

    def render_then_mutate(**kwargs: Any) -> None:
        original_render(**kwargs)
        with source_path.open("ab") as source:
            source.write(b"changed-after-render")

    monkeypatch.setattr(pipeline, "_write_annotated_com_video", render_then_mutate)

    with pytest.raises(
        ComQualificationArtifactValidationError, match=r"changed during qualification"
    ):
        qualify_com(artifacts, ComQualificationConfig())

    assert not any(path.exists() for path in _qualification_paths(artifacts))


def test_qualify_com_publish_failure_restores_previous_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, _ = _build_qualification_fixture(tmp_path)
    qualify_com(artifacts, ComQualificationConfig())
    output_paths = _qualification_paths(artifacts)
    contents_before = {path: path.read_bytes() for path in output_paths}
    original_replace = Path.replace

    def fail_staged_csv(source: Path, target: Path) -> Path:
        if (
            ".com-qualification-staging-" in source.parent.name
            and source.name == "com_stride_qc.csv"
        ):
            raise OSError("intentional staged publish failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_staged_csv)

    with pytest.raises(ArtifactPublishError) as exc_info:
        qualify_com(artifacts, ComQualificationConfig())

    assert isinstance(exc_info.value.__cause__, OSError)
    assert "intentional staged publish failure" in str(exc_info.value.__cause__)
    assert {path: path.read_bytes() for path in output_paths} == contents_before
    assert not tuple(artifacts.glob("*.backup-*"))
