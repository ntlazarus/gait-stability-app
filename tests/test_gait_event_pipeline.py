from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from gait_stability.gait_event_pipeline import (
    STEP4_ARTIFACT_NAMES,
    GaitEventArtifactValidationError,
    GaitEventPipelineConfig,
    _build_samples,
    _csv_value,
    _read_processed,
    _select_bout,
    _validate_metadata,
    detect_gait_events,
)
from gait_stability.gait_events import (
    GAIT_EVENT_FIELDS,
    STRIDE_FIELDS,
    GaitEventConfig,
    detect_candidate_events,
)
from gait_stability.pose_contracts import MEDIAPIPE_LANDMARK_NAMES
from gait_stability.pose_preprocessing import PROCESSED_FIELDS, REQUIRED_GAIT_LANDMARKS
from gait_stability.video_ingestion import sha256_file

FRAME_COUNT, FPS = 120, 30.0
LEFT_PEAKS, RIGHT_PEAKS = (18, 54, 90), (36, 72, 108)
SIGNAL_PEAKS = {"left_heel": LEFT_PEAKS, "right_heel": RIGHT_PEAKS}
ANKLE_PEAKS = {"left_ankle": LEFT_PEAKS, "right_ankle": RIGHT_PEAKS}
FALSE_FIELD_TOKENS = ("usable", "interpolat", "nonfinite", "out_of_image", "changed")


@dataclass(frozen=True)
class SyntheticStep3:
    directory: Path
    processed: Path
    quality: Path
    metadata: Path
    video: Path
    video_available: bool

    @property
    def inputs(self) -> tuple[Path, Path, Path]:
        return self.processed, self.quality, self.metadata


def _triangle(frame: int, peaks: tuple[int, ...], amplitude: float) -> float:
    distance = min((abs(frame - peak) for peak in peaks), default=5)
    return 0.05 + amplitude * max(0.0, 1.0 - distance / 4.0)


def _x_value(frame: int, name: str) -> float:
    if name in SIGNAL_PEAKS:
        return 0.5 + _triangle(frame, SIGNAL_PEAKS[name], 0.2)
    if name in ANKLE_PEAKS:
        return 0.5 + _triangle(frame, ANKLE_PEAKS[name], 0.18)
    return 0.5


def _processed_row(frame: int, landmark_id: int, name: str) -> dict[str, str]:
    x = f"{_x_value(frame, name):.8f}"
    row = {field: "" for field in PROCESSED_FIELDS}
    for field in PROCESSED_FIELDS:
        if any(token in field for token in FALSE_FIELD_TOKENS) or field in {
            "rejected_low_confidence",
            "final_missing",
        }:
            row[field] = "false"
    row.update(
        {
            "frame_index": str(frame),
            "nominal_timestamp_seconds": f"{frame / FPS:.12f}",
            "frame_status": "decoded_pose",
            "landmark_id": str(landmark_id),
            "landmark_name": name,
            "raw_row_present": "true",
            "raw_x_normalized": x,
            "raw_y_normalized": "0.50000000",
            "visibility": "1.0",
            "presence": "1.0",
            "confidence": "1.0",
            "x_observed_usable": "true",
            "y_observed_usable": "true",
            "observed_usable": "true",
            "missing_or_nonfinite_enabled_score": "false",
            "processed_x_normalized": x,
            "processed_y_normalized": "0.50000000",
            "x_final_missing": "false",
            "y_final_missing": "false",
        }
    )
    return row


def _write_processed(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PROCESSED_FIELDS)
        writer.writeheader()
        for frame in range(FRAME_COUNT):
            for landmark_id, name in enumerate(MEDIAPIPE_LANDMARK_NAMES):
                writer.writerow(_processed_row(frame, landmark_id, name))


def _write_video(path: Path) -> bool:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (64, 48))
    if not writer.isOpened():
        writer.release()
        return False
    for frame_index in range(FRAME_COUNT):
        frame = np.full((48, 64, 3), frame_index % 251, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.isOpened():
        decoded, _ = capture.read()
        if not decoded:
            break
        count += 1
    opened = capture.isOpened()
    capture.release()
    return opened and count == FRAME_COUNT


@pytest.fixture
def synthetic_step3(tmp_path: Path) -> SyntheticStep3:
    directory = tmp_path / "step3"
    directory.mkdir()
    processed = directory / "processed_landmarks.csv"
    quality = directory / "pose_quality.json"
    metadata = directory / "preprocessing_metadata.json"
    video = directory / "source.mp4"
    _write_processed(processed)
    video_available = _write_video(video)
    quality.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "total_frames": FRAME_COUNT,
                "required_gait_landmarks": list(REQUIRED_GAIT_LANDMARKS),
                "per_landmark": {name: {} for name in MEDIAPIPE_LANDMARK_NAMES},
            }
        ),
        encoding="utf-8",
    )
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm_version": "step3-mvp-1",
                "outputs": {
                    "processed_landmarks.csv": {
                        "path": "processed_landmarks.csv",
                        "sha256": sha256_file(processed),
                    }
                },
                "inherited_provenance": {
                    "source": {
                        "path_identifier": str(video.resolve()),
                        "sha256": sha256_file(video) if video.is_file() else "",
                        "width_pixels": 64,
                        "height_pixels": 48,
                        "nominal_fps": FPS,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return SyntheticStep3(
        directory, processed, quality, metadata, video, video_available
    )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"event_config": object()}, TypeError),
        ({"manual_start_frame": 1}, ValueError),
        ({"manual_start_frame": True, "manual_end_frame": 2}, TypeError),
        ({"manual_start_frame": 9, "manual_end_frame": 8}, ValueError),
    ],
)
def test_pipeline_config_manual_pair_order_and_type_validation(
    kwargs: dict[str, Any], error: type[Exception]
) -> None:
    kwargs.setdefault("event_config", GaitEventConfig())
    with pytest.raises(error):
        GaitEventPipelineConfig(**kwargs)


def _rewrite_csv(path: Path, defect: str) -> None:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    if defect == "schema":
        fields[0], fields[1] = fields[1], fields[0]
    elif defect == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif defect == "observed_usable":
        rows[0]["observed_usable"] = "false"
    elif defect == "raw_y_missing":
        rows[0]["raw_y_normalized"] = ""
    elif defect == "raw_y_nonfinite":
        rows[0]["raw_y_normalized"] = "nan"
    else:
        rows[0]["raw_row_present"] = "TRUE"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("defect", "match"),
    [
        ("schema", "schema must exactly equal"),
        ("order", "ordered complete"),
        ("boolean", "must be true or false"),
        ("observed_usable", "conflicts with axis observed usability"),
        ("raw_y_missing", "observed usable y requires a finite raw y"),
        ("raw_y_nonfinite", "observed usable y requires a finite raw y"),
        ("hash", "hash does not match"),
        ("quality_count", "total_frames must match"),
    ],
)
def test_internal_input_validation_rejects_contract_defects(
    synthetic_step3: SyntheticStep3, defect: str, match: str
) -> None:
    if defect in {
        "schema",
        "order",
        "boolean",
        "observed_usable",
        "raw_y_missing",
        "raw_y_nonfinite",
    }:
        _rewrite_csv(synthetic_step3.processed, defect)
        with pytest.raises(GaitEventArtifactValidationError, match=match):
            _read_processed(synthetic_step3.processed)
        return
    quality = json.loads(synthetic_step3.quality.read_text(encoding="utf-8"))
    metadata = json.loads(synthetic_step3.metadata.read_text(encoding="utf-8"))
    if defect == "hash":
        metadata["outputs"]["processed_landmarks.csv"]["sha256"] = "0" * 64
    else:
        quality["total_frames"] = FRAME_COUNT - 1
    with pytest.raises(GaitEventArtifactValidationError, match=match):
        _validate_metadata(
            metadata, quality, sha256_file(synthetic_step3.processed), FRAME_COUNT
        )


def test_internal_bout_selection_without_rendering(
    synthetic_step3: SyntheticStep3,
) -> None:
    grid, timestamps = _read_processed(synthetic_step3.processed)
    samples = _build_samples(grid)
    automatic = _select_bout(
        samples, timestamps, GaitEventPipelineConfig(GaitEventConfig())
    )
    assert (automatic.start_frame, automatic.end_frame) == (0, 119)
    assert automatic.selection_method == "automatic"
    assert (
        automatic.quality
        == "complete_primary_signal_interval_with_minimum_candidate_count"
    )
    assert "manual confirmation" in automatic.selection_reason
    assert "not periodicity" in automatic.limitations[0]
    assert len(automatic.candidates) == 1
    assert automatic.candidates[0]["accepted_preliminary_events"] == {
        "left": 3,
        "right": 3,
        "total": 6,
    }

    manual = _select_bout(
        samples,
        timestamps,
        GaitEventPipelineConfig(
            GaitEventConfig(), manual_start_frame=18, manual_end_frame=108
        ),
    )
    assert (manual.start_frame, manual.end_frame) == (18, 108)
    assert manual.selection_method == "manual"
    assert manual.selection_reason == (
        "Inclusive analysis interval explicitly selected by the user."
    )

    fallback = _select_bout(
        samples,
        timestamps,
        GaitEventPipelineConfig(
            GaitEventConfig(), automatic_minimum_accepted_events_per_side=4
        ),
    )
    assert (fallback.start_frame, fallback.end_frame) == (0, 119)
    assert fallback.selection_method == "full_recording_fallback"
    assert fallback.candidates == ()
    assert "not evidence" in fallback.limitations[0]


def _flag_x_interpolation(path: Path, name: str, frame: int) -> None:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    row = next(
        item
        for item in rows
        if item["landmark_name"] == name and int(item["frame_index"]) == frame
    )
    row.update(
        {
            "raw_x_normalized": "",
            "x_observed_usable": "false",
            "observed_usable": "false",
            "missing_or_nonfinite_enabled_score": "true",
            "pre_smoothed_x_normalized": row["processed_x_normalized"],
            "x_interpolated": "true",
        }
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PROCESSED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("landmark", "expected_quality", "expected_flag"),
    [
        ("left_knee", "high", False),
        ("left_heel", "review", True),
    ],
)
def test_build_samples_maps_only_primary_quality_provenance(
    synthetic_step3: SyntheticStep3,
    landmark: str,
    expected_quality: str,
    expected_flag: bool,
) -> None:
    _flag_x_interpolation(synthetic_step3.processed, landmark, LEFT_PEAKS[0])
    grid, _ = _read_processed(synthetic_step3.processed)
    samples = _build_samples(grid)
    event = next(
        event
        for event in detect_candidate_events(samples, GaitEventConfig(), 0, 119)
        if event.side == "left" and event.frame_index == LEFT_PEAKS[0]
    )
    assert event.detection_status == "accepted"
    assert event.confidence_or_quality == expected_quality
    assert event.primary_support_interpolated is expected_flag
    assert (
        "primary_support_contains_interpolation" in event.signal_support_notes
    ) is expected_flag


def test_build_samples_maps_exact_ankle_quality_provenance(
    synthetic_step3: SyntheticStep3,
) -> None:
    _flag_x_interpolation(synthetic_step3.processed, "left_ankle", LEFT_PEAKS[0] + 1)
    grid, _ = _read_processed(synthetic_step3.processed)
    samples = _build_samples(grid)
    event = next(
        event
        for event in detect_candidate_events(samples, GaitEventConfig(), 0, 119)
        if event.side == "left" and event.frame_index == LEFT_PEAKS[0]
    )

    assert event.detection_status == "accepted"
    assert event.ankle_peak_frame == LEFT_PEAKS[0]
    assert event.confidence_or_quality == "review"
    assert event.ankle_support_interpolated
    assert "ankle_support_contains_interpolation" in event.signal_support_notes


def test_build_samples_maps_exact_bilateral_hip_quality_provenance(
    synthetic_step3: SyntheticStep3,
) -> None:
    frame = LEFT_PEAKS[0] + 1
    _flag_x_interpolation(synthetic_step3.processed, "right_hip", frame)
    grid, _ = _read_processed(synthetic_step3.processed)
    sample = _build_samples(grid)["left"][frame]

    assert not sample.hip_observed_usable
    assert sample.hip_interpolated
    assert not sample.hip_smoothing_support_contains_interpolation


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or ()), list(reader)


def _decoded_video_properties(path: Path) -> tuple[int, int, float, int]:
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = 0
    while True:
        decoded, _ = capture.read()
        if not decoded:
            break
        frames += 1
    capture.release()
    return width, height, fps, frames


def _run_cli(directory: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, ("src", environment.get("PYTHONPATH")))
    )
    return subprocess.run(
        [sys.executable, "scripts/detect_gait_events.py", str(directory), *arguments],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def test_end_to_end_outputs_metadata_cli_and_replacement(
    synthetic_step3: SyntheticStep3,
) -> None:
    if not synthetic_step3.video_available:
        pytest.skip("mp4v is unavailable for video-dependent Step 4 tests")
    pytest.importorskip(
        "matplotlib.pyplot", reason="Step 4 rendering requires matplotlib"
    )
    input_hashes = {path.name: sha256_file(path) for path in synthetic_step3.inputs}
    artifacts = detect_gait_events(
        synthetic_step3.directory, GaitEventPipelineConfig(GaitEventConfig())
    )

    event_header, events = _read_csv(artifacts.gait_events_path)
    stride_header, strides = _read_csv(artifacts.strides_path)
    assert event_header == list(GAIT_EVENT_FIELDS)
    assert stride_header == list(STRIDE_FIELDS)
    assert [row["event_id"] for row in events] == [
        f"E{index:04d}" for index in range(1, 7)
    ]
    assert [(row["side"], int(row["frame_index"])) for row in events] == list(
        zip(("left", "right") * 3, range(18, 109, 18), strict=True)
    )
    assert all(row["detection_status"] == "accepted" for row in events)
    assert all(0 <= int(row["frame_index"]) <= 119 for row in events)
    assert [
        (row["side"], int(row["start_frame"]), int(row["end_frame"])) for row in strides
    ] == [
        ("left", 18, 54),
        ("right", 36, 72),
        ("left", 54, 90),
        ("right", 72, 108),
    ]
    assert [float(row["duration_seconds"]) for row in strides] == pytest.approx(
        [1.2] * 4
    )
    assert {row["included_in_stride_construction"] for row in events} == {"true"}
    assert {
        (
            row["ankle_support_observed_usable"],
            row["ankle_support_interpolated"],
            row["ankle_support_smoothing_contains_interpolation"],
        )
        for row in events
    } == {("true", "false", "false")}
    assert all(
        row["signal_support_notes"] == "raw_peak_agreement|ankle_peak_agreement"
        for row in events
    )
    assert _csv_value(("first", "second")) == "first|second"
    assert _csv_value(None) == ""

    bout = json.loads(artifacts.walking_bout_path.read_text(encoding="utf-8"))
    assert (
        bout["schema_version"],
        bout["start_frame"],
        bout["end_frame"],
        bout["start_timestamp_seconds"],
        bout["end_timestamp_seconds"],
        bout["selection_method"],
    ) == (1, 0, 119, 0.0, pytest.approx(119 / FPS), "automatic")
    assert bout["selection_reason"].startswith(
        "Selected qualifying complete-primary-signal"
    )
    assert (
        bout["quality"]
        == "complete_primary_signal_interval_with_minimum_candidate_count"
    )
    assert bout["boundary_inclusivity"] == (
        "start and end frames/timestamps are inclusive"
    )
    assert len(bout["limitations"]) == len(bout["candidate_bouts"]) == 1
    assert (
        bout["candidate_bouts"][0]["start_frame"],
        bout["candidate_bouts"][0]["end_frame"],
    ) == (0, 119)

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    method = metadata["method"]
    assert (metadata["schema_version"], metadata["algorithm_version"]) == (
        1,
        "step4-mvp-1",
    )
    assert "heel_side_x" in method["formula"]
    assert "normalized image width" in method["units"]
    assert method["direction"] == "user-declared and not inferred"
    assert method["detector_smoothing"].startswith("none")
    assert "clean cue count" in method["temporal_conflict_rule"]
    assert "nearby-peak selection" in method["temporal_conflict_rule"]
    assert "accepted otherwise review" in method["quality_rule"]
    assert method["stride_rule"].startswith("consecutive accepted same-side")
    assert method["toe_off"] == "omitted"
    assert method["phase_metrics"].startswith("no stance")
    assert "not stable across parameter" in method["event_id_stability"]
    assert "2% of image width" in method["prominence_rule"]
    assert "near-simultaneous contacts" in method["opposite_side_gate"]
    assert "not independent corroboration" in method["raw_cue"]
    assert "ankle and both hips" in method["support_landmark"]
    assert "ankle-plus-bilateral-hip" in method["quality_rule"]
    assert metadata["config"]["detector"] == asdict(GaitEventConfig())
    assert metadata["counts"]["events_by_side_type_status_quality"] == {
        side: {"candidate_initial_contact": {"accepted": {"high": 3}}}
        for side in ("left", "right")
    }
    assert metadata["counts"]["complete_strides_by_side_quality"] == {
        side: {"high": 2, "review": 0, "low": 0} for side in ("left", "right")
    }
    assert metadata["schemas"]["gait_events.csv"] == list(GAIT_EVENT_FIELDS)
    assert metadata["schemas"]["strides.csv"] == list(STRIDE_FIELDS)
    assert metadata["inputs"] == {
        name: {"path": str(synthetic_step3.directory / name), "sha256": digest}
        for name, digest in input_hashes.items()
    }
    assert metadata["source_video"]["sha256"] == sha256_file(synthetic_step3.video)
    assert any("not force-plate-confirmed" in item for item in metadata["limitations"])
    assert metadata["validation_status"].startswith("software workflow only")
    for name, output in metadata["outputs"].items():
        if name == "gait_event_metadata.json":
            assert output["sha256"] is None
        else:
            assert output["sha256"] == sha256_file(synthetic_step3.directory / name)
    assert cv2.imread(str(artifacts.diagnostic_path)) is not None
    assert _decoded_video_properties(artifacts.annotated_video_path) == (
        64,
        48,
        pytest.approx(FPS),
        FRAME_COUNT,
    )
    assert {path.name: sha256_file(path) for path in synthetic_step3.inputs} == (
        input_hashes
    )

    first_run_id = metadata["run_id"]
    completed = _run_cli(
        synthetic_step3.directory, "--walking-direction", "image_right"
    )
    assert completed.returncode == 0
    assert completed.stdout == f"{artifacts.metadata_path}\n"
    replaced = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert replaced["run_id"] != first_run_id
    assert all(
        (synthetic_step3.directory / name).is_file() for name in STEP4_ARTIFACT_NAMES
    )
    assert {path.name: sha256_file(path) for path in synthetic_step3.inputs} == (
        input_hashes
    )


def test_cli_requires_direction_and_reports_unpaired_manual_bound(
    synthetic_step3: SyntheticStep3,
) -> None:
    missing_direction = _run_cli(synthetic_step3.directory)
    assert missing_direction.returncode == 2
    assert "--walking-direction" in missing_direction.stderr

    unpaired = _run_cli(
        synthetic_step3.directory,
        "--walking-direction",
        "image_right",
        "--manual-start-frame",
        "10",
    )
    assert unpaired.returncode == 1
    assert "manual_start_frame and manual_end_frame are both required" in (
        unpaired.stderr
    )


def test_different_video_override_must_match_inherited_provenance(
    synthetic_step3: SyntheticStep3, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not synthetic_step3.video_available:
        pytest.skip("mp4v is unavailable for video-dependent Step 4 tests")
    different_video = synthetic_step3.directory / "different.mp4"
    different_video.write_bytes(synthetic_step3.video.read_bytes() + b"different hash")

    def write_stub(path: Path, *_args: object) -> None:
        path.write_bytes(b"stub")

    for renderer in ("_write_diagnostic", "_write_annotated_video"):
        monkeypatch.setattr(
            f"gait_stability.gait_event_pipeline.{renderer}", write_stub
        )
    with pytest.raises(
        GaitEventArtifactValidationError, match="override|provenance|hash"
    ):
        detect_gait_events(
            synthetic_step3.directory,
            GaitEventPipelineConfig(GaitEventConfig()),
            video_path=different_video,
        )
