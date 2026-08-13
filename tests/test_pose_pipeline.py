from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from gait_stability import VideoNotFoundError
from gait_stability.pose_contracts import (
    CanonicalLandmark,
    PoseEstimate,
    PoseFrameResult,
    PoseFrameStatus,
)
from gait_stability.pose_pipeline import (
    LANDMARK_FIELDS,
    POSE_ARTIFACT_NAMES,
    AnnotatedVideoWriteError,
    PosePipelineError,
    _metadata_payload,
    _publish_pose_artifacts,
    _render_pose,
    _write_csv_artifacts,
    estimate_pose_video,
)
from gait_stability.video_ingestion import (
    ArtifactPublishError,
    DecodedVideoFrame,
    VideoMetadata,
)


class SyntheticEstimator:
    def __init__(self) -> None:
        self.closed = False

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "synthetic-test",
            "configuration": {},
            "backend_temporal_behavior": "none",
        }

    def estimate_frame(
        self,
        image_rgb: Any,
        frame_index: int,
        nominal_timestamp_seconds: float,
    ) -> PoseEstimate:
        assert image_rgb.shape == (24, 32, 3)
        if frame_index == 1:
            return PoseEstimate((), frame_index * 100)
        return PoseEstimate(
            (
                CanonicalLandmark(
                    frame_index,
                    nominal_timestamp_seconds,
                    11,
                    "left_shoulder",
                    0.25,
                    0.5,
                    -0.1,
                    0.8,
                    0.9,
                    None,
                ),
            ),
            frame_index * 100,
        )

    def close(self) -> None:
        self.closed = True


def _write_synthetic_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 writer unavailable")
    try:
        for value in (20, 80, 140):
            writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    finally:
        writer.release()


def _landmark(
    frame_index: int = 7,
    timestamp: float = 0.35,
    landmark_id: int = 11,
    name: str = "left_shoulder",
    x: float = 0.25,
    y: float = 0.5,
) -> CanonicalLandmark:
    return CanonicalLandmark(
        frame_index=frame_index,
        nominal_timestamp_seconds=timestamp,
        landmark_id=landmark_id,
        landmark_name=name,
        x_normalized=x,
        y_normalized=y,
        z_backend_relative=None,
        visibility=None,
        presence=None,
        confidence=None,
    )


def _video_metadata(tmp_path: Path) -> VideoMetadata:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic source")
    destination = tmp_path / "outputs" / "source"
    destination.mkdir(parents=True)
    return VideoMetadata(
        source_path=source.resolve(),
        filename=source.name,
        extension=".mp4",
        container_indicator="mp4",
        file_size_bytes=source.stat().st_size,
        sha256="source-digest",
        inspected_at_utc="2026-01-01T00:00:00Z",
        width_pixels=32,
        height_pixels=24,
        nominal_fps=20.0,
        nominal_frame_count=3,
        nominal_duration_seconds=0.15,
        opencv_version="test",
        capture_backend="test",
        orientation_degrees=0,
        orientation_status="zero",
        auto_orientation_status="disabled",
        readability_status="opened",
        decode_status="decoded",
        seek_verification_method="synthetic",
        sampling_method="synthetic",
        sampled_frames=(),
        artifact_directory=destination,
        metadata_path=destination / "video_metadata.json",
    )


def test_pose_pipeline_preserves_frame_statuses_and_canonical_csv(
    tmp_path: Path,
) -> None:
    source = tmp_path / "synthetic.mp4"
    _write_synthetic_video(source)
    estimator = SyntheticEstimator()

    artifacts = estimate_pose_video(source, estimator, tmp_path / "outputs")

    assert estimator.closed
    assert [frame.status.value for frame in artifacts.frames] == [
        "decoded_pose",
        "decoded_no_pose",
        "decoded_pose",
    ]
    assert [
        frame.nominal_timestamp_seconds for frame in artifacts.frames
    ] == pytest.approx([0.0, 0.1, 0.2])
    with artifacts.raw_landmarks_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert tuple(reader.fieldnames or ()) == LANDMARK_FIELDS
        landmark_rows = list(reader)
    assert [row["frame_index"] for row in landmark_rows] == ["0", "2"]
    assert landmark_rows[0]["landmark_name"] == "left_shoulder"
    assert landmark_rows[0]["z_backend_relative"] == "-0.1"
    assert landmark_rows[0]["visibility"] == "0.8"
    assert landmark_rows[0]["presence"] == "0.9"
    assert landmark_rows[0]["confidence"] == ""
    with artifacts.pose_frames_path.open(newline="", encoding="utf-8") as file:
        frame_rows = list(csv.DictReader(file))
    assert [row["status"] for row in frame_rows] == [
        "decoded_pose",
        "decoded_no_pose",
        "decoded_pose",
    ]
    assert [row["backend_timestamp_milliseconds"] for row in frame_rows] == [
        "0",
        "100",
        "200",
    ]
    metadata = json.loads(artifacts.pose_metadata_path.read_text(encoding="utf-8"))
    assert metadata["frame_counts"]["pose_detected"] == 2
    assert metadata["frame_counts"]["decoded_no_pose"] == 1
    assert (
        "visibility is not substituted" in metadata["confidence_fields"]["confidence"]
    )
    capture = cv2.VideoCapture(str(artifacts.annotated_video_path))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 32
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 24
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0, abs=0.1)
    finally:
        capture.release()


def test_canonical_csv_serializes_all_nullable_landmark_fields(tmp_path: Path) -> None:
    frame = PoseFrameResult(
        frame_index=7,
        nominal_timestamp_seconds=0.35,
        status=PoseFrameStatus.DECODED_POSE,
        landmarks=(_landmark(),),
    )

    _write_csv_artifacts(tmp_path, [frame])

    with (tmp_path / "raw_landmarks.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows == [
        {
            "frame_index": "7",
            "nominal_timestamp_seconds": "0.35",
            "landmark_id": "11",
            "landmark_name": "left_shoulder",
            "x_normalized": "0.25",
            "y_normalized": "0.5",
            "z_backend_relative": "",
            "visibility": "",
            "presence": "",
            "confidence": "",
        }
    ]
    with (tmp_path / "pose_frames.csv").open(newline="", encoding="utf-8") as file:
        frame_rows = list(csv.DictReader(file))
    assert frame_rows[0]["backend_timestamp_milliseconds"] == ""


def test_decode_failure_serializes_blank_backend_timestamp(tmp_path: Path) -> None:
    frame = PoseFrameResult(
        frame_index=3,
        nominal_timestamp_seconds=0.15,
        status=PoseFrameStatus.DECODE_FAILURE,
        landmarks=(),
        detail="synthetic failure",
    )

    _write_csv_artifacts(tmp_path, [frame])

    with (tmp_path / "pose_frames.csv").open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    assert row["backend_timestamp_milliseconds"] == ""
    assert row["status"] == "decode_failure"


def test_render_pose_uses_unclamped_pixels_connections_and_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((11, 21, 3), dtype=np.uint8)
    landmarks = (
        _landmark(landmark_id=11, x=-0.1, y=0.0),
        _landmark(landmark_id=12, name="right_shoulder", x=1.1, y=1.0),
    )
    lines: list[tuple[tuple[int, int], tuple[int, int]]] = []
    circles: list[tuple[int, int]] = []
    labels: list[str] = []
    rectangles: list[tuple[tuple[int, int], tuple[int, int], int]] = []

    monkeypatch.setattr(
        cv2,
        "line",
        lambda _image, start, end, *_args: lines.append((start, end)),
    )
    monkeypatch.setattr(
        cv2,
        "circle",
        lambda _image, center, *_args: circles.append(center),
    )
    monkeypatch.setattr(
        cv2,
        "putText",
        lambda _image, text, *_args: labels.append(text),
    )
    monkeypatch.setattr(cv2, "getTextSize", lambda *_args: ((40, 8), 2))
    monkeypatch.setattr(
        cv2,
        "rectangle",
        lambda _image, start, end, _color, thickness: rectangles.append(
            (start, end, thickness)
        ),
    )

    rendered = _render_pose(image, landmarks, 7, 0.35, PoseFrameStatus.DECODED_POSE)

    assert rendered is not image
    assert np.array_equal(image, np.zeros_like(image))
    assert lines == [((-2, 0), (22, 10))]
    assert circles == [(-2, 0), (22, 10)]
    assert labels == ["frame 7 | nominal 0.350s | decoded_pose"]
    assert rectangles == [((1, 0), (20, 10), -1)]


def test_pose_metadata_distinguishes_source_backend_model_and_counts(
    tmp_path: Path,
) -> None:
    video = _video_metadata(tmp_path)
    estimator = SimpleNamespace(
        provenance={
            "backend": "synthetic-test",
            "library": "synthetic-library",
            "library_version": "1.2.3",
            "model_filename": "not-a-real-model.task",
            "model_sha256": "model-digest",
        }
    )
    frames = [
        PoseFrameResult(0, 0.0, PoseFrameStatus.DECODED_POSE, (_landmark(0, 0.0),)),
        PoseFrameResult(1, 0.05, PoseFrameStatus.DECODED_NO_POSE, ()),
        PoseFrameResult(
            2, 0.1, PoseFrameStatus.DECODE_FAILURE, (), "synthetic decode failure"
        ),
    ]

    payload = _metadata_payload(video, estimator, frames, "disabled for Step 2")  # type: ignore[arg-type]

    assert payload["source"]["sha256"] == "source-digest"
    assert payload["backend"]["backend"] == "synthetic-test"
    assert payload["backend"]["model_sha256"] == "model-digest"
    assert payload["frame_counts"] == {
        "expected_nominal": 3,
        "attempted": 3,
        "decoded": 2,
        "pose_detected": 1,
        "decoded_no_pose": 1,
        "decode_failure": 1,
        "annotated_frames_written": 3,
    }
    assert payload["outputs"]["raw_landmarks"] == "raw_landmarks.csv"
    assert "outside [0, 1]" in payload["coordinates"]["x_normalized"]
    assert "hip" in payload["coordinates"]["z_backend_relative"]
    assert payload["source"]["step_2_processing_auto_orientation_status"] == (
        "disabled for Step 2"
    )
    assert payload["backend_field_policy"]["omitted_backend_outputs"] == [
        "world landmarks",
        "segmentation masks",
    ]
    assert (
        "not a frame or pose quality"
        in payload["backend_field_policy"]["landmark_count"]
    )


def test_invalid_step1_input_error_propagates_without_pose_artifacts(
    tmp_path: Path,
) -> None:
    estimator = SyntheticEstimator()
    output_root = tmp_path / "outputs"

    with pytest.raises(VideoNotFoundError, match="Video does not exist"):
        estimate_pose_video(tmp_path / "missing.mp4", estimator, output_root)

    assert estimator.closed
    assert not output_root.exists()


def test_pose_publication_replaces_only_step2_artifacts(tmp_path: Path) -> None:
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "video_metadata.json").write_text("step 1", encoding="utf-8")
    sample_directory = destination / "sample_frames"
    sample_directory.mkdir()
    (sample_directory / "sample.jpg").write_bytes(b"sample")
    for name in POSE_ARTIFACT_NAMES:
        (destination / name).write_text(f"old {name}", encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    for name in POSE_ARTIFACT_NAMES:
        (staging / name).write_text(f"new {name}", encoding="utf-8")

    _publish_pose_artifacts(staging, destination)

    assert (destination / "video_metadata.json").read_text(encoding="utf-8") == "step 1"
    assert (sample_directory / "sample.jpg").read_bytes() == b"sample"
    assert all(
        (destination / name).read_text(encoding="utf-8") == f"new {name}"
        for name in POSE_ARTIFACT_NAMES
    )
    assert not list(destination.glob(".*.backup-*"))


def test_pose_publication_failure_restores_all_previous_step2_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "destination"
    destination.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    for name in POSE_ARTIFACT_NAMES:
        (destination / name).write_text(f"old {name}", encoding="utf-8")
        (staging / name).write_text(f"new {name}", encoding="utf-8")
    original_replace = Path.replace

    def fail_third_staged_publish(path: Path, target: Path) -> Path:
        if path.parent == staging and path.name == POSE_ARTIFACT_NAMES[2]:
            raise OSError("synthetic publication failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_third_staged_publish)

    with pytest.raises(ArtifactPublishError, match="atomically publish"):
        _publish_pose_artifacts(staging, destination)

    assert all(
        (destination / name).read_text(encoding="utf-8") == f"old {name}"
        for name in POSE_ARTIFACT_NAMES
    )
    assert not list(destination.glob(".*.backup-*"))


def test_annotated_writer_open_failure_preserves_existing_step1_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video_metadata(tmp_path)
    video.metadata_path.write_text("step 1 metadata", encoding="utf-8")
    estimator = SyntheticEstimator()
    writer = SimpleNamespace(isOpened=lambda: False, release=lambda: None)

    class FakeVideoWriter:
        fourcc = staticmethod(cv2.VideoWriter.fourcc)

        def __new__(cls, *_args: object) -> Any:
            return writer

    monkeypatch.setattr(
        "gait_stability.pose_pipeline.load_matching_video_metadata",
        lambda _source, _output_root: video,
    )
    monkeypatch.setattr("gait_stability.pose_pipeline.cv2.VideoWriter", FakeVideoWriter)

    with pytest.raises(AnnotatedVideoWriteError, match="Could not open"):
        estimate_pose_video(video.source_path, estimator, tmp_path / "outputs")

    assert estimator.closed
    assert video.metadata_path.read_text(encoding="utf-8") == "step 1 metadata"
    assert not any(
        (video.artifact_directory / name).exists() for name in POSE_ARTIFACT_NAMES
    )
    assert not list(
        video.artifact_directory.parent.glob(
            f".{video.artifact_directory.name}.pose-staging-*"
        )
    )


def test_raster_mismatch_closes_resources_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video_metadata(tmp_path)
    video.metadata_path.write_text("step 1 metadata", encoding="utf-8")
    sample = video.artifact_directory / "sample_frames" / "sample.jpg"
    sample.parent.mkdir()
    sample.write_bytes(b"step 1 sample")
    estimator = SyntheticEstimator()

    class FakeWriter:
        def __init__(self) -> None:
            self.released = False
            self.writes = 0

        def isOpened(self) -> bool:
            return True

        def write(self, _image: Any) -> None:
            self.writes += 1

        def release(self) -> None:
            self.released = True

    writer = FakeWriter()
    monkeypatch.setattr(
        "gait_stability.pose_pipeline.load_matching_video_metadata",
        lambda _source, _output_root: video,
    )
    monkeypatch.setattr(
        "gait_stability.pose_pipeline.decode_video_frames",
        lambda _source, _video: iter(
            [DecodedVideoFrame(0, 0.0, "decoded", np.zeros((32, 24, 3)))]
        ),
    )

    class FakeVideoWriter:
        fourcc = staticmethod(cv2.VideoWriter.fourcc)

        def __new__(cls, *_args: object) -> Any:
            return writer

    monkeypatch.setattr("gait_stability.pose_pipeline.cv2.VideoWriter", FakeVideoWriter)

    with pytest.raises(
        PosePipelineError,
        match=r"Decoded frame 0.*expected \(24, 32, 3\).*3-channel BGR",
    ):
        estimate_pose_video(video.source_path, estimator, tmp_path / "outputs")

    assert estimator.closed
    assert writer.released
    assert writer.writes == 0
    assert video.metadata_path.read_text(encoding="utf-8") == "step 1 metadata"
    assert sample.read_bytes() == b"step 1 sample"
    assert not any(
        (video.artifact_directory / name).exists() for name in POSE_ARTIFACT_NAMES
    )
    assert not list(
        video.artifact_directory.parent.glob(
            f".{video.artifact_directory.name}.pose-staging-*"
        )
    )
