from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from gait_stability import video_ingestion


class FakeImage:
    size = 1


class FakeCapture:
    def __init__(
        self,
        *,
        opened: bool = True,
        width: float = 640.0,
        height: float = 480.0,
        fps: float = 25.0,
        frame_count: float = 101.0,
        orientation: float = 0.0,
        decode: bool = True,
        decode_fail_indices: set[int] | None = None,
    ) -> None:
        self.opened = opened
        self.decode = decode
        self.decode_fail_indices = decode_fail_indices or set()
        self.decode_attempts: list[int] = []
        self.released = False
        self.position = 0.0
        self.properties = {
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_COUNT: frame_count,
        }
        orientation_property = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
        if orientation_property is not None:
            self.properties[orientation_property] = orientation

    def isOpened(self) -> bool:
        return self.opened

    def release(self) -> None:
        self.released = True

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            return self.position
        return self.properties[property_id]

    def set(self, property_id: int, value: float) -> bool:
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            self.position = value
            return True
        assert property_id == cv2.CAP_PROP_ORIENTATION_AUTO
        assert value == 0.0
        return True

    def read(self) -> tuple[bool, FakeImage | None]:
        frame_index = int(self.position)
        self.decode_attempts.append(frame_index)
        if not self.decode or frame_index in self.decode_fail_indices:
            return False, None
        self.position += 1
        return True, FakeImage()

    def getBackendName(self) -> str:
        return "FAKE_BACKEND"


def make_source(tmp_path: Path, name: str = "walk.mp4") -> Path:
    source = tmp_path / name
    source.write_bytes(b"deterministic fake video bytes")
    return source


def install_capture(monkeypatch: pytest.MonkeyPatch, capture: FakeCapture) -> list[str]:
    opened_paths: list[str] = []

    def fake_video_capture(path: str) -> FakeCapture:
        opened_paths.append(path)
        return capture

    monkeypatch.setattr(video_ingestion.cv2, "VideoCapture", fake_video_capture)
    return opened_paths


def install_file_writing_imwrite(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    written_paths: list[Path] = []

    def fake_imwrite(path: str, image: Any) -> bool:
        assert isinstance(image, FakeImage)
        destination = Path(path)
        destination.write_bytes(f"fake image {destination.name}".encode())
        written_paths.append(destination)
        return True

    monkeypatch.setattr(video_ingestion.cv2, "imwrite", fake_imwrite)
    return written_paths


def assert_no_destination_or_staging(output_root: Path, stem: str) -> None:
    assert not (output_root / stem).exists()
    assert not list(output_root.glob(f".{stem}.staging-*"))


def test_sample_indices_uses_exact_quarter_positions() -> None:
    assert video_ingestion._sample_indices(101) == (0, 25, 50, 75, 100)


def test_sample_indices_short_video_deduplicates_in_order() -> None:
    assert video_ingestion._sample_indices(3) == (0, 1, 2)
    assert video_ingestion._sample_indices(1) == (0,)


def test_inspect_video_writes_complete_metadata_and_artifact_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_source(tmp_path, "Subject Walk.MOV")
    output_root = tmp_path / "results"
    capture = FakeCapture()
    opened_paths = install_capture(monkeypatch, capture)
    staged_image_paths = install_file_writing_imwrite(monkeypatch)

    metadata = video_ingestion.inspect_video(source, output_root)

    destination = output_root / "Subject Walk"
    metadata_path = destination / "video_metadata.json"
    expected_indices = (0, 25, 50, 75, 100)
    expected_relative_paths = tuple(
        f"sample_frames/frame_{index:03d}.jpg" for index in expected_indices
    )
    assert opened_paths == [str(source.resolve())]
    assert capture.released
    assert metadata.source_path == source.resolve()
    assert metadata.filename == "Subject Walk.MOV"
    assert metadata.extension == ".mov"
    assert metadata.container_indicator == "mov"
    assert metadata.file_size_bytes == len(b"deterministic fake video bytes")
    assert (
        metadata.sha256 == hashlib.sha256(b"deterministic fake video bytes").hexdigest()
    )
    assert metadata.inspected_at_utc.endswith("Z")
    assert metadata.width_pixels == 640
    assert metadata.height_pixels == 480
    assert metadata.nominal_fps == 25.0
    assert metadata.nominal_frame_count == 101
    assert metadata.nominal_duration_seconds == pytest.approx(4.04)
    assert metadata.opencv_version == cv2.__version__
    assert metadata.capture_backend == "FAKE_BACKEND"
    assert metadata.orientation_degrees == 0
    assert metadata.orientation_status.startswith("zero:")
    assert metadata.auto_orientation_status.startswith("disabled:")
    assert metadata.readability_status == "opened"
    assert metadata.decode_status == "all selected frames decoded exactly"
    assert metadata.seek_verification_method.startswith(
        "Each exact or fallback attempt re-seeks on the same capture"
    )
    assert "failed decode may leave some backends unrecoverable" in (
        metadata.seek_verification_method
    )
    assert "backend-report dependent" in metadata.seek_verification_method
    assert metadata.sampling_method == video_ingestion.SAMPLING_METHOD
    assert metadata.artifact_directory == destination.resolve()
    assert metadata.metadata_path == metadata_path.resolve()
    assert (
        tuple(record.requested_frame_index for record in metadata.sampled_frames)
        == expected_indices
    )
    assert (
        tuple(record.frame_index for record in metadata.sampled_frames)
        == expected_indices
    )
    assert tuple(
        record.requested_nominal_timestamp_seconds for record in metadata.sampled_frames
    ) == pytest.approx((0.0, 1.0, 2.0, 3.0, 4.0))
    assert tuple(
        record.nominal_timestamp_seconds for record in metadata.sampled_frames
    ) == pytest.approx((0.0, 1.0, 2.0, 3.0, 4.0))
    assert (
        tuple(record.relative_image_path for record in metadata.sampled_frames)
        == expected_relative_paths
    )
    assert {record.fallback_distance_frames for record in metadata.sampled_frames} == {
        0
    }
    assert {record.decode_status for record in metadata.sampled_frames} == {
        "decoded_exact"
    }

    assert destination.is_dir()
    assert metadata_path.is_file()
    published_paths = sorted(
        path.relative_to(destination).as_posix() for path in destination.rglob("*")
    )
    assert published_paths == [
        "sample_frames",
        *expected_relative_paths,
        "video_metadata.json",
    ]
    assert all(
        (destination / relative_path).is_file()
        for relative_path in expected_relative_paths
    )
    assert all(".staging-" in str(path.parent.parent) for path in staged_image_paths)
    expected_json = json.loads(json.dumps(metadata.to_dict()))
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == expected_json
    assert expected_json["auto_orientation_status"].startswith("disabled:")


@pytest.mark.parametrize(
    ("reported", "expected_degrees", "status_prefix"),
    [
        (0.0, 0, "zero:"),
        (360.0, 0, "zero:"),
        (90.0, 90, "reported:"),
        (-90.0, 270, "reported:"),
        (45.0, None, "invalid:"),
        (float("nan"), None, "invalid:"),
    ],
)
def test_capture_orientation_classifies_reported_values(
    reported: float, expected_degrees: int | None, status_prefix: str
) -> None:
    orientation, status = video_ingestion._capture_orientation(
        FakeCapture(orientation=reported)
    )

    assert orientation == expected_degrees
    assert status.startswith(status_prefix)


def test_capture_orientation_is_unavailable_without_opencv_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(video_ingestion.cv2, "CAP_PROP_ORIENTATION_META")

    orientation, status = video_ingestion._capture_orientation(FakeCapture())

    assert orientation is None
    assert status.startswith("unavailable:")


def test_disable_auto_orientation_is_unsupported_without_opencv_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(video_ingestion.cv2, "CAP_PROP_ORIENTATION_AUTO")

    status = video_ingestion._disable_auto_orientation(FakeCapture())

    assert status.startswith("unsupported:")


def test_disable_auto_orientation_records_rejected_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeCapture()
    monkeypatch.setattr(capture, "set", lambda _property_id, _value: False)

    status = video_ingestion._disable_auto_orientation(capture)

    assert status.startswith("rejected:")


def test_disable_auto_orientation_records_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeCapture()

    def raise_backend_error(_property_id: int, _value: float) -> bool:
        raise cv2.error("simulated auto-orientation error")

    monkeypatch.setattr(capture, "set", raise_backend_error)

    status = video_ingestion._disable_auto_orientation(capture)

    assert status.startswith("error:")


@pytest.mark.parametrize(
    ("kind", "expected_error"),
    [
        ("missing", video_ingestion.VideoNotFoundError),
        ("directory", video_ingestion.VideoNotAFileError),
        ("extension", video_ingestion.UnsupportedVideoFormatError),
    ],
)
def test_inspect_video_rejects_invalid_source_paths(
    tmp_path: Path,
    kind: str,
    expected_error: type[video_ingestion.VideoInspectionError],
) -> None:
    source = tmp_path / "walk.mp4"
    if kind == "directory":
        source.mkdir()
    elif kind == "extension":
        source = make_source(tmp_path, "walk.avi")

    with pytest.raises(expected_error):
        video_ingestion.inspect_video(source, tmp_path / "output")

    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("descendant", [Path("walk.mp4"), Path("raw") / "walk.mp4"])
def test_inspect_video_rejects_source_inside_output_destination_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    descendant: Path,
) -> None:
    output_root = tmp_path / "outputs"
    destination = output_root / "walk"
    source = destination / descendant
    source.parent.mkdir(parents=True)
    original_bytes = b"original video must be preserved"
    source.write_bytes(original_bytes)
    existing_paths = sorted(
        path.relative_to(destination) for path in destination.rglob("*")
    )
    opened_paths: list[str] = []

    def fail_if_capture_opens(path: str) -> None:
        opened_paths.append(path)
        pytest.fail("VideoCapture must not open an unsafe source configuration")

    monkeypatch.setattr(video_ingestion.cv2, "VideoCapture", fail_if_capture_opens)

    with pytest.raises(
        video_ingestion.SourceInsideOutputDestinationError,
        match=(
            r"would contain and could replace or delete source video"
            r".*different output root"
        ),
    ):
        video_ingestion.inspect_video(source, output_root)

    assert source.read_bytes() == original_bytes
    assert opened_paths == []
    assert (
        sorted(path.relative_to(destination) for path in destination.rglob("*"))
        == existing_paths
    )
    assert not list(output_root.glob(".walk.staging-*"))
    assert not list(output_root.glob(".walk.backup-*"))


def test_inspect_video_allows_source_in_sibling_of_output_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_directory = tmp_path / "walk"
    source_directory.mkdir()
    source = make_source(source_directory, "walk.mp4")
    output_root = tmp_path / "outputs"
    capture = FakeCapture()
    opened_paths = install_capture(monkeypatch, capture)
    install_file_writing_imwrite(monkeypatch)

    metadata = video_ingestion.inspect_video(source, output_root)

    assert opened_paths == [str(source.resolve())]
    assert source.read_bytes() == b"deterministic fake video bytes"
    assert metadata.artifact_directory == (output_root / "walk").resolve()


def test_inspect_video_rejects_capture_that_cannot_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_source(tmp_path)
    capture = FakeCapture(opened=False)
    install_capture(monkeypatch, capture)
    output_root = tmp_path / "output"

    with pytest.raises(video_ingestion.VideoOpenError):
        video_ingestion.inspect_video(source, output_root)

    assert capture.released
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("capture_options", "expected_error"),
    [
        ({"fps": 0.0}, video_ingestion.InvalidVideoFPSError),
        ({"fps": float("nan")}, video_ingestion.InvalidVideoFPSError),
        ({"frame_count": 0.0}, video_ingestion.InvalidVideoFrameCountError),
        ({"frame_count": 10.5}, video_ingestion.InvalidVideoFrameCountError),
        ({"width": 0.0}, video_ingestion.InvalidVideoResolutionError),
        ({"height": float("nan")}, video_ingestion.InvalidVideoResolutionError),
    ],
)
def test_inspect_video_rejects_invalid_metadata_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_options: dict[str, float],
    expected_error: type[video_ingestion.VideoInspectionError],
) -> None:
    source = make_source(tmp_path)
    capture = FakeCapture(**capture_options)
    install_capture(monkeypatch, capture)
    output_root = tmp_path / "output"

    with pytest.raises(expected_error):
        video_ingestion.inspect_video(source, output_root)

    assert capture.released
    assert_no_destination_or_staging(output_root, source.stem)


@pytest.mark.parametrize(
    ("property_id", "expected_error", "message"),
    [
        (
            cv2.CAP_PROP_FRAME_WIDTH,
            video_ingestion.InvalidVideoResolutionError,
            "width metadata",
        ),
        (
            cv2.CAP_PROP_FRAME_HEIGHT,
            video_ingestion.InvalidVideoResolutionError,
            "height metadata",
        ),
        (
            cv2.CAP_PROP_FPS,
            video_ingestion.InvalidVideoFPSError,
            "nominal FPS metadata",
        ),
        (
            cv2.CAP_PROP_FRAME_COUNT,
            video_ingestion.InvalidVideoFrameCountError,
            "frame count metadata",
        ),
    ],
)
def test_metadata_property_errors_are_translated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    property_id: int,
    expected_error: type[video_ingestion.VideoInspectionError],
    message: str,
) -> None:
    source = make_source(tmp_path)
    capture = FakeCapture()
    original_get = capture.get

    def get_with_error(requested_property_id: int) -> float:
        if requested_property_id == property_id:
            raise cv2.error("simulated metadata read error")
        return original_get(requested_property_id)

    monkeypatch.setattr(capture, "get", get_with_error)
    install_capture(monkeypatch, capture)
    output_root = tmp_path / "output"

    with pytest.raises(expected_error, match=message) as raised:
        video_ingestion.inspect_video(source, output_root)

    assert isinstance(raised.value.__cause__, cv2.error)
    assert capture.released
    assert_no_destination_or_staging(output_root, source.stem)


@pytest.mark.parametrize(
    "operation", ["seek", "position_before", "read", "position_after"]
)
def test_selected_frame_opencv_errors_are_translated_with_frame_context(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    capture = FakeCapture()
    original_get = capture.get
    original_set = capture.set

    def set_with_error(property_id: int, value: float) -> bool:
        if operation == "seek":
            raise cv2.error("simulated seek error")
        return original_set(property_id, value)

    position_queries = 0

    def get_with_error(property_id: int) -> float:
        nonlocal position_queries
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            position_queries += 1
            if operation == "position_before" and position_queries == 1:
                raise cv2.error("simulated position error")
            if operation == "position_after" and position_queries == 2:
                raise cv2.error("simulated position error")
        return original_get(property_id)

    def read_with_error() -> tuple[bool, FakeImage | None]:
        if operation == "read":
            raise cv2.error("simulated read error")
        return FakeCapture.read(capture)

    monkeypatch.setattr(capture, "set", set_with_error)
    monkeypatch.setattr(capture, "get", get_with_error)
    monkeypatch.setattr(capture, "read", read_with_error)

    with pytest.raises(
        video_ingestion.SelectedFrameDecodeError, match="frame 17"
    ) as raised:
        video_ingestion._seek_and_decode(capture, 17)

    assert isinstance(raised.value.__cause__, cv2.error)


def test_seek_and_decode_rejects_mismatched_reported_position_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeCapture()
    original_get = capture.get

    def get_with_mismatched_position(property_id: int) -> float:
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            return 16.0
        return original_get(property_id)

    monkeypatch.setattr(capture, "get", get_with_mismatched_position)

    with pytest.raises(
        video_ingestion.SelectedFrameDecodeError,
        match=r"Seek to frame 17 reported position 16",
    ):
        video_ingestion._seek_and_decode(capture, 17)

    assert capture.decode_attempts == []


def test_seek_and_decode_rejects_reported_zero_after_decoding_frame_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeCapture()
    original_get = capture.get
    position_queries = 0

    def get_with_mismatched_position_after(property_id: int) -> float:
        nonlocal position_queries
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            position_queries += 1
            if position_queries == 2:
                return 0.0
        return original_get(property_id)

    monkeypatch.setattr(capture, "get", get_with_mismatched_position_after)

    with pytest.raises(
        video_ingestion.SelectedFrameDecodeError,
        match=r"Decoded frame 0, but OpenCV reported position 0 afterward",
    ):
        video_ingestion._seek_and_decode(capture, 0)

    assert capture.decode_attempts == [0]


def test_all_backward_decode_attempts_fail_with_explicit_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_source(tmp_path)
    capture = FakeCapture(decode=False)
    install_capture(monkeypatch, capture)
    install_file_writing_imwrite(monkeypatch)
    output_root = tmp_path / "output"

    with pytest.raises(
        video_ingestion.SelectedFrameDecodeError,
        match=r"requested frame 0; attempted inclusive frame range 0-0",
    ) as raised:
        video_ingestion.inspect_video(source, output_root)

    assert isinstance(raised.value.__cause__, video_ingestion.SelectedFrameDecodeError)
    assert_no_destination_or_staging(output_root, source.stem)


def test_terminal_frame_uses_truthful_bounded_backward_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_source(tmp_path)
    capture = FakeCapture(decode_fail_indices={100})
    install_capture(monkeypatch, capture)
    written_paths = install_file_writing_imwrite(monkeypatch)

    metadata = video_ingestion.inspect_video(source, tmp_path / "output")

    terminal = metadata.sampled_frames[-1]
    assert capture.decode_attempts[-2:] == [100, 99]
    assert terminal.requested_frame_index == 100
    assert terminal.frame_index == 99
    assert terminal.requested_nominal_timestamp_seconds == pytest.approx(4.0)
    assert terminal.nominal_timestamp_seconds == pytest.approx(3.96)
    assert terminal.fallback_distance_frames == 1
    assert terminal.relative_image_path == "sample_frames/frame_099.jpg"
    assert terminal.decode_status == "decoded_after_backward_fallback"
    assert (
        metadata.decode_status == "bounded backward fallback used for selected frames"
    )
    assert written_paths[-1].name == "frame_099.jpg"
    persisted = json.loads(metadata.metadata_path.read_text(encoding="utf-8"))
    assert persisted["sampled_frames"][-1] == {
        "requested_frame_index": 100,
        "frame_index": 99,
        "requested_nominal_timestamp_seconds": 4.0,
        "nominal_timestamp_seconds": 3.96,
        "fallback_distance_frames": 1,
        "relative_image_path": "sample_frames/frame_099.jpg",
        "decode_status": "decoded_after_backward_fallback",
    }


def test_terminal_frame_failure_reports_all_eleven_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_source(tmp_path)
    capture = FakeCapture(decode_fail_indices=set(range(90, 101)))
    install_capture(monkeypatch, capture)
    install_file_writing_imwrite(monkeypatch)
    output_root = tmp_path / "output"

    with pytest.raises(
        video_ingestion.SelectedFrameDecodeError,
        match=r"requested frame 100; attempted inclusive frame range 90-100",
    ) as raised:
        video_ingestion.inspect_video(source, output_root)

    assert capture.decode_attempts[-11:] == list(range(100, 89, -1))
    assert isinstance(raised.value.__cause__, video_ingestion.SelectedFrameDecodeError)
    assert "frame 90" in str(raised.value.__cause__)
    assert_no_destination_or_staging(output_root, source.stem)


def test_backward_fallback_reuses_an_already_written_actual_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_source(tmp_path)
    capture = FakeCapture(
        frame_count=12.0,
        decode_fail_indices={9, 10, 11},
    )
    install_capture(monkeypatch, capture)
    written_paths = install_file_writing_imwrite(monkeypatch)

    metadata = video_ingestion.inspect_video(source, tmp_path / "output")

    assert tuple(
        record.requested_frame_index for record in metadata.sampled_frames
    ) == (
        0,
        3,
        6,
        8,
        11,
    )
    earlier = metadata.sampled_frames[-2]
    terminal = metadata.sampled_frames[-1]
    assert terminal.frame_index == earlier.frame_index == 8
    assert terminal.fallback_distance_frames == 3
    assert terminal.relative_image_path == earlier.relative_image_path
    assert terminal.decode_status == "decoded_after_backward_fallback_reused_sample"
    assert [path.name for path in written_paths].count("frame_08.jpg") == 1
    assert len(written_paths) == 4
    persisted = json.loads(metadata.metadata_path.read_text(encoding="utf-8"))
    assert len(persisted["sampled_frames"]) == 5
    assert persisted["sampled_frames"][-1]["requested_frame_index"] == 11
    assert persisted["sampled_frames"][-1]["frame_index"] == 8


def test_write_failure_preserves_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_source(tmp_path)
    capture = FakeCapture()
    install_capture(monkeypatch, capture)
    monkeypatch.setattr(video_ingestion.cv2, "imwrite", lambda _path, _image: False)
    output_root = tmp_path / "output"
    destination = output_root / source.stem
    destination.mkdir(parents=True)
    old_artifact = destination / "old.txt"
    old_artifact.write_text("keep me", encoding="utf-8")

    with pytest.raises(video_ingestion.SelectedFrameWriteError):
        video_ingestion.inspect_video(source, output_root)

    assert destination.is_dir()
    assert old_artifact.read_text(encoding="utf-8") == "keep me"
    assert list(destination.iterdir()) == [old_artifact]
    assert not list(output_root.glob(f".{source.stem}.staging-*"))


def test_successful_inspection_replaces_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_source(tmp_path)
    install_capture(monkeypatch, FakeCapture())
    install_file_writing_imwrite(monkeypatch)
    output_root = tmp_path / "output"
    destination = output_root / source.stem
    destination.mkdir(parents=True)
    obsolete = destination / "obsolete.txt"
    obsolete.write_text("old output", encoding="utf-8")

    metadata = video_ingestion.inspect_video(source, output_root)

    assert metadata.artifact_directory == destination.resolve()
    assert not obsolete.exists()
    assert (destination / "video_metadata.json").is_file()
    assert not list(output_root.glob(f".{source.stem}.backup-*"))


def test_backup_cleanup_failure_warns_after_successful_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifacts"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")

    def fail_cleanup(_path: Path) -> None:
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(video_ingestion.shutil, "rmtree", fail_cleanup)

    with pytest.warns(RuntimeWarning, match="old-output cleanup failed"):
        video_ingestion._publish_staging(staging, destination)

    assert (destination / "new.txt").read_text(encoding="utf-8") == "new"
    backups = list(tmp_path.glob(".artifacts.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "old.txt").read_text(encoding="utf-8") == "old"


def test_publish_failure_restores_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "artifacts"
    destination.mkdir()
    old_artifact = destination / "old.txt"
    old_artifact.write_text("old", encoding="utf-8")
    missing_staging = tmp_path / "missing-staging"

    with pytest.raises(video_ingestion.ArtifactPublishError, match="Could not publish"):
        video_ingestion._publish_staging(missing_staging, destination)

    assert old_artifact.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".artifacts.backup-*"))


def test_inspect_generated_mp4_with_real_opencv_io(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.mp4"
    width, height = 32, 24
    frame_count = 9
    fps = 10.0
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 writer could not open with the mp4v codec")
    try:
        for index in range(frame_count):
            frame = np.full((height, width, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    output_root = tmp_path / "output"
    metadata = video_ingestion.inspect_video(source, output_root)

    assert metadata.width_pixels == width
    assert metadata.height_pixels == height
    assert metadata.nominal_fps == pytest.approx(fps)
    assert metadata.nominal_frame_count == frame_count
    assert metadata.nominal_duration_seconds == pytest.approx(frame_count / fps)
    assert tuple(record.frame_index for record in metadata.sampled_frames) == (
        0,
        2,
        4,
        6,
        8,
    )
    assert metadata.seek_verification_method.startswith(
        "Each exact or fallback attempt re-seeks on the same capture"
    )
    assert metadata.metadata_path.is_file()
    assert all(
        (metadata.artifact_directory / record.relative_image_path).is_file()
        for record in metadata.sampled_frames
    )
    persisted = json.loads(metadata.metadata_path.read_text(encoding="utf-8"))
    assert persisted["sampled_frames"] == [
        {
            "requested_frame_index": record.requested_frame_index,
            "frame_index": record.frame_index,
            "requested_nominal_timestamp_seconds": (
                record.requested_nominal_timestamp_seconds
            ),
            "nominal_timestamp_seconds": record.nominal_timestamp_seconds,
            "fallback_distance_frames": record.fallback_distance_frames,
            "relative_image_path": record.relative_image_path,
            "decode_status": "decoded_exact",
        }
        for record in metadata.sampled_frames
    ]
