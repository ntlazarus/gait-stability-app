from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import pytest

from gait_stability import video_ingestion


class FakeImage:
    size = 1


class SequentialCapture:
    def __init__(
        self,
        outcomes: list[tuple[bool, Any | None] | cv2.error],
        *,
        opened: bool = True,
        seek_outcomes: list[bool | cv2.error] | None = None,
    ) -> None:
        self.outcomes = outcomes
        self.opened = opened
        self.read_count = 0
        self.seek_calls: list[tuple[int, float]] = []
        self.released = False
        self.seek_outcomes = seek_outcomes or []

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, Any | None]:
        outcome = self.outcomes[self.read_count]
        self.read_count += 1
        if isinstance(outcome, cv2.error):
            raise outcome
        return outcome

    def set(self, property_id: int, value: float) -> bool:
        self.seek_calls.append((property_id, value))
        if self.seek_outcomes:
            outcome = self.seek_outcomes.pop(0)
            if isinstance(outcome, cv2.error):
                raise outcome
            return outcome
        return True

    def release(self) -> None:
        self.released = True


def _decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture: SequentialCapture,
    *,
    frame_count: int = 4,
    fps: float = 20.0,
) -> list[video_ingestion.DecodedVideoFrame]:
    source = tmp_path / "synthetic.mp4"
    source.write_bytes(b"synthetic")
    metadata = SimpleNamespace(nominal_frame_count=frame_count, nominal_fps=fps)
    monkeypatch.setattr(video_ingestion.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(
        video_ingestion, "_disable_auto_orientation", lambda _cap: "off"
    )
    return list(video_ingestion.decode_video_frames(source, metadata))


def test_decoder_preserves_nominal_indices_timestamps_and_recovers_sequentially(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = [FakeImage(), FakeImage(), FakeImage()]
    capture = SequentialCapture(
        [(True, images[0]), (False, None), (True, images[1]), (True, images[2])]
    )

    frames = _decode(tmp_path, monkeypatch, capture)

    assert [frame.frame_index for frame in frames] == [0, 1, 2, 3]
    assert [frame.nominal_timestamp_seconds for frame in frames] == pytest.approx(
        [0.0, 0.05, 0.1, 0.15]
    )
    assert [frame.decode_status for frame in frames] == [
        "decoded",
        "decode_failure",
        "decoded",
        "decoded",
    ]
    assert frames[1].image_bgr is None
    assert frames[1].decode_detail == (
        "OpenCV returned no decoded image; recovery seek to frame 2 accepted"
    )
    assert {frame.auto_orientation_status for frame in frames} == {"off"}
    assert capture.seek_calls == [(cv2.CAP_PROP_POS_FRAMES, 2.0)]
    assert capture.read_count == 4
    assert capture.released


@pytest.mark.parametrize(
    ("seek_outcome", "expected_detail"),
    [
        (False, "capture backend rejected seek request"),
        (cv2.error("synthetic seek exception"), "synthetic seek exception"),
    ],
)
def test_decoder_marks_remaining_slots_failed_when_recovery_seek_is_unverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seek_outcome: bool | cv2.error,
    expected_detail: str,
) -> None:
    capture = SequentialCapture(
        [(True, FakeImage()), (False, None)], seek_outcomes=[seek_outcome]
    )

    frames = _decode(tmp_path, monkeypatch, capture, frame_count=4)

    assert [frame.decode_status for frame in frames] == [
        "decoded",
        "decode_failure",
        "decode_failure",
        "decode_failure",
    ]
    assert expected_detail in (frames[1].decode_detail or "")
    assert all("not attempted" in (frame.decode_detail or "") for frame in frames[2:])
    assert capture.read_count == 2
    assert capture.released


def test_decoder_records_opencv_exception_and_keeps_recovery_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = SequentialCapture(
        [
            cv2.error("synthetic decode exception"),
            (False, None),
            (True, FakeImage()),
        ]
    )

    frames = _decode(tmp_path, monkeypatch, capture, frame_count=3, fps=25.0)

    assert [frame.decode_status for frame in frames] == [
        "decode_failure",
        "decode_failure",
        "decoded",
    ]
    assert "synthetic decode exception" in (frames[0].decode_detail or "")
    assert capture.seek_calls == [
        (cv2.CAP_PROP_POS_FRAMES, 1.0),
        (cv2.CAP_PROP_POS_FRAMES, 2.0),
    ]
    assert capture.read_count == 3
    assert capture.released


def test_decoder_open_failure_releases_capture_and_propagates_step1_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = SequentialCapture([], opened=False)
    source = tmp_path / "synthetic.mp4"
    source.write_bytes(b"synthetic")
    metadata = SimpleNamespace(nominal_frame_count=1, nominal_fps=25.0)
    monkeypatch.setattr(video_ingestion.cv2, "VideoCapture", lambda _path: capture)

    with pytest.raises(video_ingestion.VideoOpenError, match="could not open"):
        list(video_ingestion.decode_video_frames(source, metadata))

    assert capture.released
