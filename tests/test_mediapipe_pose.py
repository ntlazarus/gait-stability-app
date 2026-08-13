from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from gait_stability.mediapipe_pose import MediaPipePoseError, MediaPipePoseEstimator
from gait_stability.pose_contracts import MEDIAPIPE_LANDMARK_NAMES


class FakeLandmarker:
    def __init__(self, results: list[Any]) -> None:
        self.results = results
        self.timestamps: list[int] = []
        self.closed = False

    def detect_for_video(self, _image: Any, timestamp_ms: int) -> Any:
        self.timestamps.append(timestamp_ms)
        return self.results.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeMediaPipe:
    class ImageFormat:
        SRGB = "srgb"

    def __init__(self) -> None:
        self.images: list[Any] = []

    def Image(self, **kwargs: Any) -> Any:  # noqa: N802 - mirrors MediaPipe API
        image = SimpleNamespace(**kwargs)
        self.images.append(image)
        return image


def _estimator(results: list[Any]) -> tuple[MediaPipePoseEstimator, FakeLandmarker]:
    estimator = MediaPipePoseEstimator.__new__(MediaPipePoseEstimator)
    estimator._mp = FakeMediaPipe()
    estimator._last_timestamp_ms = -1
    landmarker = FakeLandmarker(results)
    estimator._landmarker = landmarker
    return estimator, landmarker


def test_mediapipe_adapter_maps_all_33_names_confidences_and_relative_z() -> None:
    backend_landmarks = [
        SimpleNamespace(
            x=index / 100,
            y=index / 200,
            z=-index / 300,
            visibility=None if index == 0 else 0.8,
            presence=None if index == 1 else 0.9,
        )
        for index in range(33)
    ]
    estimator, _landmarker = _estimator(
        [SimpleNamespace(pose_landmarks=[backend_landmarks])]
    )
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    estimate = estimator.estimate_frame(image, 4, 0.125)
    landmarks = estimate.landmarks

    assert len(landmarks) == 33
    assert tuple(item.landmark_name for item in landmarks) == MEDIAPIPE_LANDMARK_NAMES
    assert tuple(item.landmark_id for item in landmarks) == tuple(range(33))
    assert {item.frame_index for item in landmarks} == {4}
    assert {item.nominal_timestamp_seconds for item in landmarks} == {0.125}
    assert landmarks[0].visibility is None
    assert landmarks[1].presence is None
    assert landmarks[32].z_backend_relative == pytest.approx(-32 / 300)
    assert all(item.confidence is None for item in landmarks)
    assert estimator._mp.images[0].image_format == "srgb"
    assert estimator._mp.images[0].data is image
    assert estimate.backend_timestamp_milliseconds == 125


def test_mediapipe_adapter_returns_empty_tuple_for_decoded_no_pose() -> None:
    estimator, _landmarker = _estimator([SimpleNamespace(pose_landmarks=[])])

    estimate = estimator.estimate_frame(np.zeros((1, 1, 3)), 0, 0.0)

    assert estimate.landmarks == ()
    assert estimate.backend_timestamp_milliseconds == 0


def test_mediapipe_adapter_submits_strictly_increasing_integer_milliseconds() -> None:
    no_pose = SimpleNamespace(pose_landmarks=[])
    estimator, landmarker = _estimator([no_pose, no_pose, no_pose, no_pose])

    for timestamp in (0.0, 0.0, 0.0004, 0.01):
        estimator.estimate_frame(np.zeros((1, 1, 3)), 0, timestamp)

    assert landmarker.timestamps == [0, 1, 2, 10]


@pytest.mark.parametrize("count", [32, 34])
def test_mediapipe_adapter_rejects_noncanonical_landmark_count(count: int) -> None:
    backend_landmarks = [
        SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.8, presence=0.9)
        for _ in range(count)
    ]
    estimator, _landmarker = _estimator(
        [SimpleNamespace(pose_landmarks=[backend_landmarks])]
    )

    with pytest.raises(
        MediaPipePoseError, match=f"returned {count} landmarks; expected exactly 33"
    ):
        estimator.estimate_frame(np.zeros((1, 1, 3)), 0, 0.0)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("min_pose_detection_confidence", -0.01),
        ("min_pose_presence_confidence", 1.01),
        ("min_tracking_confidence", float("nan")),
    ],
)
def test_mediapipe_adapter_rejects_invalid_thresholds_before_import_or_model_use(
    tmp_path: Path, keyword: str, value: float
) -> None:
    model = tmp_path / "model.task"
    model.write_bytes(b"not a real model")

    with pytest.raises(ValueError, match=keyword):
        MediaPipePoseEstimator(model, **{keyword: value})


def test_mediapipe_adapter_rejects_missing_model_before_import(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        MediaPipePoseEstimator(tmp_path / "missing.task")


def test_mediapipe_adapter_wraps_landmarker_creation_runtime_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.task"
    model.write_bytes(b"synthetic model")
    mediapipe = ModuleType("mediapipe")
    tasks = ModuleType("mediapipe.tasks")
    python = ModuleType("mediapipe.tasks.python")
    vision = ModuleType("mediapipe.tasks.python.vision")

    class Options:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    class Landmarker:
        @staticmethod
        def create_from_options(_options: Any) -> None:
            raise RuntimeError("synthetic model initialization failure")

    python.BaseOptions = Options  # type: ignore[attr-defined]
    vision.PoseLandmarker = Landmarker  # type: ignore[attr-defined]
    vision.PoseLandmarkerOptions = Options  # type: ignore[attr-defined]
    vision.RunningMode = SimpleNamespace(VIDEO="video")  # type: ignore[attr-defined]
    mediapipe.tasks = tasks  # type: ignore[attr-defined]
    tasks.python = python  # type: ignore[attr-defined]
    python.vision = vision  # type: ignore[attr-defined]
    for name, module in {
        "mediapipe": mediapipe,
        "mediapipe.tasks": tasks,
        "mediapipe.tasks.python": python,
        "mediapipe.tasks.python.vision": vision,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    with pytest.raises(
        MediaPipePoseError,
        match=r"could not initialize Pose Landmarker.*synthetic model initialization",
    ) as raised:
        MediaPipePoseEstimator(model)

    assert isinstance(raised.value.__cause__, RuntimeError)
