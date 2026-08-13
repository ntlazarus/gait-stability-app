"""MediaPipe Tasks adapter; no MediaPipe types escape this module."""

from __future__ import annotations

import math
from importlib.metadata import version
from pathlib import Path
from typing import Any

from gait_stability.pose_contracts import (
    MEDIAPIPE_LANDMARK_NAMES,
    CanonicalLandmark,
    PoseEstimate,
)
from gait_stability.video_ingestion import sha256_file


class MediaPipePoseError(RuntimeError):
    """Raised when MediaPipe returns a result outside the canonical contract."""


class MediaPipePoseEstimator:
    """CPU MediaPipe Pose Landmarker configured for one pose in VIDEO mode."""

    def __init__(
        self,
        model_path: Path,
        *,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model asset does not exist: {model_path}"
            )
        thresholds = {
            "min_pose_detection_confidence": min_pose_detection_confidence,
            "min_pose_presence_confidence": min_pose_presence_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        }
        for name, value in thresholds.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1 inclusive")
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        self._mp: Any = __import__("mediapipe")
        self._model_path = model_path.resolve()
        self._thresholds = thresholds
        self._last_timestamp_ms = -1
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            output_segmentation_masks=False,
            **self._thresholds,
        )
        try:
            self._landmarker: Any = PoseLandmarker.create_from_options(options)
        except RuntimeError as exc:
            raise MediaPipePoseError(
                f"MediaPipe could not initialize Pose Landmarker with model "
                f"{self._model_path}: {exc}"
            ) from exc

    @property
    def provenance(self) -> dict[str, Any]:
        digest = sha256_file(self._model_path)
        return {
            "backend": "MediaPipe Tasks Pose Landmarker",
            "library": "mediapipe",
            "library_version": version("mediapipe"),
            "delegate": "CPU",
            "running_mode": "VIDEO",
            "num_poses": 1,
            "model_path_identifier": str(self._model_path),
            "model_filename": self._model_path.name,
            "model_size_bytes": self._model_path.stat().st_size,
            "model_sha256": digest,
            "configuration": self._thresholds,
            "backend_temporal_behavior": (
                "MediaPipe VIDEO mode may use backend-internal tracking and temporal "
                "smoothing; outputs remain raw backend pose estimates and are not "
                "project-smoothed trajectories"
            ),
        }

    def estimate_frame(
        self,
        image_rgb: Any,
        frame_index: int,
        nominal_timestamp_seconds: float,
    ) -> PoseEstimate:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = max(
            round(nominal_timestamp_seconds * 1000), self._last_timestamp_ms + 1
        )
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return PoseEstimate((), timestamp_ms)
        backend_landmarks = result.pose_landmarks[0]
        expected_count = len(MEDIAPIPE_LANDMARK_NAMES)
        if len(backend_landmarks) != expected_count:
            raise MediaPipePoseError(
                "MediaPipe returned "
                f"{len(backend_landmarks)} landmarks; expected exactly {expected_count}"
            )
        landmarks: list[CanonicalLandmark] = []
        for landmark_id, landmark in enumerate(backend_landmarks):
            landmarks.append(
                CanonicalLandmark(
                    frame_index=frame_index,
                    nominal_timestamp_seconds=nominal_timestamp_seconds,
                    landmark_id=landmark_id,
                    landmark_name=MEDIAPIPE_LANDMARK_NAMES[landmark_id],
                    x_normalized=float(landmark.x),
                    y_normalized=float(landmark.y),
                    z_backend_relative=float(landmark.z),
                    visibility=(
                        float(landmark.visibility)
                        if landmark.visibility is not None
                        else None
                    ),
                    presence=(
                        float(landmark.presence)
                        if landmark.presence is not None
                        else None
                    ),
                    confidence=None,
                )
            )
        return PoseEstimate(tuple(landmarks), timestamp_ms)

    def close(self) -> None:
        self._landmarker.close()
