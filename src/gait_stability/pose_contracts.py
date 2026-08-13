"""Backend-independent immutable contracts for raw pose estimates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class PoseFrameStatus(StrEnum):
    """Mutually exclusive outcomes for one nominal source-frame slot."""

    DECODED_POSE = "decoded_pose"
    DECODED_NO_POSE = "decoded_no_pose"
    DECODE_FAILURE = "decode_failure"


@dataclass(frozen=True, slots=True)
class CanonicalLandmark:
    """One raw model-estimated landmark in normalized image coordinates."""

    frame_index: int
    nominal_timestamp_seconds: float
    landmark_id: int
    landmark_name: str
    x_normalized: float
    y_normalized: float
    z_backend_relative: float | None
    visibility: float | None
    presence: float | None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class PoseFrameResult:
    """Pose-estimation outcome for one nominal source-frame slot."""

    frame_index: int
    nominal_timestamp_seconds: float
    status: PoseFrameStatus
    landmarks: tuple[CanonicalLandmark, ...]
    detail: str | None = None
    backend_timestamp_milliseconds: int | None = None


@dataclass(frozen=True, slots=True)
class PoseEstimate:
    """Backend-independent landmarks and their exact submission timestamp."""

    landmarks: tuple[CanonicalLandmark, ...]
    backend_timestamp_milliseconds: int | None = None


class PoseEstimator(Protocol):
    """Minimal image-level interface implemented by raw pose backends."""

    @property
    def provenance(self) -> dict[str, Any]: ...

    def estimate_frame(
        self,
        image_rgb: Any,
        frame_index: int,
        nominal_timestamp_seconds: float,
    ) -> PoseEstimate | tuple[CanonicalLandmark, ...]: ...

    def close(self) -> None: ...


MEDIAPIPE_LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

# Canonical integer indices, intentionally independent of MediaPipe drawing APIs.
CANONICAL_POSE_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (27, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (28, 32),
)


def normalized_to_pixel(value: float, extent_pixels: int) -> int:
    """Convert a normalized image coordinate to its nearest pixel coordinate."""
    return round(value * (extent_pixels - 1))
