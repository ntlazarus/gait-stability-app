"""Raw pose-estimation pipeline and dependency-light artifact serialization."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gait_stability.pose_contracts import (
    CANONICAL_POSE_CONNECTIONS,
    CanonicalLandmark,
    PoseEstimate,
    PoseEstimator,
    PoseFrameResult,
    PoseFrameStatus,
    normalized_to_pixel,
)
from gait_stability.video_ingestion import (
    ArtifactPublishError,
    VideoMetadata,
    decode_video_frames,
    inspect_video,
    load_matching_video_metadata,
)

POSE_ARTIFACT_NAMES = (
    "raw_landmarks.csv",
    "pose_frames.csv",
    "pose_metadata.json",
    "annotated_pose.mp4",
)
LANDMARK_FIELDS = tuple(CanonicalLandmark.__dataclass_fields__)
FRAME_FIELDS = (
    "frame_index",
    "nominal_timestamp_seconds",
    "backend_timestamp_milliseconds",
    "status",
    "landmark_count",
    "detail",
)


class PosePipelineError(Exception):
    """Expected raw-pose processing or artifact failure."""


class AnnotatedVideoWriteError(PosePipelineError):
    """Raised when the annotated video cannot be opened or written."""


@dataclass(frozen=True, slots=True)
class PoseRunArtifacts:
    """Published paths and frame results from one pose run."""

    artifact_directory: Path
    raw_landmarks_path: Path
    pose_frames_path: Path
    pose_metadata_path: Path
    annotated_video_path: Path
    frames: tuple[PoseFrameResult, ...]


def _validate_decoded_raster(
    image_bgr: Any, video: VideoMetadata, frame_index: int
) -> None:
    expected_shape = (video.height_pixels, video.width_pixels, 3)
    actual_shape = getattr(image_bgr, "shape", None)
    if not isinstance(image_bgr, np.ndarray) or actual_shape != expected_shape:
        raise PosePipelineError(
            f"Decoded frame {frame_index} has raster shape {actual_shape!r}; "
            f"expected {expected_shape!r} for a 3-channel BGR image"
        )


def _render_pose(
    image_bgr: Any,
    landmarks: tuple[CanonicalLandmark, ...],
    frame_index: int,
    timestamp_seconds: float,
    status: PoseFrameStatus,
) -> Any:
    rendered = image_bgr.copy()
    height, width = rendered.shape[:2]
    by_id = {landmark.landmark_id: landmark for landmark in landmarks}
    for start_id, end_id in CANONICAL_POSE_CONNECTIONS:
        if start_id not in by_id or end_id not in by_id:
            continue
        start = by_id[start_id]
        end = by_id[end_id]
        cv2.line(
            rendered,
            (
                normalized_to_pixel(start.x_normalized, width),
                normalized_to_pixel(start.y_normalized, height),
            ),
            (
                normalized_to_pixel(end.x_normalized, width),
                normalized_to_pixel(end.y_normalized, height),
            ),
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    for landmark in landmarks:
        cv2.circle(
            rendered,
            (
                normalized_to_pixel(landmark.x_normalized, width),
                normalized_to_pixel(landmark.y_normalized, height),
            ),
            3,
            (40, 255, 40),
            -1,
            cv2.LINE_AA,
        )
    label = f"frame {frame_index} | nominal {timestamp_seconds:.3f}s | {status.value}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    available_width = max(width - 8, 1)
    if text_width > available_width:
        font_scale *= available_width / text_width
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
    origin_x = min(4, max(width - 1, 0))
    origin_y = min(max(text_height + 4, 0), max(height - baseline - 1, 0))
    background_start = (max(origin_x - 3, 0), max(origin_y - text_height - 3, 0))
    background_end = (
        min(origin_x + text_width + 3, max(width - 1, 0)),
        min(origin_y + baseline + 3, max(height - 1, 0)),
    )
    cv2.rectangle(rendered, background_start, background_end, (20, 20, 20), -1)
    cv2.putText(
        rendered,
        label,
        (origin_x, origin_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return rendered


def _publish_pose_artifacts(staging: Path, destination: Path) -> None:
    """Publish only Step 2 files, rolling all of them back on rename failure."""
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for name in POSE_ARTIFACT_NAMES:
            target = destination / name
            staged = staging / name
            if target.exists():
                backup = destination / f".{name}.backup-{uuid.uuid4().hex}"
                target.replace(backup)
                backups[target] = backup
            staged.replace(target)
            published.append(target)
    except OSError as exc:
        for target in published:
            with suppress(OSError):
                target.unlink()
        for target, backup in backups.items():
            with suppress(OSError):
                backup.replace(target)
        raise ArtifactPublishError(
            "Could not atomically publish pose artifacts"
        ) from exc
    for backup in backups.values():
        with suppress(OSError):
            backup.unlink()


def _write_csv_artifacts(staging: Path, frames: list[PoseFrameResult]) -> None:
    with (staging / "raw_landmarks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=LANDMARK_FIELDS)
        writer.writeheader()
        for frame in frames:
            for landmark in frame.landmarks:
                writer.writerow(asdict(landmark))
    with (staging / "pose_frames.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        for frame in frames:
            writer.writerow(
                {
                    "frame_index": frame.frame_index,
                    "nominal_timestamp_seconds": frame.nominal_timestamp_seconds,
                    "backend_timestamp_milliseconds": (
                        frame.backend_timestamp_milliseconds
                    ),
                    "status": frame.status.value,
                    "landmark_count": len(frame.landmarks),
                    "detail": frame.detail,
                }
            )


def _metadata_payload(
    video: VideoMetadata,
    estimator: PoseEstimator,
    frames: list[PoseFrameResult],
    processing_auto_orientation_status: str | None = None,
) -> dict[str, Any]:
    counts = {status.value: 0 for status in PoseFrameStatus}
    for frame in frames:
        counts[frame.status.value] += 1
    decoded = (
        counts[PoseFrameStatus.DECODED_POSE] + counts[PoseFrameStatus.DECODED_NO_POSE]
    )
    return {
        "schema_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": (
            "raw monocular RGB markerless pose-model estimates only; no project "
            "interpolation, smoothing, gait events, COM, or metrics"
        ),
        "source": {
            "path_identifier": str(video.source_path),
            "filename": video.filename,
            "sha256": video.sha256,
            "width_pixels": video.width_pixels,
            "height_pixels": video.height_pixels,
            "nominal_fps": video.nominal_fps,
            "orientation_degrees": video.orientation_degrees,
            "orientation_status": video.orientation_status,
            "step_1_auto_orientation_status": video.auto_orientation_status,
            "step_2_processing_auto_orientation_status": (
                processing_auto_orientation_status
            ),
            "project_rotation": "none",
            "project_mirroring": "none",
        },
        "capture_assumptions": {
            "input": "single monocular RGB video",
            "calibration": (
                "none: no camera intrinsics, extrinsics, lens-distortion correction, "
                "physical scale, or multi-view reconstruction"
            ),
            "camera_view": "not required, declared, or verified",
            "gait_direction": "not required, declared, or verified",
            "camera_placement": "not required, declared, or verified",
            "mirroring": "not required, detected, or verified",
        },
        "frame_counts": {
            "expected_nominal": video.nominal_frame_count,
            "attempted": len(frames),
            "decoded": decoded,
            "pose_detected": counts[PoseFrameStatus.DECODED_POSE],
            "decoded_no_pose": counts[PoseFrameStatus.DECODED_NO_POSE],
            "decode_failure": counts[PoseFrameStatus.DECODE_FAILURE],
            "annotated_frames_written": len(frames),
        },
        "coordinates": {
            "x_normalized": (
                "dimensionless pose-model estimate in the RGB image plane; 0 image "
                "left, 1 image right; may be outside [0, 1]"
            ),
            "y_normalized": (
                "dimensionless pose-model estimate in the RGB image plane; 0 image "
                "top, 1 image bottom; may be outside [0, 1]"
            ),
            "z_backend_relative": (
                "learned monocular model-relative depth with the midpoint of the hips "
                "as origin and magnitude roughly on the same scale as x; smaller is "
                "closer to the camera. It is not camera depth, laboratory depth, or "
                "a metric/physical coordinate"
            ),
            "pixel_rendering": (
                "round(normalized * (dimension_pixels - 1)); stored normalized "
                "values are unchanged"
            ),
            "body_side_labels": (
                "left/right are model-assigned body-side labels, not image-side labels"
            ),
            "foot_labels": (
                "foot_index is the pose model's landmark label, not an invented toe "
                "or contact point; heel is a model landmark label, not a "
                "ground-contact location"
            ),
        },
        "confidence_fields": {
            "visibility": (
                "raw model score associated with landmark visibility; not an observed "
                "or calibrated probability, uncertainty, or accuracy measure"
            ),
            "presence": (
                "raw model score associated with landmark presence; not an observed "
                "or calibrated probability, uncertainty, or accuracy measure"
            ),
            "confidence": (
                "nullable generic field; null because MediaPipe exposes no single "
                "generic per-landmark confidence; visibility is not substituted"
            ),
        },
        "timestamps": {
            "field": "nominal_timestamp_seconds",
            "method": (
                "frame_index / nominal_fps; actual presentation timestamps unavailable"
            ),
            "backend_submission": (
                "backend_timestamp_milliseconds records the exact integer timestamp "
                "submitted for each decoded frame; blank for decode failures. "
                "MediaPipe uses rounded nominal milliseconds, increased by 1 ms only "
                "when needed for strict VIDEO-mode monotonicity"
            ),
        },
        "frame_statuses": {
            "decoded_pose": "decoded image with a nonempty backend pose result",
            "decoded_no_pose": "decoded image with an empty backend pose result",
            "decode_failure": "nominal source-frame slot without a decoded image",
        },
        "backend": estimator.provenance,
        "backend_field_policy": {
            "raw_definition": (
                "selected backend fields copied without project postprocessing; the "
                "backend itself may apply model-internal temporal processing"
            ),
            "thresholds": (
                "configuration values are backend processing thresholds, not project "
                "quality filters or validated accuracy cutoffs"
            ),
            "omitted_backend_outputs": [
                "world landmarks",
                "segmentation masks",
            ],
            "landmark_count": (
                "number of returned canonical rows only; not a frame or pose quality "
                "measure"
            ),
        },
        "schemas": {
            "raw_landmarks.csv": {
                "relationship": (
                    "zero or more landmark rows keyed to each pose_frames.csv "
                    "frame_index"
                ),
                "columns": list(LANDMARK_FIELDS),
            },
            "pose_frames.csv": {
                "relationship": (
                    "exactly one row per attempted nominal frame slot, including "
                    "no-pose and decode failures"
                ),
                "columns": list(FRAME_FIELDS),
            },
        },
        "outputs": {
            "raw_landmarks": "raw_landmarks.csv",
            "frame_manifest": "pose_frames.csv",
            "annotated_video": "annotated_pose.mp4",
            "pose_metadata": "pose_metadata.json",
        },
        "annotated_video": {
            "dimensions_pixels": [video.width_pixels, video.height_pixels],
            "nominal_fps": video.nominal_fps,
            "audio": "not preserved",
            "decode_failure_behavior": (
                "black placeholder frame preserves nominal ordering and count"
            ),
            "pose_drawing": (
                "draws every returned landmark row and available canonical connection "
                "without visibility, presence, confidence, or bounds filtering; "
                "out-of-range normalized coordinates are not clipped"
            ),
            "status_label": "one white label on an opaque dark filled background",
        },
        "limitations": [
            (
                "Coordinates are pose-model estimates, not measured anatomy or "
                "ground truth; landmarks are not anatomical joint centers, laboratory "
                "coordinates, or clinical measurements."
            ),
            (
                "Outputs have not been validated against manual labels or a reference "
                "measurement system."
            ),
            (
                "No gait events, center of mass, gait metrics, stability metrics, "
                "or clinical conclusions are produced."
            ),
            (
                "OpenCV container frame counts and nominal FPS may differ from "
                "actual timestamps or decodable content."
            ),
        ],
    }


def estimate_pose_video(
    video_path: str | Path,
    estimator: PoseEstimator,
    output_root: Path = Path("outputs"),
) -> PoseRunArtifacts:
    """Estimate raw pose for all nominal frames and publish Step 2 artifacts."""
    source = Path(video_path).expanduser().resolve()
    try:
        video = load_matching_video_metadata(source, output_root)
        if video is None:
            video = inspect_video(source, output_root)
        destination = video.artifact_directory
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.pose-staging-", dir=destination.parent
            )
        )
    except Exception:
        estimator.close()
        raise
    writer: Any | None = None
    frames: list[PoseFrameResult] = []
    processing_auto_orientation_status: str | None = None
    try:
        writer = cv2.VideoWriter(
            str(staging / "annotated_pose.mp4"),
            cv2.VideoWriter.fourcc(*"mp4v"),
            video.nominal_fps,
            (video.width_pixels, video.height_pixels),
        )
        if not writer.isOpened():
            raise AnnotatedVideoWriteError("Could not open annotated MP4 writer")
        for decoded in decode_video_frames(source, video):
            processing_auto_orientation_status = decoded.auto_orientation_status
            if decoded.image_bgr is None:
                result = PoseFrameResult(
                    decoded.frame_index,
                    decoded.nominal_timestamp_seconds,
                    PoseFrameStatus.DECODE_FAILURE,
                    (),
                    decoded.decode_detail,
                    None,
                )
                image = np.zeros(
                    (video.height_pixels, video.width_pixels, 3), dtype=np.uint8
                )
            else:
                _validate_decoded_raster(decoded.image_bgr, video, decoded.frame_index)
                image_rgb = cv2.cvtColor(decoded.image_bgr, cv2.COLOR_BGR2RGB)
                estimate = estimator.estimate_frame(
                    image_rgb,
                    decoded.frame_index,
                    decoded.nominal_timestamp_seconds,
                )
                if isinstance(estimate, PoseEstimate):
                    landmarks = estimate.landmarks
                    backend_timestamp_milliseconds = (
                        estimate.backend_timestamp_milliseconds
                    )
                else:
                    landmarks = estimate
                    backend_timestamp_milliseconds = None
                status = (
                    PoseFrameStatus.DECODED_POSE
                    if landmarks
                    else PoseFrameStatus.DECODED_NO_POSE
                )
                result = PoseFrameResult(
                    decoded.frame_index,
                    decoded.nominal_timestamp_seconds,
                    status,
                    landmarks,
                    None,
                    backend_timestamp_milliseconds,
                )
                image = decoded.image_bgr
            frames.append(result)
            rendered = _render_pose(
                image,
                result.landmarks,
                result.frame_index,
                result.nominal_timestamp_seconds,
                result.status,
            )
            try:
                writer.write(rendered)
            except cv2.error as exc:
                raise AnnotatedVideoWriteError(
                    f"Could not write annotated frame {result.frame_index}"
                ) from exc
        writer.release()
        writer = None
        if len(frames) != video.nominal_frame_count:
            raise PosePipelineError(
                f"Expected {video.nominal_frame_count} nominal frame slots, "
                f"processed {len(frames)}"
            )
        annotated = staging / "annotated_pose.mp4"
        if not annotated.is_file() or annotated.stat().st_size == 0:
            raise AnnotatedVideoWriteError(
                "Annotated writer produced no video artifact"
            )
        _write_csv_artifacts(staging, frames)
        (staging / "pose_metadata.json").write_text(
            json.dumps(
                _metadata_payload(
                    video, estimator, frames, processing_auto_orientation_status
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _publish_pose_artifacts(staging, destination)
        return PoseRunArtifacts(
            destination,
            destination / "raw_landmarks.csv",
            destination / "pose_frames.csv",
            destination / "pose_metadata.json",
            destination / "annotated_pose.mp4",
            tuple(frames),
        )
    finally:
        if writer is not None:
            writer.release()
        estimator.close()
        shutil.rmtree(staging, ignore_errors=True)
