"""Deterministic inspection and representative-frame extraction for videos."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import uuid
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

SUPPORTED_EXTENSIONS = frozenset({".mp4", ".mov"})
SAMPLE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
SAMPLING_METHOD = (
    "round(fraction * (frame_count - 1)) for fractions "
    "[0%, 25%, 50%, 75%, 100%], de-duplicated in order"
)


class VideoInspectionError(Exception):
    """Base class for expected video-inspection failures."""


class VideoNotFoundError(VideoInspectionError):
    """Raised when the source path does not exist."""


class VideoNotAFileError(VideoInspectionError):
    """Raised when the source path is not a regular file."""


class UnsupportedVideoFormatError(VideoInspectionError):
    """Raised when the filename extension is unsupported."""


class SourceInsideOutputDestinationError(VideoInspectionError):
    """Raised when publishing artifacts could replace or delete the source."""


class VideoOpenError(VideoInspectionError):
    """Raised when OpenCV cannot open the source video."""


class InvalidVideoResolutionError(VideoInspectionError):
    """Raised when width or height metadata is invalid."""


class InvalidVideoFPSError(VideoInspectionError):
    """Raised when nominal frame-rate metadata is invalid."""


class InvalidVideoFrameCountError(VideoInspectionError):
    """Raised when nominal frame-count metadata is invalid."""


class SelectedFrameDecodeError(VideoInspectionError):
    """Raised when a selected source frame cannot be reliably decoded."""


class SelectedFrameWriteError(VideoInspectionError):
    """Raised when a decoded sample image cannot be written."""


class ArtifactPublishError(VideoInspectionError):
    """Raised when completed staged artifacts cannot be safely published."""


@dataclass(frozen=True, slots=True)
class SampledFrameRecord:
    """Provenance for one representative frame."""

    frame_index: int
    nominal_timestamp_seconds: float
    relative_image_path: str
    decode_status: str


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Observed file and nominal container metadata from one inspection."""

    source_path: Path
    filename: str
    extension: str
    container_indicator: str
    file_size_bytes: int
    sha256: str
    inspected_at_utc: str
    width_pixels: int
    height_pixels: int
    nominal_fps: float
    nominal_frame_count: int
    nominal_duration_seconds: float
    opencv_version: str
    capture_backend: str
    orientation_degrees: int | None
    orientation_status: str
    auto_orientation_status: str
    readability_status: str
    decode_status: str
    seek_verification_method: str
    sampling_method: str
    sampled_frames: tuple[SampledFrameRecord, ...]
    artifact_directory: Path
    metadata_path: Path

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        result = asdict(self)
        result["source_path"] = str(self.source_path)
        result["artifact_directory"] = str(self.artifact_directory)
        result["metadata_path"] = str(self.metadata_path)
        return result


def _validated_positive_integer(value: float, field: str) -> int:
    if (
        not math.isfinite(value)
        or value <= 0
        or not math.isclose(value, round(value), abs_tol=0.01)
    ):
        error_type = (
            InvalidVideoFrameCountError
            if field == "frame count"
            else InvalidVideoResolutionError
        )
        raise error_type(f"Video reports invalid {field}: {value!r}")
    return int(round(value))


def _sample_indices(frame_count: int) -> tuple[int, ...]:
    indices: list[int] = []
    for fraction in SAMPLE_FRACTIONS:
        index = round(fraction * (frame_count - 1))
        if index not in indices:
            indices.append(index)
    return tuple(indices)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _capture_backend(capture: cv2.VideoCapture) -> str:
    try:
        return capture.getBackendName()
    except cv2.error:
        return "unavailable"


def _capture_orientation(capture: cv2.VideoCapture) -> tuple[int | None, str]:
    orientation_property = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    if orientation_property is None:
        return None, "unavailable: OpenCV orientation metadata property is absent"

    try:
        value = capture.get(orientation_property)
    except cv2.error:
        return None, "unavailable: capture backend could not read orientation metadata"
    if not math.isfinite(value):
        return None, f"invalid: non-finite orientation metadata {value!r}"

    normalized = value % 360
    rounded = round(normalized)
    if not math.isclose(normalized, rounded, abs_tol=0.01) or rounded % 90 != 0:
        return None, f"invalid: non-conventional orientation metadata {value:g} degrees"
    conventional = rounded % 360
    if conventional == 0:
        return 0, "zero: capture backend reported no rotation"
    return conventional, "reported: capture backend reported orientation metadata"


def _disable_auto_orientation(capture: cv2.VideoCapture) -> str:
    orientation_auto_property = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_auto_property is None:
        return "unsupported: OpenCV auto-orientation property is absent"
    try:
        disabled = capture.set(orientation_auto_property, 0.0)
    except cv2.error:
        return "error: capture backend raised while disabling auto-orientation"
    if not disabled:
        return "rejected: capture backend did not disable auto-orientation"
    return "disabled: capture backend accepted auto-orientation disable request"


def _read_metadata_property(
    capture: cv2.VideoCapture,
    property_id: int,
    field: str,
    error_type: type[VideoInspectionError],
) -> float:
    try:
        return capture.get(property_id)
    except cv2.error as exc:
        raise error_type(f"OpenCV could not read video {field} metadata") from exc


def _seek_and_decode(capture: cv2.VideoCapture, frame_index: int) -> Any:
    try:
        seek_succeeded = capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    except cv2.error as exc:
        raise SelectedFrameDecodeError(
            f"OpenCV raised while seeking to selected frame {frame_index}"
        ) from exc
    if not seek_succeeded:
        raise SelectedFrameDecodeError(
            f"OpenCV could not seek to selected frame {frame_index}"
        )

    try:
        reported_before = capture.get(cv2.CAP_PROP_POS_FRAMES)
    except cv2.error as exc:
        raise SelectedFrameDecodeError(
            f"OpenCV could not query position before decoding selected frame "
            f"{frame_index}"
        ) from exc
    if (
        math.isfinite(reported_before)
        and (reported_before > 0 or frame_index == 0)
        and not math.isclose(reported_before, frame_index, abs_tol=0.5)
    ):
        raise SelectedFrameDecodeError(
            f"Seek to frame {frame_index} reported position {reported_before:g}"
        )

    try:
        decoded, image = capture.read()
    except cv2.error as exc:
        raise SelectedFrameDecodeError(
            f"OpenCV raised while decoding selected frame {frame_index}"
        ) from exc
    if not decoded or image is None or image.size == 0:
        raise SelectedFrameDecodeError(
            f"OpenCV could not decode selected frame {frame_index}"
        )

    try:
        reported_after = capture.get(cv2.CAP_PROP_POS_FRAMES)
    except cv2.error as exc:
        raise SelectedFrameDecodeError(
            f"OpenCV could not query position after decoding selected frame "
            f"{frame_index}"
        ) from exc
    expected_after = frame_index + 1
    if (
        math.isfinite(reported_after)
        and reported_after > 0
        and not math.isclose(reported_after, expected_after, abs_tol=0.5)
    ):
        raise SelectedFrameDecodeError(
            f"Decoded frame {frame_index}, but OpenCV reported position "
            f"{reported_after:g} afterward"
        )
    return image


def _publish_staging(staging: Path, destination: Path) -> None:
    backup: Path | None = None
    if destination.exists():
        backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
        try:
            destination.replace(backup)
        except OSError as exc:
            raise ArtifactPublishError(
                f"Could not prepare existing output for replacement: {destination}"
            ) from exc

    try:
        staging.replace(destination)
    except OSError as exc:
        if backup is not None:
            try:
                backup.replace(destination)
            except OSError as rollback_exc:
                raise ArtifactPublishError(
                    f"Could not publish {destination} and rollback also failed; "
                    f"previous output remains at {backup}"
                ) from rollback_exc
        raise ArtifactPublishError(
            f"Could not publish artifacts to {destination}"
        ) from exc

    if backup is not None:
        try:
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
        except OSError as exc:
            warnings.warn(
                f"Artifacts were published to {destination}, but old-output cleanup "
                f"failed at {backup}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def inspect_video(
    video_path: str | Path, output_root: Path = Path("outputs")
) -> VideoMetadata:
    """Inspect a local MP4/MOV and stage all artifacts before replacing output.

    Timestamps and duration are nominal values derived from container metadata.
    Random access in compressed video is OpenCV-backend dependent; reported seek
    positions are checked when the backend provides meaningful values.
    """
    source = Path(video_path).expanduser()
    if not source.exists():
        raise VideoNotFoundError(f"Video does not exist: {source}")
    if not source.is_file():
        raise VideoNotAFileError(f"Video path is not a regular file: {source}")
    extension = source.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedVideoFormatError(
            f"Unsupported video extension {source.suffix!r}; expected {supported}"
        )

    source = source.resolve()
    destination = output_root.expanduser().resolve() / source.stem
    if source.is_relative_to(destination):
        raise SourceInsideOutputDestinationError(
            f"Output destination {destination} would contain and could replace or "
            f"delete source video {source}; select a different output root"
        )

    try:
        capture = cv2.VideoCapture(str(source))
    except cv2.error as exc:
        raise VideoOpenError(f"OpenCV could not open video: {source}") from exc
    if not capture.isOpened():
        capture.release()
        raise VideoOpenError(f"OpenCV could not open video: {source}")

    auto_orientation_status = _disable_auto_orientation(capture)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{source.stem}.staging-", dir=destination.parent)
    )

    try:
        width = _validated_positive_integer(
            _read_metadata_property(
                capture,
                cv2.CAP_PROP_FRAME_WIDTH,
                "width",
                InvalidVideoResolutionError,
            ),
            "width",
        )
        height = _validated_positive_integer(
            _read_metadata_property(
                capture,
                cv2.CAP_PROP_FRAME_HEIGHT,
                "height",
                InvalidVideoResolutionError,
            ),
            "height",
        )
        fps = _read_metadata_property(
            capture, cv2.CAP_PROP_FPS, "nominal FPS", InvalidVideoFPSError
        )
        if not math.isfinite(fps) or fps <= 0:
            raise InvalidVideoFPSError(f"Video reports invalid nominal FPS: {fps!r}")
        frame_count = _validated_positive_integer(
            _read_metadata_property(
                capture,
                cv2.CAP_PROP_FRAME_COUNT,
                "frame count",
                InvalidVideoFrameCountError,
            ),
            "frame count",
        )
        duration = frame_count / fps

        backend = _capture_backend(capture)
        orientation_degrees, orientation_status = _capture_orientation(capture)
        sample_directory = staging / "sample_frames"
        sample_directory.mkdir()
        padding = max(1, len(str(frame_count - 1)))
        records: list[SampledFrameRecord] = []
        for frame_index in _sample_indices(frame_count):
            image = _seek_and_decode(capture, frame_index)
            filename = f"frame_{frame_index:0{padding}d}.jpg"
            image_path = sample_directory / filename
            try:
                written = cv2.imwrite(str(image_path), image)
            except cv2.error as exc:
                raise SelectedFrameWriteError(
                    f"OpenCV could not write selected frame {frame_index} "
                    f"to {image_path}"
                ) from exc
            if not written:
                raise SelectedFrameWriteError(
                    f"OpenCV could not write selected frame {frame_index} "
                    f"to {image_path}"
                )
            records.append(
                SampledFrameRecord(
                    frame_index=frame_index,
                    nominal_timestamp_seconds=frame_index / fps,
                    relative_image_path=(Path("sample_frames") / filename).as_posix(),
                    decode_status="decoded",
                )
            )

        metadata_path = destination / "video_metadata.json"
        metadata = VideoMetadata(
            source_path=source,
            filename=source.name,
            extension=extension,
            container_indicator=extension.removeprefix("."),
            file_size_bytes=source.stat().st_size,
            sha256=_sha256(source),
            inspected_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            width_pixels=width,
            height_pixels=height,
            nominal_fps=fps,
            nominal_frame_count=frame_count,
            nominal_duration_seconds=duration,
            opencv_version=cv2.__version__,
            capture_backend=backend,
            orientation_degrees=orientation_degrees,
            orientation_status=orientation_status,
            auto_orientation_status=auto_orientation_status,
            readability_status="opened",
            decode_status="all selected frames decoded",
            seek_verification_method=(
                "OpenCV CAP_PROP_POS_FRAMES checked before and after decode when "
                "meaningful; exact compressed-frame identity remains capture-backend "
                "dependent"
            ),
            sampling_method=SAMPLING_METHOD,
            sampled_frames=tuple(records),
            artifact_directory=destination,
            metadata_path=metadata_path,
        )
        staged_metadata = staging / "video_metadata.json"
        staged_metadata.write_text(
            json.dumps(metadata.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        _publish_staging(staging, destination)
        return metadata
    finally:
        capture.release()
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
