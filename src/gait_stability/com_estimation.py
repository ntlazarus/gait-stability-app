"""Pure center-of-mass proxy calculations for video-derived gait analysis.

This module implements the de Leva 1996 14-segment anthropometric model
(adjustments to Zatsiorsky-Seluyanov, DOI 10.1016/0021-9290(95)00178-6)
using MediaPipe 2D landmark endpoints as segment proxies.

All calculations operate in normalized image coordinates (x right, y down).
No physical scale, camera calibration, or 3D reconstruction is performed.
Coordinate values are not constrained to [0, 1].

Unsupported segments: the head segment is always unavailable because
standard MediaPipe Pose lacks a defensible source-compatible vertex/neck
joint-centre line. Head coefficient and mass are retained in the model
for provenance but never participate in frame COM calculations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Anthropometric model constants (de Leva 1996, Table 4)
# ---------------------------------------------------------------------------

DE_LEVA_MALE: dict[str, dict[str, float]] = {
    # r values: proximal-to-distal COM ratio (0 = proximal endpoint, 1 = distal)
    # Source: de Leva 1996 Table 4 adjusted-parameter r values, proximal reference.
    # Visual3D adjusted-parameter documentation transcribed independently.
    "head": {"mass": 0.0694, "r": 0.5002},
    "trunk": {"mass": 0.4346, "r": 0.5138},
    "upper_arm": {"mass": 0.0271, "r": 0.5772},
    "forearm": {"mass": 0.0162, "r": 0.4574},
    "hand": {"mass": 0.0061, "r": 0.7900},
    "thigh": {"mass": 0.1416, "r": 0.4095},
    "shank": {"mass": 0.0433, "r": 0.4395},
    "foot": {"mass": 0.0137, "r": 0.4415},
}

DE_LEVA_FEMALE: dict[str, dict[str, float]] = {
    # r values: proximal-to-distal COM ratio (0 = proximal endpoint, 1 = distal)
    # Source: de Leva 1996 Table 4 adjusted-parameter r values, proximal reference.
    # Visual3D adjusted-parameter documentation transcribed independently.
    "head": {"mass": 0.0668, "r": 0.4841},
    "trunk": {"mass": 0.4257, "r": 0.4964},
    "upper_arm": {"mass": 0.0255, "r": 0.5754},
    "forearm": {"mass": 0.0138, "r": 0.4559},
    "hand": {"mass": 0.0056, "r": 0.7474},
    "thigh": {"mass": 0.1478, "r": 0.3612},
    "shank": {"mass": 0.0481, "r": 0.4352},
    "foot": {"mass": 0.0129, "r": 0.4014},
}

MODEL_MASS_TOTAL_MALE: float = 1.0000
MODEL_MASS_TOTAL_FEMALE: float = 0.9999

# Maximum represented body-mass fraction when all supported segments are usable.
# Head is unsupported (no defensible MediaPipe source-compatible vertex/neck
# joint-centre line), so full model coverage cannot occur.
# male: 1.0000 - 0.0694 = 0.9306
# female: 0.9999 - 0.0668 = 0.9331
REPRESENTED_MASS_MAX_MALE: float = 0.9306
REPRESENTED_MASS_MAX_FEMALE: float = 0.9331

# Segments that are never available for COM calculation regardless of
# endpoint availability. Head is unsupported because standard MediaPipe
# Pose lacks a defensible source-compatible vertex/neck joint-centre line.
# Nose is NOT the vertex; the published source figure/reference must be
# consulted and the implementation omits head entirely.
UNSUPPORTED_SEGMENTS: tuple[str, ...] = ("head",)

SEGMENT_NAMES: tuple[str, ...] = (
    "head",
    "trunk",
    "left_upper_arm",
    "right_upper_arm",
    "left_forearm",
    "right_forearm",
    "left_hand",
    "right_hand",
    "left_thigh",
    "right_thigh",
    "left_shank",
    "right_shank",
    "left_foot",
    "right_foot",
)

BILATERAL_SEGMENTS: tuple[str, ...] = (
    "upper_arm",
    "forearm",
    "hand",
    "thigh",
    "shank",
    "foot",
)

SEGMENT_ENDPOINTS: dict[str, tuple[str, str]] = {
    "head": ("_unsupported_head_proximal", "_unsupported_head_distal"),
    "trunk": ("shoulder_midpoint", "hip_midpoint"),
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_forearm": ("right_elbow", "right_wrist"),
    "left_hand": ("left_wrist", "left_index_pinky_midpoint_proxy"),
    "right_hand": ("right_wrist", "right_index_pinky_midpoint_proxy"),
    "left_thigh": ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_shank": ("left_knee", "left_ankle"),
    "right_shank": ("right_knee", "right_ankle"),
    "left_foot": ("left_ankle", "left_foot_index"),
    "right_foot": ("right_ankle", "right_foot_index"),
}

DERIVED_LANDMARKS: dict[str, tuple[str, str]] = {
    "shoulder_midpoint": ("left_shoulder", "right_shoulder"),
    "hip_midpoint": ("left_hip", "right_hip"),
    "left_index_pinky_midpoint_proxy": ("left_index", "left_pinky"),
    "right_index_pinky_midpoint_proxy": ("right_index", "right_pinky"),
}

# ---------------------------------------------------------------------------
# CSV field definitions (exported)
# ---------------------------------------------------------------------------

_BASE_PROXY_FIELDS: tuple[str, ...] = (
    "frame_index",
    "timestamp_seconds",
    "frame_status",
    "com_x",
    "com_y",
    "mass_coverage",
    "model_total_mass",
    "usable",
    "usable_segments",
    "missing_segments",
    "contributors_raw_observed",
    "contributors_x_interpolated",
    "contributors_y_interpolated",
    "contributors_x_smoothing_changed",
    "contributors_y_smoothing_changed",
    "contributors_x_smoothing_support_interpolation",
    "contributors_y_smoothing_support_interpolation",
    "contributors_other_qc_limited",
    "mass_x_interpolated",
    "mass_y_interpolated",
    "mass_x_smoothing_changed",
    "mass_y_smoothing_changed",
    "mass_x_smoothing_support_interpolation",
    "mass_y_smoothing_support_interpolation",
    "mass_other_qc_limited",
    "mass_missing",
)

_SEG_PER_FIELDS: tuple[str, ...] = ()
for _s in SEGMENT_NAMES:
    _SEG_PER_FIELDS = _SEG_PER_FIELDS + (
        f"seg_{_s}_com_x",
        f"seg_{_s}_com_y",
        f"seg_{_s}_usable",
        f"seg_{_s}_mass_fraction",
        f"seg_{_s}_contributors",
        f"seg_{_s}_qc_flags",
    )

COM_PROXY_FIELDS: tuple[str, ...] = _BASE_PROXY_FIELDS + _SEG_PER_FIELDS

STRIDE_COM_FIELDS: tuple[str, ...] = (
    "stride_id",
    "side",
    "start_event_id",
    "end_event_id",
    "start_frame",
    "end_frame",
    "start_timestamp_seconds",
    "end_timestamp_seconds",
    "duration_seconds",
    "quality",
    "contralateral_event_id",
    "contralateral_event_count",
    "source",
    "review_status",
    "automatic_stride_id",
    "review_intent",
    "review_changes",
    "provenance_notes",
    "sample_kind",
    "normalized_index",
    "progression",
    "method",
    "source_frame_index",
    "source_timestamp_seconds",
    "target_timestamp_seconds",
    "left_source_frame_index",
    "left_source_timestamp_seconds",
    "right_source_frame_index",
    "right_source_timestamp_seconds",
    "com_x",
    "com_y",
    "usable",
    "mass_coverage",
    "min_endpoint_coverage",
    "contributors",
    "qc_flags",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComEstimationConfig:
    """Configuration for COM proxy estimation.

    Attributes:
        anthropometry_sex: Required sex for anthropometric coefficients.
        minimum_mass_coverage: Minimum fraction of model mass that must be
            usable to mark frame as usable. Default 0.90. Equality passes.
            Frames below threshold retain finite COM but usable=False.
            A zero-coverage frame is never usable even if threshold=0.
            Female model total 0.9999 means threshold 1.0 can never pass.
        normalized_stride_samples: Number of samples in normalized stride grid.
            Default 101 (0..100 inclusive). Must be integer >= 2.
    """

    anthropometry_sex: Literal["male", "female"]
    minimum_mass_coverage: float = 0.90
    normalized_stride_samples: int = 101

    def __post_init__(self) -> None:
        if self.anthropometry_sex not in ("male", "female"):
            raise ValueError("anthropometry_sex must be 'male' or 'female'")
        if isinstance(self.minimum_mass_coverage, bool) or not isinstance(
            self.minimum_mass_coverage, (int, float)
        ):
            raise TypeError("minimum_mass_coverage must be a number")
        if not math.isfinite(self.minimum_mass_coverage):
            raise ValueError("minimum_mass_coverage must be finite")
        if not 0.0 <= self.minimum_mass_coverage <= 1.0:
            raise ValueError("minimum_mass_coverage must be between 0 and 1")
        if isinstance(self.normalized_stride_samples, bool) or not isinstance(
            self.normalized_stride_samples, int
        ):
            raise TypeError("normalized_stride_samples must be an integer")
        if self.normalized_stride_samples < 2:
            raise ValueError("normalized_stride_samples must be >= 2")


def _get_model(sex: Literal["male", "female"]) -> dict[str, dict[str, float]]:
    if sex == "male":
        return DE_LEVA_MALE
    return DE_LEVA_FEMALE


def _get_model_total(sex: Literal["male", "female"]) -> float:
    if sex == "male":
        return MODEL_MASS_TOTAL_MALE
    return MODEL_MASS_TOTAL_FEMALE


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LandmarkProvenance:
    """Provenance for a single landmark coordinate from Step 3."""

    landmark_name: str
    raw_observed_usable: bool
    x_interpolated: bool
    y_interpolated: bool
    x_smoothing_changed: bool
    y_smoothing_changed: bool
    x_smoothing_support_contains_interpolation: bool
    y_smoothing_support_contains_interpolation: bool

    def __post_init__(self) -> None:
        if not isinstance(self.landmark_name, str) or not self.landmark_name:
            raise ValueError(
                "LandmarkProvenance.landmark_name must be a nonempty string"
            )
        for _fld in (
            "raw_observed_usable",
            "x_interpolated",
            "y_interpolated",
            "x_smoothing_changed",
            "y_smoothing_changed",
            "x_smoothing_support_contains_interpolation",
            "y_smoothing_support_contains_interpolation",
        ):
            if not isinstance(getattr(self, _fld), bool):
                raise TypeError(
                    f"LandmarkProvenance.{_fld} must be bool, "
                    f"got {type(getattr(self, _fld)).__name__}"
                )


def _missing_provenance(name: str) -> LandmarkProvenance:
    return LandmarkProvenance(
        landmark_name=name,
        raw_observed_usable=False,
        x_interpolated=False,
        y_interpolated=False,
        x_smoothing_changed=False,
        y_smoothing_changed=False,
        x_smoothing_support_contains_interpolation=False,
        y_smoothing_support_contains_interpolation=False,
    )


def _aggregate_provenance(
    provs: tuple[LandmarkProvenance, ...],
) -> tuple[str, ...]:
    """Return union contributor names from a set of provenance records."""
    return tuple(sorted({p.landmark_name for p in provs}))


def _any_x_interpolated(provs: tuple[LandmarkProvenance, ...]) -> bool:
    return any(p.x_interpolated for p in provs)


def _any_y_interpolated(provs: tuple[LandmarkProvenance, ...]) -> bool:
    return any(p.y_interpolated for p in provs)


def _any_x_smoothing_changed(provs: tuple[LandmarkProvenance, ...]) -> bool:
    return any(p.x_smoothing_changed for p in provs)


def _any_y_smoothing_changed(provs: tuple[LandmarkProvenance, ...]) -> bool:
    return any(p.y_smoothing_changed for p in provs)


def _any_x_smoothing_interp(provs: tuple[LandmarkProvenance, ...]) -> bool:
    return any(p.x_smoothing_support_contains_interpolation for p in provs)


def _any_y_smoothing_interp(provs: tuple[LandmarkProvenance, ...]) -> bool:
    return any(p.y_smoothing_support_contains_interpolation for p in provs)


def _all_raw_observed(provs: tuple[LandmarkProvenance, ...]) -> bool:
    return all(p.raw_observed_usable for p in provs)


@dataclass(frozen=True, slots=True)
class SegmentProvenance:
    """Aggregated provenance for one segment's COM calculation.

    Contributors are the actual underlying MediaPipe landmark names (or
    derived-name components) whose Step 3 QC flags propagate.
    """

    segment_name: str
    proximal_landmark: str
    distal_landmark: str
    contributors: tuple[str, ...]
    usable: bool
    mass_fraction: float
    r: float
    all_raw_observed: bool
    any_x_interpolated: bool
    any_y_interpolated: bool
    any_x_smoothing_changed: bool
    any_y_smoothing_changed: bool
    any_x_smoothing_support_interpolation: bool
    any_y_smoothing_support_interpolation: bool
    other_qc_limited: bool

    def provenance_category(self) -> str:
        if not self.usable:
            return "missing"
        if self.any_x_interpolated or self.any_y_interpolated:
            return "interpolated"
        if (
            self.any_x_smoothing_support_interpolation
            or self.any_y_smoothing_support_interpolation
        ):
            return "smoothing_interpolation"
        if self.all_raw_observed:
            return "raw_observed"
        if self.any_x_smoothing_changed or self.any_y_smoothing_changed:
            return "smoothed_only"
        return "other_qc_limited"

    def __post_init__(self) -> None:
        if not isinstance(self.segment_name, str) or not self.segment_name:
            raise ValueError("SegmentProvenance.segment_name must be a nonempty string")
        if not isinstance(self.proximal_landmark, str) or not self.proximal_landmark:
            raise ValueError(
                "SegmentProvenance.proximal_landmark must be a nonempty string"
            )
        if not isinstance(self.distal_landmark, str) or not self.distal_landmark:
            raise ValueError(
                "SegmentProvenance.distal_landmark must be a nonempty string"
            )
        if not isinstance(self.contributors, tuple):
            raise TypeError("SegmentProvenance.contributors must be a tuple")
        for _c in self.contributors:
            if not isinstance(_c, str) or not _c:
                raise ValueError(
                    "SegmentProvenance.contributors entries must be nonempty strings"
                )
        if not isinstance(self.usable, bool):
            raise TypeError(
                f"SegmentProvenance.usable must be bool, "
                f"got {type(self.usable).__name__}"
            )
        for _num in ("mass_fraction", "r"):
            _val = getattr(self, _num)
            if isinstance(_val, bool):
                raise TypeError(f"SegmentProvenance.{_num} must be a number, got bool")
            if not isinstance(_val, (int, float)):
                raise TypeError(
                    f"SegmentProvenance.{_num} must be a number, "
                    f"got {type(_val).__name__}"
                )
            if not math.isfinite(_val):
                raise ValueError(f"SegmentProvenance.{_num} must be finite")
        if self.mass_fraction < 0.0:
            raise ValueError("SegmentProvenance.mass_fraction must be >= 0")
        if not (0.0 <= self.r <= 1.0):
            raise ValueError("SegmentProvenance.r must be between 0 and 1")
        for _fld in (
            "all_raw_observed",
            "any_x_interpolated",
            "any_y_interpolated",
            "any_x_smoothing_changed",
            "any_y_smoothing_changed",
            "any_x_smoothing_support_interpolation",
            "any_y_smoothing_support_interpolation",
            "other_qc_limited",
        ):
            if not isinstance(getattr(self, _fld), bool):
                raise TypeError(
                    f"SegmentProvenance.{_fld} must be bool, "
                    f"got {type(getattr(self, _fld)).__name__}"
                )


@dataclass(frozen=True, slots=True)
class Point2D:
    """2D point in normalized image coordinates (x right, y down)."""

    x: float
    y: float

    def __post_init__(self) -> None:
        for _attr in ("x", "y"):
            _val = getattr(self, _attr)
            if isinstance(_val, bool):
                raise TypeError(f"Point2D {_attr} must be a finite number, got bool")
            if not isinstance(_val, (int, float)):
                raise TypeError(
                    f"Point2D {_attr} must be a finite number, "
                    f"got {type(_val).__name__}"
                )
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("Point2D coordinates must be finite")


@dataclass(frozen=True, slots=True)
class SegmentComResult:
    """Result of COM calculation for one segment."""

    segment_name: str
    com: Point2D | None
    mass_fraction: float
    usable: bool
    provenance: SegmentProvenance


@dataclass(frozen=True, slots=True)
class FrameComResult:
    """Center-of-mass proxy result for one frame.

    The COM is the mass-weighted centroid of usable segments, normalized by
    represented usable mass. This is a represented-segment centroid: it
    divides by the sum of usable mass fractions (the represented body mass),
    never by the full model total. mass_coverage is the raw sum of usable
    mass fractions (unrenormalized). A zero-coverage frame has com=None
    and usable=False regardless of the threshold setting.

    Unsupported segments (e.g. head) never participate in frame COM
    calculations regardless of endpoint availability.
    """

    frame_index: int
    timestamp_seconds: float
    frame_status: str
    com: Point2D | None
    mass_coverage: float
    usable: bool
    segment_results: tuple[SegmentComResult, ...]
    model_total_mass: float

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError(
                f"frame_index must be a nonnegative integer, "
                f"got {type(self.frame_index).__name__}"
            )
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if isinstance(self.timestamp_seconds, bool) or not isinstance(
            self.timestamp_seconds, (int, float)
        ):
            raise TypeError(
                f"timestamp_seconds must be a finite number, "
                f"got {type(self.timestamp_seconds).__name__}"
            )
        if not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0.0:
            raise ValueError("timestamp_seconds must be a nonnegative finite number")
        if not isinstance(self.frame_status, str) or not self.frame_status:
            raise ValueError("frame_status must be a nonempty string")
        if self.com is not None and not isinstance(self.com, Point2D):
            raise TypeError(
                f"com must be Point2D or None, got {type(self.com).__name__}"
            )
        if isinstance(self.mass_coverage, bool) or not isinstance(
            self.mass_coverage, (int, float)
        ):
            raise TypeError(
                f"mass_coverage must be a finite number, "
                f"got {type(self.mass_coverage).__name__}"
            )
        if not math.isfinite(self.mass_coverage):
            raise ValueError("mass_coverage must be finite")
        if self.mass_coverage < 0.0:
            raise ValueError("mass_coverage must be >= 0")
        if not isinstance(self.usable, bool):
            raise TypeError(f"usable must be bool, got {type(self.usable).__name__}")
        if not isinstance(self.segment_results, tuple):
            raise TypeError("segment_results must be a tuple")
        for _sr in self.segment_results:
            if not isinstance(_sr, SegmentComResult):
                raise TypeError(
                    f"segment_results entries must be SegmentComResult, "
                    f"got {type(_sr).__name__}"
                )
        if isinstance(self.model_total_mass, bool) or not isinstance(
            self.model_total_mass, (int, float)
        ):
            raise TypeError(
                f"model_total_mass must be a positive finite number, "
                f"got {type(self.model_total_mass).__name__}"
            )
        if not math.isfinite(self.model_total_mass) or self.model_total_mass <= 0.0:
            raise ValueError("model_total_mass must be a positive finite number")
        # -- consistency --
        if self.usable and self.com is None:
            raise ValueError("usable is True but com is None")
        if self.mass_coverage == 0.0 and self.usable:
            raise ValueError("mass_coverage is 0.0 but usable is True")
        if self.mass_coverage == 0.0 and self.com is not None:
            raise ValueError("mass_coverage is 0.0 but com is not None")

    def usable_segments(self) -> tuple[str, ...]:
        return tuple(sr.segment_name for sr in self.segment_results if sr.usable)

    def missing_segments(self) -> tuple[str, ...]:
        return tuple(sr.segment_name for sr in self.segment_results if not sr.usable)

    def provenance_summary(self) -> dict[str, int]:
        """Count of segments per provenance category."""
        counts: dict[str, int] = {
            "raw_observed": 0,
            "smoothed_only": 0,
            "interpolated": 0,
            "smoothing_interpolation": 0,
            "missing": 0,
            "other_qc_limited": 0,
        }
        for sr in self.segment_results:
            cat = sr.provenance.provenance_category()
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    # -- nonexclusive per-segment name queries (SEGMENT_NAMES order,
    #    usable segments only) --

    def _usable_provs(
        self,
    ) -> tuple[SegmentProvenance, ...]:
        """Segment provenance for usable segments, stable SEGMENT_NAMES order."""
        return tuple(sr.provenance for sr in self.segment_results if sr.usable)

    def segments_all_raw_observed(self) -> tuple[str, ...]:
        """Segment names where usable and all_raw_observed is True."""
        return tuple(p.segment_name for p in self._usable_provs() if p.all_raw_observed)

    def segments_x_interpolated(self) -> tuple[str, ...]:
        """Segment names where usable and any_x_interpolated is True."""
        return tuple(
            p.segment_name for p in self._usable_provs() if p.any_x_interpolated
        )

    def segments_y_interpolated(self) -> tuple[str, ...]:
        """Segment names where usable and any_y_interpolated is True."""
        return tuple(
            p.segment_name for p in self._usable_provs() if p.any_y_interpolated
        )

    def segments_x_smoothing_changed(self) -> tuple[str, ...]:
        """Segment names where usable and any_x_smoothing_changed is True."""
        return tuple(
            p.segment_name for p in self._usable_provs() if p.any_x_smoothing_changed
        )

    def segments_y_smoothing_changed(self) -> tuple[str, ...]:
        """Segment names where usable and any_y_smoothing_changed is True."""
        return tuple(
            p.segment_name for p in self._usable_provs() if p.any_y_smoothing_changed
        )

    def segments_x_smoothing_support_interpolation(self) -> tuple[str, ...]:
        """Segment names where usable and x smoothing support interpolation."""
        return tuple(
            p.segment_name
            for p in self._usable_provs()
            if p.any_x_smoothing_support_interpolation
        )

    def segments_y_smoothing_support_interpolation(self) -> tuple[str, ...]:
        """Segment names where usable and y smoothing support interpolation."""
        return tuple(
            p.segment_name
            for p in self._usable_provs()
            if p.any_y_smoothing_support_interpolation
        )

    def segments_other_qc_limited(self) -> tuple[str, ...]:
        """Segment names where usable and other_qc_limited is True."""
        return tuple(p.segment_name for p in self._usable_provs() if p.other_qc_limited)

    def provenance_mass_totals(self) -> dict[str, float]:
        """Nonexclusive mass sums for each independent QC flag.

        Returns one total per independent flag among usable segments,
        plus ``"missing"`` for non-usable segments.  A single usable
        segment may contribute to more than one flag total (nonexclusive).
        """
        totals: dict[str, float] = {
            "raw_observed": 0.0,
            "x_interpolated": 0.0,
            "y_interpolated": 0.0,
            "x_smoothing_changed": 0.0,
            "y_smoothing_changed": 0.0,
            "x_smoothing_support_interpolation": 0.0,
            "y_smoothing_support_interpolation": 0.0,
            "other_qc_limited": 0.0,
            "missing": 0.0,
        }
        for sr in self.segment_results:
            p = sr.provenance
            if not p.usable:
                totals["missing"] += sr.mass_fraction
                continue
            if p.all_raw_observed:
                totals["raw_observed"] += sr.mass_fraction
            if p.any_x_interpolated:
                totals["x_interpolated"] += sr.mass_fraction
            if p.any_y_interpolated:
                totals["y_interpolated"] += sr.mass_fraction
            if p.any_x_smoothing_changed:
                totals["x_smoothing_changed"] += sr.mass_fraction
            if p.any_y_smoothing_changed:
                totals["y_smoothing_changed"] += sr.mass_fraction
            if p.any_x_smoothing_support_interpolation:
                totals["x_smoothing_support_interpolation"] += sr.mass_fraction
            if p.any_y_smoothing_support_interpolation:
                totals["y_smoothing_support_interpolation"] += sr.mass_fraction
            if p.other_qc_limited:
                totals["other_qc_limited"] += sr.mass_fraction
        return totals


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------


def _segment_com(proximal: Point2D, distal: Point2D, r: float) -> Point2D:
    x = proximal.x + r * (distal.x - proximal.x)
    y = proximal.y + r * (distal.y - proximal.y)
    return Point2D(x, y)


def _compute_derived_landmarks(
    landmarks: dict[str, Point2D],
) -> dict[str, Point2D]:
    result = dict(landmarks)
    for derived_name, (left_name, right_name) in DERIVED_LANDMARKS.items():
        left = landmarks.get(left_name)
        right = landmarks.get(right_name)
        if left is not None and right is not None:
            result[derived_name] = Point2D(
                (left.x + right.x) / 2.0, (left.y + right.y) / 2.0
            )
    return result


def _collect_endpoint_contributors(
    endpoint_name: str,
    landmark_provenance: dict[str, LandmarkProvenance],
) -> tuple[LandmarkProvenance, ...]:
    """Collect all underlying landmark provenance for an endpoint.

    For derived landmarks, returns the component MediaPipe landmarks.
    For direct landmarks, returns just that landmark's provenance.
    """
    if endpoint_name in DERIVED_LANDMARKS:
        left_name, right_name = DERIVED_LANDMARKS[endpoint_name]
        left_prov = landmark_provenance.get(left_name)
        right_prov = landmark_provenance.get(right_name)
        provs: list[LandmarkProvenance] = []
        if left_prov is not None:
            provs.append(left_prov)
        else:
            provs.append(_missing_provenance(left_name))
        if right_prov is not None:
            provs.append(right_prov)
        else:
            provs.append(_missing_provenance(right_name))
        return tuple(provs)
    prov = landmark_provenance.get(endpoint_name)
    if prov is not None:
        return (prov,)
    return (_missing_provenance(endpoint_name),)


def _build_segment_provenance(
    segment_name: str,
    proximal_name: str,
    distal_name: str,
    landmark_provenance: dict[str, LandmarkProvenance],
    mass_fraction: float,
    r: float,
    proximal_available: bool,
    distal_available: bool,
) -> SegmentProvenance:
    """Build aggregated provenance for one segment's COM calculation.

    ``proximal_available`` and ``distal_available`` are determined by the
    caller from ``all_landmarks.get(...) is not None``; a derived endpoint
    requires *both* component landmarks to be present (consistent with
    ``_compute_derived_landmarks``).  ``usable`` is True only when both
    endpoints are available.
    """
    prox_provs = _collect_endpoint_contributors(proximal_name, landmark_provenance)
    dist_provs = _collect_endpoint_contributors(distal_name, landmark_provenance)
    all_provs = prox_provs + dist_provs

    contributors = _aggregate_provenance(all_provs)
    usable = proximal_available and distal_available

    all_raw = _all_raw_observed(all_provs)
    any_xi = _any_x_interpolated(all_provs)
    any_yi = _any_y_interpolated(all_provs)
    any_xsc = _any_x_smoothing_changed(all_provs)
    any_ysc = _any_y_smoothing_changed(all_provs)
    any_xssi = _any_x_smoothing_interp(all_provs)
    any_yssi = _any_y_smoothing_interp(all_provs)
    other_qc = (
        usable
        and not all_raw
        and not any_xi
        and not any_yi
        and not any_xssi
        and not any_yssi
        and not any_xsc
        and not any_ysc
    )

    return SegmentProvenance(
        segment_name=segment_name,
        proximal_landmark=proximal_name,
        distal_landmark=distal_name,
        contributors=contributors,
        usable=usable,
        mass_fraction=mass_fraction,
        r=r,
        all_raw_observed=all_raw,
        any_x_interpolated=any_xi,
        any_y_interpolated=any_yi,
        any_x_smoothing_changed=any_xsc,
        any_y_smoothing_changed=any_ysc,
        any_x_smoothing_support_interpolation=any_xssi,
        any_y_smoothing_support_interpolation=any_yssi,
        other_qc_limited=other_qc,
    )


def estimate_frame_com(
    processed_landmarks: dict[str, Point2D],
    landmark_provenance: dict[str, LandmarkProvenance],
    config: ComEstimationConfig,
    frame_index: int,
    timestamp_seconds: float,
    frame_status: str,
) -> FrameComResult:
    """Estimate proxy COM for a single frame.

    COM is the mass-weighted centroid normalized by represented usable mass.
    mass_coverage is unrenormalized: sum of usable mass fractions without
    rescaling. A zero-coverage frame produces com=None and usable=False.

    ``SegmentComResult.usable`` exactly equals ``SegmentProvenance.usable``;
    endpoint availability is determined by ``all_landmarks.get(...) is not
    None`` (derived endpoints require both component landmarks).
    """
    # -- strict input validation --
    if not isinstance(config, ComEstimationConfig):
        raise TypeError(
            f"config must be ComEstimationConfig, got {type(config).__name__}"
        )
    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        raise TypeError(
            f"frame_index must be a nonnegative integer, "
            f"got {type(frame_index).__name__}"
        )
    if frame_index < 0:
        raise ValueError("frame_index must be >= 0")
    if isinstance(timestamp_seconds, bool) or not isinstance(
        timestamp_seconds, (int, float)
    ):
        raise TypeError(
            f"timestamp_seconds must be a finite number, "
            f"got {type(timestamp_seconds).__name__}"
        )
    if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0.0:
        raise ValueError("timestamp_seconds must be a nonnegative finite number")
    if not isinstance(frame_status, str) or not frame_status:
        raise ValueError("frame_status must be a nonempty string")
    for _k, _v in processed_landmarks.items():
        if not isinstance(_k, str):
            raise TypeError(
                f"processed_landmarks keys must be str, got {type(_k).__name__}"
            )
        if not isinstance(_v, Point2D):
            raise TypeError(
                f"processed_landmarks values must be Point2D, got {type(_v).__name__}"
            )
    for _pk, _pv in landmark_provenance.items():
        if not isinstance(_pk, str):
            raise TypeError(
                f"landmark_provenance keys must be str, got {type(_pk).__name__}"
            )
        if not isinstance(_pv, LandmarkProvenance):
            raise TypeError(
                f"landmark_provenance values must be LandmarkProvenance, "
                f"got {type(_pv).__name__}"
            )

    model = _get_model(config.anthropometry_sex)
    model_total = _get_model_total(config.anthropometry_sex)
    all_landmarks = _compute_derived_landmarks(processed_landmarks)

    segment_results: list[SegmentComResult] = []
    total_mass_covered = 0.0
    com_numerator_x = 0.0
    com_numerator_y = 0.0

    for segment_name in SEGMENT_NAMES:
        # Compute base_name for model lookup
        if segment_name in ("head", "trunk"):
            base_name = segment_name
        else:
            base_name = segment_name.split("_", 1)[1]

        # Unsupported segments are always unavailable regardless of endpoints.
        if segment_name in UNSUPPORTED_SEGMENTS:
            provenance = _build_segment_provenance(
                segment_name,
                SEGMENT_ENDPOINTS[segment_name][0],
                SEGMENT_ENDPOINTS[segment_name][1],
                landmark_provenance,
                model[base_name]["mass"],
                model[base_name]["r"],
                False,
                False,
            )
            seg_result = SegmentComResult(
                segment_name=segment_name,
                com=None,
                mass_fraction=model[base_name]["mass"],
                usable=False,
                provenance=provenance,
            )
            segment_results.append(seg_result)
            continue

        mass_fraction = model[base_name]["mass"]
        r = model[base_name]["r"]
        prox_name, dist_name = SEGMENT_ENDPOINTS[segment_name]

        # Endpoint availability from the computed landmark set.
        # Derived endpoints (e.g. shoulder_midpoint) are present in
        # all_landmarks only when *both* component landmarks were given.
        proximal_available = all_landmarks.get(prox_name) is not None
        distal_available = all_landmarks.get(dist_name) is not None

        provenance = _build_segment_provenance(
            segment_name,
            prox_name,
            dist_name,
            landmark_provenance,
            mass_fraction,
            r,
            proximal_available,
            distal_available,
        )

        if proximal_available and distal_available:
            com = _segment_com(all_landmarks[prox_name], all_landmarks[dist_name], r)
            seg_result = SegmentComResult(
                segment_name=segment_name,
                com=com,
                mass_fraction=mass_fraction,
                usable=provenance.usable,
                provenance=provenance,
            )
            segment_results.append(seg_result)
            total_mass_covered += mass_fraction
            com_numerator_x += mass_fraction * com.x
            com_numerator_y += mass_fraction * com.y
        else:
            seg_result = SegmentComResult(
                segment_name=segment_name,
                com=None,
                mass_fraction=mass_fraction,
                usable=provenance.usable,
                provenance=provenance,
            )
            segment_results.append(seg_result)

    com_val: Point2D | None
    if total_mass_covered > 0.0:
        com_val = Point2D(
            com_numerator_x / total_mass_covered,
            com_numerator_y / total_mass_covered,
        )
    else:
        com_val = None

    # Zero coverage => never usable, regardless of threshold
    if total_mass_covered <= 0.0:
        usable = False
    else:
        usable = total_mass_covered >= config.minimum_mass_coverage

    return FrameComResult(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        frame_status=frame_status,
        com=com_val,
        mass_coverage=total_mass_covered,
        usable=usable,
        segment_results=tuple(segment_results),
        model_total_mass=model_total,
    )


# ---------------------------------------------------------------------------
# Stride normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrideComSample:
    """One COM sample in a normalized stride.

    For original samples, source_frame_index/source_timestamp_seconds are exact.
    For normalized samples with method='exact', source_frame_index/timestamp
    are the matching frame. For method='linear', left/right source frames
    and timestamps bracket the interpolation. For method='none', no valid
    bracket exists.
    """

    progression: float
    com: Point2D | None
    mass_coverage: float
    usable: bool
    sample_kind: Literal["original", "normalized"]
    normalized_index: int | None
    method: Literal["exact", "linear", "none"]
    source_frame_index: int | None
    source_timestamp_seconds: float | None
    target_timestamp_seconds: float | None
    left_source_frame_index: int | None
    left_source_timestamp_seconds: float | None
    right_source_frame_index: int | None
    right_source_timestamp_seconds: float | None
    min_endpoint_coverage: float
    contributors: tuple[str, ...]
    qc_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.progression, bool) or not isinstance(
            self.progression, (int, float)
        ):
            raise TypeError(
                f"progression must be a finite number, "
                f"got {type(self.progression).__name__}"
            )
        if not math.isfinite(self.progression):
            raise ValueError("progression must be finite")
        if not (0.0 <= self.progression <= 100.0):
            raise ValueError("progression must be between 0 and 100")
        if self.com is not None and not isinstance(self.com, Point2D):
            raise TypeError(
                f"com must be Point2D or None, got {type(self.com).__name__}"
            )
        if isinstance(self.mass_coverage, bool) or not isinstance(
            self.mass_coverage, (int, float)
        ):
            raise TypeError(
                f"mass_coverage must be a finite number, "
                f"got {type(self.mass_coverage).__name__}"
            )
        if not math.isfinite(self.mass_coverage):
            raise ValueError("mass_coverage must be finite")
        if self.mass_coverage < 0.0:
            raise ValueError("mass_coverage must be >= 0")
        if not isinstance(self.usable, bool):
            raise TypeError(f"usable must be bool, got {type(self.usable).__name__}")
        if self.sample_kind not in ("original", "normalized"):
            raise ValueError(
                f"sample_kind must be 'original' or 'normalized', "
                f"got {self.sample_kind!r}"
            )
        if self.normalized_index is not None:
            if isinstance(self.normalized_index, bool) or not isinstance(
                self.normalized_index, int
            ):
                raise TypeError(
                    f"normalized_index must be an integer or None, "
                    f"got {type(self.normalized_index).__name__}"
                )
            if self.normalized_index < 0:
                raise ValueError("normalized_index must be >= 0")
        if self.method not in ("exact", "linear", "none"):
            raise ValueError(
                f"method must be 'exact', 'linear', or 'none', got {self.method!r}"
            )
        # Source frame index fields
        for _attr in (
            "source_frame_index",
            "left_source_frame_index",
            "right_source_frame_index",
        ):
            _val = getattr(self, _attr)
            if _val is not None:
                if isinstance(_val, bool) or not isinstance(_val, int):
                    raise TypeError(
                        f"{_attr} must be an integer or None, got {type(_val).__name__}"
                    )
                if _val < 0:
                    raise ValueError(f"{_attr} must be >= 0")
        # Timestamp fields
        for _attr in (
            "source_timestamp_seconds",
            "target_timestamp_seconds",
            "left_source_timestamp_seconds",
            "right_source_timestamp_seconds",
        ):
            _val = getattr(self, _attr)
            if _val is not None:
                if isinstance(_val, bool) or not isinstance(_val, (int, float)):
                    raise TypeError(
                        f"{_attr} must be a finite number or None, "
                        f"got {type(_val).__name__}"
                    )
                if not math.isfinite(_val):
                    raise ValueError(f"{_attr} must be finite")
        if isinstance(self.min_endpoint_coverage, bool) or not isinstance(
            self.min_endpoint_coverage, (int, float)
        ):
            raise TypeError(
                f"min_endpoint_coverage must be a finite number, "
                f"got {type(self.min_endpoint_coverage).__name__}"
            )
        if not math.isfinite(self.min_endpoint_coverage):
            raise ValueError("min_endpoint_coverage must be finite")
        if self.min_endpoint_coverage < 0.0:
            raise ValueError("min_endpoint_coverage must be >= 0")
        if not isinstance(self.contributors, tuple):
            raise TypeError("contributors must be a tuple")
        for _c in self.contributors:
            if not isinstance(_c, str) or not _c:
                raise ValueError("contributors entries must be nonempty strings")
        if not isinstance(self.qc_flags, tuple):
            raise TypeError("qc_flags must be a tuple")
        for _f in self.qc_flags:
            if not isinstance(_f, str) or not _f:
                raise ValueError("qc_flags entries must be nonempty strings")
        # -- consistency --
        if self.usable and self.com is None:
            raise ValueError("usable is True but com is None")
        if self.sample_kind == "original" and self.normalized_index is not None:
            raise ValueError(
                "sample_kind is 'original' but normalized_index is not None"
            )
        if self.sample_kind == "normalized" and self.normalized_index is None:
            raise ValueError("sample_kind is 'normalized' but normalized_index is None")

    @property
    def frame_index(self) -> int | None:
        """Backward-compatible alias for source_frame_index."""
        return self.source_frame_index


def _sample_qc_flags(
    fr: FrameComResult,
) -> tuple[str, ...]:
    """Build QC flags for a sample from its frame's nonexclusive mass totals.

    Includes each independent QC flag whose accumulated mass among usable
    segments is positive.  Missing-mass segments are excluded from the
    flag set.
    """
    flags: list[str] = []
    totals = fr.provenance_mass_totals()
    for _flag in (
        "raw_observed",
        "x_interpolated",
        "y_interpolated",
        "x_smoothing_changed",
        "y_smoothing_changed",
        "x_smoothing_support_interpolation",
        "y_smoothing_support_interpolation",
        "other_qc_limited",
    ):
        if totals.get(_flag, 0.0) > 0.0:
            flags.append(_flag)
    return tuple(flags)


def _sample_contributors(fr: FrameComResult) -> tuple[str, ...]:
    names: set[str] = set()
    for sr in fr.segment_results:
        if sr.usable:
            names.update(sr.provenance.contributors)
    return tuple(sorted(names))


def _find_bracket(
    stride_frames: list[FrameComResult],
    target_ts: float,
) -> tuple[FrameComResult, FrameComResult] | None:
    """Find the unique adjacent pair bracketing *target_ts*.

    Iterates consecutive pairs; returns (left, right) where
    left.timestamp < target < right.timestamp, or ``None``.
    """
    for left, right in zip(stride_frames, stride_frames[1:], strict=False):
        if left.timestamp_seconds < target_ts < right.timestamp_seconds:
            return (left, right)
    return None


def normalize_stride_com(
    frame_results: tuple[FrameComResult, ...],
    stride_start_frame: int,
    stride_end_frame: int,
    stride_start_timestamp: float,
    stride_end_timestamp: float,
    config: ComEstimationConfig,
) -> tuple[StrideComSample, ...]:
    """Normalize COM samples to a fixed stride grid.

    Collects frames ``stride_start_frame .. stride_end_frame`` (inclusive),
    validates boundary and ordering invariants, then produces exactly
    ``config.normalized_stride_samples`` normalized rows by exact-match or
    bracket-pair interpolation on timestamps.  Original rows are emitted
    for every frame in the stride.
    """
    duration = stride_end_timestamp - stride_start_timestamp
    if duration <= 0.0:
        raise ValueError(f"Stride duration must be positive, got {duration}")

    # ---- collect frames in bounds, keyed by index ----
    stride_by_index: dict[int, FrameComResult] = {}
    for fr in frame_results:
        if stride_start_frame <= fr.frame_index <= stride_end_frame:
            stride_by_index[fr.frame_index] = fr

    # ---- validate: every index in [start, end] must be present ----
    expected_count = stride_end_frame - stride_start_frame + 1
    if len(stride_by_index) != expected_count:
        raise ValueError(
            f"Stride requires all frame indices"
            f" {stride_start_frame}..{stride_end_frame}"
            f" ({expected_count} frames),"
            f" but found {len(stride_by_index)}"
        )

    # ---- build sorted frame list (by frame_index, then by timestamp) ----
    sorted_indices = sorted(stride_by_index.keys())
    for idx_pos, idx in enumerate(sorted_indices):
        if idx != stride_start_frame + idx_pos:
            raise ValueError(
                f"Frame indices are not consecutive: expected"
                f" {stride_start_frame + idx_pos}, got {idx}"
            )

    stride_frames: list[FrameComResult] = [stride_by_index[i] for i in sorted_indices]

    # ---- validate strictly increasing finite timestamps ----
    for k in range(len(stride_frames)):
        ts = stride_frames[k].timestamp_seconds
        if not math.isfinite(ts):
            raise ValueError(
                f"Frame {stride_frames[k].frame_index} has non-finite timestamp {ts}"
            )
        if k > 0:
            prev_ts = stride_frames[k - 1].timestamp_seconds
            if ts <= prev_ts:
                raise ValueError(
                    f"Timestamps not strictly increasing: frame"
                    f" {stride_frames[k - 1].frame_index} ({prev_ts}) >= frame"
                    f" {stride_frames[k].frame_index} ({ts})"
                )

    # ---- validate first/last timestamps match supplied boundaries ----
    if abs(stride_frames[0].timestamp_seconds - stride_start_timestamp) > 1e-9:
        raise ValueError(
            f"First frame timestamp {stride_frames[0].timestamp_seconds} does not match"
            f" stride_start_timestamp {stride_start_timestamp} within 1e-9"
        )
    if abs(stride_frames[-1].timestamp_seconds - stride_end_timestamp) > 1e-9:
        raise ValueError(
            f"Last frame timestamp {stride_frames[-1].timestamp_seconds} does not match"
            f" stride_end_timestamp {stride_end_timestamp} within 1e-9"
        )

    # ---- original samples: one per frame in index order ----
    original_samples: list[StrideComSample] = []
    for fr_orig in stride_frames:
        progression = (
            (fr_orig.timestamp_seconds - stride_start_timestamp) / duration * 100.0
        )
        progression = max(0.0, min(100.0, progression))
        original_samples.append(
            StrideComSample(
                progression=progression,
                com=fr_orig.com,
                mass_coverage=fr_orig.mass_coverage,
                usable=fr_orig.usable,
                sample_kind="original",
                normalized_index=None,
                method="exact",
                source_frame_index=fr_orig.frame_index,
                source_timestamp_seconds=fr_orig.timestamp_seconds,
                target_timestamp_seconds=fr_orig.timestamp_seconds,
                left_source_frame_index=None,
                left_source_timestamp_seconds=None,
                right_source_frame_index=None,
                right_source_timestamp_seconds=None,
                min_endpoint_coverage=fr_orig.mass_coverage,
                contributors=_sample_contributors(fr_orig),
                qc_flags=_sample_qc_flags(fr_orig),
            )
        )

    # ---- normalized grid: N canonical progressions ----
    n_samples = config.normalized_stride_samples
    normalized_samples: list[StrideComSample] = []

    for i in range(n_samples):
        progression = i * 100.0 / (n_samples - 1)
        target_ts = stride_start_timestamp + (progression / 100.0) * duration

        # 1. Exact frame match by timestamp (within 1e-9)
        exact_fr: FrameComResult | None = None
        for fr in stride_frames:
            if abs(fr.timestamp_seconds - target_ts) <= 1e-9:
                exact_fr = fr
                break

        if exact_fr is not None:
            # Preserve that frame's COM/coverage; usable is frame.usable
            normalized_samples.append(
                StrideComSample(
                    progression=progression,
                    com=exact_fr.com,
                    mass_coverage=exact_fr.mass_coverage,
                    usable=exact_fr.usable,
                    sample_kind="normalized",
                    normalized_index=i,
                    method="exact",
                    source_frame_index=exact_fr.frame_index,
                    source_timestamp_seconds=exact_fr.timestamp_seconds,
                    target_timestamp_seconds=target_ts,
                    left_source_frame_index=None,
                    left_source_timestamp_seconds=None,
                    right_source_frame_index=None,
                    right_source_timestamp_seconds=None,
                    min_endpoint_coverage=exact_fr.mass_coverage,
                    contributors=_sample_contributors(exact_fr),
                    qc_flags=_sample_qc_flags(exact_fr),
                )
            )
            continue

        # 2. Bracket by time: find unique adjacent pair straddling target
        bracket = _find_bracket(stride_frames, target_ts)
        if bracket is not None:
            left_fr, right_fr = bracket
            left_ts = left_fr.timestamp_seconds
            right_ts = right_fr.timestamp_seconds

            # Interpolate only if both endpoints are usable with COM
            # AND have identical usable_segments sets. Different segment
            # sets imply different denominators; blending centroids with
            # different denominators is undefined.
            both_usable = (
                left_fr.usable
                and left_fr.com is not None
                and right_fr.usable
                and right_fr.com is not None
            )
            same_segments = left_fr.usable_segments() == right_fr.usable_segments()
            if both_usable and same_segments:
                assert left_fr.com is not None and right_fr.com is not None
                frac = (target_ts - left_ts) / (right_ts - left_ts)
                interp_x = left_fr.com.x + frac * (right_fr.com.x - left_fr.com.x)
                interp_y = left_fr.com.y + frac * (right_fr.com.y - left_fr.com.y)
                min_cov = min(left_fr.mass_coverage, right_fr.mass_coverage)
                all_contrib = tuple(
                    sorted(
                        set(_sample_contributors(left_fr))
                        | set(_sample_contributors(right_fr))
                    )
                )
                all_qc = tuple(
                    sorted(
                        set(_sample_qc_flags(left_fr)) | set(_sample_qc_flags(right_fr))
                    )
                )
                normalized_samples.append(
                    StrideComSample(
                        progression=progression,
                        com=Point2D(interp_x, interp_y),
                        mass_coverage=min_cov,
                        usable=True,
                        sample_kind="normalized",
                        normalized_index=i,
                        method="linear",
                        source_frame_index=None,
                        source_timestamp_seconds=None,
                        target_timestamp_seconds=target_ts,
                        left_source_frame_index=left_fr.frame_index,
                        left_source_timestamp_seconds=left_ts,
                        right_source_frame_index=right_fr.frame_index,
                        right_source_timestamp_seconds=right_ts,
                        min_endpoint_coverage=min_cov,
                        contributors=all_contrib,
                        qc_flags=all_qc,
                    )
                )
                continue

            # Bracket found but unusable endpoints, missing COM,
            # or different usable segment sets between adjacent frames.
            # Do not blend centroids with different denominators.
            union_contrib = tuple(
                sorted(
                    set(_sample_contributors(left_fr))
                    | set(_sample_contributors(right_fr))
                )
            )
            union_qc = list(
                sorted(set(_sample_qc_flags(left_fr)) | set(_sample_qc_flags(right_fr)))
            )
            if both_usable and not same_segments:
                if "represented_segment_set_changed" not in union_qc:
                    union_qc.append("represented_segment_set_changed")
            normalized_samples.append(
                StrideComSample(
                    progression=progression,
                    com=None,
                    mass_coverage=0.0,
                    usable=False,
                    sample_kind="normalized",
                    normalized_index=i,
                    method="none",
                    source_frame_index=None,
                    source_timestamp_seconds=None,
                    target_timestamp_seconds=target_ts,
                    left_source_frame_index=left_fr.frame_index,
                    left_source_timestamp_seconds=left_ts,
                    right_source_frame_index=right_fr.frame_index,
                    right_source_timestamp_seconds=right_ts,
                    min_endpoint_coverage=0.0,
                    contributors=union_contrib,
                    qc_flags=tuple(sorted(union_qc)),
                )
            )
            continue

        # 3. No bracket found (target equals boundary or outside stride)
        normalized_samples.append(
            StrideComSample(
                progression=progression,
                com=None,
                mass_coverage=0.0,
                usable=False,
                sample_kind="normalized",
                normalized_index=i,
                method="none",
                source_frame_index=None,
                source_timestamp_seconds=None,
                target_timestamp_seconds=target_ts,
                left_source_frame_index=None,
                left_source_timestamp_seconds=None,
                right_source_frame_index=None,
                right_source_timestamp_seconds=None,
                min_endpoint_coverage=0.0,
                contributors=(),
                qc_flags=(),
            )
        )

    return tuple(original_samples + normalized_samples)


# ---------------------------------------------------------------------------
# Pipeline integration helper
# ---------------------------------------------------------------------------


def create_landmark_provenance_from_processed(
    processed_row: dict[str, Any],
) -> LandmarkProvenance:
    """Create LandmarkProvenance from a processed_landmarks.csv row."""
    return LandmarkProvenance(
        landmark_name=str(processed_row["landmark_name"]),
        raw_observed_usable=processed_row["observed_usable"] == "true",
        x_interpolated=processed_row["x_interpolated"] == "true",
        y_interpolated=processed_row["y_interpolated"] == "true",
        x_smoothing_changed=processed_row["x_smoothing_changed"] == "true",
        y_smoothing_changed=processed_row["y_smoothing_changed"] == "true",
        x_smoothing_support_contains_interpolation=(
            processed_row["x_smoothing_support_contains_interpolation"] == "true"
        ),
        y_smoothing_support_contains_interpolation=(
            processed_row["y_smoothing_support_contains_interpolation"] == "true"
        ),
    )
