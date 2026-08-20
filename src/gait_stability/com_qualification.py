"""Pure COM coverage, QC, and feasibility qualification calculations (Step 5b).

This module consumes Step 5a artifacts (FrameComResult, StrideComSample,
reviewed strides) and produces auditable coverage/QC diagnostics without
modifying Step 5a COM coordinates.

Key distinctions preserved:
- Absolute mass_coverage (total-body mass fraction) vs supported_mass_coverage
  (fraction of theoretically representable mass)
- Structurally unsupported segments (head) vs persistent/intermittent missing
  supported segments
- Engineering/QC qualification categories vs biomechanical validity
- Normalized sample availability recomputed per threshold from source frames only
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Literal

from gait_stability.com_estimation import (
    BILATERAL_SEGMENTS,
    DE_LEVA_FEMALE,
    DE_LEVA_MALE,
    REPRESENTED_MASS_MAX_FEMALE,
    REPRESENTED_MASS_MAX_MALE,
    SEGMENT_ENDPOINTS,
    SEGMENT_NAMES,
    UNSUPPORTED_SEGMENTS,
    FrameComResult,
    StrideComSample,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default sensitivity grid for absolute mass_coverage thresholds
DEFAULT_COVERAGE_THRESHOLDS: tuple[float, ...] = (
    0.80,
    0.82,
    0.84,
    0.86,
    0.88,
    0.90,
)

# Segment mass fractions by sex (from de Leva model)
SEGMENT_MASS_FRACTIONS: dict[str, dict[str, float]] = {
    "male": {seg: vals["mass"] for seg, vals in DE_LEVA_MALE.items()},
    "female": {seg: vals["mass"] for seg, vals in DE_LEVA_FEMALE.items()},
}

# Theoretical maximum supported mass fraction per sex (excludes head)
THEORETICAL_SUPPORTED_MASS: dict[str, float] = {
    "male": REPRESENTED_MASS_MAX_MALE,  # 0.9306
    "female": REPRESENTED_MASS_MAX_FEMALE,  # 0.9331
}

# Segment to base model name mapping
SEGMENT_TO_BASE: dict[str, str] = {}
for seg in SEGMENT_NAMES:
    if seg in ("head", "trunk"):
        SEGMENT_TO_BASE[seg] = seg
    else:
        SEGMENT_TO_BASE[seg] = seg.split("_", 1)[1]

# Tiny tolerance for floating-point excess clamping
_FLOAT_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComQualificationConfig:
    """Configuration for COM coverage qualification sensitivity analysis.

    Attributes:
        coverage_thresholds: Sorted tuple of absolute mass_coverage thresholds
            for sensitivity analysis. Must be finite, unique, strictly increasing,
            within [0, 1]. Default: (0.80, 0.82, 0.84, 0.86, 0.88, 0.90).

    Note:
        Primary Step 5a threshold (minimum_mass_coverage) and anthropometry sex
        are arguments to the aggregate calculator, not defaults inferred here.
    """

    coverage_thresholds: tuple[float, ...] = DEFAULT_COVERAGE_THRESHOLDS

    def __post_init__(self) -> None:
        if not isinstance(self.coverage_thresholds, tuple):
            raise TypeError("coverage_thresholds must be a tuple")
        if len(self.coverage_thresholds) == 0:
            raise ValueError("coverage_thresholds must not be empty")

        seen: set[float] = set()
        prev: float | None = None
        for i, th in enumerate(self.coverage_thresholds):
            if isinstance(th, bool) or not isinstance(th, (int, float)):
                raise TypeError(f"coverage_thresholds[{i}] must be a number")
            if not math.isfinite(th):
                raise ValueError(f"coverage_thresholds[{i}] must be finite")
            if not 0.0 <= th <= 1.0:
                raise ValueError(f"coverage_thresholds[{i}] must be between 0 and 1")
            if th in seen:
                raise ValueError(f"coverage_thresholds[{i}] duplicate value {th}")
            if prev is not None and th <= prev:
                raise ValueError(
                    f"coverage_thresholds must be strictly increasing: {prev} -> {th}"
                )
            seen.add(th)
            prev = th


# ---------------------------------------------------------------------------
# Typed input row structures (consumed from Step 5a and Step 3 artifacts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedFrameComQCRow:
    """Single frame COM quality-control row from com_proxy.csv.

    This mirrors the exported CSV columns needed for qualification without
    depending on private pipeline types.
    """

    frame_index: int
    timestamp_seconds: float
    frame_status: str
    com_x: float | None
    com_y: float | None
    mass_coverage: float
    usable: bool
    usable_segments: tuple[str, ...]
    missing_segments: tuple[str, ...]
    contributors_raw_observed: tuple[str, ...]
    contributors_x_interpolated: tuple[str, ...]
    contributors_y_interpolated: tuple[str, ...]
    contributors_x_smoothing_changed: tuple[str, ...]
    contributors_y_smoothing_changed: tuple[str, ...]
    contributors_x_smoothing_support_interpolation: tuple[str, ...]
    contributors_y_smoothing_support_interpolation: tuple[str, ...]
    contributors_other_qc_limited: tuple[str, ...]
    mass_x_interpolated: float
    mass_y_interpolated: float
    mass_x_smoothing_changed: float
    mass_y_smoothing_changed: float
    mass_x_smoothing_support_interpolation: float
    mass_y_smoothing_support_interpolation: float
    mass_other_qc_limited: float
    mass_missing: float


@dataclass(frozen=True, slots=True)
class ParsedStrideComSample:
    """Single stride COM sample from stride_com.csv."""

    stride_id: str
    side: str
    sample_kind: Literal["original", "normalized"]
    normalized_index: int | None
    progression: float
    method: Literal["exact", "linear", "none"]
    source_frame_index: int | None
    source_timestamp_seconds: float | None
    target_timestamp_seconds: float | None
    left_source_frame_index: int | None
    left_source_timestamp_seconds: float | None
    right_source_frame_index: int | None
    right_source_timestamp_seconds: float | None
    com_x: float | None
    com_y: float | None
    mass_coverage: float
    usable: bool
    min_endpoint_coverage: float
    contributors: tuple[str, ...]
    qc_flags: tuple[str, ...]
    # Review provenance extras
    automatic_stride_id: str
    review_intent: str
    review_changes: str
    provenance_notes: str


@dataclass(frozen=True, slots=True)
class ParsedReviewedStride:
    """Single reviewed stride window from reviewed_strides.csv."""

    stride_id: str
    side: str
    start_event_id: str
    end_event_id: str
    start_frame: int
    end_frame: int
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    duration_seconds: float
    quality: str
    contralateral_event_id: str | None
    contralateral_event_count: int
    sequence_notes: tuple[str, ...]
    source: str
    review_status: str
    automatic_stride_id: str
    review_intent: str
    review_changes: str
    provenance_notes: str


@dataclass(frozen=True, slots=True)
class ProcessedLandmarkQCRow:
    """Single processed landmark QC row from processed_landmarks.csv.

    Directly mirrors the Step 3 processed landmark QC flags without
    aggregation from segment provenance.
    """

    frame_index: int
    landmark_name: str
    final_available: bool
    raw_observed_usable: bool
    x_interpolated: bool
    y_interpolated: bool
    x_smoothing_changed: bool
    y_smoothing_changed: bool
    x_smoothing_support_interpolation: bool
    y_smoothing_support_interpolation: bool

    def __post_init__(self) -> None:
        # Validate boolean fields
        bool_fields = [
            "final_available",
            "raw_observed_usable",
            "x_interpolated",
            "y_interpolated",
            "x_smoothing_changed",
            "y_smoothing_changed",
            "x_smoothing_support_interpolation",
            "y_smoothing_support_interpolation",
        ]
        for field_name in bool_fields:
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be bool, got {type(value).__name__}"
                )
        if self.frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        if not self.landmark_name:
            raise ValueError("landmark_name must be nonempty")


@dataclass(frozen=True, slots=True)
class ReviewedStrideWindow:
    """Reviewed stride window with validated frame index bounds."""

    stride_id: str
    side: str
    start_frame: int
    end_frame: int
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    duration_seconds: float
    automatic_stride_id: str
    review_intent: str
    review_changes: str
    provenance_notes: str


@dataclass(frozen=True, slots=True)
class NormalizedAvailabilityResult:
    """Normalized sample availability for one stride at one threshold.

    Recomputed from source FrameComResult rows only, using canonical grid
    timestamps. No coordinate interpolation or calculation.
    """

    stride_id: str
    threshold: float
    canonical_N: int
    canonical_timestamps: tuple[float, ...]

    exact_match_usable: int
    linear_interpolation_usable: int
    unavailable: int
    total_normalized: int

    exact_match_fraction: float
    linear_interpolation_fraction: float
    usable_fraction: float

    start_endpoint_available: bool
    end_endpoint_available: bool


# ---------------------------------------------------------------------------
# Summary output dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentCoverageSummary:
    """Per-segment coverage diagnostics.

    Structurally unsupported (head) is reported explicitly but excluded from
    persistent/intermittent missing and lost representable mass totals.
    """

    segment_name: str
    mass_fraction: float
    is_supported: bool
    is_structurally_unsupported: bool  # head only

    total_frames: int
    usable_frames: int
    usable_fraction: float

    raw_observed_frames: int
    raw_observed_fraction: float

    x_interpolated_frames: int
    x_interpolated_fraction: float
    y_interpolated_frames: int
    y_interpolated_fraction: float

    x_smoothing_changed_frames: int
    x_smoothing_changed_fraction: float
    y_smoothing_changed_frames: int
    y_smoothing_changed_fraction: float

    x_smoothing_support_interpolation_frames: int
    x_smoothing_support_interpolation_fraction: float
    y_smoothing_support_interpolation_frames: int
    y_smoothing_support_interpolation_fraction: float

    other_qc_limited_frames: int
    other_qc_limited_fraction: float

    missing_frames: int
    missing_fraction: float

    longest_contiguous_missing_run: int
    lost_representable_mass: float  # mass_fraction * missing_fraction
    # (0 if structurally unsupported)

    missingness_pattern: Literal[
        "structurally_unsupported", "persistent", "intermittent", "none"
    ]


@dataclass(frozen=True, slots=True)
class LandmarkCoverageSummary:
    """Per-landmark coverage diagnostics from processed rows (nonexclusive mass)."""

    landmark_name: str
    total_frames: int

    raw_observed_usable_frames: int
    raw_observed_usable_fraction: float

    x_interpolated_frames: int
    x_interpolated_fraction: float
    y_interpolated_frames: int
    y_interpolated_fraction: float

    x_smoothing_changed_frames: int
    x_smoothing_changed_fraction: float
    y_smoothing_changed_frames: int
    y_smoothing_changed_fraction: float

    x_smoothing_support_interpolation_frames: int
    x_smoothing_support_interpolation_fraction: float
    y_smoothing_support_interpolation_frames: int
    y_smoothing_support_interpolation_fraction: float

    final_missing_frames: int
    final_missing_fraction: float

    longest_contiguous_missing_run: int

    # NONEXCLUSIVE: affected mass across frames where landmark contributed to
    # a usable segment. Overlaps across landmarks are intentional.
    nonexclusive_affected_mass_fraction: float


@dataclass(frozen=True, slots=True)
class AsymmetrySummary:
    """Left/right asymmetry for bilateral segments or meaningful landmarks."""

    pair_name: str  # e.g., "upper_arm" or "wrist"
    left_name: str
    right_name: str
    mass_fraction_per_side: float  # segment mass (0 for landmarks)

    left_usable_fraction: float
    right_usable_fraction: float

    left_raw_observed_fraction: float
    right_raw_observed_fraction: float

    left_missing_fraction: float
    right_missing_fraction: float

    usability_difference: float  # left - right
    raw_observed_difference: float
    missing_difference: float


@dataclass(frozen=True, slots=True)
class StrideCoverageSummary:
    """Per-stride coverage qualification (engineering/QC category)."""

    stride_id: str
    side: str

    frame_count: int
    finite_com_frames: int
    finite_com_fraction: float

    # At primary threshold
    usable_frames_primary: int
    usable_fraction_primary: float

    # Mass coverage distributions (all frames in stride)
    mass_coverage_min: float
    mass_coverage_max: float
    mass_coverage_mean: float
    mass_coverage_median: float

    supported_mass_coverage_min: float
    supported_mass_coverage_max: float
    supported_mass_coverage_mean: float
    supported_mass_coverage_median: float

    # Supported segment missing burden
    # (persistent + intermittent, excludes structurally unsupported)
    supported_segment_missing_count: int
    supported_segment_missing_frames: int
    supported_segment_missing_max_consecutive: int

    # Longest unusable interval at primary threshold
    longest_unusable_interval_frames: int
    longest_unusable_interval_seconds: float

    # Normalized sample availability at primary threshold
    normalized_samples_total: int
    normalized_samples_usable: int
    normalized_samples_usable_fraction: float
    normalized_exact_match_count: int
    normalized_linear_interpolation_count: int

    # Qualification (engineering/QC only)
    qualification_category: Literal[
        "policy_complete_at_threshold",
        "usable_samples_only",
        "insufficient_coverage",
        "no_usable_frames",
        "endpoint_unavailable",
    ]
    failure_reasons: tuple[str, ...]

    # Explicit stride boolean diagnostics for policy-complete decomposition
    all_original_frames_policy_eligible: bool
    all_supported_segments_represented: bool
    represented_segment_set_invariant: bool
    normalized_grid_complete: bool
    endpoints_policy_eligible: bool
    all_contributing_segments_raw_observed: bool

    # All threshold sensitivity for this stride
    threshold_sensitivity: dict[float, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ThresholdSensitivityResult:
    """Coverage sensitivity at one absolute threshold across all frames."""

    threshold: float
    equivalent_supported_threshold: float  # threshold / theoretical_supported

    total_frames: int
    usable_frames: int
    usable_fraction: float

    longest_usable_interval_frames: int
    longest_usable_interval_seconds: float

    strides_with_any_usable: int
    total_strides: int

    policy_complete_strides: int

    # Per-stride usable original frame counts
    stride_usable_original_frames: dict[str, int]

    # Normalized stride samples at this threshold (recomputed)
    normalized_total_samples: int
    normalized_exact_match_usable: int
    normalized_linear_interpolation_usable: int
    normalized_usable_samples: int
    normalized_usable_fraction: float


@dataclass(frozen=True, slots=True)
class AggregateCoverageResult:
    """Aggregate coverage across all frames and strides."""

    theoretical_supported_mass_fraction: float
    empirical_max_mass_coverage: float
    empirical_max_supported_mass_coverage: float

    total_frames: int
    finite_com_frames: int
    finite_com_fraction: float

    # At primary threshold
    usable_frames_primary: int
    usable_fraction_primary: float

    # Mass coverage distributions
    mass_coverage_min: float
    mass_coverage_max: float
    mass_coverage_mean: float
    mass_coverage_median: float

    supported_mass_coverage_min: float
    supported_mass_coverage_max: float
    supported_mass_coverage_mean: float
    supported_mass_coverage_median: float

    # Supported segment missing burden
    # (persistent + intermittent, excludes structurally unsupported)
    supported_segment_missing_frames: int
    supported_segment_missing_max_consecutive: int

    # Longest unusable interval at primary threshold
    longest_unusable_interval_frames: int
    longest_unusable_interval_seconds: float

    # Normalized availability at primary threshold
    normalized_total_samples: int
    normalized_exact_match_usable: int
    normalized_linear_interpolation_usable: int
    normalized_usable_samples: int
    normalized_usable_fraction: float

    # Summaries
    segment_summaries: tuple[SegmentCoverageSummary, ...]
    landmark_summaries: tuple[LandmarkCoverageSummary, ...]
    asymmetry_summaries: tuple[AsymmetrySummary, ...]
    stride_summaries: tuple[StrideCoverageSummary, ...]
    threshold_sensitivity: tuple[ThresholdSensitivityResult, ...]


# ---------------------------------------------------------------------------
# Core pure functions
# ---------------------------------------------------------------------------


def theoretical_supported_mass_fraction(sex: Literal["male", "female"]) -> float:
    """Return the theoretical maximum supported mass fraction for the given sex.

    Male: 0.9306 (1.0000 - 0.0694 head)
    Female: 0.9331 (0.9999 - 0.0668 head)

    Args:
        sex: "male" or "female"

    Returns:
        Theoretical supported mass fraction.
    """
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")
    return THEORETICAL_SUPPORTED_MASS[sex]


def supported_mass_coverage(mass_coverage: float, theoretical: float) -> float:
    """Compute supported_mass_coverage = mass_coverage / theoretical.

    Clamps tiny floating-point excess to 1.0 if within tolerance.
    Raises for materially out-of-range values.

    Args:
        mass_coverage: Absolute mass coverage (sum of usable segment masses).
        theoretical: Theoretical maximum supported mass fraction.

    Returns:
        Dimensionless ratio in [0, 1].

    Raises:
        ValueError: If theoretical <= 0 or ratio materially exceeds 1.0 or below 0.
    """
    if not math.isfinite(mass_coverage):
        raise ValueError("mass_coverage must be finite")
    if not math.isfinite(theoretical):
        raise ValueError("theoretical must be finite")
    if theoretical <= 0.0:
        raise ValueError("theoretical_supported_mass_fraction must be positive")

    ratio = mass_coverage / theoretical
    if ratio > 1.0:
        if ratio - 1.0 <= _FLOAT_TOLERANCE:
            return 1.0
        raise ValueError(
            f"supported_mass_coverage {ratio:.6f} exceeds 1.0 materially "
            f"(mass_coverage={mass_coverage:.6f}, theoretical={theoretical:.6f})"
        )
    if ratio < 0.0:
        raise ValueError(
            f"supported_mass_coverage {ratio:.6f} below 0 "
            f"(mass_coverage={mass_coverage:.6f})"
        )
    return ratio


def longest_contiguous_run(values: list[bool], target: bool = True) -> int:
    """Length of longest contiguous run of `target` in boolean list.

    Args:
        values: Boolean sequence.
        target: Value to count (True for longest True run, False for longest False).

    Returns:
        Maximum consecutive count.
    """
    max_run = 0
    current = 0
    for v in values:
        if v == target:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run


def longest_contiguous_run_duration(
    values: list[bool],
    timestamps: list[float],
    target: bool = True,
) -> tuple[int, float]:
    """Longest contiguous run of `target` with its duration in seconds.

    Args:
        values: Boolean sequence aligned with timestamps.
        timestamps: Strictly increasing frame timestamps in seconds.
        target: Value to find runs for.

    Returns:
        (max_run_frames, max_run_seconds). Seconds is end_timestamp - start_timestamp
        of the longest run (0.0 for single-frame runs).
    """
    if not values or not timestamps or len(values) != len(timestamps):
        return 0, 0.0

    max_run = 0
    max_duration = 0.0
    current_start: int | None = None

    for i, v in enumerate(values):
        if v == target:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                run_len = i - current_start
                if run_len > max_run:
                    max_run = run_len
                    max_duration = timestamps[i - 1] - timestamps[current_start]
                current_start = None

    if current_start is not None:
        run_len = len(values) - current_start
        if run_len > max_run:
            max_run = run_len
            max_duration = timestamps[-1] - timestamps[current_start]

    return max_run, max_duration


def frame_eligible_at_threshold(
    frame: FrameComResult,
    threshold: float,
) -> bool:
    """Re-evaluate frame eligibility at a given threshold.

    A frame is eligible iff:
    - COM is finite (com is not None)
    - mass_coverage > 0
    - mass_coverage >= threshold (equality passes)

    Args:
        frame: FrameComResult from Step 5a.
        threshold: Absolute mass_coverage threshold.

    Returns:
        True if frame meets eligibility at this threshold.
    """
    if frame.com is None:
        return False
    if frame.mass_coverage <= 0.0:
        return False
    return frame.mass_coverage >= threshold


def recompute_normalized_availability(
    stride_frames: list[FrameComResult],
    stride_window: ReviewedStrideWindow,
    canonical_timestamps: list[float],
    threshold: float,
) -> NormalizedAvailabilityResult:
    """Recompute normalized sample availability for a stride at a threshold.

    Uses ONLY existing frame rows (no coordinate interpolation). Rules:
    - Exact timestamp match with eligible frame -> exact_match_usable
    - Otherwise, only adjacent consecutive frame indices
      (right.frame_index == left.frame_index + 1) with BOTH frames eligible
      AND IDENTICAL usable supported-segment tuples may supply linear availability
    - All other cases -> unavailable

    Validates:
    - Canonical timestamps are finite and strictly increasing
    - Stride frames have finite timestamps and strict index+timestamp ordering
    - Nonconsecutive frame indices bracket -> unavailable

    Args:
        stride_frames: FrameComResult for frames in [start_frame, end_frame],
            in frame_index order, consecutive.
        stride_window: Reviewed stride window metadata.
        canonical_timestamps: N canonical timestamps for normalized grid.
        threshold: Absolute mass_coverage threshold for eligibility.

    Returns:
        NormalizedAvailabilityResult with exact/linear/none counts and endpoints.
    """
    # Validate canonical timestamps: finite and strictly increasing
    if canonical_timestamps:
        prev_ts = None
        for ts in canonical_timestamps:
            if not math.isfinite(ts):
                raise ValueError("canonical_timestamps must be finite")
            if prev_ts is not None and ts <= prev_ts:
                raise ValueError("canonical_timestamps must be strictly increasing")
            prev_ts = ts

    # Validate stride frames: finite timestamps, strict index and timestamp ordering
    if stride_frames:
        prev_fr = None
        for fr in stride_frames:
            if not math.isfinite(fr.timestamp_seconds):
                raise ValueError("stride frame timestamps must be finite")
            if prev_fr is not None:
                if fr.frame_index <= prev_fr.frame_index:
                    raise ValueError("stride frame indices must be strictly increasing")
                if fr.timestamp_seconds <= prev_fr.timestamp_seconds:
                    raise ValueError(
                        "stride frame timestamps must be strictly increasing"
                    )
            prev_fr = fr

    if not stride_frames:
        return NormalizedAvailabilityResult(
            stride_id=stride_window.stride_id,
            threshold=threshold,
            canonical_N=len(canonical_timestamps),
            canonical_timestamps=tuple(canonical_timestamps),
            exact_match_usable=0,
            linear_interpolation_usable=0,
            unavailable=len(canonical_timestamps),
            total_normalized=len(canonical_timestamps),
            exact_match_fraction=0.0,
            linear_interpolation_fraction=0.0,
            usable_fraction=0.0,
            start_endpoint_available=False,
            end_endpoint_available=False,
        )

    # Eligibility and usable segment sets for each frame in stride
    eligible: dict[int, bool] = {}
    usable_segments_map: dict[int, tuple[str, ...]] = {}
    for fr in stride_frames:
        elig = frame_eligible_at_threshold(fr, threshold)
        eligible[fr.frame_index] = elig
        if elig:
            usable_segments_map[fr.frame_index] = fr.usable_segments()
        else:
            usable_segments_map[fr.frame_index] = ()

    exact_match = 0
    linear_interp = 0
    unavailable = 0

    for target_ts in canonical_timestamps:
        # 1. Exact timestamp match (within 1e-9)
        exact_frame: FrameComResult | None = None
        for fr in stride_frames:
            if abs(fr.timestamp_seconds - target_ts) <= 1e-9:
                exact_frame = fr
                break

        if exact_frame is not None:
            if eligible[exact_frame.frame_index]:
                exact_match += 1
            else:
                unavailable += 1
            continue

        # 2. Find bracket: adjacent consecutive frame indices straddling target_ts
        bracket_found = False
        for i in range(len(stride_frames) - 1):
            left_fr = stride_frames[i]
            right_fr = stride_frames[i + 1]
            if left_fr.timestamp_seconds < target_ts < right_fr.timestamp_seconds:
                bracket_found = True
                # Enforce no-gap rule: right.frame_index == left.frame_index + 1
                if right_fr.frame_index != left_fr.frame_index + 1:
                    unavailable += 1
                # Both must be eligible AND have identical usable segment sets
                elif (
                    eligible[left_fr.frame_index]
                    and eligible[right_fr.frame_index]
                    and usable_segments_map[left_fr.frame_index]
                    == usable_segments_map[right_fr.frame_index]
                ):
                    linear_interp += 1
                else:
                    unavailable += 1
                break

        if not bracket_found:
            unavailable += 1

    total = len(canonical_timestamps)
    return NormalizedAvailabilityResult(
        stride_id=stride_window.stride_id,
        threshold=threshold,
        canonical_N=total,
        canonical_timestamps=tuple(canonical_timestamps),
        exact_match_usable=exact_match,
        linear_interpolation_usable=linear_interp,
        unavailable=unavailable,
        total_normalized=total,
        exact_match_fraction=exact_match / total if total else 0.0,
        linear_interpolation_fraction=linear_interp / total if total else 0.0,
        usable_fraction=(exact_match + linear_interp) / total if total else 0.0,
        start_endpoint_available=eligible.get(stride_window.start_frame, False),
        end_endpoint_available=eligible.get(stride_window.end_frame, False),
    )


def _get_segment_mass(sex: Literal["male", "female"], segment: str) -> float:
    """Get mass fraction for a segment.

    Accepts either a canonical segment instance name (e.g., "left_upper_arm")
    or a base model segment name (e.g., "upper_arm"). Validates the name exists
    in the selected anthropometric model.

    Args:
        sex: "male" or "female" for the anthropometric model.
        segment: Canonical segment name (e.g., "left_upper_arm") or base model
            name (e.g., "upper_arm", "head", "trunk").

    Returns:
        Segment mass fraction from the de Leva model.

    Raises:
        ValueError: If segment name is not recognized in the model.
    """
    model_fractions = SEGMENT_MASS_FRACTIONS[sex]

    # First try direct lookup (base model names like "upper_arm", "head", "trunk")
    if segment in model_fractions:
        return model_fractions[segment]

    # Then try canonical-to-base mapping (e.g., "left_upper_arm" -> "upper_arm")
    base = SEGMENT_TO_BASE.get(segment)
    if base is not None and base in model_fractions:
        return model_fractions[base]

    # Neither worked - provide a clear error with valid options
    valid_names = set(model_fractions.keys()) | set(SEGMENT_TO_BASE.keys())
    raise ValueError(
        f"Unknown segment name '{segment}' for {sex} model. "
        f"Valid names: {sorted(valid_names)}"
    )


def _is_structurally_unsupported(segment: str) -> bool:
    """Check if segment is structurally unsupported (head only)."""
    return segment in UNSUPPORTED_SEGMENTS


def compute_segment_coverage(
    segment_name: str,
    frame_results: tuple[FrameComResult, ...],
    sex: Literal["male", "female"],
) -> SegmentCoverageSummary:
    """Compute per-segment coverage summary from FrameComResult tuple.

    Excludes structurally unsupported head from persistent/intermittent missing
    and lost representable mass totals.
    """
    mass_fraction = _get_segment_mass(sex, segment_name)
    is_supported = segment_name not in UNSUPPORTED_SEGMENTS
    is_structurally_unsupported = _is_structurally_unsupported(segment_name)

    total_frames = len(frame_results)
    usable = []
    raw_obs = []
    xi = []
    yi = []
    xsc = []
    ysc = []
    xssi = []
    yssi = []
    other = []
    missing = []

    for fr in frame_results:
        seg_result = next(
            (s for s in fr.segment_results if s.segment_name == segment_name),
            None,
        )
        if seg_result is None:
            missing.append(True)
            usable.append(False)
            raw_obs.append(False)
            xi.append(False)
            yi.append(False)
            xsc.append(False)
            ysc.append(False)
            xssi.append(False)
            yssi.append(False)
            other.append(False)
            continue

        prov = seg_result.provenance
        usable.append(seg_result.usable)
        raw_obs.append(prov.all_raw_observed)
        xi.append(prov.any_x_interpolated)
        yi.append(prov.any_y_interpolated)
        xsc.append(prov.any_x_smoothing_changed)
        ysc.append(prov.any_y_smoothing_changed)
        xssi.append(prov.any_x_smoothing_support_interpolation)
        yssi.append(prov.any_y_smoothing_support_interpolation)
        other.append(prov.other_qc_limited)
        missing.append(not seg_result.usable)

    usable_frames = sum(usable)
    usable_fraction = usable_frames / total_frames if total_frames else 0.0

    raw_observed_frames = sum(raw_obs)
    raw_observed_fraction = raw_observed_frames / total_frames if total_frames else 0.0

    x_interpolated_frames = sum(xi)
    x_interpolated_fraction = (
        x_interpolated_frames / total_frames if total_frames else 0.0
    )
    y_interpolated_frames = sum(yi)
    y_interpolated_fraction = (
        y_interpolated_frames / total_frames if total_frames else 0.0
    )

    x_smoothing_changed_frames = sum(xsc)
    x_smoothing_changed_fraction = (
        x_smoothing_changed_frames / total_frames if total_frames else 0.0
    )
    y_smoothing_changed_frames = sum(ysc)
    y_smoothing_changed_fraction = (
        y_smoothing_changed_frames / total_frames if total_frames else 0.0
    )

    x_smoothing_support_interpolation_frames = sum(xssi)
    x_smoothing_support_interpolation_fraction = (
        x_smoothing_support_interpolation_frames / total_frames if total_frames else 0.0
    )
    y_smoothing_support_interpolation_frames = sum(yssi)
    y_smoothing_support_interpolation_fraction = (
        y_smoothing_support_interpolation_frames / total_frames if total_frames else 0.0
    )

    other_qc_limited_frames = sum(other)
    other_qc_limited_fraction = (
        other_qc_limited_frames / total_frames if total_frames else 0.0
    )

    # For structurally unsupported segments, missing frames/fraction/run
    # are 0/not counted
    if is_structurally_unsupported:
        missing_frames = 0
        missing_fraction = 0.0
        longest_contiguous_missing_run = 0
    else:
        missing_frames = sum(missing)
        missing_fraction = missing_frames / total_frames if total_frames else 0.0
        longest_contiguous_missing_run = longest_contiguous_run(missing, True)

    # Determine missingness pattern
    if is_structurally_unsupported:
        missingness_pattern: Literal[
            "structurally_unsupported", "persistent", "intermittent", "none"
        ] = "structurally_unsupported"
    elif missing_frames == 0:
        missingness_pattern = "none"
    elif missing_frames == total_frames:
        missingness_pattern = "persistent"
    else:
        missingness_pattern = "intermittent"

    # Exclude structurally unsupported from lost representable mass
    lost_representable_mass = mass_fraction * missing_fraction if is_supported else 0.0

    return SegmentCoverageSummary(
        segment_name=segment_name,
        mass_fraction=mass_fraction,
        is_supported=is_supported,
        is_structurally_unsupported=is_structurally_unsupported,
        total_frames=total_frames,
        usable_frames=usable_frames,
        usable_fraction=usable_fraction,
        raw_observed_frames=raw_observed_frames,
        raw_observed_fraction=raw_observed_fraction,
        x_interpolated_frames=x_interpolated_frames,
        x_interpolated_fraction=x_interpolated_fraction,
        y_interpolated_frames=y_interpolated_frames,
        y_interpolated_fraction=y_interpolated_fraction,
        x_smoothing_changed_frames=x_smoothing_changed_frames,
        x_smoothing_changed_fraction=x_smoothing_changed_fraction,
        y_smoothing_changed_frames=y_smoothing_changed_frames,
        y_smoothing_changed_fraction=y_smoothing_changed_fraction,
        x_smoothing_support_interpolation_frames=x_smoothing_support_interpolation_frames,
        x_smoothing_support_interpolation_fraction=x_smoothing_support_interpolation_fraction,
        y_smoothing_support_interpolation_frames=y_smoothing_support_interpolation_frames,
        y_smoothing_support_interpolation_fraction=y_smoothing_support_interpolation_fraction,
        other_qc_limited_frames=other_qc_limited_frames,
        other_qc_limited_fraction=other_qc_limited_fraction,
        missing_frames=missing_frames,
        missing_fraction=missing_fraction,
        longest_contiguous_missing_run=longest_contiguous_missing_run,
        lost_representable_mass=lost_representable_mass,
        missingness_pattern=missingness_pattern,
    )


def compute_landmark_coverage(
    landmark_name: str,
    landmark_rows: tuple[ProcessedLandmarkQCRow, ...],
    frame_results: tuple[FrameComResult, ...],
    sex: Literal["male", "female"],
    segment_dependency_map: dict[str, tuple[str, ...]],
) -> LandmarkCoverageSummary:
    """Compute per-landmark coverage from direct processed landmark rows.

    Uses processed landmark QC flags directly (not derived from segment provenance).
    Mass contributions are NONEXCLUSIVE (landmark serves multiple segments).
    nonexclusive_affected_mass_fraction = average supported model mass potentially
    disabled on frames where that landmark is final-missing: sum masses of dependent
    supported segments that are unusable on those frames, divided by total frames.
    Does not count structural unsupported head.
    """
    total_frames = len(frame_results)

    # Filter rows for this landmark (should be one per frame)
    lm_rows_by_frame: dict[int, ProcessedLandmarkQCRow] = {
        r.frame_index: r for r in landmark_rows if r.landmark_name == landmark_name
    }

    raw_obs = []
    xi = []
    yi = []
    xsc = []
    ysc = []
    xssi = []
    yssi = []
    missing = []

    affected_mass_sum = 0.0

    for fr in frame_results:
        fi = fr.frame_index
        row = lm_rows_by_frame.get(fi)

        if row is None:
            # No row for this landmark on this frame -> treat as missing
            raw_obs.append(False)
            xi.append(False)
            yi.append(False)
            xsc.append(False)
            ysc.append(False)
            xssi.append(False)
            yssi.append(False)
            missing.append(True)
            continue

        # Direct flags from processed rows
        final_available = row.final_available
        raw_obs.append(row.raw_observed_usable)
        xi.append(row.x_interpolated)
        yi.append(row.y_interpolated)
        xsc.append(row.x_smoothing_changed)
        ysc.append(row.y_smoothing_changed)
        xssi.append(row.x_smoothing_support_interpolation)
        yssi.append(row.y_smoothing_support_interpolation)
        missing.append(not final_available)

        # Compute affected mass on frames where landmark is final-missing
        if not final_available:
            frame_affected_mass = 0.0
            for seg_name in segment_dependency_map.get(landmark_name, ()):
                if _is_structurally_unsupported(seg_name):
                    continue  # Do not count structural unsupported head
                seg_result = next(
                    (s for s in fr.segment_results if s.segment_name == seg_name),
                    None,
                )
                if seg_result is not None and not seg_result.usable:
                    frame_affected_mass += seg_result.mass_fraction
            affected_mass_sum += frame_affected_mass

    raw_observed_usable_frames = sum(raw_obs)
    raw_observed_usable_fraction = (
        raw_observed_usable_frames / total_frames if total_frames else 0.0
    )

    x_interpolated_frames = sum(xi)
    x_interpolated_fraction = (
        x_interpolated_frames / total_frames if total_frames else 0.0
    )
    y_interpolated_frames = sum(yi)
    y_interpolated_fraction = (
        y_interpolated_frames / total_frames if total_frames else 0.0
    )

    x_smoothing_changed_frames = sum(xsc)
    x_smoothing_changed_fraction = (
        x_smoothing_changed_frames / total_frames if total_frames else 0.0
    )
    y_smoothing_changed_frames = sum(ysc)
    y_smoothing_changed_fraction = (
        y_smoothing_changed_frames / total_frames if total_frames else 0.0
    )

    x_smoothing_support_interpolation_frames = sum(xssi)
    x_smoothing_support_interpolation_fraction = (
        x_smoothing_support_interpolation_frames / total_frames if total_frames else 0.0
    )
    y_smoothing_support_interpolation_frames = sum(yssi)
    y_smoothing_support_interpolation_fraction = (
        y_smoothing_support_interpolation_frames / total_frames if total_frames else 0.0
    )

    final_missing_frames = sum(missing)
    final_missing_fraction = (
        final_missing_frames / total_frames if total_frames else 0.0
    )

    longest_contiguous_missing_run = longest_contiguous_run(missing, True)

    # Average affected mass per frame (nonexclusive across landmarks)
    nonexclusive_affected_mass_fraction = (
        affected_mass_sum / total_frames if total_frames else 0.0
    )

    return LandmarkCoverageSummary(
        landmark_name=landmark_name,
        total_frames=total_frames,
        raw_observed_usable_frames=raw_observed_usable_frames,
        raw_observed_usable_fraction=raw_observed_usable_fraction,
        x_interpolated_frames=x_interpolated_frames,
        x_interpolated_fraction=x_interpolated_fraction,
        y_interpolated_frames=y_interpolated_frames,
        y_interpolated_fraction=y_interpolated_fraction,
        x_smoothing_changed_frames=x_smoothing_changed_frames,
        x_smoothing_changed_fraction=x_smoothing_changed_fraction,
        y_smoothing_changed_frames=y_smoothing_changed_frames,
        y_smoothing_changed_fraction=y_smoothing_changed_fraction,
        x_smoothing_support_interpolation_frames=x_smoothing_support_interpolation_frames,
        x_smoothing_support_interpolation_fraction=x_smoothing_support_interpolation_fraction,
        y_smoothing_support_interpolation_frames=y_smoothing_support_interpolation_frames,
        y_smoothing_support_interpolation_fraction=y_smoothing_support_interpolation_fraction,
        final_missing_frames=final_missing_frames,
        final_missing_fraction=final_missing_fraction,
        longest_contiguous_missing_run=longest_contiguous_missing_run,
        nonexclusive_affected_mass_fraction=nonexclusive_affected_mass_fraction,
    )


def build_segment_dependency_map() -> dict[str, tuple[str, ...]]:
    """Build landmark -> segment dependency map from SEGMENT_ENDPOINTS.

    Returns mapping from landmark name to tuple of segment names that use it.
    Includes derived landmarks (midpoints) as their component landmarks.
    """
    dep_map: dict[str, set[str]] = {}

    for segment_name, (prox, dist) in SEGMENT_ENDPOINTS.items():
        if segment_name in UNSUPPORTED_SEGMENTS:
            continue
        # Add proximal landmark
        components: tuple[str, ...]
        if prox in ("shoulder_midpoint", "hip_midpoint"):
            if prox == "shoulder_midpoint":
                components = ("left_shoulder", "right_shoulder")
            elif prox == "hip_midpoint":
                components = ("left_hip", "right_hip")
            else:
                components = (prox,)
        elif prox.endswith("_midpoint_proxy"):
            if prox == "left_index_pinky_midpoint_proxy":
                components = ("left_index", "left_pinky")
            elif prox == "right_index_pinky_midpoint_proxy":
                components = ("right_index", "right_pinky")
            else:
                components = (prox,)
        else:
            components = (prox,)

        for comp in components:
            dep_map.setdefault(comp, set()).add(segment_name)

        # Add distal landmark
        if dist in ("shoulder_midpoint", "hip_midpoint"):
            if dist == "shoulder_midpoint":
                components = ("left_shoulder", "right_shoulder")
            elif dist == "hip_midpoint":
                components = ("left_hip", "right_hip")
            else:
                components = (dist,)
        elif dist.endswith("_midpoint_proxy"):
            if dist == "left_index_pinky_midpoint_proxy":
                components = ("left_index", "left_pinky")
            elif dist == "right_index_pinky_midpoint_proxy":
                components = ("right_index", "right_pinky")
            else:
                components = (dist,)
        else:
            components = (dist,)

        for comp in components:
            dep_map.setdefault(comp, set()).add(segment_name)

    return {k: tuple(sorted(v)) for k, v in dep_map.items()}


def compute_asymmetry_summaries(
    segment_summaries: tuple[SegmentCoverageSummary, ...],
    landmark_summaries: tuple[LandmarkCoverageSummary, ...],
    sex: Literal["male", "female"],
) -> tuple[AsymmetrySummary, ...]:
    """Compute left/right asymmetry for bilateral segments and meaningful landmarks.

    Segment asymmetry uses segment summaries directly.
    Landmark asymmetry uses direct LandmarkCoverageSummary pairs.
    """
    results: list[AsymmetrySummary] = []

    # Segment asymmetry from segment summaries
    seg_by_name = {s.segment_name: s for s in segment_summaries}
    for base in BILATERAL_SEGMENTS:
        left_seg = f"left_{base}"
        right_seg = f"right_{base}"
        mass_fraction = _get_segment_mass(sex, base)

        left_sum = seg_by_name.get(left_seg)
        right_sum = seg_by_name.get(right_seg)

        if left_sum is None or right_sum is None:
            continue

        results.append(
            AsymmetrySummary(
                pair_name=base,
                left_name=left_seg,
                right_name=right_seg,
                mass_fraction_per_side=mass_fraction,
                left_usable_fraction=left_sum.usable_fraction,
                right_usable_fraction=right_sum.usable_fraction,
                left_raw_observed_fraction=left_sum.raw_observed_fraction,
                right_raw_observed_fraction=right_sum.raw_observed_fraction,
                left_missing_fraction=left_sum.missing_fraction,
                right_missing_fraction=right_sum.missing_fraction,
                usability_difference=left_sum.usable_fraction
                - right_sum.usable_fraction,
                raw_observed_difference=left_sum.raw_observed_fraction
                - right_sum.raw_observed_fraction,
                missing_difference=left_sum.missing_fraction
                - right_sum.missing_fraction,
            )
        )

    # Landmark asymmetry from landmark summaries (direct pairs)
    lm_by_name: dict[str, LandmarkCoverageSummary] = {
        s.landmark_name: s for s in landmark_summaries
    }
    bilateral_landmarks = (
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
        ("left_foot_index", "right_foot_index"),
    )

    for left_lm, right_lm in bilateral_landmarks:
        left_lm_sum = lm_by_name.get(left_lm)
        right_lm_sum = lm_by_name.get(right_lm)

        if left_lm_sum is None or right_lm_sum is None:
            continue

        left_usable = 1.0 - left_lm_sum.final_missing_fraction
        right_usable = 1.0 - right_lm_sum.final_missing_fraction

        results.append(
            AsymmetrySummary(
                pair_name=left_lm.split("_", 1)[1],  # e.g., "shoulder"
                left_name=left_lm,
                right_name=right_lm,
                mass_fraction_per_side=0.0,  # landmarks don't have mass
                left_usable_fraction=left_usable,
                right_usable_fraction=right_usable,
                left_raw_observed_fraction=left_lm_sum.raw_observed_usable_fraction,
                right_raw_observed_fraction=right_lm_sum.raw_observed_usable_fraction,
                left_missing_fraction=left_lm_sum.final_missing_fraction,
                right_missing_fraction=right_lm_sum.final_missing_fraction,
                usability_difference=left_usable - right_usable,
                raw_observed_difference=left_lm_sum.raw_observed_usable_fraction
                - right_lm_sum.raw_observed_usable_fraction,
                missing_difference=left_lm_sum.final_missing_fraction
                - right_lm_sum.final_missing_fraction,
            )
        )

    return tuple(results)


def compute_threshold_sensitivity(
    frame_results: tuple[FrameComResult, ...],
    stride_windows: list[ReviewedStrideWindow],
    stride_samples_by_id: dict[str, tuple[StrideComSample, ...]],
    canonical_timestamps_by_stride: dict[str, list[float]],
    config: ComQualificationConfig,
    primary_threshold: float,
    anthropometry_sex: Literal["male", "female"],
) -> tuple[ThresholdSensitivityResult, ...]:
    """Compute sensitivity across all configured thresholds.

    Recomputes normalized availability per threshold from source frames only.
    Never reuses Step 5a's original `usable` or normalized rows.
    """
    theoretical = theoretical_supported_mass_fraction(anthropometry_sex)
    timestamps = [fr.timestamp_seconds for fr in frame_results]

    # Build stride frame index mapping
    stride_frame_map: dict[str, tuple[int, int]] = {}
    for s in stride_windows:
        stride_frame_map[s.stride_id] = (s.start_frame, s.end_frame)

    results: list[ThresholdSensitivityResult] = []

    for th in config.coverage_thresholds:
        # Frame-level eligibility at this threshold
        eligible_flags = [frame_eligible_at_threshold(fr, th) for fr in frame_results]
        usable_frames = sum(eligible_flags)
        usable_fraction = usable_frames / len(frame_results) if frame_results else 0.0

        # Longest contiguous usable interval (frames and seconds)
        longest_usable_frames, longest_usable_seconds = longest_contiguous_run_duration(
            eligible_flags, timestamps, True
        )

        # Stride-level analysis
        strides_with_any = 0
        policy_complete = 0
        stride_usable_original: dict[str, int] = {}

        normalized_total = 0
        normalized_exact = 0
        normalized_linear = 0
        normalized_usable = 0

        for s in stride_windows:
            stride_id = s.stride_id
            start_f, end_f = stride_frame_map[stride_id]
            stride_frames = [
                fr for fr in frame_results if start_f <= fr.frame_index <= end_f
            ]
            stride_frames.sort(key=lambda f: f.frame_index)

            # Usable original frames in this stride at this threshold
            stride_usable = sum(
                1 for fr in stride_frames if frame_eligible_at_threshold(fr, th)
            )
            stride_usable_original[stride_id] = stride_usable

            if stride_usable > 0:
                strides_with_any += 1

            # Policy complete at threshold: ALL original frames eligible,
            # invariant segment set, all normalized points available,
            # endpoints available
            all_eligible = all(
                frame_eligible_at_threshold(fr, th) for fr in stride_frames
            )
            if all_eligible and stride_frames:
                first_segments = stride_frames[0].usable_segments()
                invariant = all(
                    fr.usable_segments() == first_segments for fr in stride_frames
                )
                # Recompute normalized availability at this threshold
                norm_result = recompute_normalized_availability(
                    stride_frames,
                    s,
                    canonical_timestamps_by_stride.get(stride_id, []),
                    th,
                )
                all_norm_avail = (
                    norm_result.unavailable == 0
                    and norm_result.start_endpoint_available
                    and norm_result.end_endpoint_available
                )
                if invariant and all_norm_avail:
                    policy_complete += 1

            # Accumulate normalized counts (recomputed per threshold)
            norm_result = recompute_normalized_availability(
                stride_frames,
                s,
                canonical_timestamps_by_stride.get(stride_id, []),
                th,
            )
            normalized_total += norm_result.total_normalized
            normalized_exact += norm_result.exact_match_usable
            normalized_linear += norm_result.linear_interpolation_usable
            normalized_usable += (
                norm_result.exact_match_usable + norm_result.linear_interpolation_usable
            )

        equivalent_supported = th / theoretical if theoretical > 0 else 0.0
        if (
            equivalent_supported > 1.0
            and equivalent_supported - 1.0 <= _FLOAT_TOLERANCE
        ):
            equivalent_supported = 1.0

        results.append(
            ThresholdSensitivityResult(
                threshold=th,
                equivalent_supported_threshold=equivalent_supported,
                total_frames=len(frame_results),
                usable_frames=usable_frames,
                usable_fraction=usable_fraction,
                longest_usable_interval_frames=longest_usable_frames,
                longest_usable_interval_seconds=longest_usable_seconds,
                strides_with_any_usable=strides_with_any,
                total_strides=len(stride_windows),
                policy_complete_strides=policy_complete,
                stride_usable_original_frames=stride_usable_original,
                normalized_total_samples=normalized_total,
                normalized_exact_match_usable=normalized_exact,
                normalized_linear_interpolation_usable=normalized_linear,
                normalized_usable_samples=normalized_usable,
                normalized_usable_fraction=(
                    normalized_usable / normalized_total if normalized_total else 0.0
                ),
            )
        )

    return tuple(results)


def compute_stride_coverage(
    stride_id: str,
    stride_samples: tuple[StrideComSample, ...],
    frame_results: tuple[FrameComResult, ...],
    stride_window: ReviewedStrideWindow,
    canonical_timestamps: list[float],
    primary_threshold: float,
    anthropometry_sex: Literal["male", "female"],
    coverage_thresholds: tuple[float, ...],
) -> StrideCoverageSummary:
    """Compute per-stride coverage qualification at primary threshold.

    Policy-complete at threshold requires:
    - Every original frame eligible at primary threshold
      (all_original_frames_policy_eligible)
    - Invariant represented segment set across all frames
      (represented_segment_set_invariant)
    - All normalized points available via exact or linear
      (normalized_grid_complete)
    - Start and end endpoints available (endpoints_policy_eligible)

    Does NOT require all supported segments represented or all raw observed;
    those are independently exposed.
    """
    start_f = stride_window.start_frame
    end_f = stride_window.end_frame
    stride_frames = [fr for fr in frame_results if start_f <= fr.frame_index <= end_f]
    stride_frames.sort(key=lambda f: f.frame_index)

    frame_count = len(stride_frames)
    finite_com = sum(1 for fr in stride_frames if fr.com is not None)
    finite_com_fraction = finite_com / frame_count if frame_count else 0.0

    # Mass coverage distributions
    coverages = [fr.mass_coverage for fr in stride_frames]
    mass_cov_min = min(coverages) if coverages else 0.0
    mass_cov_max = max(coverages) if coverages else 0.0
    mass_cov_mean = sum(coverages) / len(coverages) if coverages else 0.0
    mass_cov_median = statistics.median(coverages) if coverages else 0.0

    theoretical = theoretical_supported_mass_fraction(anthropometry_sex)
    supported_coverages = [supported_mass_coverage(c, theoretical) for c in coverages]
    sup_min = min(supported_coverages) if supported_coverages else 0.0
    sup_max = max(supported_coverages) if supported_coverages else 0.0
    sup_mean = (
        sum(supported_coverages) / len(supported_coverages)
        if supported_coverages
        else 0.0
    )
    sup_median = statistics.median(supported_coverages) if supported_coverages else 0.0

    # Supported segment missing burden
    # (persistent + intermittent, excludes structurally unsupported)
    supported_segment_missing_count = 0
    supported_segment_missing_frames = 0
    supported_segment_missing_runs: list[int] = []

    for seg_name in SEGMENT_NAMES:
        if _is_structurally_unsupported(seg_name):
            continue
        segment_missing_at_least_once = False
        missing_run = 0
        for fr in stride_frames:
            seg_res = next(
                (s for s in fr.segment_results if s.segment_name == seg_name), None
            )
            if seg_res and not seg_res.usable:
                if not segment_missing_at_least_once:
                    segment_missing_at_least_once = True
                    supported_segment_missing_count += 1
                supported_segment_missing_frames += 1
                missing_run += 1
            else:
                if missing_run > 0:
                    supported_segment_missing_runs.append(missing_run)
                    missing_run = 0
        if missing_run > 0:
            supported_segment_missing_runs.append(missing_run)

    supported_segment_missing_max_consecutive = (
        max(supported_segment_missing_runs) if supported_segment_missing_runs else 0
    )

    # Primary threshold usability
    primary_eligible = [
        frame_eligible_at_threshold(fr, primary_threshold) for fr in stride_frames
    ]
    primary_usable = sum(primary_eligible)
    usable_fraction_primary = primary_usable / frame_count if frame_count else 0.0

    # Longest unusable interval at primary threshold
    primary_unusable = [not e for e in primary_eligible]
    longest_unusable_frames, longest_unusable_seconds = longest_contiguous_run_duration(
        primary_unusable,
        [fr.timestamp_seconds for fr in stride_frames],
        True,
    )

    # Normalized sample availability at primary threshold (recomputed)
    norm_result = recompute_normalized_availability(
        stride_frames, stride_window, canonical_timestamps, primary_threshold
    )

    # Explicit stride boolean diagnostics
    all_original_frames_policy_eligible = all(primary_eligible)
    first_segments = stride_frames[0].usable_segments() if stride_frames else ()
    represented_segment_set_invariant = all(
        fr.usable_segments() == first_segments for fr in stride_frames
    )
    # all_supported_segments_represented: every supported segment is
    # usable at least once
    supported_seg_names = [
        s for s in SEGMENT_NAMES if not _is_structurally_unsupported(s)
    ]
    all_supported_segments_represented = all(
        any(
            next(
                (
                    s
                    for s in fr.segment_results
                    if s.segment_name == seg_name and s.usable
                ),
                None,
            )
            is not None
            for fr in stride_frames
        )
        for seg_name in supported_seg_names
    )
    normalized_grid_complete = (
        norm_result.unavailable == 0
        and norm_result.start_endpoint_available
        and norm_result.end_endpoint_available
    )
    endpoints_policy_eligible = (
        norm_result.start_endpoint_available and norm_result.end_endpoint_available
    )
    # all_contributing_segments_raw_observed: all segments that contributed
    # to COM are raw-observed on all frames
    all_contributing_segments_raw_observed = all(
        all(
            next(
                (
                    s
                    for s in fr.segment_results
                    if s.segment_name == seg_name and s.provenance.all_raw_observed
                ),
                None,
            )
            is not None
            for seg_name in first_segments
        )
        for fr in stride_frames
    )

    # Qualification category
    failure_reasons: list[str] = []
    category: Literal[
        "policy_complete_at_threshold",
        "usable_samples_only",
        "insufficient_coverage",
        "no_usable_frames",
        "endpoint_unavailable",
    ]

    if frame_count == 0:
        category = "endpoint_unavailable"
        failure_reasons.append("no frames in stride")
    elif finite_com == 0:
        category = "no_usable_frames"
        failure_reasons.append("no finite COM frames in stride")
    elif not all_original_frames_policy_eligible:
        # Not all frames eligible
        category = "insufficient_coverage"
        failure_reasons.append(
            f"usable fraction {usable_fraction_primary:.3f} < 1.0 at primary threshold"
        )
    else:
        # All frames eligible - check policy-complete criteria
        all_norm_avail = normalized_grid_complete
        if represented_segment_set_invariant and all_norm_avail:
            category = "policy_complete_at_threshold"
        else:
            category = "usable_samples_only"
            if not represented_segment_set_invariant:
                failure_reasons.append("represented segment set not invariant")
            if norm_result.unavailable > 0:
                failure_reasons.append("not all normalized samples available")
            if not norm_result.start_endpoint_available:
                failure_reasons.append("start endpoint COM unavailable")
            if not norm_result.end_endpoint_available:
                failure_reasons.append("end endpoint COM unavailable")

    # Threshold sensitivity for this stride using configured thresholds
    threshold_sensitivity: dict[float, dict[str, Any]] = {}
    for th in coverage_thresholds:
        stride_usable = sum(
            1 for fr in stride_frames if frame_eligible_at_threshold(fr, th)
        )
        norm_at_th = recompute_normalized_availability(
            stride_frames, stride_window, canonical_timestamps, th
        )
        threshold_sensitivity[th] = {
            "usable_original_frames": stride_usable,
            "usable_fraction": stride_usable / frame_count if frame_count else 0.0,
            "normalized_usable": norm_at_th.exact_match_usable
            + norm_at_th.linear_interpolation_usable,
            "normalized_exact": norm_at_th.exact_match_usable,
            "normalized_linear": norm_at_th.linear_interpolation_usable,
        }

    return StrideCoverageSummary(
        stride_id=stride_id,
        side=stride_window.side,
        frame_count=frame_count,
        finite_com_frames=finite_com,
        finite_com_fraction=finite_com_fraction,
        usable_frames_primary=primary_usable,
        usable_fraction_primary=usable_fraction_primary,
        mass_coverage_min=mass_cov_min,
        mass_coverage_max=mass_cov_max,
        mass_coverage_mean=mass_cov_mean,
        mass_coverage_median=mass_cov_median,
        supported_mass_coverage_min=sup_min,
        supported_mass_coverage_max=sup_max,
        supported_mass_coverage_mean=sup_mean,
        supported_mass_coverage_median=sup_median,
        supported_segment_missing_count=supported_segment_missing_count,
        supported_segment_missing_frames=supported_segment_missing_frames,
        supported_segment_missing_max_consecutive=supported_segment_missing_max_consecutive,
        longest_unusable_interval_frames=longest_unusable_frames,
        longest_unusable_interval_seconds=longest_unusable_seconds,
        normalized_samples_total=norm_result.total_normalized,
        normalized_samples_usable=norm_result.exact_match_usable
        + norm_result.linear_interpolation_usable,
        normalized_samples_usable_fraction=norm_result.usable_fraction,
        normalized_exact_match_count=norm_result.exact_match_usable,
        normalized_linear_interpolation_count=norm_result.linear_interpolation_usable,
        qualification_category=category,
        failure_reasons=tuple(failure_reasons),
        all_original_frames_policy_eligible=all_original_frames_policy_eligible,
        all_supported_segments_represented=all_supported_segments_represented,
        represented_segment_set_invariant=represented_segment_set_invariant,
        normalized_grid_complete=normalized_grid_complete,
        endpoints_policy_eligible=endpoints_policy_eligible,
        all_contributing_segments_raw_observed=all_contributing_segments_raw_observed,
        threshold_sensitivity=threshold_sensitivity,
    )


def compute_qualification(
    frame_results: tuple[FrameComResult, ...],
    stride_results: dict[str, tuple[StrideComSample, ...]],
    reviewed_strides: list[ParsedReviewedStride],
    config: ComQualificationConfig,
    primary_threshold: float,
    anthropometry_sex: Literal["male", "female"],
    landmark_rows: tuple[ProcessedLandmarkQCRow, ...],
) -> AggregateCoverageResult:
    """Main entry point: compute full qualification from Step 5a results
    and Step 3 landmark QC.

    Args:
        frame_results: Tuple of FrameComResult from Step 5a (all frames).
        stride_results: Dict mapping stride_id to StrideComSample tuples.
        reviewed_strides: List of ParsedReviewedStride from reviewed_strides.csv.
        config: Qualification configuration (threshold grid only).
        primary_threshold: The Step 5a configured minimum_mass_coverage.
        anthropometry_sex: "male" or "female" as used in Step 5a.
        landmark_rows: Tuple of ProcessedLandmarkQCRow from processed_landmarks.csv.

    Returns:
        AggregateCoverageResult with all diagnostics.
    """
    if not frame_results:
        raise ValueError("frame_results must not be empty")
    if not reviewed_strides:
        raise ValueError("reviewed_strides must not be empty")
    if anthropometry_sex not in ("male", "female"):
        raise ValueError("anthropometry_sex must be 'male' or 'female'")

    sex = anthropometry_sex
    theoretical = theoretical_supported_mass_fraction(sex)

    # Build stride windows from reviewed strides
    stride_windows = [
        ReviewedStrideWindow(
            stride_id=s.stride_id,
            side=s.side,
            start_frame=s.start_frame,
            end_frame=s.end_frame,
            start_timestamp_seconds=s.start_timestamp_seconds,
            end_timestamp_seconds=s.end_timestamp_seconds,
            duration_seconds=s.duration_seconds,
            automatic_stride_id=s.automatic_stride_id,
            review_intent=s.review_intent,
            review_changes=s.review_changes,
            provenance_notes=s.provenance_notes,
        )
        for s in reviewed_strides
    ]

    # Build canonical timestamps for each stride from its normalized samples
    canonical_timestamps_by_stride: dict[str, list[float]] = {}
    for s in reviewed_strides:
        samples = stride_results.get(s.stride_id, ())
        norm_samples = [sp for sp in samples if sp.sample_kind == "normalized"]
        norm_samples.sort(key=lambda x: x.normalized_index or 0)
        canonical_timestamps_by_stride[s.stride_id] = [
            sp.target_timestamp_seconds
            for sp in norm_samples
            if sp.target_timestamp_seconds is not None
        ]

    # Aggregate frame-level stats
    total_frames = len(frame_results)
    finite_com = sum(1 for fr in frame_results if fr.com is not None)
    finite_com_fraction = finite_com / total_frames if total_frames else 0.0

    coverages = [fr.mass_coverage for fr in frame_results]
    mass_cov_min = min(coverages) if coverages else 0.0
    mass_cov_max = max(coverages) if coverages else 0.0
    mass_cov_mean = sum(coverages) / len(coverages) if coverages else 0.0
    mass_cov_median = statistics.median(coverages) if coverages else 0.0

    theoretical = theoretical_supported_mass_fraction(sex)
    supported_coverages = [supported_mass_coverage(c, theoretical) for c in coverages]
    sup_min = min(supported_coverages) if supported_coverages else 0.0
    sup_max = max(supported_coverages) if supported_coverages else 0.0
    sup_mean = (
        sum(supported_coverages) / len(supported_coverages)
        if supported_coverages
        else 0.0
    )
    sup_median = statistics.median(supported_coverages) if supported_coverages else 0.0

    empirical_max = mass_cov_max
    empirical_max_supported = sup_max

    # Primary threshold
    primary_eligible = [
        frame_eligible_at_threshold(fr, primary_threshold) for fr in frame_results
    ]
    primary_usable = sum(primary_eligible)
    usable_fraction_primary = primary_usable / total_frames if total_frames else 0.0

    # Supported segment missing burden
    # (persistent + intermittent, excludes structurally unsupported)
    supported_segment_missing_frames = 0
    supported_segment_missing_runs: list[int] = []
    for seg_name in SEGMENT_NAMES:
        if _is_structurally_unsupported(seg_name):
            continue
        missing_run = 0
        for fr in frame_results:
            seg_res = next(
                (s for s in fr.segment_results if s.segment_name == seg_name), None
            )
            if seg_res and not seg_res.usable:
                supported_segment_missing_frames += 1
                missing_run += 1
            else:
                if missing_run > 0:
                    supported_segment_missing_runs.append(missing_run)
                    missing_run = 0
        if missing_run > 0:
            supported_segment_missing_runs.append(missing_run)
    supported_segment_missing_max_consecutive = (
        max(supported_segment_missing_runs) if supported_segment_missing_runs else 0
    )

    # Longest unusable interval at primary threshold
    primary_unusable = [not e for e in primary_eligible]
    timestamps = [fr.timestamp_seconds for fr in frame_results]
    longest_unusable_frames, longest_unusable_seconds = longest_contiguous_run_duration(
        primary_unusable, timestamps, True
    )

    # Normalized availability at primary threshold (aggregate)
    normalized_total = 0
    normalized_exact = 0
    normalized_linear = 0
    normalized_usable = 0

    for window in stride_windows:
        stride_frames = [
            fr
            for fr in frame_results
            if window.start_frame <= fr.frame_index <= window.end_frame
        ]
        stride_frames.sort(key=lambda f: f.frame_index)
        norm_result = recompute_normalized_availability(
            stride_frames,
            window,
            canonical_timestamps_by_stride.get(window.stride_id, []),
            primary_threshold,
        )
        normalized_total += norm_result.total_normalized
        normalized_exact += norm_result.exact_match_usable
        normalized_linear += norm_result.linear_interpolation_usable
        normalized_usable += (
            norm_result.exact_match_usable + norm_result.linear_interpolation_usable
        )

    normalized_usable_fraction = (
        normalized_usable / normalized_total if normalized_total else 0.0
    )

    # Segment summaries
    segment_summaries = tuple(
        compute_segment_coverage(seg, frame_results, sex) for seg in SEGMENT_NAMES
    )

    # Landmark summaries using direct processed landmark rows
    dep_map = build_segment_dependency_map()
    # Use ALL landmarks in dependency map (required landmarks) even
    # if persistently missing
    all_landmarks = set(dep_map.keys())
    # Also include any landmarks that appear in processed rows
    for row in landmark_rows:
        all_landmarks.add(row.landmark_name)

    landmark_summaries = tuple(
        compute_landmark_coverage(lm, landmark_rows, frame_results, sex, dep_map)
        for lm in sorted(all_landmarks)
    )

    # Asymmetry summaries using direct segment and landmark summaries
    asymmetry_summaries = compute_asymmetry_summaries(
        segment_summaries, landmark_summaries, sex
    )

    # Stride summaries
    stride_window_by_id = {w.stride_id: w for w in stride_windows}
    stride_summaries = tuple(
        compute_stride_coverage(
            s.stride_id,
            stride_results.get(s.stride_id, ()),
            frame_results,
            stride_window_by_id[s.stride_id],
            canonical_timestamps_by_stride.get(s.stride_id, []),
            primary_threshold,
            sex,
            config.coverage_thresholds,
        )
        for s in reviewed_strides
    )

    # Threshold sensitivity
    threshold_sensitivity = compute_threshold_sensitivity(
        frame_results,
        stride_windows,
        stride_results,
        canonical_timestamps_by_stride,
        config,
        primary_threshold,
        sex,
    )

    return AggregateCoverageResult(
        theoretical_supported_mass_fraction=theoretical,
        empirical_max_mass_coverage=empirical_max,
        empirical_max_supported_mass_coverage=empirical_max_supported,
        total_frames=total_frames,
        finite_com_frames=finite_com,
        finite_com_fraction=finite_com_fraction,
        usable_frames_primary=primary_usable,
        usable_fraction_primary=usable_fraction_primary,
        mass_coverage_min=mass_cov_min,
        mass_coverage_max=mass_cov_max,
        mass_coverage_mean=mass_cov_mean,
        mass_coverage_median=mass_cov_median,
        supported_mass_coverage_min=sup_min,
        supported_mass_coverage_max=sup_max,
        supported_mass_coverage_mean=sup_mean,
        supported_mass_coverage_median=sup_median,
        supported_segment_missing_frames=supported_segment_missing_frames,
        supported_segment_missing_max_consecutive=supported_segment_missing_max_consecutive,
        longest_unusable_interval_frames=longest_unusable_frames,
        longest_unusable_interval_seconds=longest_unusable_seconds,
        normalized_total_samples=normalized_total,
        normalized_exact_match_usable=normalized_exact,
        normalized_linear_interpolation_usable=normalized_linear,
        normalized_usable_samples=normalized_usable,
        normalized_usable_fraction=normalized_usable_fraction,
        segment_summaries=segment_summaries,
        landmark_summaries=landmark_summaries,
        asymmetry_summaries=asymmetry_summaries,
        stride_summaries=stride_summaries,
        threshold_sensitivity=threshold_sensitivity,
    )
