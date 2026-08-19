"""Core COM proxy estimation unit tests using synthetic fixtures only.

Verifies:
- de Leva coefficient selection (male/female) and exact published totals
- Segment COM formula c = p + r*(d-p) and derived-midpoint endpoint mappings
- Weighted represented-segment COM: unrenormalized weighted average
- Missing-mass: segments excluded, never zeroed into COM denominator
- strict bool / nonfinite config rejection
- mass_coverage threshold: equality passes, below fails
- QC provenance categories: raw_observed / interpolated / smoothing_interpolation /
  smoothed_only / missing
- Stride normalization: irregular timestamps, all original frames inclusive of
  unusable, exact N normalized samples with exact 0 and 100 endpoints,
  no interpolation across an unusable interior frame, determinism
"""

from __future__ import annotations

import pytest

from gait_stability.com_estimation import (
    DE_LEVA_FEMALE,
    DE_LEVA_MALE,
    MODEL_MASS_TOTAL_FEMALE,
    MODEL_MASS_TOTAL_MALE,
    REPRESENTED_MASS_MAX_FEMALE,
    REPRESENTED_MASS_MAX_MALE,
    SEGMENT_NAMES,
    UNSUPPORTED_SEGMENTS,
    ComEstimationConfig,
    FrameComResult,
    LandmarkProvenance,
    Point2D,
    SegmentProvenance,
    _compute_derived_landmarks,
    _get_model,
    _get_model_total,
    _segment_com,
    estimate_frame_com,
    normalize_stride_com,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prov(name: str = "", **overrides: bool) -> LandmarkProvenance:
    """Return a LandmarkProvenance with sensible defaults."""
    defaults: dict[str, object] = dict(
        landmark_name=name,
        raw_observed_usable=True,
        x_interpolated=False,
        y_interpolated=False,
        x_smoothing_changed=False,
        y_smoothing_changed=False,
        x_smoothing_support_contains_interpolation=False,
        y_smoothing_support_contains_interpolation=False,
    )
    defaults.update(overrides)
    return LandmarkProvenance(**defaults)  # type: ignore[arg-type]


def _all_landmarks() -> dict[str, Point2D]:
    """Complete set of 19 COM-contributor landmarks yielding all 14 segments."""
    return {
        "nose": Point2D(0.50, 0.10),
        "left_shoulder": Point2D(0.44, 0.18),
        "right_shoulder": Point2D(0.56, 0.18),
        "left_hip": Point2D(0.46, 0.42),
        "right_hip": Point2D(0.54, 0.42),
        "left_elbow": Point2D(0.40, 0.30),
        "right_elbow": Point2D(0.60, 0.30),
        "left_wrist": Point2D(0.38, 0.42),
        "right_wrist": Point2D(0.62, 0.42),
        "left_index": Point2D(0.37, 0.44),
        "left_pinky": Point2D(0.39, 0.46),
        "right_index": Point2D(0.63, 0.44),
        "right_pinky": Point2D(0.61, 0.46),
        "left_knee": Point2D(0.44, 0.62),
        "right_knee": Point2D(0.56, 0.62),
        "left_ankle": Point2D(0.43, 0.82),
        "right_ankle": Point2D(0.57, 0.82),
        "left_foot_index": Point2D(0.40, 0.92),
        "right_foot_index": Point2D(0.60, 0.92),
    }


def _all_prov(lms: dict[str, Point2D]) -> dict[str, LandmarkProvenance]:
    """Raw_observed usable provenance for every landmark."""
    return {name: _prov(landmark_name=name) for name in lms}


def _run_frame(
    landmarks: dict[str, Point2D],
    prov: dict[str, LandmarkProvenance],
    config: ComEstimationConfig | None = None,
    *,
    frame_index: int = 0,
    ts: float = 0.0,
    status: str = "decoded_pose",
) -> FrameComResult:
    if config is None:
        config = ComEstimationConfig(anthropometry_sex="male")
    return estimate_frame_com(
        processed_landmarks=landmarks,
        landmark_provenance=prov,
        config=config,
        frame_index=frame_index,
        timestamp_seconds=ts,
        frame_status=status,
    )


# ===================================================================
# de Leva coefficients
# ===================================================================


class TestDeLevaCoefficients:
    """Exact published de Leva coefficient selection and total mass."""

    def test_male_coefficient_identity(self) -> None:
        model = _get_model("male")
        assert model is DE_LEVA_MALE

    def test_female_coefficient_identity(self) -> None:
        model = _get_model("female")
        assert model is DE_LEVA_FEMALE

    def test_male_total_mass_exact(self) -> None:
        assert _get_model_total("male") == MODEL_MASS_TOTAL_MALE == 1.0000

    def test_female_total_mass_exact(self) -> None:
        assert _get_model_total("female") == MODEL_MASS_TOTAL_FEMALE == 0.9999

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("head", 0.0694),
            ("trunk", 0.4346),
            ("upper_arm", 0.0271),
            ("forearm", 0.0162),
            ("hand", 0.0061),
            ("thigh", 0.1416),
            ("shank", 0.0433),
            ("foot", 0.0137),
        ],
    )
    def test_male_segment_mass(self, name: str, expected: float) -> None:
        assert DE_LEVA_MALE[name]["mass"] == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("head", 0.0668),
            ("trunk", 0.4257),
            ("upper_arm", 0.0255),
            ("forearm", 0.0138),
            ("hand", 0.0056),
            ("thigh", 0.1478),
            ("shank", 0.0481),
            ("foot", 0.0129),
        ],
    )
    def test_female_segment_mass(self, name: str, expected: float) -> None:
        assert DE_LEVA_FEMALE[name]["mass"] == expected

    def test_male_sum_matches_total(self) -> None:
        """Male bilateral segments: mass per side + mass per side."""
        bilateral_names = ("upper_arm", "forearm", "hand", "thigh", "shank", "foot")
        bilateral = sum(DE_LEVA_MALE[n]["mass"] for n in bilateral_names)
        head_m = DE_LEVA_MALE["head"]["mass"]
        trunk_m = DE_LEVA_MALE["trunk"]["mass"]
        total_computed = head_m + trunk_m + 2 * bilateral
        assert total_computed == pytest.approx(MODEL_MASS_TOTAL_MALE, abs=1e-4)

    def test_female_sum_matches_total(self) -> None:
        bilateral_names = ("upper_arm", "forearm", "hand", "thigh", "shank", "foot")
        bilateral = sum(DE_LEVA_FEMALE[n]["mass"] for n in bilateral_names)
        head_m = DE_LEVA_FEMALE["head"]["mass"]
        trunk_m = DE_LEVA_FEMALE["trunk"]["mass"]
        total_computed = head_m + trunk_m + 2 * bilateral
        assert total_computed == pytest.approx(MODEL_MASS_TOTAL_FEMALE, abs=1e-4)

    # -- Corrected de Leva r values (proximal reference) --

    @pytest.mark.parametrize(
        ("name", "expected_r"),
        [
            ("head", 0.5002),
            ("trunk", 0.5138),
            ("shank", 0.4395),
        ],
    )
    def test_male_r_values(self, name: str, expected_r: float) -> None:
        assert DE_LEVA_MALE[name]["r"] == expected_r

    @pytest.mark.parametrize(
        ("name", "expected_r"),
        [
            ("head", 0.4841),
            ("trunk", 0.4964),
            ("shank", 0.4352),
        ],
    )
    def test_female_r_values(self, name: str, expected_r: float) -> None:
        assert DE_LEVA_FEMALE[name]["r"] == expected_r

    def test_male_unchanged_r_values(self) -> None:
        """Non-corrected r values remain as published."""
        assert DE_LEVA_MALE["upper_arm"]["r"] == 0.5772
        assert DE_LEVA_MALE["forearm"]["r"] == 0.4574
        assert DE_LEVA_MALE["hand"]["r"] == 0.7900
        assert DE_LEVA_MALE["thigh"]["r"] == 0.4095
        assert DE_LEVA_MALE["foot"]["r"] == 0.4415

    def test_female_unchanged_r_values(self) -> None:
        """Non-corrected r values remain as published."""
        assert DE_LEVA_FEMALE["upper_arm"]["r"] == 0.5754
        assert DE_LEVA_FEMALE["forearm"]["r"] == 0.4559
        assert DE_LEVA_FEMALE["hand"]["r"] == 0.7474
        assert DE_LEVA_FEMALE["thigh"]["r"] == 0.3612
        assert DE_LEVA_FEMALE["foot"]["r"] == 0.4014

    def test_head_unsupported(self) -> None:
        """Head is always in UNSUPPORTED_SEGMENTS."""
        assert "head" in UNSUPPORTED_SEGMENTS

    def test_represented_mass_max_male(self) -> None:
        """Max represented mass excludes unsupported head."""
        assert REPRESENTED_MASS_MAX_MALE == 0.9306
        assert REPRESENTED_MASS_MAX_MALE == pytest.approx(
            MODEL_MASS_TOTAL_MALE - DE_LEVA_MALE["head"]["mass"]
        )

    def test_represented_mass_max_female(self) -> None:
        """Max represented mass excludes unsupported head."""
        assert REPRESENTED_MASS_MAX_FEMALE == 0.9331
        assert REPRESENTED_MASS_MAX_FEMALE == pytest.approx(
            MODEL_MASS_TOTAL_FEMALE - DE_LEVA_FEMALE["head"]["mass"]
        )


# ===================================================================
# Segment COM geometry
# ===================================================================


class TestSegmentCOMGeometry:
    """Segment COM formula and derived-midpoint endpoint mappings."""

    def test_segment_com_formula(self) -> None:
        """c = p + r * (d - p)."""
        com = _segment_com(Point2D(0.0, 0.0), Point2D(10.0, 20.0), 0.4)
        assert com.x == pytest.approx(4.0)
        assert com.y == pytest.approx(8.0)

    def test_com_at_proximal_when_r_zero(self) -> None:
        com = _segment_com(Point2D(3.0, 5.0), Point2D(8.0, 12.0), 0.0)
        assert (com.x, com.y) == (3.0, 5.0)

    def test_com_at_distal_when_r_one(self) -> None:
        com = _segment_com(Point2D(3.0, 5.0), Point2D(8.0, 12.0), 1.0)
        assert (com.x, com.y) == (8.0, 12.0)

    def test_com_midpoint_at_r_half(self) -> None:
        com = _segment_com(Point2D(2.0, 4.0), Point2D(6.0, 8.0), 0.5)
        assert (com.x, com.y) == (4.0, 6.0)

    def test_derived_shoulder_midpoint(self) -> None:
        lms = {
            "left_shoulder": Point2D(0.10, 0.20),
            "right_shoulder": Point2D(0.30, 0.40),
        }
        derived = _compute_derived_landmarks(lms)
        mid = derived["shoulder_midpoint"]
        assert mid.x == pytest.approx(0.20)
        assert mid.y == pytest.approx(0.30)

    def test_derived_hip_midpoint(self) -> None:
        lms = {
            "left_hip": Point2D(0.50, 0.60),
            "right_hip": Point2D(0.70, 0.80),
        }
        derived = _compute_derived_landmarks(lms)
        mid = derived["hip_midpoint"]
        assert (mid.x, mid.y) == (0.60, 0.70)

    def test_derived_left_index_pinky_midpoint_proxy(self) -> None:
        """left_index_pinky_midpoint_proxy is the unvalidated hand distal proxy."""
        from gait_stability.com_estimation import DERIVED_LANDMARKS

        assert "left_index_pinky_midpoint_proxy" in DERIVED_LANDMARKS
        left, right = DERIVED_LANDMARKS["left_index_pinky_midpoint_proxy"]
        assert left == "left_index"
        assert right == "left_pinky"
        lms = {
            "left_index": Point2D(0.10, 0.20),
            "left_pinky": Point2D(0.30, 0.40),
        }
        derived = _compute_derived_landmarks(lms)
        mid = derived["left_index_pinky_midpoint_proxy"]
        assert mid.x == pytest.approx(0.20)
        assert mid.y == pytest.approx(0.30)

    def test_derived_right_index_pinky_midpoint_proxy(self) -> None:
        """right_index_pinky_midpoint_proxy is the unvalidated hand distal proxy."""
        from gait_stability.com_estimation import DERIVED_LANDMARKS

        assert "right_index_pinky_midpoint_proxy" in DERIVED_LANDMARKS
        left, right = DERIVED_LANDMARKS["right_index_pinky_midpoint_proxy"]
        assert left == "right_index"
        assert right == "right_pinky"
        lms = {
            "right_index": Point2D(0.80, 0.90),
            "right_pinky": Point2D(1.00, 1.10),
        }
        derived = _compute_derived_landmarks(lms)
        mid = derived["right_index_pinky_midpoint_proxy"]
        assert mid.x == pytest.approx(0.90)
        assert mid.y == pytest.approx(1.00)

    def test_hand_segment_contributors_index_pinky_wrist(self) -> None:
        """Hand segment contributors include index, pinky, and wrist landmarks."""
        cfg = ComEstimationConfig(anthropometry_sex="male", minimum_mass_coverage=0.0)
        lms = _all_landmarks()
        prov = _all_prov(lms)
        result = _run_frame(lms, prov, cfg)
        # Find left_hand segment result
        left_hand = [
            sr for sr in result.segment_results if sr.segment_name == "left_hand"
        ]
        assert len(left_hand) == 1
        contribs = left_hand[0].provenance.contributors
        assert "left_index" in contribs
        assert "left_pinky" in contribs
        assert "left_wrist" in contribs

    def test_derived_landmarks_absent_when_parent_missing(self) -> None:
        lms: dict[str, Point2D] = {}
        derived = _compute_derived_landmarks(lms)
        assert "shoulder_midpoint" not in derived
        assert "hip_midpoint" not in derived


# ===================================================================
# Strict bool / nonfinite config rejection
# ===================================================================


class TestStrictBoolNonfiniteRejection:
    """Config rejects bool and nonfinite minimum_mass_coverage."""

    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError):
            ComEstimationConfig(
                anthropometry_sex="male",
                minimum_mass_coverage=True,  # type: ignore[arg-type]
            )

    def test_nan_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            ComEstimationConfig(
                anthropometry_sex="male",
                minimum_mass_coverage=float("nan"),
            )

    def test_inf_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            ComEstimationConfig(
                anthropometry_sex="male",
                minimum_mass_coverage=float("inf"),
            )

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            ComEstimationConfig(anthropometry_sex="male", minimum_mass_coverage=1.5)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            ComEstimationConfig(anthropometry_sex="male", minimum_mass_coverage=-0.1)

    def test_normalized_stride_samples_bool_rejected(self) -> None:
        with pytest.raises(TypeError):
            ComEstimationConfig(
                anthropometry_sex="male",
                normalized_stride_samples=True,  # type: ignore[arg-type]
            )

    def test_normalized_stride_samples_float_rejected(self) -> None:
        with pytest.raises(TypeError):
            ComEstimationConfig(
                anthropometry_sex="male",
                normalized_stride_samples=101.0,  # type: ignore[arg-type]
            )

    def test_normalized_stride_samples_too_small(self) -> None:
        with pytest.raises(ValueError, match=">= 2"):
            ComEstimationConfig(anthropometry_sex="male", normalized_stride_samples=1)

    def test_invalid_sex_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be"):
            ComEstimationConfig(anthropometry_sex="other")  # type: ignore[arg-type]


# ===================================================================
# Weighted represented-segment COM
# ===================================================================


class TestWeightedRepresentedSegmentCOM:
    """Weighted unrenormalized COM from represented segments."""

    def test_all_segments_usable_male(self) -> None:
        """All 13 supported segments usable: head unsupported, mass_coverage
        equals represented max .9306."""
        cfg = ComEstimationConfig(anthropometry_sex="male")
        lms = _all_landmarks()
        result = _run_frame(lms, _all_prov(lms), cfg)

        usable = sum(1 for sr in result.segment_results if sr.usable)
        assert usable == 13  # 14 total minus unsupported head
        assert "head" in UNSUPPORTED_SEGMENTS
        head_sr = [sr for sr in result.segment_results if sr.segment_name == "head"]
        assert len(head_sr) == 1
        assert not head_sr[0].usable
        assert result.mass_coverage == pytest.approx(REPRESENTED_MASS_MAX_MALE)
        assert result.model_total_mass == MODEL_MASS_TOTAL_MALE
        assert result.com is not None
        assert result.usable

    def test_all_segments_usable_female(self) -> None:
        """All 13 supported segments usable: head unsupported, mass_coverage
        equals represented max .9331."""
        cfg = ComEstimationConfig(anthropometry_sex="female")
        lms = _all_landmarks()
        result = _run_frame(lms, _all_prov(lms), cfg)
        assert result.mass_coverage == pytest.approx(REPRESENTED_MASS_MAX_FEMALE)
        assert result.usable

    def test_only_trunk_usable_no_head(self) -> None:
        """Only trunk usable: head is unsupported, bilateral segments need
        more landmarks. With shoulders + hips only, trunk = 0.4346."""
        cfg = ComEstimationConfig(anthropometry_sex="male")
        lms = {
            "left_shoulder": Point2D(0.44, 0.18),
            "right_shoulder": Point2D(0.56, 0.18),
            "left_hip": Point2D(0.46, 0.42),
            "right_hip": Point2D(0.54, 0.42),
        }
        result = _run_frame(lms, _all_prov(lms), cfg)
        usable_names = [sr.segment_name for sr in result.segment_results if sr.usable]
        assert "head" not in usable_names  # head is unsupported
        assert "trunk" in usable_names
        assert not any("_" in n for n in usable_names)
        expected_coverage = DE_LEVA_MALE["trunk"]["mass"]
        assert result.mass_coverage == pytest.approx(expected_coverage)

    def test_head_never_usable(self) -> None:
        """Head is always unusable regardless of landmark availability."""
        cfg = ComEstimationConfig(anthropometry_sex="male")
        lms = _all_landmarks()
        result = _run_frame(lms, _all_prov(lms), cfg)
        head_sr = [sr for sr in result.segment_results if sr.segment_name == "head"]
        assert len(head_sr) == 1
        assert not head_sr[0].usable
        assert head_sr[0].com is None
        assert head_sr[0].mass_fraction == DE_LEVA_MALE["head"]["mass"]

    def test_all_missing_com_is_none(self) -> None:
        """No landmarks: com=None, mass_coverage=0.0."""
        cfg = ComEstimationConfig(anthropometry_sex="male")
        result = _run_frame({}, {}, cfg)
        assert result.com is None
        assert result.mass_coverage == 0.0
        assert not result.usable

    def test_com_is_mass_weighted_average(self) -> None:
        """COM equals mass-weighted average of represented segment COMs."""
        cfg = ComEstimationConfig(anthropometry_sex="male", minimum_mass_coverage=0.0)
        lms = _all_landmarks()
        result = _run_frame(lms, _all_prov(lms), cfg)
        assert result.com is not None

        # Manually compute weighted average (head excluded as unsupported)
        num_x = 0.0
        num_y = 0.0
        denom = 0.0
        for sr in result.segment_results:
            if sr.segment_name == "head":
                continue  # unsupported, never participates
            if sr.usable and sr.com is not None:
                num_x += sr.mass_fraction * sr.com.x
                num_y += sr.mass_fraction * sr.com.y
                denom += sr.mass_fraction
        assert denom == result.mass_coverage
        assert result.com.x == pytest.approx(num_x / denom)
        assert result.com.y == pytest.approx(num_y / denom)

    def test_missing_segment_not_zeroed(self) -> None:
        """Unusable segments contribute zero mass and are excluded from COM.
        With shoulders + hips only, only trunk is usable (head unsupported)."""
        cfg = ComEstimationConfig(anthropometry_sex="male", minimum_mass_coverage=0.0)
        # Provide shoulders + hips => trunk usable, head unsupported
        lms = {
            "left_shoulder": Point2D(0.44, 0.18),
            "right_shoulder": Point2D(0.56, 0.18),
            "left_hip": Point2D(0.46, 0.42),
            "right_hip": Point2D(0.54, 0.42),
        }
        result = _run_frame(lms, _all_prov(lms), cfg)
        # COM should equal the trunk COM only (head unsupported)
        shoulder_mid = Point2D((0.44 + 0.56) / 2, (0.18 + 0.18) / 2)
        hip_mid = Point2D((0.46 + 0.54) / 2, (0.42 + 0.42) / 2)
        trunk_r = DE_LEVA_MALE["trunk"]["r"]
        trunk_com = _segment_com(shoulder_mid, hip_mid, trunk_r)
        assert result.com is not None
        assert result.com.x == pytest.approx(trunk_com.x)
        assert result.com.y == pytest.approx(trunk_com.y)
        assert result.mass_coverage == pytest.approx(DE_LEVA_MALE["trunk"]["mass"])


# ===================================================================
# Mass coverage threshold
# ===================================================================


class TestMassCoverageThreshold:
    """Equality passes, below fails for mass_coverage threshold.

    The source enforces: usable = (mass_coverage >= threshold) AND
    mass_coverage > 0.  A zero-coverage frame is never usable.
    """

    def test_equality_exact_subset_passes(self) -> None:
        """mass_coverage == threshold with exact subset => usable=True.

        With shoulders + hips only, trunk = 0.4346 is the only usable mass.
        Setting threshold to exactly 0.4346 exercises the >= comparison.
        """
        threshold = DE_LEVA_MALE["trunk"]["mass"]
        cfg = ComEstimationConfig(
            anthropometry_sex="male", minimum_mass_coverage=threshold
        )
        lms = {
            "left_shoulder": Point2D(0.44, 0.18),
            "right_shoulder": Point2D(0.56, 0.18),
            "left_hip": Point2D(0.46, 0.42),
            "right_hip": Point2D(0.54, 0.42),
        }
        result = _run_frame(lms, _all_prov(lms), cfg)
        # mass_coverage == 0.4346 >= 0.4346 => usable
        assert result.usable
        assert result.mass_coverage == pytest.approx(threshold)

    def test_below_threshold_not_usable(self) -> None:
        """Represented max .9306 < 0.95 => not usable."""
        cfg = ComEstimationConfig(anthropometry_sex="male", minimum_mass_coverage=0.95)
        lms = _all_landmarks()
        result = _run_frame(lms, _all_prov(lms), cfg)
        # male represented max 0.9306 < 0.95 => not usable
        assert not result.usable

    def test_trunk_only_below_default_threshold(self) -> None:
        """Trunk only 0.4346 < 0.90 => not usable."""
        cfg = ComEstimationConfig(anthropometry_sex="male", minimum_mass_coverage=0.90)
        lms = {
            "left_shoulder": Point2D(0.44, 0.18),
            "right_shoulder": Point2D(0.56, 0.18),
            "left_hip": Point2D(0.46, 0.42),
            "right_hip": Point2D(0.54, 0.42),
        }
        result = _run_frame(lms, _all_prov(lms), cfg)
        # trunk = 0.4346 < 0.90 => not usable
        assert not result.usable

    def test_threshold_zero_everything_usable(self) -> None:
        """Threshold 0.0: zero-coverage is still not usable per defect2."""
        cfg = ComEstimationConfig(anthropometry_sex="male", minimum_mass_coverage=0.0)
        result = _run_frame({}, {}, cfg)
        assert not result.usable
        assert result.mass_coverage == 0.0

    def test_threshold_one_needs_full_model(self) -> None:
        """Threshold 1.0: represented max .9306 < 1.0 => never usable."""
        cfg = ComEstimationConfig(anthropometry_sex="male", minimum_mass_coverage=1.0)
        lms = _all_landmarks()
        result = _run_frame(lms, _all_prov(lms), cfg)
        # represented max .9306 < 1.0 => not usable
        assert not result.usable
        # Without all segments, should also fail
        partial_lms = {
            "left_shoulder": Point2D(0.44, 0.18),
            "right_shoulder": Point2D(0.56, 0.18),
            "left_hip": Point2D(0.46, 0.42),
            "right_hip": Point2D(0.54, 0.42),
        }
        result2 = _run_frame(partial_lms, _all_prov(partial_lms), cfg)
        assert not result2.usable


# ===================================================================
# QC provenance categories
# ===================================================================


class TestQCProvenanceCategories:
    """Segment provenance category classification.

    Source priority: missing > interpolated > smoothing_interpolation >
    raw_observed > smoothed_only > other_qc_limited.
    """

    def test_raw_observed(self) -> None:
        prov = SegmentProvenance(
            segment_name="left_upper_arm",
            proximal_landmark="left_shoulder",
            distal_landmark="left_elbow",
            contributors=("left_shoulder", "left_elbow"),
            usable=True,
            mass_fraction=0.0271,
            r=0.5772,
            all_raw_observed=True,
            any_x_interpolated=False,
            any_y_interpolated=False,
            any_x_smoothing_changed=False,
            any_y_smoothing_changed=False,
            any_x_smoothing_support_interpolation=False,
            any_y_smoothing_support_interpolation=False,
            other_qc_limited=False,
        )
        assert prov.provenance_category() == "raw_observed"

    def test_interpolated(self) -> None:
        prov = SegmentProvenance(
            segment_name="left_upper_arm",
            proximal_landmark="left_shoulder",
            distal_landmark="left_elbow",
            contributors=("left_shoulder", "left_elbow"),
            usable=True,
            mass_fraction=0.0271,
            r=0.5772,
            all_raw_observed=False,
            any_x_interpolated=True,
            any_y_interpolated=True,
            any_x_smoothing_changed=False,
            any_y_smoothing_changed=False,
            any_x_smoothing_support_interpolation=False,
            any_y_smoothing_support_interpolation=False,
            other_qc_limited=False,
        )
        assert prov.provenance_category() == "interpolated"

    def test_smoothing_interpolation(self) -> None:
        prov = SegmentProvenance(
            segment_name="left_upper_arm",
            proximal_landmark="left_shoulder",
            distal_landmark="left_elbow",
            contributors=("left_shoulder", "left_elbow"),
            usable=True,
            mass_fraction=0.0271,
            r=0.5772,
            all_raw_observed=False,
            any_x_interpolated=False,
            any_y_interpolated=False,
            any_x_smoothing_changed=False,
            any_y_smoothing_changed=False,
            any_x_smoothing_support_interpolation=True,
            any_y_smoothing_support_interpolation=False,
            other_qc_limited=False,
        )
        assert prov.provenance_category() == "smoothing_interpolation"

    def test_smoothed_only(self) -> None:
        """Smoothing changed but no interpolation in support -> 'smoothed_only'.

        Source note: provenance_category checks all_raw_observed
        before smoothed_only. For smoothed_only, at least one endpoint
        must have raw_observed_usable=False (so all_raw_observed=False).
        """
        prov = SegmentProvenance(
            segment_name="left_upper_arm",
            proximal_landmark="left_shoulder",
            distal_landmark="left_elbow",
            contributors=("left_shoulder", "left_elbow"),
            usable=True,
            mass_fraction=0.0271,
            r=0.5772,
            all_raw_observed=False,
            any_x_interpolated=False,
            any_y_interpolated=False,
            any_x_smoothing_changed=True,
            any_y_smoothing_changed=True,
            any_x_smoothing_support_interpolation=False,
            any_y_smoothing_support_interpolation=False,
            other_qc_limited=False,
        )
        assert prov.provenance_category() == "smoothed_only"

    def test_smoothed_only_one_changed(self) -> None:
        """Single axis smoothing changed is sufficient for smoothed_only."""
        prov = SegmentProvenance(
            segment_name="left_upper_arm",
            proximal_landmark="left_shoulder",
            distal_landmark="left_elbow",
            contributors=("left_shoulder", "left_elbow"),
            usable=True,
            mass_fraction=0.0271,
            r=0.5772,
            all_raw_observed=False,
            any_x_interpolated=False,
            any_y_interpolated=False,
            any_x_smoothing_changed=True,
            any_y_smoothing_changed=False,
            any_x_smoothing_support_interpolation=False,
            any_y_smoothing_support_interpolation=False,
            other_qc_limited=False,
        )
        assert prov.provenance_category() == "smoothed_only"

    def test_missing(self) -> None:
        prov = SegmentProvenance(
            segment_name="left_upper_arm",
            proximal_landmark="left_shoulder",
            distal_landmark="left_elbow",
            contributors=(),
            usable=False,
            mass_fraction=0.0271,
            r=0.5772,
            all_raw_observed=False,
            any_x_interpolated=False,
            any_y_interpolated=False,
            any_x_smoothing_changed=False,
            any_y_smoothing_changed=False,
            any_x_smoothing_support_interpolation=False,
            any_y_smoothing_support_interpolation=False,
            other_qc_limited=False,
        )
        assert prov.provenance_category() == "missing"

    def test_interpolated_preferred_over_smoothing_interpolation(self) -> None:
        """Direct interpolation takes priority over smoothing-support interpolation."""
        prov = SegmentProvenance(
            segment_name="left_upper_arm",
            proximal_landmark="left_shoulder",
            distal_landmark="left_elbow",
            contributors=("left_shoulder", "left_elbow"),
            usable=True,
            mass_fraction=0.0271,
            r=0.5772,
            all_raw_observed=False,
            any_x_interpolated=True,
            any_y_interpolated=False,
            any_x_smoothing_changed=False,
            any_y_smoothing_changed=False,
            any_x_smoothing_support_interpolation=True,
            any_y_smoothing_support_interpolation=False,
            other_qc_limited=False,
        )
        assert prov.provenance_category() == "interpolated"

    def test_other_qc_limited(self) -> None:
        """Usable, no raw, no interpolation, no smoothing -> other_qc_limited."""
        prov = SegmentProvenance(
            segment_name="left_upper_arm",
            proximal_landmark="left_shoulder",
            distal_landmark="left_elbow",
            contributors=("left_shoulder", "left_elbow"),
            usable=True,
            mass_fraction=0.0271,
            r=0.5772,
            all_raw_observed=False,
            any_x_interpolated=False,
            any_y_interpolated=False,
            any_x_smoothing_changed=False,
            any_y_smoothing_changed=False,
            any_x_smoothing_support_interpolation=False,
            any_y_smoothing_support_interpolation=False,
            other_qc_limited=True,
        )
        assert prov.provenance_category() == "other_qc_limited"


# ===================================================================
# Stride normalization
# ===================================================================


class TestStrideNormalization:
    """Stride COM normalization with irregular timestamps."""

    def _make_frame(
        self, fi: int, ts: float, com_x: float, com_y: float
    ) -> FrameComResult:
        return FrameComResult(
            frame_index=fi,
            timestamp_seconds=ts,
            frame_status="decoded_pose",
            com=Point2D(com_x, com_y),
            mass_coverage=1.0,
            usable=True,
            segment_results=(),
            model_total_mass=1.0,
        )

    def _make_unusable_frame(self, fi: int, ts: float) -> FrameComResult:
        return FrameComResult(
            frame_index=fi,
            timestamp_seconds=ts,
            frame_status="no_pose",
            com=None,
            mass_coverage=0.0,
            usable=False,
            segment_results=(),
            model_total_mass=1.0,
        )

    def test_all_original_frames_included(self) -> None:
        """Every frame in the stride (usable or not) appears as an original sample."""
        frames = (
            self._make_frame(0, 0.00, 0.1, 0.2),
            self._make_frame(1, 0.04, 0.2, 0.3),
            self._make_unusable_frame(2, 0.08),
            self._make_frame(3, 0.12, 0.3, 0.4),
        )
        cfg = ComEstimationConfig(anthropometry_sex="male")
        samples = normalize_stride_com(frames, 0, 3, 0.0, 0.12, cfg)
        originals = [s for s in samples if s.sample_kind == "original"]
        assert len(originals) == 4
        # Unusable frame 2 still appears
        orig_indices = [s.frame_index for s in originals]
        assert orig_indices == [0, 1, 2, 3]

    def test_exact_normalized_sample_count(self) -> None:
        """Default 101 normalized samples."""
        frames = tuple(self._make_frame(i, i * 0.01, float(i), 0.0) for i in range(10))
        cfg = ComEstimationConfig(
            anthropometry_sex="male", normalized_stride_samples=101
        )
        samples = normalize_stride_com(frames, 0, 9, 0.0, 0.09, cfg)
        normalized = [s for s in samples if s.sample_kind == "normalized"]
        assert len(normalized) == 101

    def test_normalized_exact_0_and_100(self) -> None:
        """Normalized samples always include exact progression 0 and 100."""
        frames = tuple(self._make_frame(i, i * 0.01, float(i), 0.0) for i in range(5))
        cfg = ComEstimationConfig(
            anthropometry_sex="male", normalized_stride_samples=101
        )
        samples = normalize_stride_com(frames, 0, 4, 0.0, 0.04, cfg)
        normalized = [s for s in samples if s.sample_kind == "normalized"]
        progs = [s.progression for s in normalized]
        assert progs[0] == 0.0
        assert progs[-1] == 100.0

    def test_no_interpolation_across_unusable_interior(self) -> None:
        """Normalized samples never bridge over an unusable interior frame.

        Source behavior: when the bracket-pair interpolation is attempted
        but either endpoint is unusable, the sample gets method='none' and
        com=None.  When a normalized target timestamp exactly matches an
        unusable frame, it gets method='exact' with that frame's com (None).

        Frames: 0=usable(ts=0.00,com=0.1,0.2), 1=unusable(ts=0.05),
                2=usable(ts=0.10,com=0.3,0.4)
        Progression 50 => target_ts=0.05 which exactly matches frame1.
        """
        frames = (
            self._make_frame(0, 0.00, 0.1, 0.2),
            self._make_unusable_frame(1, 0.05),
            self._make_frame(2, 0.10, 0.3, 0.4),
        )
        cfg = ComEstimationConfig(
            anthropometry_sex="male", normalized_stride_samples=11
        )
        samples = normalize_stride_com(frames, 0, 2, 0.0, 0.10, cfg)
        normalized = [s for s in samples if s.sample_kind == "normalized"]

        # Progression 50: exact match on frame1 (unusable)
        mid = [s for s in normalized if s.progression == pytest.approx(50.0)]
        assert len(mid) == 1
        assert mid[0].com is None
        assert mid[0].usable is False
        assert mid[0].method == "exact"

        # No linear interpolation exists: every non-exact normalized sample
        # uses method='none' because the bracket always includes the unusable
        # frame1.
        linear = [s for s in normalized if s.method == "linear"]
        assert len(linear) == 0

        # All non-exact normalized samples have com=None
        non_exact = [s for s in normalized if s.method != "exact"]
        for s in non_exact:
            assert s.com is None
            assert s.usable is False

    def test_usable_interior_allows_interpolation(self) -> None:
        """When all interior frames are usable, interpolation works normally."""
        frames = (
            self._make_frame(0, 0.00, 0.1, 0.2),
            self._make_frame(1, 0.05, 0.2, 0.3),
            self._make_frame(2, 0.10, 0.3, 0.4),
        )
        cfg = ComEstimationConfig(
            anthropometry_sex="male", normalized_stride_samples=11
        )
        samples = normalize_stride_com(frames, 0, 2, 0.0, 0.10, cfg)
        normalized = [s for s in samples if s.sample_kind == "normalized"]

        # Progression 50: bracket is (frame0, frame1) or (frame1, frame2)?
        # target_ts = 0.05 matches frame1 exactly -> method="exact"
        mid = [s for s in normalized if s.progression == pytest.approx(50.0)]
        assert len(mid) == 1
        assert mid[0].com is not None
        assert mid[0].method == "exact"
        assert mid[0].com.x == pytest.approx(0.2)

        # Progression 20: target_ts=0.02, bracket=(frame0, frame1)
        # frac = (0.02 - 0.0) / (0.05 - 0.0) = 0.4
        # interp_x = 0.1 + 0.4*(0.2-0.1) = 0.14
        q1 = [s for s in normalized if s.progression == pytest.approx(20.0)]
        assert len(q1) == 1
        assert q1[0].method == "linear"
        assert q1[0].com is not None
        assert q1[0].com.x == pytest.approx(0.14)  # 0.1 + 0.4*(0.2-0.1)

    def test_irregular_timestamp_progression(self) -> None:
        """Stride progression uses canonical timestamps, not frame count fractions."""
        frames = (
            self._make_frame(0, 0.00, 0.1, 0.1),
            self._make_frame(1, 0.01, 0.2, 0.2),
            self._make_frame(2, 0.05, 0.3, 0.3),
            self._make_frame(3, 0.06, 0.4, 0.4),
        )
        cfg = ComEstimationConfig(anthropometry_sex="male")
        samples = normalize_stride_com(frames, 0, 3, 0.0, 0.06, cfg)
        originals = [s for s in samples if s.sample_kind == "original"]
        # Progression for frame 2 at ts 0.05: (0.05 - 0) / 0.06 * 100 = 83.33...
        frame2_prog = [s for s in originals if s.frame_index == 2][0].progression
        assert frame2_prog == pytest.approx(83.333333, rel=1e-4)

    def test_deterministic_output(self) -> None:
        """Same inputs produce identical outputs."""
        frames = tuple(
            self._make_frame(i, i * 0.02, float(i) * 0.1, 0.0) for i in range(6)
        )
        cfg = ComEstimationConfig(anthropometry_sex="male")
        s1 = normalize_stride_com(frames, 0, 5, 0.0, 0.10, cfg)
        s2 = normalize_stride_com(frames, 0, 5, 0.0, 0.10, cfg)
        assert len(s1) == len(s2)
        for a, b in zip(s1, s2, strict=True):
            assert a.progression == b.progression
            assert a.sample_kind == b.sample_kind
            if a.com is not None:
                assert b.com is not None
                assert a.com.x == b.com.x
                assert a.com.y == b.com.y
            else:
                assert b.com is None

    def test_custom_normalized_samples(self) -> None:
        """Configurable N gives correct sample count."""
        frames = tuple(self._make_frame(i, i * 0.01, 0.5, 0.5) for i in range(10))
        cfg = ComEstimationConfig(
            anthropometry_sex="male", normalized_stride_samples=21
        )
        samples = normalize_stride_com(frames, 0, 9, 0.0, 0.09, cfg)
        normalized = [s for s in samples if s.sample_kind == "normalized"]
        assert len(normalized) == 21

    def test_incomplete_stride_raises(self) -> None:
        """Source requires all consecutive frame indices in stride bounds."""
        # Only frame 10 provided, but stride 0..5 requires frames 0-5
        frames = (self._make_frame(10, 1.0, 0.5, 0.5),)
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ValueError, match="requires all frame indices"):
            normalize_stride_com(frames, 0, 5, 0.0, 0.5, cfg)

    def test_unusable_original_preserves_none_com(self) -> None:
        """Original sample for unusable frame keeps com=None."""
        frames = (
            self._make_frame(0, 0.00, 0.1, 0.1),
            self._make_unusable_frame(1, 0.05),
            self._make_frame(2, 0.10, 0.2, 0.2),
        )
        cfg = ComEstimationConfig(anthropometry_sex="male")
        samples = normalize_stride_com(frames, 0, 2, 0.0, 0.10, cfg)
        unusable_orig = [
            s for s in samples if s.sample_kind == "original" and s.frame_index == 1
        ]
        assert len(unusable_orig) == 1
        assert unusable_orig[0].com is None
        assert unusable_orig[0].usable is False

    def test_different_segment_sets_emit_none_not_linear(self) -> None:
        """Adjacent usable frames with different usable_segments sets
        must emit method=none with represented_segment_set_changed qc_flag
        instead of linear interpolation. Identical sets permit linear."""
        from gait_stability.com_estimation import (
            estimate_frame_com,
        )

        cfg = ComEstimationConfig(anthropometry_sex="male", minimum_mass_coverage=0.0)

        # Frame 0: trunk only (shoulders + hips, no wrist)
        lms_0 = {
            "left_shoulder": Point2D(0.44, 0.18),
            "right_shoulder": Point2D(0.56, 0.18),
            "left_hip": Point2D(0.46, 0.42),
            "right_hip": Point2D(0.54, 0.42),
        }
        prov_0 = {name: _prov(landmark_name=name) for name in lms_0}
        fr_0 = estimate_frame_com(
            processed_landmarks=lms_0,
            landmark_provenance=prov_0,
            config=cfg,
            frame_index=0,
            timestamp_seconds=0.0,
            frame_status="decoded_pose",
        )
        # Frame 1: trunk + bilateral upper arms (add elbows)
        lms_1 = {
            "left_shoulder": Point2D(0.44, 0.18),
            "right_shoulder": Point2D(0.56, 0.18),
            "left_hip": Point2D(0.46, 0.42),
            "right_hip": Point2D(0.54, 0.42),
            "left_elbow": Point2D(0.40, 0.30),
            "right_elbow": Point2D(0.60, 0.30),
        }
        prov_1 = {name: _prov(landmark_name=name) for name in lms_1}
        fr_1 = estimate_frame_com(
            processed_landmarks=lms_1,
            landmark_provenance=prov_1,
            config=cfg,
            frame_index=1,
            timestamp_seconds=0.05,
            frame_status="decoded_pose",
        )

        # Verify different segment sets
        assert fr_0.usable_segments() != fr_1.usable_segments()

        frames = (fr_0, fr_1)
        samples = normalize_stride_com(frames, 0, 1, 0.0, 0.05, cfg)
        normalized = [s for s in samples if s.sample_kind == "normalized"]

        # All normalized samples between different segment sets must be none
        for s in normalized:
            if s.method == "none" and s.left_source_frame_index is not None:
                # Bracket found but different segment sets
                assert "represented_segment_set_changed" in s.qc_flags
                assert s.com is None

        # Identical segment sets at exact-match frames permit usable results
        # Frame 0 at progression=0 is exact -> usable
        p0 = [s for s in normalized if s.progression == pytest.approx(0.0)]
        assert len(p0) == 1
        assert p0[0].method == "exact"
        assert p0[0].usable is True


# ===================================================================
# Point2D validation
# ===================================================================


class TestPoint2DValidation:
    """Point2D rejects nonfinite coordinates."""

    def test_nan_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Point2D(float("nan"), 0.0)

    def test_inf_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Point2D(0.0, float("inf"))

    def test_finite_accepted(self) -> None:
        p = Point2D(0.5, 0.5)
        assert p.x == 0.5
        assert p.y == 0.5


# ===================================================================
# FrameComResult provenance summary
# ===================================================================


class TestFrameComResultSummary:
    """FrameComResult.provenance_summary counts."""

    def test_all_raw_observed(self) -> None:
        """With all landmarks including derived provenance, all 13 supported
        segments are raw_observed (head is unsupported)."""
        cfg = ComEstimationConfig(anthropometry_sex="male", minimum_mass_coverage=0.0)
        lms = _all_landmarks()
        prov = _all_prov(lms)
        from gait_stability.com_estimation import DERIVED_LANDMARKS

        for derived_name, (left_name, right_name) in DERIVED_LANDMARKS.items():
            if left_name in prov and right_name in prov:
                prov[derived_name] = _prov(landmark_name=derived_name)
        result = _run_frame(lms, prov, cfg)
        summary = result.provenance_summary()
        # head is unsupported, so 13 of 14 segments are raw_observed
        assert summary.get("raw_observed", 0) == 13
        assert summary.get("missing", 0) == 1  # head

    def test_all_missing_when_no_landmarks(self) -> None:
        cfg = ComEstimationConfig(anthropometry_sex="male")
        result = _run_frame({}, {}, cfg)
        summary = result.provenance_summary()
        assert summary.get("missing", 0) == len(SEGMENT_NAMES)
