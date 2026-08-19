"""COM pipeline integration tests using synthetic canonical artifact fixtures.

Builds the six required inputs from scratch (processed_landmarks.csv,
preprocessing_metadata.json, pose_frames.csv, reviewed_gait_events.csv,
reviewed_strides.csv, review_resolution_metadata.json) with real SHA-256
hashes, then runs estimate_com() end-to-end.

Verifies:
- Pipeline success and output schema
- Metadata fields (citation, coefficients, coverage, schemas, limitations,
  inputs/outputs, scientific_unresolved, reviewed-not-ground-truth)
- Stride/com provenance assertions
- Reviewed-only stride segmentation
- Schema / timestamp / status / hash / algorithm / blocking / endpoints
  malformed cases
- True prepublication input mutation via monkeypatch
- True transactional rollback via monkeypatch on Path.replace
- Regression: boundary change alters stride output but not frame COM
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from gait_stability.com_estimation import (
    MODEL_MASS_TOTAL_MALE,
    REPRESENTED_MASS_MAX_MALE,
    SEGMENT_NAMES,
    ComEstimationConfig,
)
from gait_stability.com_pipeline import (
    COM_ALGORITHM_VERSION,
    COM_OUTPUT_ARTIFACT_NAMES,
    COM_PROXY_FIELD_NAMES,
    COM_SCHEMA_VERSION,
    REVIEWED_STRIDE_FIELDS,
    STRIDE_COM_FIELD_NAMES,
    ComArtifactValidationError,
    ComPipelineError,
    estimate_com,
)
from gait_stability.pose_contracts import MEDIAPIPE_LANDMARK_NAMES
from gait_stability.pose_pipeline import FRAME_FIELDS
from gait_stability.pose_preprocessing import (
    PREPROCESSING_ALGORITHM_VERSION,
    PREPROCESSING_SCHEMA_VERSION,
    PROCESSED_FIELDS,
)
from gait_stability.review_resolution import (
    REVIEW_RESOLUTION_ALGORITHM_VERSION,
    REVIEW_RESOLUTION_SCHEMA_VERSION,
    REVIEWED_GAIT_EVENT_FIELDS,
)
from gait_stability.video_ingestion import ArtifactPublishError

# ---------------------------------------------------------------------------
# Canonical fixture builder
# ---------------------------------------------------------------------------

# All 33 MEDIAPIPE_LANDMARK_NAMES with deterministic coordinates.
# Landmark IDs are the canonical indices 0..32.
# The 19 COM contributors use coordinates that yield valid segment COMs.
# The 14 non-contributor landmarks use simple synthetic positions.

_ALL_LANDMARK_COORDS: dict[str, tuple[float, float]] = {
    "nose": (0.50, 0.10),
    "left_eye_inner": (0.47, 0.08),
    "left_eye": (0.46, 0.07),
    "left_eye_outer": (0.45, 0.06),
    "right_eye_inner": (0.53, 0.08),
    "right_eye": (0.54, 0.07),
    "right_eye_outer": (0.55, 0.06),
    "left_ear": (0.43, 0.09),
    "right_ear": (0.57, 0.09),
    "mouth_left": (0.47, 0.13),
    "mouth_right": (0.53, 0.13),
    "left_shoulder": (0.44, 0.18),
    "right_shoulder": (0.56, 0.18),
    "left_elbow": (0.40, 0.30),
    "right_elbow": (0.60, 0.30),
    "left_wrist": (0.38, 0.42),
    "right_wrist": (0.62, 0.42),
    "left_pinky": (0.39, 0.46),
    "right_pinky": (0.61, 0.46),
    "left_index": (0.37, 0.44),
    "right_index": (0.63, 0.44),
    "left_thumb": (0.36, 0.43),
    "right_thumb": (0.64, 0.43),
    "left_hip": (0.46, 0.42),
    "right_hip": (0.54, 0.42),
    "left_knee": (0.44, 0.62),
    "right_knee": (0.56, 0.62),
    "left_ankle": (0.43, 0.82),
    "right_ankle": (0.57, 0.82),
    "left_heel": (0.42, 0.88),
    "right_heel": (0.58, 0.88),
    "left_foot_index": (0.40, 0.92),
    "right_foot_index": (0.60, 0.92),
}

# Canonical landmark IDs from MEDIAPIPE_LANDMARK_NAMES (0..32)
_LANDMARK_IDS: dict[str, int] = {
    name: idx for idx, name in enumerate(MEDIAPIPE_LANDMARK_NAMES)
}

# Verify our fixture covers all 33 landmarks
assert len(_ALL_LANDMARK_COORDS) == len(MEDIAPIPE_LANDMARK_NAMES)
assert set(_ALL_LANDMARK_COORDS.keys()) == set(MEDIAPIPE_LANDMARK_NAMES)

# Frames: 0, 1, 2 with timestamps 0.0, 0.033, 0.067
_FRAMES = [
    {
        "frame_index": "0",
        "nominal_timestamp_seconds": "0.0",
        "backend_timestamp_milliseconds": "100",
        "status": "decoded_pose",
        "landmark_count": str(len(_ALL_LANDMARK_COORDS)),
        "detail": "",
    },
    {
        "frame_index": "1",
        "nominal_timestamp_seconds": "0.033",
        "backend_timestamp_milliseconds": "133",
        "status": "decoded_pose",
        "landmark_count": str(len(_ALL_LANDMARK_COORDS)),
        "detail": "",
    },
    {
        "frame_index": "2",
        "nominal_timestamp_seconds": "0.067",
        "backend_timestamp_milliseconds": "167",
        "status": "decoded_pose",
        "landmark_count": str(len(_ALL_LANDMARK_COORDS)),
        "detail": "",
    },
]


def _processed_row(
    frame_index: int, ts: float, lm_name: str, lm_id: int, x: float, y: float
) -> dict[str, str]:
    """Build one row of processed_landmarks.csv matching PROCESSED_FIELDS."""
    return {
        "frame_index": str(frame_index),
        "nominal_timestamp_seconds": str(ts),
        "frame_status": "decoded_pose",
        "landmark_id": str(lm_id),
        "landmark_name": lm_name,
        "raw_row_present": "true",
        "raw_x_normalized": str(x),
        "raw_y_normalized": str(y),
        "raw_z_backend_relative": "0.0",
        "visibility": "0.9",
        "presence": "0.9",
        "confidence": "0.9",
        "x_observed_usable": "true",
        "y_observed_usable": "true",
        "observed_usable": "true",
        "rejected_low_confidence": "false",
        "missing_or_nonfinite_enabled_score": "false",
        "nonfinite_x_coordinate": "false",
        "nonfinite_y_coordinate": "false",
        "out_of_image_x": "false",
        "out_of_image_y": "false",
        "pre_smoothed_x_normalized": str(x),
        "pre_smoothed_y_normalized": str(y),
        "processed_x_normalized": str(x),
        "processed_y_normalized": str(y),
        "x_interpolated": "false",
        "y_interpolated": "false",
        "x_smoothing_changed": "false",
        "y_smoothing_changed": "false",
        "x_smoothing_support_contains_interpolation": "false",
        "y_smoothing_support_contains_interpolation": "false",
        "x_final_missing": "false",
        "y_final_missing": "false",
        "final_missing": "false",
    }


def _build_processed_landmarks_csv(path: Path) -> None:
    """Write processed_landmarks.csv for all 3 frames, all 33 landmarks."""
    fieldnames = list(PROCESSED_FIELDS)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame in _FRAMES:
            fi = int(frame["frame_index"])
            ts = float(frame["nominal_timestamp_seconds"])
            for lm_name, (x, y) in _ALL_LANDMARK_COORDS.items():
                writer.writerow(
                    _processed_row(fi, ts, lm_name, _LANDMARK_IDS[lm_name], x, y)
                )


def _build_pose_frames_csv(path: Path) -> None:
    """Write pose_frames.csv."""
    fieldnames = list(FRAME_FIELDS)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame in _FRAMES:
            writer.writerow(frame)


def _build_reviewed_gait_events_csv(path: Path) -> None:
    """Write reviewed_gait_events.csv with 2 events bounding a stride."""
    fieldnames = list(REVIEWED_GAIT_EVENT_FIELDS)
    events = [
        {
            "event_id": "RE001",
            "automatic_event_id": "E0001",
            "side": "left",
            "event_type": "candidate_initial_contact",
            "automatic_frame_index": "0",
            "automatic_timestamp_seconds": "0.0",
            "automatic_disposition": "accepted_unchanged",
            "automatic_quality": "high",
            "automatic_peak_value": "0.12",
            "automatic_prominence": "0.08",
            "automatic_rejection_reasons": "",
            "manual_event_review_status": "unreviewed",
            "stride_review_provenance": "",
            "reviewed_frame_index": "0",
            "reviewed_timestamp_seconds": "0.0",
            "reviewed_accepted": "true",
            "reviewed_rejected": "false",
            "reviewed_included_in_stride": "true",
            "reviewed_quality": "high",
            "resolution_disposition": "accepted_unchanged",
            "replaces_event_id": "",
            "replaced_by_event_id": "",
            "source": "automatic",
            "review_notes": "",
        },
        {
            "event_id": "RE002",
            "automatic_event_id": "E0002",
            "side": "left",
            "event_type": "candidate_initial_contact",
            "automatic_frame_index": "2",
            "automatic_timestamp_seconds": "0.067",
            "automatic_disposition": "accepted_unchanged",
            "automatic_quality": "high",
            "automatic_peak_value": "0.15",
            "automatic_prominence": "0.10",
            "automatic_rejection_reasons": "",
            "manual_event_review_status": "unreviewed",
            "stride_review_provenance": "",
            "reviewed_frame_index": "2",
            "reviewed_timestamp_seconds": "0.067",
            "reviewed_accepted": "true",
            "reviewed_rejected": "false",
            "reviewed_included_in_stride": "true",
            "reviewed_quality": "high",
            "resolution_disposition": "accepted_unchanged",
            "replaces_event_id": "",
            "replaced_by_event_id": "",
            "source": "automatic",
            "review_notes": "",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in events:
            writer.writerow(ev)


def _build_reviewed_strides_csv(path: Path) -> None:
    """Write reviewed_strides.csv with exact REVIEWED_STRIDE_FIELDS header."""
    from gait_stability.com_pipeline import REVIEWED_STRIDE_FIELDS

    fieldnames = list(REVIEWED_STRIDE_FIELDS)
    strides = [
        {
            "stride_id": "RS001",
            "side": "left",
            "start_event_id": "RE001",
            "end_event_id": "RE002",
            "start_frame": "0",
            "end_frame": "2",
            "start_timestamp_seconds": "0.0",
            "end_timestamp_seconds": "0.067",
            "duration_seconds": "0.067",
            "quality": "high",
            "contralateral_event_id": "",
            "contralateral_event_count": "0",
            "sequence_notes": "",
            "source": "automatic",
            "review_status": "accept",
            "automatic_stride_id": "S0001",
            "review_intent": "accept",
            "review_changes": "",
            "provenance_notes": "",
        }
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in strides:
            writer.writerow(s)


def _sha256_file_content(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _build_preprocessing_metadata(path: Path, pf_path: Path, pl_path: Path) -> None:
    """Write preprocessing_metadata.json with real hashes."""
    pf_hash = _sha256_file_content(pf_path)
    pl_hash = _sha256_file_content(pl_path)
    meta = {
        "schema_version": PREPROCESSING_SCHEMA_VERSION,
        "algorithm_version": PREPROCESSING_ALGORITHM_VERSION,
        "inputs": {
            "pose_frames.csv": {
                "path": str(pf_path),
                "sha256": pf_hash,
            }
        },
        "outputs": {
            "processed_landmarks.csv": {
                "path": str(pl_path),
                "sha256": pl_hash,
            }
        },
        "config": {"dummy": True},
        "inherited_provenance": {},
    }
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _build_review_metadata(
    path: Path,
    rge_path: Path,
    rs_path: Path,
    pm_path: Path,
    pf_path: Path,
    stride_count: int,
) -> None:
    """Write review_resolution_metadata.json with real hashes."""
    rge_hash = _sha256_file_content(rge_path)
    rs_hash = _sha256_file_content(rs_path)
    pm_hash = _sha256_file_content(pm_path)
    pf_hash = _sha256_file_content(pf_path)
    meta = {
        "schema_version": REVIEW_RESOLUTION_SCHEMA_VERSION,
        "algorithm_version": REVIEW_RESOLUTION_ALGORITHM_VERSION,
        "outputs": {
            "reviewed_gait_events.csv": {
                "path": str(rge_path),
                "sha256": rge_hash,
            },
            "reviewed_strides.csv": {
                "path": str(rs_path),
                "sha256": rs_hash,
            },
        },
        "inputs": {
            "preprocessing_metadata.json": {
                "path": str(pm_path),
                "sha256": pm_hash,
            }
        },
        "timestamp_source": {
            "path": str(pf_path),
            "sha256": pf_hash,
        },
        "blocking_unresolved": [],
        "scientific_unresolved": [
            "Not validated against force plates or clinical measurements"
        ],
        "counts": {
            "reviewed_strides": stride_count,
        },
    }
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _build_fixture(tmp: Path) -> Path:
    """Build the complete 6-input canonical artifact directory."""
    d = tmp / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    pf = d / "pose_frames.csv"
    _build_pose_frames_csv(pf)
    pl = d / "processed_landmarks.csv"
    _build_processed_landmarks_csv(pl)
    _build_preprocessing_metadata(d / "preprocessing_metadata.json", pf, pl)
    rge = d / "reviewed_gait_events.csv"
    _build_reviewed_gait_events_csv(rge)
    rs = d / "reviewed_strides.csv"
    _build_reviewed_strides_csv(rs)
    _build_review_metadata(
        d / "review_resolution_metadata.json",
        rge,
        rs,
        d / "preprocessing_metadata.json",
        pf,
        stride_count=1,
    )
    return d


# ---------------------------------------------------------------------------
# Tests: success path
# ---------------------------------------------------------------------------


class TestPipelineSuccess:
    """Run the full pipeline and verify outputs."""

    @pytest.fixture()
    def artifacts_dir(self, tmp_path: Path) -> Path:
        return _build_fixture(tmp_path)

    @pytest.fixture()
    def result(self, artifacts_dir: Path) -> Any:
        cfg = ComEstimationConfig(anthropometry_sex="male")
        return estimate_com(artifacts_dir, cfg)

    @pytest.fixture()
    def meta(self, result: Any, artifacts_dir: Path) -> Any:
        return json.loads(
            (artifacts_dir / "com_metadata.json").read_text(encoding="utf-8")
        )

    @pytest.fixture()
    def proxy_rows(self, result: Any, artifacts_dir: Path) -> list[dict[str, str]]:
        with (artifacts_dir / "com_proxy.csv").open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @pytest.fixture()
    def stride_rows(self, result: Any, artifacts_dir: Path) -> list[dict[str, str]]:
        with (artifacts_dir / "stride_com.csv").open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_all_output_files_exist(self, result: Any, artifacts_dir: Path) -> None:
        for name in COM_OUTPUT_ARTIFACT_NAMES:
            p = artifacts_dir / name
            assert p.exists(), f"Missing output: {name}"

    def test_com_proxy_schema_version(self, meta: Any) -> None:
        assert meta["schema_version"] == COM_SCHEMA_VERSION

    def test_algorithm_version(self, meta: Any) -> None:
        assert meta["algorithm_version"] == COM_ALGORITHM_VERSION

    def test_metadata_config(self, meta: Any) -> None:
        cfg = meta["config"]
        assert cfg["anthropometry_sex"] == "male"
        assert cfg["minimum_mass_coverage"] == 0.90
        assert cfg["normalized_stride_samples"] == 101

    def test_frame_counts(self, meta: Any) -> None:
        fc = meta["frame_counts"]
        assert fc["total"] == 3
        assert fc["usable"] == 3
        assert fc["unusable"] == 0

    def test_com_proxy_csv_header(self, result: Any, artifacts_dir: Path) -> None:
        with (artifacts_dir / "com_proxy.csv").open(encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert tuple(header) == COM_PROXY_FIELD_NAMES

    def test_stride_com_csv_header(self, result: Any, artifacts_dir: Path) -> None:
        with (artifacts_dir / "stride_com.csv").open(encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert tuple(header) == STRIDE_COM_FIELD_NAMES

    def test_com_proxy_row_count(self, proxy_rows: list[dict[str, str]]) -> None:
        assert len(proxy_rows) == 3

    def test_stride_com_has_original_and_normalized(
        self, stride_rows: list[dict[str, str]]
    ) -> None:
        kinds = {r["sample_kind"] for r in stride_rows}
        assert "original" in kinds
        assert "normalized" in kinds

    def test_stride_com_normalized_count(
        self, stride_rows: list[dict[str, str]]
    ) -> None:
        normalized = [r for r in stride_rows if r["sample_kind"] == "normalized"]
        assert len(normalized) == 101

    def test_com_proxy_usable_all_true(self, proxy_rows: list[dict[str, str]]) -> None:
        for row in proxy_rows:
            assert row["usable"] == "true"

    def test_com_proxy_nonempty_com(self, proxy_rows: list[dict[str, str]]) -> None:
        for row in proxy_rows:
            assert row["com_x"] != ""
            assert row["com_y"] != ""

    def test_metadata_inputs_hashes_nonempty(self, meta: Any) -> None:
        for name, info in meta["inputs"].items():
            assert "sha256" in info and info["sha256"], f"No hash for {name}"

    def test_metadata_has_limitations(self, meta: Any) -> None:
        assert isinstance(meta["limitations"], list)
        assert len(meta["limitations"]) > 0

    def test_metadata_has_algorithm_section(self, meta: Any) -> None:
        alg = meta["algorithm"]
        assert "de Leva" in alg["model"]
        assert alg["segments"] == len(SEGMENT_NAMES)

    def test_metadata_stride_statistics(self, meta: Any) -> None:
        ss = meta["stride_statistics"]
        assert len(ss) == 1
        assert ss[0]["stride_id"] == "RS001"
        assert ss[0]["side"] == "left"

    def test_stride_com_preserves_review_provenance(
        self, stride_rows: list[dict[str, str]]
    ) -> None:
        """stride_com.csv rows carry the four review-provenance extras."""
        for row in stride_rows:
            assert row["automatic_stride_id"] == "S0001"
            assert row["review_intent"] == "accept"
            assert row["review_changes"] == ""
            assert row["provenance_notes"] == ""

    def test_com_proxy_contributors_raw_observed_nonempty(
        self, proxy_rows: list[dict[str, str]]
    ) -> None:
        """All raw_observed contributors should be populated for usable frames."""
        for row in proxy_rows:
            assert row["contributors_raw_observed"] != ""

    # ------------------------------------------------------------------
    # Metadata: citation, coefficients, coverage, schemas, limitations
    # ------------------------------------------------------------------

    def test_metadata_citation_exact(self, meta: Any) -> None:
        """Exact de Leva citation fields."""
        alg = meta["algorithm"]
        assert alg["citation_author"] == "Paolo de Leva"
        assert (
            alg["citation_title"]
            == "Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters"
        )
        assert alg["citation_journal"] == "Journal of Biomechanics"
        assert alg["citation_volume"] == 29
        assert alg["citation_issue"] == 9
        assert alg["citation_pages"] == "1223-1230"
        assert alg["citation_year"] == 1996
        assert alg["doi"] == "10.1016/0021-9290(95)00178-6"

    def test_metadata_coefficient_values(self, meta: Any) -> None:
        """Coefficient table contains exact male mass fractions and corrected r."""
        ct = meta["algorithm"]["coefficient_table"]
        assert ct["head"]["mass_fraction"] == 0.0694
        assert ct["trunk"]["mass_fraction"] == 0.4346
        assert ct["thigh"]["mass_fraction"] == 0.1416
        # Corrected r values
        assert ct["head"]["centroid_ratio_r"] == 0.5002
        assert ct["trunk"]["centroid_ratio_r"] == 0.5138
        assert ct["shank"]["centroid_ratio_r"] == 0.4395

    def test_metadata_model_total_male(self, meta: Any) -> None:
        assert meta["algorithm"]["model_total_mass"] == MODEL_MASS_TOTAL_MALE

    def test_metadata_published_reference_endpoints(self, meta: Any) -> None:
        """published_reference_endpoints exists with anatomical descriptions."""
        pre = meta["algorithm"]["published_reference_endpoints"]
        assert isinstance(pre, dict)
        assert "head" in pre
        assert "trunk" in pre
        assert "thigh" in pre
        # Check proximal/distal keys exist
        for seg in ("head", "trunk", "thigh"):
            assert "proximal" in pre[seg]
            assert "distal" in pre[seg]

    def test_metadata_proxy_approximation_status(self, meta: Any) -> None:
        """segment_endpoint_definitions carries approximation_status."""
        sed = meta["algorithm"]["segment_endpoint_definitions"]
        assert isinstance(sed, dict)
        for seg_name in SEGMENT_NAMES:
            assert seg_name in sed
            assert "approximation_status" in sed[seg_name]
            assert sed[seg_name]["approximation_status"] in (
                "unsupported",
                "joint_centre_proxy",
                "approximate",
            )

    def test_metadata_six_input_paths_and_hashes(
        self, meta: Any, artifacts_dir: Path
    ) -> None:
        """All six inputs have absolute paths and sha256 hashes."""
        required_inputs = {
            "processed_landmarks.csv",
            "preprocessing_metadata.json",
            "pose_frames.csv",
            "reviewed_gait_events.csv",
            "reviewed_strides.csv",
            "review_resolution_metadata.json",
        }
        assert set(meta["inputs"].keys()) == required_inputs
        for name in required_inputs:
            info = meta["inputs"][name]
            assert info["path"] == str(artifacts_dir / name)
            assert isinstance(info["sha256"], str)
            assert len(info["sha256"]) == 64  # SHA-256 hex

    def test_metadata_output_hashes_and_self_semantics(
        self, meta: Any, artifacts_dir: Path
    ) -> None:
        """Output artifacts have path+sha256; com_metadata.json is self-null."""
        outputs = meta["outputs"]
        # com_proxy.csv, stride_com.csv, com_diagnostic.png have real hashes
        for name in ("com_proxy.csv", "stride_com.csv", "com_diagnostic.png"):
            assert name in outputs
            assert outputs[name]["path"] == str(artifacts_dir / name)
            assert isinstance(outputs[name]["sha256"], str)
            assert len(outputs[name]["sha256"]) == 64
        # com_metadata.json has self-null semantics
        assert "com_metadata.json" in outputs
        assert outputs["com_metadata.json"]["sha256"] is None
        assert "sha256_semantics" in outputs["com_metadata.json"]

    def test_metadata_carried_scientific_unresolved(self, meta: Any) -> None:
        """scientific_unresolved is carried from review_resolution_metadata."""
        assert isinstance(meta["carried_scientific_unresolved"], list)
        assert len(meta["carried_scientific_unresolved"]) > 0
        assert any("force plates" in s for s in meta["carried_scientific_unresolved"])

    def test_metadata_coverage_rationale(self, meta: Any) -> None:
        """coverage.rationale documents the default 0.90 QC policy."""
        cov = meta["coverage"]
        assert "rationale" in cov
        assert "0.90" in cov["rationale"] or "90%" in cov["rationale"]
        assert cov["threshold"] == 0.90

    def test_metadata_reviewed_strides_are_canonical(self, meta: Any) -> None:
        """reviewed_strides_are_canonical_qc_windows statement exists."""
        assert "reviewed_strides_are_canonical_qc_windows" in meta
        text = meta["reviewed_strides_are_canonical_qc_windows"]
        assert "not" in text.lower()  # must state what it is NOT

    def test_metadata_schemas_section(self, meta: Any) -> None:
        """Schemas describe com_proxy, stride_com, and com_metadata."""
        schemas = meta["schemas"]
        assert "com_proxy.csv" in schemas
        assert "stride_com.csv" in schemas
        assert "com_metadata.json" in schemas
        # com_proxy.csv and stride_com.csv have column lists
        assert isinstance(schemas["com_proxy.csv"], dict)
        assert "columns" in schemas["com_proxy.csv"]
        assert len(schemas["com_proxy.csv"]["columns"]) > 0

    def test_metadata_validation_status(self, meta: Any) -> None:
        """validation_status explicitly says NOT validated."""
        vs = meta["validation_status"]
        assert "not" in vs.lower() or "NOT" in vs
        assert "validated" in vs.lower()

    def test_metadata_no_gap_rule(self, meta: Any) -> None:
        """no_gap_rule documents interpolation adjacency constraint."""
        alg = meta["algorithm"]
        assert "no_gap_rule" in alg
        assert (
            "adjacent" in alg["no_gap_rule"].lower()
            or "consecutive" in alg["no_gap_rule"].lower()
        )

    def test_metadata_coordinates_and_camera_view(self, meta: Any) -> None:
        """coordinates and camera_view document 2D image-plane nature."""
        alg = meta["algorithm"]
        assert "image-plane" in alg["coordinates"] or "normalized" in alg["coordinates"]
        assert "2D" in alg["camera_view"] or "2d" in alg["camera_view"].lower()

    def test_metadata_unsupported_head(self, meta: Any) -> None:
        """Head is listed as unsupported with explanatory note."""
        alg = meta["algorithm"]
        assert "head" in alg["unsupported_segments"]
        assert alg["unsupported_segments_note"] is not None
        assert (
            "vertex" in alg["unsupported_segments_note"].lower()
            or "neck" in alg["unsupported_segments_note"].lower()
        )

    def test_metadata_represented_mass_max(self, meta: Any) -> None:
        """Represented mass maxima account for unsupported head."""
        alg = meta["algorithm"]
        assert alg["represented_mass_max"] == REPRESENTED_MASS_MAX_MALE
        note = alg["represented_mass_max_note"]
        assert "theoretical maximum" in note.lower() or "unsupported" in note.lower()
        assert "empirical_max_mass_coverage" in alg
        assert isinstance(alg["empirical_max_mass_coverage"], float)

    def test_metadata_projection_assumptions(self, meta: Any) -> None:
        """Projection assumptions document weak-perspective requirements."""
        alg = meta["algorithm"]
        assert "weak-perspective" in alg["projection_assumptions"].lower()

    def test_metadata_identical_segment_set_rule(self, meta: Any) -> None:
        """identical_segment_set_rule documents different-denominator prohibition."""
        alg = meta["algorithm"]
        assert "identical_segment_set_rule" in alg
        assert "different" in alg["identical_segment_set_rule"].lower()
        assert "represented_segment_set_changed" in alg["identical_segment_set_rule"]

    def test_metadata_qc_propagation(self, meta: Any) -> None:
        """qc_propagation documents nonexclusive flag semantics."""
        assert "nonexclusive" in meta["qc_propagation"].lower()

    def test_metadata_model_total_note(self, meta: Any) -> None:
        """model_total_mass_note documents male=1.0000 female=0.9999."""
        alg = meta["algorithm"]
        note = alg["model_total_mass_note"]
        assert "1.0000" in note
        assert "0.9999" in note

    def test_metadata_coefficient_formula(self, meta: Any) -> None:
        """coefficient_table includes centroid_formula for each base segment."""
        ct = meta["algorithm"]["coefficient_table"]
        # coefficient_table is keyed by base names (head, trunk, upper_arm,
        # forearm, hand, thigh, shank, foot) not full bilateral names.
        base_names = [
            "head",
            "trunk",
            "upper_arm",
            "forearm",
            "hand",
            "thigh",
            "shank",
            "foot",
        ]
        for seg_name in base_names:
            assert "centroid_formula" in ct[seg_name], (
                f"missing centroid_formula for {seg_name}"
            )
            assert "proximal" in ct[seg_name]["centroid_formula"]

    def test_metadata_coverage_unrenormalized(self, meta: Any) -> None:
        """coverage.unrenormalized explains represented mass fraction semantics."""
        cov = meta["coverage"]
        assert "unrenormalized" in cov
        assert ".9331" in cov["unrenormalized"] or "0.9331" in cov["unrenormalized"]

    def test_com_proxy_frame_counts_head_unusable(
        self, proxy_rows: list[dict[str, str]]
    ) -> None:
        """All frames have head segment marked unusable."""
        for row in proxy_rows:
            assert row["seg_head_usable"] == "false"
            assert row["seg_head_com_x"] == ""
            assert row["seg_head_com_y"] == ""

    def test_com_proxy_usable_count_13(self, proxy_rows: list[dict[str, str]]) -> None:
        """Exactly 13 of 14 segments are usable per frame."""
        for row in proxy_rows:
            usable_count = sum(
                1 for seg in SEGMENT_NAMES if row[f"seg_{seg}_usable"] == "true"
            )
            assert usable_count == 13


# ---------------------------------------------------------------------------
# Tests: stride/com provenance assertions
# ---------------------------------------------------------------------------


class TestStrideComProvenance:
    """stride_com.csv carries review-provenance extras and normalized fields."""

    @pytest.fixture()
    def artifacts_dir(self, tmp_path: Path) -> Path:
        d = _build_fixture(tmp_path)
        cfg = ComEstimationConfig(anthropometry_sex="male")
        estimate_com(d, cfg)
        return d

    def test_four_review_extras_present(self, artifacts_dir: Path) -> None:
        """Every row carries the four stride review extras."""
        with (artifacts_dir / "stride_com.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert "automatic_stride_id" in row
            assert "review_intent" in row
            assert "review_changes" in row
            assert "provenance_notes" in row

    def test_normalized_sample_source_fields(self, artifacts_dir: Path) -> None:
        """Normalized samples have target_timestamp_seconds and linear/none method."""
        with (artifacts_dir / "stride_com.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        normalized = [r for r in rows if r["sample_kind"] == "normalized"]
        for row in normalized:
            assert "target_timestamp_seconds" in row
            assert row["method"] in ("linear", "exact", "none")

    def test_com_x_com_y_populated_for_usable(self, artifacts_dir: Path) -> None:
        """Usable stride samples have nonempty com_x and com_y."""
        with (artifacts_dir / "stride_com.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row["usable"] == "true":
                assert row["com_x"] != ""
                assert row["com_y"] != ""

    def test_stride_side_matches_events(self, artifacts_dir: Path) -> None:
        """Every stride row has side=='left' (matching our fixture events)."""
        with (artifacts_dir / "stride_com.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert row["side"] == "left"


class TestComProxyProvenance:
    """com_proxy.csv carries per-segment provenance flags."""

    @pytest.fixture()
    def artifacts_dir(self, tmp_path: Path) -> Path:
        d = _build_fixture(tmp_path)
        cfg = ComEstimationConfig(anthropometry_sex="male")
        estimate_com(d, cfg)
        return d

    def test_segment_qc_flags_present(self, artifacts_dir: Path) -> None:
        """Every segment has a qc_flags column."""
        with (artifacts_dir / "com_proxy.csv").open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        for seg_name in SEGMENT_NAMES:
            assert f"seg_{seg_name}_qc_flags" in row

    def test_segment_usable_flags(self, artifacts_dir: Path) -> None:
        """All supported segment usable flags are true; head is false."""
        with (artifacts_dir / "com_proxy.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for seg_name in SEGMENT_NAMES:
                if seg_name == "head":
                    assert row[f"seg_{seg_name}_usable"] == "false"
                else:
                    assert row[f"seg_{seg_name}_usable"] == "true"

    def test_contributor_aggregation_columns(self, artifacts_dir: Path) -> None:
        """com_proxy.csv has all contributor aggregation columns."""
        expected_cols = [
            "contributors_raw_observed",
            "contributors_x_interpolated",
            "contributors_y_interpolated",
            "contributors_x_smoothing_changed",
            "contributors_y_smoothing_changed",
            "contributors_x_smoothing_support_interpolation",
            "contributors_y_smoothing_support_interpolation",
            "contributors_other_qc_limited",
        ]
        with (artifacts_dir / "com_proxy.csv").open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        for col in expected_cols:
            assert col in row

    def test_mass_coverage_columns(self, artifacts_dir: Path) -> None:
        """com_proxy.csv has nonexclusive mass per QC flag category."""
        with (artifacts_dir / "com_proxy.csv").open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        # Nonexclusive mass totals per independent QC flag
        assert "mass_x_interpolated" in row
        assert "mass_y_interpolated" in row
        assert "mass_x_smoothing_changed" in row
        assert "mass_y_smoothing_changed" in row
        assert "mass_x_smoothing_support_interpolation" in row
        assert "mass_y_smoothing_support_interpolation" in row
        assert "mass_other_qc_limited" in row
        assert "mass_missing" in row


# ---------------------------------------------------------------------------
# Tests: reviewed-only stride segmentation
# ---------------------------------------------------------------------------


class TestReviewedOnlySegmentation:
    """Only reviewed strides produce stride_com.csv entries."""

    @pytest.fixture()
    def artifacts_dir(self, tmp_path: Path) -> Path:
        d = _build_fixture(tmp_path)
        cfg = ComEstimationConfig(anthropometry_sex="male")
        estimate_com(d, cfg)
        return d

    def test_stride_com_only_reviewed_strides(self, artifacts_dir: Path) -> None:
        with (artifacts_dir / "stride_com.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        stride_ids = {r["stride_id"] for r in rows}
        assert stride_ids == {"RS001"}


# ---------------------------------------------------------------------------
# Tests: review_status / review_intent mismatch
# ---------------------------------------------------------------------------


class TestReviewStatusMismatch:
    """Pipeline carries review_status through; mismatched status is not a
    pipeline-level validation error."""

    def test_mismatched_review_status_rejected(self, tmp_path: Path) -> None:
        """review_status=reject with review_intent=accept still succeeds
        because review_status is informational, not validated at pipeline level.
        But the stride CSV hash must match the metadata record."""
        d = _build_fixture(tmp_path)
        # Overwrite strides with mismatched review_status
        rs = d / "reviewed_strides.csv"
        fieldnames = list(REVIEWED_STRIDE_FIELDS)
        with rs.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "stride_id": "RS001",
                    "side": "left",
                    "start_event_id": "RE001",
                    "end_event_id": "RE002",
                    "start_frame": "0",
                    "end_frame": "2",
                    "start_timestamp_seconds": "0.0",
                    "end_timestamp_seconds": "0.067",
                    "duration_seconds": "0.067",
                    "quality": "high",
                    "contralateral_event_id": "",
                    "contralateral_event_count": "0",
                    "sequence_notes": "",
                    "source": "automatic",
                    "review_status": "reject",
                    "automatic_stride_id": "S0001",
                    "review_intent": "accept",
                    "review_changes": "",
                    "provenance_notes": "",
                }
            )
        # Re-hash reviewed_strides.csv and update review_resolution_metadata
        new_rs_hash = _sha256_file_content(rs)
        rr_meta_path = d / "review_resolution_metadata.json"
        rr_meta = json.loads(rr_meta_path.read_text(encoding="utf-8"))
        rr_meta["outputs"]["reviewed_strides.csv"]["sha256"] = new_rs_hash
        rr_meta_path.write_text(
            json.dumps(rr_meta, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        cfg = ComEstimationConfig(anthropometry_sex="male")
        # Pipeline should accept any review_status value (it's informational,
        # not validated against review_intent at pipeline level).
        # This test documents that review_status is carried through.
        artifacts = estimate_com(d, cfg)
        assert artifacts.com_proxy_path.exists()


# ---------------------------------------------------------------------------
# Tests: malformed inputs
# ---------------------------------------------------------------------------


class TestMalformedInputs:
    """Schema / hash / status / blocking / endpoint malformed cases."""

    def test_missing_processed_landmarks(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        (d / "processed_landmarks.csv").unlink()
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="missing"):
            estimate_com(d, cfg)

    def test_missing_preprocessing_metadata(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        (d / "preprocessing_metadata.json").unlink()
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="missing"):
            estimate_com(d, cfg)

    def test_missing_pose_frames(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        (d / "pose_frames.csv").unlink()
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="missing"):
            estimate_com(d, cfg)

    def test_missing_reviewed_gait_events(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        (d / "reviewed_gait_events.csv").unlink()
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="missing"):
            estimate_com(d, cfg)

    def test_missing_reviewed_strides(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        (d / "reviewed_strides.csv").unlink()
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="missing"):
            estimate_com(d, cfg)

    def test_missing_review_resolution_metadata(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        (d / "review_resolution_metadata.json").unlink()
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="missing"):
            estimate_com(d, cfg)

    def test_nonexistent_directory(self) -> None:
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="does not exist"):
            estimate_com("/nonexistent/path", cfg)

    def test_pose_frames_hash_mismatch(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        meta_path = d / "preprocessing_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["inputs"]["pose_frames.csv"]["sha256"] = "0" * 64
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="hash"):
            estimate_com(d, cfg)

    def test_reviewed_gait_events_hash_mismatch(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        rr_path = d / "review_resolution_metadata.json"
        meta = json.loads(rr_path.read_text(encoding="utf-8"))
        meta["outputs"]["reviewed_gait_events.csv"]["sha256"] = "0" * 64
        rr_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="hash"):
            estimate_com(d, cfg)

    def test_reviewed_strides_hash_mismatch(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        rr_path = d / "review_resolution_metadata.json"
        meta = json.loads(rr_path.read_text(encoding="utf-8"))
        meta["outputs"]["reviewed_strides.csv"]["sha256"] = "0" * 64
        rr_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="hash"):
            estimate_com(d, cfg)

    def test_preprocessing_schema_version_wrong(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        meta_path = d / "preprocessing_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["schema_version"] = 99
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="schema_version"):
            estimate_com(d, cfg)

    def test_review_resolution_schema_version_wrong(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        rr_path = d / "review_resolution_metadata.json"
        meta = json.loads(rr_path.read_text(encoding="utf-8"))
        meta["schema_version"] = 99
        rr_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="schema_version"):
            estimate_com(d, cfg)

    def test_reviewed_gait_events_empty(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        rge = d / "reviewed_gait_events.csv"
        with rge.open("w", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(REVIEWED_GAIT_EVENT_FIELDS))
            writer.writeheader()
        _build_review_metadata(
            d / "review_resolution_metadata.json",
            d / "reviewed_gait_events.csv",
            d / "reviewed_strides.csv",
            d / "preprocessing_metadata.json",
            d / "pose_frames.csv",
            stride_count=1,
        )
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="at least one row"):
            estimate_com(d, cfg)

    def test_pose_frames_header_mismatch(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        pf = d / "pose_frames.csv"
        with pf.open("w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index", "wrong_header", "x"])
            writer.writerow(["0", "0.0", "0.1"])
        _build_preprocessing_metadata(
            d / "preprocessing_metadata.json", pf, d / "processed_landmarks.csv"
        )
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="header"):
            estimate_com(d, cfg)

    def test_processed_landmarks_header_mismatch(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        pl = d / "processed_landmarks.csv"
        with pl.open("w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index", "wrong_column"])
            writer.writerow(["0", "0.1"])
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="header"):
            estimate_com(d, cfg)

    def test_reviewed_strides_missing_required_field(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        rs = d / "reviewed_strides.csv"
        from gait_stability.com_pipeline import REVIEWED_STRIDE_FIELDS

        fieldnames = [f for f in REVIEWED_STRIDE_FIELDS if f != "stride_id"]
        with rs.open("w", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({f: "" for f in fieldnames})
        _build_review_metadata(
            d / "review_resolution_metadata.json",
            d / "reviewed_gait_events.csv",
            d / "reviewed_strides.csv",
            d / "preprocessing_metadata.json",
            d / "pose_frames.csv",
            stride_count=1,
        )
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="header"):
            estimate_com(d, cfg)

    def test_nonexistent_directory_is_blocking(self) -> None:
        """Nonexistent directory is caught before any file I/O."""
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError):
            estimate_com("/tmp/completely_nonexistent_xyz_999", cfg)

    def test_pose_frames_nonfinite_timestamp(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        pf = d / "pose_frames.csv"
        with pf.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(FRAME_FIELDS))
            writer.writeheader()
            writer.writerow(
                {
                    "frame_index": "0",
                    "nominal_timestamp_seconds": "nan",
                    "backend_timestamp_milliseconds": "100",
                    "status": "decoded_pose",
                    "landmark_count": "1",
                    "detail": "",
                }
            )
        _build_preprocessing_metadata(
            d / "preprocessing_metadata.json", pf, d / "processed_landmarks.csv"
        )
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="finite"):
            estimate_com(d, cfg)

    def test_pose_frames_duplicate_frame_index(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        pf = d / "pose_frames.csv"
        with pf.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(FRAME_FIELDS))
            writer.writeheader()
            writer.writerow(
                {
                    "frame_index": "0",
                    "nominal_timestamp_seconds": "0.0",
                    "backend_timestamp_milliseconds": "100",
                    "status": "decoded_pose",
                    "landmark_count": "0",
                    "detail": "",
                }
            )
            writer.writerow(
                {
                    "frame_index": "0",
                    "nominal_timestamp_seconds": "0.033",
                    "backend_timestamp_milliseconds": "133",
                    "status": "decoded_pose",
                    "landmark_count": "0",
                    "detail": "",
                }
            )
        _build_preprocessing_metadata(
            d / "preprocessing_metadata.json", pf, d / "processed_landmarks.csv"
        )
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="duplicate"):
            estimate_com(d, cfg)

    # ------------------------------------------------------------------
    # Strict malformed data: processed boolean, coordinate, algorithm,
    # basename, blocking, scientific, timestamp_source, event disposition
    # ------------------------------------------------------------------

    def test_processed_malformed_boolean(self, tmp_path: Path) -> None:
        """processed_landmarks with 'True' (capital T) boolean rejected."""
        d = _build_fixture(tmp_path)
        pl = d / "processed_landmarks.csv"
        fieldnames = list(PROCESSED_FIELDS)
        with pl.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for frame in _FRAMES:
                fi = int(frame["frame_index"])
                ts = float(frame["nominal_timestamp_seconds"])
                for lm_name, (x, y) in _ALL_LANDMARK_COORDS.items():
                    row = _processed_row(fi, ts, lm_name, _LANDMARK_IDS[lm_name], x, y)
                    # Inject a capital-T boolean on first row only
                    if fi == 0 and lm_name == "nose":
                        row["raw_row_present"] = "True"
                    writer.writerow(row)
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="true or false"):
            estimate_com(d, cfg)

    def test_processed_nonfinite_coordinate(self, tmp_path: Path) -> None:
        """processed_x_normalized with nan value rejected."""
        d = _build_fixture(tmp_path)
        pl = d / "processed_landmarks.csv"
        fieldnames = list(PROCESSED_FIELDS)
        with pl.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for frame in _FRAMES:
                fi = int(frame["frame_index"])
                ts = float(frame["nominal_timestamp_seconds"])
                for lm_name, (x, y) in _ALL_LANDMARK_COORDS.items():
                    row = _processed_row(fi, ts, lm_name, _LANDMARK_IDS[lm_name], x, y)
                    if fi == 0 and lm_name == "nose":
                        row["processed_x_normalized"] = "nan"
                    writer.writerow(row)
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="finite"):
            estimate_com(d, cfg)

    def test_processed_status_mismatch_with_manifest(self, tmp_path: Path) -> None:
        """processed frame_status != pose_frames status -> error."""
        d = _build_fixture(tmp_path)
        pl = d / "processed_landmarks.csv"
        fieldnames = list(PROCESSED_FIELDS)
        with pl.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for frame in _FRAMES:
                fi = int(frame["frame_index"])
                ts = float(frame["nominal_timestamp_seconds"])
                for lm_name, (x, y) in _ALL_LANDMARK_COORDS.items():
                    row = _processed_row(fi, ts, lm_name, _LANDMARK_IDS[lm_name], x, y)
                    if fi == 0:
                        row["frame_status"] = "no_pose"
                    writer.writerow(row)
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="frame_status"):
            estimate_com(d, cfg)

    def test_processed_timestamp_mismatch(self, tmp_path: Path) -> None:
        """processed timestamp != pose_frames timestamp -> error."""
        d = _build_fixture(tmp_path)
        pl = d / "processed_landmarks.csv"
        fieldnames = list(PROCESSED_FIELDS)
        with pl.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for frame in _FRAMES:
                fi = int(frame["frame_index"])
                ts = float(frame["nominal_timestamp_seconds"])
                for lm_name, (x, y) in _ALL_LANDMARK_COORDS.items():
                    row = _processed_row(fi, ts, lm_name, _LANDMARK_IDS[lm_name], x, y)
                    if fi == 0:
                        row["nominal_timestamp_seconds"] = "999.0"
                    writer.writerow(row)
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="timestamp"):
            estimate_com(d, cfg)

    def test_preprocessing_algorithm_version_wrong(self, tmp_path: Path) -> None:
        """preprocessing algorithm_version mismatch."""
        d = _build_fixture(tmp_path)
        meta_path = d / "preprocessing_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["algorithm_version"] = "wrong-version-999"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="algorithm_version"):
            estimate_com(d, cfg)

    def test_preprocessing_input_path_basename_wrong(self, tmp_path: Path) -> None:
        """preprocessing inputs.pose_frames.csv path basename mismatch."""
        d = _build_fixture(tmp_path)
        meta_path = d / "preprocessing_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["inputs"]["pose_frames.csv"]["path"] = "/some/other/wrong_file.csv"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="basename"):
            estimate_com(d, cfg)

    def test_review_algorithm_version_wrong(self, tmp_path: Path) -> None:
        """review_resolution algorithm_version mismatch."""
        d = _build_fixture(tmp_path)
        rr_path = d / "review_resolution_metadata.json"
        meta = json.loads(rr_path.read_text(encoding="utf-8"))
        meta["algorithm_version"] = "wrong-review-version"
        rr_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="algorithm_version"):
            estimate_com(d, cfg)

    def test_blocking_unresolved_nonempty(self, tmp_path: Path) -> None:
        """blocking_unresolved must be empty list."""
        d = _build_fixture(tmp_path)
        rr_path = d / "review_resolution_metadata.json"
        meta = json.loads(rr_path.read_text(encoding="utf-8"))
        meta["blocking_unresolved"] = ["some_blocking_issue"]
        rr_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="blocking_unresolved"):
            estimate_com(d, cfg)

    def test_scientific_unresolved_wrong_type(self, tmp_path: Path) -> None:
        """scientific_unresolved must be list of strings."""
        d = _build_fixture(tmp_path)
        rr_path = d / "review_resolution_metadata.json"
        meta = json.loads(rr_path.read_text(encoding="utf-8"))
        meta["scientific_unresolved"] = [123, True]  # not strings
        rr_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="scientific_unresolved"):
            estimate_com(d, cfg)

    def test_timestamp_source_hash_mismatch(self, tmp_path: Path) -> None:
        """timestamp_source hash does not match actual pose_frames hash."""
        d = _build_fixture(tmp_path)
        rr_path = d / "review_resolution_metadata.json"
        meta = json.loads(rr_path.read_text(encoding="utf-8"))
        meta["timestamp_source"]["sha256"] = "0" * 64
        rr_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="hash"):
            estimate_com(d, cfg)

    def test_timestamp_source_path_basename_wrong(self, tmp_path: Path) -> None:
        """timestamp_source path basename must be pose_frames.csv."""
        d = _build_fixture(tmp_path)
        rr_path = d / "review_resolution_metadata.json"
        meta = json.loads(rr_path.read_text(encoding="utf-8"))
        meta["timestamp_source"]["path"] = "/some/other/wrong.csv"
        rr_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="basename"):
            estimate_com(d, cfg)

    def test_preprocessing_metadata_hash_mismatch_from_review(
        self, tmp_path: Path
    ) -> None:
        """review_resolution metadata stores wrong hash for preprocessing_metadata."""
        d = _build_fixture(tmp_path)
        rr_path = d / "review_resolution_metadata.json"
        meta = json.loads(rr_path.read_text(encoding="utf-8"))
        meta["inputs"]["preprocessing_metadata.json"]["sha256"] = "0" * 64
        rr_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="hash"):
            estimate_com(d, cfg)

    def test_reviewed_event_disposition_invalid(self, tmp_path: Path) -> None:
        """resolution_disposition='retained' is not in valid set."""
        d = _build_fixture(tmp_path)
        rge = d / "reviewed_gait_events.csv"
        fieldnames = list(REVIEWED_GAIT_EVENT_FIELDS)
        with rge.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "event_id": "RE001",
                    "automatic_event_id": "E0001",
                    "side": "left",
                    "event_type": "candidate_initial_contact",
                    "automatic_frame_index": "0",
                    "automatic_timestamp_seconds": "0.0",
                    "automatic_disposition": "accepted_unchanged",
                    "automatic_quality": "high",
                    "automatic_peak_value": "0.12",
                    "automatic_prominence": "0.08",
                    "automatic_rejection_reasons": "",
                    "manual_event_review_status": "unreviewed",
                    "stride_review_provenance": "",
                    "reviewed_frame_index": "0",
                    "reviewed_timestamp_seconds": "0.0",
                    "reviewed_accepted": "true",
                    "reviewed_rejected": "false",
                    "reviewed_included_in_stride": "true",
                    "reviewed_quality": "high",
                    "resolution_disposition": "retained",
                    "replaces_event_id": "",
                    "replaced_by_event_id": "",
                    "source": "automatic",
                    "review_notes": "",
                }
            )
        _build_review_metadata(
            d / "review_resolution_metadata.json",
            d / "reviewed_gait_events.csv",
            d / "reviewed_strides.csv",
            d / "preprocessing_metadata.json",
            d / "pose_frames.csv",
            stride_count=0,
        )
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="resolution_disposition"):
            estimate_com(d, cfg)

    def test_reviewed_event_endpoint_not_in_pose_manifest(self, tmp_path: Path) -> None:
        """reviewed frame_index referencing nonexistent pose frame."""
        d = _build_fixture(tmp_path)
        rge = d / "reviewed_gait_events.csv"
        fieldnames = list(REVIEWED_GAIT_EVENT_FIELDS)
        with rge.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "event_id": "RE001",
                    "automatic_event_id": "E0001",
                    "side": "left",
                    "event_type": "candidate_initial_contact",
                    "automatic_frame_index": "0",
                    "automatic_timestamp_seconds": "0.0",
                    "automatic_disposition": "accepted_unchanged",
                    "automatic_quality": "high",
                    "automatic_peak_value": "0.12",
                    "automatic_prominence": "0.08",
                    "automatic_rejection_reasons": "",
                    "manual_event_review_status": "unreviewed",
                    "stride_review_provenance": "",
                    "reviewed_frame_index": "999",
                    "reviewed_timestamp_seconds": "0.0",
                    "reviewed_accepted": "true",
                    "reviewed_rejected": "false",
                    "reviewed_included_in_stride": "true",
                    "reviewed_quality": "high",
                    "resolution_disposition": "accepted_unchanged",
                    "replaces_event_id": "",
                    "replaced_by_event_id": "",
                    "source": "automatic",
                    "review_notes": "",
                }
            )
        _build_review_metadata(
            d / "review_resolution_metadata.json",
            d / "reviewed_gait_events.csv",
            d / "reviewed_strides.csv",
            d / "preprocessing_metadata.json",
            d / "pose_frames.csv",
            stride_count=0,
        )
        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError, match="reviewed_frame_index"):
            estimate_com(d, cfg)


# ---------------------------------------------------------------------------
# Tests: input mutation (true prepublication mutation via monkeypatch)
# ---------------------------------------------------------------------------


class TestInputMutationProtection:
    """Pipeline re-verifies input hashes after processing."""

    def test_input_change_detected(self, tmp_path: Path) -> None:
        """If an input changes after initial hash snapshot, pipeline raises."""
        d = _build_fixture(tmp_path)
        pf = d / "pose_frames.csv"

        with pf.open("a", encoding="utf-8") as f:
            f.write("extra line\n")

        cfg = ComEstimationConfig(anthropometry_sex="male")
        with pytest.raises(ComArtifactValidationError):
            estimate_com(d, cfg)

    def test_true_prepublication_mutation_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutate processed_landmarks after initial validation but before
        final hash recheck; pipeline detects and raises ComArtifactValidationError
        naming the changed input, with no partial outputs published."""
        d = _build_fixture(tmp_path)
        cfg = ComEstimationConfig(anthropometry_sex="male")

        # Patch _write_com_proxy (a late staging function) to mutate
        # processed_landmarks.csv after the initial hash snapshot but before
        # the final _recheck_input_hashes call.
        original_write_com_proxy = None

        from gait_stability import com_pipeline as _mod

        original_write_com_proxy = _mod._write_com_proxy

        def _mutate_during_write(*args: Any, **kwargs: Any) -> None:
            # Before writing the real output, tamper with the input
            pl = d / "processed_landmarks.csv"
            with pl.open("a", encoding="utf-8") as f:
                f.write("tampered\n")
            return original_write_com_proxy(*args, **kwargs)

        monkeypatch.setattr(_mod, "_write_com_proxy", _mutate_during_write)

        with pytest.raises(ComArtifactValidationError, match="changed"):
            estimate_com(d, cfg)

        # No partial outputs should have been published to destination
        for name in COM_OUTPUT_ARTIFACT_NAMES:
            assert not (d / name).exists(), f"Partial output {name} should not exist"


# ---------------------------------------------------------------------------
# Tests: transactional rollback (true Path.replace monkeypatch)
# ---------------------------------------------------------------------------


class TestTransactionalRollback:
    """On failure during publication, previously published outputs are restored."""

    def test_no_partial_outputs_on_failure(self, tmp_path: Path) -> None:
        """Existing outputs survive a failed re-run."""
        d = _build_fixture(tmp_path)
        cfg = ComEstimationConfig(anthropometry_sex="male")
        estimate_com(d, cfg)
        for name in COM_OUTPUT_ARTIFACT_NAMES:
            assert (d / name).exists()

        pf = d / "pose_frames.csv"
        with pf.open("a", encoding="utf-8") as f:
            f.write("corrupted\n")

        with pytest.raises((ComPipelineError, ComArtifactValidationError)):
            estimate_com(d, cfg)

        assert (d / "com_proxy.csv").exists()

    def test_rollback_restores_prior_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First publish succeeds; second run fails during the second
        Path.replace call. All four outputs must be restored byte-for-byte
        to the originals from the first successful run."""
        d = _build_fixture(tmp_path)
        cfg = ComEstimationConfig(anthropometry_sex="male")
        estimate_com(d, cfg)

        # Snapshot the original output bytes
        original_bytes: dict[str, bytes] = {}
        for name in COM_OUTPUT_ARTIFACT_NAMES:
            original_bytes[name] = (d / name).read_bytes()

        # Monkeypatch Path.replace: fail on the second call (which is
        # when the first output file is moved to destination, after backup).
        call_count = 0
        original_replace = Path.replace

        def _fail_on_second_replace(self: Path, target: Path | str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("Simulated publication failure")
            return original_replace(self, target)

        monkeypatch.setattr(Path, "replace", _fail_on_second_replace)

        with pytest.raises(ArtifactPublishError, match="Step 5"):
            estimate_com(d, cfg)

        # Restore Path.replace for assertion reads
        monkeypatch.setattr(Path, "replace", original_replace)

        # Verify all four outputs restored to original bytes
        for name in COM_OUTPUT_ARTIFACT_NAMES:
            assert (d / name).exists(), f"{name} missing after rollback"
            current = (d / name).read_bytes()
            assert current == original_bytes[name], (
                f"{name} bytes differ after rollback"
            )


# ---------------------------------------------------------------------------
# Tests: regression boundary - stride changes but frame COM unchanged
# ---------------------------------------------------------------------------


class TestRegressionBoundary:
    """Changing stride boundaries changes stride_com but not com_proxy."""

    def test_boundary_change_stride_only(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        cfg = ComEstimationConfig(anthropometry_sex="male")

        # First run
        estimate_com(d, cfg)
        com1 = (d / "com_proxy.csv").read_text(encoding="utf-8")
        stride1 = (d / "stride_com.csv").read_text(encoding="utf-8")

        # Change the reviewed event RE002 to frame 1 (was frame 2)
        # and update the stride to match.
        rge = d / "reviewed_gait_events.csv"
        rge_fieldnames = list(REVIEWED_GAIT_EVENT_FIELDS)
        with rge.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rge_fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "event_id": "RE001",
                    "automatic_event_id": "E0001",
                    "side": "left",
                    "event_type": "candidate_initial_contact",
                    "automatic_frame_index": "0",
                    "automatic_timestamp_seconds": "0.0",
                    "automatic_disposition": "accepted_unchanged",
                    "automatic_quality": "high",
                    "automatic_peak_value": "0.12",
                    "automatic_prominence": "0.08",
                    "automatic_rejection_reasons": "",
                    "manual_event_review_status": "unreviewed",
                    "stride_review_provenance": "",
                    "reviewed_frame_index": "0",
                    "reviewed_timestamp_seconds": "0.0",
                    "reviewed_accepted": "true",
                    "reviewed_rejected": "false",
                    "reviewed_included_in_stride": "true",
                    "reviewed_quality": "high",
                    "resolution_disposition": "accepted_unchanged",
                    "replaces_event_id": "",
                    "replaced_by_event_id": "",
                    "source": "automatic",
                    "review_notes": "",
                }
            )
            writer.writerow(
                {
                    "event_id": "RE002",
                    "automatic_event_id": "E0002",
                    "side": "left",
                    "event_type": "candidate_initial_contact",
                    "automatic_frame_index": "1",
                    "automatic_timestamp_seconds": "0.033",
                    "automatic_disposition": "accepted_unchanged",
                    "automatic_quality": "high",
                    "automatic_peak_value": "0.15",
                    "automatic_prominence": "0.10",
                    "automatic_rejection_reasons": "",
                    "manual_event_review_status": "unreviewed",
                    "stride_review_provenance": "",
                    "reviewed_frame_index": "1",
                    "reviewed_timestamp_seconds": "0.033",
                    "reviewed_accepted": "true",
                    "reviewed_rejected": "false",
                    "reviewed_included_in_stride": "true",
                    "reviewed_quality": "high",
                    "resolution_disposition": "accepted_unchanged",
                    "replaces_event_id": "",
                    "replaced_by_event_id": "",
                    "source": "automatic",
                    "review_notes": "",
                }
            )

        rs = d / "reviewed_strides.csv"
        with rs.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(REVIEWED_STRIDE_FIELDS))
            writer.writeheader()
            writer.writerow(
                {
                    "stride_id": "RS001",
                    "side": "left",
                    "start_event_id": "RE001",
                    "end_event_id": "RE002",
                    "start_frame": "0",
                    "end_frame": "1",
                    "start_timestamp_seconds": "0.0",
                    "end_timestamp_seconds": "0.033",
                    "duration_seconds": "0.033",
                    "quality": "high",
                    "contralateral_event_id": "",
                    "contralateral_event_count": "0",
                    "sequence_notes": "",
                    "source": "automatic",
                    "review_status": "unreviewed",
                    "automatic_stride_id": "S0001",
                    "review_intent": "accept",
                    "review_changes": "",
                    "provenance_notes": "",
                }
            )

        _build_review_metadata(
            d / "review_resolution_metadata.json",
            d / "reviewed_gait_events.csv",
            d / "reviewed_strides.csv",
            d / "preprocessing_metadata.json",
            d / "pose_frames.csv",
            stride_count=1,
        )

        estimate_com(d, cfg)
        com2 = (d / "com_proxy.csv").read_text(encoding="utf-8")
        stride2 = (d / "stride_com.csv").read_text(encoding="utf-8")

        # com_proxy should be identical (same frames, same landmarks)
        assert com1 == com2
        # stride_com should differ (different stride boundaries)
        assert stride1 != stride2
