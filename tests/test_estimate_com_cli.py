"""CLI tests for estimate_com.py entry point.

Verifies:
- --anthropometry-sex is required
- --minimum-mass-coverage and --normalized-stride-samples are supported
- Missing sex yields error
- Invalid sex yields error
- Success prints metadata path
- Failure prints error to stderr
- Custom options are forwarded correctly
- Parameterized invalid coverage/samples values yield clean errors
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from gait_stability.com_pipeline import REVIEWED_STRIDE_FIELDS
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

# ---------------------------------------------------------------------------
# Canonical fixture builder (matches pipeline tests exactly)
# ---------------------------------------------------------------------------

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

_LANDMARK_IDS: dict[str, int] = {
    name: idx for idx, name in enumerate(MEDIAPIPE_LANDMARK_NAMES)
}

assert len(_ALL_LANDMARK_COORDS) == len(MEDIAPIPE_LANDMARK_NAMES)
assert set(_ALL_LANDMARK_COORDS.keys()) == set(MEDIAPIPE_LANDMARK_NAMES)

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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _build_fixture(tmp: Path) -> Path:
    d = tmp / "cli_artifacts"
    d.mkdir(parents=True, exist_ok=True)

    pf = d / "pose_frames.csv"
    with pf.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(FRAME_FIELDS))
        writer.writeheader()
        for frame in _FRAMES:
            writer.writerow(frame)

    pl = d / "processed_landmarks.csv"
    with pl.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(PROCESSED_FIELDS))
        writer.writeheader()
        for frame in _FRAMES:
            fi = int(frame["frame_index"])
            ts = float(frame["nominal_timestamp_seconds"])
            for lm_name, (x, y) in _ALL_LANDMARK_COORDS.items():
                writer.writerow(
                    _processed_row(fi, ts, lm_name, _LANDMARK_IDS[lm_name], x, y)
                )

    pf_hash = _sha256_file(pf)
    pl_hash = _sha256_file(pl)
    (d / "preprocessing_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": PREPROCESSING_SCHEMA_VERSION,
                "algorithm_version": PREPROCESSING_ALGORITHM_VERSION,
                "inputs": {"pose_frames.csv": {"path": str(pf), "sha256": pf_hash}},
                "outputs": {
                    "processed_landmarks.csv": {"path": str(pl), "sha256": pl_hash}
                },
                "config": {"dummy": True},
                "inherited_provenance": {},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rge = d / "reviewed_gait_events.csv"
    with rge.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(REVIEWED_GAIT_EVENT_FIELDS))
        writer.writeheader()
        for ev in [
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
        ]:
            writer.writerow(ev)

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
        )

    rge_hash = _sha256_file(rge)
    rs_hash = _sha256_file(rs)
    pm_hash = _sha256_file(d / "preprocessing_metadata.json")
    pf_hash2 = _sha256_file(pf)
    (d / "review_resolution_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": REVIEW_RESOLUTION_SCHEMA_VERSION,
                "algorithm_version": REVIEW_RESOLUTION_ALGORITHM_VERSION,
                "outputs": {
                    "reviewed_gait_events.csv": {
                        "path": str(rge),
                        "sha256": rge_hash,
                    },
                    "reviewed_strides.csv": {
                        "path": str(rs),
                        "sha256": rs_hash,
                    },
                },
                "inputs": {
                    "preprocessing_metadata.json": {
                        "path": str(d / "preprocessing_metadata.json"),
                        "sha256": pm_hash,
                    }
                },
                "timestamp_source": {
                    "path": str(pf),
                    "sha256": pf_hash2,
                },
                "blocking_unresolved": [],
                "scientific_unresolved": ["Not validated"],
                "counts": {"reviewed_strides": 1},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return d


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/estimate_com.py", *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCLIRequiresSex:
    """--anthropometry-sex is required."""

    def test_missing_sex_fails(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        result = _run_cli(str(d))
        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "failed" in result.stderr.lower()

    def test_empty_string_sex_fails(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        result = _run_cli(str(d), "--anthropometry-sex", "")
        assert result.returncode != 0


class TestCLISexValidation:
    """--anthropometry-sex validates choices."""

    def test_male_accepted(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        result = _run_cli(str(d), "--anthropometry-sex", "male")
        assert result.returncode == 0

    def test_female_accepted(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        result = _run_cli(str(d), "--anthropometry-sex", "female")
        assert result.returncode == 0

    def test_invalid_sex_rejected(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        result = _run_cli(str(d), "--anthropometry-sex", "other")
        assert result.returncode != 0
        assert "invalid" in result.stderr.lower() or "error" in result.stderr.lower()


class TestCLICustomOptions:
    """--minimum-mass-coverage and --normalized-stride-samples are forwarded."""

    def test_custom_coverage(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        result = _run_cli(
            str(d), "--anthropometry-sex", "male", "--minimum-mass-coverage", "0.80"
        )
        assert result.returncode == 0
        meta = json.loads((d / "com_metadata.json").read_text(encoding="utf-8"))
        assert meta["config"]["minimum_mass_coverage"] == 0.80

    def test_custom_normalized_stride_samples(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        result = _run_cli(
            str(d),
            "--anthropometry-sex",
            "female",
            "--normalized-stride-samples",
            "51",
        )
        assert result.returncode == 0
        meta = json.loads((d / "com_metadata.json").read_text(encoding="utf-8"))
        assert meta["config"]["normalized_stride_samples"] == 51
        assert meta["config"]["anthropometry_sex"] == "female"

    def test_defaults_used(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        result = _run_cli(str(d), "--anthropometry-sex", "male")
        assert result.returncode == 0
        meta = json.loads((d / "com_metadata.json").read_text(encoding="utf-8"))
        assert meta["config"]["minimum_mass_coverage"] == 0.90
        assert meta["config"]["normalized_stride_samples"] == 101


class TestCLISuccessOutput:
    """Success prints metadata path to stdout."""

    def test_stdout_is_metadata_path(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        result = _run_cli(str(d), "--anthropometry-sex", "male")
        assert result.returncode == 0
        stdout = result.stdout.strip()
        assert stdout.endswith("com_metadata.json")
        assert Path(stdout).exists()


class TestCLIFailureOutput:
    """Failure prints error to stderr."""

    def test_bad_directory_stderr(self) -> None:
        result = _run_cli("/nonexistent_xyz", "--anthropometry-sex", "male")
        assert result.returncode != 0
        assert "failed" in result.stderr.lower() or "error" in result.stderr.lower()


class TestCLIMissingDirectory:
    """Missing directory exits with error."""

    def test_missing_dir(self) -> None:
        result = _run_cli("/tmp/nonexistent_com_dir_xyz", "--anthropometry-sex", "male")
        assert result.returncode == 1


# ---------------------------------------------------------------------------
# Tests: parameterized invalid coverage/samples CLI values
# ---------------------------------------------------------------------------


class TestCLIInvalidParameterization:
    """Invalid --minimum-mass-coverage and --normalized-stride-samples
    values are rejected with no traceback."""

    @pytest.mark.parametrize(
        "bad_coverage",
        ["-0.1", "1.1", "nan", "inf"],
        ids=["negative", "above-one", "nan", "inf"],
    )
    def test_invalid_coverage_rejected(self, tmp_path: Path, bad_coverage: str) -> None:
        d = _build_fixture(tmp_path)
        result = _run_cli(
            str(d),
            "--anthropometry-sex",
            "male",
            "--minimum-mass-coverage",
            bad_coverage,
        )
        assert result.returncode != 0
        # Must be argparse error, not a Python traceback
        assert "Traceback" not in result.stderr
        assert (
            "error" in result.stderr.lower()
            or "invalid" in result.stderr.lower()
            or "not a number" in result.stderr.lower()
        )

    @pytest.mark.parametrize(
        "bad_samples",
        ["0", "1"],
        ids=["zero", "one"],
    )
    def test_invalid_stride_samples_rejected(
        self, tmp_path: Path, bad_samples: str
    ) -> None:
        d = _build_fixture(tmp_path)
        result = _run_cli(
            str(d),
            "--anthropometry-sex",
            "male",
            "--normalized-stride-samples",
            bad_samples,
        )
        assert result.returncode != 0
        assert "Traceback" not in result.stderr
        assert (
            "error" in result.stderr.lower()
            or "invalid" in result.stderr.lower()
            or ">= 2" in result.stderr
        )

    def test_nonnumeric_coverage_rejected(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        result = _run_cli(
            str(d),
            "--anthropometry-sex",
            "male",
            "--minimum-mass-coverage",
            "abc",
        )
        assert result.returncode != 0
        assert "Traceback" not in result.stderr
        assert (
            "not a number" in result.stderr.lower() or "error" in result.stderr.lower()
        )

    def test_nonnumeric_samples_rejected(self, tmp_path: Path) -> None:
        d = _build_fixture(tmp_path)
        result = _run_cli(
            str(d),
            "--anthropometry-sex",
            "male",
            "--normalized-stride-samples",
            "xyz",
        )
        assert result.returncode != 0
        assert "Traceback" not in result.stderr
        assert (
            "not an integer" in result.stderr.lower()
            or "error" in result.stderr.lower()
        )
