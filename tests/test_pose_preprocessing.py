from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

import gait_stability.pose_preprocessing as pose_preprocessing
from gait_stability.pose_contracts import MEDIAPIPE_LANDMARK_NAMES
from gait_stability.pose_pipeline import FRAME_FIELDS, LANDMARK_FIELDS
from gait_stability.pose_preprocessing import (
    PROCESSED_FIELDS,
    STEP3_ARTIFACT_NAMES,
    PoseArtifactValidationError,
    PosePreprocessingConfig,
    RawPoseArtifacts,
    _publish_step3_artifacts,
    preprocess_pose,
)
from gait_stability.video_ingestion import ArtifactPublishError, sha256_file


def _write_step2_artifacts(
    directory: Path,
    timestamps: list[float],
    landmarks: list[dict[str, object]],
) -> None:
    directory.mkdir()
    counts = {index: 0 for index in range(len(timestamps))}
    for landmark in landmarks:
        counts[int(landmark["frame_index"])] += 1
    with (directory / "pose_frames.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        for index, timestamp in enumerate(timestamps):
            count = counts[index]
            writer.writerow(
                {
                    "frame_index": index,
                    "nominal_timestamp_seconds": timestamp,
                    "backend_timestamp_milliseconds": round(timestamp * 1000),
                    "status": "decoded_pose" if count else "decoded_no_pose",
                    "landmark_count": count,
                    "detail": "",
                }
            )
    with (directory / "raw_landmarks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=LANDMARK_FIELDS)
        writer.writeheader()
        for values in landmarks:
            landmark_id = int(values.get("landmark_id", 27))
            row: dict[str, object] = {
                "frame_index": values["frame_index"],
                "nominal_timestamp_seconds": timestamps[int(values["frame_index"])],
                "landmark_id": landmark_id,
                "landmark_name": MEDIAPIPE_LANDMARK_NAMES[landmark_id],
                "x_normalized": values.get("x", 0.2),
                "y_normalized": values.get("y", 0.4),
                "z_backend_relative": values.get("z", -0.1),
                "visibility": values.get("visibility", 0.9),
                "presence": values.get("presence", 0.9),
                "confidence": values.get("confidence", ""),
            }
            writer.writerow(row)
    (directory / "pose_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source": {"sha256": "synthetic-video", "nominal_fps": 10.0},
                "backend": {"backend": "synthetic", "model_sha256": "model-hash"},
                "capture_assumptions": {"input": "synthetic monocular RGB"},
                "outputs": {
                    "raw_landmarks": "raw_landmarks.csv",
                    "frame_manifest": "pose_frames.csv",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _processed_rows(path: Path, landmark_name: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert tuple(reader.fieldnames or ()) == PROCESSED_FIELDS
        return [row for row in reader if row["landmark_name"] == landmark_name]


def test_config_is_frozen_and_validates_signal_parameters() -> None:
    config = PosePreprocessingConfig()
    with pytest.raises(AttributeError):
        config.max_gap_frames = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="positive odd"):
        PosePreprocessingConfig(smoothing_window_frames=2)
    with pytest.raises(ValueError, match="between 0 and 1"):
        PosePreprocessingConfig(visibility_threshold=float("nan"))
    with pytest.raises(ValueError, match="Unknown diagnostic"):
        PosePreprocessingConfig(diagnostic_landmarks=("not_canonical",))
    with pytest.raises(TypeError, match="max_gap_frames must be an integer"):
        PosePreprocessingConfig(max_gap_frames=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="use_visibility must be a bool"):
        PosePreprocessingConfig(use_visibility=1)  # type: ignore[arg-type]


def test_preprocessing_preserves_raw_values_and_distinguishes_rejections(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(
        artifact_directory,
        [0.0, 0.1, 0.2, 0.3],
        [
            {"frame_index": 0, "x": 1.2, "visibility": 0.5, "presence": 0.5},
            {"frame_index": 1, "x": 0.3, "visibility": 0.49},
            {"frame_index": 2, "x": 0.4, "presence": ""},
            {"frame_index": 3, "x": "nan", "visibility": 0.9},
        ],
    )

    artifacts = preprocess_pose(
        artifact_directory,
        PosePreprocessingConfig(
            max_gap_frames=0, smoothing_window_frames=1, write_diagnostic=False
        ),
    )

    rows = _processed_rows(artifacts.processed_landmarks_path, "left_ankle")
    assert len(rows) == 4
    assert rows[0]["raw_x_normalized"] == "1.2"
    assert rows[0]["observed_usable"] == "true"
    assert rows[0]["out_of_image_x"] == "true"
    assert rows[0]["processed_x_normalized"] == "1.2"
    assert rows[0]["raw_z_backend_relative"] == "-0.1"
    assert rows[1]["rejected_low_confidence"] == "true"
    assert rows[1]["missing_or_nonfinite_enabled_score"] == "false"
    assert rows[2]["rejected_low_confidence"] == "false"
    assert rows[2]["missing_or_nonfinite_enabled_score"] == "true"
    assert rows[3]["nonfinite_x_coordinate"] == "true"
    assert rows[3]["nonfinite_y_coordinate"] == "false"
    assert rows[3]["x_observed_usable"] == "false"
    assert rows[3]["y_observed_usable"] == "true"
    assert rows[3]["observed_usable"] == "false"
    assert rows[3]["raw_x_normalized"] == "nan"
    assert rows[3]["pre_smoothed_x_normalized"] == ""
    assert rows[3]["pre_smoothed_y_normalized"] == "0.4"
    assert rows[3]["processed_y_normalized"] == "0.4"
    assert all(
        row["final_missing"] == ("false" if index == 0 else "true")
        for index, row in enumerate(rows)
    )
    assert artifacts.diagnostic_path is None
    assert not (artifact_directory / "pose_trajectory_diagnostic.png").exists()


def test_scalar_coordinates_remain_usable_and_interpolate_independently(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(
        artifact_directory,
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        [
            {"frame_index": 0, "x": 0.0, "y": "nan"},
            {"frame_index": 2, "x": 2.0, "y": "nan"},
            {"frame_index": 3, "x": "nan", "y": 3.0},
            {"frame_index": 5, "x": "nan", "y": 5.0},
        ],
    )

    artifacts = preprocess_pose(
        artifact_directory,
        PosePreprocessingConfig(
            max_gap_frames=1, smoothing_window_frames=1, write_diagnostic=False
        ),
    )

    rows = _processed_rows(artifacts.processed_landmarks_path, "left_ankle")
    assert rows[0]["x_observed_usable"] == "true"
    assert rows[0]["y_observed_usable"] == "false"
    assert rows[0]["observed_usable"] == "false"
    assert rows[0]["processed_x_normalized"] == "0.0"
    assert rows[0]["processed_y_normalized"] == ""
    assert rows[1]["x_interpolated"] == "true"
    assert rows[1]["processed_x_normalized"] == "1.0"
    assert rows[1]["y_interpolated"] == "false"
    assert rows[3]["x_observed_usable"] == "false"
    assert rows[3]["y_observed_usable"] == "true"
    assert rows[3]["processed_x_normalized"] == ""
    assert rows[3]["processed_y_normalized"] == "3.0"
    assert rows[4]["x_interpolated"] == "false"
    assert rows[4]["y_interpolated"] == "true"
    assert rows[4]["processed_y_normalized"] == "4.0"

    quality = json.loads(artifacts.pose_quality_path.read_text(encoding="utf-8"))
    assert quality["interpolated_fraction"] == pytest.approx(2 / (6 * 33))
    assert quality["interpolated_coordinate_fraction"] == pytest.approx(
        2 / (6 * 33 * 2)
    )
    assert quality["denominators"]["overall_coordinate_fractions"].endswith(
        "x 2 planar coordinates"
    )
    ankle = quality["per_landmark"]["left_ankle"]
    assert (
        ankle["x_coordinate_gaps"]["interpolated_gap_summary"][
            "longest_missing_sample_count"
        ]
        == 1
    )
    assert (
        ankle["y_coordinate_gaps"]["interpolated_gap_summary"][
            "longest_missing_sample_count"
        ]
        == 1
    )
    assert ankle["point_union_gaps"]["semantics"].startswith("union across x and y")


def test_adjacent_x_and_y_gaps_remain_separate_scalar_gap_runs(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(
        artifact_directory,
        [0.0, 0.1, 0.2, 0.3],
        [
            {"frame_index": 0, "x": 0.0, "y": 0.0},
            {"frame_index": 1, "x": "nan", "y": 1.0},
            {"frame_index": 2, "x": 2.0, "y": "nan"},
            {"frame_index": 3, "x": 3.0, "y": 3.0},
        ],
    )

    artifacts = preprocess_pose(
        artifact_directory,
        PosePreprocessingConfig(
            max_gap_frames=1, smoothing_window_frames=1, write_diagnostic=False
        ),
    )

    quality = json.loads(artifacts.pose_quality_path.read_text(encoding="utf-8"))
    ankle = quality["per_landmark"]["left_ankle"]
    x_gaps = ankle["x_coordinate_gaps"]["interpolated_gaps"]
    y_gaps = ankle["y_coordinate_gaps"]["interpolated_gaps"]
    assert [(gap["start_frame_index"], gap["end_frame_index"]) for gap in x_gaps] == [
        (1, 1)
    ]
    assert [(gap["start_frame_index"], gap["end_frame_index"]) for gap in y_gaps] == [
        (2, 2)
    ]
    assert x_gaps[0]["start_timestamp_seconds"] == 0.1
    assert x_gaps[0]["end_timestamp_seconds"] == 0.1
    assert x_gaps[0]["nominal_sample_span_seconds"] == 0.0
    assert (
        ankle["x_coordinate_gaps"]["interpolated_gap_summary"][
            "longest_missing_sample_count"
        ]
        == 1
    )
    assert (
        ankle["y_coordinate_gaps"]["interpolated_gap_summary"][
            "longest_missing_sample_count"
        ]
        == 1
    )
    assert (
        ankle["point_union_gaps"]["interpolated_gap_summary"][
            "longest_missing_sample_count"
        ]
        == 2
    )


def test_short_gap_uses_timestamps_but_long_and_boundary_gaps_remain_missing(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(
        artifact_directory,
        [0.0, 0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [
            {"frame_index": 0, "x": 0.0, "y": 0.0},
            {"frame_index": 3, "x": 1.0, "y": 2.0},
            {"frame_index": 7, "x": 2.0, "y": 4.0},
        ],
    )

    artifacts = preprocess_pose(
        artifact_directory,
        PosePreprocessingConfig(
            max_gap_frames=2, smoothing_window_frames=1, write_diagnostic=False
        ),
    )

    rows = _processed_rows(artifacts.processed_landmarks_path, "left_ankle")
    assert float(rows[1]["pre_smoothed_x_normalized"]) == pytest.approx(0.2)
    assert float(rows[2]["pre_smoothed_x_normalized"]) == pytest.approx(0.8)
    assert rows[1]["x_interpolated"] == "true"
    assert rows[2]["y_interpolated"] == "true"
    assert all(rows[index]["final_missing"] == "true" for index in (4, 5, 6))
    missing_landmark_rows = _processed_rows(
        artifacts.processed_landmarks_path, "right_ankle"
    )
    assert all(row["final_missing"] == "true" for row in missing_landmark_rows)
    assert all(row["x_interpolated"] == "false" for row in missing_landmark_rows)
    quality = json.loads(artifacts.pose_quality_path.read_text(encoding="utf-8"))
    ankle = quality["per_landmark"]["left_ankle"]
    assert ankle["interpolated_count"] == 2
    assert ankle["interpolated_gap_summary"]["longest_missing_sample_count"] == 2
    assert (
        ankle["x_coordinate_gaps"]["interpolated_gap_summary"][
            "longest_missing_sample_count"
        ]
        == 2
    )
    assert ankle["remaining_missing_count"] == 3
    assert ankle["remaining_missing_gap_summary"]["longest_missing_sample_count"] == 3
    assert ankle["interpolated_gaps"][0][
        "nominal_sample_span_seconds"
    ] == pytest.approx(0.3)
    assert ankle["interpolated_gaps"][0][
        "bracketing_duration_seconds"
    ] == pytest.approx(0.5)


def test_low_confidence_raw_point_is_retained_and_interpolated_with_both_flags(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(
        artifact_directory,
        [0.0, 0.1, 0.2],
        [
            {"frame_index": 0, "x": 0.0},
            {"frame_index": 1, "x": 0.9, "visibility": 0.1},
            {"frame_index": 2, "x": 0.4},
        ],
    )

    artifacts = preprocess_pose(
        artifact_directory,
        PosePreprocessingConfig(
            max_gap_frames=1, smoothing_window_frames=1, write_diagnostic=False
        ),
    )

    rejected = _processed_rows(artifacts.processed_landmarks_path, "left_ankle")[1]
    assert rejected["raw_row_present"] == "true"
    assert rejected["raw_x_normalized"] == "0.9"
    assert rejected["visibility"] == "0.1"
    assert rejected["rejected_low_confidence"] == "true"
    assert rejected["observed_usable"] == "false"
    assert rejected["x_interpolated"] == "true"
    assert rejected["y_interpolated"] == "true"
    assert float(rejected["processed_x_normalized"]) == pytest.approx(0.2)
    assert rejected["final_missing"] == "false"


def test_leading_and_trailing_boundary_gaps_are_never_extrapolated(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(
        artifact_directory,
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        [{"frame_index": 2, "x": 0.2}, {"frame_index": 3, "x": 0.3}],
    )

    artifacts = preprocess_pose(
        artifact_directory,
        PosePreprocessingConfig(
            max_gap_frames=3, smoothing_window_frames=1, write_diagnostic=False
        ),
    )

    rows = _processed_rows(artifacts.processed_landmarks_path, "left_ankle")
    for index in (0, 1, 4, 5):
        assert rows[index]["processed_x_normalized"] == ""
        assert rows[index]["processed_y_normalized"] == ""
        assert rows[index]["x_interpolated"] == "false"
        assert rows[index]["y_interpolated"] == "false"
        assert rows[index]["final_missing"] == "true"

    quality = json.loads(artifacts.pose_quality_path.read_text(encoding="utf-8"))
    gaps = quality["per_landmark"]["left_ankle"]["remaining_missing_gaps"]
    assert [(gap["start_frame_index"], gap["end_frame_index"]) for gap in gaps] == [
        (0, 1),
        (4, 5),
    ]
    assert all(gap["bracketing_duration_seconds"] is None for gap in gaps)


def test_centered_smoothing_uses_symmetric_segment_support_and_records_flags(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(
        artifact_directory,
        [0.0, 0.05, 0.2, 0.3, 0.4, 0.5, 0.6],
        [
            {"frame_index": 0, "x": 0.0},
            {"frame_index": 2, "x": 2.0},
            {"frame_index": 5, "x": 0.0},
            {"frame_index": 6, "x": 2.0},
        ],
    )

    artifacts = preprocess_pose(
        artifact_directory,
        PosePreprocessingConfig(
            max_gap_frames=1, smoothing_window_frames=3, write_diagnostic=False
        ),
    )

    rows = _processed_rows(artifacts.processed_landmarks_path, "left_ankle")
    assert float(rows[1]["pre_smoothed_x_normalized"]) == pytest.approx(0.5)
    assert float(rows[1]["processed_x_normalized"]) == pytest.approx(5 / 6)
    assert rows[1]["x_interpolated"] == "true"
    assert rows[1]["x_smoothing_changed"] == "true"
    assert rows[1]["x_smoothing_support_contains_interpolation"] == "true"
    assert rows[0]["processed_x_normalized"] == "0.0"
    assert rows[3]["final_missing"] == "true"
    assert rows[4]["final_missing"] == "true"
    assert rows[5]["processed_x_normalized"] == "0.0"
    assert rows[6]["processed_x_normalized"] == "2.0"


def test_smoothing_reduces_high_frequency_jitter_without_crossing_long_gap(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "walk"
    timestamps = [index * 0.1 for index in range(13)]
    observed_indices = (*range(5), *range(8, 13))
    landmarks = [
        {
            "frame_index": index,
            "x": 0.4 + (0.1 if index % 2 else -0.1),
            "y": 0.6 + (0.1 if index % 2 else -0.1),
        }
        for index in observed_indices
    ]
    _write_step2_artifacts(artifact_directory, timestamps, landmarks)
    raw_hashes = {
        name: sha256_file(artifact_directory / name)
        for name in ("raw_landmarks.csv", "pose_frames.csv", "pose_metadata.json")
    }

    artifacts = preprocess_pose(
        artifact_directory,
        PosePreprocessingConfig(
            max_gap_frames=2, smoothing_window_frames=3, write_diagnostic=False
        ),
    )

    rows = _processed_rows(artifacts.processed_landmarks_path, "left_ankle")
    assert [float(row["nominal_timestamp_seconds"]) for row in rows] == timestamps
    assert [int(row["frame_index"]) for row in rows] == list(range(len(timestamps)))
    assert len(rows) == len(timestamps)
    for coordinate in ("x", "y"):
        raw = [float(rows[index][f"raw_{coordinate}_normalized"]) for index in range(5)]
        processed = [
            float(rows[index][f"processed_{coordinate}_normalized"])
            for index in range(5)
        ]
        raw_total_variation = math.fsum(
            abs(raw[index + 1] - raw[index]) for index in range(len(raw) - 1)
        )
        processed_total_variation = math.fsum(
            abs(processed[index + 1] - processed[index])
            for index in range(len(processed) - 1)
        )
        assert processed_total_variation < raw_total_variation
    for index in (5, 6, 7):
        assert rows[index]["processed_x_normalized"] == ""
        assert rows[index]["processed_y_normalized"] == ""
        assert rows[index]["final_missing"] == "true"
    assert all(
        float(rows[index]["raw_x_normalized"])
        == pytest.approx(0.4 + (0.1 if index % 2 else -0.1))
        for index in observed_indices
    )
    assert all(rows[index]["raw_x_normalized"] == "" for index in (5, 6, 7))
    assert {
        name: sha256_file(artifact_directory / name) for name in raw_hashes
    } == raw_hashes


def test_quality_report_has_exact_counts_fractions_and_gap_lengths(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(
        artifact_directory,
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        [
            {"frame_index": 0, "x": 0.0},
            {"frame_index": 1, "x": 0.9, "visibility": 0.1},
            {"frame_index": 3, "x": 0.6},
        ],
    )

    artifacts = preprocess_pose(
        artifact_directory,
        PosePreprocessingConfig(
            max_gap_frames=2, smoothing_window_frames=1, write_diagnostic=False
        ),
    )

    quality = json.loads(artifacts.pose_quality_path.read_text(encoding="utf-8"))
    ankle = quality["per_landmark"]["left_ankle"]
    assert ankle["raw_row_count"] == 3
    assert ankle["raw_row_coverage"] == pytest.approx(1 / 2)
    assert ankle["usable_observation_count"] == 2
    assert ankle["usable_observation_coverage"] == pytest.approx(1 / 3)
    assert ankle["low_confidence_rejection_count"] == 1
    assert ankle["low_confidence_rejection_fraction_of_raw_rows"] == pytest.approx(
        1 / 3
    )
    assert ankle["low_confidence_rejection_coverage"] == pytest.approx(1 / 6)
    assert ankle["initial_missing_count"] == 4
    assert ankle["initial_missing_coverage"] == pytest.approx(2 / 3)
    assert ankle["interpolated_count"] == 2
    assert ankle["interpolation_coverage"] == pytest.approx(1 / 3)
    assert ankle["remaining_missing_count"] == 2
    assert ankle["remaining_missing_coverage"] == pytest.approx(1 / 3)
    assert ankle["initial_missing_gap_summary"]["count"] == 2
    assert ankle["initial_missing_gap_summary"]["longest_missing_sample_count"] == 2
    assert ankle["interpolated_gap_summary"]["longest_missing_sample_count"] == 2
    assert ankle["remaining_missing_gap_summary"]["longest_missing_sample_count"] == 2
    assert quality["interpolated_fraction"] == pytest.approx(2 / (6 * 33))
    assert quality["remaining_missing_fraction"] == pytest.approx(
        (6 * 33 - 4) / (6 * 33)
    )


def test_zero_raw_rows_have_explicit_null_raw_row_fractions(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(artifact_directory, [0.0], [])

    artifacts = preprocess_pose(
        artifact_directory, PosePreprocessingConfig(write_diagnostic=False)
    )

    quality = json.loads(artifacts.pose_quality_path.read_text(encoding="utf-8"))
    ankle = quality["per_landmark"]["left_ankle"]
    assert ankle["raw_row_count"] == 0
    assert ankle["low_confidence_rejection_fraction_of_raw_rows"] is None
    assert ankle["missing_or_nonfinite_enabled_score_fraction_of_raw_rows"] is None


def test_enabled_generic_confidence_rejects_null_but_disabled_scores_are_ignored(
    tmp_path: Path,
) -> None:
    disabled_directory = tmp_path / "disabled"
    landmarks = [
        {
            "frame_index": 0,
            "visibility": "",
            "presence": "",
            "confidence": "",
        },
        {
            "frame_index": 1,
            "visibility": "",
            "presence": "",
            "confidence": 0.4,
        },
    ]
    _write_step2_artifacts(disabled_directory, [0.0, 0.1], landmarks)
    disabled = preprocess_pose(
        disabled_directory,
        PosePreprocessingConfig(
            use_visibility=False,
            use_presence=False,
            use_confidence=False,
            smoothing_window_frames=1,
            write_diagnostic=False,
        ),
    )
    disabled_rows = _processed_rows(disabled.processed_landmarks_path, "left_ankle")
    assert all(row["observed_usable"] == "true" for row in disabled_rows)
    assert all(
        row["missing_or_nonfinite_enabled_score"] == "false" for row in disabled_rows
    )
    assert all(row["rejected_low_confidence"] == "false" for row in disabled_rows)

    enabled_directory = tmp_path / "enabled"
    _write_step2_artifacts(enabled_directory, [0.0, 0.1], landmarks)
    enabled = preprocess_pose(
        enabled_directory,
        PosePreprocessingConfig(
            use_visibility=False,
            use_presence=False,
            use_confidence=True,
            max_gap_frames=0,
            smoothing_window_frames=1,
            write_diagnostic=False,
        ),
    )
    enabled_rows = _processed_rows(enabled.processed_landmarks_path, "left_ankle")
    assert enabled_rows[0]["missing_or_nonfinite_enabled_score"] == "true"
    assert enabled_rows[0]["rejected_low_confidence"] == "false"
    assert enabled_rows[1]["missing_or_nonfinite_enabled_score"] == "false"
    assert enabled_rows[1]["rejected_low_confidence"] == "true"
    metadata = json.loads(
        enabled.preprocessing_metadata_path.read_text(encoding="utf-8")
    )
    assert metadata["confidence_semantics"]["enabled_fields"] == ["confidence"]


def test_complete_grid_quality_and_metadata_are_auditable(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(
        artifact_directory,
        [0.0, 0.1],
        [{"frame_index": 0, "x": 0.2}, {"frame_index": 1, "x": 0.4}],
    )
    input_hash = sha256_file(artifact_directory / "raw_landmarks.csv")

    artifacts = preprocess_pose(
        artifact_directory,
        PosePreprocessingConfig(smoothing_window_frames=1, write_diagnostic=False),
    )

    with artifacts.processed_landmarks_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2 * 33
    assert sum(row["raw_row_present"] == "true" for row in rows) == 2
    quality = json.loads(artifacts.pose_quality_path.read_text(encoding="utf-8"))
    assert list(quality)[:2] == ["schema_version", "semantics"]
    semantics = quality["semantics"]
    assert "heuristic planar usability" in semantics["observed_usable"]
    assert "not positional or anatomical accuracy" in semantics["observed_usable"]
    assert "bounded interpolation" in semantics["processed_completeness"]
    assert "same nominal frame" in semantics["simultaneous_all_12"]
    assert semantics["legacy_aliases"]["fields"] == [
        "frames_with_all_12_required_gait_landmarks",
        "required_landmark_coverage",
    ]
    assert "observed_usable" in semantics["legacy_aliases"]["ambiguity"]
    assert (
        "not be interpreted as processed completeness or accuracy"
        in semantics["legacy_aliases"]["ambiguity"]
    )
    assert quality["total_frames"] == 2
    assert quality["pose_detected_frames"] == 2
    assert quality["frames_with_all_12_required_gait_landmarks"] == 0
    assert quality["required_landmark_coverage"] == 0.0
    assert quality["frames_with_all_required_landmarks_observed_usable"] == 0
    assert quality["frames_with_all_required_landmarks_processed_complete"] == 0
    assert quality["interpolated_fraction"] == 0.0
    assert quality["remaining_missing_fraction"] == pytest.approx(64 / 66)
    assert quality["interpolated_coordinate_fraction"] == 0.0
    assert quality["remaining_missing_coordinate_fraction"] == pytest.approx(128 / 132)
    assert "quality_label" not in quality
    metadata = json.loads(
        artifacts.preprocessing_metadata_path.read_text(encoding="utf-8")
    )
    assert metadata["algorithm_version"] == "step3-mvp-1"
    assert metadata["inputs"]["raw_landmarks.csv"]["sha256"] == input_hash
    assert metadata["inherited_provenance"]["backend"]["model_sha256"] == "model-hash"
    assert metadata["config"]["use_confidence"] is False
    score_fields = metadata["confidence_semantics"]["fields"]
    assert "associated with landmark visibility" in score_fields["visibility"]
    assert "associated with landmark presence" in score_fields["presence"]
    for score_name in ("visibility", "presence"):
        assert (
            "not calibrated accuracy, probability, or ground truth"
            in score_fields[score_name]
        )
    assert metadata["coordinates"]["z_backend_relative"].startswith("preserved raw")
    assert metadata["interpolation"]["maximum_missing_samples"] == 3
    assert metadata["smoothing"]["configured_window_frames"] == 1
    assert (
        "no fixed group delay under uniform sampling" in metadata["smoothing"]["phase"]
    )
    for unsupported_preservation_claim in (
        "extrema",
        "threshold crossings",
        "derivatives",
        "gait-event timing",
    ):
        assert unsupported_preservation_claim in metadata["smoothing"]["phase"]
    assert metadata["outputs"]["processed_landmarks.csv"]["sha256"] == sha256_file(
        artifacts.processed_landmarks_path
    )
    assert metadata["outputs"]["preprocessing_metadata.json"]["sha256"] is None
    assert set(PROCESSED_FIELDS) >= {
        "raw_row_present",
        "observed_usable",
        "rejected_low_confidence",
        "x_interpolated",
        "y_interpolated",
        "x_final_missing",
        "y_final_missing",
        "final_missing",
    }
    for name in ("raw_landmarks.csv", "pose_frames.csv", "pose_metadata.json"):
        assert metadata["inputs"][name]["sha256"] == sha256_file(
            artifact_directory / name
        )
    assert metadata["config"] == {
        "visibility_threshold": 0.5,
        "presence_threshold": 0.5,
        "confidence_threshold": 0.5,
        "use_visibility": True,
        "use_presence": True,
        "use_confidence": False,
        "max_gap_frames": 3,
        "smoothing_window_frames": 1,
        "diagnostic_landmarks": [
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_hip",
            "right_hip",
        ],
        "write_diagnostic": False,
    }


def test_successful_publication_without_diagnostic_removes_stale_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(artifact_directory, [0.0], [{"frame_index": 0}])
    step2_hashes = {
        name: sha256_file(artifact_directory / name)
        for name in ("raw_landmarks.csv", "pose_frames.csv", "pose_metadata.json")
    }
    stale = artifact_directory / "pose_trajectory_diagnostic.png"
    stale.write_text("stale diagnostic")
    staging_directories: list[Path] = []
    original_mkdtemp = pose_preprocessing.tempfile.mkdtemp

    def capture_staging_directory(*args: object, **kwargs: object) -> str:
        staging = original_mkdtemp(*args, **kwargs)  # type: ignore[arg-type]
        staging_directories.append(Path(staging))
        return staging

    monkeypatch.setattr(
        pose_preprocessing.tempfile, "mkdtemp", capture_staging_directory
    )

    artifacts = preprocess_pose(
        artifact_directory,
        PosePreprocessingConfig(smoothing_window_frames=1, write_diagnostic=False),
    )

    assert artifacts.diagnostic_path is None
    assert not stale.exists()
    assert {
        name: sha256_file(artifact_directory / name) for name in step2_hashes
    } == step2_hashes
    assert len(staging_directories) == 1
    assert staging_directories[0].name.startswith("walk.preprocessing-staging-")
    assert not staging_directories[0].name.startswith(".")


def test_step3_publication_failure_restores_prior_complete_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "destination"
    staging = tmp_path / "staging"
    destination.mkdir()
    staging.mkdir()
    for name in STEP3_ARTIFACT_NAMES:
        (destination / name).write_text(f"old {name}", encoding="utf-8")
        (staging / name).write_text(f"new {name}", encoding="utf-8")
    step2_names = ("raw_landmarks.csv", "pose_frames.csv", "pose_metadata.json")
    for name in step2_names:
        (destination / name).write_text(f"step 2 {name}", encoding="utf-8")
    original_replace = Path.replace
    backup_targets: list[Path] = []

    def fail_publish(path: Path, target: Path) -> Path:
        if ".backup-" in target.name:
            backup_targets.append(target)
        if path.parent == staging and path.name == "pose_quality.json":
            raise OSError("synthetic failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publish)

    with pytest.raises(ArtifactPublishError, match="atomically publish"):
        _publish_step3_artifacts(staging, destination)

    assert all(
        (destination / name).read_text(encoding="utf-8") == f"old {name}"
        for name in STEP3_ARTIFACT_NAMES
    )
    assert all(
        (destination / name).read_text(encoding="utf-8") == f"step 2 {name}"
        for name in step2_names
    )
    assert backup_targets
    assert all(not path.name.startswith(".") for path in backup_targets)


def test_input_mutation_before_publication_preserves_prior_step3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(artifact_directory, [0.0], [{"frame_index": 0, "x": 0.2}])
    old = artifact_directory / "processed_landmarks.csv"
    old.write_text("old step 3", encoding="utf-8")
    source = artifact_directory / "raw_landmarks.csv"
    original_metadata_payload = pose_preprocessing._metadata_payload

    def mutate_after_load(*args: object, **kwargs: object) -> object:
        payload = original_metadata_payload(*args, **kwargs)
        source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return payload

    monkeypatch.setattr(
        "gait_stability.pose_preprocessing._metadata_payload", mutate_after_load
    )

    with pytest.raises(
        PoseArtifactValidationError, match="changed during preprocessing"
    ):
        preprocess_pose(
            artifact_directory, PosePreprocessingConfig(write_diagnostic=False)
        )

    assert old.read_text(encoding="utf-8") == "old step 3"


def test_publication_reports_incomplete_rollback_and_backup_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "destination"
    staging = tmp_path / "staging"
    destination.mkdir()
    staging.mkdir()
    for name in STEP3_ARTIFACT_NAMES:
        (destination / name).write_text(f"old {name}", encoding="utf-8")
        (staging / name).write_text(f"new {name}", encoding="utf-8")
    original_replace = Path.replace

    def fail_publish_and_restore(path: Path, target: Path) -> Path:
        if path.parent == staging and path.name == "pose_quality.json":
            raise OSError("synthetic publish failure")
        # Regression: generated backup basenames must not begin with '.'
        if ".backup-" in path.name and target.name == "processed_landmarks.csv":
            raise OSError("synthetic restore failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publish_and_restore)

    with pytest.raises(
        ArtifactPublishError, match="restoration may be incomplete"
    ) as error:
        _publish_step3_artifacts(staging, destination)

    assert "processed_landmarks.csv" in str(error.value)
    # Regression: error message should contain 'backup-' (visible backup name)
    assert "backup-" in str(error.value)
    # Regression: generated backup basenames must not begin with '.'
    backup_files = list(destination.glob("processed_landmarks.csv.backup-*"))
    # Regression: backup basenames must not begin with "."
    assert not any(Path(b).name.startswith(".") for b in backup_files)


@pytest.mark.parametrize(
    ("metadata_update", "expected_error"),
    [
        ({"schema_version": None}, "schema_version must be exactly 2"),
        ({"schema_version": 1}, "schema_version must be exactly 2"),
        (
            {"schemas": {"raw_landmarks.csv": {"columns": ["not", "canonical"]}}},
            "embedded columns conflict for raw_landmarks.csv",
        ),
    ],
)
def test_explicit_conflicting_pose_metadata_contract_is_rejected(
    tmp_path: Path, metadata_update: dict[str, object], expected_error: str
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(artifact_directory, [0.0], [{"frame_index": 0}])
    metadata_path = artifact_directory / "pose_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(metadata_update)
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    with pytest.raises(PoseArtifactValidationError, match=expected_error):
        preprocess_pose(
            artifact_directory, PosePreprocessingConfig(write_diagnostic=False)
        )


@pytest.mark.parametrize(
    ("malformation", "expected_error"),
    [("short", "missing value"), ("extra", "extra columns")],
)
def test_malformed_csv_rows_raise_expected_validation_error(
    tmp_path: Path, malformation: str, expected_error: str
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(artifact_directory, [0.0], [{"frame_index": 0}])
    raw_path = artifact_directory / "raw_landmarks.csv"
    lines = raw_path.read_text(encoding="utf-8").splitlines()
    if malformation == "short":
        lines[-1] = lines[-1].rsplit(",", maxsplit=1)[0]
    else:
        lines[-1] += ",unexpected"
    raw_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(PoseArtifactValidationError, match=expected_error):
        preprocess_pose(
            artifact_directory, PosePreprocessingConfig(write_diagnostic=False)
        )


def test_custom_input_paths_cannot_overlap_each_other_or_step3_outputs(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(artifact_directory, [0.0], [{"frame_index": 0}])
    raw_path = artifact_directory / "raw_landmarks.csv"
    metadata_path = artifact_directory / "pose_metadata.json"

    overlapping_inputs = RawPoseArtifacts(
        artifact_directory=artifact_directory,
        raw_landmarks_path=raw_path,
        pose_frames_path=raw_path,
        pose_metadata_path=metadata_path,
    )
    with pytest.raises(PoseArtifactValidationError, match="paths overlap"):
        preprocess_pose(
            overlapping_inputs, PosePreprocessingConfig(write_diagnostic=False)
        )

    overlapping_output = RawPoseArtifacts(
        artifact_directory=artifact_directory,
        raw_landmarks_path=artifact_directory / "processed_landmarks.csv",
        pose_frames_path=artifact_directory / "pose_frames.csv",
        pose_metadata_path=metadata_path,
    )
    with pytest.raises(PoseArtifactValidationError, match="paths overlap"):
        preprocess_pose(
            overlapping_output, PosePreprocessingConfig(write_diagnostic=False)
        )


@pytest.mark.parametrize(
    ("malformation", "expected_error"),
    [
        ("pose_frames_header", "schema mismatch"),
        ("metadata_root", "root must be an object"),
        ("invalid_json", "valid JSON"),
    ],
)
def test_malformed_csv_or_metadata_is_rejected(
    tmp_path: Path, malformation: str, expected_error: str
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(artifact_directory, [0.0], [{"frame_index": 0}])
    if malformation == "pose_frames_header":
        frames_path = artifact_directory / "pose_frames.csv"
        lines = frames_path.read_text(encoding="utf-8").splitlines()
        lines[0] = "wrong,header"
        frames_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif malformation == "metadata_root":
        (artifact_directory / "pose_metadata.json").write_text("[]\n", encoding="utf-8")
    else:
        (artifact_directory / "pose_metadata.json").write_text(
            "{not valid json\n", encoding="utf-8"
        )

    with pytest.raises(PoseArtifactValidationError, match=expected_error):
        preprocess_pose(
            artifact_directory, PosePreprocessingConfig(write_diagnostic=False)
        )


def test_invalid_source_relationship_fails_before_replacing_existing_step3(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(artifact_directory, [0.0], [{"frame_index": 0}])
    raw_path = artifact_directory / "raw_landmarks.csv"
    lines = raw_path.read_text(encoding="utf-8").splitlines()
    raw_path.write_text("\n".join([*lines, lines[-1]]) + "\n", encoding="utf-8")
    old_output = artifact_directory / "processed_landmarks.csv"
    old_output.write_text("old step 3", encoding="utf-8")

    with pytest.raises(PoseArtifactValidationError, match="duplicate frame/landmark"):
        preprocess_pose(
            artifact_directory, PosePreprocessingConfig(write_diagnostic=False)
        )

    assert old_output.read_text(encoding="utf-8") == "old step 3"


def test_optional_diagnostic_is_generated_and_hashed_when_matplotlib_is_available(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    artifact_directory = tmp_path / "walk"
    _write_step2_artifacts(artifact_directory, [0.0], [{"frame_index": 0}])

    artifacts = preprocess_pose(
        artifact_directory,
        PosePreprocessingConfig(
            smoothing_window_frames=1,
            diagnostic_landmarks=("left_ankle",),
        ),
    )

    assert artifacts.diagnostic_path is not None
    assert artifacts.diagnostic_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    metadata = json.loads(
        artifacts.preprocessing_metadata_path.read_text(encoding="utf-8")
    )
    assert metadata["outputs"]["pose_trajectory_diagnostic.png"][
        "sha256"
    ] == sha256_file(artifacts.diagnostic_path)
