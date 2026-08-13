from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gait_stability import VideoOpenError
from gait_stability.mediapipe_pose import MediaPipePoseError
from scripts import estimate_pose as cli


@pytest.mark.parametrize("text", ["0", "0.25", "1"])
def test_probability_parser_accepts_inclusive_unit_interval(text: str) -> None:
    assert cli._probability(text) == float(text)


@pytest.mark.parametrize("text", ["-0.01", "1.01", "nan", "inf", "-inf"])
def test_probability_parser_rejects_non_probabilities(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="between 0 and 1"):
        cli._probability(text)


def test_pose_cli_passes_thresholds_and_prints_metadata_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "walk.mp4"
    model = tmp_path / "pose.task"
    output_root = tmp_path / "results"
    metadata_path = output_root / "walk" / "pose_metadata.json"
    estimator = object()
    constructor_calls: list[tuple[Path, dict[str, float]]] = []
    pipeline_calls: list[tuple[Path, object, Path]] = []

    def fake_constructor(path: Path, **thresholds: float) -> object:
        constructor_calls.append((path, thresholds))
        return estimator

    def fake_pipeline(input_path: Path, backend: object, output: Path) -> Any:
        pipeline_calls.append((input_path, backend, output))
        return SimpleNamespace(pose_metadata_path=metadata_path)

    monkeypatch.setattr(cli, "MediaPipePoseEstimator", fake_constructor)
    monkeypatch.setattr(cli, "estimate_pose_video", fake_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "estimate_pose.py",
            str(source),
            "--model",
            str(model),
            "--output-root",
            str(output_root),
            "--min-pose-detection-confidence",
            "0.2",
            "--min-pose-presence-confidence",
            "0.3",
            "--min-tracking-confidence",
            "0.4",
        ],
    )

    assert cli.main() == 0
    assert constructor_calls == [
        (
            model,
            {
                "min_pose_detection_confidence": 0.2,
                "min_pose_presence_confidence": 0.3,
                "min_tracking_confidence": 0.4,
            },
        )
    ]
    assert pipeline_calls == [(source, estimator, output_root)]
    captured = capsys.readouterr()
    assert captured.out == f"{metadata_path}\n"
    assert captured.err == ""


def test_pose_cli_expected_error_returns_one_and_reports_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli, "MediaPipePoseEstimator", lambda _path, **_kwargs: object()
    )

    def fail_pipeline(*_args: object) -> None:
        raise VideoOpenError("synthetic decoder failure")

    monkeypatch.setattr(cli, "estimate_pose_video", fail_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "estimate_pose.py",
            str(tmp_path / "broken.mp4"),
            "--model",
            str(tmp_path / "pose.task"),
        ],
    )

    assert cli.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Pose estimation failed: synthetic decoder failure\n"


def test_pose_cli_reports_mediapipe_initialization_error_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_constructor(_path: Path, **_kwargs: float) -> None:
        raise MediaPipePoseError("synthetic backend initialization failure")

    monkeypatch.setattr(cli, "MediaPipePoseEstimator", fail_constructor)
    monkeypatch.setattr(
        cli,
        "estimate_pose_video",
        lambda *_args: pytest.fail(
            "pipeline must not run after initialization failure"
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "estimate_pose.py",
            str(tmp_path / "walk.mp4"),
            "--model",
            str(tmp_path / "pose.task"),
        ],
    )

    assert cli.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Pose estimation failed: synthetic backend initialization failure\n"
    )
    assert "Traceback" not in captured.err
