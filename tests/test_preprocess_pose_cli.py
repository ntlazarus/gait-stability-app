from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gait_stability.pose_preprocessing import PoseArtifactValidationError
from scripts import preprocess_pose as cli


@pytest.mark.parametrize("text", ["0", "0.5", "1"])
def test_probability_accepts_finite_inclusive_values(text: str) -> None:
    assert cli._probability(text) == float(text)


@pytest.mark.parametrize("text", ["-0.1", "1.1", "nan", "inf"])
def test_probability_rejects_invalid_values(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        cli._probability(text)


def test_cli_builds_config_and_prints_metadata_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact_directory = tmp_path / "walk"
    metadata_path = artifact_directory / "preprocessing_metadata.json"
    calls: list[tuple[Path, object]] = []

    def fake_preprocess(path: Path, config: object) -> object:
        calls.append((path, config))
        return SimpleNamespace(preprocessing_metadata_path=metadata_path)

    monkeypatch.setattr(cli, "preprocess_pose", fake_preprocess)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocess_pose.py",
            str(artifact_directory),
            "--visibility-threshold",
            "0.6",
            "--disable-presence",
            "--enable-confidence",
            "--confidence-threshold",
            "0.4",
            "--max-gap-frames",
            "2",
            "--smoothing-window-frames",
            "5",
            "--diagnostic-landmarks",
            "left_ankle,right_ankle",
            "--no-diagnostic",
        ],
    )

    assert cli.main() == 0
    assert len(calls) == 1
    path, config = calls[0]
    assert path == artifact_directory
    assert config.visibility_threshold == 0.6  # type: ignore[attr-defined]
    assert config.use_presence is False  # type: ignore[attr-defined]
    assert config.use_confidence is True  # type: ignore[attr-defined]
    assert config.max_gap_frames == 2  # type: ignore[attr-defined]
    assert config.smoothing_window_frames == 5  # type: ignore[attr-defined]
    assert config.diagnostic_landmarks == (  # type: ignore[attr-defined]
        "left_ankle",
        "right_ankle",
    )
    assert config.write_diagnostic is False  # type: ignore[attr-defined]
    captured = capsys.readouterr()
    assert captured.out == f"{metadata_path}\n"
    assert captured.err == ""


def test_cli_reports_expected_error_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*_args: object) -> None:
        raise PoseArtifactValidationError("synthetic invalid artifacts")

    monkeypatch.setattr(cli, "preprocess_pose", fail)
    monkeypatch.setattr(sys, "argv", ["preprocess_pose.py", str(tmp_path / "walk")])

    assert cli.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Pose preprocessing failed: synthetic invalid artifacts\n"
    assert "Traceback" not in captured.err
