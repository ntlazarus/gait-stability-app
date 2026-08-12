from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gait_stability import VideoOpenError
from scripts import inspect_video as cli


def test_cli_success_prints_metadata_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "walk.mp4"
    output_root = tmp_path / "artifacts"
    metadata_path = output_root / "walk" / "video_metadata.json"
    calls: list[tuple[Path, Path]] = []

    def fake_inspect_video(input_path: Path, output_path: Path) -> SimpleNamespace:
        calls.append((input_path, output_path))
        return SimpleNamespace(metadata_path=metadata_path)

    monkeypatch.setattr(cli, "inspect_video", fake_inspect_video)
    monkeypatch.setattr(
        sys,
        "argv",
        ["inspect_video.py", str(source), "--output-root", str(output_root)],
    )

    assert cli.main() == 0
    assert calls == [(source, output_root)]
    captured = capsys.readouterr()
    assert captured.out == f"{metadata_path}\n"
    assert captured.err == ""


def test_cli_expected_error_returns_one_and_reports_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "broken.mp4"

    def fail_inspection(_input_path: Path, _output_path: Path) -> None:
        raise VideoOpenError("fake decoder could not open input")

    monkeypatch.setattr(cli, "inspect_video", fail_inspection)
    monkeypatch.setattr(sys, "argv", ["inspect_video.py", str(source)])

    assert cli.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Video inspection failed: fake decoder could not open input\n"
    )
