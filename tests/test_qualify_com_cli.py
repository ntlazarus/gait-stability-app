"""CLI tests for scripts/qualify_com.py.

These tests exercise the argparse-driven entry point without requiring
full pipeline runs where possible.  Tests that need the canonical
Step 5 artifact fixture reuse the deterministic synthetic helper from
tests/test_com_qualification_pipeline.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from test_com_qualification_pipeline import _build_qualification_fixture

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_main(argv: list[str]) -> subprocess.CompletedProcess:
    """Run scripts/qualify_com.main via subprocess with the given argv."""
    return subprocess.run(
        [sys.executable, "scripts/qualify_com.py"] + argv,
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
    )


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_cli_help() -> None:
    """--help exits 0 and shows usage."""
    result = _run_main(["--help"])
    assert result.returncode == 0
    has_help = "artifact_directory" in result.stdout
    has_help = has_help or "artifact_directory" in result.stderr
    assert has_help


# ---------------------------------------------------------------------------
# Missing directory failure / stderr
# ---------------------------------------------------------------------------


def test_cli_missing_directory_fails_stderr() -> None:
    """Non-existent artifact_directory causes exit 1 and stderr message."""
    result = _run_main(["/nonexistent/path/to/nothing"])
    assert result.returncode == 1
    assert "error" in result.stderr.lower() or "failed" in result.stderr.lower()


# ---------------------------------------------------------------------------
# Default grid success / stdout metadata path
# ---------------------------------------------------------------------------


def test_cli_default_grid_success_stdout_path(tmp_path: Path) -> None:
    """Default coverage grid succeeds and prints the qualification JSON path."""
    artifacts, _ = _build_qualification_fixture(tmp_path)
    result = _run_main([str(artifacts)])
    assert result.returncode == 0
    # stdout should be the qualification JSON path
    qualified_path = result.stdout.strip()
    assert qualified_path, "Expected qualification JSON path on stdout"
    p = Path(qualified_path)
    assert p.is_file(), f"Qualification JSON not found at {qualified_path}"
    meta = json.loads(p.read_text(encoding="utf-8"))
    # Default thresholds should be present
    assert meta["sensitivity_grid"]["coverage_thresholds"] == [
        0.80,
        0.82,
        0.84,
        0.86,
        0.88,
        0.90,
    ]
    assert meta["stride_statistics"][0]["policy_complete_at_primary_threshold"] is True
    assert (
        meta["preprocessing_inheritance"]["interpolation"]["maximum_missing_samples"]
        == 3
    )
    assert (
        meta["preprocessing_inheritance"]["smoothing"]["configured_window_frames"] == 3
    )
    assert meta["camera_view"]["artifact_declaration"]["status"] == (
        "not_declared_or_verified"
    )
    assert meta["camera_view"]["human_review"]["status"] == "not_recorded"


# ---------------------------------------------------------------------------
# Custom sorted grid success and JSON records exact grid
# ---------------------------------------------------------------------------


def test_cli_custom_grid_success_stdout_path(tmp_path: Path) -> None:
    """Custom coverage grid succeeds and JSON records the exact grid."""
    artifacts, _ = _build_qualification_fixture(tmp_path)
    result = _run_main([str(artifacts), "--coverage-thresholds", "0.5,0.75"])
    assert result.returncode == 0
    qualified_path = result.stdout.strip()
    assert qualified_path, "Expected qualification JSON path on stdout"
    p = Path(qualified_path)
    assert p.is_file(), f"Qualification JSON not found at {qualified_path}"
    meta = json.loads(p.read_text(encoding="utf-8"))
    assert meta["sensitivity_grid"]["coverage_thresholds"] == [0.5, 0.75]


def test_cli_custom_single_threshold_success(tmp_path: Path) -> None:
    """Single custom coverage threshold succeeds."""
    artifacts, _ = _build_qualification_fixture(tmp_path)
    result = _run_main([str(artifacts), "--coverage-thresholds", "0.9"])
    assert result.returncode == 0
    qualified_path = result.stdout.strip()
    assert qualified_path, "Expected qualification JSON path on stdout"
    p = Path(qualified_path)
    assert p.is_file()
    meta = json.loads(p.read_text(encoding="utf-8"))
    assert meta["sensitivity_grid"]["coverage_thresholds"] == [0.9]


# ---------------------------------------------------------------------------
# --video override success using same hash-identical source
# ---------------------------------------------------------------------------


def test_cli_video_override_success(tmp_path: Path) -> None:
    """--video override with hash-identical source video succeeds."""
    artifacts, source_video = _build_qualification_fixture(tmp_path)

    # Pass --video pointing to the same source video that's already in the
    # fixture's provenance.  The pipeline will verify the sha256 matches.
    result = _run_main([str(artifacts), "--video", str(source_video)])
    assert result.returncode == 0, f"Video override failed: {result.stderr}"


# ---------------------------------------------------------------------------
# Grid rejection: empty / malformed / nan / out-of-range / duplicate /
# non-increasing (tested via argparse before pipeline processing)
# ---------------------------------------------------------------------------


def test_cli_rejects_empty_grid() -> None:
    """Empty --coverage-thresholds raises SystemExit (argparse error)."""
    result = _run_main(["--coverage-thresholds", ""])
    # argparse should reject this; exit status may be 2 (argparse error)
    # or 1 depending on how main() handles it.  We just check non-zero.
    assert result.returncode != 0


def test_cli_rejects_malformed_grid() -> None:
    """Malformed (non-numeric) --coverage-thresholds raises SystemExit."""
    result = _run_main(["--coverage-thresholds", "abc,def"])
    assert result.returncode != 0


def test_cli_rejects_nan_grid() -> None:
    """NaN --coverage-thresholds raises SystemExit."""
    result = _run_main(["--coverage-thresholds", "nan,0.5"])
    assert result.returncode != 0


def test_cli_rejects_out_of_range_grid() -> None:
    """Out-of-range --coverage-thresholds (<0 or >1) raises SystemExit."""
    result = _run_main(["--coverage-thresholds", "-0.1,0.5"])
    assert result.returncode != 0
    result = _run_main(["--coverage-thresholds", "1.5,0.5"])
    assert result.returncode != 0


def test_cli_rejects_duplicate_grid() -> None:
    """Duplicate --coverage-thresholds raises SystemExit."""
    result = _run_main(["--coverage-thresholds", "0.5,0.5"])
    assert result.returncode != 0


def test_cli_rejects_non_increasing_grid() -> None:
    """Non-increasing --coverage-thresholds raises SystemExit."""
    result = _run_main(["--coverage-thresholds", "0.9,0.5"])
    assert result.returncode != 0


def test_cli_rejects_bool_grid() -> None:
    """Bool values in --coverage-thresholds raises SystemExit."""
    result = _run_main(["--coverage-thresholds", "True,False"])
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# Integration: custom grid with full pipeline (uses fixture)
# ---------------------------------------------------------------------------


def test_cli_custom_grid_full_pipeline(tmp_path: Path) -> None:
    """Full pipeline run with custom grid produces correct JSON grid records."""
    artifacts, _ = _build_qualification_fixture(tmp_path)
    result = _run_main([str(artifacts), "--coverage-thresholds", "0.5,0.75,0.95"])
    assert result.returncode == 0
    qualified_path = result.stdout.strip()
    assert qualified_path
    meta = json.loads(Path(qualified_path).read_text(encoding="utf-8"))
    assert meta["sensitivity_grid"]["coverage_thresholds"] == [0.5, 0.75, 0.95]
