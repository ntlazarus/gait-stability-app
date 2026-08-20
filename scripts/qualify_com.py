#!/usr/bin/env python3
"""Command-line entry point for Step 5b COM qualification."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from gait_stability import (
    ArtifactPublishError,
    ComQualificationConfig,
    ComQualificationPipelineError,
    qualify_com,
)


def _coverage_thresholds(value: str) -> tuple[float, ...]:
    """Parse comma-separated coverage thresholds."""
    parts = value.split(",")
    thresholds = []
    for part in parts:
        part = part.strip()
        if not part:
            raise argparse.ArgumentTypeError("empty threshold value")
        try:
            th = float(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"'{part}' is not a number") from exc
        if not math.isfinite(th):
            raise argparse.ArgumentTypeError(f"'{part}' is not finite")
        if not 0.0 <= th <= 1.0:
            raise argparse.ArgumentTypeError(f"'{part}' must be between 0 and 1")
        thresholds.append(th)

    # Validate uniqueness and strict increase
    seen = set()
    prev = None
    for _i, th in enumerate(thresholds):
        if th in seen:
            raise argparse.ArgumentTypeError(f"duplicate threshold value: {th}")
        if prev is not None and th <= prev:
            raise argparse.ArgumentTypeError(
                f"thresholds must be strictly increasing: {prev} -> {th}"
            )
        seen.add(th)
        prev = th

    if not thresholds:
        raise argparse.ArgumentTypeError("at least one threshold required")

    return tuple(thresholds)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Step 5b COM coverage qualification on Step 5a artifacts."
    )
    parser.add_argument(
        "artifact_directory",
        type=Path,
        help="Directory containing Step 5a outputs and upstream inputs",
    )
    parser.add_argument(
        "--coverage-thresholds",
        type=_coverage_thresholds,
        default=(0.80, 0.82, 0.84, 0.86, 0.88, 0.90),
        help="Comma-separated absolute mass_coverage thresholds for sensitivity grid "
        "(default: 0.80,0.82,0.84,0.86,0.88,0.90)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Override source video path (must match inherited provenance hash)",
    )
    args = parser.parse_args()

    try:
        config = ComQualificationConfig(coverage_thresholds=args.coverage_thresholds)
        artifacts = qualify_com(args.artifact_directory, config, video_path=args.video)
    except (
        ArtifactPublishError,
        ComQualificationPipelineError,
        OSError,
        ValueError,
        TypeError,
    ) as exc:
        print(f"COM qualification failed: {exc}", file=sys.stderr)
        return 1

    print(artifacts.qualification_json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
