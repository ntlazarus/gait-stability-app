#!/usr/bin/env python3
"""Command-line entry point for Step 5 COM proxy estimation."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from gait_stability.com_estimation import ComEstimationConfig
from gait_stability.com_pipeline import ComPipelineError, estimate_com
from gait_stability.video_ingestion import ArtifactPublishError


def _positive_int(value: str) -> int:
    """Argparse type converter for positive integers."""
    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if ivalue < 2:
        raise argparse.ArgumentTypeError(f"must be >= 2, got {value!r}")
    return ivalue


def _coverage_float(value: str) -> float:
    """Argparse type converter for coverage threshold."""
    try:
        fvalue = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
    if not math.isfinite(fvalue):
        raise argparse.ArgumentTypeError(f"{value!r} is not finite")
    if not 0.0 <= fvalue <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value!r}")
    return fvalue


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate segment-weighted anthropometric COM proxy "
        "from processed pose landmarks."
    )
    parser.add_argument(
        "artifact_directory",
        type=Path,
        help=(
            "Directory containing processed_landmarks.csv, "
            "preprocessing_metadata.json, pose_frames.csv, "
            "reviewed_gait_events.csv, reviewed_strides.csv, and "
            "review_resolution_metadata.json"
        ),
    )
    parser.add_argument(
        "--anthropometry-sex",
        required=True,
        choices=["male", "female"],
        help="Anthropometric sex for de Leva coefficients (required)",
    )
    parser.add_argument(
        "--minimum-mass-coverage",
        type=_coverage_float,
        default=0.90,
        dest="minimum_mass_coverage",
        help="Minimum mass coverage threshold (default: 0.90)",
    )
    parser.add_argument(
        "--normalized-stride-samples",
        type=_positive_int,
        default=101,
        dest="normalized_stride_samples",
        help="Number of normalized stride samples (default: 101)",
    )
    args = parser.parse_args()

    try:
        config = ComEstimationConfig(
            anthropometry_sex=args.anthropometry_sex,
            minimum_mass_coverage=args.minimum_mass_coverage,
            normalized_stride_samples=args.normalized_stride_samples,
        )
        artifacts = estimate_com(args.artifact_directory, config)
    except (
        ArtifactPublishError,
        ComPipelineError,
        OSError,
        ValueError,
        TypeError,
    ) as exc:
        print(f"COM estimation failed: {exc}", file=sys.stderr)
        return 1
    print(artifacts.com_metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
