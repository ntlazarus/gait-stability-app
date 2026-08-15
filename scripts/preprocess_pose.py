#!/usr/bin/env python3
"""Command-line entry point for Step 3 pose preprocessing."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from gait_stability import (
    ArtifactPublishError,
    PosePreprocessingConfig,
    PosePreprocessingError,
    preprocess_pose,
)


def _probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be finite and between 0 and 1 inclusive")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_odd_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed % 2 == 0:
        raise argparse.ArgumentTypeError("must be a positive odd integer")
    return parsed


def _landmark_list(value: str) -> tuple[str, ...]:
    landmarks = tuple(name.strip() for name in value.split(",") if name.strip())
    if not landmarks:
        raise argparse.ArgumentTypeError("must contain at least one landmark name")
    return landmarks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quality-assess and minimally preprocess Step 2 pose artifacts."
    )
    parser.add_argument(
        "artifact_directory",
        type=Path,
        help=(
            "Directory containing raw_landmarks.csv, pose_frames.csv, and "
            "pose_metadata.json"
        ),
    )
    parser.add_argument("--visibility-threshold", type=_probability, default=0.5)
    parser.add_argument("--presence-threshold", type=_probability, default=0.5)
    parser.add_argument("--confidence-threshold", type=_probability, default=0.5)
    parser.add_argument("--disable-visibility", action="store_true")
    parser.add_argument("--disable-presence", action="store_true")
    parser.add_argument("--enable-confidence", action="store_true")
    parser.add_argument("--max-gap-frames", type=_nonnegative_integer, default=3)
    parser.add_argument(
        "--smoothing-window-frames", type=_positive_odd_integer, default=3
    )
    parser.add_argument(
        "--diagnostic-landmarks",
        type=_landmark_list,
        default=None,
        help="Comma-separated canonical names; defaults to ankles, heels, and hips",
    )
    parser.add_argument("--no-diagnostic", action="store_true")
    args = parser.parse_args()
    config_kwargs = {
        "visibility_threshold": args.visibility_threshold,
        "presence_threshold": args.presence_threshold,
        "confidence_threshold": args.confidence_threshold,
        "use_visibility": not args.disable_visibility,
        "use_presence": not args.disable_presence,
        "use_confidence": args.enable_confidence,
        "max_gap_frames": args.max_gap_frames,
        "smoothing_window_frames": args.smoothing_window_frames,
        "write_diagnostic": not args.no_diagnostic,
    }
    if args.diagnostic_landmarks is not None:
        config_kwargs["diagnostic_landmarks"] = args.diagnostic_landmarks
    try:
        config = PosePreprocessingConfig(**config_kwargs)
        artifacts = preprocess_pose(args.artifact_directory, config)
    except (ArtifactPublishError, OSError, PosePreprocessingError, ValueError) as exc:
        print(f"Pose preprocessing failed: {exc}", file=sys.stderr)
        return 1
    print(artifacts.preprocessing_metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
