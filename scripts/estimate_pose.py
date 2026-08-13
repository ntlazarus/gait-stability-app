#!/usr/bin/env python3
"""Command-line entry point for raw MediaPipe pose estimation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gait_stability import PosePipelineError, VideoInspectionError, estimate_pose_video
from gait_stability.mediapipe_pose import MediaPipePoseError, MediaPipePoseEstimator


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1 inclusive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate raw pose and generate canonical CSV/annotated artifacts."
    )
    parser.add_argument("input", type=Path, help="Path to the source MP4/MOV")
    parser.add_argument(
        "--model", type=Path, required=True, help="Pose Landmarker .task path"
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--min-pose-detection-confidence", type=_probability, default=0.5
    )
    parser.add_argument(
        "--min-pose-presence-confidence", type=_probability, default=0.5
    )
    parser.add_argument("--min-tracking-confidence", type=_probability, default=0.5)
    args = parser.parse_args()
    try:
        estimator = MediaPipePoseEstimator(
            args.model,
            min_pose_detection_confidence=args.min_pose_detection_confidence,
            min_pose_presence_confidence=args.min_pose_presence_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        artifacts = estimate_pose_video(args.input, estimator, args.output_root)
    except (
        FileNotFoundError,
        ImportError,
        MediaPipePoseError,
        OSError,
        PosePipelineError,
        VideoInspectionError,
    ) as exc:
        print(f"Pose estimation failed: {exc}", file=sys.stderr)
        return 1
    print(artifacts.pose_metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
