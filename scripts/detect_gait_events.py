#!/usr/bin/env python3
"""Command-line entry point for Step 4 candidate gait-event artifacts."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from gait_stability import (
    ArtifactPublishError,
    GaitEventConfig,
    GaitEventPipelineConfig,
    GaitEventPipelineError,
    detect_gait_events,
)


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> int:
    defaults = GaitEventConfig()
    parser = argparse.ArgumentParser(
        description="Create Step 4 candidate gait-event and stride review artifacts."
    )
    parser.add_argument("artifact_directory", type=Path)
    parser.add_argument(
        "--walking-direction",
        choices=("image_right", "image_left"),
        required=True,
        help="Manually established forward direction in the stored image",
    )
    parser.add_argument("--manual-start-frame", type=_nonnegative_integer)
    parser.add_argument("--manual-end-frame", type=_nonnegative_integer)
    parser.add_argument(
        "--video", type=Path, help="Override inherited source video path"
    )
    parser.add_argument(
        "--peak-radius-frames",
        type=_positive_integer,
        default=defaults.peak_radius_frames,
    )
    parser.add_argument(
        "--prominence-window-frames",
        type=_positive_integer,
        default=defaults.prominence_window_frames,
    )
    parser.add_argument(
        "--min-prominence", type=_nonnegative_float, default=defaults.min_prominence
    )
    parser.add_argument(
        "--reversal-half-window-frames",
        type=_positive_integer,
        default=defaults.reversal_half_window_frames,
    )
    parser.add_argument(
        "--derivative-deadband",
        type=_nonnegative_float,
        default=defaults.derivative_deadband,
    )
    parser.add_argument(
        "--min-forward-relative-x",
        type=_finite_float,
        default=defaults.min_forward_relative_x,
    )
    parser.add_argument(
        "--raw-agreement-window-frames",
        type=_nonnegative_integer,
        default=defaults.raw_agreement_window_frames,
    )
    parser.add_argument(
        "--ankle-agreement-window-frames",
        type=_nonnegative_integer,
        default=defaults.ankle_agreement_window_frames,
    )
    parser.add_argument(
        "--same-side-min-interval-seconds",
        type=_nonnegative_float,
        default=defaults.same_side_min_interval_seconds,
    )
    parser.add_argument(
        "--opposite-side-min-interval-seconds",
        type=_nonnegative_float,
        default=defaults.opposite_side_min_interval_seconds,
    )
    parser.add_argument(
        "--same-side-max-interval-warning-seconds",
        type=_positive_float,
        default=defaults.same_side_max_interval_warning_seconds,
    )
    parser.add_argument(
        "--automatic-minimum-bout-duration-seconds", type=_positive_float, default=3.0
    )
    parser.add_argument(
        "--automatic-minimum-accepted-events-per-side",
        type=_positive_integer,
        default=2,
    )
    parser.add_argument(
        "--event-flash-radius-frames", type=_nonnegative_integer, default=2
    )
    args = parser.parse_args()
    try:
        event_config = GaitEventConfig(
            direction=args.walking_direction,
            peak_radius_frames=args.peak_radius_frames,
            prominence_window_frames=args.prominence_window_frames,
            min_prominence=args.min_prominence,
            reversal_half_window_frames=args.reversal_half_window_frames,
            derivative_deadband=args.derivative_deadband,
            min_forward_relative_x=args.min_forward_relative_x,
            raw_agreement_window_frames=args.raw_agreement_window_frames,
            ankle_agreement_window_frames=args.ankle_agreement_window_frames,
            same_side_min_interval_seconds=args.same_side_min_interval_seconds,
            opposite_side_min_interval_seconds=args.opposite_side_min_interval_seconds,
            same_side_max_interval_warning_seconds=args.same_side_max_interval_warning_seconds,
        )
        config = GaitEventPipelineConfig(
            event_config=event_config,
            manual_start_frame=args.manual_start_frame,
            manual_end_frame=args.manual_end_frame,
            automatic_minimum_bout_duration_seconds=args.automatic_minimum_bout_duration_seconds,
            automatic_minimum_accepted_events_per_side=args.automatic_minimum_accepted_events_per_side,
            event_flash_radius_frames=args.event_flash_radius_frames,
        )
        artifacts = detect_gait_events(
            args.artifact_directory, config, video_path=args.video
        )
    except (ArtifactPublishError, GaitEventPipelineError, OSError, ValueError) as exc:
        print(f"Gait-event processing failed: {exc}", file=sys.stderr)
        return 1
    print(artifacts.metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
