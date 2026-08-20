#!/usr/bin/env python3
"""Command-line entry point for Step 5c clean-capture qualification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gait_stability import (
    ArtifactPublishError,
    CaptureQualificationError,
    qualify_clean_capture,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qualify a clean capture from current and prior Step 5b evidence."
    )
    parser.add_argument(
        "artifact_directory",
        type=Path,
        help="Directory containing the current canonical com_qualification.json",
    )
    parser.add_argument(
        "capture_review",
        type=Path,
        help="External clean-capture review JSON",
    )
    parser.add_argument(
        "--prior-qualification",
        type=Path,
        required=True,
        help="Prior comparable Step 5b com_qualification.json",
    )
    args = parser.parse_args()
    try:
        artifacts = qualify_clean_capture(
            args.artifact_directory,
            args.capture_review,
            args.prior_qualification,
        )
    except (
        CaptureQualificationError,
        ArtifactPublishError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Capture qualification failed: {exc}", file=sys.stderr)
        return 1
    print(artifacts.qualification_json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
