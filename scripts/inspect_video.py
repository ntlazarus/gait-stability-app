#!/usr/bin/env python3
"""Command-line entry point for deterministic video inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gait_stability import VideoInspectionError, inspect_video


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect an MP4/MOV and extract representative frames."
    )
    parser.add_argument("input", type=Path, help="Path to the source video")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Artifact root directory (default: outputs)",
    )
    args = parser.parse_args()

    try:
        metadata = inspect_video(args.input, args.output_root)
    except (VideoInspectionError, OSError) as exc:
        print(f"Video inspection failed: {exc}", file=sys.stderr)
        return 1

    print(metadata.metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
