#!/usr/bin/env python3
"""Command-line entry point for Step 4b gait review resolution.

Resolves manual review and correction of automatic gait-event and stride
detections, producing reviewed_gait_events.csv, reviewed_strides.csv, and
review_resolution_metadata.json in the artifact directory.
"""

from __future__ import annotations

import sys

from gait_stability.review_resolution import (
    ReviewResolutionError,
    main_cli,
)


def main() -> int:
    try:
        return main_cli()
    except ReviewResolutionError as exc:
        print(f"Review resolution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
