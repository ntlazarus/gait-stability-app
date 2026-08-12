# Project State

## Snapshot metadata

- Snapshot UTC: `2026-08-12T21:29:28Z`
- Git branch: `main`
- HEAD: `a25c66b959cc0484baaa73c48190935d2c3123ff` (`a25c66b chore: add application state snapshot workflow`)
- Working tree: dirty; `README.md`, `src/gait_stability/video_ingestion.py`,
  and `tests/test_video_ingestion.py` are modified, and `docs/` is untracked

## What the project currently is

The executable application is MVP Step 1: a local Python workflow that inspects
an `.mp4` or `.mov` video, records file and decoder metadata, and writes
deterministically selected decoded frame samples.

It is not yet a gait-analysis pipeline. Pose estimation, gait events,
center-of-mass estimation, speed, gait or stability metrics, user interfaces,
APIs, clinical usability, and scientific or clinical validation are not
implemented.

## Current capabilities

- Public Python API:
  `gait_stability.inspect_video(video_path, output_root=Path("outputs"))`.
- CLI: `python scripts/inspect_video.py INPUT [--output-root OUTPUTS]`.
- Validates the input path, supported extension, OpenCV opening, positive frame
  dimensions, positive nominal FPS, and a positive integer frame count.
- Rejects an input nested inside its deterministic output destination.
- Records source provenance: resolved path, byte size, SHA-256 digest, and
  inspection time.
- Records OpenCV backend and version, an extension-derived container indicator,
  orientation metadata/status, and whether disabling OpenCV auto-orientation
  was accepted.
- Records readability/decode status, requested and actual sampling positions,
  requested and actual nominal timestamps, fallback distance, image path, and
  seek method.
- Deterministically requests positions at 0%, 25%, 50%, 75%, and 100% over
  `frame_count - 1`, using Python rounding and ordered de-duplication for short
  videos.
- Attempts an exact decode first. On failure, it tries up to 10 immediately
  preceding indices in descending order on the same OpenCV capture.
- Writes JPEG samples as `frame_<zero-padded actual source index>.jpg`. If
  fallback reaches an actual frame already written for another request, the
  image is reused while both requested targets retain metadata records.
- If neither the requested index nor its bounded fallback range decodes, the
  workflow reports an explicit error and does not publish staged output.
- Stages all artifacts before replacing an existing same-stem destination and
  rolls back reported publish failures.

## Quick start

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/inspect_video.py INPUT.mp4
```

Python 3.11 or newer is required. Replace `INPUT.mp4` with a local, non-sensitive
`.mp4` or `.mov` file. The default destination is `outputs/`.

## Key workflows

### Inspect a local video with the CLI

Purpose: validate and characterize a supported video and save representative
decoded frames.

```bash
python scripts/inspect_video.py INPUT.mp4 --output-root outputs
```

- Required input: one readable local `.mp4` or `.mov` video.
- Important option: `--output-root` changes the artifact root; it defaults to
  `outputs`.
- Success: the command exits successfully and creates
  `outputs/<video-stem>/video_metadata.json` plus JPEGs under
  `outputs/<video-stem>/sample_frames/`.
- Failure: validation or decoding errors are reported explicitly and produce a
  nonzero exit status.

### Inspect a local video from Python

```python
from pathlib import Path

from gait_stability import inspect_video

metadata = inspect_video(Path("INPUT.mov"), output_root=Path("outputs"))
```

This invokes the same inspection and artifact-writing behavior as the CLI and
returns the inspection metadata.

## Inputs

- Supported formats are selected by filename extension: `.mp4` and `.mov`.
- The file must exist, open through OpenCV, and expose valid dimensions,
  nominal FPS, and frame count.
- No participant data is bundled or required. Use appropriately governed local
  files; `data/` is ignored by Git.
- No camera calibration, scale, camera view, gait events, or anthropometric
  information is consumed because biomechanical measurements are not
  implemented.
- Input cannot be located inside the deterministic destination
  `<output-root>/<video-stem>/`.

## Outputs

For an input stem `walk`, the workflow writes:

```text
<output-root>/walk/
├── video_metadata.json
└── sample_frames/
    └── frame_<zero-padded actual source index>.jpg
```

`video_metadata.json` records source provenance, container indicator, decoder
and orientation behavior, dimensions, nominal timing, readability/decode
status, and sampling/seek details. Each `sampled_frames` entry includes
`requested_frame_index`, actual `frame_index`,
`requested_nominal_timestamp_seconds`, actual `nominal_timestamp_seconds`,
`fallback_distance_frames`, `relative_image_path`, and `decode_status`. Sample
JPEGs are decoded image observations named by actual frame index; multiple
requested targets can refer to one reused image.

These outputs describe observed file, container, and OpenCV backend properties
and decoded frame samples. They are not pose-model estimates, derived
biomechanical measurements, experimental stability proxies, or validated
clinical measurements.

Artifacts are fully staged before destination replacement. Existing output for
the same stem is replaced. Different input files with the same stem collide
under one output root.

## Configuration

- `pyproject.toml` defines package metadata, Python compatibility, runtime and
  development dependencies, and Ruff, mypy, and pytest settings.
- Runtime dependency: `opencv-python-headless`.
- Development dependencies: `mypy`, `pytest`, and `ruff`.
- Mypy targets Python 3.13 because current NumPy/OpenCV stubs use newer syntax;
  package runtime metadata permits Python 3.11 and newer.
- The primary runtime setting is the API `output_root` argument or CLI
  `--output-root` option.
- `README.md` contains usage and limitations. `AGENTS.md` defines scientific
  and repository workflow constraints.

## Testing and verification

Normal verification from the repository root:

```bash
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/python -m mypy src scripts
.venv/bin/python -m pytest
```

Fresh snapshot evidence:

- Ruff formatting passed for 27 files.
- Ruff linting passed.
- Pytest passed 47 tests, including a generated synthetic MP4 through the real
  OpenCV path.
- Mypy passed for 3 source files in a fresh clean environment.
- The Git diff check passed.
- `python scripts/inspect_video.py --help` passed in the inspected environment.
- A missing-file CLI smoke test produced the expected explicit error and
  nonzero status.
- No real participant video was processed by agents. A user-reported decode
  failure at frame 300 motivated the fallback change, but that video has not
  been confirmed as successfully rerun.

## Known limitations

- Duration and sample timestamps are nominal values derived from frame count
  and FPS, not actual presentation timestamps.
- Each exact or fallback attempt re-seeks on the same OpenCV capture. A failed
  decode can leave some backends unrecoverable even when an earlier frame might
  otherwise be readable.
- `CAP_PROP_POS_FRAMES` checking establishes decoded-frame identity only when
  backend position reporting is meaningful; it does not guarantee accurate
  presentation timing.
- OpenCV auto-orientation may be unsupported or rejected. The application does
  not explicitly rotate frames; dimensions and samples reflect backend decoded
  output behavior.
- A process interruption between the two renames used while replacing an
  existing destination can leave a hidden `.backup-*` sibling that requires
  manual recovery.
- Same-stem videos collide under one output root, and prior same-stem output is
  replaced.
- Container identification is inferred from the extension, not a full container
  probe.
- There is no pose quality assessment, landmark confidence, gait event
  detection, calibration, coordinate normalization, or biomechanical analysis.
- There is no UI, service API, clinical workflow, or scientific/clinical
  validation. Outputs must not be interpreted as clinical conclusions.

## Repository map

```text
src/gait_stability/   Reusable package and public inspection API
scripts/              Thin executable CLI entry points
tests/                Unit and integration tests, including synthetic video tests
docs/PROJECT_STATE.md Canonical current-application snapshot
.opencode/            OpenCode commands, agents, and skills
outputs/              Ignored generated inspection artifacts
data/                 Ignored local data; participant data must not be committed
```

## Next logical capabilities

The following are not implemented:

- Pose landmark extraction with confidence and quality preservation.
- Configurable pose quality control, bounded interpolation, and smoothing.
- Validated gait-event detection and biomechanical coordinate handling.
- Clearly defined and validated gait and candidate stability metrics.
- Research reporting, overlays, UI/API access, and validation against reference
  measurements.
