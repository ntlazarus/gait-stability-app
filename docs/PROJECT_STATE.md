# Project State

## Snapshot metadata

- Snapshot UTC: `2026-08-14T16:44:54Z`
- Git branch: `main`
- HEAD: `e4df68d` (`chore: support for fallback OpenCode models`)
- Working tree: dirty; the Step 3 implementation and related documentation are
  uncommitted
- Verification environment: Python `3.13.5` at
  `/tmp/opencode/gait-step3-venv`

## What the project currently is

The executable MVP is a local, research-only Python pipeline with three stages:

1. Step 1 inspects MP4/MOV video and saves metadata plus representative frames.
2. Step 2 estimates raw monocular pose with MediaPipe Tasks Pose Landmarker 1.0.0.
3. Step 3 validates, quality-gates, interpolates bounded gaps, smooths planar
   trajectories, and produces auditable quality and provenance artifacts.

The pipeline produces video observations, unvalidated pose-model estimates, and
processed normalized trajectories. It does not produce validated gait events,
physical measurements, stability metrics, diagnoses, or clinical conclusions.

## Current capabilities

### Step 1: video inspection

- CLI: `python scripts/inspect_video.py INPUT [--output-root OUTPUTS]`.
- API: `gait_stability.inspect_video(video_path, output_root)`.
- Accepts local `.mp4` and `.mov` files and validates decoder opening, dimensions,
  nominal FPS, and nominal integer frame count.
- Records source hash/size, OpenCV/backend and orientation status, nominal timing,
  and frames requested at 0%, 25%, 50%, 75%, and 100% of the nominal range.
- A failed representative decode uses at most 10 preceding frame indices and
  records requested and actual indices. Complete staged output replaces the
  same-stem destination through unique, non-dot same-directory staging and backup
  paths.

### Step 2: raw pose estimation

- CLI: `python scripts/estimate_pose.py INPUT --model MODEL.task [options]`.
- API: `gait_stability.estimate_pose_video(video_path, estimator, output_root)`
  through the backend-independent `PoseEstimator` contract.
- Uses MediaPipe Tasks 1.0.0 in CPU `VIDEO` mode with one pose and an explicit
  local model. It reuses matching Step 1 metadata or reruns inspection.
- Processes every nominal frame slot as `decoded_pose`, `decoded_no_pose`, or
  `decode_failure`; annotated output preserves nominal order and uses black
  placeholders for decode failures.
- Preserves 33 canonical landmark rows with normalized image `x`/`y`, learned
  backend-relative `z`, visibility, presence, and nullable generic confidence.
  World landmarks, segmentation masks, and audio are omitted.

### Step 3: pose quality and preprocessing

- CLI: `python scripts/preprocess_pose.py ARTIFACT_DIRECTORY [options]`.
- API: `gait_stability.preprocess_pose(directory_or_contract, config)`.
- Requires canonical Step 2 schema version 2 artifacts. It validates schemas,
  frame/landmark relationships, distinct input/output paths, and input hashes.
- Builds a complete nominal-frame by 33-landmark audit grid while retaining raw
  values and quality scores unchanged.
- By default, finite visibility and presence must each be `>=0.5`; generic
  confidence is disabled. These are engineering heuristics, not calibrated
  accuracy cutoffs.
- Processes normalized image-plane `x` and `y` independently. It linearly
  interpolates only interior scalar gaps of at most 3 frames when both endpoints
  are raw-observed usable; it never extrapolates, recursively interpolates, or
  processes `z` or confidence.
- Applies a centered, unweighted, 3-frame boxcar independently within contiguous
  nonmissing scalar segments. Window `1` disables smoothing. Segment endpoints
  retain one-sample support.
- Reports raw usability, rejections, interpolation, smoothing support, remaining
  missingness, required-landmark simultaneous coverage, gap runs, hashes, runtime,
  configuration, and inherited provenance. It assigns no quality label.
- Publishes the Step 3 set through unique, non-dot same-directory staging and
  backup paths, with rollback on reported rename failure, and preserves the Step
  2 inputs unchanged.

## Quick start

Python 3.11 or newer is supported. For the headless MediaPipe 1.0.0 environment,
run from the repository root in this order:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,pose]'
.venv/bin/python -m pip install --no-deps mediapipe==1.0.0
.venv/bin/python -m pip install ./compat/opencv-contrib-python-headless-provider
.venv/bin/python -m pip check
```

MediaPipe declares GUI OpenCV, so `--no-deps` plus the local metadata-only
provider is required to retain the actual headless OpenCV implementation. Obtain
the full Pose Landmarker `.task` manually as documented in `models/README.md`;
the application never downloads a model.

Run the complete current workflow:

```bash
.venv/bin/python scripts/inspect_video.py path/to/walk.mp4 --output-root outputs
.venv/bin/python scripts/estimate_pose.py path/to/walk.mp4 \
  --model models/pose_landmarker_full.task --output-root outputs
.venv/bin/python scripts/preprocess_pose.py outputs/walk
```

## Key workflows

### Inspect a video

```bash
.venv/bin/python scripts/inspect_video.py path/to/walk.mp4 \
  --output-root outputs
```

- Input: one readable local `.mp4` or `.mov` outside its deterministic output
  destination.
- Options: `--output-root` defaults to `outputs`.
- Outputs: `outputs/<video-stem>/video_metadata.json` and JPEG observations under
  `sample_frames/`.
- Success: the metadata path is printed after all requested samples are decoded
  exactly or through recorded bounded fallback.

### Estimate raw pose

```bash
.venv/bin/python scripts/estimate_pose.py path/to/walk.mp4 \
  --model models/pose_landmarker_full.task \
  --output-root outputs \
  --min-pose-detection-confidence 0.5 \
  --min-pose-presence-confidence 0.5 \
  --min-tracking-confidence 0.5
```

- Inputs: a Step 1-compatible video and existing Pose Landmarker `.task` file.
- Options: all three backend thresholds default to `0.5` and accept `[0,1]`;
  `--output-root` defaults to `outputs`.
- Outputs: `raw_landmarks.csv`, `pose_frames.csv`, `pose_metadata.json`, and
  `annotated_pose.mp4` beside retained Step 1 artifacts.
- Success: the pose metadata path is printed after every nominal slot is recorded
  and all four Step 2 artifacts are published through unique, non-dot
  same-directory staging and backup paths.

### Quality-assess and preprocess pose

```bash
.venv/bin/python scripts/preprocess_pose.py outputs/walk \
  --visibility-threshold 0.5 \
  --presence-threshold 0.5 \
  --max-gap-frames 3 \
  --smoothing-window-frames 3
```

- Input: one artifact directory containing canonical `raw_landmarks.csv`,
  `pose_frames.csv`, and schema-version-2 `pose_metadata.json`.
- Options: disable visibility/presence gates with `--disable-visibility` or
  `--disable-presence`; enable the generic confidence gate with
  `--enable-confidence`; select plot rows with `--diagnostic-landmarks`; suppress
  plotting with `--no-diagnostic`. Run `--help` for all controls.
- Outputs: `processed_landmarks.csv`, `pose_quality.json`,
  `preprocessing_metadata.json`, and normally
  `pose_trajectory_diagnostic.png` in the same artifact directory.
- Success: the preprocessing metadata path is printed after input hashes are
  rechecked and the complete Step 3 set is published.

## Inputs

- Video selection is extension-based (`.mp4`/`.mov`); actual codec/container
  support, seeking, frame positions, and orientation behavior depend on OpenCV.
- Different videos with the same stem share an output destination. A source
  nested inside its own destination is rejected.
- Step 2 uses monocular RGB and an explicit ignored local `.task` model. It does
  not require or verify camera view, gait direction, placement, or mirroring.
- No camera calibration, physical scale, distortion correction, anthropometry,
  multi-view reconstruction, or gait events are inputs.
- Step 3 accepts only the canonical Step 2 CSV/JSON contract. Finite normalized
  coordinates outside `[0,1]` remain usable but are flagged; nonfinite values are
  unusable. Nominal timestamps are not verified presentation timestamps.

## Outputs

For an input stem `walk`:

```text
outputs/walk/
|-- video_metadata.json
|-- sample_frames/*.jpg
|-- raw_landmarks.csv
|-- pose_frames.csv
|-- pose_metadata.json
|-- annotated_pose.mp4
|-- processed_landmarks.csv
|-- pose_quality.json
|-- preprocessing_metadata.json
`-- pose_trajectory_diagnostic.png
```

- Direct observations: decoded Step 1 JPEGs and source/decoder metadata.
- Pose-model estimates: Step 2 landmark CSV and annotated video. `x`/`y` are
  dimensionless image-plane estimates; `z` is learned model-relative depth.
- Derived research data: Step 3's complete audit grid, heuristic usability flags,
  bounded interpolants, smoothed normalized trajectories, descriptive quality
  report, diagnostic plot, and provenance metadata.
- No output is a validated physical measurement, gait event, gait/stability
  metric, fall-risk estimate, diagnosis, or clinical result.

## Configuration

- `pyproject.toml`: Python `>=3.11`; headless OpenCV; optional pose/plot and dev
  dependencies; Ruff, mypy, and pytest settings. Mypy parses as Python 3.13.
- `requirements-pose-headless.txt` and
  `compat/opencv-contrib-python-headless-provider/`: reproducible headless
  MediaPipe dependency setup.
- `models/README.md`: manual model acquisition and hash/size provenance.
- Step 2 defaults: detection, pose presence, and tracking thresholds `0.5`.
- Step 3 defaults: visibility `0.5`, presence `0.5`, generic confidence disabled,
  maximum gap `3` frames, centered boxcar window `3`, and diagnostic enabled for
  ankles, heels, and hips.

## Testing and verification

Normal verification from the repository root:

```bash
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/python -m mypy src scripts
.venv/bin/python -m pytest
.venv/bin/python -m pip check
git diff --check
```

Fresh session evidence using Python `3.13.5` at
`/tmp/opencode/gait-step3-venv`:

- Ruff formatting passed for 46 files; Ruff lint passed.
- Mypy passed for 9 source files.
- Pytest passed all 130 tests in 8.97 seconds.
- `pip check` reported no broken requirements.
- `git diff --check` passed.

Tests cover Step 1 validation/decoding/publication, Step 2 contracts/backend/CLI
and artifact handling, and Step 3 schema validation, independent scalar gating,
bounded interpolation, smoothing, quality/provenance semantics, diagnostics,
input preservation, CLI behavior, and publication rollback.

### Aggregate Step 3 exercise

Only aggregate, non-identifying evidence from the ignored `outputs/A_Video`
exercise is recorded:

- 301 nominal frames; pose was detected on 300.
- All 12 required gait landmarks were simultaneously raw-usable on 299/301
  frames and processed-complete on 300/301 frames.
- One left-knee low-confidence sample at frame 43 formed a one-sample interior gap
  and was interpolated. Terminal frame 300 remains missing.
- The recorded Step 2 input hashes still match all three current raw artifacts.
- The run used visibility/presence thresholds `0.5`, maximum gap 3 frames, and a
  centered boxcar window of 3 frames.
- Diagnostics show periodic lower-limb trajectories, but some abrupt raw foot
  departures remain. No automated spike, swap, or tracking-discontinuity
  detection exists.

Readiness is bounded: these outputs are sufficient only for exploratory Step 4
gait-event algorithm development on frames 0-299, with manual review and
raw-versus-smoothed sensitivity checks. They are not validated contact events,
measurements, or clinical outputs.

## Known limitations

- Timing uses nominal frame index/FPS rather than verified presentation time.
  Nominal frame count can exceed decodable content.
- Orientation, seeking, codecs, and frame-position reporting are backend-dependent;
  the project requests auto-orientation off but performs no rotation or mirroring.
- MediaPipe `VIDEO` mode may internally track or smooth. Step 2 "raw" means no
  project postprocessing of selected backend fields.
- Visibility and presence are model scores, not calibrated confidence, uncertainty,
  accuracy, or ground truth. Landmarks are not anatomical joint centers, and heel
  or `foot_index` labels are not validated ground-contact points.
- Step 3 interpolation can hide phase-dependent missingness or cross an unknown
  gait event. Boxcar smoothing attenuates amplitude and does not preserve extrema,
  derivatives, threshold crossings, or event timing.
- Step 3 does not detect spikes, left/right swaps, tracking discontinuities, camera
  motion, or phase-dependent missingness.
- Publication rolls back reported rename failures, but abrupt interruption may
  leave a visible `*.backup-*` or `*.staging-*` recovery path requiring manual
  recovery. Non-dot transaction names intentionally avoid Finder-hidden output
  directories and remain visible through container bind mounts. A `*.backup-*`
  path is the prior published output; a `*.staging-*` path is an unpublished or
  incomplete candidate. Do not merge them. If the final destination is missing
  or incomplete, first move it aside, rename the backup to the final name, and
  verify the restored output before deleting leftovers. Step 2 or Step 3 backup
  cleanup failures may leave visible backup files after successful publication.
- There is no calibration, physical unit conversion, gait-event/contact detection,
  COM estimation, gait or stability metric, reference-system validation, UI, or
  service API.

## Repository map

```text
src/gait_stability/   Step 1 ingestion, Step 2 pose, Step 3 preprocessing APIs
scripts/              Thin CLIs for inspect, estimate, and preprocess workflows
tests/                Deterministic unit, CLI, and pipeline-boundary tests
docs/                 Plans, prompts, and this canonical current-state snapshot
models/README.md       Local ignored pose-model acquisition/provenance instructions
compat/                Metadata-only headless OpenCV compatibility distribution
outputs/               Ignored generated artifacts
data/                  Ignored local data; subject data must not be committed
.opencode/             Agent, command, and scientific workflow configuration
```

## Next logical capabilities

- Step 4 candidate gait-event detection with explicit event definitions, manual
  review, raw-versus-smoothed sensitivity analysis, and reference-label validation.
- Automated spike, swap, tracking-discontinuity, camera-motion, and phase-dependent
  missingness detection.
- Calibrated or explicitly normalized COM/gait measurements and documented
  candidate stability metrics, each with validation evidence.
- Research reporting and a UI/API that invoke rather than reimplement the pipeline.
