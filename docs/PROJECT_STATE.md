# Project State

## Snapshot metadata

- Snapshot UTC: `2026-08-12T23:19:40Z`
- Git branch: `main`
- HEAD: `b28aa85` (`chore: modify OpenCode permission settings`)
- Working tree: dirty due to the uncommitted MVP Step 2 implementation
- Verification environment: Python `3.13.5` at
  `/tmp/opencode/gait-step2-venv`

## What the project currently is

The current executable MVP is a local Python pipeline with two workflows:

1. Step 1 inspects an MP4/MOV, records source and decoder metadata, and saves
   deterministic representative decoded frames.
2. Step 2 runs MediaPipe Tasks Pose Landmarker 1.0.0 over every nominal frame
   slot and writes canonical raw pose estimates, frame outcomes, provenance,
   and an annotated video.

Step 2 stops at unvalidated raw pose-model output. Pose quality preprocessing,
gait events, center-of-mass estimation, gait/stability metrics, reporting UI/API,
and scientific or clinical validation are not implemented.

## Current capabilities

### Step 1: video inspection

- CLI: `python scripts/inspect_video.py INPUT [--output-root OUTPUTS]`.
- API: `gait_stability.inspect_video(video_path, output_root=Path("outputs"))`.
- Accepts `.mp4` and `.mov` by extension and validates path, decoder opening,
  dimensions, nominal FPS, and nominal integer frame count.
- Records resolved source path, file size and SHA-256; OpenCV version/backend;
  extension-derived container indicator; orientation/auto-orientation status;
  dimensions; and nominal timing.
- Requests frames at 0%, 25%, 50%, 75%, and 100% of `frame_count - 1`, with
  ordered de-duplication. If an exact request fails, it tries at most the 10
  preceding indices and records requested and actual indices separately.
- Stages a complete inspection result before replacing the same-stem output.

### Step 2: raw pose estimation

- CLI: `python scripts/estimate_pose.py INPUT --model MODEL.task [options]`.
- API: `gait_stability.estimate_pose_video(video_path, estimator, output_root)`
  through a backend-independent `PoseEstimator` contract.
- Uses MediaPipe Tasks Pose Landmarker 1.0.0, CPU `VIDEO` mode, one pose, no
  segmentation masks, and the explicit local `.task` model supplied by the user.
- Reuses Step 1 output only when its source path and SHA-256 match; otherwise it
  runs Step 1 first.
- Processes every nominal frame slot and distinguishes `decoded_pose`,
  `decoded_no_pose`, and `decode_failure`; failed decodes are never filled with
  landmarks. A verified seek is attempted after a non-terminal decode failure.
- Preserves all 33 MediaPipe landmarks with canonical IDs/names, normalized
  image coordinates, backend-relative `z`, visibility, presence, and a nullable
  generic confidence field. World landmarks are not stored.
- Writes an annotated MP4 in nominal order. Decode failures receive labeled
  black placeholders so frame count/order are preserved; audio is not retained.
- Replaces only Step 2 artifacts, preserving Step 1 metadata and sample frames.

## Quick start

Python 3.11 or newer is supported. MediaPipe 1.0.0 declares the GUI
`opencv-contrib-python` distribution, which is unsuitable on headless Linux
without `libGL`. Install the headless environment from the repository root in
this exact order:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,pose]'
.venv/bin/python -m pip install --no-deps mediapipe==1.0.0
.venv/bin/python -m pip install ./compat/opencv-contrib-python-headless-provider
.venv/bin/python -m pip check
```

The local compatibility package contains no Python/OpenCV implementation. It
provides only MediaPipe's required distribution name and depends on the actual
`opencv-contrib-python-headless>=5.0,<5.1` implementation. Installing MediaPipe
with `--no-deps` is required to prevent pip from selecting GUI OpenCV first.
`requirements-pose-headless.txt` records the equivalent requirements but does
not remove that ordering constraint.

Download the full Pose Landmarker model manually as documented in
`models/README.md`; the application does not download it. The model is an
ignored local runtime input, not bundled with the repository. Then run:

```bash
.venv/bin/python scripts/estimate_pose.py path/to/walk.mp4 \
  --model models/pose_landmarker_full.task \
  --output-root outputs
```

## Key workflows

### Inspect a video

```bash
.venv/bin/python scripts/inspect_video.py path/to/walk.mp4 \
  --output-root outputs
```

- Input: one readable local `.mp4` or `.mov` outside its deterministic output
  destination.
- Output: `outputs/<video-stem>/video_metadata.json` and JPEGs under
  `sample_frames/`.
- Success: the metadata path is printed and all selected frames have exact or
  explicitly recorded bounded-fallback decodes.

### Estimate raw pose

```bash
.venv/bin/python scripts/estimate_pose.py path/to/walk.mp4 \
  --model models/pose_landmarker_full.task \
  --output-root outputs \
  --min-pose-detection-confidence 0.5 \
  --min-pose-presence-confidence 0.5 \
  --min-tracking-confidence 0.5
```

- Inputs: a Step 1-compatible video and an existing MediaPipe Pose Landmarker
  `.task` asset.
- Options: the three confidence thresholds accept finite values in `[0, 1]` and
  default to `0.5`; `--output-root` defaults to `outputs`.
- Output: the four Step 2 artifacts under `outputs/<video-stem>/`, alongside
  retained Step 1 artifacts.
- Success: `pose_metadata.json` is printed after all nominal slots and all four
  staged Step 2 files are published.

## Inputs

- Video formats are selected by `.mp4`/`.mov` filename extension and decoded by
  OpenCV; codec/container support therefore depends on the installed backend.
- Step 1 rejects invalid metadata and a source nested inside
  `<output-root>/<video-stem>/`. Different inputs with the same stem share a
  destination and can replace prior same-stem artifacts.
- Step 2 consumes monocular RGB frames. It does not require, record, or verify
  camera view, gait direction, camera placement, or mirroring.
- No camera intrinsics/extrinsics, distortion correction, physical scale,
  multi-view reconstruction, anthropometry, or gait events are consumed.
- The model file is explicit, local, ignored/untracked, and identified in output
  provenance by resolved path, filename, byte size, and SHA-256.

## Outputs

For an input stem `walk`:

```text
<output-root>/walk/
|-- video_metadata.json
|-- sample_frames/frame_<actual-source-index>.jpg
|-- raw_landmarks.csv
|-- pose_frames.csv
|-- pose_metadata.json
`-- annotated_pose.mp4
```

- `video_metadata.json`: observed file/OpenCV metadata, nominal timing, source
  provenance, and requested-versus-actual representative-frame decode records.
- `sample_frames/*.jpg`: decoded image observations from Step 1.
- `raw_landmarks.csv`: zero or more pose-model estimate rows per detected frame.
  Columns are `frame_index`, `nominal_timestamp_seconds`, `landmark_id`,
  `landmark_name`, `x_normalized`, `y_normalized`, `z_backend_relative`,
  `visibility`, `presence`, and `confidence`.
- `pose_frames.csv`: exactly one status row per attempted nominal slot. Columns
  are `frame_index`, `nominal_timestamp_seconds`,
  `backend_timestamp_milliseconds`, `status`, `landmark_count`, and `detail`.
  Backend timestamps are blank for decode failures; `landmark_count` is a row
  count, not a quality score.
- `pose_metadata.json` (schema version 2): source/model hashes, dimensions/FPS,
  backend/version/configuration, frame counts, coordinate/confidence/timestamp
  semantics, schemas, output names, rendering behavior, and limitations.
- `annotated_pose.mp4`: pose-model estimates over decoded frames at source
  decoded dimensions and nominal FPS/order, plus status labels/placeholders.

CSV was chosen deliberately to avoid pandas, PyArrow, and other Parquet engine
dependencies and to remain inspectable with Python's standard library. Pose
CSV rows and overlays are unvalidated pose-model estimates, not direct
observations, derived biomechanical measurements, stability proxies, or clinical
measurements. No derived measurement output exists yet.

## Configuration

- `pyproject.toml`: Python `>=3.11`, headless OpenCV runtime dependency, optional
  MediaPipe transitive requirements, dev dependencies, and Ruff/mypy/pytest
  settings. Mypy targets Python 3.13 because current dependency stubs use newer
  syntax.
- `requirements-pose-headless.txt`: documented headless pose requirements.
- `compat/opencv-contrib-python-headless-provider/`: metadata-only compatibility
  distribution for MediaPipe's GUI OpenCV package-name requirement.
- `models/README.md`: manual model acquisition and hash/size provenance steps.
- Pose CLI thresholds: detection, presence, and tracking all default to `0.5`.
  They are backend processing thresholds, not project quality filters or
  validated accuracy cutoffs.
- Output location: `--output-root`, default `outputs`.

## Testing and verification

Normal verification from the repository root in the prepared environment:

```bash
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/python -m mypy src scripts
.venv/bin/python -m pytest
.venv/bin/python -m pip check
git diff --check
```

Fresh supplied evidence after final fixes, using Python 3.13.5 at
`/tmp/opencode/gait-step2-venv`:

- `ruff format --check .`, `ruff check .`, `python -m mypy src scripts`, and
  `git diff --check` passed.
- `python -m pytest` passed all 93 tests. Tests cover Step 1 validation,
  fallback/publishing, real synthetic OpenCV I/O, sequential decode outcomes,
  canonical serialization, MediaPipe adapter contracts, rendering, metadata,
  CLI behavior, and Step 2 artifact preservation/rollback.
- `python -m pip check` reported no broken requirements.

A separate orchestrator-run local integration used MediaPipe Tasks Pose
Landmarker 1.0.0 full model in CPU `VIDEO` mode. Only these supplied aggregate,
non-identifying results are recorded here; its input/artifacts were not inspected
while creating this snapshot:

- Source metadata: 301 nominal frames, 30 FPS, 1440x1080.
- Processing: 301 attempted; 300 decoded with pose detected; one terminal decode
  failure; 301 annotated frames including the failure placeholder.
- All required shoulder, hip, knee, ankle, heel, and `foot_index` labels were
  returned on all 300 detected frames.
- Frames 0, 75, 150, 225, and 299 plus frame 300's failure placeholder were
  manually inspected. The skeleton was generally aligned with shoulders, hips,
  knees, ankles, heels, and feet in those samples, with expected side-view
  overlap/occlusion risk and no obvious total tracking loss in decoded samples.
- Ignored, untracked artifacts were
  `outputs/A_Video/raw_landmarks.csv`, `outputs/A_Video/pose_frames.csv`,
  `outputs/A_Video/pose_metadata.json`, and
  `outputs/A_Video/annotated_pose.mp4`.

This integration check demonstrates execution and sample-level visual
plausibility only; it is not pose accuracy, biomechanical, gait, stability, or
clinical validation.

## Known limitations

- Frame timestamps and duration are nominal values derived from frame index,
  nominal FPS, and nominal frame count, not verified presentation timestamps.
  MediaPipe submission timestamps are rounded nominal milliseconds and made
  strictly increasing when necessary.
- OpenCV-reported nominal frame count can exceed decodable content. Step 2 emits
  explicit failures/placeholders; an unverified recovery seek causes remaining
  nominal slots to be marked failed rather than mislabeled.
- OpenCV seeking, frame-position reporting, codec support, and orientation
  behavior are backend-dependent. The project requests auto-orientation off but
  does not rotate or mirror frames itself.
- Normalized `x`/`y` are dimensionless image-plane estimates and may be outside
  `[0, 1]`. Pixel overlays use `round(normalized * (dimension - 1))` without
  clipping. MediaPipe `z` is learned model-relative monocular depth, not camera,
  laboratory, metric, or physical depth.
- Left/right are model body-side labels. `foot_index` and `heel` are model labels,
  not verified toe/contact points or ground-contact locations. Landmarks are not
  anatomical joint centers.
- Visibility and presence are raw model scores, not observed/calibrated
  probabilities, uncertainty, or accuracy. Generic `confidence` remains null.
- The project performs no confidence filtering, interpolation, or smoothing.
  MediaPipe `VIDEO` mode may internally track or temporally smooth, so "raw"
  means selected backend fields are unchanged by project postprocessing.
- Overlay rendering draws all returned landmarks/connections without confidence
  or bounds filtering. It is for inspection, not measurement.
- Outputs are unvalidated pose-model estimates. There is no calibration,
  physical unit conversion, gait-event detection, COM estimation, gait metric,
  stability metric, fall-risk assessment, diagnosis, or clinical validity.
- There is no UI/service API. Models, source videos, and generated outputs are
  local ignored artifacts and must not be treated as repository-bundled assets.

## Repository map

```text
src/gait_stability/   Step 1 ingestion, pose contracts/backend, Step 2 pipeline
scripts/              Thin inspection and pose-estimation CLIs
tests/                Deterministic unit/CLI/integration-boundary tests
docs/                 MVP plan/prompts and this canonical current-state snapshot
models/README.md       Local ignored model acquisition/provenance instructions
compat/                Metadata-only headless OpenCV dependency compatibility
outputs/               Ignored generated artifacts
data/                  Ignored local data; participant data must not be committed
.opencode/             OpenCode agents, commands, and scientific workflow skills
```

## Next logical capabilities

The following Step 3+ capabilities are not implemented:

- Pose quality reports, confidence filtering, missing-data policy, bounded
  interpolation, and trajectory smoothing while retaining raw estimates.
- Validated gait-event/stride detection and explicit biomechanical coordinate
  handling.
- Calibrated or clearly normalized COM and gait measurements.
- Defined and validated candidate stability metrics and research reporting.
- A UI/API that invokes, rather than reimplements, the scientific pipeline.
