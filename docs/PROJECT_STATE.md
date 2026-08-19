# Project State

## Snapshot metadata

- Snapshot UTC: `2026-08-19T19:38:50Z`
- Git branch: `main`
- HEAD: `f2fc44f1b7cdf95606052974e96e38e523529693`
- Working tree: dirty; uncommitted Step 5 source, CLI, tests, API export, and
  method documentation are present. Generated outputs are ignored. Current
  unrelated docs/theory are excluded and untouched by this snapshot update.

## What the project currently is

The executable product is a local, research-only Python video gait pipeline:

1. Step 1 inspects MP4/MOV video and records metadata and sample frames.
2. Step 2 estimates raw monocular pose with MediaPipe Tasks Pose Landmarker.
3. Step 3 quality-gates, bounded-interpolates, and smooths image-plane pose.
4. Step 4 detects candidate initial contacts and candidate stride intervals.
5. Step 4b resolves manual event/stride reviews and corrected boundaries.
6. Step 5 computes a represented-segment mass-weighted 2D centroid proxy for
   every frame and original plus normalized samples for reviewed strides.

Step 5 software is implemented. Its output is an experimental image-plane proxy,
not measured whole-body COM or a stability, fall-risk, diagnostic, validated, or
clinical output. Toe-off, stance/swing/double-support measures and stability
scores are not implemented.

## Current capabilities

### Steps 1-3: video, pose, and pose QC

- `scripts/inspect_video.py` validates local `.mp4`/`.mov` decoding and writes
  source/decoder metadata plus representative JPEGs.
- `scripts/estimate_pose.py` runs a user-supplied MediaPipe Pose Landmarker
  `.task` model in CPU video mode and writes the canonical 33-landmark raw pose
  contract, frame manifest, metadata, and annotated video.
- `scripts/preprocess_pose.py` creates a complete frame-by-landmark audit grid,
  applies configurable visibility/presence QC, bounded interior interpolation,
  and centered smoothing in normalized image `x`/`y`, and preserves missingness
  and processing provenance. It does not process `z` into derived measures.

### Steps 4 and 4b: gait-event segmentation and review

- `scripts/detect_gait_events.py` uses direction-normalized heel motion relative
  to the bilateral hip midpoint to produce auditable accepted/rejected candidate
  initial contacts and same-side candidate strides. Results are unvalidated
  segmentation/QC artifacts and require manual confirmation.
- `scripts/resolve_gait_reviews.py` applies explicit event and stride review CSVs,
  canonical `pose_frames.csv` timestamps, bounded rejected-event promotion, and
  shared-boundary propagation. It transactionally publishes reviewed events,
  reviewed strides, and review-resolution metadata without altering automatic
  inputs.

### Step 5: represented-segment COM proxy

- CLI: `.venv/bin/python scripts/estimate_com.py ARTIFACT_DIRECTORY
  --anthropometry-sex {male,female} [--minimum-mass-coverage VALUE
  --normalized-stride-samples N]`.
- API: `gait_stability.estimate_com(artifact_directory, config)` with an explicit
  `ComEstimationConfig(anthropometry_sex="male" | "female", ...)`. Sex is required
  and is never inferred from video, landmarks, metadata, names, or other fields.
- Uses corrected de Leva adjusted 14-segment coefficients. The head is
  intentionally unsupported because standard MediaPipe Pose has no defensible
  source-compatible vertex/neck joint-centre line; the other 13 represented
  segments are supported. The theoretical maximum represented fractions are
  `0.9306` male and `0.9331` female.
- Citation: de Leva P. "Adjustments to Zatsiorsky-Seluyanov's segment inertia
  parameters." *Journal of Biomechanics*. 1996;29(9):1223-1230.
  DOI `10.1016/0021-9290(95)00178-6`.
- For each frame, each supported segment centroid is
  `proximal + r * (distal - proximal)`. The frame proxy is the mass-weighted
  centroid of usable represented segments divided by their represented mass.
  `mass_coverage` remains the raw, unrenormalized sum of contributing published
  mass fractions. The default usability gate is `>=0.90`; finite below-threshold
  centroids remain recorded but are not usable.
- Segment and aggregate QC preserves raw-observed, axis interpolation, smoothing,
  smoothing-over-interpolation, missing, and other-limited provenance. Per-segment
  and aggregate QC flags/mass sums are nonexclusive.
- Frame COM is computed independently of strides. Reviewed Step 4b strides only
  define the rows subsequently copied and normalized in `stride_com.csv`.

## Quick start

Python 3.11 or newer is supported. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,pose]'
.venv/bin/python -m pip install --no-deps mediapipe==1.0.0
.venv/bin/python -m pip install ./compat/opencv-contrib-python-headless-provider
.venv/bin/python -m pip check
```

MediaPipe declares GUI OpenCV; the local provider retains the headless OpenCV
implementation. Obtain the full Pose Landmarker `.task` manually as documented
in `models/README.md`; the application does not download models.

Run the complete implemented workflow:

```bash
.venv/bin/python scripts/inspect_video.py path/to/walk.mp4 --output-root outputs
.venv/bin/python scripts/estimate_pose.py path/to/walk.mp4 \
  --model models/pose_landmarker_full.task --output-root outputs
.venv/bin/python scripts/preprocess_pose.py outputs/walk
.venv/bin/python scripts/detect_gait_events.py outputs/walk \
  --walking-direction image_right
.venv/bin/python scripts/resolve_gait_reviews.py outputs/walk \
  path/to/assumption_responses.txt
.venv/bin/python scripts/estimate_com.py outputs/walk \
  --anthropometry-sex male
```

The documented user environment remains `.venv/bin`. The verification runner
described below used the repository-ignored `venv/bin` environment.

## Key workflows

### Inspect, estimate, and preprocess pose

- Inspect: `.venv/bin/python scripts/inspect_video.py INPUT --output-root outputs`.
  Success writes `outputs/<stem>/video_metadata.json` and `sample_frames/*.jpg`.
- Pose: `.venv/bin/python scripts/estimate_pose.py INPUT --model MODEL.task
  --output-root outputs`. Success writes `raw_landmarks.csv`, `pose_frames.csv`,
  `pose_metadata.json`, and `annotated_pose.mp4`.
- Preprocess: `.venv/bin/python scripts/preprocess_pose.py outputs/<stem>`.
  Success writes `processed_landmarks.csv`, `pose_quality.json`,
  `preprocessing_metadata.json`, and normally
  `pose_trajectory_diagnostic.png`.

### Detect and review candidate events

- Detect: `.venv/bin/python scripts/detect_gait_events.py outputs/<stem>
  --walking-direction {image_right,image_left}`. It consumes canonical Step 3
  artifacts and writes `walking_bout.json`, `gait_events.csv`, `strides.csv`,
  `gait_event_diagnostic.png`, `annotated_gait_events.mp4`, and
  `gait_event_metadata.json` as one transaction.
- Review: `.venv/bin/python scripts/resolve_gait_reviews.py outputs/<stem>
  ASSUMPTION_RESPONSE_DOCUMENT`. It requires the Step 4 outputs plus complete
  `gait_event_reviews.csv` and `strides_reviews.csv`, and writes
  `reviewed_gait_events.csv`, `reviewed_strides.csv`, and
  `review_resolution_metadata.json`.

### Estimate the Step 5 proxy

```bash
.venv/bin/python scripts/estimate_com.py ARTIFACT_DIRECTORY \
  --anthropometry-sex {male,female} \
  [--minimum-mass-coverage 0.90] \
  [--normalized-stride-samples 101]
```

- `--anthropometry-sex` is mandatory. `--minimum-mass-coverage` accepts `[0,1]`
  and defaults to `0.90`; `--normalized-stride-samples` is an integer `>=2` and
  defaults to `101`.
- Success prints the `com_metadata.json` path after all outputs have been staged,
  all six input hashes rechecked, and the four-file set published transactionally
  with replacement backups and rollback.

## Inputs

Step 5 requires exactly these six canonical files in one artifact directory:

1. `processed_landmarks.csv`
2. `preprocessing_metadata.json`
3. `pose_frames.csv`
4. `reviewed_gait_events.csv`
5. `reviewed_strides.csv`
6. `review_resolution_metadata.json`

The pipeline snapshots all six SHA-256 hashes before parsing and rechecks all six
immediately before publication. It strictly validates exact CSV headers/order,
canonical landmark and frame grids, booleans/numerics/statuses, strictly increasing
canonical timestamps, frame/event/stride timestamp correspondence, schema and
algorithm versions, canonical basenames, metadata-linked hashes, reviewed stride
endpoints/counts, and empty Step 4b `blocking_unresolved`.

Only canonical reviewed Step 4b artifacts are accepted. There is no fallback to
automatic Step 4 events or strides. Step 3 provenance must link the same
`pose_frames.csv` and `processed_landmarks.csv`; Step 4b provenance must link the
same preprocessing metadata, pose timestamp source, reviewed events, and reviewed
strides.

Capture inputs are monocular RGB. Step 5 uses normalized image-plane `x` (right)
and `y` (down), with no `z`, camera calibration, physical scale, 3D reconstruction,
or physical units. It assumes weak-perspective/equivalent endpoint depth, static,
low-distortion, near-sagittal capture with little out-of-plane motion; these
assumptions are user-established and unverified by the software.

## Outputs

Step 5 transactionally publishes four files in `ARTIFACT_DIRECTORY`:

- `com_proxy.csv`: one row per nominal frame. Broad fields include frame/time/
  status, represented-segment centroid `com_x`/`com_y`, raw `mass_coverage`, model
  mass, usability, usable/missing segments, nonexclusive aggregate contributor and
  mass QC, and each segment's centroid, mass, usability, contributors, and QC.
- `stride_com.csv`: reviewed stride identity/bounds/review provenance plus all
  original in-bound frame rows and exactly `N` normalized rows per stride (`101`
  by default), with progression, exact/linear/none method, source or bracket
  timestamps/frames, centroid, usability, coverage, contributors, and QC.
- `com_diagnostic.png`: required diagnostic of frame centroid coordinates,
  coverage/usability, and reviewed-stride context.
- `com_metadata.json`: configuration, selected coefficients and citation, exact
  input/output provenance and hashes (self-hash intentionally null), schemas,
  frame/stride counts, coverage/QC semantics, assumptions, limitations, runtime,
  warnings, and carried scientific unresolved items.

Normalized progression is fixed and timestamp-based:

```text
progression = (timestamp - stride_start_timestamp) / stride_duration * 100
```

The grid includes exact `0` and `100` and has exactly the configured sample count.
An exact timestamp copies that frame. Linear interpolation is permitted only
between adjacent usable consecutive frames with identical represented segment
sets. Invalid/unusable gaps and changed segment sets are never bridged; such rows
use `method=none` and no centroid. Step 5 performs no landmark interpolation or
extrapolation; Step 3 processing provenance is propagated.

## Configuration

- `pyproject.toml`: Python/dependency and Ruff, mypy, and pytest configuration.
- `requirements-pose-headless.txt` and `compat/`: headless MediaPipe environment.
- `models/README.md`: local pose-model acquisition and provenance.
- `docs/GAIT_EVENT_METHOD.md`: Step 4 event method and limits.
- `docs/Step4b_review_resolution.md`: Step 4b contracts and semantics.
- `docs/COM_PROXY_METHOD.md`: Step 5 formulas, endpoints, coefficients, QC,
  normalization, coordinate assumptions, and validation limits.

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

Focused Step 5 tests:

```bash
.venv/bin/python -m pytest \
  tests/test_com_estimation.py tests/test_com_pipeline.py \
  tests/test_estimate_com_cli.py
```

Fresh full verification on `2026-08-19` used the repository-ignored `venv/bin`
runner while user documentation remains `.venv/bin`: Ruff format check passed for
63 files; Ruff lint passed; mypy passed 17 source/script files; pytest passed all
409 tests in `141.53s`; pip check passed; and Git diff check passed. Focused Step 5
verification passed 197 tests in `109.45s`.

### Aggregate ignored-artifact smoke test

The current aggregate smoke evidence is research-only, nonidentifying, and from
ignored generated artifacts. It is not participant description, clinical evidence,
or reference validation.

- The user explicitly selected the male coefficient set; it was not inferred.
- Inputs contained 31 reviewed event rows, 16 accepted events, and 14 reviewed
  strides. Corrected shared boundaries were consumed at S0003 end/S0005 start
  frame 86 and S0006 end/S0008 start frame 140.
- `com_proxy.csv` had 301 frame rows: 300 finite represented centroids, but zero
  usable at the default `0.90` threshold. Maximum coverage was `0.8812` because
  the head is unsupported and the left arm chain was unavailable; all 301 head
  rows carried explicit unsupported QC.
- All 14 strides had exactly 101 normalized rows. Across 1,414 normalized rows,
  65 were exact with finite centroids, zero were usable, and 1,349 had
  `method=none`.
- All recorded hashes matched and `com_diagnostic.png` was readable (`1184x1797`
  RGB).

This is a conservative QC outcome. It is no evidence that the software is invalid,
and no evidence that the finite represented centroids are valid COM measurements.

## Known limitations

- The represented centroid is not whole-body measured COM. Missing segments alter
  its represented-mass denominator; `mass_coverage` is coverage, not positional
  accuracy. Anthropometric coefficients are population averages, not individual
  measurements.
- MediaPipe landmarks are model estimates and endpoint proxies, not anatomical
  joint centers. The hand distal midpoint and foot endpoint are unvalidated
  surrogates. The head is unsupported, so full model coverage is impossible.
- No `z`, scale, depth, physical units, laboratory frame, camera calibration, or
  perspective correction is used. Weak-perspective, near-sagittal, static-camera,
  low-distortion assumptions are not machine-verified.
- Step 3 smoothing/interpolation affects Step 5 positions. Step 5 does not bridge
  invalid gaps, but normalization remains time-based and is not validated gait
  cycle percentage.
- Reviewed events/strides are QC segmentation windows, not ground truth or
  force-plate-confirmed contacts. Event timing, stride duration, the COM proxy,
  and any relationship to stability remain externally unvalidated.
- No output is a stability score, fall-risk estimate, diagnosis, validation result,
  or clinical result. There are no force-plate/motion-capture comparisons,
  repeated-session studies, population validation, or clinical validation.
- Transaction interruption or cleanup failure may leave staging/backup recovery
  files; they must not be mixed with the published four-file set.

## Repository map

```text
src/gait_stability/   Reusable Steps 1-5 APIs and typed contracts
scripts/              Thin CLI entry points for Steps 1-5
tests/                Deterministic unit, CLI, and pipeline-boundary tests
docs/                 Method documentation and this canonical snapshot
models/README.md       Ignored local model acquisition/provenance instructions
compat/                Headless OpenCV compatibility distribution
outputs/               Ignored generated research artifacts
data/                  Ignored local data; subject data must not be committed
.opencode/             Agent and scientific workflow configuration
```

## Next logical capabilities

Step 5 software is implemented, but the current artifact is **NO-GO** for
downstream feature extraction under default coverage because no frame or normalized
sample is usable. The next work is improving capture/pose coverage and externally
validating event timing and the represented centroid against appropriate reference
systems. A stability score is not the next justified step.
