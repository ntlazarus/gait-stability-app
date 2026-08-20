# Project State

## Snapshot metadata

- Snapshot UTC: `2026-08-20T14:56:51Z`
- Git branch: `main`
- HEAD: `a8aa92df8f1f245b10b520508101c2faac816f96`
- Working tree: dirty; intended uncommitted Step 5b source, CLI, tests, API
  exports, and method documentation are present. Generated outputs are ignored
  and are not committed.

## What the project currently is

The executable product is a local, research-only Python video gait pipeline:

1. Step 1 inspects MP4/MOV video and records metadata and sample frames.
2. Step 2 estimates raw monocular pose with MediaPipe Tasks Pose Landmarker.
3. Step 3 quality-gates, bounded-interpolates, and smooths image-plane pose.
4. Step 4 detects candidate initial contacts and candidate stride intervals.
5. Step 4b resolves manual event/stride reviews and corrected boundaries.
6. Step 5a computes a represented-segment mass-weighted 2D centroid proxy for
   every frame and original plus normalized samples for reviewed strides.
7. Standalone Step 5b qualifies Step 5a coverage, missingness, and stride sample
   completeness and renders a diagnostic COM-proxy overlay.

The distinctions are material: video pixels and decoder/container properties are
observations; raw landmarks are pose-model estimates; Step 3 outputs are gated,
optionally interpolated and smoothed pose trajectories; Step 5a is an unvalidated
represented-segment COM proxy; and Step 5b is engineering/QC qualification of
proxy completeness, not scientific measurement validation. None is measured
whole-body COM, a stability or fall-risk result, a diagnosis, or a clinical
output. Toe-off, stance/swing/double-support measures and stability scores are not
implemented.

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

### Step 5a: represented-segment COM proxy

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

### Step 5b: COM engineering/QC qualification

- CLI: `.venv/bin/python scripts/qualify_com.py ARTIFACT_DIRECTORY
  [--coverage-thresholds GRID] [--video PATH]`.
- API: `gait_stability.qualify_com(artifact_directory,
  ComQualificationConfig(...), video_path=...)`.
- Reads and strictly cross-validates Step 5a outputs, their six upstream inputs,
  and the provenance-matched source video. It computes coverage, segment and
  direct-landmark missingness, left/right QC asymmetry, threshold sensitivity,
  normalized availability, and per-stride engineering completeness without
  changing or recomputing Step 5a coordinates.
- `mass_coverage` remains the absolute unrenormalized fraction of published total
  body-model mass represented in a frame. `supported_mass_coverage =
  mass_coverage / theoretical_supported_mass_fraction` reports completeness
  relative to the implementation's supported-segment ceiling. Neither quantity
  measures positional accuracy, anatomical accuracy, confidence, or validity.
- The default absolute `mass_coverage` sensitivity grid is
  `0.80,0.82,0.84,0.86,0.88,0.90`. It is predeclared, unvalidated, and diagnostic;
  Step 5b does not select or optimize a threshold or change Step 5a's inherited
  primary gate.
- Qualification categories and `policy_complete_at_threshold` describe
  engineering sample completeness only. Capture suitability is not machine
  established and requires independent human review.

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
.venv/bin/python scripts/qualify_com.py outputs/walk
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

### Estimate the Step 5a proxy

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

### Qualify the Step 5a proxy with Step 5b

```bash
.venv/bin/python scripts/qualify_com.py ARTIFACT_DIRECTORY \
  [--coverage-thresholds 0.80,0.82,0.84,0.86,0.88,0.90] \
  [--video PATH]
```

- Run Step 5a first. The optional threshold grid must contain finite, unique,
  strictly increasing values in `[0,1]`. `--video` only relocates the inherited
  source; its SHA-256 must match provenance.
- Success prints the `com_qualification.json` path after the three outputs are
  staged, all input/source hashes are rechecked, and publication completes with
  replacement backups and rollback.

## Inputs

Step 5a requires exactly these six canonical files in one artifact directory:

1. `processed_landmarks.csv`
2. `preprocessing_metadata.json`
3. `pose_frames.csv`
4. `reviewed_gait_events.csv`
5. `reviewed_strides.csv`
6. `review_resolution_metadata.json`

Step 5a snapshots all six SHA-256 hashes before parsing and rechecks all six
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

Step 5b additionally requires the four Step 5a outputs and the provenance-matched
source video. It normally resolves the video from inherited Step 3 provenance;
`--video` may provide a moved hash-identical file. It reads but does not modify
the Step 5a or upstream artifacts.

Capture inputs are monocular RGB. Steps 5a/5b use normalized image-plane `x` (right)
and `y` (down), with no `z`, camera calibration, physical scale, 3D reconstruction,
or physical units. They require weak-perspective/equivalent endpoint depth, static,
low-distortion, near-sagittal capture with little out-of-plane motion and adequate
visibility of the full body, both arms, and both feet. These assumptions are
user-established and unverified by the software.

## Outputs

Step 5a transactionally publishes four files in `ARTIFACT_DIRECTORY`:

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

Step 5b transactionally publishes three files in the same directory:

- `com_qualification.json`: schema/versioned aggregate frame, segment, landmark,
  asymmetry, stride, threshold-sensitivity, provenance, capture-review, validation,
  and Step 6 readiness evidence. It records hashes for the other two outputs but
  intentionally cannot self-record its own hash.
- `com_stride_qc.csv`: one row per reviewed stride with frame/coverage summaries,
  normalized availability, qualification category, policy-complete flag, component
  booleans, and failure reasons at the inherited primary threshold.
- `annotated_com.mp4`: provenance-matched source video with processed pose and the
  finite represented-segment proxy colored by the inherited primary coverage gate,
  plus explicit research-only/validation warnings. It is a diagnostic rendering,
  not a new observation or measurement.

Artifact interpretation:

- The source video is the observed capture; decoded samples remain decoder outputs.
- `raw_landmarks.csv` and overlays contain pose-model estimates, not observations.
- `processed_landmarks.csv` contains QC-gated and potentially interpolated/smoothed
  trajectories, not raw or validated measurements.
- Step 5a COM files contain an experimental represented-segment image-plane proxy.
- Step 5b files qualify engineering completeness and provenance of that proxy; a
  passing category would not validate physical COM or a stability measurement.

Normalized progression is fixed and timestamp-based:

```text
progression = (timestamp - stride_start_timestamp) / stride_duration * 100
```

The grid includes exact `0` and `100` and has exactly the configured sample count.
An exact timestamp copies that frame. Linear interpolation is permitted only
between adjacent usable consecutive frames with identical represented segment
sets. Invalid/unusable gaps and changed segment sets are never bridged; such rows
use `method=none` and no centroid. Step 5a performs no landmark interpolation or
extrapolation; Step 3 processing provenance is propagated.

## Configuration

- `pyproject.toml`: Python/dependency and Ruff, mypy, and pytest configuration.
- `requirements-pose-headless.txt` and `compat/`: headless MediaPipe environment.
- `models/README.md`: local pose-model acquisition and provenance.
- `docs/GAIT_EVENT_METHOD.md`: Step 4 event method and limits.
- `docs/Step4b_review_resolution.md`: Step 4b contracts and semantics.
- `docs/COM_PROXY_METHOD.md`: Step 5a formulas, endpoints, coefficients, QC,
  normalization, coordinate assumptions, and validation limits.
- `docs/COM_QUALIFICATION_METHOD.md`: Step 5b formulas, input/output contracts,
  coverage semantics, threshold sensitivity, engineering criteria, capture review,
  and readiness boundaries.

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

Focused Step 5a tests:

```bash
.venv/bin/python -m pytest \
  tests/test_com_estimation.py tests/test_com_pipeline.py \
  tests/test_estimate_com_cli.py
```

Focused Step 5b tests:

```bash
.venv/bin/python -m pytest \
  tests/test_com_qualification.py \
  tests/test_com_qualification_normalization.py \
  tests/test_com_qualification_summaries.py \
  tests/test_com_qualification_pipeline.py \
  tests/test_qualify_com_cli.py
```

Prior fresh full verification for Step 5a on `2026-08-19` used the
repository-ignored `venv/bin`
runner while user documentation remains `.venv/bin`: Ruff format check passed for
63 files; Ruff lint passed; mypy passed 17 source/script files; pytest passed all
409 tests in `141.53s`; pip check passed; and Git diff check passed. Focused Step 5
verification passed 197 tests in `109.45s`.

Fresh final verification on `2026-08-20` used the repository-ignored `venv/bin`
runner while documented user commands remain `.venv/bin`: `ruff format --check .`
passed with 73 files already formatted; `ruff check .` passed; mypy passed with no
issues in 20 source files; pytest passed all 512 tests in `187.35s`; pip check
reported no broken requirements; and `git diff --check` passed.

### Aggregate ignored-artifact evidence

The current aggregate smoke evidence is research-only, nonidentifying, and from
ignored generated artifacts. It is not participant description, clinical evidence,
or reference validation.

- The male coefficient set was explicitly selected; it was not inferred from the
  video, pose, metadata, names, or participant characteristics.
- Inputs contained 31 reviewed event rows, 16 accepted events, and 14 reviewed
  strides; reviewed boundary corrections were consumed by Step 5a.
- `com_proxy.csv` had 301 frame rows: 300 finite represented centroids, but zero
  usable at the primary `0.90` threshold. Maximum absolute `mass_coverage` was
  `0.8812`; the left upper arm, forearm, and hand were persistently unavailable,
  while the head is structurally unsupported rather than capture-missing.
- All 14 strides had exactly 101 normalized rows. Across 1,414 normalized rows,
  65 were exact with finite centroids, zero were usable, and 1,349 had
  `method=none`.
- All recorded hashes matched and `com_diagnostic.png` was readable (`1184x1797`
  RGB).
- Step 5b reported `300/301` finite-COM frames, `0/301` primary-eligible frames,
  and `0/14` policy-complete strides at the inherited primary `0.90` threshold.
- At each sensitivity threshold from `0.80` through `0.88`, it reported `300/301`
  eligible frames and `14/14` policy-complete strides. This sharp sensitivity is
  not threshold validation, does not establish accuracy, and is not a reason to
  tune or relax the primary threshold on this artifact.
- Clean-capture human review is required but was not recorded. The inherited
  capture declaration did not establish or machine-verify the required static,
  near-sagittal, low-distortion, full-body view.

These are aggregate, nonidentifying engineering diagnostics from ignored generated
artifacts. They are not participant description, threshold validation, clinical
evidence, or reference validation. The finite represented centroids remain
unvalidated proxy values.

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
- Step 3 smoothing/interpolation affects Step 5a positions. Step 5a does not bridge
  invalid gaps, but normalization remains time-based and is not validated gait
  cycle percentage.
- Reviewed events/strides are QC segmentation windows, not ground truth or
  force-plate-confirmed contacts. Event timing, stride duration, the COM proxy,
  and any relationship to stability remain externally unvalidated.
- No output is a stability score, fall-risk estimate, diagnosis, validation result,
  or clinical result. There are no force-plate/motion-capture comparisons,
  repeated-session studies, population validation, or clinical validation.
- Step 5b coverage can characterize completeness but cannot detect positional bias,
  establish camera/capture suitability, validate a threshold, or establish COM
  accuracy. High supported-mass coverage only means completeness relative to the
  supported model ceiling.
- Transaction interruption or cleanup failure may leave staging/backup recovery
  files; they must not be mixed with either published artifact set.

## Repository map

```text
src/gait_stability/   Reusable Steps 1-5b APIs and typed contracts
scripts/              Thin CLI entry points for Steps 1-5b
tests/                Deterministic unit, CLI, and pipeline-boundary tests
docs/                 Method documentation and this canonical snapshot
models/README.md       Ignored local model acquisition/provenance instructions
compat/                Headless OpenCV compatibility distribution
outputs/               Ignored generated research artifacts
data/                  Ignored local data; subject data must not be committed
.opencode/             Agent and scientific workflow configuration
```

## Next logical capabilities

Step 6 engineering exploratory readiness is **CONDITIONAL**, pending independently
reviewed clean-capture evidence that establishes full-body/limb visibility, suitable
capture geometry, and sufficiently complete proxy trajectories under a documented,
defensible QC policy. Sensitivity results from the current artifact do not satisfy
that requirement.

Step 6 scientific measurement readiness is **NO-GO / not established**, pending
reference validation and calibrated/validated inputs including scale/geometry,
ground or gravity alignment, gait events, and downstream measurements. The next
logical work is independent clean-capture review and appropriate reference-system
validation, not threshold tuning or a stability score.
