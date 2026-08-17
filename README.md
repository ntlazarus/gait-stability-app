# Gait Stability from Video: OpenCode Overview

## Project Purpose

This project builds a research-grade Python pipeline for estimating gait
characteristics and candidate stability metrics from walking video. It is not a
diagnostic medical device. Unvalidated outputs must not be presented as clinical
conclusions.

## Configuration and Runtime

- `opencode.docker.json` is the tracked source configuration.
- `run-opencode.sh` copies that file to the ignored, generated `opencode.json`
  and launches the `local-opencode` Docker image.
- The default agent is `orchestrator`.
- `AGENTS.md` defines repository, scientific, software, and workflow rules.
- The Codebase Memory MCP server and plugin are configured for repository
  context and indexing hooks.
- Project skills are loaded from `.opencode/skills`.

## Agents

| Agent | Role | Model |
| --- | --- | --- |
| `orchestrator` | Primary; plans, delegates, and verifies | `openai/gpt-5.6-sol` |
| `researcher` | Read-only evidence research | `openai/gpt-5.6-terra` |
| `python-implementer` | Bounded Python changes | `opencode/nemotron-3-ultra-free` |
| `test-engineer` | Deterministic tests | `opencode/nemotron-3-ultra-free` |
| `biomechanics-reviewer` | Read-only scientific and measurement review | `openai/gpt-5.6-sol` |
| `code-reviewer` | Read-only engineering review | `opencode/nemotron-3-ultra-free` |

## Skills and Commands

Project skills:

- `experiment-reproducibility`, `gait-biomechanics`, `pose-quality-control`,
  `stability-metrics`, `validation-and-claims`, and `video-ingestion`.

Slash commands:

- `/plan-feature`: plan bounded feature work.
- `/record-experiment`: record reproducibility and provenance details.
- `/review-change`: review a repository change.
- `/run-checks`: run the configured verification workflow.

## Working Model

The orchestrator inspects relevant repository state, develops a bounded plan,
and delegates implementation, testing, and research to specialized agents. It
requires biomechanics review when scientific or measurement semantics change
and code review for nontrivial code changes. It then independently verifies the
required checks and final diff rather than relying on an implementation agent's
report.

All work must preserve the distinctions among:

- observed video;
- pose estimates;
- processed trajectories;
- biomechanical estimates;
- validated measurements; and
- unvalidated proxies.

These categories must not be conflated, and clinical validity must not be
claimed without supporting validation evidence.

## Video Inspection

Install the package and inspect a local MP4 or MOV:

```bash
python -m pip install -e '.[dev]'
python scripts/inspect_video.py path/to/walk.mp4 --output-root outputs
```

The command writes `outputs/<video_stem>/video_metadata.json` and deterministic
JPEG samples under `sample_frames/`. Nominal timestamps and duration come from
container metadata. If a requested representative frame cannot be decoded, the
inspector tries at most the 10 preceding source indices and records requested and
actual indices, timestamps, fallback distance, and decode status separately.
Images are named by actual source index, and fallback collisions reuse the
existing image without dropping the requested target's metadata record.
Each fallback attempt re-seeks on the same capture; a failed decode may leave
some backends unrecoverable. The inspector checks finite nonnegative OpenCV frame
positions, but frame identity remains backend-report dependent when position
reporting is unavailable, and actual timestamps are not verified. Orientation
metadata is recorded when the backend reports a finite conventional value. A
separate `auto_orientation_status` records whether the backend accepted,
rejected, did not support, or errored on the request to disable OpenCV
auto-orientation. Rejected or unsupported requests do not establish that decoded
frames are unrotated. The inspector performs no additional rotation.

Artifacts are staged completely in a unique, non-dot same-directory path before
output replacement. Reported rename failures trigger rollback of the prior
result, but replacing an existing output requires two same-filesystem renames.
Abrupt interruption may leave a visible `*.backup-*` or `*.staging-*` recovery
path requiring manual recovery. A `*.backup-*` path contains the prior published
output; a `*.staging-*` path contains an unpublished or incomplete candidate. Do
not merge them. If the final destination is missing or incomplete, first move it
aside, rename the backup to the final name, and verify the restored output before
deleting any leftovers. Step 2 or Step 3 backup-cleanup failures may leave visible
backup files even after publication succeeds. This stage only records video
properties; it performs no pose, gait, biomechanical, or clinical assessment.

The package runtime remains Python 3.11 or newer. Static checks currently target
Python 3.13 so mypy can parse the NumPy stubs installed with OpenCV 5 in the
development environment.

## Raw Pose Estimation

Step 2 uses MediaPipe Tasks Pose Landmarker 1.0.0 in CPU `VIDEO` mode with one
pose. MediaPipe 1.0.0 declares the GUI `opencv-contrib-python` distribution,
which imports unsuccessfully on headless Linux without `libGL`. Python package
metadata cannot express that `opencv-contrib-python-headless` replaces that
distribution. Install the headless runtime exactly as follows:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,pose]'
.venv/bin/python -m pip install --no-deps mediapipe==1.0.0
.venv/bin/python -m pip install ./compat/opencv-contrib-python-headless-provider
.venv/bin/python -m pip check
```

The local compatibility package contains no Python or OpenCV implementation. It
only supplies the distribution name required by MediaPipe and depends on
`opencv-contrib-python-headless>=5.0,<5.1`, which is the actual `cv2` provider.
Its project name and `5.0.0.0` version intentionally mirror the GUI distribution
requirement for dependency-resolution purposes; they do not identify another
OpenCV implementation or an application release.
The equivalent requirements are recorded in `requirements-pose-headless.txt`,
but MediaPipe must still be installed with `--no-deps` to prevent pip from
selecting its GUI OpenCV dependency first. This is an unavoidable limitation of
standard dependency metadata, not a runtime substitution performed by the app.

The `pose` optional dependency group lists MediaPipe 1.0.0's declared transitive
runtime requirements (`absl-py`, `certifi`, `flatbuffers`, `matplotlib`, and
`sounddevice`). They are installed to satisfy the backend distribution; the gait
application uses Matplotlib only for Step 3's optional diagnostic plot and does
not directly import the other listed packages. MediaPipe itself is installed
separately with `--no-deps` only to avoid its GUI OpenCV distribution.

Download the recommended full Pose Landmarker model manually using the official
MediaPipe model link documented in `models/README.md`; processing never downloads
a model. Then run:

```bash
.venv/bin/python scripts/estimate_pose.py path/to/walk.mp4 \
  --model models/pose_landmarker_full.task \
  --output-root outputs \
  --min-pose-detection-confidence 0.5 \
  --min-pose-presence-confidence 0.5 \
  --min-tracking-confidence 0.5
```

The command reuses matching Step 1 metadata and samples. If they are absent or
the source path/hash is stale, it runs inspection first. It uses unique, non-dot
same-directory staging and backup paths to replace only these Step 2 files under
`outputs/<video_stem>/`:

- `raw_landmarks.csv`: one row per detected canonical landmark, preserving
  nominal frame/time, landmark ID/name, normalized image `x`/`y`, backend-relative
  `z`, visibility, presence, and nullable generic confidence;
- `pose_frames.csv`: exactly one row per attempted nominal frame, explicitly
  distinguishing `decoded_pose`, `decoded_no_pose`, and `decode_failure`, and
  recording the exact backend-submitted integer timestamp for decoded frames;
- `pose_metadata.json`: source/model hashes, model size, versions, thresholds,
  schemas, counts, coordinate/confidence semantics, and artifact paths;
- `annotated_pose.mp4`: source decoded dimensions, nominal FPS/order, canonical
  skeleton, and frame status. Decode failures use labeled black placeholders to
  preserve nominal count and ordering. Audio is not retained.

CSV is used deliberately so the MVP has no pandas, PyArrow, or other Parquet
engine dependency and artifacts remain inspectable with Python's standard
library. `landmark_count` is a row count, not a quality score.

Timestamps are nominal `frame_index / nominal_fps`, not verified presentation
timestamps. `backend_timestamp_milliseconds` preserves the exact timestamp sent
to MediaPipe and is blank for decode failures. Step 2 separately records whether
its own processing capture accepted the request to disable OpenCV
auto-orientation; it does not reuse Step 1's status. The project performs no
explicit rotation or mirroring. Camera view, gait direction, placement, and
mirroring are not required or verified.

The input is monocular RGB. No intrinsics, extrinsics, distortion correction,
physical scale, or multi-view reconstruction is used. Normalized `x` and `y` are
dimensionless image-plane pose-model estimates and remain unchanged; overlay
pixels are derived as `round(normalized * (dimension - 1))` without clipping.
The overlay draws every returned row without quality filtering and places one
status label on an opaque background. Left/right names are model body-side labels,
not image-side labels. `foot_index` is the model's label, not a project-defined
toe/contact point, and `heel` is not a ground-contact location. MediaPipe `z` is
learned monocular model-relative depth, with the hip midpoint as origin and
roughly the same scale as `x`; it is not camera/laboratory depth or metric.

`visibility` is a raw backend model score associated with landmark visibility.
`presence` is a separate raw backend model score associated with landmark
presence. Neither score is calibrated accuracy, probability, or ground truth,
and neither is an anatomical or positional measurement. Backend thresholds
control backend processing and are not validated quality cutoffs. MediaPipe
visibility is not renamed as generic confidence. `decoded_pose` means only that
the backend returned a nonempty pose result. "Raw" means selected backend fields
are unchanged by project postprocessing; MediaPipe `VIDEO` mode may internally
track or temporally smooth. World landmarks and segmentation masks are
deliberately omitted.
There is no project interpolation, gait-event detection, COM estimation, gait or
stability metric, biomechanical validation, or clinical interpretation.
Landmarks are not anatomical joint centers, laboratory coordinates, or clinical
measurements, and outputs have not been validated against labels or a reference
measurement system.

## Pose Quality and Preprocessing

Step 3 consumes an existing Step 2 artifact directory and writes a separate,
auditable processed dataset alongside the raw artifacts:

```bash
.venv/bin/python scripts/preprocess_pose.py outputs/walk \
  --visibility-threshold 0.5 \
  --presence-threshold 0.5 \
  --max-gap-frames 3 \
  --smoothing-window-frames 3
```

The Python API accepts either the artifact directory or an explicit
`RawPoseArtifacts` contract:

```python
from gait_stability import PosePreprocessingConfig, preprocess_pose

artifacts = preprocess_pose(
    "outputs/walk",
    PosePreprocessingConfig(max_gap_frames=3, smoothing_window_frames=3),
)
```

The default filter requires finite visibility and presence scores greater than
or equal to `0.5`; generic confidence is disabled because Step 2's current
MediaPipe backend does not provide it. These thresholds are configurable
engineering heuristics, not calibrated probabilities or validated accuracy
cutoffs. In Step 3 metadata, `visibility` remains the raw backend model score
associated with landmark visibility and `presence` remains the distinct raw
backend model score associated with landmark presence; neither is calibrated
accuracy, probability, or ground truth. Use `--disable-visibility`,
`--disable-presence`, or
`--enable-confidence` only when the input schema's score applicability is
understood. Finite normalized `x`/`y` values outside `[0, 1]` remain usable but
are flagged; nonfinite coordinates are unusable.

The command validates all three Step 2 inputs before processing. It derives a
complete nominal frame by 33-landmark audit grid while preserving sparse raw
rows, coordinates, backend-relative `z`, and confidence fields unchanged.
The three resolved input paths must be distinct from one another and from every
Step 3 output path. Step 2 metadata must declare schema version 2, and explicit
embedded CSV schemas must match. Input hashes are captured before parsing and
rechecked immediately before publication; a concurrent change fails without
replacing the prior Step 3 set.
Only normalized image-plane `x` and `y` are processed. Interior scalar gaps of
at most `--max-gap-frames` missing samples are interpolated linearly by nominal
timestamp when both endpoints are raw-observed usable for that scalar
coordinate. After landmark-level enabled-score gating, finite `x` and `y` are
independently usable: one remains observed and processable when the other is
nonfinite. The audit field `observed_usable` retains planar meaning and is true
only when both `x_observed_usable` and `y_observed_usable` are true. There is no
recursive interpolation, extrapolation, confidence interpolation, or `z`
processing.

Smoothing is a centered unweighted moving average applied independently within
each contiguous nonmissing scalar segment. The odd window defaults to three;
one disables smoothing. Each point uses the largest available symmetric odd
support, so segment endpoints retain their pre-smoothed value. This boxcar
method is noncausal and index-symmetric, with no fixed group delay under uniform
sampling, and is not time-weighted for irregular timestamps. This centering does
not preserve extrema, threshold crossings, derivatives, or gait-event timing.
The filter can attenuate trajectory amplitude.
At 30 fps the unvalidated default spans three samples (about 0.1 seconds of
samples) and about 0.067 seconds between support endpoints. All 33 canonical
landmarks are eligible for coordinate-level interpolation.

Step 3 stages these files completely in unique, non-dot same-directory paths
before publication. Publication is rollback-capable for reported rename
failures, but abrupt process interruption may leave a visible `*.backup-*` or
`*.staging-*` recovery path requiring manual recovery:

- `processed_landmarks.csv`: complete audit grid with raw, pre-smoothed, and
  processed planar values plus explicit raw-presence, usability, rejection,
  coordinate-level raw-observed usability, interpolation, smoothing, and
  final-missing flags;
- `pose_quality.json`: frame and required-landmark coverage, per-landmark
  coverage and confidence summaries, missing/interpolated gap records and
  durations, point and scalar-coordinate missing/interpolation fractions, and
  explicit denominator semantics, without a quality label;
- `preprocessing_metadata.json`: run/configuration, hashes, inherited source and
  model provenance, versions, Git state, algorithms, schemas, and limitations;
- `pose_trajectory_diagnostic.png`: raw-versus-processed ankle, heel, and hip
  `x`/`y` trajectories over nominal time when optional Matplotlib is available.

Use `--diagnostic-landmarks left_ankle,right_ankle` to select canonical plot
landmarks or `--no-diagnostic` when Matplotlib is unavailable. Disabling the
diagnostic removes any stale Step 3 diagnostic during successful set
replacement. Run `python scripts/preprocess_pose.py --help` for all confidence,
gap, smoothing, and plotting controls.

Step 3 does not detect swaps, spikes, camera motion, phase-dependent missingness,
or gait events, so interpolation may unknowingly cross an event. It performs no
camera calibration, physical conversion, COM estimation, gait or stability
metric, clinical interpretation, or biomechanical validation. Monocular
normalized trajectories remain nonphysical pose-model estimates, and nominal
timestamps remain unverified presentation times.

Quality reports distinguish frames where all 12 required gait landmarks were
raw planar-observed usable from frames where all 12 are processed complete.
`observed_usable` is only heuristic planar enabled-score and finite-coordinate
gating; it does not establish positional or anatomical accuracy. "All 12" means
all 12 named required gait landmarks satisfy the condition simultaneously in the
same nominal frame, not any 12 landmarks or coverage accumulated across frames.
Processed completeness may include bounded interpolation and does not imply that
a landmark was observed or accurate. The legacy fields
`frames_with_all_12_required_gait_landmarks` and `required_landmark_coverage` are
ambiguously named aliases for simultaneous-all-12 `observed_usable`, not processed
completeness or accuracy.
Per-landmark low-confidence and enabled-score-missing fractions report both
nominal-frame coverage and fractions among raw returned rows; the latter is
`null` when no raw row exists. Gap runs are reported separately for `x` and `y`
with start/end frame indices and nominal timestamps. A one-sample gap has a
zero-second sample span, and boundary gaps have no bracketing duration. Legacy
point-union gap fields remain explicitly labeled as unions and must not be used
to assess the scalar `--max-gap-frames` bound. Finite out-of-image coordinates
remain usable and flagged.

## Candidate Gait Events and Strides

Step 4 consumes the three canonical Step 3 artifacts in place. Walking direction
must be manually established:

```bash
.venv/bin/python scripts/detect_gait_events.py outputs/walk \
  --walking-direction image_right
```

Use both `--manual-start-frame` and `--manual-end-frame` for an inclusive manual
analysis interval. `--video path/to/walk.mp4` overrides only the inherited file
location; the selected file must match the inherited Step 3 source SHA-256 and
cannot be a different recording. The command transactionally replaces only these
six Step 4 artifacts:

- `walking_bout.json`
- `gait_events.csv`
- `strides.csv`
- `gait_event_diagnostic.png`
- `annotated_gait_events.mp4`
- `gait_event_metadata.json`

Events are **video-derived candidate initial contacts**, not force-confirmed heel
strikes. `high` means deterministic algorithmic support, not probability or
validated accuracy. The pelvis midpoint is a proxy, not COM; toe-off, COM,
stance, swing, double-support, stability metrics, and clinical outputs are omitted.
Automatic bout selection uses complete-signal and candidate-count evidence, not a
periodicity test, and requires manual confirmation before interpretation. Camera
view, direction, mirroring, and static-camera assumptions are not automatically
verified. See
`docs/GAIT_EVENT_METHOD.md` for formulas, artifact semantics, and validation
limitations.

## Detailed Rules

LLMs should read `AGENTS.md` first, then consult `.opencode/agents/`,
`.opencode/skills/`, and `.opencode/commands/` for the detailed role,
permission, scientific, and workflow instructions.
