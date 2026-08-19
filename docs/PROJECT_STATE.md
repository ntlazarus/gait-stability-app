# Project State

## Snapshot metadata

- Snapshot UTC: `2026-08-19T00:39:12Z`
- Git branch: `main`
- HEAD: `a0845d11cac3039176a1aa0af6dffa0f1caaa19c`
- Working tree: dirty; uncommitted Step 4b source, tests, and documentation are
  present. Generated outputs are ignored, and no subject-derived temporary files
  remain in `git status`. Unrelated untracked docs/theory are excluded from this
  feature.

## What the project currently is

The executable MVP is a local, research-only Python pipeline with four full
stages plus an optional manual-review resolution stage:

1. Step 1 inspects MP4/MOV video and saves metadata and representative frames.
2. Step 2 estimates raw monocular pose with MediaPipe Tasks Pose Landmarker 1.0.0.
3. Step 3 quality-gates pose, interpolates bounded gaps, smooths planar
   trajectories, and records complete QC and preprocessing provenance.
4. Step 4 selects an analyzable interval, detects video-derived candidate initial
   contacts (ICs), constructs candidate stride intervals, and creates diagnostic
   and annotated review artifacts.
5. Step 4b resolves manual review and correction of automatic gait-event and
   stride detections, producing reviewed event and stride outputs with full
   provenance.

The outputs remain research artifacts. The project has no validated measurements,
toe-off, stance/swing/double-support metrics, COM estimate, stability metric, fall-
risk result, diagnosis, UI, or API.

## Current capabilities

### Step 1: video inspection

- CLI: `python scripts/inspect_video.py INPUT [--output-root OUTPUTS]`.
- API: `gait_stability.inspect_video(video_path, output_root)`.
- Validates a local `.mp4` or `.mov` decoder, dimensions, nominal FPS, and nominal
  integer frame count; records source hash/size, OpenCV/backend orientation status,
  nominal timing, and frames requested at 0%, 25%, 50%, 75%, and 100%.
- A failed representative decode tries at most 10 preceding indices and records
  requested versus actual indices. Publication uses same-directory staging,
  replacement, and rollback paths.

### Step 2: raw pose estimation

- CLI: `python scripts/estimate_pose.py INPUT --model MODEL.task [options]`.
- API: `gait_stability.estimate_pose_video(video_path, estimator, output_root)` via
  the backend-independent `PoseEstimator` contract.
- Uses MediaPipe Tasks 1.0.0 in CPU `VIDEO` mode, one pose, and an explicit local
  model. Matching Step 1 artifacts are reused; otherwise inspection runs first.
- Records every nominal slot as `decoded_pose`, `decoded_no_pose`, or
  `decode_failure`; the annotated video retains order and uses black placeholders
  for decode failures.
- Preserves 33 canonical landmark rows with normalized image `x`/`y`, learned
  backend-relative `z`, visibility, presence, and nullable generic confidence.
  World landmarks, segmentation masks, and audio are omitted.

### Step 3: pose quality and preprocessing

- CLI: `python scripts/preprocess_pose.py ARTIFACT_DIRECTORY [options]`.
- API: `gait_stability.preprocess_pose(directory_or_contract, config)`.
- Requires canonical Step 2 schema-version-2 artifacts, validates schemas and
  hashes, and creates a complete nominal-frame by 33-landmark audit grid while
  preserving raw values and scores.
- Defaults require finite visibility and presence each `>=0.5`; generic confidence
  is disabled. These are engineering heuristics, not calibrated accuracy cutoffs.
- Processes normalized image-plane `x` and `y` independently. It linearly
  interpolates only interior scalar gaps of at most 3 frames between raw-observed
  usable endpoints, with no extrapolation, recursive interpolation, `z`, or score
  processing.
- Applies a centered, unweighted 3-frame boxcar within contiguous nonmissing scalar
  segments; window `1` disables smoothing. Reports coverage, rejection, gaps,
  interpolation, smoothing support, remaining missingness, hashes, configuration,
  runtime, and inherited provenance without assigning a quality label.

### Step 4: candidate gait events and strides

- CLI: `python scripts/detect_gait_events.py ARTIFACT_DIRECTORY --walking-direction
  {image_right,image_left} [options]`.
- API: `gait_stability.detect_gait_events(artifact_directory, config, video_path=...)`.
- Requires canonical `processed_landmarks.csv`, `pose_quality.json`, and
  `preprocessing_metadata.json`; it validates their contract and hashes and
  preserves Step 1-3 inputs unchanged.
- Uses the direction-normalized ipsilateral heel `x` relative to the bilateral hip
  midpoint proxy:

  ```text
  s_side(t) = d * (heel_side_x(t) - (left_hip_x(t) + right_hip_x(t)) / 2)
  d = +1 for image_right; d = -1 for image_left
  ```

- Forms candidates at strict local maxima (one earlier-midpoint candidate for a
  flat peak), then applies configurable prominence, forward-position, and velocity-
  reversal gates. Defaults are peak radius 2 frames, prominence window 10 frames,
  minimum prominence `0.02` normalized image width, reversal half-window 2 frames,
  derivative deadband `0`, and minimum forward relative `x` of `0`.
- Raw heel-plus-bilateral-hip and processed ankle-plus-bilateral-hip local peaks,
  each within 2 frames by default, are correlated support cues rather than
  independent corroboration. Primary, ankle, and hip observed-usability,
  interpolation, and interpolation-affected smoothing provenance are retained over
  the exact relevant support.
- Step 4 adds no smoothing or interpolation. Missing processed primary `x` splits
  the signal; no event or terminal sample is fabricated. All formed local maxima,
  including rejected candidates, remain auditable.
- Without manual bounds, qualifying bout candidates are maximal contiguous runs
  with complete bilateral primary processed signals, duration `>=3` seconds, and
  at least 2 accepted preliminary candidates per side. Selection ranks total
  accepted count, duration, then earlier start. This is not periodicity, walking
  onset/offset, or steady-state classification.
- Paired manual start/end bounds are inclusive and mean only user-selected analysis.
  If no automatic run qualifies, `full_recording_fallback` retains the entire stored
  nominal range without claiming it is all walking. Every mode requires manual
  confirmation.
- Default hard minimum gates are `0.5` seconds for same-side events and `0.15`
  seconds for opposite-side events; equality passes. A same-side interval over `3`
  seconds is retained with a warning. Nonalternation is also retained as QC context.
- Accepted events are `high` only when the full primary support is raw-observed
  usable and interpolation-clean, a raw peak agrees, and the ankle-plus-bilateral-
  hip cue is present and clean. Other accepted events are `review`; rejected
  candidates are `low`. These are deterministic support labels, not accuracy or
  probability. All event and stride rows default to `review_status=unreviewed`.
- Candidate strides pair consecutive accepted same-side candidate ICs. They are
  segmentation/QC intervals, not validated stride measurements.

### Step 4b: review resolution

- CLI: `.venv/bin/python scripts/resolve_gait_reviews.py ARTIFACT_DIRECTORY
  ASSUMPTION_RESPONSE_DOCUMENT`.
- API: `gait_stability.resolve_gait_reviews(artifact_directory,
  assumption_response_document)`.
- Requires seven canonical Step 4 artifacts in the artifact directory:
  `gait_events.csv`, `strides.csv`, `gait_event_reviews.csv`,
  `strides_reviews.csv`, `gait_event_metadata.json`, `pose_frames.csv`, and
  `preprocessing_metadata.json`. Also requires an explicit plain-text assumption
  response document path. All input hashes are captured before processing and
  rechecked before publication.
- Canonical timestamps are read from `pose_frames.csv` and validated against
  `preprocessing_metadata.json` hash. No frame-index or FPS recomputation is
  performed; corrected frames use exact `pose_frames.csv` lookups.
- Review statuses for events: `unreviewed` (baseline preserved),
  `retain_rejection` (confirmed rejection), `promote_to_candidate` (rejected→accepted).
  Stride review statuses: `accept` (unchanged boundaries) or `correct` (frame edit).
- Bounded promotion rule: a `promote_to_candidate` event must be linked as the
  replacement target of an event-level stride boundary correction from a
  `correct`-status stride review. Standalone promotions without a linked stride-
  boundary correction are rejected before publication.
- Resolved events preserve automatic frame/time/status/quality/provenance fields
  and add reviewed frame, timestamp, accepted/rejected flags, reviewed quality,
  resolution disposition, and full provenance notes.
- `reviewed_quality` is a Step 4b QC review category, not an accuracy label.
  `"review"` marks corrected or promoted boundaries; `"high"`/`"low"` carry the
  detector's `automatic_quality` forward for unchanged events; replaced events are
  `"low"` as a Step 4b exclusion/QC flag.
- Reviewed strides are regenerated by calling `construct_strides` on reviewed
  `GaitEvent` copies with reviewed frame/time/status/inclusion/quality. Shared
  consecutive same-side boundaries propagate, and dependent start/end timestamps,
  durations, and contralateral info are recomputed. Automatic stride IDs and review
  provenance are preserved.
- Three outputs are transactionally published: `reviewed_gait_events.csv`,
  `reviewed_strides.csv`, and `review_resolution_metadata.json`. Original automatic
  and manual-review inputs are untouched throughout.
- Original automatic/manual-review inputs are preserved; input hashes are
  independently confirmed unchanged after processing; consistency assertions are
  passed; blocking review-resolution ambiguity is empty.
- Assumption document content is hashed provenance only and is not machine-
  evaluated. `scientific_unresolved` items are carried downstream. Reviewed
  intervals are candidate temporal segmentation/QC windows using nominal pose-frame
  timestamps; they are not force-plate-confirmed contacts, validated stride
  durations, or measured toe-off/stance/spatial metrics.

## Quick start

Python 3.11 or newer is supported. From the repository root, create the headless
MediaPipe 1.0.0 environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,pose]'
.venv/bin/python -m pip install --no-deps mediapipe==1.0.0
.venv/bin/python -m pip install ./compat/opencv-contrib-python-headless-provider
.venv/bin/python -m pip check
```

MediaPipe declares GUI OpenCV, so `--no-deps` plus the local metadata-only provider
retains the actual headless OpenCV implementation. Obtain the full Pose Landmarker
`.task` manually as described in `models/README.md`; the application does not
download models.

Run all four stages plus optional review resolution:

```bash
.venv/bin/python scripts/inspect_video.py path/to/walk.mp4 --output-root outputs
.venv/bin/python scripts/estimate_pose.py path/to/walk.mp4 \
  --model models/pose_landmarker_full.task --output-root outputs
.venv/bin/python scripts/preprocess_pose.py outputs/walk
.venv/bin/python scripts/detect_gait_events.py outputs/walk \
  --walking-direction image_right
# Step 4b (optional, requires manual review CSVs and assumption document):
.venv/bin/python scripts/resolve_gait_reviews.py outputs/walk \
  path/to/assumption_responses.txt
```

## Key workflows

### Inspect a video

```bash
.venv/bin/python scripts/inspect_video.py path/to/walk.mp4 --output-root outputs
```

- Input: one readable local `.mp4` or `.mov` outside its deterministic destination.
- Output: `outputs/<stem>/video_metadata.json` and JPEG observations under
  `sample_frames/`.
- Success: the metadata path prints after exact or recorded bounded-fallback sample
  decoding.

### Estimate raw pose

```bash
.venv/bin/python scripts/estimate_pose.py path/to/walk.mp4 \
  --model models/pose_landmarker_full.task --output-root outputs \
  --min-pose-detection-confidence 0.5 \
  --min-pose-presence-confidence 0.5 \
  --min-tracking-confidence 0.5
```

- Inputs: a Step 1-compatible video and existing Pose Landmarker `.task` file.
- Key options: the three backend thresholds default to `0.5` and accept `[0,1]`;
  `--output-root` defaults to `outputs`.
- Outputs: `raw_landmarks.csv`, `pose_frames.csv`, `pose_metadata.json`, and
  `annotated_pose.mp4` beside Step 1 artifacts.

### Quality-assess and preprocess pose

```bash
.venv/bin/python scripts/preprocess_pose.py outputs/walk \
  --visibility-threshold 0.5 --presence-threshold 0.5 \
  --max-gap-frames 3 --smoothing-window-frames 3
```

- Input: canonical Step 2 artifacts in one artifact directory.
- Key options: `--disable-visibility`, `--disable-presence`, `--enable-confidence`,
  `--diagnostic-landmarks`, and `--no-diagnostic`.
- Outputs: `processed_landmarks.csv`, `pose_quality.json`,
  `preprocessing_metadata.json`, and normally `pose_trajectory_diagnostic.png`.

### Detect candidate ICs and strides

```bash
.venv/bin/python scripts/detect_gait_events.py outputs/walk --walking-direction image_right
```

- Inputs: the three canonical Step 3 artifacts and the inherited source video.
  `--video` may override only the path; its SHA-256 must match inherited source
  provenance.
- Manual interval: provide both `--manual-start-frame START` and
  `--manual-end-frame END`; both bounds are inclusive.
- Key detector options: `--peak-radius-frames`, `--prominence-window-frames`,
  `--min-prominence`, `--reversal-half-window-frames`, `--derivative-deadband`,
  `--min-forward-relative-x`, `--raw-agreement-window-frames`,
  `--ankle-agreement-window-frames`, `--same-side-min-interval-seconds`,
  `--opposite-side-min-interval-seconds`, and
  `--same-side-max-interval-warning-seconds`.
- Key workflow options: `--automatic-minimum-bout-duration-seconds`,
  `--automatic-minimum-accepted-events-per-side`, and
  `--event-flash-radius-frames`.
- Six outputs: `walking_bout.json`, `gait_events.csv`, `strides.csv`,
  `gait_event_diagnostic.png`, `annotated_gait_events.mp4`, and
  `gait_event_metadata.json` in the same artifact directory.
- Success: the metadata path prints after source and input hashes are rechecked and
  the complete six-file set is transactionally published.

### Resolve manual review (Step 4b)

```bash
.venv/bin/python scripts/resolve_gait_reviews.py outputs/walk \
  path/to/assumption_responses.txt
```

- Inputs: seven Step 4 artifacts in the artifact directory (`gait_events.csv`,
  `strides.csv`, `gait_event_reviews.csv`, `strides_reviews.csv`,
  `gait_event_metadata.json`, `pose_frames.csv`,
  `preprocessing_metadata.json`) plus an explicit plain-text assumption response
  document.
- All input SHA-256 hashes are captured before processing and rechecked before
  publication. `pose_frames.csv` hash is validated against
  `preprocessing_metadata.json`.
- Event review statuses: `unreviewed`, `retain_rejection`, `promote_to_candidate`.
  Stride review statuses: `accept`, `correct`. Every automatic event must have
  exactly one review row; every automatic stride must have exactly one review row.
- Bounded promotion rule: `promote_to_candidate` must be linked as the replacement
  target of a stride-boundary correction; standalone promotions are rejected.
- Three outputs: `reviewed_gait_events.csv`, `reviewed_strides.csv`, and
  `review_resolution_metadata.json`.
- Reviewed strides are regenerated through `construct_strides` with reviewed
  frame/time/status/inclusion/quality; shared consecutive same-side boundaries
  propagate; dependent timestamps, durations, and contralateral info are recomputed.
- Original automatic and manual-review inputs are untouched; transactional publish
  with UUID backups and rollback.
- Success: the metadata path prints after input hash reconfirmation and consistency
  assertions pass.

## Inputs

- Video selection is extension-based (`.mp4`/`.mov`); codec/container support,
  seeking, frame positions, and orientation behavior depend on OpenCV. Same-stem
  videos share a destination; a source nested in its destination is rejected.
- Step 2 uses monocular RGB and an explicit ignored local `.task` model. It does not
  require or verify camera view, direction, placement, static capture, or mirroring.
- There is no camera calibration, physical scale, distortion correction,
  anthropometry, multi-view reconstruction, or reference gait-event input.
- Step 3 accepts only the canonical Step 2 CSV/JSON contract. Finite normalized
  coordinates outside `[0,1]` remain usable but flagged; nonfinite values are
  unusable.
- Step 4 expects a manually established anatomical left/right interpretation,
  stored-image walking direction, and a near-sagittal static-camera treadmill or
  compatible capture. These assumptions are not automatically verified.
- Step 4b requires manually prepared `gait_event_reviews.csv` (one row per
  automatic event with review status) and `strides_reviews.csv` (one row per
  automatic stride with review status and optional frame edits), plus an explicit
  plain-text assumption response document whose content is hashed only and not
  machine-evaluated.
- All timestamps are nominal frame-derived times, not verified presentation times.

## Outputs

For input stem `walk`:

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
|-- pose_trajectory_diagnostic.png
|-- walking_bout.json
|-- gait_events.csv
|-- strides.csv
|-- gait_event_diagnostic.png
|-- annotated_gait_events.mp4
|-- gait_event_metadata.json
|-- reviewed_gait_events.csv          (Step 4b)
|-- reviewed_strides.csv              (Step 4b)
|-- review_resolution_metadata.json   (Step 4b)
```

- Observed video: decoded Step 1 JPEGs and source/decoder metadata.
- Pose-model estimates: Step 2 landmarks and pose overlay; normalized `x`/`y` and
  backend-relative `z` are not laboratory coordinates or anatomical ground truth.
- Step 3 processed trajectories: complete audit grid, heuristic usability flags,
  bounded interpolants, smoothed normalized trajectories, descriptive QC,
  diagnostics, and provenance.
- Step 4 unvalidated research proxies: selected interval, accepted and rejected
  candidate IC rows, candidate same-side stride intervals, diagnostic plot, overlay,
  and method/run provenance. `high` is algorithmic support only; all rows are
  `unreviewed` by default.
- Step 4b reviewed outputs: reviewed event and stride rows with automatic and
  reviewed frame/time/status/quality/provenance; `reviewed_quality` is a Step 4b
  QC review category, not an accuracy label. Original inputs are untouched.
- No artifact is a validated event, physical measurement, stability/fall-risk
  estimate, diagnosis, or clinical result.

## Configuration

- `pyproject.toml`: Python `>=3.11`, headless OpenCV, optional pose/plot and dev
  dependencies, plus Ruff, mypy, and pytest settings. Mypy parses as Python 3.13.
- `requirements-pose-headless.txt` and
  `compat/opencv-contrib-python-headless-provider/`: headless MediaPipe dependency
  setup.
- `models/README.md`: manual model acquisition and hash/size provenance.
- `docs/GAIT_EVENT_METHOD.md`: Step 4 formula, coordinate/unit semantics, quality
  rules, artifact definitions, and validation limits.
- `docs/Step4b_review_resolution.md`: Step 4b review resolution semantics,
  artifact contracts, and validation details.
- Step 2 defaults: detection, pose presence, and tracking thresholds `0.5`.
- Step 3 defaults: visibility/presence `0.5`, generic confidence disabled, maximum
  gap 3 frames, centered boxcar window 3, and ankle/heel/hip diagnostic enabled.
- Step 4 defaults are recorded in `gait_event_metadata.json`; walking direction is
  always required rather than defaulted by the CLI.
- Step 4b requires both positional CLI arguments (`artifact_directory` and
  `assumption_response_document`); no additional options are available.

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

Focused Step 4b tests:

```bash
.venv/bin/python -m pytest tests/test_review_resolution.py
```

The orchestrator's fresh final verification on 2026-08-19 used the repository-
ignored Linux `venv/bin` environment because `.venv` is incompatible on this
runner; the documented user commands remain `.venv/bin`. `ruff format --check .`
passed for 57 files; `ruff check .` passed; mypy passed 14 source/script files;
pytest collected and passed all 212 tests in 30.77 seconds with no skips; `pip
check` reported no broken requirements; and `git diff --check` passed. Tests
cover Steps 1-3 plus Step 4 formulas, direction handling, peak and temporal
gates, missing/provenance behavior, bout modes, schemas, rendering, transactional
replacement, CLI errors, source/input hash preservation, and Step 4b review
resolution (23 tests).

### Aggregate current lab exercise

Only aggregate, non-identifying evidence supplied from the current ignored lab run
is recorded; this is not participant description or clinical evidence:

- Automatic bout: frames 0-299 inclusive, nominal `0-9.9667 s`.
- Accepted candidate ICs: 8 left at frames
  `[12,48,82,121,157,193,229,266]` and 8 right at
  `[30,66,102,138,175,211,247,283]`, with exact accepted-event alternation.
- Candidate strides: 7 per side. Left durations ranged `1.133-1.300 s`, median
  `1.200 s`; right durations ranged `1.200-1.233 s`, median `1.200 s`.
- Fifteen rejected low local maxima were preserved. No accepted candidate had
  interpolation-affected support.
- The final actual Step 4 rerun passed with automatic frames 0-299 and unchanged
  event, stride, and rejected-candidate counts. The diagnostic PNG was readable.
- The annotated MP4 was 1440x1080 at 30 fps and retained 301 frames, including the
  terminal decode-failure placeholder. Step 1-3 artifact hashes were unchanged.
- Manual visual checks at early, middle, and late accepted candidates on both sides
  found frame-level plausible leading-heel contact.

Step 4b aggregate evidence for the same artifact directory:

- 31 automatic candidate events in the input `gait_events.csv`.
- 16 events accepted after resolution: 14 accepted unchanged, 1 corrected, 1
  promoted from automatic rejection.
- 15 events nonaccepted after resolution including 1 replaced original; 14
  reviewed candidate stride intervals.
- Specific boundary changes: E0007 auto left frame82 replaced by promoted E0009
  frame86; canonical 2.8666666666666667 s. E0012 auto right frame138 corrected to
  reviewed140; canonical 4.666666666666667 s.
- Affected stride boundaries: S0003 end=S0005 start=86; S0006 end=S0008 start=140.
  Affected durations: S0003 1.2666666666666666, S0005 1.1666666666666665,
  S0006 1.266666666666667, S0008 1.166666666666666 seconds.
- Original input SHA hashes independently confirmed unchanged; consistency
  assertions passed; blocking review-resolution ambiguity empty; scientific capture
  caveats remain.

Readiness is conditional: the exercise supports manual review, correction, and
sensitivity work only. It has no reference-system validation, does not validate event
timing or stride duration, and is not identifying, diagnostic, or clinical evidence.

## Known limitations

- Timing is nominal frame index/FPS; frame count may exceed decodable content and
  event/contact timing bias is unknown.
- Monocular image projection has no physical scale or laboratory frame. Pose labels
  are model estimates, not anatomical joint centers or measured contact markers.
- Capture direction, anatomical side, view, mirroring, camera placement, static-
  camera context, and manual bounds require human establishment or confirmation.
- Step 3 interpolation can cross an unknown event, and its centered smoothing can
  attenuate or alter extrema, derivatives, thresholds, and Step 4 event timing.
- Step 4 detector thresholds are capture/framing dependent. Raw and ankle cues are
  correlated with the primary trajectory, not independent evidence.
- Automatic bouts use signal completeness and candidate counts, not periodicity or
  objective walking/steady-state classification. Fallback bounds infer no walking.
- Automatic Step 4 event/stride rows start `unreviewed`; `high` is deterministic
  support only. Step 4b reviewed outputs carry resolved statuses. Local heel maxima
  may be biased relative to physical contact.
- Rejected local peaks and abrupt foot departures may reflect true motion, tracking
  spikes, swaps, occlusion, or other artifacts. No automated spike, swap, tracking-
  discontinuity, camera-motion, or phase-dependent-missingness QC exists.
- Step 4b `reviewed_quality` is a QC review category, not validated accuracy.
  Promoted candidates retain automatic rejection provenance. Manual corrections are
  not force-plate or reference-validated.
- Assumption document content is hashed provenance only and is not machine-
  evaluated; scientific capture caveats remain unresolved and are carried downstream.
- No toe-off, stance, swing, double support, COM, gait/stability metric, fall risk,
  diagnosis, UI, or API is implemented.
- There are no manual reference labels, force plates, marker/reference-system
  comparisons, repeated-session studies, population validation, or clinical
  validation.
- Event and stride IDs are deterministic only for the same inputs, run
  configuration, and algorithm; they are not stable across configuration or method
  changes. Run IDs are run-scoped.
- Abrupt interruption or cleanup failure can leave visible `*.staging-*` or
  `*.backup-*` transaction recovery files. A backup is the prior published artifact;
  a staging path is unpublished/incomplete and must not be merged with the output.

## Repository map

```text
src/gait_stability/   Steps 1-4 plus 4b reusable ingestion, pose, QC, event, and review-resolution APIs
scripts/              Thin CLIs for inspect, estimate, preprocess, detect, and resolve
tests/                Deterministic unit, CLI, and pipeline-boundary tests
docs/                 Method documentation and this canonical state snapshot
models/README.md       Local ignored pose-model acquisition/provenance instructions
compat/                Metadata-only headless OpenCV compatibility distribution
outputs/               Ignored generated artifacts
data/                  Ignored local data; subject data must not be committed
.opencode/             Agent, command, and scientific workflow configuration
```

## Next logical capabilities

- Validate candidate event timing against manual labels and an appropriate marker,
  force, or other reference system; evaluate capture and smoothing sensitivity.
- Add tracking spike, left/right swap, occlusion, and discontinuity QC.
- Explore Step 5 COM proxies only after Step 4 review and sensitivity work; do not
  treat exploratory COM or stability outputs as validated measurements.
