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

Artifacts are staged completely before output replacement. Reported rename
failures trigger rollback of the prior result, but replacing an existing output
requires two same-filesystem renames. Interruption between them can leave the
prior result in a hidden `.backup-*` sibling requiring manual recovery. This
stage only records video properties; it performs no pose, gait, biomechanical,
or clinical assessment.

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
application does not import them directly. MediaPipe itself is installed
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
the source path/hash is stale, it runs inspection first. It stages and replaces
only these Step 2 files under `outputs/<video_stem>/`:

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

Visibility and presence are raw model scores, not observed or calibrated
probabilities, uncertainty, or accuracy. Backend thresholds control backend
processing and are not validated quality cutoffs. MediaPipe visibility is not
renamed as generic confidence. `decoded_pose` means only that the backend returned
a nonempty pose result. "Raw" means selected backend fields are unchanged by
project postprocessing; MediaPipe `VIDEO` mode may internally track or temporally
smooth. World landmarks and segmentation masks are deliberately omitted.
There is no project interpolation, gait-event detection, COM estimation, gait or
stability metric, biomechanical validation, or clinical interpretation.
Landmarks are not anatomical joint centers, laboratory coordinates, or clinical
measurements, and outputs have not been validated against labels or a reference
measurement system.

## Detailed Rules

LLMs should read `AGENTS.md` first, then consult `.opencode/agents/`,
`.opencode/skills/`, and `.opencode/commands/` for the detailed role,
permission, scientific, and workflow instructions.
