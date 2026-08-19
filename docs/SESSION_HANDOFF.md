# Session Hand-off

## Repository state

- Date: `2026-08-19`
- Branch: `main`
- HEAD: `f2fc44f1b7cdf95606052974e96e38e523529693`
- Working tree: uncommitted.
- User request: Step 5 exploratory video-derived 2D COM proxy. Implementation is
  complete but not committed.
- `docs/PROJECT_STATE.md` is already refreshed.

The exact intended Step 5 changed files are:

- `docs/PROJECT_STATE.md`
- `docs/COM_PROXY_METHOD.md`
- `scripts/estimate_com.py`
- `src/gait_stability/__init__.py`
- `src/gait_stability/com_estimation.py`
- `src/gait_stability/com_pipeline.py`
- `tests/test_com_estimation.py`
- `tests/test_com_pipeline.py`
- `tests/test_estimate_com_cli.py`

This hand-off, `docs/SESSION_HANDOFF.md`, is an additional file. Do not include
current unrelated `docs/theory` changes in Step 5 work.

## Implementation

Architecture:

- `com_estimation.py` contains typed configuration/data models, corrected de
  Leva coefficients, frame segment/aggregate calculations, QC provenance, and
  timestamp-based stride normalization.
- `com_pipeline.py` strictly loads and validates the six canonical upstream
  artifacts, computes every frame and reviewed stride, builds diagnostics and
  metadata, and transactionally publishes the output set.
- `scripts/estimate_com.py` is the thin CLI. `src/gait_stability/__init__.py`
  exports the public API.

Public API:

```python
from gait_stability import ComEstimationConfig, estimate_com

artifacts = estimate_com(
    artifact_directory,
    ComEstimationConfig(
        anthropometry_sex="male",  # or "female"; always explicit
        minimum_mass_coverage=0.90,
        normalized_stride_samples=101,
    ),
)
```

CLI:

```bash
.venv/bin/python scripts/estimate_com.py ARTIFACT_DIRECTORY \
  --anthropometry-sex {male,female} \
  [--minimum-mass-coverage 0.90] \
  [--normalized-stride-samples 101]
```

The four transactionally published outputs are:

- `com_proxy.csv`: per-frame represented centroid, coverage/usability,
  aggregate provenance, and per-segment results/QC.
- `stride_com.csv`: reviewed-stride identity/provenance plus all original
  in-bound frame samples and exactly `N` normalized samples per stride.
- `com_diagnostic.png`: required frame-coordinate, coverage/usability, and
  reviewed-stride diagnostic.
- `com_metadata.json`: configuration, coefficients/citation, schemas, counts,
  assumptions/limitations, runtime, and exact input/output provenance/hashes.

## Scientific decisions

- The implementation uses corrected de Leva adjusted constants. Head, trunk,
  and shank values were corrected during review. Citation: de Leva P.,
  *Journal of Biomechanics* 1996;29(9):1223-1230, DOI
  `10.1016/0021-9290(95)00178-6`; adjusted proximal-reference values are from
  Table 4.
- Anthropometric sex is explicitly required as `male` or `female`; it is never
  inferred from video, landmarks, metadata, names, or other fields.
- The 14-segment model is retained for provenance, but the head is unsupported
  because MediaPipe Pose has no defensible source-compatible vertex/neck
  joint-centre line. The other 13 represented segments are supported.
- The theoretical maximum represented mass is `0.9306` for male and `0.9331`
  for female coefficients.
- Segment centroid is `proximal + r * (distal - proximal)`. The aggregate is a
  mass-weighted **represented-segment centroid**, divided by usable represented
  mass. `mass_coverage` is the raw, unrenormalized sum of contributing published
  mass fractions.
- Coordinates are normalized image-plane `x` right and `y` down. There is no
  `z`, physical scale, calibration, physical unit, laboratory frame, or 3D
  reconstruction.
- Projection assumes static, near-sagittal, low-distortion weak-perspective or
  equivalent endpoint depth with little out-of-plane motion. These assumptions
  are user-established and not machine-verified.
- Reviewed strides define temporal windows but are not ground truth or
  force-plate-confirmed gait events.
- The default `0.90` minimum-mass-coverage policy is an engineering QC gate,
  not a positional-accuracy or validation threshold. Finite below-threshold
  centroids remain recorded but are marked unusable.

## Normalization and provenance

- Frame COM is calculated independently of strides. Every frame within a
  reviewed stride is emitted as an `original` sample.
- Normalized progression is timestamp-based:

```text
progression = (timestamp - stride_start_timestamp) / stride_duration * 100
```

- The normalized grid contains exactly `N` points including `0` and `100`.
  Exact timestamps copy the frame. Linear interpolation is allowed only between
  adjacent, consecutive, usable frames with identical represented segment sets.
  Gaps, unusable endpoints, and changed segment sets produce `method=none` and
  no centroid. Step 5 performs no landmark interpolation or extrapolation.
- Step 3 raw-observed, per-axis interpolation, smoothing change,
  smoothing-over-interpolation, missing, and other-limited provenance is
  propagated through segment and aggregate outputs. QC flags and mass sums are
  nonexclusive; processed smoothed coordinates are never called raw.

The six canonical inputs are exactly:

1. `processed_landmarks.csv`
2. `preprocessing_metadata.json`
3. `pose_frames.csv`
4. `reviewed_gait_events.csv`
5. `reviewed_strides.csv`
6. `review_resolution_metadata.json`

The pipeline accepts only canonical reviewed Step 4b lineage, with no fallback
to automatic Step 4 events/strides. It strictly checks exact schemas/header
order, canonical frame/landmark grids and timestamps, values/statuses,
event/stride endpoints and counts, versions/basenames, empty
`blocking_unresolved`, and linked upstream hashes. All six SHA-256 hashes are
snapshotted before parsing and rechecked immediately before publication. Output
paths cannot alias inputs or each other. The four outputs are staged and
published as one replacement/backup/rollback transaction; input drift or any
failure prevents partial publication.

## Verification evidence

Fresh final verification on `2026-08-19`:

- Ruff format check passed for 63 files.
- Ruff lint passed.
- mypy passed 17 source/script files.
- All 409 pytest tests passed in `141.53s`.
- `pip check` passed.
- `git diff --check` passed.
- Focused Step 5 tests passed: 197 tests in `109.45s`.
- Biomechanics review and code review found no blockers after fixes.

The recorded final suite used the repository-ignored `venv/bin` runner:

```bash
venv/bin/ruff format --check .
venv/bin/ruff check .
venv/bin/python -m mypy src scripts
venv/bin/python -m pytest
venv/bin/python -m pip check
git diff --check
```

The exact focused command was:

```bash
venv/bin/python -m pytest \
  tests/test_com_estimation.py tests/test_com_pipeline.py \
  tests/test_estimate_com_cli.py
```

The documented user environment remains `.venv/bin`; equivalent verification
commands are the same commands above with `venv/bin` replaced by `.venv/bin`.

## Aggregate smoke test

The smoke test used the user-selected male coefficient set and current ignored
`outputs/A_Video` artifacts:

```bash
venv/bin/python scripts/estimate_com.py outputs/A_Video \
  --anthropometry-sex male
```

Aggregate evidence only:

- Inputs: 31 reviewed event rows, 16 accepted events, and 14 reviewed strides.
- Corrected shared boundaries at frames 86 and 140 were consumed.
- `com_proxy.csv`: 301 frame rows, 300 finite centroids, zero usable at `0.90`,
  maximum coverage `0.8812`. The limiting omissions were unsupported head plus
  unavailable left arm.
- Normalized output: `14 x 101` rows, zero usable, 65 finite exact, and 1,349
  `method=none`.
- Recorded hashes matched and the diagnostic was readable.

Do not add participant details. Generated artifacts, videos, subject data,
landmarks, overlays, and current `outputs/A_Video` files must not be committed.

## Readiness and next actions

Step 5 software is complete. The current artifact is **NO-GO** for downstream
feature extraction because no frame or normalized sample passes the default
coverage gate. Next work should improve capture/pose coverage and perform
external reference validation. A stability score is not the next justified
step.

For the next session:

1. Inspect `git status` and the complete `git diff`, confirming only the intended
   Step 5 files and this hand-off are selected.
2. Optionally rerun the final suite using the commands above.
3. Decide with the user whether to commit; do not commit automatically.
4. Keep ignored generated outputs and unrelated `docs/theory` changes excluded.

Tests demonstrate software behavior and contract enforcement. They do **not**
validate COM accuracy, gait stability, fall risk, diagnosis, or any clinical
interpretation.
