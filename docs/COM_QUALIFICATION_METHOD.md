# COM Qualification Method (Step 5b)

## Purpose and scope

Step 5b is a standalone engineering/QC qualification stage for the Step 5a
represented-segment COM proxy. It measures how much supported model mass is
represented, identifies where coverage is lost, and describes frame- and
stride-level sample completeness. It does not recompute or change Step 5a COM
coordinates and does not calculate a gait-stability metric.

The qualification is research-only. It is not validation of physical COM,
gait stability, fall risk, diagnostic performance, or clinical performance.

## Standalone flow and command

Run Step 5a first, then run:

```bash
.venv/bin/python scripts/qualify_com.py ARTIFACT_DIRECTORY
```

The default absolute coverage sensitivity grid is
`0.80,0.82,0.84,0.86,0.88,0.90`. To supply a different fixed grid:

```bash
.venv/bin/python scripts/qualify_com.py ARTIFACT_DIRECTORY \
  --coverage-thresholds 0.80,0.85,0.90
```

The values must be finite, unique, strictly increasing, and within `[0, 1]`.
The source video is normally resolved from Step 3 inherited provenance. If the
same file has moved, `--video PATH` may override its location, but its SHA-256
must equal the inherited source hash.

Step 5b reads, but does not modify:

- Step 5a outputs: `com_proxy.csv`, `stride_com.csv`, `com_diagnostic.png`, and
  `com_metadata.json`;
- upstream artifacts: `processed_landmarks.csv`,
  `preprocessing_metadata.json`, `pose_frames.csv`,
  `reviewed_gait_events.csv`, `reviewed_strides.csv`, and
  `review_resolution_metadata.json`;
- the provenance-matched source video used to render the diagnostic overlay.

It publishes three files into the same artifact directory:
`com_qualification.json`, `com_stride_qc.csv`, and `annotated_com.mp4`. The CLI
prints the path to `com_qualification.json` on success.

## Coverage quantities

### Total-body mass coverage

Step 5a `mass_coverage` retains its original meaning:

```text
mass_coverage = sum(m_i for usable supported segments i)
```

It is the unrenormalized fraction of the published total-body model mass that
contributes to that frame's represented-segment centroid. Its units are a
dimensionless body-mass fraction. This represented-mass sum is also the
centroid denominator, but the reported coverage value is not rescaled to full
body mass. It is not a probability or confidence score.

### Theoretical supported mass

`theoretical_supported_mass_fraction` is the largest total-body mass fraction
the implemented landmark/segment model can represent when every supported
segment is usable. The structurally unsupported head is excluded:

| Anthropometric model | Published model total | Head mass | Theoretical supported maximum |
|---|---:|---:|---:|
| Male | 1.0000 | 0.0694 | 0.9306 |
| Female | 0.9999 | 0.0668 | 0.9331 |

The female maximum follows the published rounded model total of `0.9999`; it is
not calculated as `1.0000 - 0.0668`.

### Supported-mass coverage

```text
supported_mass_coverage = mass_coverage
                          / theoretical_supported_mass_fraction
```

This dimensionless ratio answers what fraction of the mass supportable by the
current implementation is represented in a frame. It does not change the COM
centroid and does not turn the proxy into whole-body COM. A high value means
high completeness relative to this model's ceiling, not high positional
accuracy, anatomical accuracy, confidence, or validity.

The calculation rejects nonfinite values, a nonpositive theoretical maximum,
negative ratios, and material ratios above 1. A floating-point excess above 1
of at most `1e-12` is clamped to exactly `1.0`.

## Camera view and coordinate interpretation

The method requires a static, near-sagittal, low-distortion,
weak-perspective-compatible side view with minimal out-of-plane motion. This
method requirement is separate from the current artifact's inherited capture
declaration. Step 5b records that declaration, whether it was machine verified,
and whether human review was recorded; it does not infer that the required view
was satisfied. In the current implementation machine verification is false and
human review of capture suitability remains required.

Coordinates are normalized 2D image-plane coordinates (`x` right, `y` down),
with no physical scale or reconstructed depth. The image axes are not laboratory
progression, vertical, gravity, or ground axes. No camera roll or tilt correction
is applied.

## Structural unsupportedness and supported-segment nonstructural missingness

The head is always reported as structurally unsupported. Standard MediaPipe
Pose does not provide the source-compatible vertex/neck joint-centre line; the
nose is not treated as the vertex. Head mass is retained for anthropometric
provenance but never contributes to a frame COM calculation, supported-segment
missing burden, or lost representable mass. Its `missingness_pattern` is
`structurally_unsupported`; its capture-missing frame count, fraction, and
longest run are emitted as zero sentinels and are semantically not applicable.

A supported segment has nonstructural missingness on a frame when its Step 5a
segment result is unusable because the required processed endpoint geometry is
not available. Such a segment contributes neither mass nor a segment centroid
on that frame. Supported-segment missingness is explicitly classified as:

- `none`: missing on zero frames;
- `intermittent`: missing on more than zero but fewer than all frames;
- `persistent`: missing on every frame.

Step 5b diagnoses this existing state; it performs no additional interpolation,
smoothing, extrapolation, or landmark fabrication.

## Inherited interpolation and smoothing

The JSON embeds the inherited Step 3 preprocessing configuration and run
identity. Interpolation records the method, maximum missing-sample gap, and
whether gaps may cross gait events. Smoothing records the method, configured
window in frames, phase, and edge behavior. Segment and direct-landmark
diagnostics also report coordinate-specific interpolation, smoothing changes,
and smoothing support that contains interpolation.

These fields establish processing provenance, not the accuracy of interpolated
or smoothed coordinates. Their presence does not make an unavailable segment
available, and Step 5b does not independently validate the resulting anatomy or
trajectory.

## Coverage diagnostics

### Segment diagnostics

For every model segment, including the unsupported head, the JSON records mass,
support status, total and usable frames/fractions, raw-observed status,
coordinate-specific interpolation, smoothing change, smoothing support that
contains interpolation, other QC limitation, missing frames/fraction, and the
longest contiguous missing run. These provenance flags are nonexclusive.

For a supported segment:

```text
lost_representable_mass = segment_mass_fraction * missing_fraction
```

The value is zero for the structurally unsupported head. It is an average
coverage-loss attribution over frames, not a positional-error estimate.

### Direct-landmark diagnostics

Required endpoint landmarks are summarized directly from
`processed_landmarks.csv`, rather than inferred from segment labels. The JSON
records raw-observed usability, x/y interpolation, x/y smoothing changes,
smoothing support containing interpolation, final missingness, and longest
missing run.

`nonexclusive_affected_mass_fraction` is the per-frame average of supported
segment mass potentially disabled on frames where that landmark is finally
missing. It sums dependent supported segments that are unusable on those
frames. It is deliberately nonexclusive: one missing segment can be attributed
to more than one required landmark, and a landmark can affect multiple
segments. Therefore these values must not be summed across landmarks as unique
lost body mass.

### Asymmetry diagnostics

Left-minus-right differences are reported for usable, raw-observed, and missing
fractions. Segment pairs cover upper arm, forearm, hand, thigh, shank, and foot.
Direct-landmark pairs cover shoulder, elbow, wrist, hip, knee, ankle, and foot
index. Segment rows include per-side mass; landmark rows use zero mass because
landmarks themselves have no anthropometric mass. These are capture/pose QC
asymmetries, not biomechanical gait asymmetry measures.

## Coverage-policy sensitivity

The configured grid contains fixed absolute `mass_coverage` thresholds. At each
threshold, Step 5b independently reports:

- eligible frame count/fraction and longest contiguous eligible interval;
- reviewed strides with any eligible frame and per-stride eligible frame counts;
- number of strides that are policy-complete at that threshold;
- normalized total, exact, adjacent-linear, usable, and unavailable counts.

It also reports `equivalent_supported_threshold = absolute_threshold /
theoretical_supported_mass_fraction` as a descriptive conversion. The policy
still operates on absolute `mass_coverage`.

The default grid is a predeclared project engineering default. It is unvalidated,
was not selected or tuned on the current artifact, and is not learned or
optimized. Step 5b does not select an optimal threshold, alter Step 5a's
inherited primary threshold, or choose a threshold to maximize availability.
The primary qualification labels always use `minimum_mass_coverage` inherited
from `com_metadata.json`; grid results are sensitivity diagnostics only.

Because the male and female theoretical supported mass fractions differ, the
same absolute `mass_coverage` threshold maps to a different
`equivalent_supported_threshold` by sex. The absolute gate itself is unchanged.

## Exact threshold and zero behavior

A frame is eligible at a tested or primary threshold exactly when all of these
are true:

```text
COM is finite
mass_coverage > 0
mass_coverage >= threshold
```

Exact equality passes. Zero mass coverage never passes, including when the
threshold is zero. A finite COM below threshold remains available in Step 5a's
source artifact but is ineligible for qualification at that threshold.

## Normalized availability

Step 5b recomputes availability at each threshold from source frame rows; it
does not reuse the Step 5a `usable` flags or normalized-row usability and does
not recompute interpolated coordinates.

For each existing canonical target timestamp from `stride_com.csv`:

- an exact timestamp match within `1e-9` is available only if that frame is
  eligible;
- otherwise, availability is linear only when the target lies strictly between
  two adjacent rows whose frame indices are consecutive, both frames are
  eligible, and their ordered `usable_segments()` tuples are identical;
- every other case is unavailable; unusable frames, gaps, boundaries, and
  changed represented-segment sets are never bridged.

The result counts availability only; Step 5b does not calculate a new normalized
COM coordinate.

## Per-stride engineering criterion

Every canonical reviewed stride is summarized over its inclusive frame bounds.
The primary-threshold diagnostics include frame and finite-COM counts,
absolute and supported coverage distributions, supported-segment missing
burden, longest unusable interval, normalized availability, and failure reasons.
`supported_segment_missing_count` is the number of distinct supported segment
names missing at least once in the stride. `supported_segment_missing_frames` is
the summed segment-frame burden: it increments once for each missing supported
segment on each frame. The values therefore need not be equal.

`policy_complete_at_threshold` means all of the following hold:

1. Every original frame is eligible at the inherited primary threshold.
2. The ordered represented-segment set is invariant across all stride frames.
3. Every canonical normalized target is available by the exact or adjacent
   same-segment-set rule above.
4. The stride start and end frames are eligible.

This is an engineering/QC completeness category, not evidence that the path is
biomechanically valid or accurate. In particular, it can pass while supported
anatomy is persistently absent if the remaining represented segment set is
invariant and its mass remains above the policy threshold.

Six independent component booleans are reported for interpretation:

- `all_original_frames_policy_eligible`: every stride frame passes the primary
  policy gate;
- `all_supported_segments_represented`: every supported segment is usable on at
  least one stride frame;
- `represented_segment_set_invariant`: the ordered usable segment tuple does not
  change across stride frames;
- `normalized_grid_complete`: every normalized target, including endpoints, is
  available under the exact/adjacent rule;
- `endpoints_policy_eligible`: both stride endpoint frames are eligible;
- `all_contributing_segments_raw_observed`: all segments in the contributing set
  are raw-observed on all stride frames.

The policy-complete category requires all original frames to be policy-eligible,
an invariant represented segment set, a complete normalized grid, and eligible
endpoints. `all_supported_segments_represented` and
`all_contributing_segments_raw_observed` are reported diagnostics but are not
required by the policy-complete definition.

Other implemented categories are:

- `usable_samples_only`: all original frames are eligible, but the segment set
  varies or normalized/end-point completeness fails;
- `insufficient_coverage`: at least one frame is not eligible at the primary
  threshold, provided at least one finite COM exists;
- `no_usable_frames`: no finite COM exists in the stride (the category name does
  not specifically mean zero primary-threshold-eligible frames);
- `endpoint_unavailable`: the reviewed stride window contains no frame rows.

## Annotated video

`annotated_com.mp4` decodes the provenance-matched source video and draws
available processed pose connections. A finite represented-segment COM proxy is
drawn green when the inherited Step 5a primary gate passes and orange when it
has positive coverage but fails that gate; no COM marker is drawn when COM is
absent. The zero-coverage red-marker branch cannot normally be visible because
zero coverage has no finite COM.

Text identifies the frame, labels the point as a represented-segment proxy, and
shows `Primary coverage gate @ threshold: ELIGIBLE/INELIGIBLE`, absolute and
supported mass coverage, absent supported segments, and containing reviewed
stride IDs. The rendered `threshold` is the inherited numeric primary threshold.
The overlay explicitly states that it is a research-only proxy and not validated
physical COM. It is a diagnostic rendering, not a new measurement artifact.

## Artifact schemas

### `com_qualification.json`

Schema version 1 and algorithm version `step5b-com-qualification-1` include:

- run identity, timestamp, scope, runtime/dependency/git provenance, and no
  randomness;
- paths and SHA-256 hashes for the source video, Step 5a outputs, and upstream
  inputs;
- inherited Step 5a run/configuration, anthropometric coefficient table,
  coverage policy, and input/output hashes;
- sex, theoretical supported mass, structural unsupported-segment explanation,
  formulas, coordinate/camera assumptions, and exact gate semantics;
- configured sensitivity grid and nested aggregate, segment, direct-landmark,
  asymmetry, stride, and per-threshold results;
- output paths/hashes, warnings, limitations, validation status,
  clean-capture status, and Step 6 readiness.

The JSON records `sha256: null` for itself because embedding its own digest would
change the digest. It records hashes for `com_stride_qc.csv` and
`annotated_com.mp4`.

### `com_stride_qc.csv`

This file has one row per reviewed stride, preserving reviewed-stride order. It
contains stride identity, bounds and review provenance; finite and
primary-eligible frame counts; absolute and supported coverage min/mean/median/
max; supported-segment missing burden; longest primary-unusable
interval; normalized exact/linear/unavailable availability; qualification
category, policy-complete-at-primary-threshold flag, six explicit component
booleans, and pipe-separated failure reasons.

### `annotated_com.mp4`

The video preserves the source frame count, dimensions, and nominal frame rate,
uses the MP4V codec, and contains the diagnostic semantics described above.

## Provenance, validation, and publication

Before calculation, Step 5b validates exact schemas and cross-artifact
relationships, including Step 3/4b lineage, Step 5a model/configuration,
coefficient table, source-video identity, frame/timestamp/status alignment,
reviewed-stride ordering, and normalized sample structure. Stored SHA-256 values
must match actual files. A relocated `--video` is accepted only by hash.

All input and source hashes are snapshotted before reading and rechecked after
the three outputs have been staged. A detected mid-run input change aborts
publication. Existing outputs are moved to UUID-named backups, all three staged
outputs are moved into place, and backups are restored if publication fails.
Backups are removed after successful publication and staging is cleaned up.

## Limitations and validation status

- Coverage describes represented mass, not landmark accuracy or COM accuracy.
- High absolute or supported coverage can coexist with biased landmarks,
  projection error, anthropometric mismatch, or an inaccurate centroid.
- Coordinates remain normalized 2D image-plane values (`x` right, `y` down),
  with no physical scale, camera calibration, depth reconstruction, or `[0, 1]`
  guarantee.
- MediaPipe points are proxies rather than anatomical joint centres; the hand
  endpoint and unsupported head limitations from Step 5a remain unchanged.
- Population-average sex-specific coefficients are not individual
  anthropometry.
- Static, near-sagittal, low-distortion, weak-perspective capture assumptions
  are user-established and not automatically verified.
- Reviewed stride boundaries and time normalization are not ground truth or
  validated gait-cycle percentage.
- Step 3 interpolation and smoothing affect the input trajectories.
- There is no force-plate, motion-capture, COM, gait-stability, fall-risk, or
  clinical validation.

## Step 6 readiness criteria

Readiness must be interpreted from evidence, not from a relaxed threshold that
makes one artifact pass. Engineering readiness for exploratory work and
scientific measurement readiness are separate decisions.

### Engineering exploratory readiness

An engineering exploratory GO requires coverage failures to be measurable and
explainable; structurally unsupported anatomy to be separated from supported-
segment nonstructural missingness; stride policy completeness and threshold
sensitivity to be characterized; and at least one independently reviewed clean
capture to demonstrate sufficiently complete proxy trajectories under a
documented, defensible QC policy.

Clean-capture review must establish that the full body, both arms, and both feet
remain visible as required; occlusion is minimal; the camera is static and the
view is near-sagittal; capture distortion and out-of-plane motion are acceptably
limited; and the required landmarks and represented-segment trajectories are
complete across the reviewed gait cycles. Step 5b does not machine-assess
all-body continuity, visibility, occlusion, orientation, camera motion,
subject/background separation, or gait-cycle adequacy, so artifact declarations
alone do not establish clean capture.

`CONDITIONAL` applies when the qualification capability is available but clean-
capture suitability or sufficiently complete clean-capture trajectories have
not been externally established. This is the status emitted by
`com_qualification.json`. Engineering NO-GO applies when appropriately captured
videos still have substantial unexplained coverage loss or when provenance or
completeness cannot be established. An engineering GO authorizes exploratory
pipeline work only; it is not evidence of COM measurement validity.

### Scientific measurement readiness

Independent validation against an appropriate reference COM measurement remains
separate scientific work. Step 5b does not assess reference-system agreement,
and scientific readiness is therefore NO-GO/not assessed for downstream claims
that require laboratory-equivalent COM or derived stability measurements. Such
claims require a defined reference protocol and accuracy, agreement,
repeatability, and population/capture-condition validation beyond this coverage
qualification.
