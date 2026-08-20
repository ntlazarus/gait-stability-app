# COM Proxy Method

## Citation

This implementation uses the de Leva 1996 anthropometric model (adjustments to
Zatsiorsky-Seluyanov):

> de Leva P. Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters.
> *Journal of Biomechanics*. 1996;29(9):1223-1230.
> DOI: [10.1016/0021-9290(95)00178-6](https://doi.org/10.1016/0021-9290(95)00178-6)

Coefficient r values are from Table 4 of that paper (adjusted-parameter values,
proximal reference). Visual3D adjusted-parameter documentation may be recorded
as independent transcription.

## Overview

Step 5 computes a **represented-segment mass-weighted 2D centroid proxy** from
processed 2D pose landmarks. The centroid is renormalized by the sum of usable
segment mass fractions (the represented body mass). It is an image-plane proxy,
not a 3D or laboratory measurement. No physical scale, camera calibration, or
gait-stability metric is produced.

The denominator for the centroid is the represented usable mass (sum of usable
mass fractions), **not** the full model total or body mass. `mass_coverage` is
the only unrenormalized quantity: the raw sum of usable mass fractions without
rescaling.

Step 5b qualifies the completeness and provenance of these existing outputs
without changing Step 5a coordinates. See
[`COM_QUALIFICATION_METHOD.md`](COM_QUALIFICATION_METHOD.md) for the standalone
command, diagnostic artifacts, threshold sensitivity, and stride-level
engineering criteria.

## Unsupported segments

The head segment is always unavailable because standard MediaPipe Pose lacks
a defensible source-compatible vertex/neck joint-centre line. Nose is **not**
the vertex; the published source figure/reference must be consulted and the
implementation omits head entirely. Head coefficient and mass are retained in
the model for provenance but never participate in frame COM calculations.

Maximum represented body-mass fraction when all supported segments are usable:

- male: 0.9306 (= 1.0000 - 0.0694)
- female: 0.9331 (= 0.9999 - 0.0668)

Full model coverage cannot occur because head is unsupported.

## Camera view requirements

This implementation requires a single 2D side-view (sagittal or near-sagittal).
Depth is not reconstructed. The coordinate values are not constrained to [0, 1];
they can extend beyond the normalized image bounds depending on the pose
estimator and camera setup.

### Projection assumptions

The implementation requires static, near-sagittal, low-distortion,
weak-perspective/equivalent endpoint-depth capture with minimal out-of-plane
motion. These assumptions are user-established and not machine-verified.
Affine segment-fraction projection is not generally valid under perspective
depth differences.

## Anthropometric coefficients

The 14-segment model uses the following mass fractions and COM ratios (r,
proximal-to-distal, 0 = proximal endpoint, 1 = distal endpoint).
These values are from de Leva 1996, Table 4 (adjusted-parameter r values,
proximal reference).

### Male (de Leva 1996, Table 4)

| Segment     | Mass fraction | r     |
|-------------|---------------|-------|
| Head        | 0.0694        | 0.5002|
| Trunk       | 0.4346        | 0.5138|
| Upper arm   | 0.0271        | 0.5772|
| Forearm     | 0.0162        | 0.4574|
| Hand        | 0.0061        | 0.7900|
| Thigh       | 0.1416        | 0.4095|
| Shank       | 0.0433        | 0.4395|
| Foot        | 0.0137        | 0.4415|

Published model total: 1.0000

### Female (de Leva 1996, Table 4)

| Segment     | Mass fraction | r     |
|-------------|---------------|-------|
| Head        | 0.0668        | 0.4841|
| Trunk       | 0.4257        | 0.4964|
| Upper arm   | 0.0255        | 0.5754|
| Forearm     | 0.0138        | 0.4559|
| Hand        | 0.0056        | 0.7474|
| Thigh       | 0.1478        | 0.3612|
| Shank       | 0.0481        | 0.4352|
| Foot        | 0.0129        | 0.4014|

Published model total: 0.9999

Bilateral segments (upper arm, forearm, hand, thigh, shank, foot) use the same
mass fraction and r per side. The total includes both left and right instances.

## Segment endpoint mappings

### Source model endpoints (published anatomical definitions)

The de Leva 1996 model defines segments using joint-centre-based endpoints:

| Segment   | Proximal endpoint              | Distal endpoint                    |
|-----------|--------------------------------|-------------------------------------|
| Head      | Vertex                         | Neck joint-centre                   |
| Trunk     | Shoulder joint-centre midpoint | Hip joint-centre midpoint           |
| Upper arm | Shoulder joint-centre          | Elbow joint-centre                  |
| Forearm   | Elbow joint-centre             | Wrist joint-centre                  |
| Hand      | Wrist joint-centre             | Third metacarpal head               |
| Thigh     | Hip joint-centre               | Knee joint-centre                   |
| Shank     | Knee joint-centre              | Ankle joint-centre                  |
| Foot      | Ankle joint-centre             | Toe                                 |

**Head**: The exact source figure/reference must be consulted; implementation
omits head. Do not invent or call nose the vertex.

### Proxy mappings (MediaPipe 2D landmarks)

These are the MediaPipe 2D landmark endpoints used as segment proxies.
They are **not** anatomical joint centers.

| Segment         | Proximal landmark                    | Distal landmark                          |
|-----------------|--------------------------------------|------------------------------------------|
| Head            | _unsupported_head_proximal           | _unsupported_head_distal                 |
| Trunk           | shoulder_midpoint                    | hip_midpoint                             |
| Left upper arm  | left_shoulder                        | left_elbow                               |
| Right upper arm | right_shoulder                       | right_elbow                              |
| Left forearm    | left_elbow                           | left_wrist                               |
| Right forearm   | right_elbow                          | right_wrist                              |
| Left hand       | left_wrist                           | left_index_pinky_midpoint_proxy          |
| Right hand      | right_wrist                          | right_index_pinky_midpoint_proxy         |
| Left thigh      | left_hip                             | left_knee                                |
| Right thigh     | right_hip                            | right_knee                               |
| Left shank      | left_knee                            | left_ankle                               |
| Right shank     | right_knee                           | right_ankle                              |
| Left foot       | left_ankle                           | left_foot_index                          |
| Right foot      | right_ankle                          | right_foot_index                         |

### Derived landmarks (computed per frame)

- `shoulder_midpoint` = midpoint(left_shoulder, right_shoulder)
- `hip_midpoint` = midpoint(left_hip, right_hip)
- `left_index_pinky_midpoint_proxy` = midpoint(left_index, left_pinky)
- `right_index_pinky_midpoint_proxy` = midpoint(right_index, right_pinky)

Derived endpoints are the arithmetic mean of their component MediaPipe
landmarks. Their provenance tracks the underlying component landmarks.

**Hand proxy note**: The midpoint of MediaPipe index and pinky points is an
unvalidated distal hand endpoint surrogate. It is not the anatomical hand
midpoint or third metacarpal head defined in the published model.

## Formulas

### Segment COM

For segment with proximal endpoint p and distal endpoint d:

```
c = p + r * (d - p)
```

where r is the published COM ratio from de Leva 1996.

### Whole-body COM proxy (represented-segment centroid)

```
COM_x = sum_i(m_i * c_ix) / sum_i(m_i)
COM_y = sum_i(m_i * c_iy) / sum_i(m_i)
```

where the sum is over all **usable supported** segments i (head is always
excluded), m_i is the published mass fraction, and c_i is the segment COM.

**Centroid denominator**: The denominator is the sum of usable mass fractions
only, not the total model mass or body mass. This is the represented body
mass fraction. Missing-mass segments and unsupported segments are excluded
from the weighted average and the denominator. The COM is therefore a
**represented-segment centroid**, renormalized by represented usable mass,
not an unrenormalized whole-body COM.

**Exact-unusable behavior**: Frames with zero usable mass have `com=None`
and `usable=False` regardless of the threshold setting. Frames with nonzero
usable mass below the threshold retain their finite COM proxy but are marked
`usable=False`.

## Endpoint source vs. proxy

Endpoints are processed 2D pose landmarks from Step 3. They are not
anatomical joint centers, not force-plate contacts, and not measured
anatomy. Derived endpoints (midpoints) are computed as the arithmetic mean
of their component landmarks.

## Coordinate system

All calculations operate in normalized image coordinates:
- x: image left to right
- y: image top to bottom

Coordinate values are **not** constrained to [0, 1]. The normalization
depends on the pose estimator and may extend beyond the normalized image
bounds.

No physical scale, camera calibration, or 3D reconstruction is performed.

## Quality control

### Mass coverage threshold

`mass_coverage` is the raw represented body-mass-fraction sum (CSV field
name preserved as `mass_coverage` per Step 5 design). The default
`minimum_mass_coverage` of 0.90 is an auditable engineering QC gate: frames
below this threshold are marked unusable but their COM proxy is retained
(not discarded). This bounds omitted published body-mass fraction to
approximately/at most 10 percentage points under rounded coefficients;
it is not a positional accuracy bound. Full supported max and female
rounding explicit: female max after omitted head is .9331.

Step 5b preserves this absolute total-body quantity and additionally reports
`supported_mass_coverage = mass_coverage /
theoretical_supported_mass_fraction`. The latter is completeness relative to
the implementation's supported-segment ceiling (`0.9306` male, `0.9331`
female), not total-body coverage and not an accuracy, confidence, or validity
score.

### Below-threshold behavior

Frames with mass coverage below the threshold retain their finite COM proxy
estimate but are marked `usable=false`. This preserves the computation for
downstream inspection while flagging reduced reliability.

### Zero-coverage frames

Frames with zero usable mass (no segments have both endpoints present) are
marked `usable=false` regardless of the threshold setting. Their COM proxy
is `None`.

### Per-segment QC flags (nonexclusive)

Each segment's COM carries independent QC flags:

- `raw_observed`: all endpoint landmarks were observed as usable in Step 3
- `x_interpolated` / `y_interpolated`: either coordinate was interpolated
- `x_smoothing_changed` / `y_smoothing_changed`: either coordinate was
  changed by smoothing
- `x_smoothing_support_interpolation` / `y_smoothing_support_interpolation`:
  smoothing window contained interpolated samples
- `other_qc_limited`: usable but none of the above flags are set
- `missing`: segment endpoints unavailable

These flags are **nonexclusive**: a single segment may have multiple flags
simultaneously. The `provenance_category()` display label is exclusive
(derived from the same flags) but the mass totals in com_proxy.csv are
nonexclusive per-flag sums.

Processed smoothed coordinates are never called raw.

## Missing data

- Segments with one or both missing endpoints contribute no mass and no
  COM to the weighted average.
- Unsupported segments (head) never contribute regardless of endpoint
  availability.
- The COM proxy is `None` only when all supported segments are missing.
- No landmarks are fabricated, interpolated, or extrapolated in Step 5.
- Interpolation and smoothing are performed in Step 3 and propagate
  through the COM calculations.

## Normalization

Stride normalization uses canonical timestamps (not frame-number fractions).
Progression is computed as:

```
progression = (timestamp - stride_start_timestamp) / duration * 100
```

where `duration = stride_end_timestamp - stride_start_timestamp`.

### Original samples

Every frame within the stride bounds (usable or not) appears as an
`original` sample with its exact frame index, timestamp, and COM proxy.

### Normalized samples

A fixed grid of N canonical progressions (0, 100/(N-1), ..., 100) is
created. For each canonical progression:
- If an exact usable frame exists at that progression, it is used directly.
- Otherwise, the two nearest bracketing usable frames are used for linear
  interpolation, **if and only if** they have consecutive frame indices
  (no-gap rule) **and** identical `usable_segments()` tuples.
- If bracketing is not possible (gap, boundary, or different segment sets),
  the normalized sample is `None` with method='none'.

No unusable frames are ever used as bracketing endpoints for normalized
samples. Non-consecutive usable frames do not bridge gaps. Adjacent frames
with different usable segment sets do not blend centroids with different
denominators; the `represented_segment_set_changed` QC flag is emitted.

### Canonical reviewed strides

Reviewed strides from Step 4b are used as-is for temporal boundaries. They
are not ground truth; they are reviewed candidate temporal segmentations
using nominal pose-frame timestamps. No force-plate-confirmed contacts,
validated stride durations, or measured toe-off / stance / swing / spatial
metrics are produced.

## Frame vs. stride

- `com_proxy.csv`: one row per frame in the recording, with per-frame COM
  and per-segment details.
- `stride_com.csv`: one row per original and normalized sample per stride,
  with `sample_kind` distinguishing original frames from normalized grid
  points.

## Not a stability metric

This COM proxy is a prerequisite for future stability metrics (e.g.,
extrapolated COM, margin of stability). By itself, it is not a stability
assessment, fall-risk estimate, or clinical result.

## Limitations

- 2D image-plane projection: no depth, no laboratory frame.
- Anthropometric coefficients are population averages, not individual.
- No camera calibration or physical scale; coordinates are not constrained
  to [0, 1].
- Normalized progression is not validated gait-cycle percentage.
- Step 3 interpolation and smoothing affect landmark positions and
  propagate through COM calculations.
- Head segment is unsupported; full model coverage cannot occur.
  Represented body mass fraction is at most .9306 (male) or .9331 (female).
- The COM proxy is a represented-segment centroid, not a whole-body
  center of mass.
- Derived endpoints (midpoints) are arithmetic means, not anatomical
  joint centers.
- Hand distal endpoint proxy uses midpoint of MediaPipe index and pinky
  points, an unvalidated distal hand endpoint surrogate, not anatomical
  hand midpoint or third metacarpal head.
- Single side-view requirement: the model assumes the walker is
  approximately perpendicular to the camera axis.
- Projection assumes static, near-sagittal, low-distortion,
  weak-perspective capture; not machine-verified.
- No validated accuracy or clinical interpretation.

## Unresolved scientific questions

- The appropriate mass coverage threshold for clinical or research use
  is not established.
- The impact of pose-estimator error on COM accuracy is not quantified.
- The relationship between this image-plane proxy and laboratory-measured
  COM is not validated.
- Stride normalization using time-based progression vs. gait-cycle
  percentage has not been compared.
