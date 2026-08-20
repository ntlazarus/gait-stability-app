# Clean-Capture Qualification Method (Step 5c)

## Purpose and scope

Step 5c applies a frozen engineering policy to one current and one prior Step 5b
COM qualification and an external clean-capture review. It does not rerun pose
estimation, preprocessing, gait-event detection or review, COM estimation, or
Step 5b QC. It publishes one auditable qualification record for exploratory
engineering feasibility.

The evidence layers must remain distinct:

- the observed video is the camera recording inspected by the reviewer;
- pose estimates are model-produced image landmarks, not measured anatomy;
- processed trajectories are pose estimates after inherited Step 3 missing-data,
  interpolation, and smoothing rules;
- the Step 5a represented-segment COM proxy is an unvalidated, mass-weighted 2D
  centroid of usable supported segments, not physical whole-body COM;
- Step 5b is quantitative completeness and provenance QC for that proxy; and
- Step 5c is a frozen engineering-feasibility decision over Step 5b evidence and
  external review, not a new measurement or scientific validation stage.

An engineering `GO` permits exploratory pipeline work only. Scientific
readiness is always `NO-GO` with status `not_established`; Step 5c cannot
establish COM, gait-event, stability-measurement, diagnostic, or clinical
validity.

## CLI and Python API

Run Step 5b for the current capture and a prior capture before Step 5c:

```bash
.venv/bin/python scripts/qualify_capture.py ARTIFACT_DIRECTORY \
  CAPTURE_REVIEW_JSON --prior-qualification PRIOR_QUALIFICATION_JSON
```

`ARTIFACT_DIRECTORY` must contain the current canonical
`com_qualification.json`. The CLI prints the single published output path on
success and exits nonzero on validation or publication failure.

The equivalent Python API is:

```python
from gait_stability.capture_qualification import qualify_clean_capture

artifacts = qualify_clean_capture(
    artifact_directory,
    capture_review_path,
    prior_qualification_path,
)
output_path = artifacts.qualification_json_path
```

The three arguments may be strings or `pathlib.Path` objects. The API returns a
`CaptureQualificationArtifacts` value containing the artifact directory and
qualification JSON path.

## Inputs and output

Step 5c consumes:

- **current**: the canonical Step 5b qualification in the supplied artifact
  directory and all current source, annotated-video, stride-QC, preprocessing,
  and review-resolution artifacts referenced by it;
- **prior**: a distinct prior Step 5b qualification and its corresponding
  referenced lineage, used as a comparator rather than pooled evidence; and
- **review**: the strict external-review JSON described below. If the supplied
  review path does not identify a file, Step 5c still publishes a pending
  `NO-GO` record rather than treating review as implicitly satisfactory.

Input paths and the output path must not alias. The sole published output is
`capture_qualification.json` in the current artifact directory. Step 5c does
not alter any input artifact and does not emit a revised video, trajectory,
event, COM, or Step 5b result.

## Capture protocol

The intended protocol is a fixed, static-camera, approximately sagittal side
view with the complete person framed from head through both feet throughout the
usable walk. Both arms and feet should remain visible; major self-occlusion,
out-of-plane motion, turns, camera pan/tilt/zoom/reframing, camera motion, severe
lens or perspective distortion, and avoidable landmark loss should be limited.
Lighting and subject/background separation must support tracking. The recording
must contain several complete candidate gait cycles, and anatomical left/right
and image-space walking direction must be establishable.

This protocol is an engineering capture requirement, not a claim that any
recording satisfies it. The review's `capture_protocol` is a required nonempty
declaration of the protocol actually reviewed. Step 5c compares inherited
capture-protocol declarations between current and prior records but does not
machine-interpret that free text or infer protocol compliance from it.

## External review contract

Start from [`capture_review_template.json`](capture_review_template.json). It is
deliberately non-qualifying: its zero hashes are placeholders, its reviewer is
an `automated_assistant` with uncertain independence, whole-video flags are
false, the declared-direction consistency check is false, and every item is
`uncertain`. An independent human
reviewer must replace every placeholder with review-specific evidence. Do not
change field names, add fields, or delete fields.

The top-level object has exactly these keys:

- `schema_version`, exactly integer `1`;
- nonempty `review_id` and UTC ISO-8601 `reviewed_at_utc`;
- `reviewer`, containing exactly `reviewer_type`, `identifier`, `role`, and
  `independence`;
- boolean `whole_source_video_inspected` and
  `whole_annotated_com_video_inspected`;
- nonempty `capture_protocol`;
- `artifact_hashes`, containing exactly `source_video`, `annotated_com.mp4`, and
  `com_qualification.json`;
- `walking_direction`, exactly `image_left` or `image_right`;
- boolean `declared_direction_matches_inherited_step4`;
- nonempty `orientation_notes`; and
- exact `capture_items` and `annotated_com_items` objects.

`reviewer_type` is exactly `human` or `automated_assistant`.
`independence` is exactly `independent`, `not_independent`, or `uncertain`.
Each checklist item contains exactly `status` and `note`. Status is exactly one
of:

- `confirmed`: the reviewer found the stated condition satisfied; `note` may be
  empty;
- `uncertain`: available evidence does not establish the condition; a nonblank
  note is required and this prevents `GO` but is not itself a hard blocker;
- `not_confirmed`: the condition was reviewed and not established; a nonblank
  note is required and this is a hard `NO-GO` blocker; or
- `not_applicable`: the reviewer declares the required condition inapplicable;
  a nonblank note is required and this is a hard `NO-GO` blocker.

### Required capture items

The exact capture checklist keys are:

- `full_body_in_frame`
- `head_to_feet_framing_adequate`
- `both_arms_visible`
- `both_feet_visible`
- `approximately_sagittal_view`
- `fixed_camera_no_pan_tilt_zoom_or_reframing`
- `camera_motion_absent_or_negligible`
- `minimal_subject_out_of_plane_motion_and_no_turns`
- `no_obvious_severe_lens_or_perspective_distortion`
- `major_self_occlusion_limited`
- `lighting_adequate`
- `subject_background_separation_adequate`
- `several_complete_candidate_gait_cycles_present`
- `anatomical_left_right_established`
- `walking_direction_established`
- `bilateral_com_landmarks_trackable`
- `bilateral_event_landmarks_trackable`

### Required annotated-COM items

The exact annotated-video checklist keys are:

- `trajectory_continuity_adequate`
- `no_visibly_nonphysiological_tracking_jumps`
- `no_identity_loss_or_whole_pose_reset`
- `no_visible_anatomical_left_right_label_swap`
- `no_unexplained_sustained_supported_segment_dropout`
- `no_obvious_centroid_relocation_with_limb_disappearance`
- `no_visible_tracking_discontinuity_within_0_2_seconds_of_reviewed_event_boundaries`
- `proxy_grossly_follows_tracked_body`

The `0_2_seconds` item asks only whether a visible tracking discontinuity occurs
within plus or minus 0.2 nominal seconds of a reviewed event boundary. It does
not ask the reviewer to confirm contact timing, correct the boundary, expand the
window, or establish event accuracy.

## Human and machine boundaries

An independent human must inspect the entire observed source video and entire
annotated COM video and set both whole-video flags true. The human establishes
capture suitability, visible continuity, anatomical left/right, and walking
direction, and records uncertainty or failures. A reviewer is complete for the
policy only when `reviewer_type` is `human`, independence is `independent`, and
both whole-video flags are true.

The machine strictly parses schema and values, verifies hashes and inherited
lineage, extracts Step 5b quantities, checks current/prior comparability, and
applies the frozen policy. `declared_direction_matches_inherited_step4` is a
machine consistency check on the review declaration, not a measure of human
agreement or inter-rater agreement. The machine does not verify that a named
person performed the review, independently inspect the pixels, infer camera
suitability, validate anatomy, or turn reviewer declarations into scientific
ground truth.

## Hash and provenance linkage

Every review hash must be a 64-character lowercase hexadecimal SHA-256 digest.
The review hashes must exactly match the current observed source video, current
annotated COM video, and current Step 5b qualification bytes. Placeholder,
stale, uppercase, malformed, or mismatched hashes are rejected.

For both current and prior records, Step 5c verifies referenced file existence
and stored hashes; exact Step 5a and Step 5b schema/algorithm versions; the
primary threshold and frozen grid; sex and anthropometry consistency; Step 3
schema/algorithm, source hash, pose-model hash, preprocessing configuration,
and capture assumptions; and Step 4b schema/algorithm, preprocessing linkage,
and inherited image-space walking direction. The review direction must match
the inherited Step 4 direction. All consumed hashes are snapshotted and checked
again immediately before publication, including source-video hashes; mutation
aborts publication. Git provenance is intentionally best-effort so absence of a
Git worktree does not prevent qualification.

The Step 5a anthropometric coefficient sex is recorded as user-supplied and
never inferred by this pipeline. The output records the value inherited from
the specific Step 5a input; this generic method does not prescribe or claim a
particular selection. Sex-specific de Leva coefficients remain population
averages, not measurements of individual anthropometry.

## Evidence definitions and units

All fractions are dimensionless. Frame fractions use all frames in the entire
recording, not only finite, eligible, selected, or stride-contained frames:

```text
finite_com_fraction = finite COM proxy frames / all recording frames

primary_eligible_fraction = finite, nonzero COM proxy frames with
                            mass_coverage >= 0.90
                            / all recording frames
```

`mass_coverage` is the unrenormalized sum of represented supported-segment mass
fractions and is an absolute fraction of modeled total body mass. It is not a
confidence, probability, positional error, or validity score.

The longest usable interval is the longest contiguous run of primary-eligible
frames. Its frame count is exact for the stored sequence; its duration is in
nominal seconds derived from recorded nominal timestamps, not an independently
validated clock. Coordinates remain normalized 2D image-plane coordinates
(`x` right, `y` down) with no physical scale, depth, gravity alignment, or
laboratory coordinate system.

Reviewed stride counts include all reviewed candidate strides. The
policy-complete fraction denominator is all reviewed candidate strides, and the
normalized usable fraction denominator is all canonical normalized samples
across all reviewed candidate strides. No failed or incomplete recording,
stride, frame, or normalized sample is silently removed from these denominators.

Reviewed candidate events and stride boundaries are inherited Step 4 outputs,
not force-plate-confirmed contacts or validated gait events. A candidate stride
is policy-complete only under the Step 5b primary-threshold definition: every
original frame is eligible, the represented segment set is invariant, every
canonical normalized target is available under exact or adjacent
same-segment-set rules, and both endpoints are eligible. This category is QC
completeness, not biomechanical validity.

## Frozen threshold grid

Step 5c accepts only the absolute mass-coverage grid
`0.80, 0.82, 0.84, 0.86, 0.88, 0.90`, in that order, with exactly one Step 5b
entry per threshold. The primary threshold is exactly `0.90`. Grid entries are
descriptive sensitivity results; Step 5c does not search, tune, rank, select, or
optimize a threshold for either recording. The policy is applied at the frozen
primary threshold regardless of which grid value would make availability look
best.

## Prior comparison and comparability

The raw current value, raw prior value, and current-minus-prior empirical maximum
absolute mass-coverage difference are reported. A comparable prior requires
matching sex, pose-model SHA-256,
preprocessing configuration, and inherited capture-protocol declaration.
Step 5a and Step 5b schema/algorithm versions, the primary gate, and sensitivity
grid are fixed and validated for both inputs. The recordings need not be the
same file; each retains separate source and lineage hashes.

Failure of comparability is a `CONDITIONAL` condition, not evidence of
improvement. For a noncomparable pair, the raw values and arithmetic delta are
retained but `coverage_delta_status` is `not_interpretable_for_go`; they do not
support a causal conclusion about capture quality, framing, visibility, or any
other acquisition change. `GO` also requires current empirical maximum mass
coverage to exceed the prior value by at least `0.01` absolute. This comparison
is a frozen engineering check on one current/prior pair, not an optimized
baseline, statistical effect, repeatability estimate, or population inference.

## Frozen engineering decision policy

All of these are required for engineering `GO`:

| Criterion | GO requirement |
|---|---:|
| Finite COM fraction | `>= 0.95` |
| Primary-eligible fraction at absolute coverage 0.90 | `>= 0.90` |
| Longest primary-eligible interval | `>= 3.0` nominal seconds |
| Persistent missingness among supported segments | none |
| Reviewed candidate strides | `>= 3` |
| Policy-complete strides | `>= 3` |
| Policy-complete stride fraction | `>= 0.75` |
| Bilateral policy-complete strides | `>= 1` on each side |
| Normalized usable fraction | `>= 0.90` |
| Empirical maximum absolute mass coverage | `>= 0.90` |
| Comparable prior | true |
| Empirical maximum improvement over prior | `>= 0.01` absolute |
| Review | independent human inspected both whole videos |
| Checklist | every required item `confirmed` |
| Declared direction consistency | review declaration matches inherited Step 4 direction |
| Upstream provenance valid | all hashes and inherited lineage match |

The following hard blockers produce `NO-GO`:

- any persistent missingness among supported segments;
- primary-eligible fraction strictly below `0.50`;
- fewer than `2` policy-complete strides;
- normalized usable fraction strictly below `0.50`;
- missing, nonhuman, incomplete, or non-independent whole-video review;
- any checklist item `not_confirmed` or `not_applicable`;
- declared review direction mismatch with inherited Step 4; or
- provenance mismatch.

Values exactly at the hard minima `0.50`, `2`, and `0.50` are not hard
blockers, but remain `CONDITIONAL` because they fail GO thresholds. In the
absence of a hard blocker, any unmet GO criterion produces `CONDITIONAL`.
An `uncertain` checklist item makes the `all_review_items_confirmed`
`go_required` criterion fail, prevents GO, and is recorded as a warning. A
`not_confirmed` or required `not_applicable` item remains an explicit hard
blocker. Only when there are no blockers and every GO criterion passes is the
engineering decision `GO`.

The pure evaluator's `provenance_valid` argument is an upstream-validation flag:
the caller must set it from completed hash and lineage validation. The pure
function does not inspect files. `False` is a hard `NO-GO` blocker.

## Output structure

The schema-version-1 output records:

- algorithm, scope, run, timestamp, runtime, dependency, Git, and randomness
  provenance;
- paths and SHA-256 hashes for current, prior, review, and inherited artifacts;
- the parsed review or `null` when review is missing;
- inherited direction and the declared-direction consistency result;
- the frozen grid, thresholds, hard blockers, and definitions;
- complete current and prior evidence, distributions, missing-segment,
  asymmetry, stride, normalized, and threshold-grid results;
- comparability checks, raw current/prior maximum coverage, arithmetic delta,
  and delta interpretation status;
- each criterion result, blockers, warnings, and engineering decision states;
- scientific `NO-GO`/`not_established`, coordinate/camera/filter/event
  provenance, limitations, and validation status; and
- one output entry whose self-hash is `null`, because embedding a file's own
  digest would change that digest.

`evaluation_state` is `evaluated` once the quantitative policy has been applied.
`external_human_review_state` is separately `confirmed`, `pending`, or
`incomplete`: `confirmed` requires an independent human, both whole-video flags,
and all checklist items confirmed; `pending` means no human review is available;
all other present human-review states are `incomplete`. The legacy `state`
remains for compatibility and is either `evaluated` or
`pending_external_human_review`; it must not be used as a proxy for whether
quantitative evaluation ran. Quantitative hard blockers can establish `NO-GO`
while external human review is still pending.

When review is absent, its input status is `missing`, its hash and parsed review
are null, legacy engineering `state` is `pending_external_human_review`,
`evaluation_state` is `evaluated`, `external_human_review_state` is `pending`,
and the decision is `NO-GO`. A present but malformed, hash-mismatched, or
direction-mismatched review is rejected rather than published as missing.

## Limitations

- One current capture and one prior comparator cannot establish test-retest
  repeatability, generalization, robustness across cameras or environments,
  sensitivity to capture variation, or population performance.
- Human visual confirmation is subjective and declarative; Step 5c does not
  authenticate reviewer identity or measure inter-rater agreement.
- Good visibility, continuity, and mass coverage do not establish landmark or
  COM accuracy and can coexist with systematic pose or projection bias.
- The proxy omits unsupported anatomy, uses user-supplied sex selection with
  population-average anthropometric coefficients and pose landmarks as
  joint-centre surrogates, and remains a represented-segment centroid.
- Inherited interpolation and smoothing can affect trajectories; Step 5c does
  not recompute or validate those operations.
- Candidate events, stride boundaries, and normalized progression are not
  reference-system validated.
- No laboratory COM, motion-capture, force-plate, stability, fall-risk,
  diagnostic, or clinical validation is performed. Scientific readiness remains
  `NO-GO`/`not_established` regardless of the engineering decision.
