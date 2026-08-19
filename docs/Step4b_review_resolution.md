# Step 4b: Review Resolution

## Overview

Step 4b resolves manual review and correction of automatic gait-event and
stride detections from Step 4. It validates review inputs, resolves overrides
against automatic detector output, and produces reviewed artifact files.

The pipeline preserves automatic detector provenance unless an explicit review
override is provided. Manual review is a separate process from validation;
review overrides are not force-plate or reference-validated.

## Inputs (8 files in artifact directory)

1. **gait_events.csv** - Automatic gait events with fields from `GAIT_EVENT_FIELDS`,
   all automatic candidates; statuses `accepted`/`rejected_candidate`.

2. **strides.csv** - Automatic strides with fields from `STRIDE_FIELDS`.

3. **gait_event_reviews.csv** - Event reviews.
   Header: `event_id,frame_index,timestamp_seconds,side,detection_status,review_status`
   Review statuses: `unreviewed`, `retain_rejection`, `promote_to_candidate`.

4. **strides_reviews.csv** - Stride reviews.
   Exact `STRIDE_FIELDS` copied from `strides.csv`, only
   `start_frame/end_frame/review_status` intended to differ.
   Statuses: `accept`, `correct`.
   *Ignore its timestamps/durations as final values.*

5. **gait_event_metadata.json** - Step 4 metadata including
   `config.detector` configuration.

6. **pose_frames.csv** - Canonical frame->timestamp mapping.
   Header: `frame_index,nominal_timestamp_seconds,backend_timestamp_milliseconds,status,landmark_count,detail`.
   *Validate its hash against `preprocessing_metadata.json inputs.pose_frames.csv.sha256`.*

7. **preprocessing_metadata.json** - Step 3 preprocessing provenance.
   Contains `inputs.pose_frames.csv.sha256` for hash validation.

8. **assumption-response document** - Explicit plain-text document path, hash
   and record. Documents capture assumptions (orientation, mirroring, anatomical
   labels, walking direction, view, static camera, treadmill compatibility,
   no turns, single walker, no major occlusions, nominal 30 fps timing).
   The path is recorded and the file is hashed; it is not parsed as JSON.

## Outputs (3 files in artifact directory)

1. **reviewed_gait_events.csv** - Reviewed events with columns:
   `event_id,automatic_event_id,side,event_type,automatic_frame_index,
   automatic_timestamp_seconds,automatic_disposition,automatic_quality,
   automatic_peak_value,automatic_prominence,automatic_rejection_reasons,
   manual_event_review_status,stride_review_provenance,reviewed_frame_index,
   reviewed_timestamp_seconds,reviewed_accepted,reviewed_rejected,
   reviewed_included_in_stride,reviewed_quality,resolution_disposition,
   replaces_event_id,replaced_by_event_id,source,review_notes`

   - `resolution_disposition`: `accepted_unchanged` | `corrected` |
     `promoted_from_rejected_candidate` | `rejected` | `replaced`
   - `reviewed_quality`: `"review"` for corrected/promoted accepted events;
     `"high"` / `"low"` carry the detector's `automatic_quality` forward for
     unchanged events; replaced events are assigned `"low"`.
   - Counts `rejected` and `replaced` may overlap: an event that was replaced
     also counts as rejected (reviewed_rejected is true).
   - Automatic detector values stay semantically attached to automatic frame;
     detector fields are not represented as measured at a manual-only frame.
   - CSV booleans lowercase; tuples pipe-joined; blank null.

2. **reviewed_strides.csv** - Reviewed strides with normal `STRIDE_FIELDS` plus
   `automatic_stride_id`, `review_intent`, `review_changes`,
   `provenance_notes`. Generated from `construct_strides` on reviewed events;
   never copies timestamps/durations from strides_reviews. Stride quality
   derives from reviewed endpoints and is not validated measurement quality.
   Stale timestamps/durations in strides_reviews.csv must match automatic
   strides to prevent silent data corruption.

3. **review_resolution_metadata.json** - Metadata for the review-resolution run.
   - `schema_version`: `1`
   - `algorithm_version`: `step4b-review-resolution-1`
   - Run time, git runtime, no randomness
   - Path/hash all 8 inputs (assumption path as separate record)
   - Output path/hash/self-null
   - Step4 linkage/config
   - Counts: automatic events, reviewed accepted, accepted unchanged,
     corrected, promoted, rejected, reviewed strides
   - Exact boundary changes with source/reason/IDs/frames/times
   - Timestamp source/hash
   - Regeneration method
   - Checks actually performed/passed
   - Baseline unreviewed counts/items
   - Blocking unresolved list
   - Scientific limitations and no COM/Step5
   - Metadata self hash null with explanation

## Resolution Rules

### Event review statuses

- **unreviewed**: Keep automatic disposition visibly; no override.
- **retain_rejection**: Only for automatic rejected candidates; keep rejected.
- **promote_to_candidate**: Only for automatic rejected candidates; accept them,
  retaining automatic-rejection provenance. The manual frame/timestamp comes from
  pose_frames lookup. Promoted events become accepted; the promoted candidate ID
  becomes the replacement boundary ID.

### Stride review statuses

- **accept**: No change from automatic. All other cells must match the automatic
  row.
- **correct**: Requires at least one change in `start_frame/end_frame/review_status`.
  All other cells must match the automatic row. Convert changed boundary to
  event-level correction keyed by automatic event ID; conflicting changes fail.

### Boundary correction propagation

- Corrected frame matching a same-side automatic rejected candidate can replace
  original accepted boundary only when candidate is explicitly
  `promote_to_candidate`.
- Original becomes `replaced/rejected`; candidate accepted/promoted;
  preserve `replaces/replaced_by`.
- Corrected frame with no candidate: retain original event ID, mark `corrected`,
  keep automatic frame/timestamp distinct from reviewed frame/timestamp;
  never claim detector output at manual frame.
- `retain_rejection` only for automatic rejected.
- `promote_to_candidate` only for automatic rejected and accepts it, retaining
  automatic-rejection provenance.
- Promotions must be linked as the replacement target of a stride boundary
  correction; standalone promotions are rejected.
- Existing event IDs remain canonical. Manual-only correction keeps original ID.

### Validation before publish

- Stride endpoint IDs/sides/frames/timestamps exactly match reviewed events.
- Stride endpoints must be `detection_status == accepted`,
  `included_in_stride_construction == true`, and
  `event_type == candidate_initial_contact`.
- All reviewed times exactly match canonical pose_frames.
- `duration=end-start`; positive.
- No stride uses rejected event.
- Consecutive same-side strides share boundary.
- All corrections propagate to all dependent strides.
- Input hashes unchanged.

## Scientific scope and Step 5 use

Reviewed events and strides are **manually review-aware, video-derived
candidate initial-contact boundaries and candidate temporal segmentation /
QC windows**.  Timestamps are nominal pose-frame timestamps and durations
are segmentation / QC only.

These outputs are **not** force-confirmed contacts, laboratory-equivalent
event timing, validated stride duration, toe-off, stance / swing,
spatial gait metrics, center-of-mass (COM), stability, fall-risk, or
clinical outputs.

The assumption-response document is recorded and hashed for provenance
only; its content is not machine-evaluated.  The metadata field
`scientific_unresolved` must be carried downstream to any consumer of
these artifacts.  The metadata field `blocking_unresolved` covers only
computational review-resolution ambiguity, not scientific or clinical
interpretation.

## CLI Usage

```bash
python scripts/resolve_gait_reviews.py ARTIFACT_DIRECTORY \
  ASSUMPTION_RESPONSE_DOCUMENT_PATH
```

- `artifact_directory`: Path to the artifact directory containing Step 4 inputs.
- `assumption_response_document_path`: Path to the explicit plain-text
  assumption-response document (not JSON).

Produces: `reviewed_gait_events.csv`, `reviewed_strides.csv`,
`review_resolution_metadata.json` in the artifact directory.

## Example

```bash
python scripts/resolve_gait_reviews.py outputs/walk \
  configs/assumption_responses.txt
```
