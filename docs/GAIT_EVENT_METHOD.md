# Step 4 Candidate Gait-Event Method

## Scope and terminology

Step 4 reports **video-derived candidate initial contacts** and same-side stride
boundaries for research review. It does not report force-confirmed foot contact,
toe-off, center of mass (COM), stability, fall risk, diagnosis, or any clinical
conclusion. A `high` quality label means that deterministic algorithmic support
conditions were met. It is not a probability, accuracy estimate, or validation
claim. Sequence regularity is event-train quality control (QC), not pose-signal
confidence.

## Definition and formula

For anatomical side `s`, the primary signal is the direction-normalized,
pelvis-proxy-relative heel trajectory:

```text
p_x(t) = (left_hip_x(t) + right_hip_x(t)) / 2
s_s(t) = d * (heel_s_x(t) - p_x(t))
d = +1 for image_right; d = -1 for image_left
```

The bilateral hip midpoint is a pelvis proxy, not whole-body COM. A candidate
initial contact is formed at a strict local maximum of `s_s(t)`, including one
deterministic earlier-midpoint candidate for a flat maximum plateau. The candidate
must pass configured prominence, forward-position, and pre/post reversal tests.
Prominence is the peak minus the larger of the preceding and following minima in
the configured support windows. Reversal velocities use differences over nominal
timestamps. Ipsilateral ankle `x` and a raw heel-plus-bilateral-hip signal are
supporting sensitivity cues; they do not define the primary peak. These cues are
correlated with the processed primary trajectory and are not independent
corroboration.

This adapts the pelvis-relative foot-position concept described by Zeni,
Richards, and Higginson, "Two simple methods for determining gait events during
treadmill and overground walking using kinematic data" (2008), PMCID
[PMC2384115](https://pmc.ncbi.nlm.nih.gov/articles/PMC2384115/). That paper's
marker/force validation does **not** validate MediaPipe, this implementation, or
the current recording.

## Coordinates, units, and calibration

- Coordinates are monocular pose-model estimates in the stored image plane.
- `x` increases from image left to image right; `y` increases from image top to
  image bottom. Left/right labels are model-assigned anatomical sides.
- `x` and `y` use dimensionless normalized image width and height. Signal and
  prominence units are normalized image width; reversal velocity units are
  normalized image width per nominal second.
- The default minimum prominence of `0.02` is 2% of image width. It depends on
  capture geometry and framing and is not transferable to another setup without
  evaluation.
- There is no camera calibration, physical scale, lens correction, laboratory
  frame, or metric conversion. Results must not be interpreted in meters.
- Walking direction is manually declared as `image_right` or `image_left`; it is
  not inferred. Camera mirroring, stored orientation, and anatomical left/right
  correctness must be manually established.

## Capture and landmark requirements

The intended context is a near-sagittal view with a static camera and treadmill
or otherwise compatible walking motion. These requirements are not automatically
verified. Perspective, camera movement, oblique views, turns, overground passage
through the image, or unrecognized mirroring can invalidate the signal meaning.

Required primary landmarks are the ipsilateral heel and both hips. Both heels
and both hips must have complete processed `x` values throughout an automatic
bout candidate. Ipsilateral ankle `x` is optional support. Processed `x` and `y`
for all canonical landmarks are used only to draw the review skeleton. The heel
landmark is a pose-model label, not a measured ground-contact point.

Initial contact candidates are the only required and produced gait events.
Toe-off is deliberately omitted because it is not sufficiently established by
this MVP method. Strides are consecutive accepted same-side candidate initial
contacts. Rejected candidates never form strides, and no missing event or partial
stride is fabricated. No stance, swing, or double-support metrics are produced.

## Bout selection

Manual start/end frames are inclusive and define a user-selected analysis
interval; they do not prove walking. Without a manual range, maximal contiguous
runs with complete bilateral primary processed signals qualify only when they
meet the configured minimum duration and minimum accepted preliminary candidate
count on each side. Every qualifying run is recorded. Selection is deterministic:
highest total accepted count, then longest duration, then earliest start.

These boundaries indicate only a complete primary-signal interval with minimum
candidate-count evidence. The implementation does not test periodicity, physical
walking onset or offset, or steady-state walking. Manual confirmation is required
before downstream interpretation. If no run qualifies, the complete stored
nominal range is retained as `full_recording_fallback`, explicitly without inferred
walking boundaries. No terminal sample is synthesized.

## Filtering, interpolation, and missing data

Step 4 adds no smoothing or interpolation. It reuses Step 3 processed trajectories,
which may contain bounded interpolation and centered moving-average smoothing as
recorded in `preprocessing_metadata.json`. Those operations can attenuate extrema
or shift candidate timing and may interpolate across an unknown event.

Primary observed usability and interpolation provenance use only the exact three
primary `x` inputs: ipsilateral heel, left hip, and right hip. Their flags are
combined with all/OR rules across the full configured support needed by the local
peak, reversal, and prominence calculations. Ankle, knee, shoulder, and unrelated
landmark interpolation cannot change primary-signal provenance. A raw supporting
cue exists only when all three primary raw `x` values are observed usable.

The selected ankle cue is direction-normalized ipsilateral ankle `x` relative to
the bilateral hip midpoint. Its existing `ankle_support_*` fields record the
combined provenance of the ankle and both hips over the exact local-peak support,
including samples adjacent to the peak or plateau. The cue frame is retained when
that support is unclean, but the event is downgraded to `review`. A clean ankle cue
requires the ankle and both hips at every support sample to be raw-observed usable,
with no interpolation or interpolation-affected smoothing in any of those inputs.

Missing processed primary `x` splits the signal into contiguous segments. Peaks
cannot bridge a gap. Interpolation-affected primary support downgrades an accepted
event to `review`; missing raw or ankle agreement also yields `review`, not an
automatic rejection. All formed local candidates, including rejected candidates,
remain in `gait_events.csv` for audit.

## Timing and event QC

Timestamps are exact stored Step 3 nominal timestamps, originally based on
nominal frame timing. They are not verified presentation timestamps. Bout bounds
are inclusive. Configured same-side and opposite-side minimum intervals resolve
hard conflicts by ranking clean support, clean cue count, greater forward
peak value, prominence, and earlier frame, in that exact order, then rejecting the
lower-ranked candidate. An unclean ankle-plus-bilateral-hip cue is not a clean
quality cue. This ranking may bias which nearby peak is selected. The default
0.15-second
opposite-side gate is an ordinary-walking engineering assumption and can reject
near-simultaneous contacts. Long same-side intervals and nonalternating adjacent
events are retained as QC notes rather than forcing alternation. Thresholds are
engineering/research assumptions and are recorded in `gait_event_metadata.json`.

## Artifacts and renderer semantics

`walking_bout.json` records selection and all qualifying alternatives.
`gait_events.csv` records accepted and rejected formed candidates.
`strides.csv` records complete same-side accepted-event pairs. The diagnostic PNG
shows processed direction-normalized signals, optional raw cues, interpolation
effects, candidate markers, and exact bounds.

The annotated MP4 uses inherited nominal dimensions and FPS, has no audio, and
preserves every nominal Step 3 frame slot from zero through the final frame. A
sequential source decode failure produces a labeled black placeholder rather than
dropping a slot. The overlay is a processed pose-model skeleton and explicitly
states `POSE-MODEL ESTIMATE`. Event flashes show anatomical side, quality, status,
and the model heel; stride IDs indicate accepted stride intervals. Renderer or
writer initialization failures abort publication clearly.

Event IDs are deterministic only for the same inputs, configuration, and algorithm.
They are not stable across parameter or algorithm changes.

## Limitations and validation status

Monocular image-plane trajectories are sensitive to pose errors, occlusion,
left/right swaps, perspective, camera movement, smoothing, direction mistakes,
mirroring, and capture geometry. Local heel maxima may not coincide with physical
contact. Automatic candidates identify complete-signal and candidate-count evidence,
not periodicity, walking pathology, or stability. A `high` stride means only that it
is bounded by two accepted same-side candidates and passes current QC; it does not
prove that no cycle was missed. Stride duration is retained for segmentation QC
only, not population-normalized interpretation.

The pure formulas and workflow can be software-tested with synthetic data, but
software correctness is not biomechanical validation. There is currently no
validation against manually labeled video, markers, force plates, another
reference system, repeated sessions, participant populations, or clinical
outcomes. Visual plausibility review is not reference-system validation.
