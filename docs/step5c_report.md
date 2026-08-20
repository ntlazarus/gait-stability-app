Step 5c completed
1. New code required
Yes, but only qualification plumbing—not a new biomechanics algorithm.
Implemented:
- src/gait_stability/capture_qualification.py
- scripts/qualify_capture.py
- docs/CLEAN_CAPTURE_QUALIFICATION_METHOD.md
- docs/capture_review_template.json
- Step 5c API exports
- 54 synthetic Step 5c tests
- Updated docs/PROJECT_STATE.md
Step 5a/5b algorithms, coefficients, segment mappings, gait-event boundaries, and the 0.90 gate were not changed.
2. Capture review
A candidate with strong gross framing was processed through Steps 1–5b using the explicitly user-selected male coefficient set.
The user selected a review template rather than immediate human confirmation. Therefore:
- Independent whole-video human review: pending
- Capture assumptions: uncertain, not confirmed
- Annotated-video review: uncertain, not confirmed
- Machine verification of camera geometry: not performed
An AI-assisted sampled-frame precheck showed the full body and below-threshold proxy marker in sampled frames, without an obvious whole-pose reset. It cannot establish continuity, left/right consistency, event-boundary behavior, or capture suitability across the complete video.
3. Prior-versus-current comparison
Metric	Prior
Finite COM frames	300/301
Theoretical supported mass	0.9306
Maximum absolute coverage	0.8812
Frames passing 0.90	0/301
Policy-complete windows at 0.90	0/14
Persistent missing segments	Left upper arm, forearm, hand
Strict controlled comparability is false because preprocessing configurations differ. The raw coverage delta is 0.0, but it is recorded as not_interpretable_for_go; no causal framing or capture-improvement conclusion is claimed.
4. Primary 0.90 result
Current candidate:
- Absolute coverage distribution: min/mean/median/max all 0.8812
- Supported-mass coverage: approximately 0.946916
- Primary-eligible frames: 0/301
- Longest passing interval: 0 seconds
- Normalized availability: 0/1,313
- Policy-complete candidate windows: 0/13
5. Threshold sensitivity
At every threshold from 0.80 through 0.88:
- Eligible frames: 301/301
- Longest interval: 10.0 seconds
- Policy-complete windows: 13/13
- Normalized availability: 1,313/1,313
At 0.90, all four quantities fall to zero. This reproduces the threshold cliff and does not justify relaxing the threshold.
6. Missingness and asymmetry
Persistent supported-segment loss:
- Left upper arm
- Left forearm
- Left hand
Transient supported-segment loss: none
Left-minus-right usability difference was -1.0 for upper arm, forearm, and hand, with corresponding elbow/wrist landmark asymmetry. Lower-limb segment asymmetries were zero.
7. Reviewed-stride completeness
Step 4b produced 13 reviewed candidate temporal windows. They remain video-derived candidate initial-contact windows, not validated gait cycles.
At 0.90:
- Policy-complete: 0/13
- Bilateral complete windows: none
- Normalized available samples: 0/1,313
No event boundaries were adjusted to improve COM qualification.
8. Annotated-COM qualitative QC
Sampled frames show:
- A finite orange, below-threshold proxy marker near the tracked body
- No gross whole-pose reset apparent in those samples
- Persistent unilateral arm-chain omission in the pose-derived evidence
Whole-video continuity, jumps, anatomical swaps, segment disappearance, centroid shifts, and ±0.2-second event-boundary behavior remain externally unreviewed. Gross visual following would be face-validity screening only—not accuracy evidence.
9. Verification
Fresh results:
- Ruff format: 78 files formatted
- Ruff lint: passed
- mypy: 22 source files passed
- Full pytest: 566 passed in 215.11s
- Focused Step 5c: 54 passed in 34.23s
- pip check: passed
- git diff --check: passed
- Biomechanics and code reviews completed; confirmed findings fixed
- Generated videos, reviews, and qualification outputs remain ignored
10. Remaining limitations
- No independent whole-video human capture review
- Current/prior records are not strictly controlled-comparable
- Persistent unilateral arm pose coverage despite complete gross framing in sampled views
- No physical scale, depth, calibration, gravity/ground alignment, or 3D reconstruction
- Candidate gait events remain unvalidated
- One capture cannot establish reliability or generalizability
- No reference-system COM validation
11. Step 6 engineering readiness
NO-GO
Independent quantitative hard blockers are:
- Persistent supported-segment absence
- Primary eligibility below the 0.50 hard minimum
- Fewer than two policy-complete windows
- Normalized availability below the 0.50 hard minimum
External human review remains pending, but completing it cannot remove these quantitative failures for this artifact.
12. Scientific measurement readiness
NO-GO / not established
No evidence establishes laboratory-equivalent COM, gait-event accuracy, stability-measurement validity, fall-risk performance, or clinical usefulness.
All source changes remain uncommitted.