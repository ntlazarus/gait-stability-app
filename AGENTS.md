# Gait Stability from Video

## Project purpose

Build a research-grade Python pipeline that estimates gait characteristics and
candidate stability metrics from walking video. The system is not a diagnostic
medical device and must not present unvalidated outputs as clinical conclusions.

## Repository rules

- Work only inside this repository.
- Use Python with reusable code under `src/gait_stability/`.
- Keep executable entry points thin and place them under `scripts/`.
- Keep exploratory work in `notebooks/`; do not make notebooks the source of truth.
- Never commit videos, subject data, extracted landmarks, model weights,
  generated overlays, credentials, or environment-specific files.
- Do not silently change metric definitions, coordinate systems, units, data
  schemas, or capture assumptions.
- Avoid speculative abstractions and premature application infrastructure.

## Agent workflow

Use this sequence:

1. Understand the requested outcome.
2. Inspect relevant code and documentation.
3. Produce a bounded implementation plan.
4. Delegate research, implementation, and testing to appropriate subagents.
5. Run focused tests.
6. Run the complete verification suite.
7. Request biomechanics review when measurement semantics change.
8. Request code review.
9. Fix confirmed findings.
10. Re-run verification before claiming completion.

The orchestrator should not directly implement code unless explicitly requested.

## Scientific requirements

Every reported metric must document:

- Definition and formula
- Units
- Coordinate system
- Required camera view
- Required calibration or scale information
- Required gait events
- Landmark dependencies
- Filtering or smoothing
- Missing-data behavior
- Known limitations
- Validation status

Do not treat pose-estimator world coordinates as ground-truth laboratory
coordinates. Preserve confidence and quality information throughout the
pipeline.

Do not silently interpolate low-confidence landmarks. Any interpolation must be
configurable, bounded, recorded, and reflected in quality outputs.

## Software requirements

- Use explicit, readable Python.
- Use typed data models at module boundaries.
- Separate pose backends from biomechanics calculations.
- Keep raw observations separate from derived metrics.
- Make every pipeline runnable from one documented command.
- Add unit tests for formulas and integration tests for pipeline boundaries.
- Use deterministic tests and synthetic pose sequences where possible.
- Do not add dependencies without documenting their purpose.

## Completion requirements

Do not claim completion without fresh evidence from:

- Formatting and linting
- Static type checking
- Unit tests
- Relevant integration tests
- Review of the final Git diff

Report remaining assumptions, unvalidated behavior, and known limitations.
## Current project state documentation

`docs/PROJECT_STATE.md` is the canonical quick-reference description of the
currently implemented application.

After a substantial feature or workflow change, refresh it with:

`/snapshot-app`

The snapshot must describe implemented behavior, executable commands, inputs,
outputs, testing instructions, and known limitations. Planned functionality
must not be represented as current functionality.
