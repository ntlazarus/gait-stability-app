---
name: gait-biomechanics
description: Guidance for gait-cycle definitions, gait events, coordinate systems, anthropometry, center-of-mass estimation, and biomechanical interpretation.
compatibility: opencode
metadata:
  domain: biomechanics
---

# Gait biomechanics

Use this skill for calculations that interpret pose trajectories
biomechanically.

## Establish conventions first

Before implementing a metric, specify:

- anatomical landmarks;
- coordinate system;
- axis directions;
- origin;
- units;
- scale source;
- camera view;
- gait direction;
- gait-event definitions.

## Gait events

Do not assume heel strike or toe-off is directly observable from a landmark
without defining the detection algorithm.

Keep gait-event detection separate from downstream metric calculations.

Preserve event confidence when available.

## Center of mass

Distinguish among:

- landmark centroid;
- pelvis or trunk proxy;
- segment-weighted anthropometric COM estimate;
- laboratory-measured whole-body COM.

Never label a simple landmark average as whole-body center of mass without
qualification.

Document anthropometric coefficients and their source.

## Dimensional consistency

Check every formula for compatible dimensions.

Keep physical-unit calculations separate from normalized-image-coordinate
calculations.

Do not mix normalized coordinates, pixels, meters, or arbitrary model-space
units without an explicit transformation.
