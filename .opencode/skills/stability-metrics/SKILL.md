---
name: stability-metrics
description: Guidance for implementing and interpreting candidate gait-stability metrics including variability, sway, COM dynamics, XCoM, and margin of stability.
compatibility: opencode
metadata:
  domain: gait-stability
---

# Stability metrics

Use this skill when implementing or reviewing metrics intended to describe gait
stability.

## Do not collapse metrics prematurely

Initially report interpretable component metrics rather than a proprietary
single stability score.

Candidate categories may include:

- temporal variability;
- spatial variability;
- step-width behavior;
- trunk motion;
- mediolateral sway;
- center-of-mass motion;
- gait asymmetry;
- extrapolated center-of-mass measures;
- margin-of-stability measures.

## For every metric define

- mathematical formula;
- units;
- direction/axis;
- analysis window;
- required gait events;
- required landmarks;
- scale dependency;
- normalization;
- filtering;
- missing-data behavior;
- interpretation;
- validation status.

## Dynamic stability

Do not implement margin of stability merely because a formula is available.

First establish defensible estimates of:

- COM position;
- COM velocity;
- base of support;
- physical scale;
- ground plane;
- pendulum length or equivalent model parameter;
- gait events.

If required inputs cannot be estimated adequately from the video configuration,
label the resulting measure as an experimental proxy or do not calculate it.
