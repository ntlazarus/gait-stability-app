---
name: pose-quality-control
description: Guidance for evaluating pose landmark visibility, confidence, missingness, occlusion, smoothing, interpolation, and tracking quality.
compatibility: opencode
metadata:
  domain: pose-estimation
---

# Pose quality control

Use this skill whenever pose landmarks are generated, cleaned, interpolated,
filtered, or evaluated.

## Preserve raw information

Retain the raw pose-estimator outputs separately from processed trajectories.

Preserve available:

- landmark confidence;
- landmark visibility;
- tracking confidence;
- frame timestamp;
- frame index;
- coordinate representation.

## Missing and unreliable landmarks

Never silently replace a low-confidence landmark with an interpolated value.

Represent missingness explicitly.

If interpolation is introduced, specify:

- eligible landmarks;
- confidence threshold;
- maximum gap length;
- interpolation method;
- whether interpolation crosses gait events;
- how interpolated samples are marked.

## Filtering

Filtering must document:

- filter family;
- cutoff or smoothing parameter;
- sampling-rate assumptions;
- causal versus zero-phase behavior;
- edge handling.

Do not choose smoothing parameters solely because the trajectory looks better.

## Quality outputs

Produce quality measures independently of gait metrics where possible.

Examples include:

- valid-frame fraction;
- valid-landmark fraction;
- landmark-specific missingness;
- longest missing-data gap;
- left/right visibility imbalance;
- tracking discontinuities.

Poor-quality input should be reportable rather than automatically converted
into apparently precise gait metrics.
