---
name: validation-and-claims
description: Guidance for validating video-derived gait measurements and keeping scientific or clinical claims proportional to available evidence.
compatibility: opencode
metadata:
  domain: validation
---

# Validation and claims

Use this skill when evaluating measurement accuracy or describing what the
system can measure.

## Validation hierarchy

Distinguish:

1. software correctness;
2. internal repeatability;
3. agreement with manually labeled video;
4. agreement with a validated reference system;
5. reliability across sessions;
6. validity across participants and capture conditions;
7. clinical validity.

Passing unit tests establishes software correctness, not measurement validity.

## Compare measurements appropriately

Depending on the metric, consider:

- absolute error;
- mean absolute error;
- RMSE;
- bias;
- limits of agreement;
- correlation;
- ICC;
- repeatability;
- sensitivity to capture conditions.

Do not use correlation alone as evidence that two measurement methods agree.

## Claims

Use precise wording such as:

- estimated;
- video-derived;
- candidate metric;
- research proxy;
- internally validated;
- externally validated.

Do not use language implying diagnosis, clinical accuracy, or medical-device
performance without the evidence required to support those claims.
