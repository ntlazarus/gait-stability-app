---
name: experiment-reproducibility
description: Guidance for making gait-analysis experiments repeatable through configuration capture, provenance, versions, hashes, and deterministic execution.
compatibility: opencode
metadata:
  domain: reproducibility
---

# Experiment reproducibility

Use this skill whenever producing experimental results.

A recorded experiment should capture, where applicable:

- run identifier;
- timestamp;
- Git commit;
- dirty-working-tree status;
- input identifiers;
- input hashes;
- configuration;
- Python version;
- dependency versions;
- pose model and version;
- model file hash;
- algorithm versions;
- random seeds;
- output paths;
- quality-control results.

Configuration used for a run should be copied or serialized into the run's
output directory.

Do not depend on notebook state for reproducibility.

A meaningful experiment should be runnable again using a documented command.
