---
name: app-state-snapshot
description: Creates or refreshes the canonical quick-reference description of what the application currently does, how to run it, how to exercise key workflows, and what remains incomplete.
compatibility: opencode
metadata:
  domain: project-documentation
---

# Application state snapshot

Use this skill to create or update `docs/PROJECT_STATE.md`.

The snapshot must describe the application as it exists in the repository now,
not as planned.

## Source of truth

Determine current behavior by inspecting:

- source code;
- executable scripts and CLI entry points;
- configuration files;
- tests;
- README and architecture documentation;
- package metadata;
- example or synthetic fixtures where appropriate.

Do not rely solely on older documentation or conversation history.

When documentation conflicts with executable code, report the discrepancy.

## Verification

Whenever practical, verify important run instructions rather than copying them
from documentation.

Use safe, non-destructive commands.

Do not:

- process real participant or patient data merely to create the snapshot;
- download large models or datasets;
- make network-dependent calls unless necessary;
- modify scientific calculations;
- change application behavior.

Synthetic or documented example inputs are preferred for smoke testing.

## Required PROJECT_STATE.md structure

# Project State

## Snapshot metadata

Include:

- snapshot date/time when available;
- current Git branch;
- current HEAD commit;
- whether the working tree is clean or dirty.

## What the project currently is

Give a short description of the current executable product/research pipeline.

Explicitly distinguish implemented functionality from planned functionality.

## Current capabilities

List user-visible or researcher-visible capabilities that actually work.

For each important capability include:

- what it does;
- implementation status;
- important assumptions or restrictions.

Do not list planned features as current capabilities.

## Quick start

Provide the shortest reliable sequence needed to:

1. prepare the environment;
2. install dependencies if necessary;
3. run the primary workflow.

Commands must be copy/pasteable from the repository root.

## Key workflows

For every meaningful current workflow document:

- purpose;
- command;
- required inputs;
- important options;
- outputs created;
- where outputs are stored;
- what a successful result looks like.

Examples may include:

- inspecting a video;
- extracting pose landmarks;
- running analysis;
- generating an overlay;
- running validation;
- starting a UI or API.

Only include workflows that currently exist.

## Inputs

Document currently supported inputs, formats, configuration, and important
capture assumptions.

Do not expose participant-identifying data.

## Outputs

Document generated files, schemas, reports, visualizations, and their locations.

Explain which outputs are:

- direct observations;
- pose-model estimates;
- derived measurements;
- experimental research proxies.

## Configuration

Identify important configuration files and the settings a user is most likely
to change.

## Testing and verification

Include commands for:

- the normal verification suite;
- focused tests if useful;
- a quick smoke test if one exists.

## Known limitations

Document important:

- software limitations;
- unsupported workflows;
- scientific assumptions;
- validation gaps;
- data-quality restrictions.

Do not imply clinical validity where none has been established.

## Repository map

Give a compact map of important directories and their purposes.

Do not inventory every file.

## Next logical capabilities

Include only a short list of features that are clearly not implemented yet.

This section exists to prevent planned functionality from being confused with
current functionality.

## Documentation rules

Keep this file concise enough to scan quickly.

Prefer exact commands and concrete behavior over architectural prose.

Remove obsolete instructions when functionality changes.

Do not preserve old behavior merely for historical record. Git provides the
history.

Never report a feature as working without repository evidence.

If an important workflow cannot currently be run or verified, say so explicitly.
