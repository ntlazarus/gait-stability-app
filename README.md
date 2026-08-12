# Gait Stability from Video: OpenCode Overview

## Project Purpose

This project builds a research-grade Python pipeline for estimating gait
characteristics and candidate stability metrics from walking video. It is not a
diagnostic medical device. Unvalidated outputs must not be presented as clinical
conclusions.

## Configuration and Runtime

- `opencode.docker.json` is the tracked source configuration.
- `run-opencode.sh` copies that file to the ignored, generated `opencode.json`
  and launches the `local-opencode` Docker image.
- The default agent is `orchestrator`.
- `AGENTS.md` defines repository, scientific, software, and workflow rules.
- The Codebase Memory MCP server and plugin are configured for repository
  context and indexing hooks.
- Project skills are loaded from `.opencode/skills`.

## Agents

| Agent | Role | Model |
| --- | --- | --- |
| `orchestrator` | Primary; plans, delegates, and verifies | `openai/gpt-5.6-sol` |
| `researcher` | Read-only evidence research | `openai/gpt-5.6-terra` |
| `python-implementer` | Bounded Python changes | `opencode/north-mini-code-free` |
| `test-engineer` | Deterministic tests | `opencode/north-mini-code-free` |
| `biomechanics-reviewer` | Read-only scientific and measurement review | `openai/gpt-5.6-sol` |
| `code-reviewer` | Read-only engineering review | `opencode/nemotron-3-ultra-free` |

## Skills and Commands

Project skills:

- `experiment-reproducibility`, `gait-biomechanics`, `pose-quality-control`,
  `stability-metrics`, `validation-and-claims`, and `video-ingestion`.

Slash commands:

- `/plan-feature`: plan bounded feature work.
- `/record-experiment`: record reproducibility and provenance details.
- `/review-change`: review a repository change.
- `/run-checks`: run the configured verification workflow.

## Working Model

The orchestrator inspects relevant repository state, develops a bounded plan,
and delegates implementation, testing, and research to specialized agents. It
requires biomechanics review when scientific or measurement semantics change
and code review for nontrivial code changes. It then independently verifies the
required checks and final diff rather than relying on an implementation agent's
report.

All work must preserve the distinctions among:

- observed video;
- pose estimates;
- processed trajectories;
- biomechanical estimates;
- validated measurements; and
- unvalidated proxies.

These categories must not be conflated, and clinical validity must not be
claimed without supporting validation evidence.

## Detailed Rules

LLMs should read `AGENTS.md` first, then consult `.opencode/agents/`,
`.opencode/skills/`, and `.opencode/commands/` for the detailed role,
permission, scientific, and workflow instructions.
