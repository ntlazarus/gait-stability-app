---
description: Refresh the quick-reference snapshot of the application's current capabilities and usage
agent: orchestrator
---

Refresh the current application snapshot.

Load the `app-state-snapshot` skill.

The canonical output is:

`docs/PROJECT_STATE.md`

Inspect the repository before writing. Determine what is actually implemented
now rather than describing the intended future product.

Inspect relevant:

- source code;
- scripts and CLI entry points;
- pyproject/package configuration;
- configuration files;
- tests;
- README;
- architecture and measurement documentation;
- current Git state.

Run safe, focused verification commands where practical so that important usage
instructions are based on fresh evidence.

Do not use real participant/patient data simply to verify documentation.
Prefer synthetic fixtures and existing safe examples.

Do not implement features or change application behavior as part of this task.

If you cannot edit the documentation directly, delegate the bounded
documentation update to an appropriate implementation subagent and then review
the resulting diff yourself.

`docs/PROJECT_STATE.md` must make it easy for someone returning to the project
later to answer:

1. What can this project do right now?
2. What can it not do yet?
3. How do I set it up?
4. How do I run the main workflow?
5. How do I exercise each important current feature?
6. What inputs does each workflow require?
7. What outputs should I expect and where are they?
8. How do I run tests and perform a quick smoke test?
9. What important scientific or software limitations should I know?
10. What are the important directories/configuration files?

Include current branch, HEAD commit, and dirty/clean working-tree state.

Remove obsolete instructions from an earlier snapshot.

At completion:

- summarize what changed in `docs/PROJECT_STATE.md`;
- identify any run instructions that could not be freshly verified;
- report documentation/code discrepancies;
- do not claim capabilities unsupported by the repository.
