---
description: Run the project's available verification checks
agent: orchestrator
---

Determine which verification commands are currently configured in this
repository and run the appropriate checks.

Prefer, when available:

1. formatting check;
2. linting;
3. static type checking;
4. unit tests;
5. relevant integration tests.

Do not invent commands that are not configured.

Report:

- commands executed;
- pass/fail status;
- important failures;
- whether failures appear related to current working-tree changes.
