# v0.8.0 progress

Branch `feat/v0.8.0-flow-runtime`, started from `1e3cc7b` (v0.7.0).

## Packages

| Package | Status | Commits | Review rounds | Open findings |
|---|---|---|---|---|
| A. Model call contract | not started | | | |
| B. Structured output | not started | | | |
| C. Pipeline control flow and policy | not started | | | |
| I. Tools | not started | | | |
| D. Observability | not started | | | |
| E. Run context | not started | | | |
| F. Artefacts and compile | not started | | | |
| G. Test kit | not started | | | |
| H. Housekeeping | not started | | | |

## Commands

| Command | Result | Commit |
|---|---|---|

## Decisions

- Chat and tool types (`Message`, `ToolCall`, `Tool`, `ChatRequest`, `ChatResponse`,
  `ChatClient`, `FinishReason`) move to the root package because `agents` imports it and
  `Predict` and pipeline steps need them. `agents` keeps type aliases.
- H2: delete `ModelRouter`. Nothing in the repo uses it, and roles would need a model
  selection API on `Predict` and `Compile` that the spec does not otherwise call for.
- `programs/kiln` and `programs/documentscout` are gitignored local projects inside the root
  module. They are not part of the repo; the final `make verify` runs in a clean worktree,
  as CI does.

## Breaking changes collected

## Open findings
