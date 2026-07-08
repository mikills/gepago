# Crucible PRD

## Problem

GEPAgo needs a reusable evaluation program for comparing model-like systems: raw models, GEPA programs, compiled artifacts, agents, HTTP endpoints, commands, and custom Go functions. Users should be able to define suites without writing Go for every eval, run them in CI, inspect failures, and compare candidate runs against baselines.

## Goals

- Provide an independently installable CLI under `programs/crucible`.
- Support dynamic JSON-defined suites and subject configs.
- Keep the core provider-agnostic through `Subject` and `Evaluator` interfaces.
- Support custom subject types and evaluators in Go.
- Include useful built-ins for tool choice and JSON reliability.
- Produce machine-readable JSON artifacts, CSV summaries, and static HTML reports.
- Support regression comparison for CI gating.

## Non-goals

- Replacing GEPA optimization primitives.
- Requiring every eval to be authored in Go.
- Locking the agent abstraction to any one provider.

## Core model

```text
subjects × cases × evaluators -> run artifact -> reports/comparisons
```

A `Subject` is anything that can turn a case input into observable output. An `Evaluator` scores that output. A `Suite` contains cases, evaluator configuration, and optional pairwise comparison configuration.

## CLI

```bash
crucible validate-suite suite.json
crucible run -suite suite.json -subjects subjects.json -out .crucible
crucible compare -baseline old.json -candidate new.json -html compare.html -fail-drop 0.05
crucible render-report -run run.json -html report.html -csv summary.csv
crucible built-ins list
```

## Provider support

Crucible agents use the provider-neutral `agents.ChatClient` interface. Built-in JSON adapters include:

- OpenAI-compatible Chat Completions via `provider_api: "chat_completions"` or `"completions"`.
- OpenAI-compatible Responses API via `provider_api: "responses"`.
- Anthropic Messages API via `provider: "anthropic"`.

Users can register custom subject type names with `RegisterSubjectType` for additional providers or internal runtimes.

## Acceptance criteria

- `go install github.com/mikills/gepago/programs/crucible/cmd/crucible@latest` works after release.
- `go test ./...` passes within the module.
- `diago audit -target ./...` passes within the module and repository.
- Command subjects/evaluators require explicit opt-in from CLI.
- Reports are self-contained and suitable for sharing.
