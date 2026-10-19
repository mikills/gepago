# v0.8.0 progress

Branch `feat/v0.8.0-flow-runtime`, started from `1e3cc7b` (v0.7.0).

## Packages

| Package | Status | Commits | Review rounds | Open findings |
|---|---|---|---|---|
| A. Model call contract | done | a1791af, 53893a4, b8fd03f | 1: 14 findings, fixed; 2: 4 findings, fixed; 3: none | none |
| B. Structured output | implementing | | | |
| C. Pipeline control flow and policy | implementing (worktree) | | | |
| I. Tools | not started | | | |
| D. Observability | not started | | | |
| E. Run context | not started | | | |
| F. Artefacts and compile | not started | | | |
| G. Test kit | not started | | | |
| H. Housekeeping | not started | | | |

## Commands

| Command | Result | Commit |
|---|---|---|
| clean worktree: `make fmt`, `git diff --exit-code`, `make test`, `make vet`, `make staticcheck` | pass | a1791af |
| same | pass | 53893a4 |
| same | pass | b8fd03f |

## Decisions

- Chat and tool types (`Message`, `ToolCall`, `Tool`, `ChatRequest`, `ChatResponse`,
  `ChatClient`, `FinishReason`) move to the root package because `agents` imports it and
  `Predict` and pipeline steps need them. `agents` keeps type aliases.
- H2: delete `ModelRouter`. Nothing in the repo uses it, and roles would need a model
  selection API on `Predict` and `Compile` that the spec does not otherwise call for.
  Done early in A's review fixes (53893a4) because A moved every model consumer to `Generator`.
- `programs/kiln` and `programs/documentscout` are gitignored local projects inside the root
  module. They are not part of the repo; the final `make verify` runs in a clean worktree,
  as CI does.

## Breaking changes collected

A (model call contract):
- `Predict`/`ChainOfThought` `LM` and `RepairLM`, `LLMJudgeMetric.LM`, `ReflectiveProposer.LM`, `LLMMergeProposer.LM`, `ReflectionLM` on `CompileConfig`, `OptimizationConfig` and `ProgramTrainConfig`, `DefaultCompileConfig` and `RetryingLanguageModel.LM` take `Generator`.
- crucible: judge and subject `LM`, `JudgeLM`, `RAGGroundingPack` and `BuildLanguageModel` use `gepa.Generator`.
- Provider `Generate` returns `(gepa.Generation, error)`; providers no longer satisfy `LanguageModel`.
- Removed: provider `LastUsage`/`LastResolvedModel`, `Predict.LastUsage`, `ChainOfThought.LastUsage`, `ProgramLastUsage`, `Pipeline.LastUsage`, `PipelineProgram.LastUsage`, `RetryingLanguageModel.LastUsage`.
- `RunWithUsage`, `ProgramEvaluator` and crucible `ProgramSubject` no longer fall back to `LastUsage`; custom programmes call `RecordUsage`.
- `RetryingLanguageModel` retries only transient errors, with capped exponential backoff and jitter.
- `agents.FinishReason` is an alias of `gepa.FinishReason`.
- `Predict` fails with `ErrOutputTruncated` on `FinishLength` instead of repairing.
- Google `CompletionTokens` includes `thoughtsTokenCount`.
- Providers return context errors bare; pi's error text changed.
- Ledger `ModelCalls` counts failed proposals and merge calls.
- Resuming a saved optimisation fails the config fingerprint when the reflection model's Go type changed (adapted legacy models keep their v0.7 fingerprint).
- OpenAI and Claude providers no longer retry inside the SDK.
- `ModelRouter`, `ModelRole` and the `Role*` constants are deleted.
- Missing or unknown finish reasons are `""`.
- `ReflectiveProposer`, `LLMMergeProposer`, `LLMJudgeMetric` and crucible's judges fail with `ErrOutputTruncated` on `FinishLength`.
- Claude `model_context_window_exceeded` maps to `length`.
- TLS, unknown-host and bad-URL errors are permanent; other unrecognised transport errors are unclassified.
- Google error text shows `REDACTED` in place of the API key.
- crucible Anthropic subject `PromptTokens` includes cache tokens.
- `UsageLedger` JSON gains three fields; a content-filter stop ends an agent with `StopContentFilter`.
- WASM: unknown host `finishReason` is an error; host usage with negative breakdown fields is rejected; predict fails on `length` without a repair.
- TypeScript: `FinishReason` and `StopReason` gain `content_filter`; `ChatResponse.finish_reason` is optional; providers omit unknown finish reasons instead of sending `stop`.
- HTTP/2 stream resets and GOAWAY are transient; bad addresses are permanent.
- Pi chat responses no longer carry `finish_reason: "stop"`.

## Open findings

- `make fmt` and `make verify` fail in the main checkout because the gitignored kiln and documentscout no longer compile against the new API. CI and the clean-worktree checks are unaffected.
- README and `docs/architecture.html` still describe `LanguageModel` (H3).
