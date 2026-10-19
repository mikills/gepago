# v0.8.0 progress

Branch `feat/v0.8.0-flow-runtime`, started from `1e3cc7b` (v0.7.0).

## Packages

| Package | Status | Commits | Review rounds | Open findings |
|---|---|---|---|---|
| A. Model call contract | done | a1791af, 53893a4, b8fd03f | 1: 14 findings, fixed; 2: 4 findings, fixed; 3: none | none |
| B. Structured output | done | 6eb19d9, 59c26b8, 04521cf, fb9e31a, 096233f, 3ff9b8c, 75803f9, c468be7 | 1: 11, 2: 3, 3: 3, 4: 2 + deferred, 5: 2 + perf, 6: 1, 7: 2, all fixed; 8: none | none |
| C. Pipeline control flow and policy | done (merged in 1306f2b) | 8a69ad8, 9b95e07, c76a7b9, 82e32ca, 3c5a14b, merge 1306f2b | 1: 5, 2: 3, 3: 1, 4: 1 finding, all fixed; 5: none | none |
| I. Tools | review round 1 | 3344220 (worktree branch) | | |
| D. Observability | review round 1 | 5351b86 (worktree branch) | | |
| E. Run context | not started | | | |
| F. Artefacts and compile | done (merged in 8212dbf) | 442358c, 4c92e2a, 309504d, merge 8212dbf | 1: 2, 2: 1, all fixed; 3: none | none |
| G. Test kit | not started | | | |
| H. Housekeeping | not started | | | |

## Commands

| Command | Result | Commit |
|---|---|---|
| clean worktree: `make fmt`, `git diff --exit-code`, `make test`, `make vet`, `make staticcheck` | pass | a1791af |
| same | pass | 53893a4 |
| same | pass | b8fd03f |
| same | pass | 8a69ad8 (C, worktree branch) |
| same | pass | 9b95e07 (C, worktree branch) |
| clean worktree: full set (fmt, diff, test, vet, staticcheck, diago, wasm-test) | pass | 59c26b8 |
| clean worktree: fmt, diff, test, vet, staticcheck | pass | c76a7b9 (C, worktree branch) |
| same | pass | 82e32ca (C, worktree branch) |
| same | pass | 3c5a14b (C, worktree branch) |
| clean worktree: full set | pass | 04521cf |
| clean worktree: fmt, diff, test, vet, staticcheck | pass | fb9e31a |
| clean worktree: full set (after merging C; one mechanical conflict in program.go resolved as the union of both sides) | pass | 1306f2b |
| clean worktree: full set | pass | 096233f |
| clean worktree: fmt, diff, test, vet, staticcheck | pass | 3ff9b8c, 75803f9, c468be7 |
| clean worktree: fmt, diff, test, vet, staticcheck | pass | 442358c (F), 5351b86 (D), worktree branches |
| same | pass | 4c92e2a, 309504d (F, worktree branch) |
| same (after merging F, no conflicts) | pass | 8212dbf |
| clean worktree: same plus `npm ci`, `make diago`, `make wasm-test` | pass | 6eb19d9 |

## Decisions

- Chat and tool types (`Message`, `ToolCall`, `Tool`, `ChatRequest`, `ChatResponse`,
  `ChatClient`, `FinishReason`) move to the root package because `agents` imports it and
  `Predict` and pipeline steps need them. `agents` keeps type aliases.
- H2: delete `ModelRouter`. Nothing in the repo uses it, and roles would need a model
  selection API on `Predict` and `Compile` that the spec does not otherwise call for.
  Done early in A's review fixes (53893a4) because A moved every model consumer to `Generator`.
- Budgets count physical provider calls and every token spent, including retried attempts
  inside `RetryingLanguageModel` (C review round 1).
- `Build` rejects any two components that produce the same candidate key.
- An `OnErrorFallback` step runs under its own budget frame: limited by its own
  `PerStepTokens` entry and the run limits, not by the failed step's limit (C review round 2).
- A step that ignores its context is not bounded by `Timeout` or `MaxDuration`: the runner
  waits for it rather than abandoning a goroutine.
- Built-in metrics compare typed Go values (from `TypedPredict` and pipeline writes) through
  a canonical form rather than `fmt.Sprint` (B review round 2).
- Providers report a refusal as a `Generation` with `FinishContentFilter`; `Predict` fails
  with `ErrContentFiltered` (matches `ErrInvalidOutput`) and never repairs it.
- To save time, I, D and F run in parallel worktrees from 096233f (none depends on another's
  types); E follows D, then G and H.
- `LoadOrTrain` always checks the fingerprint: a config's `ProgramVersion` is a label, not an explicit
  override (F review round 1).
- `gepa.TraceDatasetBuilder` mirrors `agents.TraceDatasetBuilder`: it implements
  `ReflectiveDatasetBuilder`, and `ProgramEvaluator` records a `PipelineTrace` per example (D review round 1).
- Agent chat calls go through `AdmitModelCall` and `RecordUsage`; `RetryingLanguageModel` gains `Chat`.
- `programs/kiln` and `programs/documentscout` are gitignored local projects inside the root
  module. They are not part of the repo; the final `make verify` runs in a clean worktree,
  as CI does.

## Breaking changes collected

B (structured output):
- `Field.Required` removed; outputs are required by default and `Field.Optional` opts out.
- `Predict` rejects missing or null required outputs, extra fields and values that do not match `Type`, as `ErrInvalidOutput`; `ChainOfThought` replies must include the reasoning field.
- An invalid reply is returned with its error, and the reply is in step details under `"output"`.
- `RepairPrompt` wording changed and includes the schema; prompts for typed signatures gain a schema block.
- Pipeline missing/null messages for Predict steps read `signature "x": output "b" is missing`.
- OpenAI and Claude send native structured output for strict schemas; models without support reject those requests (`gepa.GeneratorFunc(lm.Generate)` opts out).
- An OpenAI refusal returns "model refused: ..." with `FinishContentFilter`.
- WASM predict jobs apply the same output checks; the JS `Field` type drops `required` and adds `type`, `schema`, `optional`.
- `RepairPrompt(prompt, raw, err)` takes the original prompt and repeats it; wording changed.
- `ChainOfThought`'s reasoning field comes first in the output schema (Go and TS).
- Providers report refusals as `FinishContentFilter` without an error; `Predict`, the proposers and judges fail with `ErrContentFiltered`. Truncation errors end with ", N completion tokens".
- `TypedPredict.RunCandidate` returns decoded Go values.
- `Key.Get` rejects wrong-length arrays and float underflow.
- OpenAI and Google `Generate` return refusals without an error.
- TS OpenAI and Claude model callbacks send strict formats; OpenAI-compatible endpoints without structured-output support can return 400.
- Metric text for nil is "null" (was "<nil>"); non-string values compare by canonical JSON, which changes `ClassificationMetric` details.
- crucible `RawModelSubject` and `AgentSubject` fail on truncation or refusal; the OpenAI agent subject maps refusals to `content_filter`.
- `Key.Get` rejects a quoted "null", a quoted float that underflows and quoted non-JSON numbers ("01", "1.") in `,string` fields.
- `Build` checks a signature step's reads and writes against its signature names and Go types (`SignatureProgram`); a `TypedPredict` step must declare its required `Out` fields as writes with exact types.
- An absent or null `Optional` output commits the zero value of the key's type instead of failing the step.
- `StrictJSONSchema` is stricter (provider limits, unique required, scalar enums).
- WASM task and repair host requests carry `response_format`; the TS Claude chat sends `output_config` for a strict `response_format`.

C (pipeline control flow):
- `StepError.Err` is the step's own error, not a one-element `errors.Join` (messages and `errors.Is` unchanged).
- `RecordUsage` charges the run budget in the context; inside a budgeted pipeline, `Predict`, `LLMJudgeMetric`, the proposers and `RetryingLanguageModel` can return a `Budget` error before calling the model.
- Budget errors say "tokens spent" (were "tokens recorded").
- When a budget or timeout ends a step's context, a step that finishes anyway fails and its writes are dropped.
- `PipelineSpec` gains `Defaults` and `Budget` (breaks unkeyed struct literals only).

F (artefacts and compile):
- `StepError` gains `Reads`; `ProgramArtifact` gains `Fingerprint` (unkeyed literals break).
- Every loader returns `*FingerprintMismatchError` when a v0.8 artefact's fingerprint differs from the programme's or the programme has none; `LoadOrTrain` always checks.
- `TrainAndSave` and `CompiledProgram.Artifact` write `fingerprint` into the artefact JSON.
- The `ProgramEvaluator` record for an invalid output gains `step` and `reads` keys.
- For `WithPolicy` with `OnErrorFallback`, `StepError.Err` wraps a `StepError` for the failed step (text and `errors.Is` unchanged).

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
- crucible's go.mod requires gepago v0.6.0; bump it after the root v0.8.0 tag.
- F's `StepError.Reads` and reflection `SideInfo` must go through D's `Redactor` when D and F merge.
- README and `docs/architecture.html` still describe `LanguageModel` (H3).
