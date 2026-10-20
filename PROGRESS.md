# v0.8.0 progress

Status: all packages done; version and dependency bumps done; final `make verify` passes at c943353. Not pushed, no PR, not tagged.

Branch `feat/v0.8.0-flow-runtime`, started from `1e3cc7b` (v0.7.0).

## Packages

| Package | Status | Commits | Review rounds | Open findings |
|---|---|---|---|---|
| A. Model call contract | done | a1791af, 53893a4, b8fd03f | 1: 14 findings, fixed; 2: 4 findings, fixed; 3: none | none |
| B. Structured output | done | 6eb19d9, 59c26b8, 04521cf, fb9e31a, 096233f, 3ff9b8c, 75803f9, c468be7 | 1: 11, 2: 3, 3: 3, 4: 2 + deferred, 5: 2 + perf, 6: 1, 7: 2, all fixed; 8: none | none |
| C. Pipeline control flow and policy | done (merged in 1306f2b) | 8a69ad8, 9b95e07, c76a7b9, 82e32ca, 3c5a14b, merge 1306f2b | 1: 5, 2: 3, 3: 1, 4: 1 finding, all fixed; 5: none | none |
| Final review (v0.7.0..68b712a) | done | 68b712a | 1: 1 low–medium (TS artefact helpers dropped fingerprint) + 1 nit + spelling, fixed; 2: none | none |
| Version and dependency bumps | done | f023769, 657326c, c943353 | 1: 6 (Claude SDK tracing on by default, undocumented release order, redirect error kind, thinking tokens, go.mod tidiness accepted, dependency note), fixed; 2: 1 (CHANGELOG module list), fixed; 3: none | none |
| I. Tools | done (merged in 19cc05e) | 3344220, acc200b, 58cf7ec, a696312, merge 19cc05e | 1: 8, 2: 3, all fixed; 3: none | none |
| D. Observability | done (merged in fdaa382, 7e33dda) | 5351b86, 40af86e, 0e0d914, bbeaf71, 040c8a6, 72d5707, 0459b30 (worktree branch) | 1: 4 + D2 partial, 2: 4, 3: 4 + 1 decision, 4: 2, 5: 2, 6: 1, all fixed; 7: none | none |
| D/F/I integration | done | merge fdaa382, 6bd1530, merge 7e33dda, 108cc8d, 8cf4f9d, 773ba2d | 1: 2, 2: 3, 3: 1, all fixed; 4: none | none |
| E. Run context | done (feature branch fast-forwarded to 93b9125) | 802e2ab, merge ac100ad, 2e1ecb5, merge 15de703, a7313f7, 93b9125 | 1: 2 + doc nit, fixed; 2: none; flaky test found by the orchestrator, fixed; 3: 1 (test helper too broad), fixed; 4: none | none |
| F. Artefacts and compile | done (merged in 8212dbf) | 442358c, 4c92e2a, 309504d, merge 8212dbf | 1: 2, 2: 1, all fixed; 3: none | none |
| G. Test kit | done (merged in c55c605) | cb9d266, merge dcd3aad, bb759d1, merge 6830be0, ce202ba | 1: 3, 2: 3, all fixed; 3: none | none |
| H. Housekeeping | done (merged in c55c605) | d54cdd0, merge 1281d5a, 471b8bd, 8324ab0, 576108f, merge 6830be0 | 1: 3 + missing CHANGELOG entries, 2: 3, all fixed; 3: none | none |

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
| clean worktree: full set | pass | 3344220 (I, worktree branch) |
| clean worktree: fmt, diff, test, vet, staticcheck | pass | acc200b (I, worktree branch) |
| clean worktree: full set | pass | 58cf7ec, a696312 (I, worktree branch) |
| clean worktree: fmt, diff, test, vet, staticcheck | pass | 0e0d914, bbeaf71, 040c8a6 (D, worktree branch) |
| clean worktree: full set (after merging I; three conflicts in pipeline.go, pipeline_build.go, pipeline_exec.go resolved as the union of both sides) | pass | 19cc05e |
| clean worktree: fmt, diff, test, vet, staticcheck | pass | 72d5707, 0459b30 (D, worktree branch) |
| clean worktree: full set | pass | 7e33dda (D merged and integrated) |
| clean worktree: fmt, diff, test, vet, staticcheck | pass | 108cc8d, 8cf4f9d, 773ba2d (integration) |
| clean worktree: full set | pass | ac100ad (E) |
| clean worktree: full set | FAIL (make diago: disk full, "no space left on device") | 15de703 |
| clean worktree: full set (after freeing disk) | pass | 15de703 |
| clean worktree: fmt, diff, test, vet, staticcheck | pass | a7313f7, 93b9125 |
| `go test -race -count=300` on the redaction tests | pass | 93b9125 |
| clean worktree: full set | pass | 1281d5a (H), dcd3aad (G merged), bb759d1 (G) |
| clean worktree: full set | FAIL (staticcheck U1000 in run_compiled.go) | 471b8bd |
| clean worktree: full set | pass | 8324ab0, 576108f (H), 6830be0 (G+H) |
| clean worktree: full set | pass | ce202ba (G) |
| clean worktree: `npm ci`, then `make verify` (fmt, test, vet, staticcheck, diago, wasm-test; both modules), then `git diff --exit-code` | pass, no formatting diff (diago 6/6 checks per module, TS 19/19) | c55c605 |
| clean worktree: `npm ci`, then `make verify`, then `git diff --exit-code` | pass, no formatting diff (diago 6/6 per module, TS 22/22) | 68b712a (final) |
| clean worktree: `npm ci`, then `make verify`, then `git diff --exit-code` | pass, no formatting diff (diago 6/6 per module, TS 24/24) | f023769, 657326c, c943353 (final) |
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
- Inside a pipeline every tool call goes through the run's one tool session; `Build` rejects a Predict
  step with its own `ToolPolicy` or another `ToolRegistry`. Tool memory writes in a pipeline commit with
  the step's state writes. The tool cache key does not normalise number forms. Pi has no `ChatClient`
  (its print mode cannot return tool calls). Compile records pipeline traces only for `TraceDatasetBuilder`.
- Redaction covers reflection records: with a pipeline `Redactor`, `ProgramEvaluator` redacts each item's
  `Output` and `SideInfo` (Redactors found from the programme's structure, plus run-time joins for custom
  Programs); the metric sees real values. Observer contexts end with the caller's context, the run's
  `MaxDuration` or the run's end. Only comparable observers are de-duplicated.
- A Redactor's plain-string result is used verbatim as its replacement in text; a Redactor must not
  put other hidden values into its own result (to document in H). Per-field replacements from JSON
  results never enter text (integration review round 3).
- Run context: every run takes the ctx RunContext (TraceID and Attributes kept, its RunID becomes
  ParentRunID, fresh RunID); attributes pass the Redactor as "attribute:<name>"; provider logs carry
  attributes unredacted (filter in the slog handler). A run with a Redactor and no observers is
  observed silently so nested runs inherit it.
- E1 scope: agent runs inside a pipeline step keep their own `Memory.RunID` span trace id; they see the
  step's `RunContext` through ctx but agent traces do not carry it (final review, not counted).
- `RetryingLanguageModel` is not on the CHANGELOG unkeyed-literal list: it has had an unexported `mu`
  field since v0.7, so unkeyed literals were never possible outside the package (H review round 2).
- Dependencies move to the newest releases that support Go 1.25 (CI); several next releases need Go 1.26.
  `a-h/templ` and `golang.org/x/sync` stay as direct requirements for the gitignored local kiln.
  The Claude provider turns off anthropic-sdk-go's default OpenTelemetry tracing (`Config.OpenTelemetry` opts in).
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

I (tools):
- `agents` chat types (`Role`, `Message`, `ToolCall`, `Tool`, `ChatRequest`, `ChatResponse`, `ChatClient`) are aliases of new root types; `agents.JSONSchemaResponseFormat` is `gepa.ResponseFormat`.
- `Tool.Parameters` and `ResponseFormat.Schema` are `json.RawMessage` (were `map[string]any`); `ResponseFormat` JSON keys are `name`/`schema`/`strict`.
- `Tool.Validate` rejects an invalid or non-object schema; `ResponseFormat.Validate` requires a valid name and schema; `Tool.InputSchema` adds `"type":"object"`.
- `agents.ErrUnknownTool` is `gepa.ErrUnknownTool` ("unknown tool"); agent tool errors and `ToolCallRecord.Err` are prefixed `tool "<name>": `.
- `Agent.Validate` builds middleware and a tool session; agent tool panics are contained.
- Agents admit and record each chat call (run budgets apply; do not also record `RunResult.Usage`); `RunResult.Usage` includes tool usage.
- `RetryingLanguageModel` implements `ChatClient`.
- Google chat sends `parametersJsonSchema`/`responseJsonSchema`, drops the response format when tools are present, merges consecutive tool results and sends function-call ids.
- OpenAI sends `parallel_tool_calls:false` when any tool is strict.
- WASM host chat requests send normalised tool schemas; payload tools with invalid schemas are rejected.
- `Build` rejects a `Predict`/`TypedPredict` step with its own `ToolPolicy` or another `ToolRegistry`; `Predict.Validate` rejects a `ToolPolicy` without a `ToolRegistry`.
- A pipeline `ToolPolicy` governs nested agents' and pipelines' tool calls, even without `Tools`.
- Pipeline tool memory writes and cache entries commit with the step; a cached record has zero `Usage`.
- crucible callers marshal their schema maps.

D (observability):
- An `OnErrorSkip`, and a failed top-level merge under a wrapper, each use an attempt number (`extract#2 skipped`).
- `PipelineSpec`, `CompileConfig`, `ProgramTrainConfig` and `ProgramEvaluator` gain fields (unkeyed literals break).
- A failing top-level merge is now a failed step execution.
- With a pipeline `Redactor`, `ProgramEvaluator` returns redacted `Output` and `SideInfo`.

E (run context) and the D/F/I integration:
- Provider `Config` types gain `Logger`; panicking model calls are logged before the panic continues.
- Event types and `PipelineTrace` gain `RunContext`; a run's span ID is its `RunID` and the trace ID is the `RunContext` `TraceID`; the HTML report gains a Run context section.
- Observer contexts hide the RunContext, the run's tool session, tool memory and step stage.
- A pipeline with a `Redactor` and no observers calls the Redactor for every value; nested observers see outer-redacted values with a `ParentID` they never saw.
- `StepFinishedEvent.ToolCalls` and `StepTrace.ToolCalls`; Redactors receive `"tool:<name>"` and `"attribute:<name>"` keys.
- Scrubbed `json.RawMessage` keeps key order and duplicates; scrubbed numbers may become strings; `ProgramEvaluator` redacts SideInfo "reads".
- The HTML report shows JSON compact past 32 levels or 64 KiB.

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

None in the release. Follow-ups outside it:

- `make fmt` and `make verify` fail in the main checkout because the gitignored kiln and documentscout no longer compile against the new API. CI and the clean-worktree checks are unaffected.
- Release order (CONTRIBUTING.md): tag v0.8.0; in programs/crucible run `GOWORK=off go mod tidy`, commit go.sum and drop the go.work replace; then tag programs/crucible.
