# Changelog

All notable changes to `gepago` will be documented in this file.

The project follows semantic versioning once public tags are published.

## v0.8.0

`Pipeline` becomes the unit of a production flow: typed steps, bounded model calls, tools, per-step policy and a trace for every run.

### Breaking changes

#### Model calls

- Every model consumer takes a `Generator`, whose `Generate` returns a `Generation{Text, Usage, Model, FinishReason}`: `Predict` and `ChainOfThought` `LM` and `RepairLM`, `LLMJudgeMetric.LM`, `ReflectiveProposer.LM`, `LLMMergeProposer.LM`, `ReflectionLM` on `CompileConfig`, `OptimizationConfig` and `ProgramTrainConfig`, `DefaultCompileConfig` and `RetryingLanguageModel.LM`. Wrap a `LanguageModel` with `AdaptLanguageModel`.
- Usage travels in the `Generation` instead of being read back after the call: `RunWithUsage`, `ProgramEvaluator` and crucible's `ProgramSubject` no longer fall back to `LastUsage`, so a custom programme reports its usage with `RecordUsage`.
- `RecordUsage` also charges the run budget in the context; inside a budgeted pipeline, `Predict`, `LLMJudgeMetric`, the proposers and `RetryingLanguageModel` can return a `Budget` error before calling the model.
- `RetryingLanguageModel` retries only transient errors, so an error without a kind (see `TransientError`) is no longer retried, and waits with capped exponential backoff and jitter from `Delay`. It implements `ChatClient`.
- `Predict` fails with `ErrOutputTruncated` on `FinishLength` instead of running the repair prompt; `ReflectiveProposer`, `LLMMergeProposer`, `LLMJudgeMetric` and crucible's judges fail with it too. Truncation errors end with ", N completion tokens".
- A refusal is a `Generation` with `FinishContentFilter` and no error (OpenAI's text reads "model refused: ..."); `Predict`, the proposers and the judges fail on it with `ErrContentFiltered`, which matches `ErrInvalidOutput`, and never repair it.
- Missing or unknown finish reasons are `""`. `agents.FinishReason` is an alias of `gepa.FinishReason`, and a content-filter stop ends an agent with `StopContentFilter`.
- `UsageLedger` JSON gains `reasoning_tokens`, `cached_input_tokens` and `cache_creation_tokens`, and its `ModelCalls` counts failed proposals and merge calls.
- Resuming a saved optimisation fails the config fingerprint check when the reflection model's Go type changed; a model wrapped by `AdaptLanguageModel` keeps its v0.7 fingerprint.

#### Structured output

- `Field.Required` is removed: outputs are required by default and `Field.Optional` opts out. In a pipeline, an absent or null `Optional` output commits the zero value of the key's type.
- `Predict` rejects missing or null required outputs, extra fields and values that do not match `Type` as `ErrInvalidOutput`, and returns the invalid reply with its error; in a pipeline the reply is in `StepError.Details["output"]`. Missing and null outputs read `signature "x": output "b" is missing`.
- `ChainOfThought` replies must include the reasoning field, which comes first in the output schema (Go and TypeScript).
- `RepairPrompt(prompt, raw, err)` takes the original prompt and repeats it, and its wording changed to include the schema. Prompts for typed signatures gain a schema block.
- OpenAI and Claude send native structured output for strict schemas, so a model or endpoint without support rejects those requests; `gepa.GeneratorFunc(lm.Generate)` sends the schema in the prompt only.
- `Key.Get` rejects wrong-length arrays, floats that underflow, a quoted `"null"` and quoted non-JSON numbers such as `"01"` and `"1."` in `,string` fields.
- Metric text for nil is `null` (was `<nil>`), and non-string values compare by canonical JSON, which changes `ClassificationMetric` details.
- `Build` checks a signature step's reads and writes against its signature's names and Go types (`SignatureProgram`).

#### Pipelines and evaluation

- `PipelineSpec`, `CompileConfig`, `ProgramTrainConfig`, `ProgramEvaluator`, `StepError` (`Reads`), `ProgramArtifact` (`Fingerprint`) and the provider `Config`s (`Logger`) gain fields, so unkeyed struct literals of them break.
- `StepError.Err` is the step's own error, not a one-element `errors.Join`; messages and `errors.Is` are unchanged.
- A failing top-level merge is a failed step execution.
- The `ProgramEvaluator` record for an invalid output gains `step` and `reads` keys, which `DefaultReflectiveDatasetBuilder` passes to reflection.

#### Tools and agents

- The chat types `Role`, `Message`, `ToolCall`, `Tool`, `ChatRequest`, `ChatResponse` and `ChatClient` move to the root package and `agents` keeps aliases; `agents.JSONSchemaResponseFormat` is `gepa.ResponseFormat`.
- `Tool.Parameters` and `ResponseFormat.Schema` are `json.RawMessage` (were `map[string]any`), and `ResponseFormat`'s JSON keys are `name`, `schema` and `strict`; crucible marshals its schema maps.
- `Tool.Validate` rejects an invalid or non-object schema, `ResponseFormat.Validate` requires a valid name and schema, and `Tool.InputSchema` adds `"type":"object"`.
- `agents.ErrUnknownTool` is `gepa.ErrUnknownTool` ("unknown tool"), and agent tool errors and `ToolCallRecord.Err` are prefixed `tool "<name>": `.
- `Agent.Validate` builds middleware and a tool session, and a panicking agent tool is contained as a tool error.
- Agents admit and record each chat call, so run budgets apply and `RunResult.Usage` must not be recorded again; `RunResult.Usage` includes tool usage.

#### Providers

- Provider `Generate` returns `(gepa.Generation, error)`, so providers no longer satisfy `LanguageModel`.
- OpenAI and Claude no longer retry inside their SDKs; wrap them in `RetryingLanguageModel`.
- Errors are classified: 4xx is permanent except 429; 429, 5xx, network errors, HTTP/2 stream resets and GOAWAY are transient; TLS, unknown-host, bad-URL and bad-address errors are permanent; other unrecognised transport errors are unclassified. Context errors are returned bare, and pi's error text changed.
- Google `CompletionTokens` includes `thoughtsTokenCount`, its error text shows `REDACTED` in place of the API key, and its chat sends `parametersJsonSchema` and `responseJsonSchema`, drops the response format when tools are present, merges consecutive tool results and sends function-call ids.
- Claude's `model_context_window_exceeded` maps to `length`.

#### Artefacts

- Every loader returns a `*FingerprintMismatchError` when a v0.8.0 artefact's fingerprint differs from the programme's or the programme has none, and `LoadOrTrain` always checks it. `TrainAndSave` writes `fingerprint` into the artefact JSON. Artefacts from v0.7.0 and earlier have none and load as before.

#### WASM and TypeScript

- `compile` runs a `pipeline` programme as a typed `gepa.Pipeline`; its JSON shape is unchanged. A step without `inputKeys` sees only its signature's inputs (it saw every input and earlier output); a pipeline input whose first reading field sets `type` must convert to it; every input a step reads must be present (a missing signature input used to be left out of the prompt); an absent or null optional output holds its type's zero value; and a step that writes a key an earlier step read from the inputs, or steps that write one key as different types, fail validation. `runCompiled` runs pipelines the same way.
- WASM predict jobs apply the same output checks as `Predict` and fail on `length` without a repair; the JS `Field` type drops `required` and adds `type`, `schema` and `optional`.
- WASM task and repair host requests carry `response_format`; the TypeScript OpenAI and Claude model callbacks send strict formats (Claude chat as `output_config`), so OpenAI-compatible endpoints without structured output can return 400.
- An unknown host `finishReason` is an error, and host usage with negative breakdown fields is rejected.
- WASM host chat requests send normalised tool schemas, and payload tools with invalid schemas are rejected.
- TypeScript `FinishReason` and `StopReason` gain `content_filter`, `ChatResponse.finish_reason` is optional, providers omit unknown finish reasons instead of sending `stop`, and pi chat responses no longer carry `finish_reason: "stop"`.

#### Crucible

- Judge and subject `LM`, `JudgeLM`, `RAGGroundingPack` and `BuildLanguageModel` use `gepa.Generator`.
- `RawModelSubject` and `AgentSubject` fail on truncation or refusal, the OpenAI agent subject maps refusals to `content_filter`, and the Anthropic subject's `PromptTokens` includes cache tokens.

#### Removed

- `PipelineProgram` and `PipelineStep`: build a `Pipeline` with `PipelineSpec` and `NewProgramStep`. An artefact trained on a `PipelineProgram` has no fingerprint and loads into a `Pipeline` with the same step names.
- `ModelRouter`, `ModelRole` and the `Role*` constants. Nothing used them, and choosing models by role would need a model selection API on `Predict` and `Compile`.
- `LastUsage` and `LastResolvedModel` on the providers, `Predict.LastUsage`, `ChainOfThought.LastUsage`, `ProgramLastUsage`, `Pipeline.LastUsage`, `PipelineProgram.LastUsage` and `RetryingLanguageModel.LastUsage`.

### Added

- `Generator`, `Generation`, `GeneratorFunc` and `AdaptLanguageModel`; every provider implements `Generator` natively. `Usage` gains `ReasoningTokens`, `CachedInputTokens` and `CacheCreationTokens`, mapped by every provider and carried by `UsageLedger.AddUsage`.
- `FinishReason` (`stop`, `length`, `tool_calls`, `content_filter`), `ErrOutputTruncated` and `ErrContentFiltered`.
- Error kinds `Permanent`, `Transient` and `Budget`, with `IsPermanent`, `IsTransient`, `IsBudget` and `ErrorKindOf`; every provider classifies its errors.
- Structured output: `Field.Type`, `Field.Schema`, `Field.Optional` and `Field.GoType`, `Signature.JSONSchema`, `ResponseFormat` and `StrictJSONSchema`, the optional `StructuredGenerator` provider interface (OpenAI and Claude), a check of every reply before `Predict` returns it, and `TypedPredict[In, Out]`, which decodes with the rules `Key.Get` uses.
- Pipeline control flow, checked at `Build`: `WithPolicy` and `StepPolicy` (per-attempt `Timeout`, `RetryPolicy`, and `OnErrorFail`, `OnErrorSkip` with `PipelineSpec.Defaults` and `DefaultValue`, or `OnErrorFallback`), `Branch`, `Parallel`, `Fallback` and `WithCache`.
- `RunBudget` on `PipelineSpec` (`MaxDuration`, `MaxModelCalls`, `MaxTokens`, `PerStepTokens`), with `BudgetFrom(ctx).Remaining()` for steps. Exceeding a limit fails with a `Budget` error that names it, and a step that finishes after a budget or timeout ended its context fails with its writes dropped.
- Tools: `ToolRegistry` (`Register`, `Lookup`, `Schemas`), `ToolProvider`, `NewTool[Args, Out]` with a schema derived from `Args`, `NewToolFunc`, `ToolResult` (output, text, memory writes applied only on success, provenance, usage), `ToolPolicy` (allow-list, total and per-tool call limits, a cache keyed on name and canonical arguments), `ToolSession` and `ToolMemory`. In a pipeline, `PipelineSpec.Tools`, `ToolStep` and `Predict.Tools` with `MaxToolTurns` call through the run's one tool session, and a step's tool memory writes and cache entries commit with its state writes. Named `ChatMiddleware`, and `ChatClient` on the OpenAI, Claude and Google providers (pi's print mode cannot return tool calls).
- Observability: `PipelineObserver` with `RunStarted`, `StepStarted`, `StepFinished` (duration, usage, error, reads, writes, `ToolCalls`, cached, attempt) and `RunFinished`, built on `UsageSpan` with a run trace and a span per step; a panicking observer is contained. `RecordingObserver` records a serialisable `PipelineTrace` (with `StepTrace.ToolCalls`), `TraceDatasetBuilder` turns traces into reflection records, and `WriteHTMLRunReport` and `HTMLRunReport` render one run step by step, showing JSON indented up to 32 levels and 64 KiB and compact beyond.
- `Redactor` on `PipelineSpec`: every value an event, trace, run report or reflection record carries passes through it by state key, tool calls under `tool:<name>` and run attributes under `attribute:<name>`, and the text of each redacted value is removed from error messages and details. Scrubbed JSON keeps its key order and duplicate keys, and a number whose text changes becomes a string. A nested run's values pass through every enclosing Redactor too, so its observers see outer-redacted values, and spans whose `ParentID` belongs to an enclosing run they do not observe. A run with a Redactor and no observers still redacts every value, for the runs nested in it. With a Redactor, `ProgramEvaluator` redacts each item's `Output` and `SideInfo`, `reads` included, while the metric scores the real values.
- Run context: `RunContext{RunID, ParentRunID, TraceID, Attributes}`, `WithRunContext` and `RunContextFrom`. A run keeps the caller's trace ID and attributes, takes the caller's run as its parent and gets a fresh `RunID`, so nested pipelines form one trace. Events and `PipelineTrace` carry the `RunContext`, a run's span ID is its `RunID` and its trace ID is the `TraceID`, and the run report has a Run context section. Observer contexts carry neither the run's `RunContext` nor its `ToolMemory`, so an observer's own work is independent of the run. Provider `Config.Logger` writes one record per model call with the run context, a call that panics included.
- Artefacts: `Pipeline.Fingerprint` from step names, keys and types, the `Fingerprinter` interface, `ProgramArtifact.Fingerprint`, `CompiledProgram.Artifact` and `FingerprintMismatchError`. A `ProgramVersion` stated by both the artefact and `LoadCompiledProgramVersion` or `LoadCompiledVersion` stands in for the fingerprint.
- `DefaultReflectiveDatasetBuilder` names the failing step of an invalid output and carries its `StepError.Details` and the values it read.
- Test kit: package `gepatest` with `FakeGenerator`, a `Generator`, `StructuredGenerator` and `ChatClient` that answers from reply scripts keyed by step path or name, prompt substring or a default, each reply with its own text or tool calls, usage, finish reason and error, and records every call; and `AssertTrace`, which checks a `PipelineTrace`'s step executions (path, attempt, error text or kind, cached and skipped flags, usage, tool calls) and the run's error and usage, sorted by path and attempt, and reports a diff. `StepPathFrom(ctx)` returns the path a running step has in traces.

### Fixed

- A provider call that fails after spending tokens returns that usage with its error, so it is recorded exactly once, also when calls run concurrently; v0.7 read it back through `LastUsage`.
- `Predict` no longer sends a truncated reply through the repair prompt.
- `RetryingLanguageModel` no longer retries permanent errors or a call whose context has ended.
- Google provider error messages no longer include the API key.
- A panicking agent tool no longer crashes the agent run.

## v0.7.0

### Breaking changes

- `UsageReporter` implementations must reset their usage when each call starts. `Predict`, `LLMJudgeMetric` and `RetryingLanguageModel` now record usage for calls that return an error, so an implementation that keeps the previous call's usage counts it twice.
- `RetryingLanguageModel` has pointer receivers and must be used as `&gepa.RetryingLanguageModel{...}`. `LastUsage` now sums every attempt of the most recent call.
- `ProgramEvaluator` scores examples that fail with `ErrInvalidOutput` as zero instead of returning an error.

### Added

- Typed `PipelineSpec`/`Pipeline` with `Key[T]` state: build-time read/write and type checks, run-time type enforcement, copy-on-boundary values, read-only and strictly declared inputs, merge rules, signature-checked programme steps, strict decoding with `Validate`, and partial state with `StepError` details (including recovered panics) on failure. `PipelineProgram` remains for the WASM compile bridge.

### Fixed

- `ProgramEvaluator` returns usage alongside evaluation errors, so the optimiser ledger keeps tokens spent before a failure.
- The WASM compile model resets its usage on every call path.
- Crucible reports a subject's error before checking output size, so a partial output no longer hides it.

## v0.6.0 and earlier

- Standalone Go module at `github.com/mikills/gepago`.
- Provider-neutral GEPA-style optimiser and programme layer.
- Optional `agents` runtime with event traces and trace-backed reflection datasets.
- Optional OpenAI/OpenAI-compatible and Claude provider packages.
- Built-in exact-match, contains, numeric, classification, and LLM judge metrics.
- `Predict`, `ChainOfThought`, demos, JSON repair, programme compilation, persistence, and HTML reports.
- Sentiment and deterministic examples.
