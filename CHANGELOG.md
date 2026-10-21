# Changelog

All notable changes to `gepago` will be documented in this file.

The project follows semantic versioning once public tags are published.

## v0.9.0

`Pipeline` can run a real extraction flow: one transition engine with forward skips and bounded rewinds replaces the control-flow combinators, `Map` fans out over lists known only at run time, tool calls carry the caller's scope and usage billed elsewhere, and state decodes with `encoding/json/v2` on Go 1.27.

### Breaking changes

#### Toolchain and decoding

- Both modules require Go 1.27.1, and `GOEXPERIMENT=nojsonv2` is not supported. Upgrade Go.
- State values, `TypedPredict` replies and `NewTool` arguments decode with `encoding/json/v2`, which refuses input v0.8.0 accepted: duplicate object names in a raw JSON (`json.RawMessage`) state value (`TypedPredict` replies and `NewTool` arguments still keep the last), times that are not strict RFC 3339 (such as `,` before the fraction or a `+24:00` offset), base64 with line breaks, lone UTF-16 surrogate escapes in `,string` fields, invalid UTF-8 or lone surrogates inside a `json.RawMessage`, and map keys that decode to the same key (`"1"` and `"01"`). Fix such data where it is produced.
- A struct whose definition v2 refuses fails every decode, even of an object without the faulty field, and `JSONSchemaFor`, `NewTool` and `TypedPredict` (given `Inputs` included, at any depth) refuse it up front: an unexported field with a json tag, two fields with one JSON name, an unnamed embedded non-struct, an embedded struct with tag options or marshal methods, a struct without exported fields, a malformed or repeated tag option, a single-quoted name, `json:"-,"`, any option but `omitempty`, `omitzero` and `string` (`embed`, `inline`, `case:…`, `format:…`, misspellings), and `,string` on anything but a boolean, number or string (or an unnamed pointer to one) or on a type with its own unmarshal method. Remove the tag or option, or rename the field.
- A json tag name that `encoding/json` rejected, such as `temp°C`, is now the field's JSON name for decoding and encoding, as the schema already said; v0.8.0 refused such members and wrote the Go field name. Rename the tag if you relied on the Go name.
- A type with `UnmarshalJSONFrom` decodes through it and gets schema `{}`, so a struct embedding `json.Number` decodes from a bare number and refuses `{"Number":"5"}`. A `,string` `json.Number` refuses text that is not a JSON number (`"0x10"`, `"01"`, `"1."`, `"-Inf"`, `"5 "`). Send the JSON form.
- Decoding error text changed: errors from custom unmarshalers and times are wrapped as `json: cannot unmarshal … of type T: <err>`, paths name the innermost struct and the JSON field names, `,string`, `TextUnmarshaler`, `Duration` overflow and `*T` messages read differently, and with several faults the one reported can differ (a type error before a renamed field now reports the missing field). A stored Go map with float keys now converts, and bool, pointer and struct keys fail with another error. Metric text for map keys that encode alike is `{"a":1,"a#2":2}`. Match errors with `errors.Is` and `errors.As`, not text.

#### Tools

- `ToolCall` gains `Scope any` (JSON `scope`), so `==` on calls and calls as map keys panic when the scope holds a map, slice or func, and `ToolSession.Call` replaces any scope on a call with the one in its context. Set scopes with `WithToolScope` and compare calls by `ID`, `Name` and `Arguments`.
- A `NewTool` function or an agent `ToolBinding` handler invoked directly, without a session, sees the call's `Scope` rather than the scope in its context. Set `ToolCall.Scope` when you invoke a tool yourself.
- `Provenance` is `{Kind, ID, Source, Detail}` and its JSON `source` is omitted when empty. A result with an entry that fails `Provenance.Validate` (empty, a blank `Source`, `Detail` without `Source`, or only one of `Kind` and `ID`) fails the call. Give each entry a `Source` or a `Kind` and `ID`.
- `RunWithUsage`, `UsageCollector`, `ProgramEvaluator`, and an agent's `RunResult.Usage`, `Ledger` and `Event.Usage` include tools' `UnbudgetedUsage`; the agent run span's `Usage` is budgeted only. Read `Unbudgeted` on spans, events and traces for the split.
- The Google chat client refuses, before sending it and with a Permanent error, a request with a role other than gepa's four (empty included), tool parameters that are not a JSON object, or, without tools, a response-format schema that is set but not a JSON object. `gepatest.FakeGenerator.Chat` refuses only what no provider accepts, with `ErrInvalidChatRequest`: a role that is set but not one of gepa's four, tool parameters that are not a JSON object, and, without tools, a strict response format whose schema is set but not valid JSON. Send gepa's roles and object schemas.

#### Pipelines and control flow

- `StepPolicy.OnError`, `ErrorAction`, `OnErrorFail`, `OnErrorSkip` and `OnErrorFallback` are removed: a step that still fails after its last attempt fails the run. For `OnErrorSkip`, add an `OnError` `Transition` from the step to the next one, with a `Default` for each write a later step reads. `Fallback` replaces `OnErrorFallback` but does not fall back on a Permanent error, which `OnErrorFallback` did; to fall back on any error, put the secondary step right after the step and add `{From: step, OnError: true, To: secondary}` and `{From: step, To: next}`, where `next` is the step after the secondary or `gepa.End`, with a `Default` for each of their writes a later step reads. A step inside a `Parallel` or a `Map` body has no equivalent, since transitions name top-level steps only: lift it to the top level, or handle the error inside the step.
- `RetryPolicy.RetryIf` is removed: a `StepPolicy` retries Transient errors only. Wrap retryable errors with `TransientError`.
- `KeyDefault` is now `Default`, which `DefaultValue` returns. Defaults seed state before every run's first step, so steps see them and `Run` returns those no write replaced; a `MergeRule` never merges a write with a default; and `Build` refuses only a default that no step reads or writes. Rename the type, and drop defaults you do not want in the output.
- `Branch` and `Fallback` return a `Flow` of top-level steps and transitions, which cannot nest inside `Parallel`, `WithPolicy`, `WithCache` or `Map`. Add it with `spec.Add(gepa.Branch(...))`.
- Arms and a Fallback's two steps are top-level steps: errors, step paths and events name them without the flow's name, their candidate keys lose the `<branch>.` or `<fallback>.` prefix, and a routing step (named after the Branch, writing its chosen label to a string key of that name) and a join step appear in events and traces. Arms and a Fallback's steps may not write, so a later step that reads their writes needs a `Default`. Rename `OnStep` scripts and expectations, and add the defaults.
- A Branch's name may not be a pipeline input or another step's write, and `Build` refuses steps named `$end` or `$fail` and a `PerStepTokens` entry that names a Branch or Fallback. Rename them, and budget the arms or steps instead.
- `Fallback` fails with the secondary step's error only; v0.8.0 joined both errors. Read the primary's error from its event or trace.
- The pipeline fingerprint now covers transitions, `MaxVisits`, amendable keys and each `Map`'s keys and `MapOptions`, so a v0.8.0 artefact no longer loads once its pipeline uses them, as `Branch`, `Fallback` and the migrations of `OnErrorSkip` and `OnErrorFallback` do; a pipeline that uses none keeps its fingerprint. Set the same `ProgramVersion` on the artefact (`LoadProgramArtifact`, then `SaveProgramArtifact`) and pass it to `LoadCompiledProgramVersion` or `ProgramRegistry.LoadCompiledVersion`, or retrain. Retrain for `Branch` and `Fallback` anyway: the old candidate keys of their arms and steps are ignored.
- These structs gain fields, so unkeyed struct literals of them break: `PipelineSpec` (`Transitions`, `MaxVisits`), `ToolCall` (`Scope`), `Provenance` (`Kind`, `ID`), `ToolResult` and `ToolCallRecord` (`UnbudgetedUsage`), `UsageSpan`, `RunFinishedEvent` and `PipelineTrace` (`Unbudgeted`), `StepStartedEvent` (`Attributes`), `StepFinishedEvent` (`Attributes`, `Unbudgeted`), `StepTrace` (`Attributes`, `Transition`, `RewindRejected`, `Unbudgeted`), and gepatest's `Call`, `ExpectedStep` and `ExpectedTrace`. Use keyed literals.

#### Observability

- `StepFinishedEvent.Skipped`, `StepTrace.Skipped` (JSON `skipped`), the run report's skipped badge, the `skipped` key of `TraceDatasetBuilder` records and `gepatest.ExpectedStep.Skipped` are removed. A step a transition skips emits no events; read `StepTrace.Transition` on the step before it.
- Event and trace order change for flows (routing and join steps, arms at top level), and `Attempt` counts a step's executions across rewind visits. `gepatest.AssertTrace` compares `Transition`, `RewindRejected` and `Unbudgeted` in full. Update golden traces and expectations.
- Every step execution, in a pipeline with or without transitions, carries `Attributes` (`{"visit": 1}` on a first visit, plus `index` in a Map element), so every `StepTrace` JSON, golden trace and `gepatest` `Calls()` record changes. Update golden files, and compare `Calls()` by field or ignore `Attributes`; `AssertTrace` compares only the attributes an expectation lists.

### Added

- Transitions: `Transition{From, OnError, When, To, Amend, Accept}` in `PipelineSpec.Transitions`, with `End`, `Fail`, `PipelineSpec.MaxVisits` and `ErrMaxVisits`. After a top-level step finishes, the first matching transition skips forward, ends or fails the run, or rewinds to the same or an earlier step and runs the range again, bounded by the target's `MaxVisits`, one range at a time. `Build` checks them in one pass over the transitions and one over the steps.
- Guaranteed keys: a step a transition skips over, or one with an `OnError` transition forward, may not write, and `Build` refuses a later read of its writes unless the key has a `Default`.
- Rewinds: `WithAmendable(step, keys...)` declares the keys a rewind's `Amend` may set for the target's next visit only, never committed; `Accept(prev, next)` keeps the new visit or restores the old one. Every visit's usage counts.
- `Flow` and `PipelineSpec.Add`; `Branch` and `Fallback` are built from ordinary steps and transitions.
- `Map` and `MapCollect` run a body once per element of a list known only at run time, each on its own view, at most `MapOptions.MaxConcurrency` at a time and refusing more than `MaxItems`. `Map` fails fast and commits nothing; `MapCollect` writes a `MapResult{Index, Value, Error, Usage, Duration}` for every element, unless an element hits a `Budget` error or the run's context ends, which fail the step. Elements share the run's budget and tool scope.
- Events: `Attributes` on `StepStartedEvent`, `StepFinishedEvent` and `StepTrace` (`visit`, a Map element's `index`, a Map's `failed`), `StepAttributesFrom(ctx)`, and `TransitionObserver` with `TransitionTakenEvent` and `RewindRejectedEvent`, which `RecordingObserver` records as `StepTrace.Transition` (`TransitionTrace`) and `StepTrace.RewindRejected`. The run report shows visits, elements, transitions and rejected rewinds.
- Tool scope: `WithToolScope(ctx, scope)` sets an opaque scope, such as a tenant, on every tool call of a run; a tool reads it with `ToolCall.ScopeAs[T]()`, or with `ToolScopeFrom(ctx)` in a `NewTool` function or an agent `ToolBinding` handler. gepago never inspects or marshals it in-process; it must be JSON-marshalable when calls cross a process boundary.
- Unbudgeted usage: `ToolResult.UnbudgetedUsage` and `RecordUnbudgetedUsage` record usage a run reports but no budget counts. `UsageSpan`, `StepFinishedEvent`, `RunFinishedEvent`, `StepTrace`, `PipelineTrace` and `ToolCallRecord` carry it apart from `Usage`.
- `Provenance.Kind` and `ID` name a protected resource for audit, beside `Source` and `Detail`, checked by `Provenance.Validate`.
- Test kit: `gepatest.CheckToolProvider(t, provider, options...)`, with `ToolArguments(...)` and `ToolScope(...)`, and `CheckChatClient(t, client)` drive an implementation through invalid input, a panic, usage on failure, an ended context, a cache hit and concurrent calls, so you can prove your adapters. `FakeGenerator.OnVisit` and `OnElement` script one visit or one Map element (`OnElement` does not match the calls of a pipeline an element runs; script those by step path), calls record their `Attributes`, and `ErrInvalidChatRequest` marks a chat request no provider accepts. `AssertTrace` checks attributes and matches Map elements by index, giving the steps of a pipeline an element runs that element's index.
- Example `examples/map_reduce`: a deterministic pipeline with two forward skips, `MapCollect`, a `TypedPredict` reducer that rewinds once with an amendment, and a tool that reads the scope.

### Dependencies

- The Makefile pins modernize v0.52.0 and staticcheck v0.8.2 (golines stays at v0.13.0) and runs every Go tool and `gofmt` on the toolchain `go.mod` selects. CI uses Go 1.27.x.

## v0.8.0

`Pipeline` becomes the unit of a production flow: typed steps, bounded model calls, tools, per-step policy and a trace for every run.

### Breaking changes

#### Model calls

- Every model consumer takes a `Generator`, whose `Generate` returns a `Generation{Text, Usage, Model, FinishReason}`: `Predict` and `ChainOfThought` `LM` and `RepairLM`, `LLMJudgeMetric.LM`, `ReflectiveProposer.LM`, `LLMMergeProposer.LM`, `ReflectionLM` on `CompileConfig`, `OptimizationConfig` and `ProgramTrainConfig`, `DefaultCompileConfig` and `RetryingLanguageModel.LM`. Wrap a `LanguageModel` with `AdaptLanguageModel`.
- Usage travels in the `Generation` instead of being read back after the call: `RunWithUsage`, `ProgramEvaluator` and crucible's `ProgramSubject` no longer fall back to `LastUsage`, so a custom programme reports its usage with `RecordUsage`.
- `RecordUsage` also charges the run budget in the context; inside a budgeted pipeline, `Predict`, `LLMJudgeMetric`, the proposers and `RetryingLanguageModel` can return a `Budget` error before calling the model.
- `(*RetryingLanguageModel).Generate` returns `(Generation, error)`, so `*RetryingLanguageModel` no longer satisfies `LanguageModel`, and `RetryingLanguageModel` is no longer comparable (its `Jitter` field is a func). It retries only transient errors, so an error without a kind (see `TransientError`) is no longer retried, and waits with capped exponential backoff and jitter from `Delay`. It implements `ChatClient`.
- `Predict` fails with `ErrOutputTruncated` on `FinishLength` instead of running the repair prompt; `ReflectiveProposer`, `LLMMergeProposer`, `LLMJudgeMetric` and crucible's judges fail with it too. Truncation errors end with ", N completion tokens".
- A refusal is a `Generation` with `FinishContentFilter` and no error (OpenAI's text reads "model refused: ..."); `Predict`, the proposers and the judges fail on it with `ErrContentFiltered`, which matches `ErrInvalidOutput`, and never repair it.
- Missing or unknown finish reasons are `""`. `agents.FinishReason` is an alias of `gepa.FinishReason`, and a content-filter stop ends an agent with `StopContentFilter`.
- `UsageLedger.AddUsage` adds the reasoning, cached-input and cache-creation tokens a `Usage` carries, so the ledger's `reasoning_tokens`, `cached_input_tokens` and `cache_creation_tokens`, always zero before, now hold what providers report; its `ModelCalls` counts failed proposals and merge calls.
- Resuming a saved optimisation fails the config fingerprint check when the reflection model's Go type changed; a model wrapped by `AdaptLanguageModel` keeps its v0.7 fingerprint.

#### Structured output

- `Field.Required` is removed: outputs are required by default and `Field.Optional` opts out. In a pipeline, an absent or null `Optional` output commits the zero value of the key's type.
- `Field` gains a `json.RawMessage` `Schema`, so it is no longer comparable: `==` on fields and fields as map keys stop compiling.
- `Predict` rejects missing or null required outputs, extra fields and values that do not match `Type` as `ErrInvalidOutput`, and returns the invalid reply with its error; in a pipeline the reply is in `StepError.Details["output"]`. Missing and null outputs read `signature "x": output "b" is missing`.
- `ChainOfThought` replies must include the reasoning field, which comes first in the output schema (Go and TypeScript).
- `RepairPrompt(prompt, raw, err)` takes the original prompt and repeats it, and its wording changed to include the schema. Prompts for typed signatures gain a schema block.
- OpenAI and Claude send native structured output for strict schemas, so a model or endpoint without support rejects those requests; `gepa.GeneratorFunc(lm.Generate)` sends the schema in the prompt only.
- `Key.Get` rejects wrong-length arrays, floats that underflow, a quoted `"null"` and quoted non-JSON numbers such as `"01"` and `"1."` in `,string` fields.
- Metric text for nil is `null` (was `<nil>`), and non-string values compare by canonical JSON, which changes `ClassificationMetric` details.
- `Build` checks a signature step's reads and writes against its signature's names and Go types (`SignatureProgram`).

#### Pipelines and evaluation

- These structs gain fields, so unkeyed struct literals of them break: `Usage` (`ReasoningTokens`, `CachedInputTokens`, `CacheCreationTokens`), `Field` (`Type`, `Schema`, `Optional`), `Predict` (`Tools`, `MaxToolTurns`, `Middleware`, `ToolRegistry`, `ToolPolicy`), `PipelineSpec` (`Defaults`, `Budget`, `Tools`, `ToolPolicy`, `Observers`, `Redactor`), `CompileConfig` and `ProgramTrainConfig` (`DatasetBuilder`), `ProgramEvaluator` (`RecordTraces`), `StepError` (`Reads`), `ProgramArtifact` (`Fingerprint`), `agents.Agent` (`Middleware`, `ToolNames`, `ToolPolicy`, `ToolRegistry`), `agents.ToolCallRecord` (`Cached`), and the provider `Config`s (`Logger`, `Client` on OpenAI's and `OpenTelemetry` on Claude's).
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
- OpenAI no longer follows a redirect to another origin: openai-go refuses it with "request URL origin must match the configured base URL", which the provider classifies as permanent.

#### Artefacts

- Every loader returns a `*FingerprintMismatchError` when a v0.8.0 artefact's fingerprint differs from the programme's or the programme has none, and `LoadOrTrain` always checks it. `TrainAndSave` writes `fingerprint` into the artefact JSON. Artefacts from v0.7.0 and earlier have none and load as before.

#### WASM and TypeScript

- `compile` runs a `pipeline` programme as a typed `gepa.Pipeline`; its JSON shape is unchanged. A step without `inputKeys` sees only its signature's inputs (it saw every input and earlier output); a pipeline input whose first reading field sets `type` must convert to it; every input a step reads must be present (a missing signature input used to be left out of the prompt); an absent or null optional output holds its type's zero value; and validation fails for a step that writes a key it or an earlier step reads from the inputs (for example `inputKeys: ["text"]` with an output `text`: `writes pipeline input "text", which is read-only`) and for steps that write one key as different types.
- `runCompiled` runs a pipeline in the Go module as the same `gepa.Pipeline` that `compile` evaluates (through the new `pipeline.run` job), so it accepts and rejects the same programmes and inputs and sends the same prompts, typed inputs converted (a `float32` input of 16777217 prompts as 16777216). Its errors are the pipeline's (a missing input reads `pipeline input "x" is missing`), its model requests have the `predict` host request shape with role `task` and attempts counted per step, and a candidate component it leaves out keeps its seed text.
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
- `LastUsage` and `LastResolvedModel` on the providers, `Predict.LastUsage`, `ChainOfThought.LastUsage`, `ProgramLastUsage`, `Pipeline.LastUsage`, `PipelineProgram.LastUsage` and `RetryingLanguageModel.LastUsage`, so `Predict`, `ChainOfThought`, `Pipeline` and the providers no longer implement `UsageReporter`.

### Added

- `Generator`, `Generation`, `GeneratorFunc` and `AdaptLanguageModel`; every provider implements `Generator` natively. `Usage` gains `ReasoningTokens`, `CachedInputTokens` and `CacheCreationTokens`, mapped by every provider and carried by `UsageLedger.AddUsage`. For custom programmes, metrics and proposers that call a model themselves: `GenerateWithFormat`, `FinishReasonError`, `AdmitModelCall`, `RecordUsage` and `ModelCallReporter`.
- `FinishReason` (`stop`, `length`, `tool_calls`, `content_filter`), `ErrOutputTruncated` and `ErrContentFiltered`.
- `ErrorKind`, with the kinds `Permanent`, `Transient` and `Budget`, and `IsPermanent`, `IsTransient`, `IsBudget`, `ErrorKindOf`, `PermanentError`, `TransientError`, `BudgetError` and `WithErrorKind`; every provider classifies its errors.
- Structured output: `Field.Type`, `Field.Schema`, `Field.Optional` and `Field.GoType`, `Signature.JSONSchema` and `Signature.ResponseFormat`, `ResponseFormat`, `JSONSchemaFor` and `StrictJSONSchema`, the optional `StructuredGenerator` provider interface (OpenAI and Claude), `Predict.ParseOutput` and a check of every reply before `Predict` returns it, `TypedPredict[In, Out]`, which decodes with the rules `Key.Get` uses, and `SignatureProgram` and `ProgramSignature`, which let `Build` check a programme step against its signature.
- Pipeline control flow, checked at `Build`: `WithPolicy` and `StepPolicy` (per-attempt `Timeout`, `RetryPolicy`, and an `ErrorAction`: `OnErrorFail`, `OnErrorSkip` with `PipelineSpec.Defaults`, `KeyDefault` and `DefaultValue`, or `OnErrorFallback`), `Branch`, `Parallel`, `Fallback` and `WithCache`.
- `RunBudget` on `PipelineSpec` (`MaxDuration`, `MaxModelCalls`, `MaxTokens`, `PerStepTokens`), with `BudgetFrom(ctx)`, whose `BudgetMeter` reports what a step has left of each limit as a `BudgetRemaining`. Exceeding a limit fails with a `Budget` error that names it, and a step that finishes after a budget or timeout ended its context fails with its writes dropped.
- Tools: `ToolRegistry` (`NewToolRegistry`, `Register`, `Lookup`, `Schemas`), `ToolProvider`, `NewTool[Args, Out]` with a schema derived from `Args`, `NewToolFunc`, `ToolResult` (output, text, memory writes applied only on success, provenance, usage), `ToolPolicy` (allow-list, total and per-tool call limits, a cache keyed on name and canonical arguments), `ToolSession` with `NewToolSession`, and `ToolMemory` with `NewToolMemory`, `WithToolMemory`, which gives a run the memory to seed or read back, and `ToolMemoryFrom`. In a pipeline, `PipelineSpec.Tools`, `ToolStep` and `Predict.Tools` with `MaxToolTurns` call through the run's one tool session, and a step's tool memory writes and cache entries commit with its state writes. `Provenance`, the errors `ErrUnknownTool`, `ErrToolNotAllowed`, `ErrToolCallLimit`, `ErrToolTurnLimit` and `ErrInvalidToolArguments`, named `ChatMiddleware`, `ChatClientFunc`, and `ChatClient` on the OpenAI, Claude and Google providers (pi's print mode cannot return tool calls).
- Observability: `PipelineObserver` with `RunStarted`, `StepStarted`, `StepFinished` (duration, usage, error, reads, writes, `ToolCalls`, cached, attempt) and `RunFinished`, built on `UsageSpan` with a run trace and a span per step (span kinds `UsageSpanPipelineRun` and `UsageSpanPipelineStep`); a panicking observer is contained. `RecordingObserver` records a serialisable `PipelineTrace` (with `StepTrace.ToolCalls`), `TraceDatasetBuilder` turns traces into reflection records, and `WriteHTMLRunReport` and `HTMLRunReport` render one run step by step, showing JSON indented up to 32 levels and 64 KiB and compact beyond.
- `Redactor` on `PipelineSpec`: every value an event, trace, run report or reflection record carries passes through it by state key, tool calls under `tool:<name>` and run attributes under `attribute:<name>`, and the text of each redacted value is removed from error messages and details, replaced by the redacted value when that is a string and otherwise by `RedactedText`. A Redactor that panics is recovered and the value becomes `RedactionFailed`. Scrubbed JSON keeps its key order and duplicate keys, and a number whose text changes becomes a string. A nested run's values pass through every enclosing Redactor too, so its observers see outer-redacted values, and spans whose `ParentID` belongs to an enclosing run they do not observe. A run with a Redactor and no observers still redacts every value, for the runs nested in it. With a Redactor, `ProgramEvaluator` redacts each item's `Output` and `SideInfo`, `reads` included, while the metric scores the real values.
- Run context: `RunContext{RunID, ParentRunID, TraceID, Attributes}`, `WithRunContext` and `RunContextFrom`. A run keeps the caller's trace ID and attributes, takes the caller's run as its parent and gets a fresh `RunID`, so nested pipelines form one trace. Events and `PipelineTrace` carry the `RunContext`, a run's span ID is its `RunID` and its trace ID is the `TraceID`, and the run report has a Run context section. Observer contexts carry neither the run's `RunContext` nor its `ToolMemory`, so an observer's own work is independent of the run. Provider `Config.Logger` writes one record per model call with the run context, a call that panics included.
- Artefacts: `Pipeline.Fingerprint` from step names, keys and types, the `Fingerprinter` interface, `ProgramArtifact.Fingerprint`, `CompiledProgram.Artifact` and `FingerprintMismatchError`. A `ProgramVersion` stated by both the artefact and `LoadCompiledProgramVersion` or `LoadCompiledVersion` stands in for the fingerprint. The TypeScript `ProgramArtifact` has an optional `fingerprint`, which `parseProgramArtifact`, `loadProgramArtifact` and `saveProgramArtifact` keep and reject when it is not a string.
- `DefaultReflectiveDatasetBuilder` names the failing step of an invalid output and carries its `StepError.Details` and the values it read.
- WASM: `gepagoWasm.start` takes the `pipeline.run` operation, which runs a compiled pipeline programme as `compile` runs it; `runCompiled` uses it.
- Test kit: package `gepatest` with `FakeGenerator`, a `Generator`, `StructuredGenerator` and `ChatClient` that answers from reply scripts keyed by step path or name, prompt substring or a default, each reply with its own text or tool calls, usage, finish reason and error, records every call and lists the calls no script answered (`Unmatched`), whose error a step policy's fallback or skip can hide; and `AssertTrace`, which checks a `PipelineTrace`'s step executions (path, attempt, failure, error text or kind, cached and skipped flags, usage, tool calls) and the run's outcome and usage, checks that executions which did not overlap in time ran in the order listed (different steps of a `Parallel` are unordered), and reports a diff. `StepPathFrom(ctx)` returns the path a running step has in traces. `StepTrace` and `PipelineTrace` record `Failed`, set even when the error's message is empty, and `StepTrace` and `StepStartedEvent` name the step of a `Parallel` an execution belongs to in `Lane`, shared by its retries and its policy's fallback.

### Fixed

- A provider call that fails after spending tokens returns that usage with its error, so it is recorded exactly once, also when calls run concurrently; v0.7 read it back through `LastUsage`.
- `Predict` no longer sends a truncated reply through the repair prompt.
- `RetryingLanguageModel` no longer retries permanent errors or a call whose context has ended.
- Google provider error messages no longer include the API key.
- A panicking agent tool no longer crashes the agent run.
- A WASM async job (`compile`, `runAgent`, capability jobs) that fails before its first host call reports its error instead of "completed with an unexpected job id": the module now calls `onComplete` from a later JavaScript task, after `start` has returned the job id.
- In Node, a `compile` model callback that throws or rejects fails that model call with its message, as in the browser, so `compile` rejects with the host's error instead of `WASM job "job-1" cancelled`; only `close()` cancels a pending job. A pipeline's `runCompiled` reports host errors the same way.

### Dependencies

- Go modules move to their newest releases that support Go 1.25: `anthropic-sdk-go` v1.82.0, `openai-go/v3` v3.74.0, `testify` v1.12.1, `golang.org/x/sync` v0.22.0 and, in crucible, `jsonschema/v6` v6.0.3. The Makefile pins modernize v0.49.0.
- The SDKs bring new indirect dependencies into the build: OpenTelemetry (`go.opentelemetry.io/otel`, `otel/metric`, `otel/trace`, `auto/sdk`), `go-logr/logr` and `go-logr/stdr`, `cespare/xxhash/v2`, `pb33f/go-yaml` and `pb33f/ordered-map/v2`, and `coder/websocket`, while `mailru/easyjson`, `wk8/go-ordered-map/v2` and `gopkg.in/yaml.v3` drop out. Test builds swap `gopkg.in/yaml.v3`, `davecgh/go-spew` and `pmezard/go-difflib` for testify's `go.yaml.in/yaml/v3`. The module graph also gains modules no build loads, such as `otel/sdk` and `creack/pty`. The library adds no direct dependency; the Claude provider's tests import the OpenTelemetry API.
- `anthropic-sdk-go` traces every call through the global OpenTelemetry provider by default. The Claude provider turns that off, as in v0.7; `Config.OpenTelemetry` turns it on, and the SDK's content mode then puts prompts and replies in spans, past the `Redactor`.
- Crucible requires `github.com/mikills/gepago` v0.8.0.
- The npm package `@mikills/wasm` is 0.2.0 and builds with TypeScript 7.
- The WASM API version is `"3"`, and the JS loaders reject a module that reports another version: a module and JS wrapper must come from the same release.

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
