# gepago

`gepago` is a Go SDK for GEPA-style reflective optimisation and for running the flows it optimises.
It helps you improve prompts, policies, tool descriptions, and other text components
using evaluation feedback, and runs them as typed pipelines with bounded model calls, tools,
per-step policies and a trace for every run.

Original paper:
[GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457).

![GEPAgo architecture](docs/architecture.png)

## What it can do

- Call any model through the provider-neutral `Generator` interface, with usage, the resolved model and a finish reason on every call.
- Ask for structured output: typed signature fields, strict JSON schemas, native structured output where the provider has it, and validation before a reply is returned.
- Build typed `Pipeline`s with retries, timeouts, fallbacks, branches, parallel steps, step caching and run budgets.
- Give model steps tools from a registry, under a per-run policy with call limits and caching.
- Observe every run with events, recorded traces, HTML run reports, a run context and redaction.
- Run optional agent loops from the composable `agents` package, with full trace events for runs, turns, LLM calls, tool calls, errors, and run completion.
- Track usage with spans and ledgers instead of relying only on turn counts.
- Persist traces, run results, optimisation events, and optimiser state to files.
- Optimise named text candidates such as prompts, tool descriptions, routing rules, and policies.
- Load parsed text documents into examples for extraction-focused optimisation.
- Build reflective datasets from evaluation feedback, pipeline traces and agent traces.
- Use built-in exact-match, contains, numeric, classification, and LLM judge metrics.
- Compile a `Predict`, `ChainOfThought`, `TypedPredict` or `Pipeline` from examples, a metric, and a reflection model.
- Select candidates from a Pareto frontier or current-best strategy, accept or reject them with configurable score criteria, and resume optimisation from persisted state.
- Save trained candidates as fingerprinted artefacts and load them at runtime.
- Use optional OpenAI, Claude, Google/Gemini or pi providers.
- Generate self-contained HTML reports for optimisation runs and pipeline runs.

## Package layout

```text
github.com/mikills/gepago                  core optimiser, programme and pipeline layer
github.com/mikills/gepago/agents           optional evented agent runtime
github.com/mikills/gepago/gepatest         test kit: fake models and trace assertions
github.com/mikills/gepago/providers/openai OpenAI and OpenAI-compatible provider
github.com/mikills/gepago/providers/claude Anthropic Claude provider
github.com/mikills/gepago/providers/google Google Gemini provider
github.com/mikills/gepago/providers/pi     pi CLI provider
packages/gepago/wasm                       Node.js ESM WASM packages
```

The root package is intentionally provider-neutral and does not depend on the agent runtime.
Use subpackages only for the integrations you need.

## Core flow

GEPA optimises text, not model weights. You provide a candidate text bundle, evaluation cases, and feedback. The optimiser turns failures into reflection records, asks a model for a patch, evaluates the patch, and keeps useful candidates on a frontier.

```text
Candidate text -> rollout/evaluate -> feedback/traces -> reflection -> patch -> frontier
```

## Models

Every model call goes through `Generator`:

```go
type Generator interface {
    Generate(ctx context.Context, prompt string) (gepa.Generation, error)
}
```

A `Generation` holds the reply `Text`, its `Usage` (prompt, completion, reasoning and cached tokens), the `Model` the provider resolved, and a `FinishReason`: `stop`, `length`, `tool_calls` or `content_filter`. A call that fails after the provider spent tokens returns that usage with the error, so it is recorded exactly once. `Predict` fails with `ErrOutputTruncated` on a truncated reply and `ErrContentFiltered` on a refusal rather than repairing them.

Provider errors carry a kind: `gepa.IsTransient` (429, 5xx and network errors), `gepa.IsPermanent` (other 4xx) or `gepa.IsBudget` (a run budget is spent). `RetryingLanguageModel` retries transient errors only, with capped exponential backoff and jitter, and stops when its context ends.

Use `gepa.GeneratorFunc` to turn a function into a `Generator`, and `gepa.AdaptLanguageModel` to wrap an older `LanguageModel` that returns a string.

## Programme layer

For common prompt-optimisation workflows, define a task signature, a `Predict` programme,
examples, and a metric:

```go
program := gepa.Predict{
    Signature: gepa.Signature{
        Name:    "sentiment",
        Inputs:  []gepa.Field{{Name: "text"}},
        Outputs: []gepa.Field{{Name: "label"}},
    },
    Instruction: "Classify the sentiment.",
    LM:          lm, // any gepa.Generator
}

examples := []gepa.Example{
    gepa.NewIOExample(
        "positive-service",
        gepa.Prediction{"text": "I loved the fast service."},
        gepa.Prediction{"label": "positive"},
    ),
    gepa.NewIOExample(
        "negative-quality",
        gepa.Prediction{"text": "The product broke immediately."},
        gepa.Prediction{"label": "negative"},
    ),
}

split, err := gepa.SplitExamples(examples, gepa.SplitConfig{ValidationSize: 1, Shuffle: true, Seed: 1})
if err != nil {
    // handle error
}

config := gepa.DefaultCompileConfig(
    program,
    split.Train,
    split.Validation,
    gepa.ExactMatchMetric{Fields: []string{"label"}},
    lm,
)
compiled, state, err := gepa.Compile(ctx, config)
```

Use `ChainOfThought` when the programme should expose a concise reasoning field.
Use `DemosFromExamples` and the `demos` component when you want examples optimised with the instruction.
Use `LLMJudgeMetric` when exact labels are too rigid for the task.

## Structured output

Give signature fields a `Type`, a Go kind name such as `"string"`, `"int"`, `"float64"`, `"bool"`, `"slice"` or `"map"`, or an explicit JSON `Schema`, and mark outputs the reply may leave out as `Optional`:

```go
signature := gepa.Signature{
    Name:   "extract_amount",
    Inputs: []gepa.Field{{Name: "document", Type: "string"}},
    Outputs: []gepa.Field{
        {Name: "amount", Type: "int"},
        {Name: "note", Type: "string", Optional: true},
    },
}
```

`Signature.JSONSchema` returns the strict reply schema. Providers that implement `StructuredGenerator` (OpenAI and Claude) receive it as the response format; others get it in the prompt. `Predict` checks every reply before returning it: a missing or null required output, an extra field or a value that does not fit its `Type` fails with `ErrInvalidOutput` naming the field, and the repair prompt carries the schema.

`TypedPredict[In, Out]` works with Go structs and decodes the reply with the strict rules `Key.Get` uses:

```go
type Document struct {
    Document string `json:"document"`
}

type Extraction struct {
    Amount int    `json:"amount"`
    Note   string `json:"note,omitempty"`
}

extract := gepa.TypedPredict[Document, Extraction]{
    Signature:   gepa.Signature{Name: "extract_amount"},
    Instruction: "Extract the obligation amount.",
    LM:          lm,
}
result, err := extract.Run(ctx, Document{Document: "Bond obligation for 125,000."})
```

## Pipelines

For multi-step workflows, build a `Pipeline` from a `PipelineSpec`. Each step that wraps a programme has its own scoped components such as `extract.instruction`, so GEPA can improve the steps independently while `Compile` still sees one programme.

```go
document := gepa.Key[string]{Name: "document"}
amount := gepa.Key[int]{Name: "amount"}
decision := gepa.Key[string]{Name: "decision"}

extractProgram := gepa.Predict{
    Signature: gepa.Signature{
        Name:    "extract_amount",
        Inputs:  []gepa.Field{{Name: "document"}},
        Outputs: []gepa.Field{{Name: "amount", Type: "int"}},
    },
    Instruction: "Extract the obligation amount.",
    LM:          lm,
}

pipeline, err := gepa.PipelineSpec{
    Inputs: []gepa.KeyRef{document.Ref()},
    Steps: []gepa.Step{
        gepa.NewProgramStep("extract", extractProgram, []gepa.KeyRef{document.Ref()}, []gepa.KeyRef{amount.Ref()}),
        gepa.NewStep("decide", []gepa.KeyRef{amount.Ref()}, []gepa.KeyRef{decision.Ref()}, func(_ context.Context, s gepa.State) error {
            value, err := amount.Get(s)
            if err != nil {
                return err
            }
            if value > 100000 {
                decision.Set(s, "needs_review")
            } else {
                decision.Set(s, "approve")
            }
            return nil
        }),
    },
}.Build()
```

Steps declare the keys they read and write as `KeyRef`s, which carry the value type. `Build` checks the contract once:

- every key a step reads is a pipeline input, has a `Default` or is written by an earlier step, with the same type
- a key written by more than one step needs a `MergeRule`, unless transitions keep any run from committing both writes
- pipeline inputs are read-only, and no `Default` may name one

`PipelineSpec.Defaults` seed keys before the first step, so every step sees a default until a committed write replaces it; each run gets its own copy. A `MergeRule` combines writes only: the first write of a key replaces its default rather than merging with it.

At run time each step only sees the keys it declared. Its writes must have the declared type, and they are copied into state only when the step succeeds. Values are copied whenever they cross into or out of state, so a step cannot change an input or an earlier result through a shared map or slice. A run fails when its inputs hold a key not declared in `Inputs`, unless `AllowExtraInputs` is set, and input values are converted to their declared types before the first step runs.

`NewProgramStep` copies only the declared outputs from a programme's reply into state, so a model cannot overwrite a document or record identifier. For `Predict`, `ChainOfThought` and `TypedPredict` the signature's inputs must be exactly the declared reads and its outputs exactly the declared writes, with the Go types the signature gives them; a `ChainOfThought` may also declare its reasoning field, and a `TypedPredict` step may leave out the optional fields of `Out`. A reply that omits a declared output, sets it to null or gives a value that does not fit its type fails the step with `ErrInvalidOutput`, unless the signature marks that output `Optional`, in which case the step stores the zero value of the key's type. During `Compile` such an example scores zero, with the failing step, its reads and the error as reflection feedback, instead of stopping optimisation. Deterministic steps built with `NewStep` take no candidate; only steps that carry components appear in `SeedCandidate`.

`Key.Get` converts model output into `T` strictly: unknown fields, fields that match only by case, missing fields not tagged `omitempty` and nulls in non-nullable positions are errors, not zero values. If `T` has a `Validate() error` method, `Get` calls it too.

`Run` and `RunCandidate`, which `Compile` uses, return the state without the inputs: the values the steps wrote and the defaults no write replaced; `RunState` also returns the inputs. A failed run returns that state as it stood before the failing step, defaults included, together with a `*StepError` naming it and holding the values it read. A step can attach diagnostics with `WithStepDetails`, a panicking step's stack is kept under `Details["stack"]`, and usage from a failed model call is still recorded.

This mirrors the GEPA paper's compound-system formulation `Φ = (M, C, X, Y)`: the `Pipeline` is `Φ`, each step is a language module `Mᵢ`, each wrapped `Predict` carries the prompt and local input/output schema `(πᵢ, Xᵢ, Yᵢ)`, and the steps and their control flow are `C`.

### Control flow, policies and budgets

```go
reads, writes := []gepa.KeyRef{document.Ref()}, []gepa.KeyRef{amount.Ref()}
extract := gepa.WithPolicy(
    gepa.NewProgramStep("extract", extractProgram, reads, writes),
    gepa.StepPolicy{Timeout: 20 * time.Second, Retry: gepa.RetryPolicy{MaxAttempts: 3}},
)

spec := gepa.PipelineSpec{
    Inputs:   []gepa.KeyRef{document.Ref()},
    Defaults: []gepa.Default{gepa.DefaultValue(amount, 0), gepa.DefaultValue(decision, "undecided")},
    Budget:   gepa.RunBudget{MaxDuration: time.Minute, MaxModelCalls: 6, MaxTokens: 20000},
}
spec.Add(
    gepa.Fallback("extracted", extract, gepa.NewProgramStep("extract_small", smallProgram, reads, writes)),
    gepa.Branch("route", route, map[string]gepa.Step{"review": review, "approve": approve}, amount.Ref()),
)
pipeline, err := spec.Build()
```

- `Transitions` move a run from a top-level step to a later one (skipping those between), back to an earlier one (a rewind, bounded by `MaxVisits`, with `Amend` and `Accept`), to `End` or to `Fail`. A step a transition skips over may not write, so a later reader of its keys needs a `Default`, which seeds the key before the first step.
- `Branch` and `Fallback` build a `Flow` of ordinary steps and transitions that `PipelineSpec.Add` appends: `Branch` calls its condition once, writes the chosen label to a string key named after the branch and runs the arm with that label, all arms writing the same keys, and `Fallback` runs a secondary step when the primary fails with a non-permanent error. Budget their arms or steps by name; the steps a flow adds for routing and joining spend nothing.
- `WithPolicy` gives a step a per-attempt `Timeout` and retries of transient errors; a step that still fails fails the run unless an `OnError` transition or a `Fallback` takes the run elsewhere.
- `Parallel` runs steps with disjoint writes concurrently and commits them together, and `WithCache` reuses a step's result within a run when its key function returns a key it has seen.
- `Map(name, items, item, body, results, opts)` runs `body` once per element of a list known only at run time, each element on its own view holding the element under `item`, at most `MaxConcurrency` at a time, and writes the outputs in input order; the first failure cancels the rest and commits nothing. `MapCollect` runs every element and writes a `MapResult` for each, with `Error` set for the ones that failed, so a later step can tell what is missing. Elements share the run's budget and tool scope, and their events carry `Attributes["index"]`.
- `RunBudget` limits a run's duration, model calls, tokens and tokens per step; exceeding a limit fails with a `Budget` error naming it. A step reads what is left with `gepa.BudgetFrom(ctx).Remaining()`.

`Build` checks all of this before the first run: transitions that name unknown steps or overlap as rewinds, an `Accept` on an `OnError` rewind, reads of keys a step may not write without a default, arms that disagree, overlapping parallel writes, a `Map` body that does not read its item, writes other than one key of the results' element type or reads a key that is not guaranteed, and budgets that name unknown steps or a flow's routing and joining steps all fail it.

## Tools

Register tools once and refer to them by name:

```go
type lookupArgs struct {
    Customer string `json:"customer"`
}

lookup := gepa.NewTool("lookup_customer", "Look up a customer's plan.",
    func(ctx context.Context, args lookupArgs) (Plan, error) {
        return store.Plan(ctx, args.Customer)
    })
tools, err := gepa.NewToolRegistry(lookup)
```

`NewTool` derives the argument schema from `Args` and encodes `Out` as the result; `NewToolFunc` takes a hand-written schema. In a pipeline, set `PipelineSpec.Tools` and either call a tool deterministically with `gepa.ToolStep("lookup", "lookup_customer", reads, writes)` or give a `Predict` step a bounded tool loop with `Predict.Tools` (its `LM` must be a `ChatClient`, which the OpenAI, Claude and Google providers are). `PipelineSpec.ToolPolicy` sets the run's allow-list, total and per-tool call limits and result cache, and a tool's memory writes commit only with its step's writes. `Build` fails on a tool that is not registered or not allowed. Chat middleware registered with `RegisterMiddleware` wraps a step's or an agent's chat client by name, outermost first.

## Observability and redaction

`PipelineSpec.Observers` receive `RunStarted`, `StepStarted`, `StepFinished` (duration, usage, error, reads, writes, tool calls, attempt, visit and whether the result was cached) and `RunFinished` events, and a `TransitionObserver` also `TransitionTaken` and `RewindRejected`; an observer that panics does not fail the run. `RecordingObserver` turns the events into serialisable `PipelineTrace`s, which `WriteHTMLRunReport` renders step by step and `TraceDatasetBuilder` turns into reflection records during `Compile`.

```go
recorder := &gepa.RecordingObserver{}
spec.Observers = []gepa.PipelineObserver{recorder}
spec.Redactor = func(key string, value any) any {
    if key == "document" {
        return "[document]"
    }
    return value
}
pipeline, err := spec.Build()
// after a run
err = gepa.WriteHTMLRunReport("run.html", recorder.Traces()[0], gepa.HTMLReportOptions{})
```

A `Redactor` rewrites each value by its state key before it leaves the run in an event, a trace, a report or a reflection record; tool calls pass through it under `tool:<name>` and run attributes under `attribute:<name>`. State keeps the real values, and the text of every redacted value is removed from error messages. A Redactor must be safe for concurrent use. Its plain-string result is used verbatim as the replacement in text, so it must not put other hidden values into it. A pipeline with a Redactor pays the redaction cost on every value, even without observers.

## Run context

Every run has a `RunContext{RunID, ParentRunID, TraceID, Attributes}`. Put one on the context to correlate a run with your own request:

```go
ctx = gepa.WithRunContext(ctx, gepa.RunContext{
    TraceID:    requestID,
    Attributes: map[string]string{"job": jobID},
})
out, err := pipeline.Run(ctx, inputs)
```

Each run keeps the trace ID and attributes, gets a fresh `RunID` and records the caller's run as its parent, so nested pipelines form one trace. Steps, tools and models find it with `gepa.RunContextFrom(ctx)`, events, traces and run reports carry it, and providers given a `Config.Logger` log it with every model call. The library has no tenant model: attributes are plain strings, redacted like other values in events but logged as they are.

## Reusable programme artefacts

Optimisation can happen offline, then runtime applications can load the trained prompt bundle wherever the same programme can be initialised.

```go
artifact := compiled.Artifact("internal-program")
err := gepa.SaveProgramArtifact("internal-program.program.json", artifact)
```

`compiled.Artifact` records the programme's fingerprint. `Pipeline.Fingerprint()` is derived from the step names, keys and types, and every loader refuses an artefact whose fingerprint differs from the programme's with a `*FingerprintMismatchError` that shows both, so a renamed key or step never silently picks up a stale prompt. Artefacts saved before v0.8.0 have no fingerprint and still load. To load an artefact across a shape change you know is compatible, record the same `ProgramVersion` on the artefact and pass it to `LoadCompiledProgramVersion` or `LoadCompiledVersion`.

At runtime, register factories that inject app dependencies such as models, tools, stores, or retrievers:

```go
registry := gepa.NewProgramRegistry()
err := registry.Register("internal-program", func() (gepa.Program, error) {
    return buildInternalProgram(plannerLM, reporterLM, retriever), nil
})

compiled, artifact, err := registry.LoadCompiled("internal-program.program.json")
result, err := compiled.Run(ctx, input)
_ = artifact
```

Services may also train on demand, then reuse the saved artefact on later calls:

```go
compiled, artifact, state, trained, err := registry.LoadOrTrain(ctx, gepa.ProgramTrainConfig{
    Name:           "internal-program",
    ArtifactPath:   "internal-program.program.json",
    ProgramVersion: "v1",
    Trainset:       examples,
    Metric:         metric,
    ReflectionLM:   reflectorLM,
    MaxMetricCalls: 8,
})
result, err := compiled.Run(ctx, input)
_, _, _ = artifact, state, trained
```

This separates reusable programme shape from trained candidate text. The artefact chooses the programme by name; the registry supplies the runtime dependencies. Use `LoadCompiledVersion` for reviewed production artefacts and `LoadOrTrain` when a service is allowed to bootstrap or refresh a programme at runtime. `LoadOrTrain` always checks the fingerprint; its `ProgramVersion` labels the artefacts it writes.

For service-style deployments, reusable programme families can implement `ProgramTemplate` and be registered with `RegisterTemplate`. GEPAgo also includes helpers for JSONL training data (`LoadJSONLExamples`), artefact indexes (`WriteProgramArtifactManifest`), before/after prompt reports (`WriteCandidateDiffReport`), a minimal HTTP runner (`ProgramHTTPHandler`), and cross-process runtime-training locks (`FileTrainLock`, `LoadOrTrainLocked`).

Lower-level optimiser APIs remain available when you need full control.

## Test kit

Package `gepatest` provides `FakeGenerator`, a model with scripted replies, usage, finish reasons and errors per call, and `AssertTrace`, which compares a recorded run's steps, the order they ran in, their errors and usage.

A call takes the next reply of the script for its step path (as `gepa.StepPathFrom` reports it), else its step name, else the first prompt substring it contains, else the default; a script repeats its last reply once used up. `Calls` returns every call the fake received, `Unmatched` the calls no script answered (a step policy's fallback or skip can hide their error), and `Plain` hides its structured and chat methods to test the prompt-only path.

## Key extension points

- `Generator`: provider-neutral model interface; `StructuredGenerator` and `ChatClient` add response formats and tool calls.
- `Step` and `ComponentStep`: pipeline steps, deterministic or optimisable.
- `ToolProvider`: a tool a registry can hold.
- `PipelineObserver` and `Redactor`: watch runs and control what leaves them.
- `Evaluator`: scores candidates against examples.
- `ReflectiveDatasetBuilder`: turns outputs, feedback, and traces into reflection records.
- `Proposer`: creates candidate patches.
- `CandidateSelector`: chooses the next parent candidate.
- `ComponentSelector`: chooses which text components to update.
- `AcceptanceCriterion`: decides whether a proposal is kept.
- `OptimizationPersistence`: stores and loads optimiser state.

## Visual report

```go
compiled, state, err := gepa.CompileAndReport(ctx, config, "report.html", gepa.HTMLReportOptions{})
```

For lower-level optimiser use:

```go
state, err := optimizer.Run(ctx)
if err != nil {
    // handle error
}
err = gepa.WriteHTMLReport("report.html", state, gepa.HTMLReportOptions{})
```

The report is a single HTML file you can open in a browser. `WriteHTMLRunReport` does the same for one recorded pipeline run.

## Reflection traces

GEPA reflection does not require an agent runtime. Simple prompt or programme optimisation can use scores plus ordinary LLM call metadata:

```text
prompt -> completion -> score -> reflection -> improved candidate
```

Pipelines add step traces: set `DatasetBuilder: gepa.TraceDatasetBuilder{}` on the compile config and each reflection record carries the run's steps, reads, writes and errors.

Use `agents/` when the task itself is a tool-using workflow and you want richer execution traces:

```text
agent run -> turns -> LLM calls -> tool calls -> observations -> score -> reflection
```

So simple tasks may only need LLM call traces and scores. Tool-using workflows benefit from full agent traces.

## Agents

Agent runtime types live outside the core package:

```go
import gepaagents "github.com/mikills/gepago/agents"
```

Use this package when you need tool-calling loops, memory, observers, traces,
or agent-trace-backed reflection. The core optimiser remains independent of the agent runtime.

```go
recorder := gepaagents.NewTraceRecorder()
agent := gepaagents.Agent{
    Name:      "researcher",
    Client:    chatClient,
    Model:     "model-name",
    Observers: []gepaagents.Observer{recorder},
}
```

## Providers

Provider packages live outside the core package:

```go
import (
    gepaclaude "github.com/mikills/gepago/providers/claude"
    gepagoogle "github.com/mikills/gepago/providers/google"
    gepaopenai "github.com/mikills/gepago/providers/openai"
)
```

Each provider's model implements `gepa.Generator`; the OpenAI and Claude models are also a `gepa.StructuredGenerator` and a `gepa.ChatClient`:

```go
lm, err := gepaopenai.NewLanguageModel(gepaopenai.Config{
    APIKey: os.Getenv("OPENAI_API_KEY"),
    Model:  "gpt-4.1-mini",
    Logger: slog.Default(), // optional: one record per call, with the run context
})
generation, err := lm.Generate(ctx, "Say hello.")
fmt.Println(generation.Text, generation.Usage.TotalTokens, generation.FinishReason)
```

OpenAI-compatible endpoints can use `BaseURL` and provider-specific headers:

```go
lm, err := gepaopenai.NewLanguageModel(gepaopenai.Config{
    BaseURL: "https://openai-compatible.example.com/v1",
    Model:   "provider-model",
    Headers: map[string]string{
        "Authorization": "Bearer " + os.Getenv("PROVIDER_API_KEY"),
        "X-Route":       "default",
    },
})
```

An endpoint without structured-output support rejects strict response formats; wrap the model as `gepa.GeneratorFunc(lm.Generate)` to send the schema in the prompt only.

```go
lm, err := gepaclaude.NewLanguageModel(gepaclaude.Config{
    APIKey: os.Getenv("ANTHROPIC_API_KEY"),
    Model:  "claude-sonnet-4-20250514",
})
```

The Anthropic SDK's OpenTelemetry tracing is off. `OpenTelemetry: true` turns it on: a span per call through the global tracer provider and a `traceparent` header, with prompts and replies in the spans when `ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE=content`, which bypasses a pipeline's `Redactor`.

Google/Gemini supports both `Generator` and `ChatClient` for tool-calling agents and steps:

```go
chatClient, err := gepagoogle.NewChatClient(gepagoogle.Config{
    APIKey: os.Getenv("GOOGLE_API_KEY"),
    Model:  "gemini-1.5-flash",
})
```

## Examples

Run the deterministic example without API keys:

```bash
go run ./examples/deterministic
```

Run the sentiment prompt optimisation example with OpenAI:

```bash
OPENAI_API_KEY=... go run ./examples/sentiment
```

Or with Claude:

```bash
GEPA_PROVIDER=claude ANTHROPIC_API_KEY=... go run ./examples/sentiment
```

It writes `sentiment-report.html` in the current directory.

## JavaScript / WASM

The Node.js ESM WASM package is available at [`packages/gepago/wasm`](packages/gepago/wasm).
It loads the Gepago Go engine as WebAssembly and exposes JSON-safe parsing, demo,
metric, and prompt-building helpers:

```bash
cd packages/gepago/wasm
npm ci
cd ../..
make wasm-test
```

Use it from a JavaScript project with:

```bash
npm install @mikills/wasm
```

```js
import { createGepagoWasm } from "@mikills/wasm/gepago";

const gepago = await createGepagoWasm();
try {
  const result = gepago.exactMatch(
    { label: "positive" },
    { label: " POSITIVE " },
    { caseInsensitive: true, trimSpace: true },
  );
  console.log(result.score);
} finally {
  gepago.close();
}
```

The WASM package exposes asynchronous `predict`, `compile`, `optimize`, and
`runAgent` methods. Declarative `predict`, `chainOfThought`, and sequential
`pipeline` programmes are supported; `compile` and `runCompiled` both run a pipeline
as the same typed `Pipeline`, and pipeline candidates are step-scoped.
`compile` routes task, reflector, judge, and repair calls to host callbacks and
supports custom metrics, observers, and persistence callbacks. `optimize`
provides JSON-safe evaluator, proposer, dataset-builder, merge, observer, and
persistence hooks for lower-level workflows. Provider adapters keep credentials
in the host runtime. The Node APIs require Node.js 22.18 or newer; worker and browser
entries provide isolated runtimes.

## Development

Run the full local suite:

```bash
make verify
```

If `diago` is not installed, run the standard checks:

```bash
make fmt
make test
make vet
make staticcheck
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution and provider E2E notes.

## Provider E2E

Provider E2E tests are opt-in so normal `go test ./...` does not spend API credits.

If `OPENAI_API_KEY` is set, run:

```bash
go test -tags=e2e ./providers/openai -run TestOptimizerE2E -count=1 -v
```

The test starts with a weak support-ticket triage instruction, uses OpenAI to propose a stronger one,
evaluates it, and accepts it when the score improves.

If `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY` is set, run:

```bash
go test -tags=e2e ./providers/claude -run TestProposerE2E -count=1 -v
```

## License

MIT. See [LICENSE](LICENSE).
