# Gepago WASM host API

This command exposes one global `gepagoWasm` namespace. Its `version` is the
JSON protocol version (`"2"`). The existing `invoke(operation, payloadJSON)`
API returns a JSON envelope:

```json
{"ok":true,"value":...}
{"ok":false,"error":"..."}
```

The deterministic operations remain synchronous: `version`,
`parsePrediction`, `decodeDemos`, `encodeDemos`, `metric`, `buildPrompt`, and
`htmlReport`. The resumable `predict.start`/`predict.resume`/`predict.cancel` operations are
unchanged; `predict.start` accepts the `predict` and `chainOfThought` program kinds.

## Async Compile

`gepagoWasm.start(operation, payloadJSON, modelFn, onComplete)` supports
`operation === "compile"` and `operation === "pipeline.run"` (below). It returns
a module-owned `job_id` immediately. `onComplete(job_id, resultJSON)` is always
called from a later JavaScript task, even for a job that fails before its first
model call, so the caller knows the `job_id` by then. `resultJSON` is the usual
envelope:

```json
{"ok":true,"value":{"candidate":{},"state":{}}}
{"ok":false,"error":"..."}
```

`modelFn` receives a JSON request string and may return a string, an object of
shape `{text, usage?}`, or a Promise of either. Requests have this shape:

```json
{"kind":"language_model.generate","role":"task","prompt":"...","attempt":1,
 "response_format":{"name":"sentiment","schema":{"type":"object"},"strict":true}}
```

`response_format` is present on `task` and `repair` requests; a host with
structured output should constrain the reply to `schema` when `strict` is true.

`role` is `task`, `reflector`, `judge`, or `repair`. The host model is used for
all four roles; the role lets the host route them independently. Token usage
from `{prompt_tokens, completion_tokens, total_tokens}` is passed into
Gepago's usage reporters and optimizer ledger. `llmJudge` metrics use the
`judge` role.

The compile payload is:

```json
{
  "program": {"kind": "predict", "signature": {"inputs": [], "outputs": []}, "instruction": "...", "demos": [], "maxParseRetries": 1},
  "trainset": [{"id": "example-1", "input": {}, "expected": {}}],
  "valset": [{"id": "validation-1", "input": {}, "expected": {}}],
  "metric": {"kind": "exactMatch", "fields": ["label"]},
  "objective": "optional reflection objective",
  "components": ["instruction"],
  "maxMetricCalls": 10,
  "minibatchSize": 1,
  "seed": 1,
  "acceptance": "strict"
}
```

`program.kind` defaults to `predict`. `chainOfThought` accepts the same prompt
fields plus optional `reasoningField`; Compile delegates its signature and
instruction expansion to `gepa.ChainOfThought.ToPredict()`. A `pipeline` program
uses named steps whose nested programs are `predict` or `chainOfThought`:

```json
{"kind":"pipeline","steps":[
  {"name":"extract","program":{"kind":"predict","signature":{"inputs":[{"name":"text"}],"outputs":[{"name":"topic"}]},"inputKeys":["text"],"outputPrefix":"extract."}},
  {"name":"decide","program":{"kind":"chainOfThought","signature":{"inputs":[{"name":"extract.topic"}],"outputs":[{"name":"label"}]},"inputKeys":["extract.topic"]}}
],"returnAll":true}
```

Compile runs a pipeline program as a typed `gepa.Pipeline`. Each step reads its
`inputKeys`, or its signature's inputs when it lists none, and writes its
signature's outputs under its `outputPrefix`. A key takes the Go type of the
`Field.Type` that first declares it (the first step's output that writes it, or,
for a pipeline input, the input field of the first step that reads it), and
`any` when that field sets no type, so a typed pipeline input must convert to
its type. An absent or null optional output holds its type's zero value. A key
written by several steps keeps the latest value; a step that writes a pipeline
input, or steps that write one key as different types, fail validation. The
result is the last step's outputs, or with `returnAll` the inputs overlaid with
every step's outputs. Pipeline candidates are scoped (`extract.instruction`,
`decide.instruction`, etc.).

`start("pipeline.run", payloadJSON, modelFn, onComplete)` runs a compiled
pipeline program as compile runs it, with
`{"program": {...}, "candidate": {...}, "inputs": {...}}`; a component the
candidate leaves out keeps its seed text. Its model requests have the
`predict.start` host request shape (`request_id`, role `task`, attempt 1 for a
step's first call and one more for each repair), and it completes with
`{"prediction": {...}, "usage": {...}}`.

Supported metrics are `exactMatch`, `contains`, `numericMatch`,
`classification`, and `llmJudge`. `acceptance` is `strict` (the default) or
`equal`. Omitted `minibatchSize` follows Gepago's default (`3` for trainsets of
three or more, otherwise the trainset size); omitted `maxMetricCalls` follows
`DefaultCompileConfig`: `validation_size + 8 * (2 * minibatch_size + validation_size)`.
Examples are converted with
Gepago's structured `NewIOExample` API.

`gepagoWasm.cancel(job_id)` requests cancellation and returns `true`; the job
then completes with an error envelope when its goroutine observes cancellation.
The bridge waits for a Promise only on a worker goroutine, so the JavaScript
event loop is not blocked by model calls. An arbitrary Promise cannot be
force-cancelled: if a host callback itself blocks synchronously or never
settles, cleanup is necessarily best effort. A settled Promise releases its Go
callbacks, and completed jobs are removed from the module-owned job map.

## Custom optimizer callbacks

`gepagoWasm.startCapabilities(operation, payloadJSON, callbacks, onComplete)`
is an additive low-level API for JSON-safe host implementations. It returns a
job ID immediately and uses the same completion callback and envelope as
`start`. Existing `start` and `startAgent` APIs are unchanged. Each callback is
called with one JSON string and may return a JSON-compatible value or a Promise
of one. Callback errors, rejected Promises, non-JSON values, and invalid result
shapes fail the operation. Callback functions run on the host event loop; the
Go bridge waits only on its worker goroutine.

For `operation === "compile"`, use the existing compile payload but set
`metric.kind` to `"host"` and provide:

```js
{
  model: (requestJSON) => ({text: "...", usage: {prompt_tokens: 1, completion_tokens: 1, total_tokens: 2}}),
  metric: (requestJSON) => ({score: 1, feedback: "optional", objective_scores: {}, details: {}})
}
```

`model` has the same request and result protocol documented above. `metric`
receives:

```json
{"kind":"metric.score","expected":{},"actual":{}}
```

and returns a `MetricResult` object with finite numeric `score` and optional
`objective_scores`, `feedback`, and JSON-safe `details`.

For `operation === "optimize"`, the payload is a lower-level optimizer config:

```json
{
  "seedCandidate": {"instruction":"initial text"},
  "trainset": [{"id":"train-1","input":{"text":"hello"}}],
  "valset": [{"id":"val-1","input":{"text":"goodbye"}}],
  "components": ["instruction"],
  "maxMetricCalls": 20,
  "minibatchSize": 2,
  "seed": 7,
  "acceptance": "strict",
  "enableMerge": false,
  "resume": false
}
```

`evaluator` and `proposer` are required; `mergeProposer` is required when
`enableMerge` is true. Their exact request/result shapes are:

```json
{"kind":"optimizer.evaluate","candidate":{},"examples":[{"id":"x","input":{}}],"capture_traces":true}
{"items":[{"example_id":"x","output":{},"score":1,"objective_scores":{},"side_info":{},"trace":{}}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}

{"kind":"optimizer.propose","candidate":{},"dataset":{"instruction":[]},"components":["instruction"]}
{"instruction":"replacement text"}

{"kind":"optimizer.merge","left":{"id":"a","candidate":{}},"right":{"id":"b","candidate":{}},"components":["instruction"]}
{"instruction":"merged text"}
```

`observer` receives `{"kind":"optimizer.observe","event":{...}}` and its
result is ignored. Optional persistence callbacks receive
`optimizer.persistence.append_event` with `{event}`, `optimizer.persistence.save_state`
with `{state}`, and `optimizer.persistence.load_state` with no extra fields;
the load callback must return an `OptimizationState` object, or `null` when no
state has been saved yet. If persistence is
configured, all three callbacks (`appendEvent`, `saveState`, `loadState`) are
required. An optimize completion value is `{ "state": ... }`; compile retains
its existing `{ "candidate": ..., "state": ... }` value.

The command-level API intentionally does not expose Go interfaces or
filesystem access. The npm wrapper is unchanged in this extension; consumers
using it would need to call the low-level namespace through a wrapper-specific
binding or add equivalent typed helpers.

## Async Agent

`gepagoWasm.startAgent(payloadJSON, chatFn, toolFn, onComplete)` starts the
existing `agents.Agent` loop and returns a `job_id`. It is intentionally a
separate API from compile's model callback API. Completion is
`onComplete(job_id, resultJSON)`, with the normal envelope:

```json
{"ok":true,"value":{"final":{},"transcript":[],"usage":{},"ledger":{},"spans":[],"stop_reason":"complete","turns":1,"tool_calls":[]}}
{"ok":false,"error":"agent callback failed ..."}
```

The payload is declarative and contains no provider or Go callback fields:

```json
{
  "name": "assistant",
  "model": "host-model",
  "systemPrompt": "Be concise.",
  "maxTurns": 8,
  "temperature": 0.2,
  "maxTokens": 512,
  "messages": [{"role":"user","content":"Hello"}],
  "tools": [{"name":"lookup","description":"Look up a value.","parameters":{"type":"object"}}],
  "run_id": "run-123",
  "memory": {"tenant":"demo"}
}
```

`initial_messages` is accepted as an alias for `messages`. `run_id` and at
least one initial message are required. `maxTurns` and `maxTokens` cannot be
negative; zero `maxTurns` uses the Agent's default. Tool definitions require a
name and description and cannot be duplicated.

`chatFn` receives one JSON string containing `agents.ChatRequest` and returns a
JSON string containing `agents.ChatResponse` (a JSON object is also accepted),
or a Promise of either. The response message must have role `assistant`.
Usage token counts must be non-negative; an omitted/zero `total_tokens` is
normalized to prompt plus completion tokens, and an inconsistent smaller total
is rejected.

`toolFn` receives one JSON string with this shape:

```json
{"name":"lookup","call_id":"call-1","arguments":"{}","run_id":"run-123","memory":{"tenant":"demo"}}
```

It returns a string tool output, an object `{ "output": "...", "memory": {} }`,
or a Promise of either. Returned memory is replaced only after it has been
validated as a JSON object. A model-requested unknown tool and all malformed,
rejected, or throwing callbacks fail the run with a clear error envelope.
Cancellation is propagated to the Go Agent context and causes completion with
a cancellation error; host Promises themselves cannot be cancelled.

This WASM command keeps provider credentials, filesystem objects, and arbitrary
Go interfaces outside JSON. Host callbacks provide equivalent extension points:
custom metrics, evaluators, reflective dataset builders, proposers, merge
proposers, observers, and persistence are available through the async
`startCapabilities` bridge above. Node and browser adapters own providers and
storage while only JSON-safe values cross the WASM boundary.

Build with:

```sh
GOOS=js GOARCH=wasm go build -o gepago.wasm ./cmd/wasm/gepago
```
