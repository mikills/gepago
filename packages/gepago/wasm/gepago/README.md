# @mikills/wasm/gepago

ESM bindings for the production Gepago Go `js/wasm` command at
`./cmd/wasm/gepago`. The default entry point is the synchronous Node.js API;
worker entry points provide isolated instances and browser support.

## Requirements

- Node.js 22.18 or newer (native TypeScript support)
- Go installed for local builds

The package is authored in TypeScript and publishes compiled ESM JavaScript,
declarations, generated WASM, and a copied Go `wasm_exec.js`. `npm pack` builds
the artifacts automatically. The default
Node entry point requires Node.js built-ins. Browser consumers should import
`@mikills/wasm/gepago/browser` (or use the package's `browser` export).

## Independent instances

The legacy `createGepagoWasm()` API remains in-process and synchronous for its
small deterministic helpers. The Go `js/wasm` runtime owns process-global
JavaScript compatibility state and its entrypoint owns `globalThis.gepagoWasm`;
it is not safe to load two copies in one JavaScript realm. `close()` on that API
only invalidates the wrapper and cannot stop Go, so it does not make a second
same-process instance safe.

For production isolation, use one dedicated worker per instance:

```js
import { createGepagoWasmWorker } from "@mikills/wasm/gepago";

const [a, b] = await Promise.all([
  createGepagoWasmWorker(),
  createGepagoWasmWorker(),
]);
console.log(await a.version(), await b.version());
a.close();
b.close();
```

The worker facade has the same operation names, but every method is async
because calls cross the worker boundary. Host model/chat/tool callbacks are
also bridged by messages. Worker termination, rather than the legacy `close`,
releases the runtime. The package test suite verifies two concurrent Node
workers and independent predictions.

## Browser

```js
import { createGepagoWasm } from "@mikills/wasm/gepago/browser";

const gepago = await createGepagoWasm({
  // Set these when assets are served outside the package URL.
  wasmURL: "/assets/gepago.wasm",
  wasmExecURL: "/assets/wasm_exec.js",
});
try {
  console.log(await gepago.version());
} finally {
  gepago.close();
}
```

The browser adapter creates a module Worker and loads `wasm_exec.js` and the
WASM inside it, avoiding Node imports and main-thread globals. The assets must
be fetchable under the site's CSP and URL base; browser bundler/hosting
integration and cross-origin worker policy still need an application-level
smoke test. `wasm_exec.js` is the Go runtime shim and must match the generated
WASM's Go toolchain.

## Build and test locally

From the repository root:

```bash
cd packages/gepago/wasm
npm ci
npm run build
npm test
```

The build defaults to `./cmd/wasm/gepago`, uses `GOOS=js GOARCH=wasm`, and
writes `gepago/dist/gepago.wasm`. Generated WASM is ignored by git. To keep the
generated WASM outside the checkout:

```bash
wasm_dir="$(mktemp -d)"
trap 'rm -rf "$wasm_dir"' EXIT
npm run build -- --out-dir "$wasm_dir"
GEPAGO_WASM="$wasm_dir/gepago.wasm" node gepago/test/smoke.ts
```

For a development command at another path, use
`GEPAGO_WASM_SOURCE=/path/to/command npm run build` or pass `--source`.
`wasmExecPath` and `GEPAGO_WASM_EXEC` can select a development copy of
`wasm_exec.js`; otherwise the package copy is preferred, followed by Go's
`GOROOT/lib/wasm/wasm_exec.js`.

## Installation and usage

```bash
npm install @mikills/wasm
```

````js
import { createGepagoWasm } from "@mikills/wasm/gepago";

const gepago = await createGepagoWasm();

try {
  console.log(gepago.version());

  const prediction = gepago.parsePrediction(
    '```json\n{"label":"positive","count":3}\n```',
  );
  const demos = gepago.decodeDemos(
    '[{"inputs":{"text":"great"},"outputs":{"label":"positive"}}]',
  );
  const encoded = gepago.encodeDemos(demos);

  const exact = gepago.exactMatch(
    { label: "positive" },
    { label: " POSITIVE " },
    { trimSpace: true, caseInsensitive: true },
  );
  const contained = gepago.contains(
    { answer: "hello" },
    { answer: "hello world" },
  );
  const numeric = gepago.numericMatch(
    { count: 3 },
    { count: "3.05" },
    { tolerance: 0.1 },
  );
  const classification = gepago.classification(
    { label: "positive" },
    { label: "POSITIVE" },
    { ignoreCase: true },
  );

  const prompt = gepago.buildPrompt({
    signature: { inputs: [{ name: "text" }], outputs: [{ name: "label" }] },
    instruction: "Classify the text.",
    demos,
    inputs: { text: "great service" },
  );

  // The model callback receives a JSON-safe host request. It may return raw
  // text or text plus token usage; the wrapper resumes Go until completion.
  const result = await gepago.predict({
    program: {
      signature: { inputs: [{ name: "text" }], outputs: [{ name: "label" }] },
      instruction: "Classify the text.",
      demos,
      maxParseRetries: 1,
    },
    inputs: { text: "great service" },
  }, async (request) => {
    const response = await fetch("https://model.example/generate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request),
    });
    return await response.text();
  });

  const compiled = await gepago.compile({
    program: {
      signature: { inputs: [{ name: "text" }], outputs: [{ name: "label" }] },
      instruction: "Classify the text.",
    },
    trainset: [
      { id: "positive", input: { text: "great service" }, expected: { label: "positive" } },
    ],
    metric: { kind: "exactMatch", fields: ["label"] },
    components: ["instruction"],
    maxMetricCalls: 1,
  }, async (request) => {
    // Compile uses the same host callback for task, reflector, repair, and
    // optional judge calls. The request is already a JSON-safe object.
    await new Promise((resolve) => setTimeout(resolve, 1));
    return { text: '{"label":"positive"}' };
  });

  // Compile returns a candidate map. Run it with the same wire program;
  // instruction is overlaid and encoded demos are decoded for you.
  const optimizedResult = await gepago.runCompiled(
    {
      signature: { inputs: [{ name: "text" }], outputs: [{ name: "label" }] },
      instruction: "Classify the text.",
    },
    compiled.candidate,
    { text: "great service" },
    async (request) => (await fetch("https://model.example/generate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request),
    })).text(),
  );

  const report = gepago.htmlReport(compiled.state, { title: "Optimization report" });
  console.log({ encoded, exact, contained, numeric, classification, prompt, result, compiled, optimizedResult, report });
} finally {
  gepago.close();
}
````

`buildPrompt` also accepts positional `signature, instruction, demos, inputs`
arguments, plus a convenience positional form with a candidate map containing
`instruction` and encoded `demos`. All calls cross the Go JSON envelope
`{ ok: true, value }` / `{ ok: false, error }`; malformed envelopes and Go
errors are reported as `GepagoWasmError`.

`htmlReport(state, options)` renders a self-contained HTML report from a compile result.

## Provider callbacks

The package includes host-side callback factories matching the Go providers in
`providers/openai`, `providers/claude`, `providers/google`, and `providers/pi`.
Credentials are captured by the factory and are never included in a WASM
config, model request, or agent payload.

```js
import { createGepagoWasm, createOpenAIProvider } from "@mikills/wasm/gepago";

const gepago = await createGepagoWasm();
const provider = createOpenAIProvider({
  apiKey: process.env.OPENAI_API_KEY, // or omit it to use OPENAI_API_KEY
  model: "gpt-4.1-mini",
  // fetch: testFetch, // injectable for deterministic tests
});
const prediction = await gepago.predict(config, provider.model);
const run = await gepago.runAgent(agentConfig, provider.chat, tool);
```

Available factories are `createOpenAIModel`/`Chat`, `createClaudeModel`/`Chat`,
`createGoogleModel`/`Chat`, and the Node-only `createPiModel`/`Chat`. Provider
objects also expose `model`, `chat`, `predict`, `compile`, and `runAgent`.
OpenAI-compatible endpoints use `baseURL` and `headers`; Claude accepts
`ANTHROPIC_API_KEY` or `CLAUDE_API_KEY`, Google accepts `GOOGLE_API_KEY` or
`GEMINI_API_KEY`, and OpenAI accepts `OPENAI_API_KEY`. Set `apiKeyEnv` to use a
custom environment variable. Explicit keys always take precedence.

HTTP factories use an injected `fetch` when supplied, otherwise
`globalThis.fetch`. Import them from `@mikills/wasm/gepago/browser` in browser
code; that entry point contains only fetch-based OpenAI, Claude, and Gemini
adapters and does not import `child_process`. The Pi factory runs the configured
`pi` command through Node `child_process` and is only available from the Node
entry point.

## Async Compile

`compile(config, model, capabilities?)` exposes the production async compile
bridge. The config contains a JSON-safe `program`, `trainset`, optional
`valset`, and one built-in `metric` (`exactMatch`, `contains`, `numericMatch`,
`classification`, or `llmJudge`). Set `metric.kind` to `host` and provide
`capabilities.metric` for a custom metric. `capabilities.observer` and the
`appendEvent`/`saveState`/`loadState` callbacks provide optimizer events and
resumable persistence. The callback receives a normalized object of type
`CompileModelRequest` and may return text, `{ text, usage, finishReason?, model? }`,
or a Promise of either. A `finishReason` of `length` marks a truncated reply and
`content_filter` a refused one: the example scores zero and no repair is requested.
Task and repair requests carry a `response_format` (`{ name, schema, strict }`); the
OpenAI and Claude model helpers enforce it when `strict` is true. Requests are tagged with `task`, `reflector`, `judge`, or `repair`; an
`llmJudge` metric uses the `judge` role. The resolved value is
`{ candidate, state }`, including candidate scores and the optimization usage
ledger.

The wrapper validates inputs at the JSON boundary, converts model strings to
`{ text }`, rejects Go error envelopes, and cancels the module-owned job when a
model callback or completion fails. `close()` also requests cancellation for
pending compile jobs.

`runCompiled(programConfig, candidate, inputs, model)` accepts wire `predict`,
`chainOfThought`, and `pipeline` programs. COT programs add the reasoning output
and instruction once; compiled COT candidates are overlaid after that normalization.
Pipeline candidates use scoped keys such as `extract.instruction`; a component
the candidate leaves out keeps its seed text. A pipeline runs in the Go module
as the same typed `gepa.Pipeline` that `compile` evaluates, so `runCompiled`
accepts and rejects the same programs and inputs and sends the same prompts:
each step sees its `inputKeys`, or only its signature's inputs when it lists
none, typed inputs are converted to their types, its outputs are stored under
its `outputPrefix` (an optional output the reply leaves out holds its type's
zero value), and the pipeline's `returnAll` setting picks the result. Model
requests have the `predict` host request shape.

## Async Agents

`runAgent(config, chat, tool)` runs the Go agent loop with host-provided async
callbacks. The config is JSON-safe and uses the declarative agent shape:
`name`, `model`, optional `systemPrompt`, `maxTurns`, `temperature`, and
`maxTokens`, required `messages` and `run_id`, optional `tools`, and optional
object `memory`. Tools contain `name`, `description`, and optional JSON-schema
`parameters`.

The `chat` callback receives a typed `ChatRequest` object and returns a
`ChatResponse` (or a Promise). The `tool` callback receives a typed request
containing `name`, `call_id`, `arguments`, `run_id`, and the current `memory`.
It may return a string, `{ output, memory? }`, or a Promise. A returned memory
object replaces the run memory after validation. `runAgent` resolves a typed
`RunResult` containing the final message, transcript, usage, ledger, spans, and
tool-call records. Invalid JSON, callback requests/results, rejected callbacks,
and Go error envelopes reject the Promise; callback failures also request job
cancellation. `close()` cancels pending agent and compile jobs.

## Resumable Predict and limitations

`predict(config, model)` implements the version-2 resumable host protocol. The
config is `{ program: { kind?: "predict" | "chainOfThought", signature,
instruction?, demos?, maxParseRetries?, reasoningField? }, inputs }`.
Chain-of-thought normalization adds the reasoning output and instruction before
building the prompt. The async `model` callback receives `{ request_id, kind,
role, prompt, attempt, response_format? }` and returns either a raw string or
`{ text, usage: { prompt_tokens, completion_tokens, total_tokens }, finishReason?, model? }`.
The returned value is `{ prediction, usage }`. A `finishReason` of `length` or
`content_filter` fails the prediction without a repair request. `response_format`
holds the signature's JSON schema; a model with structured output should enforce it
when `strict` is true. If the callback rejects or returns
invalid data, the wrapper cancels the module-owned job and rethrows the model
error. Parse failures request repair until `maxParseRetries` is exhausted.

The package also exposes `optimize(config, capabilities)` for lower-level
JSON-safe custom evaluator, proposer, dataset-builder, merge-proposer,
observer, and persistence callbacks. `compile` accepts an optional capability
object; use `metric: { kind: "host" }` for a custom metric. Persistence
callbacks can map to Node filesystem storage or browser IndexedDB without
exposing filesystem objects to Go:

```js
import { createFilePersistence, withFileTrainLock } from "@mikills/wasm/gepago";
const persistence = createFilePersistence({ directory: "./.gepago/run-1" });
await withFileTrainLock("./.gepago/train.lock", () =>
  gepago.compile(config, model, { ...persistence, observer: async () => {} }),
);
```

Use `createIndexedDBPersistence` from `@mikills/wasm/gepago/browser` in browser
applications. `newProgramArtifact`, `saveProgramArtifact`, and
`loadProgramArtifact` provide the portable candidate bundle equivalent; pass
the loaded candidate to `runCompiled`. Provider adapters keep credentials in
the host language rather than crossing the WASM boundary.

A host Promise cannot be force-cancelled, so a callback that blocks
synchronously or never settles may keep Go-side resources alive; cancellation
is best effort in that case.

The Go WASM entrypoint keeps its runtime alive and owns the process-global
`globalThis.gepagoWasm` namespace. Consequently, the direct Node API supports
one instance per JavaScript realm. Use the worker adapter for independent
instances. A worker is isolated, but it is not free: each instance has a Go
runtime, WASM memory, and worker overhead. Cancellation of an arbitrary host
Promise remains best effort; terminating the worker is the hard isolation
boundary.
