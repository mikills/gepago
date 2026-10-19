import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile as readFileAsync } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { Worker } from "node:worker_threads";
import { connectWorker } from "./worker-client.ts";
import type {
  AgentChat,
  AgentConfig,
  AgentTool,
  AgentToolRequest,
  AsyncGepagoWasm,
  Candidate,
  ChatRequest,
  ChatResponse,
  CompileCapabilities,
  CompileConfig,
  CompileModel,
  CompileResult,
  Demo,
  JSONValue,
  Message,
  MetricResult,
  OptimizeCapabilities,
  OptimizeConfig,
  PipelineStep,
  PredictConfig,
  PredictHostRequest,
  PredictModel,
  PredictResult,
  RunProgram,
  RunResult,
  Signature,
  Tool,
  Usage,
  UsageLedger,
  UsageSpan,
  ToolCallRecord,
  BuildPromptRequest,
  ExactMatchOptions,
  ContainsOptions,
  NumericMatchOptions,
  ClassificationOptions,
  CreateGepagoWasmOptions,
  CreateGepagoWasmWorkerOptions,
  OptimizationState,
  FinishReason,
  StopReason,
  CompileModelRequest,
  CompileModelOutput,
  CompileModelResult,
  PredictModelResult,
} from "./public-types.ts";
import type {
  WireObject,
  RuntimeNamespace,
  LoadedRuntime,
} from "./internal-types.ts";

type RuntimeGlobals = typeof globalThis & {
  Go?: new () => {
    importObject: WebAssembly.Imports;
    run(instance: WebAssembly.Instance): Promise<void>;
  };
  gepagoWasm?: RuntimeNamespace;
  [key: string]: unknown;
};
type RuntimeState = {
  closed: boolean;
  namespace: RuntimeNamespace;
  hadNamespace: boolean;
  previousNamespace: unknown;
  pendingJobs: Set<string>;
};
const globalsObject = globalThis as RuntimeGlobals;

const packageRoot = path.dirname(fileURLToPath(import.meta.url));
const defaultWasmPath = path.join(packageRoot, "gepago.wasm");
const bundledWasmExecPath = path.join(packageRoot, "wasm_exec.js");

let wasmExecConstructor: RuntimeGlobals["Go"];
let wasmExecLoad: Promise<NonNullable<RuntimeGlobals["Go"]>> | undefined;
let wasmExecGeneration = 0;
let loading = false;
let activeRuntime: RuntimeState | undefined;

export class GepagoWasmError extends Error {
  readonly operation: string;
  constructor(message: string, operation: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "GepagoWasmError";
    this.operation = operation;
  }
}

function defaultGOROOT() {
  return execFileSync("go", ["env", "GOROOT"], { encoding: "utf8" }).trim();
}

function developmentWasmExecPath() {
  const goroot = process.env.GOROOT || defaultGOROOT();
  return path.join(goroot, "lib", "wasm", "wasm_exec.js");
}

async function loadWasmExec(
  explicitPath?: string,
): Promise<NonNullable<RuntimeGlobals["Go"]>> {
  if (wasmExecConstructor) return wasmExecConstructor;
  if (!wasmExecLoad) {
    const wasmExecPath =
      explicitPath ||
      process.env.GEPAGO_WASM_EXEC ||
      (existsSync(bundledWasmExecPath)
        ? bundledWasmExecPath
        : developmentWasmExecPath());
    const wasmExecURL = pathToFileURL(wasmExecPath);
    // A failed attempt may have imported the shim before reading or
    // instantiating the WASM. Use a fresh module URL on the next attempt so
    // top-level global initialisation is not hidden by the ESM module cache.
    wasmExecURL.searchParams.set(
      "gepago_wasm_generation",
      String(wasmExecGeneration),
    );
    wasmExecLoad = import(wasmExecURL.href)
      .then(() => {
        if (typeof globalsObject.Go !== "function") {
          throw new Error(`wasm_exec.js did not install Go: ${wasmExecPath}`);
        }
        wasmExecConstructor = globalsObject.Go;
        return wasmExecConstructor;
      })
      .catch((error) => {
        wasmExecLoad = undefined;
        wasmExecGeneration += 1;
        throw new Error(
          `could not load Go wasm_exec.js from ${wasmExecPath}: ${error instanceof Error ? error.message : String(error)}`,
          { cause: error },
        );
      });
  }
  return wasmExecLoad;
}

function restoreGlobal(
  name: string,
  hadOwnProperty: boolean,
  previousValue: unknown,
): void {
  if (hadOwnProperty) {
    globalsObject[name] = previousValue;
  } else {
    delete globalsObject[name];
  }
}

function assertString(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string")
    throw new TypeError(`${name} must be a string`);
}

function unwrapRuntimeCall(value: unknown, operation: string): string {
  if (typeof value === "string") return value;
  const result = assertPlainObject(value, `${operation} runtime result`);
  if (result.__gepago_wasm_error === true) {
    throw new GepagoWasmError(
      typeof result.message === "string"
        ? result.message
        : `${operation} failed in the WASM runtime`,
      operation,
    );
  }
  throw new GepagoWasmError(
    `${operation} returned an invalid runtime result`,
    operation,
  );
}

function assertPlainObject(value: unknown, name: string): WireObject {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${name} must be a plain object`);
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    throw new TypeError(`${name} must be a plain object`);
  }
  return value as WireObject;
}

function assertJSONValue(
  value: unknown,
  name: string,
  seen: WeakSet<object> = new WeakSet<object>(),
): JSONValue {
  if (value === null || typeof value === "string" || typeof value === "boolean")
    return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value))
      throw new TypeError(`${name} must contain finite JSON numbers`);
    return value as JSONValue;
  }
  if (typeof value !== "object")
    throw new TypeError(`${name} must contain only JSON values`);
  if (seen.has(value))
    throw new TypeError(`${name} must not contain circular references`);
  seen.add(value);
  if (Array.isArray(value)) {
    value.forEach((item, index) =>
      assertJSONValue(item, `${name}[${index}]`, seen),
    );
    // JSON.stringify ignores enumerable properties added to an array. Reject
    // them instead of silently dropping caller data at the WASM boundary.
    for (const key of Object.keys(value)) {
      if (!/^\d+$/.test(key))
        throw new TypeError(`${name} must contain only JSON values`);
    }
  } else {
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new TypeError(`${name} must contain only JSON values`);
    }
    const object = value as Record<string, unknown>;
    for (const key of Object.keys(object))
      assertJSONValue(object[key], `${name}.${key}`, seen);
  }
  if (Object.getOwnPropertySymbols(value).length > 0) {
    throw new TypeError(`${name} must not contain symbol properties`);
  }
  seen.delete(value);
  return value as JSONValue;
}

/*
 * The Go wasm_exec shim installs Go and, when absent, compatibility globals.
 * Keep the Go shim's runtime globals alive, but never leave Go itself behind.
 */
const wasmExecGlobalNames = ["Go", "fs", "process", "path"];

function snapshotGlobals(names: string[]): Map<string, [boolean, unknown]> {
  return new Map(
    names.map((name) => [
      name,
      [
        Object.prototype.hasOwnProperty.call(globalsObject, name),
        globalsObject[name],
      ],
    ]),
  );
}

function restoreGlobals(snapshot: Map<string, [boolean, unknown]>): void {
  for (const [name, [hadOwnProperty, previousValue]] of snapshot) {
    restoreGlobal(name, hadOwnProperty, previousValue);
  }
}

function assertPrediction(value: unknown, name: string): WireObject {
  assertPlainObject(value, name);
  assertJSONValue(value, name);
  return value as WireObject;
}

const optionalUsageFields = [
  "reasoning_tokens",
  "cached_input_tokens",
  "cache_creation_tokens",
];

function assertUsage(value: unknown, name: string): Usage {
  const usage = assertPlainObject(value, name);
  for (const field of ["prompt_tokens", "completion_tokens", "total_tokens"]) {
    if (!Number.isSafeInteger(usage[field]) || (usage[field] as number) < 0) {
      throw new TypeError(
        `${name}.${field} must be a non-negative safe integer`,
      );
    }
  }
  for (const field of optionalUsageFields) {
    if (
      usage[field] !== undefined &&
      (!Number.isSafeInteger(usage[field]) || (usage[field] as number) < 0)
    ) {
      throw new TypeError(
        `${name}.${field} must be a non-negative safe integer`,
      );
    }
  }
  assertJSONValue(usage, name);
  return usage as unknown as Usage;
}

const agentRoles = new Set(["system", "user", "assistant", "tool"]);
const agentFinishReasons = new Set([
  "stop",
  "length",
  "tool_calls",
  "content_filter",
]);
const agentStopReasons = new Set([
  "complete",
  "length",
  "max_turns",
  "tool_error",
  "content_filter",
]);

function assertAgentMessage(value: unknown, name: string): Message {
  const message = assertPlainObject(value, name);
  if (!agentRoles.has(message.role as string))
    throw new TypeError(`${name}.role is invalid`);
  if (message.content !== undefined)
    assertString(message.content, `${name}.content`);
  if (message.tool_call_id !== undefined)
    assertString(message.tool_call_id, `${name}.tool_call_id`);
  if (message.tool_calls !== undefined) {
    if (!Array.isArray(message.tool_calls))
      throw new TypeError(`${name}.tool_calls must be an array`);
    (message.tool_calls as unknown[]).forEach((call: unknown, index: number) =>
      assertAgentToolCall(call, `${name}.tool_calls[${index}]`),
    );
  }
  assertJSONValue(message, name);
  return message as unknown as Message;
}

function assertAgentToolCall(
  value: unknown,
  name: string,
): import("./public-types.ts").ToolCall {
  const call = assertPlainObject(value, name);
  for (const field of ["id", "name", "arguments"])
    assertString(call[field], `${name}.${field}`);
  assertJSONValue(call, name);
  return call as unknown as import("./public-types.ts").ToolCall;
}

function assertAgentTool(value: unknown, name: string): Tool {
  const tool = assertPlainObject(value, name);
  for (const field of ["name", "description"]) {
    assertString(tool[field], `${name}.${field}`);
    if ((tool[field] as string).trim() === "")
      throw new TypeError(`${name}.${field} must not be empty`);
  }
  if (tool.parameters !== undefined) {
    assertPlainObject(tool.parameters, `${name}.parameters`);
    assertJSONValue(tool.parameters, `${name}.parameters`);
  }
  assertJSONValue(tool, name);
  return tool as unknown as Tool;
}

function assertAgentConfig(config: unknown): AgentConfig {
  const object = assertPlainObject(config, "agent config");
  for (const field of ["name", "model", "run_id"]) {
    assertString(object[field], `agent config.${field}`);
    if ((object[field] as string).trim() === "")
      throw new TypeError(`agent config.${field} must not be empty`);
  }
  if (object.systemPrompt !== undefined)
    assertString(object.systemPrompt, "agent config.systemPrompt");
  for (const field of ["maxTurns", "maxTokens"]) {
    if (
      object[field] !== undefined &&
      (!Number.isSafeInteger(object[field]) || (object[field] as number) < 0)
    ) {
      throw new TypeError(
        `agent config.${field} must be a non-negative safe integer`,
      );
    }
  }
  if (
    object.temperature !== undefined &&
    (typeof object.temperature !== "number" ||
      !Number.isFinite(object.temperature))
  ) {
    throw new TypeError("agent config.temperature must be a finite number");
  }
  if (object.messages !== undefined && object.initial_messages !== undefined) {
    throw new TypeError(
      "agent config must use messages or initial_messages, not both",
    );
  }
  const messages = object.messages ?? object.initial_messages;
  if (!Array.isArray(messages) || messages.length === 0) {
    throw new TypeError("agent config.messages must be a non-empty array");
  }
  messages.forEach((message, index) =>
    assertAgentMessage(message, `agent config.messages[${index}]`),
  );
  if (object.tools !== undefined) {
    if (!Array.isArray(object.tools))
      throw new TypeError("agent config.tools must be an array");
    object.tools.forEach((tool, index) =>
      assertAgentTool(tool, `agent config.tools[${index}]`),
    );
    const names = object.tools.map((tool) => tool.name);
    if (new Set(names).size !== names.length)
      throw new TypeError("agent config.tools must not contain duplicates");
  }
  if (object.memory !== undefined) {
    assertPlainObject(object.memory, "agent config.memory");
    assertJSONValue(object.memory, "agent config.memory");
  }
  assertJSONValue(object, "agent config");
  return {
    name: object.name,
    model: object.model,
    ...(object.systemPrompt === undefined
      ? {}
      : { systemPrompt: object.systemPrompt }),
    ...(object.maxTurns === undefined ? {} : { maxTurns: object.maxTurns }),
    ...(object.temperature === undefined
      ? {}
      : { temperature: object.temperature }),
    ...(object.maxTokens === undefined ? {} : { maxTokens: object.maxTokens }),
    messages,
    ...(object.tools === undefined ? {} : { tools: object.tools }),
    run_id: object.run_id,
    ...(object.memory === undefined
      ? {}
      : { memory: object.memory as Record<string, JSONValue> }),
  } as AgentConfig;
}

function parseAgentJSON(encoded: unknown, name: string): unknown {
  assertString(encoded, name);
  try {
    return JSON.parse(encoded);
  } catch (error) {
    throw new GepagoWasmError(
      `${name} is invalid JSON: ${error instanceof Error ? error.message : String(error)}`,
      "agent",
      { cause: error },
    );
  }
}

function assertAgentChatRequest(value: unknown): ChatRequest {
  const request = assertPlainObject(value, "agent chat request");
  assertString(request.model, "agent chat request.model");
  if (!Array.isArray(request.messages))
    throw new TypeError("agent chat request.messages must be an array");
  request.messages.forEach((message, index) =>
    assertAgentMessage(message, `agent chat request.messages[${index}]`),
  );
  if (request.tools !== undefined) {
    if (!Array.isArray(request.tools))
      throw new TypeError("agent chat request.tools must be an array");
    request.tools.forEach((tool, index) =>
      assertAgentTool(tool, `agent chat request.tools[${index}]`),
    );
  }
  if (
    request.temperature !== undefined &&
    (typeof request.temperature !== "number" ||
      !Number.isFinite(request.temperature))
  ) {
    throw new TypeError(
      "agent chat request.temperature must be a finite number",
    );
  }
  if (
    request.max_tokens !== undefined &&
    (!Number.isSafeInteger(request.max_tokens) ||
      (request.max_tokens as number) < 0)
  ) {
    throw new TypeError(
      "agent chat request.max_tokens must be a non-negative safe integer",
    );
  }
  if (request.response_format !== undefined) {
    const format = assertPlainObject(
      request.response_format,
      "agent chat request.response_format",
    );
    assertString(format.name, "agent chat request.response_format.name");
    assertPlainObject(
      format.schema,
      "agent chat request.response_format.schema",
    );
    if (typeof format.strict !== "boolean")
      throw new TypeError(
        "agent chat request.response_format.strict must be a boolean",
      );
    assertJSONValue(format, "agent chat request.response_format");
  }
  assertJSONValue(request, "agent chat request");
  return request as unknown as ChatRequest;
}

function normalizeAgentChatResponse(value: unknown): ChatResponse {
  const response =
    typeof value === "string"
      ? parseAgentJSON(value, "agent chat response")
      : value;
  const object = assertPlainObject(response, "agent chat response");
  const message = assertAgentMessage(
    object.message,
    "agent chat response.message",
  );
  if (message.role !== "assistant") {
    throw new TypeError("agent chat response.message.role must be assistant");
  }
  const responseUsage = assertUsage(object.usage, "agent chat response.usage");
  const reported = isReportedFinishReason(object.finish_reason);
  if (reported && !agentFinishReasons.has(object.finish_reason as string)) {
    throw new TypeError("agent chat response.finish_reason is invalid");
  }
  assertJSONValue(object, "agent chat response");
  const normalized: ChatResponse = { message, usage: responseUsage };
  if (reported) normalized.finish_reason = object.finish_reason as FinishReason;
  return normalized;
}

function assertAgentToolRequest(value: unknown): AgentToolRequest {
  const request = assertPlainObject(value, "agent tool request");
  for (const field of ["name", "call_id", "arguments", "run_id"]) {
    assertString(request[field], `agent tool request.${field}`);
    if (request[field].trim() === "")
      throw new TypeError(`agent tool request.${field} must not be empty`);
  }
  assertPlainObject(request.memory, "agent tool request.memory");
  assertJSONValue(request, "agent tool request");
  return request as unknown as AgentToolRequest;
}

function normalizeAgentToolResult(
  value: unknown,
): import("./public-types.ts").AgentToolResult | string {
  if (typeof value === "string") return value;
  const result = assertPlainObject(value, "agent tool result");
  assertString(result.output, "agent tool result.output");
  if (result.memory !== undefined) {
    assertPlainObject(result.memory, "agent tool result.memory");
    assertJSONValue(result.memory, "agent tool result.memory");
  }
  assertJSONValue(result, "agent tool result");
  return result.memory === undefined
    ? { output: result.output as string }
    : {
        output: result.output as string,
        memory: result.memory as Record<string, JSONValue>,
      };
}

function assertUsageLedger(value: unknown, name: string): UsageLedger {
  const ledger = assertPlainObject(value, name);
  for (const field of [
    "runs",
    "turns",
    "model_calls",
    "tool_calls",
    "metric_calls",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "cache_creation_tokens",
    "total_tokens",
    "total_duration",
  ]) {
    if (!Number.isSafeInteger(ledger[field]) || (ledger[field] as number) < 0)
      throw new TypeError(
        `${name}.${field} must be a non-negative safe integer`,
      );
  }
  if (
    typeof ledger.total_cost_usd !== "number" ||
    !Number.isFinite(ledger.total_cost_usd) ||
    ledger.total_cost_usd < 0
  ) {
    throw new TypeError(
      `${name}.total_cost_usd must be a finite non-negative number`,
    );
  }
  assertJSONValue(ledger, name);
  return ledger as unknown as UsageLedger;
}

function assertUsageSpan(value: unknown, name: string): UsageSpan {
  const span = assertPlainObject(value, name);
  for (const field of [
    "id",
    "trace_id",
    "kind",
    "name",
    "started_at",
    "ended_at",
  ])
    assertString(span[field], `${name}.${field}`);
  if (span.parent_id !== undefined)
    assertString(span.parent_id, `${name}.parent_id`);
  if (!Number.isSafeInteger(span.duration) || (span.duration as number) < 0)
    throw new TypeError(`${name}.duration must be a non-negative safe integer`);
  assertUsage(span.usage, `${name}.usage`);
  if (span.model !== undefined) assertString(span.model, `${name}.model`);
  if (
    span.cost_usd !== undefined &&
    (typeof span.cost_usd !== "number" || !Number.isFinite(span.cost_usd))
  )
    throw new TypeError(`${name}.cost_usd must be finite`);
  if (span.err !== undefined) assertString(span.err, `${name}.err`);
  if (span.attributes !== undefined) {
    assertPlainObject(span.attributes, `${name}.attributes`);
    assertJSONValue(span.attributes, `${name}.attributes`);
  }
  assertJSONValue(span, name);
  return span as unknown as UsageSpan;
}

function assertAgentToolCallRecord(
  value: unknown,
  name: string,
): ToolCallRecord {
  const record = assertPlainObject(value, name);
  if (!Number.isSafeInteger(record.turn) || (record.turn as number) < 0)
    throw new TypeError(`${name}.turn must be a non-negative safe integer`);
  for (const field of ["call_id", "name", "arguments"])
    assertString(record[field], `${name}.${field}`);
  if (record.output !== undefined)
    assertString(record.output, `${name}.output`);
  if (record.err !== undefined) assertString(record.err, `${name}.err`);
  if (!Number.isSafeInteger(record.duration) || (record.duration as number) < 0)
    throw new TypeError(`${name}.duration must be a non-negative safe integer`);
  assertJSONValue(record, name);
  return record as unknown as ToolCallRecord;
}

function assertAgentRunResult(value: unknown): RunResult {
  const result = assertPlainObject(value, "agent run result");
  assertAgentMessage(result.final, "agent run result.final");
  if (!Array.isArray(result.transcript))
    throw new TypeError("agent run result.transcript must be an array");
  result.transcript.forEach((message, index) =>
    assertAgentMessage(message, `agent run result.transcript[${index}]`),
  );
  assertUsage(result.usage, "agent run result.usage");
  assertUsageLedger(result.ledger, "agent run result.ledger");
  const spans = result.spans === null ? [] : result.spans;
  if (!Array.isArray(spans))
    throw new TypeError("agent run result.spans must be an array");
  spans.forEach((span, index) =>
    assertUsageSpan(span, `agent run result.spans[${index}]`),
  );
  if (!agentStopReasons.has(result.stop_reason as string))
    throw new TypeError("agent run result.stop_reason is invalid");
  if (!Number.isSafeInteger(result.turns) || (result.turns as number) < 0)
    throw new TypeError(
      "agent run result.turns must be a non-negative safe integer",
    );
  const toolCalls = result.tool_calls === null ? [] : result.tool_calls;
  if (!Array.isArray(toolCalls))
    throw new TypeError("agent run result.tool_calls must be an array");
  toolCalls.forEach((call, index) =>
    assertAgentToolCallRecord(call, `agent run result.tool_calls[${index}]`),
  );
  const normalized = { ...result, spans, tool_calls: toolCalls };
  assertJSONValue(normalized, "agent run result");
  return normalized as unknown as RunResult;
}

function assertProgramPromptFields(program: WireObject, name: string): void {
  if (program.instruction !== undefined)
    assertString(program.instruction, `${name}.instruction`);
  if (program.demos !== undefined) assertDemos(program.demos);
  if (
    program.maxParseRetries !== undefined &&
    (!Number.isSafeInteger(program.maxParseRetries) ||
      (program.maxParseRetries as number) < 0)
  ) {
    throw new TypeError(
      `${name}.maxParseRetries must be a non-negative safe integer`,
    );
  }
  if (program.reasoningField !== undefined)
    assertString(program.reasoningField, `${name}.reasoningField`);
}

function normalizeChainOfThoughtProgram(
  program: WireObject,
  name: string,
): WireObject {
  const signature = assertCompileSignature(
    program.signature,
    `${name}.signature`,
  );
  assertProgramPromptFields(program, name);
  const reasoningField =
    program.reasoningField === undefined || program.reasoningField === ""
      ? "reasoning"
      : program.reasoningField;
  // as in cot.go: the reasoning field comes first, so a model that writes fields in schema
  // order reasons before it answers, and it is a string when the other outputs are typed.
  const typed = signature.outputs.some(
    (field) =>
      field.type !== undefined ||
      field.schema !== undefined ||
      field.optional === true,
  );
  const outputs = signature.outputs.some(
    (field) => field.name === reasoningField,
  )
    ? signature.outputs
    : [
        {
          name: reasoningField,
          description: "brief reasoning for the answer",
          ...(typed ? { type: "string" } : {}),
        },
        ...signature.outputs,
      ];
  const instruction = String(
    program.instruction ?? signature.description ?? "",
  ).trim();
  return {
    signature: { ...signature, outputs },
    instruction: `${instruction}\nThink step by step before answering.\nInclude a concise reasoning string in the ${reasoningField} field.`,
    ...(program.demos === undefined ? {} : { demos: program.demos }),
    ...(program.maxParseRetries === undefined
      ? {}
      : { maxParseRetries: program.maxParseRetries }),
  };
}

function normalizePredictProgram(program: unknown, name: string): WireObject {
  const object = assertPlainObject(program, name);
  const kind = object.kind ?? "predict";
  if (kind === "chainOfThought")
    return normalizeChainOfThoughtProgram(object, name);
  if (kind !== "predict") {
    throw new TypeError(`${name}.kind must be predict or chainOfThought`);
  }
  const signature = assertCompileSignature(
    object.signature,
    `${name}.signature`,
  );
  assertProgramPromptFields(object, name);
  return {
    signature,
    ...(object.instruction === undefined
      ? {}
      : { instruction: object.instruction }),
    ...(object.demos === undefined ? {} : { demos: object.demos }),
    ...(object.maxParseRetries === undefined
      ? {}
      : { maxParseRetries: object.maxParseRetries }),
  };
}

function assertCompileProgram(
  program: unknown,
  name: string = "compile config.program",
  allowPipeline = true,
): WireObject {
  const object = assertPlainObject(program, name);
  const kind = object.kind ?? "predict";
  if (kind === "predict" || kind === "chainOfThought") {
    const signature = assertCompileSignature(
      object.signature,
      `${name}.signature`,
    );
    assertProgramPromptFields(object, name);
    return {
      kind,
      signature,
      ...(object.instruction === undefined
        ? {}
        : { instruction: object.instruction }),
      ...(object.demos === undefined ? {} : { demos: object.demos }),
      ...(object.maxParseRetries === undefined
        ? {}
        : { maxParseRetries: object.maxParseRetries }),
      ...(object.reasoningField === undefined
        ? {}
        : { reasoningField: object.reasoningField }),
    };
  }
  if (kind !== "pipeline" || !allowPipeline) {
    throw new TypeError(
      `${name}.kind is unsupported; use predict, chainOfThought, or pipeline`,
    );
  }
  if (!Array.isArray(object.steps) || object.steps.length === 0) {
    throw new TypeError(`${name}.steps must be a non-empty array`);
  }
  const names = new Set();
  const steps = object.steps.map((step, index) => {
    const item = assertPlainObject(step, `${name}.steps[${index}]`);
    assertString(item.name, `${name}.steps[${index}].name`);
    const stepName = item.name.trim();
    if (stepName === "")
      throw new TypeError(`${name}.steps[${index}].name must not be empty`);
    if (stepName.includes("."))
      throw new TypeError(`${name}.steps[${index}].name must not contain '.'`);
    if (names.has(stepName))
      throw new TypeError(`${name}.steps contains duplicate name ${stepName}`);
    names.add(stepName);
    const child = assertCompileProgram(
      item.program,
      `${name}.steps[${index}].program`,
      false,
    );
    if (child.kind === "pipeline")
      throw new TypeError(
        `${name}.steps[${index}].program.kind must be predict or chainOfThought`,
      );
    const source = assertPlainObject(
      item.program,
      `${name}.steps[${index}].program`,
    );
    if (source.inputKeys !== undefined) {
      if (
        !Array.isArray(source.inputKeys) ||
        source.inputKeys.some(
          (key) => typeof key !== "string" || key.trim() === "",
        )
      ) {
        throw new TypeError(
          `${name}.steps[${index}].program.inputKeys must be an array of non-empty strings`,
        );
      }
      if (new Set(source.inputKeys).size !== source.inputKeys.length) {
        throw new TypeError(
          `${name}.steps[${index}].program.inputKeys must not contain duplicates`,
        );
      }
    }
    if (source.outputPrefix !== undefined)
      assertString(
        source.outputPrefix,
        `${name}.steps[${index}].program.outputPrefix`,
      );
    return {
      name: stepName,
      program: {
        ...child,
        ...(source.inputKeys === undefined
          ? {}
          : { inputKeys: source.inputKeys }),
        ...(source.outputPrefix === undefined
          ? {}
          : { outputPrefix: source.outputPrefix }),
      },
    };
  });
  if (object.returnAll !== undefined && typeof object.returnAll !== "boolean") {
    throw new TypeError(`${name}.returnAll must be a boolean`);
  }
  return {
    kind: "pipeline",
    steps,
    ...(object.returnAll === undefined ? {} : { returnAll: object.returnAll }),
  };
}

function assertPredictConfig(config: unknown): PredictConfig {
  const object = assertPlainObject(config, "predict config");
  const program = normalizePredictProgram(
    object.program,
    "predict config.program",
  );
  assertPrediction(object.inputs, "predict config.inputs");
  return {
    program: program as unknown as PredictConfig["program"],
    inputs: object.inputs as Record<string, JSONValue>,
  };
}

function assertCompileSignature(value: unknown, name: string): Signature {
  const signature = assertPlainObject(value, name);
  if (!Array.isArray(signature.inputs) || !Array.isArray(signature.outputs)) {
    throw new TypeError(`${name} must contain inputs and outputs arrays`);
  }
  assertJSONValue(signature, name);
  return signature as unknown as Signature;
}

function assertCompileExamples(
  value: unknown,
  name: string,
  required = false,
): unknown[] {
  if (!Array.isArray(value) || (required && value.length === 0)) {
    throw new TypeError(`${name} must be a non-empty array`);
  }
  const ids = new Set();
  value.forEach((example, index) => {
    const item = assertPlainObject(example, `${name}[${index}]`);
    if (typeof item.id !== "string" || item.id.trim() === "") {
      throw new TypeError(`${name}[${index}].id must be a non-empty string`);
    }
    if (ids.has(item.id))
      throw new TypeError(`${name} contains duplicate id ${item.id}`);
    ids.add(item.id);
    assertPrediction(item.input, `${name}[${index}].input`);
    assertPrediction(item.expected, `${name}[${index}].expected`);
    assertJSONValue(item, `${name}[${index}]`);
  });
  return value;
}

function assertCompileMetricOptions(
  value: unknown,
  name: string,
): WireObject | undefined {
  if (value === undefined) return undefined;
  const options = assertPlainObject(value, name);
  if (options.fields !== undefined)
    assertFields(options.fields, `${name}.fields`);
  for (const field of [
    "caseInsensitive",
    "trimSpace",
    "ignoreCase",
    "parseRetry",
  ]) {
    if (options[field] !== undefined && typeof options[field] !== "boolean") {
      throw new TypeError(`${name}.${field} must be a boolean`);
    }
  }
  if (
    options.tolerance !== undefined &&
    (typeof options.tolerance !== "number" ||
      !Number.isFinite(options.tolerance) ||
      options.tolerance < 0)
  ) {
    throw new TypeError(
      `${name}.tolerance must be a finite non-negative number`,
    );
  }
  if (
    options.maxScore !== undefined &&
    (typeof options.maxScore !== "number" ||
      !Number.isFinite(options.maxScore) ||
      options.maxScore < 0)
  ) {
    throw new TypeError(
      `${name}.maxScore must be a finite non-negative number`,
    );
  }
  for (const field of [
    "field",
    "positiveLabel",
    "negativeLabel",
    "rubric",
    "role",
  ]) {
    if (options[field] !== undefined)
      assertString(options[field], `${name}.${field}`);
  }
  assertJSONValue(options, name);
  return options;
}

function assertCompileMetric(value: unknown): WireObject {
  const metric = assertPlainObject(value, "compile config.metric");
  assertString(metric.kind, "compile config.metric.kind");
  if (
    ![
      "exactMatch",
      "contains",
      "numericMatch",
      "classification",
      "llmJudge",
      "host",
    ].includes(metric.kind)
  ) {
    throw new TypeError(
      `compile config.metric.kind is unsupported: ${metric.kind}`,
    );
  }
  if (metric.fields !== undefined)
    assertFields(metric.fields, "compile config.metric.fields");
  const options = assertCompileMetricOptions(
    metric.options,
    "compile config.metric.options",
  );
  for (const field of [
    "caseInsensitive",
    "trimSpace",
    "ignoreCase",
    "parseRetry",
  ]) {
    if (metric[field] !== undefined && typeof metric[field] !== "boolean") {
      throw new TypeError(`compile config.metric.${field} must be a boolean`);
    }
  }
  if (
    metric.tolerance !== undefined &&
    (typeof metric.tolerance !== "number" ||
      !Number.isFinite(metric.tolerance) ||
      metric.tolerance < 0)
  ) {
    throw new TypeError(
      "compile config.metric.tolerance must be a finite non-negative number",
    );
  }
  if (
    metric.maxScore !== undefined &&
    (typeof metric.maxScore !== "number" ||
      !Number.isFinite(metric.maxScore) ||
      metric.maxScore < 0)
  ) {
    throw new TypeError(
      "compile config.metric.maxScore must be a finite non-negative number",
    );
  }
  for (const field of [
    "field",
    "positiveLabel",
    "negativeLabel",
    "rubric",
    "role",
  ]) {
    if (metric[field] !== undefined)
      assertString(metric[field], `compile config.metric.${field}`);
  }
  const role = metric.role ?? options?.role;
  if (role !== undefined && role !== "judge") {
    throw new TypeError("compile config.metric.role must be judge");
  }
  if (
    metric.kind !== "llmJudge" &&
    metric.kind !== "host" &&
    role !== undefined
  ) {
    throw new TypeError(
      "compile config.metric.role is only supported for llmJudge",
    );
  }
  assertJSONValue(metric, "compile config.metric");
  return {
    kind: metric.kind,
    ...(metric.fields === undefined ? {} : { fields: metric.fields }),
    ...(metric.caseInsensitive === undefined
      ? {}
      : { caseInsensitive: metric.caseInsensitive }),
    ...(metric.trimSpace === undefined ? {} : { trimSpace: metric.trimSpace }),
    ...(metric.tolerance === undefined ? {} : { tolerance: metric.tolerance }),
    ...(metric.field === undefined ? {} : { field: metric.field }),
    ...(metric.positiveLabel === undefined
      ? {}
      : { positiveLabel: metric.positiveLabel }),
    ...(metric.negativeLabel === undefined
      ? {}
      : { negativeLabel: metric.negativeLabel }),
    ...(metric.ignoreCase === undefined
      ? {}
      : { ignoreCase: metric.ignoreCase }),
    ...(metric.rubric === undefined ? {} : { rubric: metric.rubric }),
    ...(metric.role === undefined ? {} : { role: metric.role }),
    ...(metric.maxScore === undefined ? {} : { maxScore: metric.maxScore }),
    ...(metric.parseRetry === undefined
      ? {}
      : { parseRetry: metric.parseRetry }),
    ...(options === undefined ? {} : { options }),
  };
}

function assertCompileConfig(config: unknown): CompileConfig {
  const object = assertPlainObject(config, "compile config");
  const program = assertCompileProgram(object.program);
  const trainset = assertCompileExamples(
    object.trainset,
    "compile config.trainset",
    true,
  );
  const valset =
    object.valset === undefined
      ? undefined
      : assertCompileExamples(object.valset, "compile config.valset");
  const metric = assertCompileMetric(object.metric);
  if (object.objective !== undefined)
    assertString(object.objective, "compile config.objective");
  if (object.components !== undefined) {
    const components = assertFields(
      object.components,
      "compile config.components",
    );
    if (components.some((component: string) => component.trim() === "")) {
      throw new TypeError(
        "compile config.components must not contain empty names",
      );
    }
    if (new Set(components).size !== components.length) {
      throw new TypeError(
        "compile config.components must not contain duplicates",
      );
    }
  }
  for (const field of ["maxMetricCalls", "minibatchSize"]) {
    if (
      object[field] !== undefined &&
      (!Number.isSafeInteger(object[field]) || (object[field] as number) <= 0)
    ) {
      throw new TypeError(
        `compile config.${field} must be a positive safe integer`,
      );
    }
  }
  if (object.seed !== undefined && !Number.isSafeInteger(object.seed)) {
    throw new TypeError("compile config.seed must be a safe integer");
  }
  if (
    object.acceptance !== undefined &&
    object.acceptance !== "strict" &&
    object.acceptance !== "equal"
  ) {
    throw new TypeError("compile config.acceptance must be strict or equal");
  }
  if (object.resume !== undefined && typeof object.resume !== "boolean") {
    throw new TypeError("compile config.resume must be a boolean");
  }
  return {
    program: program as unknown as CompileConfig["program"],
    trainset: trainset as CompileConfig["trainset"],
    ...(valset === undefined ? {} : { valset }),
    metric: metric as unknown as CompileConfig["metric"],
    ...(object.objective === undefined ? {} : { objective: object.objective }),
    ...(object.components === undefined
      ? {}
      : { components: assertFields(object.components) }),
    ...(object.maxMetricCalls === undefined
      ? {}
      : { maxMetricCalls: object.maxMetricCalls }),
    ...(object.minibatchSize === undefined
      ? {}
      : { minibatchSize: object.minibatchSize }),
    ...(object.seed === undefined ? {} : { seed: object.seed }),
    ...(object.acceptance === undefined
      ? {}
      : { acceptance: object.acceptance }),
    ...(object.resume === undefined ? {} : { resume: object.resume }),
  } as unknown as CompileConfig;
}

const compileModelRoles = new Set(["task", "reflector", "judge", "repair"]);

function parseCompileModelRequest(encoded: unknown): CompileModelRequest {
  assertString(encoded, "compile model request");
  let request;
  try {
    request = JSON.parse(encoded);
  } catch (error) {
    throw new GepagoWasmError(
      `compile model request is invalid JSON: ${error instanceof Error ? error.message : String(error)}`,
      "compile",
    );
  }
  const object = assertPlainObject(request, "compile model request");
  if (object.kind !== "language_model.generate") {
    throw new GepagoWasmError(
      "compile model request has an invalid kind",
      "compile",
    );
  }
  if (!compileModelRoles.has(object.role as string)) {
    throw new GepagoWasmError(
      "compile model request has an invalid role",
      "compile",
    );
  }
  assertString(object.prompt, "compile model request.prompt");
  if (
    object.attempt !== undefined &&
    (!Number.isSafeInteger(object.attempt) || (object.attempt as number) < 1)
  ) {
    throw new GepagoWasmError(
      "compile model request.attempt must be a positive safe integer",
      "compile",
    );
  }
  assertJSONValue(object, "compile model request");
  return object as unknown as CompileModelRequest;
}

function normalizeCompileModelResult(
  value: unknown,
): CompileModelOutput | CompileModelResult {
  return normalizeHostModelResult(value, "compile model result");
}

// isReportedFinishReason treats undefined, null and "" as a finish reason the provider did not
// report, as the Go bridge does.
function isReportedFinishReason(value: unknown): boolean {
  return value !== undefined && value !== null && value !== "";
}

// normalizeHostModelResult validates a host model's reply and keeps only the fields the WASM
// module reads: text, plus optional usage, finishReason and model.
function normalizeHostModelResult(
  value: unknown,
  name: string,
): PredictModelResult {
  if (typeof value === "string") return { text: value };
  const result = assertPlainObject(value, name);
  if (!Object.hasOwn(result, "text"))
    throw new TypeError(`${name}.text is required`);
  assertString(result.text, `${name}.text`);
  if (result.usage !== undefined) assertUsage(result.usage, `${name}.usage`);
  const reported = isReportedFinishReason(result.finishReason);
  if (reported && !agentFinishReasons.has(result.finishReason as string))
    throw new TypeError(
      `${name}.finishReason must be one of stop, length, tool_calls or content_filter`,
    );
  if (result.model !== undefined) assertString(result.model, `${name}.model`);
  assertJSONValue(result, name);
  const normalized: PredictModelResult = { text: result.text as string };
  if (result.usage !== undefined) normalized.usage = result.usage as Usage;
  if (reported) normalized.finishReason = result.finishReason as FinishReason;
  if (result.model !== undefined) normalized.model = result.model as string;
  return normalized;
}

function assertCompileResult(value: unknown): CompileResult {
  const result = assertPlainObject(value, "compile result");
  const candidate = assertPlainObject(
    result.candidate,
    "compile result.candidate",
  );
  for (const [key, text] of Object.entries(candidate))
    assertString(text, `compile result.candidate.${key}`);
  const state = assertPlainObject(result.state, "compile result.state");
  assertJSONValue({ candidate, state }, "compile result");
  return { candidate, state } as unknown as CompileResult;
}

function assertCapabilityCallbacks(
  callbacks: unknown,
  operation: string,
): Record<string, (...args: never[]) => unknown> {
  const object = assertPlainObject(callbacks, `${operation} callbacks`);
  for (const [name, callback] of Object.entries(object)) {
    if (typeof callback !== "function")
      throw new TypeError(`${operation} callbacks.${name} must be a function`);
  }
  return object as unknown as Record<string, (...args: never[]) => unknown>;
}

function assertOptimizeConfig(config: unknown): OptimizeConfig {
  const object = assertPlainObject(config, "optimize config");
  const seedCandidate = assertPlainObject(
    object.seedCandidate,
    "optimize config.seedCandidate",
  );
  for (const [key, value] of Object.entries(seedCandidate))
    assertString(value, `optimize config.seedCandidate.${key}`);
  const examples = (
    value: unknown,
    name: string,
    required: boolean,
  ): Array<{ id: string; input: JSONValue }> | undefined => {
    if (value === undefined && !required) return undefined;
    if (!Array.isArray(value) || (required && value.length === 0)) {
      throw new TypeError(`${name} must be a non-empty array`);
    }
    const ids = new Set();
    return value.map((example, index) => {
      const item = assertPlainObject(example, `${name}[${index}]`);
      assertString(item.id, `${name}[${index}].id`);
      if ((item.id as string).trim() === "" || ids.has(item.id))
        throw new TypeError(`${name} contains duplicate or empty id`);
      ids.add(item.id);
      if (!Object.hasOwn(item, "input"))
        throw new TypeError(`${name}[${index}].input is required`);
      assertJSONValue(item.input, `${name}[${index}].input`);
      return { id: item.id as string, input: item.input as JSONValue };
    });
  };
  const trainset = examples(object.trainset, "optimize config.trainset", true)!;
  const valset = examples(object.valset, "optimize config.valset", false);
  for (const name of [
    "objective",
    "background",
    "acceptance",
    "candidateSelector",
    "componentSelector",
  ]) {
    if (object[name] !== undefined)
      assertString(object[name], `optimize config.${name}`);
  }
  for (const name of ["maxMetricCalls", "minibatchSize", "seed", "topK"]) {
    if (
      object[name] !== undefined &&
      (!Number.isSafeInteger(object[name]) ||
        (name !== "seed" && (object[name] as number) <= 0))
    ) {
      throw new TypeError(`optimize config.${name} must be a valid integer`);
    }
  }
  if (
    object.epsilon !== undefined &&
    (typeof object.epsilon !== "number" ||
      !Number.isFinite(object.epsilon) ||
      object.epsilon < 0 ||
      object.epsilon > 1)
  ) {
    throw new TypeError("optimize config.epsilon must be between 0 and 1");
  }
  if (object.promptTemplates !== undefined) {
    const templates = assertPlainObject(
      object.promptTemplates,
      "optimize config.promptTemplates",
    );
    for (const [key, value] of Object.entries(templates))
      assertString(value, `optimize config.promptTemplates.${key}`);
  }
  if (object.components !== undefined) {
    if (!Array.isArray(object.components))
      throw new TypeError("optimize config.components must be an array");
    object.components.forEach((component, index) => {
      assertString(component, `optimize config.components[${index}]`);
      if ((component as string).trim() === "")
        throw new TypeError(
          "optimize config.components must not contain empty names",
        );
    });
  }
  for (const name of ["enableMerge", "resume"]) {
    if (object[name] !== undefined && typeof object[name] !== "boolean")
      throw new TypeError(`optimize config.${name} must be a boolean`);
  }
  if (
    object.acceptance !== undefined &&
    !["strict", "equal"].includes(String(object.acceptance))
  )
    throw new TypeError("optimize config.acceptance must be strict or equal");
  if (
    object.candidateSelector !== undefined &&
    !["pareto", "best", "topK", "epsilonGreedy"].includes(
      String(object.candidateSelector),
    )
  )
    throw new TypeError("optimize config.candidateSelector is invalid");
  if (
    object.componentSelector !== undefined &&
    !["all", "roundRobin"].includes(String(object.componentSelector))
  )
    throw new TypeError("optimize config.componentSelector is invalid");
  assertJSONValue(object, "optimize config");
  return {
    seedCandidate,
    trainset,
    ...(valset === undefined ? {} : { valset }),
    ...(object.objective === undefined ? {} : { objective: object.objective }),
    ...(object.background === undefined
      ? {}
      : { background: object.background }),
    ...(object.promptTemplates === undefined
      ? {}
      : { promptTemplates: object.promptTemplates }),
    ...(object.components === undefined
      ? {}
      : { components: object.components }),
    maxMetricCalls: object.maxMetricCalls ?? valset?.length ?? trainset.length,
    minibatchSize: object.minibatchSize ?? Math.min(3, trainset.length),
    seed: object.seed ?? 0,
    ...(object.acceptance === undefined
      ? {}
      : { acceptance: object.acceptance }),
    ...(object.candidateSelector === undefined
      ? {}
      : { candidateSelector: object.candidateSelector }),
    ...(object.componentSelector === undefined
      ? {}
      : { componentSelector: object.componentSelector }),
    ...(object.topK === undefined ? {} : { topK: object.topK }),
    ...(object.epsilon === undefined ? {} : { epsilon: object.epsilon }),
    ...(object.enableMerge === undefined
      ? {}
      : { enableMerge: object.enableMerge }),
    ...(object.resume === undefined ? {} : { resume: object.resume }),
  } as unknown as OptimizeConfig;
}

function assertPredictPending(value: unknown): WireObject {
  const pending = assertPlainObject(value, "predict response");
  if (
    pending.status !== "pending" ||
    typeof pending.job_id !== "string" ||
    pending.job_id === ""
  ) {
    throw new GepagoWasmError(
      "predict returned an invalid pending response",
      "predict",
    );
  }
  const request = assertPlainObject(
    pending.host_request,
    "predict host request",
  );
  if (
    typeof request.request_id !== "string" ||
    request.request_id === "" ||
    request.kind !== "language_model.generate" ||
    request.role !== "task" ||
    typeof request.prompt !== "string" ||
    !Number.isSafeInteger(request.attempt) ||
    (request.attempt as number) < 1
  ) {
    throw new GepagoWasmError(
      "predict returned an invalid host request",
      "predict",
    );
  }
  assertJSONValue(pending, "predict response");
  return pending;
}

function assertPredictComplete(value: unknown): PredictResult {
  const complete = assertPlainObject(value, "predict response");
  if (complete.status !== "complete") {
    throw new GepagoWasmError(
      "predict returned an invalid complete response",
      "predict",
    );
  }
  const result = assertPlainObject(complete.value, "predict response.value");
  assertPrediction(result.prediction, "predict response.value.prediction");
  assertUsage(result.usage, "predict response.value.usage");
  assertJSONValue(complete, "predict response");
  return result as unknown as PredictResult;
}

function normalizeModelResult(value: unknown): PredictModelResult {
  return normalizeHostModelResult(value, "predict model result");
}

function assertDemos(value: unknown): asserts value is Demo[] {
  if (!Array.isArray(value)) throw new TypeError("demos must be an array");
  value.forEach((demo, index) => {
    const object = assertPlainObject(demo, `demos[${index}]`);
    assertPrediction(object.inputs, `demos[${index}].inputs`);
    assertPrediction(object.outputs, `demos[${index}].outputs`);
    if (object.rationale !== undefined)
      assertString(object.rationale, `demos[${index}].rationale`);
    assertJSONValue(object, `demos[${index}]`);
  });
}

function assertFields(fields: unknown, name: string = "fields"): string[] {
  if (
    !Array.isArray(fields) ||
    fields.some((field) => typeof field !== "string")
  ) {
    throw new TypeError(`${name} must be an array of strings`);
  }
  return fields as string[];
}

function assertMetricOptions(options: unknown, kind: string): void {
  if (options === undefined) return;
  if (Array.isArray(options)) {
    assertFields(options);
    return;
  }
  const object = assertPlainObject(options, `${kind} options`);
  if (object.fields !== undefined) assertFields(object.fields);
  for (const name of ["caseInsensitive", "trimSpace", "ignoreCase"]) {
    if (object[name] !== undefined && typeof object[name] !== "boolean") {
      throw new TypeError(`${kind} options.${name} must be a boolean`);
    }
  }
  if (
    object.tolerance !== undefined &&
    (typeof object.tolerance !== "number" ||
      !Number.isFinite(object.tolerance) ||
      object.tolerance < 0)
  ) {
    throw new TypeError(
      `${kind} options.tolerance must be a finite non-negative number`,
    );
  }
  for (const name of ["field", "positiveLabel", "negativeLabel"]) {
    if (object[name] !== undefined)
      assertString(object[name], `${kind} options.${name}`);
  }
  assertJSONValue(object, `${kind} options`);
}

function encodePayload(operation: string, payload: unknown): string {
  assertJSONValue(payload, `${operation} payload`);
  try {
    return JSON.stringify(payload);
  } catch (error) {
    throw new TypeError(
      `could not encode ${operation} payload: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

function parseEnvelope(operation: string, encoded: unknown): unknown {
  if (typeof encoded !== "string") {
    throw new GepagoWasmError(
      `${operation} returned a non-string response`,
      operation,
    );
  }
  let response;
  try {
    response = JSON.parse(encoded);
  } catch (error) {
    throw new GepagoWasmError(
      `${operation} returned invalid JSON: ${error instanceof Error ? error.message : String(error)}`,
      operation,
    );
  }
  if (
    response === null ||
    typeof response !== "object" ||
    Array.isArray(response) ||
    (response.ok !== true && response.ok !== false)
  ) {
    throw new GepagoWasmError(
      `${operation} returned an invalid JSON envelope`,
      operation,
    );
  }
  if (response.ok === false) {
    if (typeof response.error !== "string") {
      throw new GepagoWasmError(
        `${operation} returned an invalid JSON envelope`,
        operation,
      );
    }
    throw new GepagoWasmError(response.error, operation);
  }
  if (!Object.hasOwn(response, "value")) {
    throw new GepagoWasmError(
      `${operation} returned an invalid JSON envelope`,
      operation,
    );
  }
  return response.value;
}

function assertBuildPromptRequest(
  signature: unknown,
  instruction: unknown,
  demos: unknown,
  inputs: unknown,
): void {
  assertPlainObject(signature, "signature");
  assertJSONValue(signature, "signature");
  assertString(instruction, "instruction");
  assertDemos(demos);
  assertPrediction(inputs, "inputs");
}

function metricPayload(
  kind: string,
  expected: Record<string, JSONValue>,
  actual: Record<string, JSONValue>,
  options: unknown,
): WireObject {
  const normalizedOptions = Array.isArray(options)
    ? { fields: options }
    : options;
  return { kind, expected, actual, options: normalizedOptions };
}

function candidateToPromptRequest(
  signature: Signature,
  candidate: Candidate,
  inputs: Record<string, JSONValue>,
): WireObject {
  assertPlainObject(candidate, "candidate");
  for (const [key, value] of Object.entries(candidate))
    assertString(value, `candidate.${key}`);
  let demos = [];
  if (candidate.demos !== undefined && candidate.demos !== "") {
    try {
      demos = JSON.parse(candidate.demos);
    } catch (error) {
      throw new TypeError(
        `candidate.demos must contain encoded demo JSON: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }
  return { signature, instruction: candidate.instruction ?? "", demos, inputs };
}

function assertCompiledCandidate(
  candidate: unknown,
  program: RunProgram,
): Candidate {
  const object = assertPlainObject(candidate, "compiled candidate");
  for (const [key, value] of Object.entries(object))
    assertString(value, `compiled candidate.${key}`);
  if (program.kind !== "pipeline") {
    const unsupported = Object.keys(object).filter(
      (key) => key !== "instruction" && key !== "demos",
    );
    if (unsupported.length > 0) {
      throw new TypeError(
        `runCompiled only supports Predict candidates; unsupported component(s): ${unsupported.join(", ")}`,
      );
    }
    return object as unknown as Candidate;
  }
  const steps = new Map(program.steps.map((step) => [step.name, step]));
  for (const key of Object.keys(object)) {
    const separator = key.indexOf(".");
    const stepName = separator < 0 ? "" : key.slice(0, separator);
    const component = separator < 0 ? key : key.slice(separator + 1);
    if (!steps.has(stepName) || !["instruction", "demos"].includes(component)) {
      throw new TypeError(
        `runCompiled pipeline candidate has unsupported component: ${key}`,
      );
    }
  }
  return object as unknown as Candidate;
}

function candidateForStep(candidate: Candidate, stepName: string): Candidate {
  const scoped: Candidate = {};
  const prefix = `${stepName}.`;
  for (const [key, value] of Object.entries(candidate)) {
    if (key.startsWith(prefix)) scoped[key.slice(prefix.length)] = value;
  }
  return scoped;
}

// pipelineStepReads returns the keys a pipeline step reads, as Compile runs it: its
// inputKeys, or its signature's inputs when it lists none.
function pipelineStepReads(step: PipelineStep): string[] {
  const keys = step.program.inputKeys;
  return keys === undefined || keys.length === 0
    ? step.program.signature.inputs.map((field) => field.name)
    : keys;
}

// pipelineStepWrites returns a step's outputs under its outputPrefix, as Compile commits them:
// an optional output that the reply leaves out or sets to null holds the zero value of its type.
function pipelineStepWrites(
  step: PipelineStep,
  prediction: Record<string, JSONValue>,
): Record<string, JSONValue> {
  const prefix = (step.program.outputPrefix ?? "").trim();
  const writes: Record<string, JSONValue> = {};
  for (const [key, value] of Object.entries(prediction))
    writes[`${prefix}${key}`] = value;
  for (const field of step.program.signature.outputs) {
    const key = `${prefix}${field.name}`;
    if (field.optional === true && (writes[key] ?? null) === null)
      writes[key] = zeroFieldValue(field.type);
  }
  return writes;
}

// zeroFieldValue returns the JSON form of the Go zero value of a Field type.
function zeroFieldValue(type: string | undefined): JSONValue {
  switch (type) {
    case "bool":
      return false;
    case "string":
      return "";
    case "int":
    case "int8":
    case "int16":
    case "int32":
    case "int64":
    case "uint":
    case "uint8":
    case "uint16":
    case "uint32":
    case "uint64":
    case "float32":
    case "float64":
      return 0;
    default:
      return null;
  }
}

function applyCandidateToProgram(
  program: RunProgram,
  candidate: Candidate,
  decodeDemos: (text: string) => Demo[],
): RunProgram {
  const demos =
    candidate.demos === undefined ? undefined : decodeDemos(candidate.demos);
  return {
    ...program,
    ...(candidate.instruction === undefined
      ? {}
      : { instruction: candidate.instruction }),
    ...(demos === undefined ? {} : { demos }),
  } as RunProgram;
}

/**
 * Load the production Gepago Go js/wasm module.
 *
 * The Go module currently owns one process-global namespace and never exits,
 * so this package intentionally permits only one loaded instance per process.
 */
export async function createGepagoWasm(
  options: CreateGepagoWasmOptions = {},
): Promise<LoadedRuntime> {
  if (
    options === null ||
    typeof options !== "object" ||
    Array.isArray(options)
  ) {
    throw new TypeError("createGepagoWasm options must be an object");
  }
  const { wasmPath: requestedWasmPath, wasmExecPath: requestedWasmExecPath } =
    options;
  let wasmPath = requestedWasmPath;
  let wasmExecPath = requestedWasmExecPath;
  if (wasmPath !== undefined && wasmPath !== null)
    assertString(wasmPath, "wasmPath");
  if (wasmExecPath !== undefined && wasmExecPath !== null)
    assertString(wasmExecPath, "wasmExecPath");
  if (activeRuntime || loading) {
    throw new Error(
      "Gepago WASM supports one instance per process; the existing runtime cannot be replaced",
    );
  }
  loading = true;
  let started = false;
  let namespace: RuntimeNamespace | undefined;
  const globals = snapshotGlobals(wasmExecGlobalNames);
  const [hadNamespace, previousNamespace] = [
    Object.prototype.hasOwnProperty.call(globalThis, "gepagoWasm"),
    globalsObject.gepagoWasm,
  ];
  try {
    if (globalsObject.gepagoWasm !== undefined) {
      throw new Error(
        "globalsObject.gepagoWasm is already in use; refusing to replace a process-global WASM API",
      );
    }
    wasmPath ??= process.env.GEPAGO_WASM ?? defaultWasmPath;
    wasmExecPath ??= process.env.GEPAGO_WASM_EXEC;
    const Go = await loadWasmExec(wasmExecPath);
    const go = new Go();
    const binary = await readFileAsync(wasmPath);
    const { instance } = await WebAssembly.instantiate(binary, go.importObject);
    const runPromise = go.run(instance);
    started = true;
    runPromise.catch(() => {});
    // wasm_exec.js needs its fs/path compatibility globals for the lifetime
    // of Go. Go itself is only a constructor injection point and is restored
    // immediately after the runtime starts.
    const [hadGo, previousGo] = globals.get("Go") ?? [false, undefined];
    restoreGlobal("Go", hadGo, previousGo);

    await new Promise((resolve) => setImmediate(resolve));
    namespace = globalsObject.gepagoWasm as RuntimeNamespace | undefined;
    if (
      namespace === undefined ||
      namespace === null ||
      typeof namespace.invoke !== "function" ||
      typeof namespace.start !== "function" ||
      typeof namespace.startAgent !== "function" ||
      typeof namespace.startCapabilities !== "function" ||
      typeof namespace.cancel !== "function" ||
      typeof namespace.version !== "string"
    ) {
      throw new Error(
        "Gepago WASM module did not register the expected gepagoWasm API",
      );
    }
    const invokeFunction = namespace.invoke.bind(namespace);
    const startFunction = namespace.start.bind(namespace);
    const startAgentFunction = namespace.startAgent.bind(namespace);
    const startCapabilitiesFunction =
      namespace.startCapabilities.bind(namespace);
    const cancelFunction = namespace.cancel.bind(namespace);
    const state: RuntimeState = {
      closed: false,
      namespace,
      hadNamespace,
      previousNamespace,
      pendingJobs: new Set<string>(),
    };
    activeRuntime = state;
    loading = false;

    async function predict(
      config: PredictConfig,
      model: PredictModel,
    ): Promise<PredictResult> {
      if (typeof model !== "function")
        throw new TypeError("predict model must be a function");
      const request = assertPredictConfig(config);
      let pending = assertPredictPending(invoke("predict.start", request));
      const jobID = pending.job_id;
      while (true) {
        let modelResult;
        try {
          modelResult = normalizeModelResult(
            await model(pending.host_request as unknown as PredictHostRequest),
          );
        } catch (error) {
          try {
            invoke("predict.cancel", { job_id: jobID });
          } catch {
            // Preserve the model failure. Cancellation is best effort at this boundary.
          }
          throw error;
        }
        const resumed = invoke("predict.resume", {
          job_id: jobID,
          request_id: String((pending.host_request as WireObject).request_id),
          result: modelResult,
        });
        if ((resumed as WireObject).status === "complete")
          return assertPredictComplete(resumed);
        pending = assertPredictPending(resumed);
      }
    }

    function cancelAsyncJob(jobID: string): void {
      if (!jobID) return;
      try {
        unwrapRuntimeCall(cancelFunction(jobID), "cancel");
      } catch {
        // A completion removes the job before invoking onComplete. Cancellation
        // after that point is intentionally best effort.
      }
    }

    async function runCapabilities(
      operation: string,
      payload: unknown,
      callbacks: Record<string, (...args: never[]) => unknown>,
      resultValidator: (value: unknown) => unknown,
    ): Promise<unknown> {
      if (state.closed) throw new Error("Gepago WASM instance is closed");
      const callbackObject = assertCapabilityCallbacks(callbacks, operation);
      const normalizedCallbacks: Record<
        string,
        (requestJSON: string) => Promise<unknown>
      > = {};
      for (const [name, callback] of Object.entries(callbackObject)) {
        normalizedCallbacks[name] = async (requestJSON: string) => {
          let request;
          try {
            request = JSON.parse(requestJSON);
          } catch (error) {
            throw new GepagoWasmError(
              `${operation} callback request was invalid JSON: ${error instanceof Error ? error.message : String(error)}`,
              operation,
            );
          }
          assertJSONValue(request, `${operation} ${name} callback request`);
          const result = await callback(request as never);
          if (
            result === undefined &&
            ["observer", "appendEvent", "saveState"].includes(name)
          )
            return null;
          assertJSONValue(result, `${operation} ${name} callback result`);
          return result;
        };
      }
      const payloadJSON = encodePayload(operation, payload);
      let jobID: string | undefined;
      let settled = false;
      const completion = new Promise<unknown>((resolve, reject) => {
        const onComplete = (
          completedJobID: string,
          resultJSON: string,
        ): void => {
          if (settled) return;
          settled = true;
          if (completedJobID !== jobID) {
            reject(
              new GepagoWasmError(
                `${operation} completed with an unexpected job id`,
                operation,
              ),
            );
            return;
          }
          try {
            resolve(resultValidator(parseEnvelope(operation, resultJSON)));
          } catch (error) {
            reject(error);
          }
        };
        try {
          jobID = unwrapRuntimeCall(
            startCapabilitiesFunction(
              operation,
              payloadJSON,
              normalizedCallbacks,
              onComplete,
            ),
            operation,
          );
          if (typeof jobID !== "string" || jobID === "")
            throw new GepagoWasmError(
              `${operation} start returned an invalid job id`,
              operation,
            );
          state.pendingJobs.add(jobID);
        } catch (error) {
          settled = true;
          reject(error);
        }
      });
      try {
        return await completion;
      } finally {
        if (jobID) state.pendingJobs.delete(jobID);
      }
    }

    async function runAgent(
      config: AgentConfig,
      chat: AgentChat,
      tool: AgentTool,
    ): Promise<RunResult> {
      if (state.closed) throw new Error("Gepago WASM instance is closed");
      if (typeof chat !== "function")
        throw new TypeError("runAgent chat must be a function");
      if (typeof tool !== "function")
        throw new TypeError("runAgent tool must be a function");
      const request = assertAgentConfig(config);
      const payloadJSON = encodePayload("agent", request);
      let jobID: string | undefined;
      let settled = false;
      let callbackFailure: unknown;
      const failCallback = (error: unknown): never => {
        if (callbackFailure === undefined) callbackFailure = error;
        cancelAsyncJob(jobID ?? "");
        throw error;
      };
      const chatFunction = async (
        requestJSON: string,
      ): Promise<ChatResponse> => {
        try {
          const hostRequest = assertAgentChatRequest(
            parseAgentJSON(requestJSON, "agent chat request"),
          );
          return normalizeAgentChatResponse(await chat(hostRequest));
        } catch (error) {
          return failCallback(error);
        }
      };
      const toolFunction = async (
        requestJSON: string,
      ): Promise<import("./public-types.ts").AgentToolResult | string> => {
        try {
          const hostRequest = assertAgentToolRequest(
            parseAgentJSON(requestJSON, "agent tool request"),
          );
          return normalizeAgentToolResult(await tool(hostRequest));
        } catch (error) {
          return failCallback(error);
        }
      };
      const completion = new Promise<RunResult>((resolve, reject) => {
        const onComplete = (
          completedJobID: string,
          resultJSON: string,
        ): void => {
          if (settled) return;
          settled = true;
          if (completedJobID !== jobID) {
            cancelAsyncJob(jobID ?? "");
            reject(
              new GepagoWasmError(
                "agent completed with an unexpected job id",
                "agent",
              ),
            );
            return;
          }
          try {
            resolve(assertAgentRunResult(parseEnvelope("agent", resultJSON)));
          } catch (error) {
            cancelAsyncJob(jobID ?? "");
            reject(callbackFailure ?? error);
          }
        };
        try {
          jobID = unwrapRuntimeCall(
            startAgentFunction(
              payloadJSON,
              chatFunction,
              toolFunction,
              onComplete,
            ),
            "agent",
          );
          if (typeof jobID !== "string" || jobID === "") {
            throw new GepagoWasmError(
              "agent start returned an invalid job id",
              "agent",
            );
          }
          state.pendingJobs.add(jobID);
        } catch (error) {
          settled = true;
          reject(
            error instanceof GepagoWasmError
              ? error
              : new GepagoWasmError(
                  `agent start failed: ${error instanceof Error ? error.message : String(error)}`,
                  "agent",
                  { cause: error },
                ),
          );
        }
      });
      try {
        return await completion;
      } finally {
        if (jobID) state.pendingJobs.delete(jobID);
      }
    }

    async function optimize(
      config: OptimizeConfig,
      callbacks: OptimizeCapabilities,
    ): Promise<{ state: OptimizationState }> {
      if (state.closed) throw new Error("Gepago WASM instance is closed");
      const request = assertOptimizeConfig(config);
      return runCapabilities(
        "optimize",
        request,
        callbacks as unknown as Record<string, (...args: never[]) => unknown>,
        (value) => {
          const result = assertPlainObject(value, "optimize result");
          const stateValue = assertPlainObject(
            result.state,
            "optimize result.state",
          );
          assertJSONValue(result, "optimize result");
          return { state: stateValue } as unknown as {
            state: OptimizationState;
          };
        },
      ) as Promise<{ state: OptimizationState }>;
    }

    async function runCompiled(
      programConfig: RunProgram,
      candidate: Candidate,
      inputs: Record<string, JSONValue>,
      model: PredictModel,
    ): Promise<PredictResult> {
      if (state.closed) throw new Error("Gepago WASM instance is closed");
      const program = assertCompileProgram(
        programConfig,
        "runCompiled program config",
      ) as unknown as RunProgram;
      const compiledCandidate = assertCompiledCandidate(candidate, program);
      assertPrediction(inputs, "runCompiled inputs");
      if (program.kind !== "pipeline") {
        // Normalize COT before applying the compiled instruction. Compile's COT
        // candidate already contains ToPredict's reasoning suffix; normalizing
        // after overlay would append that suffix a second time.
        return predict(
          {
            program: applyCandidateToProgram(
              normalizePredictProgram(
                programConfig,
                "runCompiled program config",
              ) as unknown as RunProgram,
              compiledCandidate,
              (text) => invoke("decodeDemos", { text }) as unknown as Demo[],
            ),
            inputs,
          } as PredictConfig,
          model,
        );
      }

      let stateInputs = { ...inputs };
      let last: Record<string, JSONValue> = {};
      let usage: Usage = {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
      };
      for (const step of program.steps) {
        const stepInputs: Record<string, JSONValue> = {};
        for (const key of pipelineStepReads(step)) {
          if (!Object.hasOwn(stateInputs, key)) {
            throw new Error(
              `pipeline step "${step.name}": required input "${key}" is missing`,
            );
          }
          stepInputs[key] = stateInputs[key];
        }
        const stepProgram = normalizePredictProgram(
          step.program,
          `runCompiled program ${step.name}`,
        ) as unknown as RunProgram;
        const result = await predict(
          {
            program: applyCandidateToProgram(
              stepProgram,
              candidateForStep(compiledCandidate, step.name),
              (text) => invoke("decodeDemos", { text }) as unknown as Demo[],
            ),
            inputs: stepInputs,
          } as PredictConfig,
          model,
        );
        usage = {
          prompt_tokens: usage.prompt_tokens + result.usage.prompt_tokens,
          completion_tokens:
            usage.completion_tokens + result.usage.completion_tokens,
          total_tokens: usage.total_tokens + result.usage.total_tokens,
        };
        last = pipelineStepWrites(step, result.prediction);
        Object.assign(stateInputs, last);
      }
      return { prediction: program.returnAll ? stateInputs : last, usage };
    }

    async function compile(
      config: CompileConfig,
      model: CompileModel,
      capabilities: CompileCapabilities = {},
    ): Promise<CompileResult> {
      if (state.closed) throw new Error("Gepago WASM instance is closed");
      if (typeof model !== "function")
        throw new TypeError("compile model must be a function");
      const request = assertCompileConfig(config);
      const capabilityObject =
        capabilities === undefined
          ? {}
          : assertPlainObject(capabilities, "compile capabilities");
      const useCapabilities =
        request.metric.kind === "host" ||
        Object.keys(capabilityObject).length > 0;
      if (useCapabilities) {
        return runCapabilities(
          "compile",
          request,
          { model, ...capabilityObject } as unknown as Record<
            string,
            (...args: never[]) => unknown
          >,
          assertCompileResult,
        ) as Promise<CompileResult>;
      }
      const payloadJSON = encodePayload("compile", request);
      let jobID: string | undefined;
      let settled = false;
      const modelFunction = async (
        requestJSON: string,
      ): Promise<CompileModelOutput> => {
        try {
          const hostRequest = parseCompileModelRequest(requestJSON);
          return normalizeCompileModelResult(await model(hostRequest));
        } catch (error) {
          // The Go bridge turns callback failures into completion errors. Cancel
          // immediately as well, so a pending model Promise cannot keep the
          // optimisation alive after a host callback failure.
          cancelAsyncJob(jobID ?? "");
          throw error;
        }
      };
      const completion = new Promise<CompileResult>((resolve, reject) => {
        const onComplete = (
          completedJobID: string,
          resultJSON: string,
        ): void => {
          if (settled) return;
          settled = true;
          if (completedJobID !== jobID) {
            cancelAsyncJob(jobID ?? "");
            reject(
              new GepagoWasmError(
                "compile completed with an unexpected job id",
                "compile",
              ),
            );
            return;
          }
          try {
            resolve(assertCompileResult(parseEnvelope("compile", resultJSON)));
          } catch (error) {
            // This also covers the Go {ok:false,error} envelope. The job is
            // normally already removed, so cancel is deliberately best effort.
            cancelAsyncJob(jobID ?? "");
            reject(error);
          }
        };
        try {
          jobID = unwrapRuntimeCall(
            startFunction("compile", payloadJSON, modelFunction, onComplete),
            "compile",
          );
          if (typeof jobID !== "string" || jobID === "") {
            throw new GepagoWasmError(
              "compile start returned an invalid job id",
              "compile",
            );
          }
          state.pendingJobs.add(jobID);
        } catch (error) {
          settled = true;
          reject(
            error instanceof GepagoWasmError
              ? error
              : new GepagoWasmError(
                  `compile start failed: ${error instanceof Error ? error.message : String(error)}`,
                  "compile",
                  { cause: error },
                ),
          );
        }
      });
      try {
        return await completion;
      } finally {
        if (jobID) state.pendingJobs.delete(jobID);
      }
    }

    function invoke(operation: string, payload: unknown): JSONValue {
      if (state.closed) throw new Error("Gepago WASM instance is closed");
      assertString(operation, "operation");
      if (operation.length === 0)
        throw new TypeError("operation must not be empty");
      const payloadJSON = encodePayload(operation, payload);
      let encoded;
      try {
        encoded = invokeFunction(operation, payloadJSON);
      } catch (error) {
        throw new GepagoWasmError(
          `${operation} invocation failed: ${error instanceof Error ? error.message : String(error)}`,
          operation,
          { cause: error },
        );
      }
      return parseEnvelope(operation, encoded) as JSONValue;
    }

    const api: LoadedRuntime = {
      invoke,
      predict,
      runAgent,
      optimize,
      runCompiled,
      compile,
      htmlReport(state, options = {}) {
        assertPlainObject(state, "htmlReport state");
        assertJSONValue(state, "htmlReport state");
        const reportOptions = assertPlainObject(options, "htmlReport options");
        if (reportOptions.title !== undefined)
          assertString(reportOptions.title, "htmlReport options.title");
        return String(invoke("htmlReport", { state, options: reportOptions }));
      },
      version() {
        return String(invoke("version", {}));
      },
      parsePrediction(raw) {
        assertString(raw, "raw");
        return invoke("parsePrediction", { raw }) as Record<string, JSONValue>;
      },
      decodeDemos(text) {
        assertString(text, "text");
        return invoke("decodeDemos", { text }) as unknown as Demo[];
      },
      encodeDemos(demos) {
        assertDemos(demos);
        return String(invoke("encodeDemos", { demos }));
      },
      exactMatch(expected, actual, options = {}) {
        assertPrediction(expected, "expected");
        assertPrediction(actual, "actual");
        assertMetricOptions(options, "exactMatch");
        return invoke(
          "metric",
          metricPayload("exactMatch", expected, actual, options),
        ) as unknown as MetricResult;
      },
      contains(expected, actual, options = {}) {
        assertPrediction(expected, "expected");
        assertPrediction(actual, "actual");
        assertMetricOptions(options, "contains");
        return invoke(
          "metric",
          metricPayload("contains", expected, actual, options),
        ) as unknown as MetricResult;
      },
      numericMatch(expected, actual, options = {}) {
        assertPrediction(expected, "expected");
        assertPrediction(actual, "actual");
        assertMetricOptions(options, "numericMatch");
        return invoke(
          "metric",
          metricPayload("numericMatch", expected, actual, options),
        ) as unknown as MetricResult;
      },
      classification(expected, actual, options = {}) {
        assertPrediction(expected, "expected");
        assertPrediction(actual, "actual");
        assertMetricOptions(options, "classification");
        return invoke(
          "metric",
          metricPayload("classification", expected, actual, options),
        ) as unknown as MetricResult;
      },
      buildPrompt(
        signatureOrRequest,
        instructionOrCandidate,
        demosOrInputs,
        inputs,
      ) {
        let request;
        if (arguments.length === 1) {
          request = assertPlainObject(
            signatureOrRequest,
            "buildPrompt request",
          );
          if (request.instruction !== undefined)
            assertString(request.instruction, "instruction");
          if (request.demos !== undefined) assertDemos(request.demos);
          request = {
            signature: request.signature,
            instruction: request.instruction ?? "",
            demos: request.demos ?? [],
            inputs: request.inputs,
          };
        } else if (arguments.length === 3) {
          request = candidateToPromptRequest(
            signatureOrRequest as Signature,
            instructionOrCandidate as Candidate,
            demosOrInputs as Record<string, JSONValue>,
          );
        } else if (arguments.length === 4) {
          request = {
            signature: signatureOrRequest,
            instruction: instructionOrCandidate,
            demos: demosOrInputs,
            inputs,
          };
        } else {
          throw new TypeError(
            "buildPrompt expects a request object or signature, instruction, demos, inputs",
          );
        }
        assertBuildPromptRequest(
          request.signature as Signature,
          request.instruction,
          request.demos,
          request.inputs,
        );
        return String(invoke("buildPrompt", request));
      },
      close() {
        if (state.closed) return;
        state.closed = true;
        for (const jobID of state.pendingJobs) cancelAsyncJob(jobID ?? "");
        // Go's current main blocks forever. This only invalidates this wrapper
        // and removes the namespace; it cannot stop the underlying Go runtime.
        if (globalsObject.gepagoWasm === state.namespace) {
          restoreGlobal(
            "gepagoWasm",
            state.hadNamespace,
            state.previousNamespace,
          );
        }
      },
    };
    return Object.freeze(api);
  } catch (error) {
    if (namespace && globalsObject.gepagoWasm === namespace) {
      restoreGlobal("gepagoWasm", hadNamespace, previousNamespace);
    }
    if (started) {
      const [hadGo, previousGo] = globals.get("Go") ?? [false, undefined];
      restoreGlobal("Go", hadGo, previousGo);
    } else {
      // No Go runtime owns these globals yet, so failed loading must not leak
      // the shim's compatibility globals either. Also invalidate the cached
      // constructor: the shim module itself was restored and must initialise
      // its globals again on a retry.
      restoreGlobals(globals);
      wasmExecConstructor = undefined;
      wasmExecLoad = undefined;
      wasmExecGeneration += 1;
    }
    loading = false;
    if (started && namespace)
      activeRuntime = {
        closed: false,
        namespace,
        hadNamespace,
        previousNamespace,
        pendingJobs: new Set<string>(),
      };
    throw error;
  }
}

/**
 * Create an isolated, asynchronous facade backed by a dedicated Node worker.
 * Each worker has its own JS global object, Go runtime, and WASM memory.
 */
export function createGepagoWasmWorker(
  options: CreateGepagoWasmWorkerOptions = {},
): Promise<AsyncGepagoWasm> {
  if (
    options === null ||
    typeof options !== "object" ||
    Array.isArray(options)
  ) {
    throw new TypeError("createGepagoWasmWorker options must be an object");
  }
  const { wasmPath, wasmExecPath } = options;
  if (wasmPath !== undefined && wasmPath !== null)
    assertString(wasmPath, "wasmPath");
  if (wasmExecPath !== undefined && wasmExecPath !== null)
    assertString(wasmExecPath, "wasmExecPath");
  const worker = new Worker(new URL("./node-worker.js", import.meta.url), {
    type: "module",
  } as import("node:worker_threads").WorkerOptions);
  const proxy = connectWorker(worker, {
    init: {
      ...(wasmPath === undefined ? {} : { wasmPath }),
      ...(wasmExecPath === undefined ? {} : { wasmExecPath }),
    },
    post: (message) => worker.postMessage(message),
    subscribe: (handler, onError) => {
      worker.on("message", handler);
      worker.on("error", onError);
      worker.on("exit", (code) => {
        onError(new Error(`Gepago WASM worker exited with code ${code}`));
      });
    },
    unsubscribe: (handler) => worker.off("message", handler),
    terminate: () => {
      void worker.terminate();
    },
  });
  return proxy.ready.then(
    () => proxy,
    (error) => {
      proxy.close();
      throw error;
    },
  );
}

export const __paths = Object.freeze({ defaultWasmPath, bundledWasmExecPath });

// Provider callbacks live outside the WASM payload. This Node entry point also
// exposes the Pi CLI adapter, which is intentionally not reachable from the
// browser entry point.
export * from "./providers.ts";
export * from "./storage.ts";
export * from "./artifacts.ts";
