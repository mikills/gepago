import type {
  AgentChat,
  AgentConfig,
  AgentTool,
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
  OptimizeCapabilities,
  OptimizeConfig,
  PredictConfig,
  PredictHostRequest,
  PredictModel,
  PredictResult,
  RunProgram,
  RunResult,
  Tool,
} from "./public-types.ts";

export type WireObject = Record<string, unknown>;
export type WireArray = unknown[];
export type AnyFunction = (...args: never[]) => unknown;
export type WorkerMessage = WireObject & {
  type?: string;
  [key: string]: unknown;
};
export type ErrorLike = { name?: unknown; message?: unknown; stack?: unknown };
export type RuntimeCallResult =
  string | (WireObject & { __gepago_wasm_error?: unknown });

export interface FileWorkerProtocol {
  post(message: WorkerMessage): void;
  subscribe(
    handler: (message: unknown) => void,
    onError: (error: unknown) => void,
  ): void;
  unsubscribe?(handler: (message: unknown) => void): void;
  terminate?(): void;
  init: unknown;
}

export interface RuntimeNamespace {
  invoke(operation: string, payload: string): RuntimeCallResult;
  start(
    operation: string,
    payload: string,
    callback: (request: string) => Promise<unknown> | unknown,
    complete: (jobID: string, result: string) => void,
  ): RuntimeCallResult;
  startAgent(
    payload: string,
    chat: (request: string) => Promise<unknown> | unknown,
    tool: (request: string) => Promise<unknown> | unknown,
    complete: (jobID: string, result: string) => void,
  ): RuntimeCallResult;
  startCapabilities(
    operation: string,
    payload: string,
    callbacks: Record<string, (request: string) => Promise<unknown>>,
    complete: (jobID: string, result: string) => void,
  ): RuntimeCallResult;
  cancel(jobID: string): RuntimeCallResult;
  version: string;
}

export interface LoadedRuntime {
  invoke(operation: string, payload: JSONValue): JSONValue;
  predict(config: PredictConfig, model: PredictModel): Promise<PredictResult>;
  runAgent(
    config: AgentConfig,
    chat: AgentChat,
    tool: AgentTool,
  ): Promise<RunResult>;
  runCompiled(
    program: RunProgram,
    candidate: Candidate,
    inputs: Record<string, JSONValue>,
    model: PredictModel,
  ): Promise<PredictResult>;
  compile(
    config: CompileConfig,
    model: CompileModel,
    capabilities?: CompileCapabilities,
  ): Promise<CompileResult>;
  optimize(
    config: OptimizeConfig,
    capabilities: OptimizeCapabilities,
  ): Promise<{ state: import("./public-types.ts").OptimizationState }>;
  htmlReport(
    state: import("./public-types.ts").OptimizationState,
    options?: { title?: string },
  ): string;
  parsePrediction(raw: string): Record<string, JSONValue>;
  decodeDemos(text: string): Demo[];
  encodeDemos(demos: Demo[]): string;
  exactMatch(
    expected: Record<string, JSONValue>,
    actual: Record<string, JSONValue>,
    options?: unknown,
  ): import("./public-types.ts").MetricResult;
  contains(
    expected: Record<string, JSONValue>,
    actual: Record<string, JSONValue>,
    options?: unknown,
  ): import("./public-types.ts").MetricResult;
  numericMatch(
    expected: Record<string, JSONValue>,
    actual: Record<string, JSONValue>,
    options?: unknown,
  ): import("./public-types.ts").MetricResult;
  classification(
    expected: Record<string, JSONValue>,
    actual: Record<string, JSONValue>,
    options?: unknown,
  ): import("./public-types.ts").MetricResult;
  buildPrompt(...args: unknown[]): string;
  close(): void;
  version(): string;
  [method: string]: unknown;
}

export interface HTTPResponseJSON extends WireObject {
  choices?: unknown;
  content?: unknown;
  candidates?: unknown;
  usage?: unknown;
  usageMetadata?: unknown;
  stop_reason?: unknown;
}

export interface ProviderRequest extends ChatRequest {
  [key: string]: unknown;
}

export interface ProviderResponse extends ChatResponse {}
export interface ProviderMessage extends Message {}
export interface ProviderTool extends Tool {}

export type ProviderBody = Record<string, unknown>;
export type ProviderPart = Record<string, unknown>;

export {
  AgentChat,
  ChatRequest,
  ChatResponse,
  CompileModel,
  Demo,
  JSONValue,
  Message,
  PredictHostRequest,
  PredictModel,
  PredictResult,
  Tool,
};
