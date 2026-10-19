export type JSONPrimitive = string | number | boolean | null;
export type JSONValue = JSONPrimitive | JSONObject | JSONValue[];
export interface JSONObject {
  [key: string]: JSONValue;
}

export type Prediction = Record<string, JSONValue>;
export type Candidate = Record<string, string>;

export type MessageRole = "system" | "user" | "assistant" | "tool";
export type FinishReason = "stop" | "length" | "tool_calls" | "content_filter";
export type StopReason =
  | "complete"
  | "length"
  | "max_turns"
  | "tool_error"
  | "content_filter";

export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

export interface Message {
  role: MessageRole;
  content?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

export interface Tool {
  name: string;
  description: string;
  parameters?: JSONObject;
}

export interface JSONSchemaResponseFormat {
  name: string;
  schema: JSONObject;
  strict: boolean;
}

export interface ChatRequest {
  model: string;
  messages: Message[];
  tools?: Tool[];
  temperature?: number;
  max_tokens?: number;
  response_format?: JSONSchemaResponseFormat;
}

export interface ChatResponse {
  message: Message;
  usage: Usage;
  /** Why the model stopped; omitted when the provider does not report it. */
  finish_reason?: FinishReason;
}

export interface AgentConfigBase {
  name: string;
  model: string;
  systemPrompt?: string;
  maxTurns?: number;
  temperature?: number;
  maxTokens?: number;
  tools?: Tool[];
  run_id: string;
  memory?: JSONObject;
}

export type AgentConfig =
  | (AgentConfigBase & { messages: Message[]; initial_messages?: never })
  /** Go bridge alias for messages. */
  | (AgentConfigBase & { messages?: never; initial_messages: Message[] });

export interface AgentToolRequest {
  name: string;
  call_id: string;
  arguments: string;
  run_id: string;
  memory: JSONObject;
}

export interface AgentToolResult {
  output: string;
  memory?: JSONObject;
}

export type ToolRequest = AgentToolRequest;
export type ToolResult = AgentToolResult;
export type AgentChat = (
  request: ChatRequest,
) => ChatResponse | Promise<ChatResponse>;
export type AgentTool = (
  request: AgentToolRequest,
) => string | AgentToolResult | Promise<string | AgentToolResult>;

/** Candidate components supported by a Predict or ChainOfThought program. */
export interface PredictCandidate {
  instruction?: string;
  demos?: string;
}

/** Candidate components for a pipeline are scoped as `step.component`. */
export type PipelineCandidate = Candidate;

export interface Demo {
  inputs: Prediction;
  outputs: Prediction;
  rationale?: string;
}

export interface MetricResult {
  score: number;
  objective_scores?: Record<string, number>;
  feedback?: string;
  details?: JSONObject;
}

export interface ExactMatchOptions {
  fields?: string[];
  caseInsensitive?: boolean;
  trimSpace?: boolean;
}

export interface ContainsOptions {
  fields?: string[];
  caseInsensitive?: boolean;
}

export interface NumericMatchOptions {
  fields?: string[];
  tolerance?: number;
}

export interface ClassificationOptions {
  field?: string;
  positiveLabel?: string;
  negativeLabel?: string;
  trimSpace?: boolean;
  ignoreCase?: boolean;
}

export type MetricOptions =
  | ExactMatchOptions
  | ContainsOptions
  | NumericMatchOptions
  | ClassificationOptions
  | string[];

export interface Field {
  name: string;
  description?: string;
  type?: string;
  schema?: Record<string, unknown>;
  optional?: boolean;
}

export interface Signature {
  name?: string;
  description?: string;
  inputs: Field[];
  outputs: Field[];
}

export interface BuildPromptRequest {
  signature: Signature;
  instruction?: string;
  demos?: Demo[];
  inputs: Prediction;
}

export type ProgramKind = "predict" | "chainOfThought" | "pipeline";

export interface PredictProgram {
  kind?: "predict";
  signature: Signature;
  instruction?: string;
  demos?: Demo[];
  maxParseRetries?: number;
}

export interface ChainOfThoughtProgram {
  kind: "chainOfThought";
  signature: Signature;
  instruction?: string;
  demos?: Demo[];
  maxParseRetries?: number;
  reasoningField?: string;
}

export interface PipelineStepProgram {
  kind?: "predict" | "chainOfThought";
  signature: Signature;
  instruction?: string;
  demos?: Demo[];
  maxParseRetries?: number;
  reasoningField?: string;
  inputKeys?: string[];
  outputPrefix?: string;
}

export interface PipelineStep {
  name: string;
  program: PipelineStepProgram;
}

export interface PipelineProgram {
  kind: "pipeline";
  steps: PipelineStep[];
  returnAll?: boolean;
}

export type PredictConfigProgram = PredictProgram | ChainOfThoughtProgram;
export type RunProgram = PredictConfigProgram | PipelineProgram;
export type CompileProgram =
  PredictProgram | ChainOfThoughtProgram | PipelineProgram;

export interface PredictConfig {
  program: PredictConfigProgram;
  inputs: Prediction;
}

export interface CompileExample {
  id: string;
  input: Prediction;
  expected: Prediction;
}

export type CompileMetricKind =
  | "exactMatch"
  | "contains"
  | "numericMatch"
  | "classification"
  | "llmJudge"
  | "host";

export interface CompileMetricOptions {
  fields?: string[];
  caseInsensitive?: boolean;
  trimSpace?: boolean;
  tolerance?: number;
  field?: string;
  positiveLabel?: string;
  negativeLabel?: string;
  ignoreCase?: boolean;
  rubric?: string;
  role?: "judge";
  maxScore?: number;
  parseRetry?: boolean;
}

export interface CompileMetric extends CompileMetricOptions {
  kind: CompileMetricKind;
  /** Options may also be supplied as a nested object for wire compatibility. */
  options?: CompileMetricOptions;
}

export interface CompileConfig {
  program: CompileProgram;
  trainset: CompileExample[];
  valset?: CompileExample[];
  metric: CompileMetric;
  objective?: string;
  components?: string[];
  maxMetricCalls?: number;
  minibatchSize?: number;
  seed?: number;
  acceptance?: "strict" | "equal";
  resume?: boolean;
}

export interface CapabilityMetricRequest {
  kind: "metric.score";
  expected: Prediction;
  actual: Prediction;
}
export interface CapabilityEvaluatorRequest {
  kind: "optimizer.evaluate";
  candidate: Candidate;
  examples: JSONValue[];
  capture_traces: boolean;
}
export interface CapabilityDatasetRequest {
  kind: "optimizer.dataset.build";
  candidate: Candidate;
  evaluation: JSONValue;
  components: string[];
}
export interface CapabilityProposerRequest {
  kind: "optimizer.propose";
  candidate: Candidate;
  dataset: JSONValue;
  components: string[];
}
export interface CapabilityMergeRequest {
  kind: "optimizer.merge";
  left: JSONValue;
  right: JSONValue;
  components: string[];
}
export interface CapabilityObserverRequest {
  kind: "optimizer.observe";
  event: JSONValue;
}
export interface CapabilityPersistenceRequest {
  kind:
    | "optimizer.persistence.append_event"
    | "optimizer.persistence.save_state"
    | "optimizer.persistence.load_state";
  event?: JSONValue;
  state?: JSONValue;
}
export type CapabilityCallback = (
  request: JSONValue,
) => JSONValue | void | Promise<JSONValue | void>;
export interface CompileCapabilities {
  metric?: (
    request: CapabilityMetricRequest,
  ) => MetricResult | Promise<MetricResult>;
  observer?: (request: CapabilityObserverRequest) => void | Promise<void>;
  appendEvent?: (request: CapabilityPersistenceRequest) => void | Promise<void>;
  saveState?: (request: CapabilityPersistenceRequest) => void | Promise<void>;
  /** Return null when no resumable state has been saved yet. */
  loadState?: (
    request: CapabilityPersistenceRequest,
  ) => JSONValue | Promise<JSONValue>;
}
export interface OptimizeConfig {
  seedCandidate: Candidate;
  trainset: Array<{ id: string; input: JSONValue }>;
  valset?: Array<{ id: string; input: JSONValue }>;
  objective?: string;
  background?: string;
  promptTemplates?: Record<string, string>;
  components?: string[];
  maxMetricCalls: number;
  minibatchSize: number;
  seed?: number;
  acceptance?: "strict" | "equal";
  candidateSelector?: "pareto" | "best" | "topK" | "epsilonGreedy";
  componentSelector?: "all" | "roundRobin";
  topK?: number;
  epsilon?: number;
  enableMerge?: boolean;
  resume?: boolean;
}
export interface OptimizeCapabilities {
  evaluator: (
    request: CapabilityEvaluatorRequest,
  ) => JSONValue | Promise<JSONValue>;
  proposer: (
    request: CapabilityProposerRequest,
  ) => JSONValue | Promise<JSONValue>;
  datasetBuilder?: (
    request: CapabilityDatasetRequest,
  ) => JSONValue | Promise<JSONValue>;
  mergeProposer?: (
    request: CapabilityMergeRequest,
  ) => JSONValue | Promise<JSONValue>;
  observer?: (request: CapabilityObserverRequest) => void | Promise<void>;
  appendEvent?: (request: CapabilityPersistenceRequest) => void | Promise<void>;
  saveState?: (request: CapabilityPersistenceRequest) => void | Promise<void>;
  /** Return null when no resumable state has been saved yet. */
  loadState?: (
    request: CapabilityPersistenceRequest,
  ) => JSONValue | Promise<JSONValue>;
}

export interface UsageLedger {
  runs: number;
  turns: number;
  model_calls: number;
  tool_calls: number;
  metric_calls: number;
  input_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  cached_input_tokens: number;
  cache_creation_tokens: number;
  total_tokens: number;
  total_cost_usd: number;
  total_duration: number;
}

export interface UsageSpan {
  id: string;
  parent_id?: string;
  trace_id: string;
  kind: string;
  name: string;
  started_at: string;
  ended_at: string;
  duration: number;
  model?: string;
  usage: Usage;
  cost_usd?: number;
  err?: string;
  attributes?: JSONObject;
}

export interface ToolCallRecord {
  turn: number;
  call_id: string;
  name: string;
  arguments: string;
  output?: string;
  err?: string;
  duration: number;
}

export type AgentToolCallRecord = ToolCallRecord;

export interface RunResult {
  final: Message;
  transcript: Message[];
  usage: Usage;
  ledger: UsageLedger;
  spans: UsageSpan[];
  stop_reason: StopReason;
  turns: number;
  tool_calls: ToolCallRecord[];
}

export interface CandidateRecord {
  id: string;
  candidate: Candidate;
  parent_ids: string[];
  discovered_at: string;
  train_score: number;
  validation_score: number;
  scores_by_example: Record<string, number>;
  objective_scores_by_example?: Record<string, Record<string, number>>;
}

export interface ProposalMetadata {
  component: string;
  prompt: string;
  raw_output: string;
  parsed: string;
}

export interface ProposalRecord {
  id: string;
  iteration: number;
  parent_ids: string[];
  components: string[];
  metadata?: ProposalMetadata[];
  minibatch_before_sum: number;
  minibatch_after_sum: number;
  accepted: boolean;
  created_at: string;
}

export interface OptimizationState {
  run_id: string;
  started_at: string;
  ended_at: string;
  candidates: CandidateRecord[];
  proposal_records: ProposalRecord[];
  lessons?: string[];
  best_candidate_id: string;
  frontier_ids: string[];
  metric_calls: number;
  iterations: number;
  ledger: UsageLedger;
  spans: UsageSpan[];
}

export interface CompileResult {
  candidate: Candidate;
  state: OptimizationState;
}

export type CompileCandidateResult = CompileResult;

export type CompileModelRole = "task" | "reflector" | "judge" | "repair";

export interface CompileModelRequest {
  kind: "language_model.generate";
  role: CompileModelRole;
  prompt: string;
  attempt?: number;
}

export interface CompileModelResult {
  text: string;
  usage?: Usage;
  /**
   * Why the model stopped; "length" marks a truncated reply that is not repaired. Omit it,
   * or pass null or "", when the provider does not report one.
   */
  finishReason?: FinishReason | "" | null;
  /** The model the provider resolved the request to. */
  model?: string;
}

export type CompileModelOutput = string | CompileModelResult;
export type CompileModel = (
  request: CompileModelRequest,
) => CompileModelOutput | Promise<CompileModelOutput>;

export interface PredictHostRequest {
  request_id: string;
  kind: "language_model.generate";
  role: "task";
  prompt: string;
  attempt: number;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  /** Part of completion_tokens spent on hidden reasoning. */
  reasoning_tokens?: number;
  /** Part of prompt_tokens read from a prompt cache. */
  cached_input_tokens?: number;
  /** Part of prompt_tokens written to a prompt cache. */
  cache_creation_tokens?: number;
}

export type PredictModelUsage = Usage;

export interface PredictModelResult {
  text: string;
  usage?: PredictModelUsage;
  /**
   * Why the model stopped; "length" fails the prediction without a repair request. Omit it,
   * or pass null or "", when the provider does not report one.
   */
  finishReason?: FinishReason | "" | null;
  /** The model the provider resolved the request to. */
  model?: string;
}

export interface PredictResult {
  prediction: Prediction;
  usage: PredictModelUsage;
}

export type PredictModel = (
  request: PredictHostRequest,
) => string | PredictModelResult | Promise<string | PredictModelResult>;

export class GepagoWasmError extends Error {
  readonly operation: string;
}

export interface GepagoWasm {
  /** Low-level JSON-boundary operation; prefer the typed helpers below. */
  invoke(operation: string, payload: JSONValue): JSONValue;
  predict(config: PredictConfig, model: PredictModel): Promise<PredictResult>;
  /** Runs the declarative Go agent loop with host-provided async callbacks. */
  runAgent(
    config: AgentConfig,
    chat: AgentChat,
    tool: AgentTool,
  ): Promise<RunResult>;
  /** Runs a Compile candidate for Predict, ChainOfThought, or Pipeline programs. */
  runCompiled(
    programConfig: RunProgram,
    candidate: PredictCandidate | PipelineCandidate,
    inputs: Prediction,
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
  ): Promise<{ state: OptimizationState }>;
  htmlReport(state: OptimizationState, options?: { title?: string }): string;
  version(): string;
  parsePrediction(raw: string): Prediction;
  decodeDemos(text: string): Demo[];
  encodeDemos(demos: Demo[]): string;
  exactMatch(
    expected: Prediction,
    actual: Prediction,
    options?: ExactMatchOptions | string[],
  ): MetricResult;
  contains(
    expected: Prediction,
    actual: Prediction,
    options?: ContainsOptions | string[],
  ): MetricResult;
  numericMatch(
    expected: Prediction,
    actual: Prediction,
    options?: NumericMatchOptions | string[],
  ): MetricResult;
  classification(
    expected: Prediction,
    actual: Prediction,
    options?: ClassificationOptions,
  ): MetricResult;
  buildPrompt(request: BuildPromptRequest): string;
  buildPrompt(
    signature: Signature,
    instruction: string,
    demos: Demo[],
    inputs: Prediction,
  ): string;
  /** Convenience form for a Candidate containing instruction and encoded demos. */
  buildPrompt(
    signature: Signature,
    candidate: Candidate,
    inputs: Prediction,
  ): string;
  /** Invalidates this wrapper and removes the process-global namespace. */
  close(): void;
}

/**
 * Worker-backed API. Every operation is asynchronous because it crosses a
 * worker boundary. Use this for independent instances in one process/tab.
 */
export interface AsyncGepagoWasm {
  readonly ready: Promise<string>;
  invoke(operation: string, payload: JSONValue): Promise<JSONValue>;
  predict(config: PredictConfig, model: PredictModel): Promise<PredictResult>;
  runAgent(
    config: AgentConfig,
    chat: AgentChat,
    tool: AgentTool,
  ): Promise<RunResult>;
  runCompiled(
    programConfig: RunProgram,
    candidate: PredictCandidate | PipelineCandidate,
    inputs: Prediction,
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
  ): Promise<{ state: OptimizationState }>;
  htmlReport(
    state: OptimizationState,
    options?: { title?: string },
  ): Promise<string>;
  version(): Promise<string>;
  parsePrediction(raw: string): Promise<Prediction>;
  decodeDemos(text: string): Promise<Demo[]>;
  encodeDemos(demos: Demo[]): Promise<string>;
  exactMatch(
    expected: Prediction,
    actual: Prediction,
    options?: ExactMatchOptions | string[],
  ): Promise<MetricResult>;
  contains(
    expected: Prediction,
    actual: Prediction,
    options?: ContainsOptions | string[],
  ): Promise<MetricResult>;
  numericMatch(
    expected: Prediction,
    actual: Prediction,
    options?: NumericMatchOptions | string[],
  ): Promise<MetricResult>;
  classification(
    expected: Prediction,
    actual: Prediction,
    options?: ClassificationOptions,
  ): Promise<MetricResult>;
  buildPrompt(request: BuildPromptRequest): Promise<string>;
  buildPrompt(
    signature: Signature,
    instruction: string,
    demos: Demo[],
    inputs: Prediction,
  ): Promise<string>;
  buildPrompt(
    signature: Signature,
    candidate: Candidate,
    inputs: Prediction,
  ): Promise<string>;
  close(): void;
}

export interface CreateGepagoWasmWorkerOptions {
  /** Node filesystem path to the WASM binary. */
  wasmPath?: string;
  /** Node filesystem path to wasm_exec.js. */
  wasmExecPath?: string;
  /** Browser Worker URL (browser entry only). */
  workerURL?: string | URL;
  /** Browser URL for the WASM binary (browser entry only). */
  wasmURL?: string | URL;
  /** Browser URL for wasm_exec.js (browser entry only). */
  wasmExecURL?: string | URL;
}

export function createGepagoWasmWorker(
  options?: CreateGepagoWasmWorkerOptions,
): Promise<AsyncGepagoWasm>;

export interface CreateGepagoWasmOptions {
  /** Defaults to GEPAGO_WASM or this package's dist/gepago.wasm. */
  wasmPath?: string;
  /** Defaults to the bundled runtime shim, then GEPAGO_WASM_EXEC/GOROOT. */
  wasmExecPath?: string;
}

export function createGepagoWasm(
  options?: CreateGepagoWasmOptions,
): Promise<GepagoWasm>;

/** Resolved package paths, exposed for diagnostics and development tooling. */
export const __paths: Readonly<{
  defaultWasmPath: string;
  bundledWasmExecPath: string;
}>;

export * from "./providers.js";
export * from "./storage.js";
export * from "./artifacts.js";
