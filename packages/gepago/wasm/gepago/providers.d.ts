import type {
  AgentChat,
  CompileModel,
  PredictModel,
  ChatResponse,
} from "./index.d.ts";

export interface ProviderFetchResponse {
  ok?: boolean;
  status?: number;
  text?: () => Promise<string>;
  json?: () => Promise<unknown>;
}

export type ProviderFetch = (
  input: string | URL,
  init?: { method?: string; headers?: Record<string, string>; body?: string },
) => Promise<ProviderFetchResponse>;

export interface HTTPProviderOptions {
  model: string;
  apiKey?: string;
  /** Node only: explicit environment variable name. */
  apiKeyEnv?: string;
  baseURL?: string;
  headers?: Record<string, string>;
  fetch?: ProviderFetch;
  /** Alias for fetch, useful when sharing a config with other clients. */
  fetchImpl?: ProviderFetch;
  maxTokens?: number;
  temperature?: number;
}

export interface NodeProviderOptions extends HTTPProviderOptions {
  /** Node only; defaults to process.env. */
  env?: Record<string, string | undefined>;
}

export interface ProviderCallbacks {
  model: PredictModel & CompileModel;
  chat: AgentChat;
  predict: PredictModel & CompileModel;
  compile: PredictModel & CompileModel;
  runAgent: AgentChat;
  modelCallback: PredictModel & CompileModel;
  chatCallback: AgentChat;
}

export function createOpenAIModel(
  options: HTTPProviderOptions | NodeProviderOptions,
): PredictModel & CompileModel;
export function createOpenAIChat(
  options: HTTPProviderOptions | NodeProviderOptions,
): AgentChat;
export function createOpenAIProvider(
  options: HTTPProviderOptions | NodeProviderOptions,
): ProviderCallbacks;
export const openaiModel: typeof createOpenAIModel;
export const openaiChat: typeof createOpenAIChat;
export const openai: typeof createOpenAIProvider;
export const createOpenAI: typeof createOpenAIProvider;

export interface ClaudeOptions extends HTTPProviderOptions {
  anthropicVersion?: string;
}
export function createClaudeModel(
  options: ClaudeOptions | NodeProviderOptions,
): PredictModel & CompileModel;
export function createClaudeChat(
  options: ClaudeOptions | NodeProviderOptions,
): AgentChat;
export function createClaudeProvider(
  options: ClaudeOptions | NodeProviderOptions,
): ProviderCallbacks;
export const claudeModel: typeof createClaudeModel;
export const claudeChat: typeof createClaudeChat;
export const claude: typeof createClaudeProvider;
export const createClaude: typeof createClaudeProvider;

export function createGoogleModel(
  options: HTTPProviderOptions | NodeProviderOptions,
): PredictModel & CompileModel;
export function createGoogleChat(
  options: HTTPProviderOptions | NodeProviderOptions,
): AgentChat;
export function createGoogleProvider(
  options: HTTPProviderOptions | NodeProviderOptions,
): ProviderCallbacks;
export const googleModel: typeof createGoogleModel;
export const googleChat: typeof createGoogleChat;
export const google: typeof createGoogleProvider;
export const createGoogle: typeof createGoogleProvider;

export interface PiOptions {
  command?: string;
  model?: string;
  extraArgs?: string[];
  cwd?: string;
  env?: Record<string, string | undefined>;
  maxBuffer?: number;
  /** Maximum time to allow the subprocess to run. Defaults to two minutes. */
  timeoutMs?: number;
  /** Time between SIGTERM and SIGKILL. */
  killTimeoutMs?: number;
  signal?: AbortSignal;
}
export function createPiModel(options?: PiOptions): PredictModel & CompileModel;
export function createPiChat(options?: PiOptions): AgentChat;
export function createPiProvider(options?: PiOptions): ProviderCallbacks;
export const piModel: typeof createPiModel;
export const piChat: typeof createPiChat;
export const pi: typeof createPiProvider;
export const createPi: typeof createPiProvider;
