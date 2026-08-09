import type { AgentChat, CompileModel, PredictModel } from "./index.d.ts";

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
  baseURL?: string;
  headers?: Record<string, string>;
  fetch?: ProviderFetch;
  fetchImpl?: ProviderFetch;
  maxTokens?: number;
  temperature?: number;
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
  options: HTTPProviderOptions,
): PredictModel & CompileModel;
export function createOpenAIChat(options: HTTPProviderOptions): AgentChat;
export function createOpenAIProvider(
  options: HTTPProviderOptions,
): ProviderCallbacks;
export const openaiModel: typeof createOpenAIModel;
export const openaiChat: typeof createOpenAIChat;
export const openai: typeof createOpenAIProvider;
export const createOpenAI: typeof createOpenAIProvider;

export interface ClaudeOptions extends HTTPProviderOptions {
  anthropicVersion?: string;
}
export function createClaudeModel(
  options: ClaudeOptions,
): PredictModel & CompileModel;
export function createClaudeChat(options: ClaudeOptions): AgentChat;
export function createClaudeProvider(options: ClaudeOptions): ProviderCallbacks;
export const claudeModel: typeof createClaudeModel;
export const claudeChat: typeof createClaudeChat;
export const claude: typeof createClaudeProvider;
export const createClaude: typeof createClaudeProvider;

export function createGoogleModel(
  options: HTTPProviderOptions,
): PredictModel & CompileModel;
export function createGoogleChat(options: HTTPProviderOptions): AgentChat;
export function createGoogleProvider(
  options: HTTPProviderOptions,
): ProviderCallbacks;
export const googleModel: typeof createGoogleModel;
export const googleChat: typeof createGoogleChat;
export const google: typeof createGoogleProvider;
export const createGoogle: typeof createGoogleProvider;
