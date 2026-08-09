import type {
  AgentChat,
  CompileModel,
  PredictModel,
  ProviderFetchResponse,
} from "./public-types.ts";
export type { AgentChat, CompileModel, PredictModel };
export type ProviderFetch = (
  input: string | URL,
  init?: { method?: string; headers?: Record<string, string>; body?: string },
) => Promise<ProviderFetchResponse>;
export interface HTTPProviderOptions {
  model: string;
  apiKey?: string;
  apiKeyEnv?: string;
  baseURL?: string;
  headers?: Record<string, string>;
  fetch?: ProviderFetch;
  fetchImpl?: ProviderFetch;
  maxTokens?: number;
  temperature?: number;
}
export interface NodeProviderOptions extends HTTPProviderOptions {
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
export interface ClaudeOptions extends HTTPProviderOptions {
  anthropicVersion?: string;
}
export interface PiOptions {
  command?: string;
  model?: string;
  extraArgs?: string[];
  cwd?: string;
  env?: Record<string, string | undefined>;
  maxBuffer?: number;
  timeoutMs?: number;
  killTimeoutMs?: number;
  signal?: AbortSignal;
}
