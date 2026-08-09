import type {
  AgentChat,
  AgentToolRequest,
  ChatRequest,
  ChatResponse,
  CompileModel,
  Message,
  PredictModel,
  PredictHostRequest,
  PredictModelResult,
  Tool,
  Usage,
} from "./public-types.ts";
import type {
  ProviderFetch,
  HTTPProviderOptions,
  ProviderCallbacks,
} from "./provider-types.ts";
import type { WireObject } from "./internal-types.ts";

export type JsonResponse = WireObject;
export type ChatFunction = AgentChat;
export type ModelFunction = PredictModel & CompileModel;

function asObject(value: unknown): WireObject {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as WireObject)
    : {};
}
function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}
function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

export function requireFetch(
  candidate: ProviderFetch | undefined,
): ProviderFetch {
  const fetchImpl = candidate ?? globalThis.fetch;
  if (typeof fetchImpl !== "function")
    throw new TypeError(
      "a fetch implementation is required (globalThis.fetch is unavailable)",
    );
  return fetchImpl;
}
export function nonEmptyString(value: unknown, name: string): string {
  if (typeof value !== "string" || value.trim() === "")
    throw new TypeError(`${name} must be a non-empty string`);
  return value.trim();
}
export function optionalString(
  value: unknown,
  name: string,
): string | undefined {
  return value === undefined || value === null
    ? undefined
    : nonEmptyString(value, name);
}
export function headersWithDefaults(
  headers: Record<string, string> | undefined,
  defaults: Record<string, string | undefined>,
): Record<string, string> {
  if (
    headers !== undefined &&
    (headers === null || typeof headers !== "object" || Array.isArray(headers))
  )
    throw new TypeError("headers must be an object");
  const output: Record<string, string> = { ...(headers ?? {}) };
  for (const [name, value] of Object.entries(defaults)) {
    if (value === undefined) continue;
    if (
      !Object.keys(output).some(
        (key) => key.toLowerCase() === name.toLowerCase(),
      )
    )
      output[name] = value;
  }
  return output;
}
export function usageFrom(value: unknown = {}): Usage {
  const usage = asObject(value);
  const number = (candidate: unknown): number =>
    typeof candidate === "number" &&
    Number.isFinite(candidate) &&
    candidate >= 0
      ? Math.max(0, Math.trunc(candidate))
      : 0;
  const prompt_tokens = number(
    usage.prompt_tokens ??
      usage.promptTokens ??
      usage.input_tokens ??
      usage.promptTokenCount,
  );
  const completion_tokens = number(
    usage.completion_tokens ??
      usage.completionTokens ??
      usage.output_tokens ??
      usage.candidatesTokenCount,
  );
  const total = number(
    usage.total_tokens ?? usage.totalTokens ?? usage.totalTokenCount,
  );
  return {
    prompt_tokens,
    completion_tokens,
    total_tokens: total || prompt_tokens + completion_tokens,
  };
}
export function jsonObject(value: unknown, name: string): WireObject {
  if (value === null || typeof value !== "object" || Array.isArray(value))
    throw new TypeError(`${name} must be an object`);
  return value as WireObject;
}
export async function postJSON(
  fetchImpl: ProviderFetch,
  url: string,
  body: WireObject,
  headers: Record<string, string>,
  provider: string,
): Promise<WireObject> {
  let response;
  try {
    response = await fetchImpl(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
  } catch (error) {
    throw new Error(
      `${provider} request failed: ${error instanceof Error ? error.message : String(error)}`,
      { cause: error },
    );
  }
  let text = "";
  let parsed: unknown;
  if (typeof response.text === "function") text = await response.text();
  else if (typeof response.json === "function") parsed = await response.json();
  if (
    !response.ok &&
    !(
      response.status !== undefined &&
      response.status >= 200 &&
      response.status < 300
    )
  )
    throw new Error(
      `${provider} request failed with status ${response.status ?? "unknown"}: ${text.trim()}`,
    );
  if (parsed !== undefined) return jsonObject(parsed, `${provider} response`);
  try {
    return jsonObject(JSON.parse(text), `${provider} response`);
  } catch (error) {
    throw new Error(
      `${provider} returned invalid JSON: ${error instanceof Error ? error.message : String(error)}`,
      { cause: error },
    );
  }
}
export function textFromParts(parts: unknown): string {
  return asArray(parts)
    .map(asObject)
    .filter((part) => part.type === "text" && typeof part.text === "string")
    .map((part) => part.text as string)
    .join("\n");
}
export function responseMessage(
  content: string,
  toolCalls: Message["tool_calls"] = [],
): Message {
  const message: Message = { role: "assistant" };
  if (content) message.content = content;
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return message;
}
export function finishReason(
  value: unknown,
  hasTools = false,
): ChatResponse["finish_reason"] {
  if (hasTools || value === "tool_use" || value === "function_call")
    return "tool_calls";
  if (["length", "max_tokens", "MAX_TOKENS"].includes(String(value)))
    return "length";
  return "stop";
}
export function modelCallbackFromChat(
  chat: AgentChat,
  modelName: string,
): ModelFunction {
  return async (
    request:
      PredictHostRequest | import("./public-types.ts").CompileModelRequest,
  ): Promise<PredictModelResult | string> => {
    const object = jsonObject(request, "model request");
    const prompt = nonEmptyString(object.prompt, "model request.prompt");
    const response = await chat({
      model: modelName,
      messages: [
        {
          role: "system",
          content:
            "You are a precise optimisation assistant. Follow the requested output format exactly.",
        },
        { role: "user", content: prompt },
      ],
    });
    return { text: response.message.content ?? "", usage: response.usage };
  };
}
export function providerCallbacks(
  model: ModelFunction,
  chat: AgentChat,
): ProviderCallbacks {
  return Object.freeze({
    model,
    chat,
    modelCallback: model,
    chatCallback: chat,
    predict: model,
    compile: model,
    runAgent: chat,
  });
}
