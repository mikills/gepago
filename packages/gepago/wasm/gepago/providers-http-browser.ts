// Browser-safe HTTP provider helpers. This module intentionally has no Node imports.
import type {
  AgentChat,
  ChatRequest,
  ChatResponse,
  CompileModel,
  Message,
  PredictModel,
  Tool,
} from "./public-types.ts";
import type {
  HTTPProviderOptions,
  ProviderCallbacks,
  ProviderFetch,
} from "./provider-types.ts";
import type { WireObject } from "./internal-types.ts";
import {
  finishReasonField,
  headersWithDefaults,
  jsonObject,
  modelCallbackFromChat,
  nonEmptyString,
  optionalString,
  postJSON,
  providerCallbacks,
  requireFetch,
  responseMessage,
  textFromParts,
  usageFrom,
} from "./providers-http.ts";

type ProviderOptions = HTTPProviderOptions & {
  model: string;
  maxTokens: number;
};
type ModelFunction = PredictModel & CompileModel;
type Body = Record<string, unknown>;
function object(value: unknown): WireObject {
  return jsonObject(value, "response");
}
function array(value: unknown): WireObject[] {
  return Array.isArray(value) ? value.map(object) : [];
}
function apiResponsePart(value: unknown): WireObject {
  return value !== null && typeof value === "object"
    ? (value as WireObject)
    : {};
}
function providerOptions(
  options: HTTPProviderOptions,
  name: string,
): ProviderOptions {
  if (options === null || typeof options !== "object" || Array.isArray(options))
    throw new TypeError(`${name} options must be an object`);
  const model = nonEmptyString(options.model, `${name} model`);
  const maxTokens = options.maxTokens === undefined ? 512 : options.maxTokens;
  if (!Number.isSafeInteger(maxTokens) || maxTokens <= 0)
    throw new TypeError(`${name} maxTokens must be a positive safe integer`);
  if (
    options.temperature !== undefined &&
    (typeof options.temperature !== "number" ||
      !Number.isFinite(options.temperature))
  )
    throw new TypeError(`${name} temperature must be a finite number`);
  return { ...options, model, maxTokens };
}
function requestModel(
  request: ChatRequest,
  configured: string,
  name: string,
): string {
  return optionalString(request.model, `${name} request.model`) ?? configured;
}
function openAIMessage(message: Message): Body {
  return {
    role: message.role,
    ...(message.content === undefined ? {} : { content: message.content }),
    ...(message.tool_call_id === undefined
      ? {}
      : { tool_call_id: message.tool_call_id }),
    ...(message.tool_calls === undefined
      ? {}
      : {
          tool_calls: message.tool_calls.map((call) => ({
            id: call.id,
            type: "function",
            function: { name: call.name, arguments: call.arguments },
          })),
        }),
  };
}
function openAITools(tools: Tool[] | undefined): Body[] | undefined {
  return tools?.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      ...(tool.parameters === undefined ? {} : { parameters: tool.parameters }),
    },
  }));
}
function openAIResponse(response: WireObject): ChatResponse {
  const choices = array(response.choices);
  const choice = choices[0];
  const raw = apiResponsePart(choice?.message);
  if (!choice || !choice.message)
    throw new Error("OpenAI chat completion returned no choices");
  const calls = array(raw.tool_calls).map((call, index) => ({
    id: String(call.id ?? `call-${index + 1}`),
    name: String(apiResponsePart(call.function).name ?? ""),
    arguments:
      typeof apiResponsePart(call.function).arguments === "string"
        ? (apiResponsePart(call.function).arguments as string)
        : JSON.stringify(apiResponsePart(call.function).arguments ?? {}),
  }));
  return {
    message: responseMessage(
      typeof raw.content === "string" ? raw.content : "",
      calls,
    ),
    usage: usageFrom(response.usage),
    ...finishReasonField(choice.finish_reason, calls.length > 0),
  };
}
function openAIRequest(
  options: ProviderOptions,
  request: ChatRequest,
): WireObject {
  const messages = request.messages;
  const body: Body = {
    model: requestModel(request, options.model, "OpenAI"),
    messages: messages.map(openAIMessage),
  };
  if (request.tools?.length) body.tools = openAITools(request.tools);
  if (request.temperature !== undefined || options.temperature !== undefined)
    body.temperature = request.temperature ?? options.temperature;
  body.max_tokens = request.max_tokens ?? options.maxTokens;
  if (request.response_format)
    body.response_format = {
      type: "json_schema",
      json_schema: {
        name: request.response_format.name,
        schema: request.response_format.schema,
        strict: request.response_format.strict,
      },
    };
  return body;
}
export function createOpenAIChat(input: HTTPProviderOptions): AgentChat {
  const options = providerOptions(input, "OpenAI");
  const key = optionalString(options.apiKey, "apiKey");
  const fetchImpl = requireFetch(options.fetch ?? options.fetchImpl);
  const baseURL = (options.baseURL ?? "https://api.openai.com/v1").replace(
    /\/$/,
    "",
  );
  const headers = headersWithDefaults(options.headers, {
    "content-type": "application/json",
    ...(key ? { authorization: `Bearer ${key}` } : {}),
  });
  return async (request) =>
    openAIResponse(
      await postJSON(
        fetchImpl,
        `${baseURL}/chat/completions`,
        openAIRequest(options, request),
        headers,
        "OpenAI",
      ),
    );
}
export function createOpenAIModel(options: HTTPProviderOptions): ModelFunction {
  const normalized = providerOptions(options, "OpenAI");
  return modelCallbackFromChat(createOpenAIChat(normalized), normalized.model);
}
export function createOpenAIProvider(
  options: HTTPProviderOptions,
): ProviderCallbacks {
  return providerCallbacks(
    createOpenAIModel(options),
    createOpenAIChat(options),
  );
}

function anthropicMessages(messages: Message[]): Body[] {
  return messages
    .filter((message) => message.role !== "system")
    .map((message) => {
      if (message.role === "tool")
        return {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: message.tool_call_id,
              content: message.content ?? "",
            },
          ],
        };
      if (message.role === "assistant" && message.tool_calls?.length) {
        const content: Body[] = [];
        if (message.content)
          content.push({ type: "text", text: message.content });
        for (const call of message.tool_calls) {
          let input: unknown;
          try {
            input = JSON.parse(call.arguments || "{}");
          } catch {
            input = { arguments: call.arguments };
          }
          content.push({
            type: "tool_use",
            id: call.id,
            name: call.name,
            input,
          });
        }
        return { role: "assistant", content };
      }
      return {
        role: message.role === "assistant" ? "assistant" : "user",
        content: message.content ?? "",
      };
    });
}
function anthropicRequest(
  options: ProviderOptions & { anthropicVersion?: string },
  request: ChatRequest,
): WireObject {
  const system = request.messages
    .filter((message) => message.role === "system")
    .map((message) => message.content ?? "")
    .filter(Boolean)
    .join("\n");
  const body: Body = {
    model: requestModel(request, options.model, "Anthropic"),
    max_tokens: request.max_tokens ?? options.maxTokens,
    messages: anthropicMessages(request.messages),
  };
  if (system) body.system = system;
  if (request.temperature !== undefined || options.temperature !== undefined)
    body.temperature = request.temperature ?? options.temperature;
  if (request.tools?.length)
    body.tools = request.tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.parameters ?? { type: "object" },
    }));
  return body;
}
function anthropicResponse(response: WireObject): ChatResponse {
  const blocks = array(response.content);
  const text = textFromParts(blocks);
  const calls = blocks
    .filter((block) => block.type === "tool_use")
    .map((block) => ({
      id: String(block.id),
      name: String(block.name),
      arguments: JSON.stringify(block.input ?? {}),
    }));
  const usage = apiResponsePart(response.usage);
  return {
    message: responseMessage(text, calls),
    usage: usageFrom({
      input_tokens:
        Number(usage.input_tokens ?? 0) +
        Number(usage.cache_creation_input_tokens ?? 0) +
        Number(usage.cache_read_input_tokens ?? 0),
      output_tokens: usage.output_tokens,
    }),
    ...finishReasonField(response.stop_reason, calls.length > 0),
  };
}
export interface ClaudeOptions extends HTTPProviderOptions {
  anthropicVersion?: string;
}
export function createClaudeChat(input: ClaudeOptions): AgentChat {
  const options = {
    ...providerOptions(input, "Claude"),
    anthropicVersion: input.anthropicVersion,
  };
  const key = optionalString(options.apiKey, "apiKey");
  if (!key) throw new Error("API key is required (set apiKey)");
  const fetchImpl = requireFetch(options.fetch ?? options.fetchImpl);
  const baseURL = (options.baseURL ?? "https://api.anthropic.com").replace(
    /\/$/,
    "",
  );
  const headers = headersWithDefaults(options.headers, {
    "content-type": "application/json",
    "x-api-key": key,
    "anthropic-version": options.anthropicVersion ?? "2023-06-01",
  });
  return async (request) =>
    anthropicResponse(
      await postJSON(
        fetchImpl,
        `${baseURL}/v1/messages`,
        anthropicRequest(options, request),
        headers,
        "Anthropic",
      ),
    );
}
export function createClaudeModel(options: ClaudeOptions): ModelFunction {
  const normalized = providerOptions(options, "Claude");
  return modelCallbackFromChat(createClaudeChat(options), normalized.model);
}
export function createClaudeProvider(
  options: ClaudeOptions,
): ProviderCallbacks {
  return providerCallbacks(
    createClaudeModel(options),
    createClaudeChat(options),
  );
}

function googleFunctionName(name: string): string {
  let output = name
    .trim()
    .replace(/[^A-Za-z0-9_.-]/g, "_")
    .replace(/^_+|_+$/g, "");
  if (!output) output = "tool";
  if (!/^[A-Za-z_]/.test(output)) output = `_${output}`;
  return output.slice(0, 63);
}
function googleToolNames(tools: Tool[]): {
  toGemini: Map<string, string>;
  fromGemini: Map<string, string>;
} {
  const toGemini = new Map<string, string>();
  const fromGemini = new Map<string, string>();
  for (const tool of tools) {
    const original = tool.name;
    if (toGemini.has(original)) continue;
    const base = googleFunctionName(original);
    let normalized = base;
    let suffix = 2;
    while (fromGemini.has(normalized)) {
      const suffixText = `_${suffix++}`;
      normalized = `${base.slice(0, 63 - suffixText.length)}${suffixText}`;
    }
    toGemini.set(original, normalized);
    fromGemini.set(normalized, original);
  }
  return { toGemini, fromGemini };
}
function googleMessages(
  messages: Message[],
  names: { toGemini: Map<string, string> },
): Body[] {
  const normalize = (name: string): string =>
    names.toGemini.get(name) ?? googleFunctionName(name);
  return messages
    .filter((message) => message.role !== "system")
    .map((message, index, all) => {
      if (message.role === "tool") {
        let name = message.tool_call_id ?? "tool";
        for (let i = index - 1; i >= 0; i--) {
          const call = all[i]?.tool_calls?.find(
            (candidate) => candidate.id === message.tool_call_id,
          );
          if (call) {
            name = normalize(call.name);
            break;
          }
        }
        return {
          role: "user",
          parts: [
            {
              functionResponse: {
                name,
                response: { result: message.content ?? "" },
              },
            },
          ],
        };
      }
      const parts: Body[] = [];
      if (message.content) parts.push({ text: message.content });
      for (const call of message.tool_calls ?? []) {
        let args: unknown;
        try {
          args = JSON.parse(call.arguments || "{}");
        } catch {
          args = { arguments: call.arguments };
        }
        parts.push({ functionCall: { name: normalize(call.name), args } });
      }
      return { role: message.role === "assistant" ? "model" : "user", parts };
    })
    .filter(
      (message) => Array.isArray(message.parts) && message.parts.length > 0,
    );
}
function googleRequest(
  options: ProviderOptions,
  request: ChatRequest,
): { body: WireObject; toolNames: Map<string, string> } {
  const names = googleToolNames(request.tools ?? []);
  const systemParts = request.messages
    .filter((message) => message.role === "system" && message.content)
    .map((message) => ({ text: message.content }));
  const body: Body = {
    ...(systemParts.length
      ? { systemInstruction: { parts: systemParts } }
      : {}),
    contents: googleMessages(request.messages, names),
  };
  if (request.tools?.length)
    body.tools = [
      {
        functionDeclarations: request.tools.map((tool) => ({
          name: names.toGemini.get(tool.name),
          description: tool.description,
          parameters: tool.parameters,
        })),
      },
    ];
  const generationConfig: Body = {
    maxOutputTokens: request.max_tokens ?? options.maxTokens,
  };
  body.generationConfig = generationConfig;
  if (request.temperature !== undefined || options.temperature !== undefined)
    generationConfig.temperature = request.temperature ?? options.temperature;
  if (request.response_format) {
    generationConfig.responseMimeType = "application/json";
    generationConfig.responseSchema = request.response_format.schema;
  }
  return { body, toolNames: names.fromGemini };
}
function googleResponse(
  response: WireObject,
  toolNames: Map<string, string>,
): ChatResponse {
  const candidate = array(response.candidates)[0];
  if (!candidate)
    throw new Error("Gemini generateContent returned no candidates");
  const content = apiResponsePart(candidate.content);
  let text = "";
  const calls: Message["tool_calls"] = [];
  for (const part of array(content.parts)) {
    if (typeof part.text === "string") text += part.text;
    const functionCall = apiResponsePart(part.functionCall);
    if (part.functionCall)
      calls.push({
        id: String(functionCall.name),
        name:
          toolNames.get(String(functionCall.name)) ?? String(functionCall.name),
        arguments: JSON.stringify(functionCall.args ?? {}),
      });
  }
  return {
    message: responseMessage(text, calls),
    usage: usageFrom(response.usageMetadata),
    ...finishReasonField(candidate.finishReason, calls.length > 0),
  };
}
export function createGoogleChat(input: HTTPProviderOptions): AgentChat {
  const options = providerOptions(input, "Google");
  const key = optionalString(options.apiKey, "apiKey");
  const fetchImpl = requireFetch(options.fetch ?? options.fetchImpl);
  const baseURL = (
    options.baseURL ?? "https://generativelanguage.googleapis.com"
  ).replace(/\/$/, "");
  const headers = headersWithDefaults(options.headers, {
    "content-type": "application/json",
  });
  return async (request) => {
    const { body, toolNames } = googleRequest(options, request);
    const url = new URL(
      `${baseURL}/v1beta/models/${encodeURIComponent(requestModel(request, options.model, "Gemini"))}:generateContent`,
    );
    if (key) url.searchParams.set("key", key);
    return googleResponse(
      await postJSON(fetchImpl, url.toString(), body, headers, "Gemini"),
      toolNames,
    );
  };
}
export function createGoogleModel(options: HTTPProviderOptions): ModelFunction {
  const normalized = providerOptions(options, "Google");
  return modelCallbackFromChat(createGoogleChat(normalized), normalized.model);
}
export function createGoogleProvider(
  options: HTTPProviderOptions,
): ProviderCallbacks {
  return providerCallbacks(
    createGoogleModel(options),
    createGoogleChat(options),
  );
}
export const openai = createOpenAIProvider;
export const claude = createClaudeProvider;
export const google = createGoogleProvider;
export const openaiModel = createOpenAIModel;
export const openaiChat = createOpenAIChat;
export const claudeModel = createClaudeModel;
export const claudeChat = createClaudeChat;
export const googleModel = createGoogleModel;
export const googleChat = createGoogleChat;
export const createOpenAI = createOpenAIProvider;
export const createClaude = createClaudeProvider;
export const createGoogle = createGoogleProvider;
