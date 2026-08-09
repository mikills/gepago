import { spawn, type ChildProcess } from "node:child_process";
import type {
  AgentChat,
  CompileModel,
  CompileModelRequest,
  PredictHostRequest,
  PredictModel,
  PredictModelResult,
  ChatRequest,
} from "./public-types.ts";
import type {
  HTTPProviderOptions,
  NodeProviderOptions,
  PiOptions,
  ProviderCallbacks,
} from "./provider-types.ts";
import {
  createOpenAIChat as browserOpenAIChat,
  createOpenAIModel as browserOpenAIModel,
  createOpenAIProvider as browserOpenAIProvider,
  createClaudeChat as browserClaudeChat,
  createClaudeModel as browserClaudeModel,
  createClaudeProvider as browserClaudeProvider,
  createGoogleChat as browserGoogleChat,
  createGoogleModel as browserGoogleModel,
  createGoogleProvider as browserGoogleProvider,
} from "./providers-http-browser.ts";
import {
  nonEmptyString,
  usageFrom,
  responseMessage,
  jsonObject,
  providerCallbacks,
} from "./providers-http.ts";

function nodeEnv(
  options: NodeProviderOptions,
  names: string[],
): string | undefined {
  const env = options.env ?? process.env;
  if (env === null || typeof env !== "object")
    throw new TypeError("env must be an object");
  const candidates =
    options.apiKeyEnv === undefined
      ? names
      : [nonEmptyString(options.apiKeyEnv, "apiKeyEnv")];
  for (const name of candidates) {
    const value = env[name];
    if (typeof value === "string" && value.trim() !== "") return value.trim();
  }
  return undefined;
}
function withKey(
  options: HTTPProviderOptions | NodeProviderOptions,
  names: string[],
  required: boolean,
): HTTPProviderOptions {
  const key = options.apiKey ?? nodeEnv(options as NodeProviderOptions, names);
  if (required && !key)
    throw new Error(
      `API key is required (set apiKey or ${options.apiKeyEnv ?? names.join(" / ")})`,
    );
  return { ...options, ...(key ? { apiKey: key } : {}) };
}
export function createOpenAIChat(
  options: HTTPProviderOptions | NodeProviderOptions,
): AgentChat {
  return browserOpenAIChat(
    withKey(options, ["OPENAI_API_KEY"], !options.baseURL),
  );
}
export function createOpenAIModel(
  options: HTTPProviderOptions | NodeProviderOptions,
): PredictModel & CompileModel {
  return browserOpenAIModel(
    withKey(options, ["OPENAI_API_KEY"], !options.baseURL),
  );
}
export function createOpenAIProvider(
  options: HTTPProviderOptions | NodeProviderOptions,
): ProviderCallbacks {
  return browserOpenAIProvider(
    withKey(options, ["OPENAI_API_KEY"], !options.baseURL),
  );
}
export function createClaudeChat(
  options: HTTPProviderOptions | NodeProviderOptions,
): AgentChat {
  return browserClaudeChat(
    withKey(
      options,
      ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
      true,
    ) as HTTPProviderOptions & { anthropicVersion?: string },
  );
}
export function createClaudeModel(
  options: HTTPProviderOptions | NodeProviderOptions,
): PredictModel & CompileModel {
  return browserClaudeModel(
    withKey(
      options,
      ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
      true,
    ) as HTTPProviderOptions & { anthropicVersion?: string },
  );
}
export function createClaudeProvider(
  options: HTTPProviderOptions | NodeProviderOptions,
): ProviderCallbacks {
  return browserClaudeProvider(
    withKey(
      options,
      ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
      true,
    ) as HTTPProviderOptions & { anthropicVersion?: string },
  );
}
export function createGoogleChat(
  options: HTTPProviderOptions | NodeProviderOptions,
): AgentChat {
  return browserGoogleChat(
    withKey(options, ["GOOGLE_API_KEY", "GEMINI_API_KEY"], false),
  );
}
export function createGoogleModel(
  options: HTTPProviderOptions | NodeProviderOptions,
): PredictModel & CompileModel {
  return browserGoogleModel(
    withKey(options, ["GOOGLE_API_KEY", "GEMINI_API_KEY"], false),
  );
}
export function createGoogleProvider(
  options: HTTPProviderOptions | NodeProviderOptions,
): ProviderCallbacks {
  return browserGoogleProvider(
    withKey(options, ["GOOGLE_API_KEY", "GEMINI_API_KEY"], false),
  );
}

type PiResult = { stdout: string; stderr: string };
function piArguments(options: PiOptions, prompt: string): string[] {
  const args = [
    "--no-tools",
    "--no-context-files",
    "--no-skills",
    "--no-prompt-templates",
    "--no-session",
  ];
  if (options.model) args.push("--model", options.model);
  args.push(...(options.extraArgs ?? []), "-p", prompt);
  return args;
}
function piSignal(child: ChildProcess, signal: NodeJS.Signals): boolean {
  if (process.platform !== "win32" && child.pid !== undefined) {
    try {
      process.kill(-child.pid, signal);
      return true;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
      return false;
    }
  }
  try {
    return child.kill(signal);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ESRCH") return false;
    throw error;
  }
}
async function runPi(options: PiOptions, prompt: string): Promise<string> {
  const command = options.command ?? "pi",
    maxBuffer = options.maxBuffer ?? 10 * 1024 * 1024,
    timeoutMs = options.timeoutMs ?? 120_000,
    killTimeoutMs = options.killTimeoutMs ?? 1_000,
    signal = options.signal;
  if (!Number.isSafeInteger(maxBuffer) || maxBuffer <= 0)
    throw new TypeError("Pi maxBuffer must be a positive safe integer");
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 0)
    throw new TypeError("Pi timeoutMs must be a non-negative safe integer");
  if (!Number.isSafeInteger(killTimeoutMs) || killTimeoutMs < 0)
    throw new TypeError("Pi killTimeoutMs must be a non-negative safe integer");
  if (signal?.aborted) throw new Error("Pi command aborted");
  try {
    const result = await new Promise<PiResult>((resolve, reject) => {
      let stdout = "",
        stderr = "",
        outputBytes = 0,
        closeCode: number | null = null,
        closeSignal: NodeJS.Signals | null = null,
        failure: Error | undefined,
        stopping = false,
        killTimer: NodeJS.Timeout | undefined,
        timeoutTimer: NodeJS.Timeout | undefined,
        settled = false;
      const child = spawn(command, piArguments(options, prompt), {
        stdio: ["ignore", "pipe", "pipe"],
        detached: process.platform !== "win32",
        ...(options.cwd === undefined ? {} : { cwd: options.cwd }),
        ...(options.env === undefined ? {} : { env: options.env }),
      });
      const stop = (error?: Error): void => {
        if (error && !failure) failure = error;
        if (stopping) return;
        stopping = true;
        try {
          piSignal(child, "SIGTERM");
        } catch (killError) {
          if (!failure)
            failure =
              killError instanceof Error
                ? killError
                : new Error(String(killError));
        }
        killTimer = setTimeout(() => {
          try {
            piSignal(child, "SIGKILL");
          } catch (killError) {
            if (!failure)
              failure =
                killError instanceof Error
                  ? killError
                  : new Error(String(killError));
          }
        }, killTimeoutMs);
        killTimer.unref();
      };
      const append = (stream: "stdout" | "stderr", chunk: unknown): void => {
        if (failure) return;
        const text = String(chunk),
          bytes = Buffer.byteLength(text);
        if (outputBytes + bytes > maxBuffer) {
          failure = new Error(
            `Pi output exceeded maxBuffer (${maxBuffer} bytes)`,
          );
          stop();
          return;
        }
        outputBytes += bytes;
        if (stream === "stdout") stdout += text;
        else stderr += text;
      };
      const onAbort = (): void => stop(new Error("Pi command aborted"));
      const finish = (): void => {
        if (settled) return;
        settled = true;
        if (timeoutTimer) clearTimeout(timeoutTimer);
        if (killTimer) clearTimeout(killTimer);
        signal?.removeEventListener("abort", onAbort);
        if (!failure && closeCode === 0) resolve({ stdout, stderr });
        else
          reject(
            failure ??
              new Error(
                `Pi exited with ${closeCode ?? `signal ${closeSignal}`}${stderr ? `: ${stderr.trim()}` : ""}`,
              ),
          );
      };
      child.stdout?.setEncoding("utf8");
      child.stderr?.setEncoding("utf8");
      child.stdout?.on("data", (chunk: unknown) => append("stdout", chunk));
      child.stderr?.on("data", (chunk: unknown) => append("stderr", chunk));
      child.once("error", (error) => stop(error));
      child.once("close", (code, childSignal) => {
        closeCode = code;
        closeSignal = childSignal;
        finish();
      });
      signal?.addEventListener("abort", onAbort, { once: true });
      timeoutTimer = setTimeout(
        () => stop(new Error(`Pi command timed out after ${timeoutMs}ms`)),
        timeoutMs,
      );
      timeoutTimer.unref();
    });
    const text = result.stdout.trim();
    if (!text) throw new Error("Pi returned empty output");
    return text;
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`Pi command failed: ${detail}`, { cause: error });
  }
}
export function createPiModel(
  input: PiOptions = {},
): PredictModel & CompileModel {
  if (input === null || typeof input !== "object" || Array.isArray(input))
    throw new TypeError("Pi options must be an object");
  const options: PiOptions = { ...input };
  if (options.command !== undefined)
    options.command = nonEmptyString(options.command, "Pi command");
  if (options.model !== undefined)
    options.model = nonEmptyString(options.model, "Pi model");
  if (
    options.extraArgs !== undefined &&
    (!Array.isArray(options.extraArgs) ||
      options.extraArgs.some((arg) => typeof arg !== "string"))
  )
    throw new TypeError("Pi extraArgs must be an array of strings");
  return async (
    request: PredictHostRequest | CompileModelRequest,
  ): Promise<PredictModelResult> => {
    const object = jsonObject(request, "Pi model request");
    const prompt = nonEmptyString(object.prompt, "Pi model request.prompt");
    return { text: await runPi(options, prompt), usage: usageFrom() };
  };
}
export function createPiChat(input: PiOptions = {}): AgentChat {
  const model = createPiModel(input);
  return async (request: ChatRequest) => {
    const prompt = request.messages
      .map((message) => `[${message.role}] ${message.content ?? ""}`)
      .join("\n");
    const result = await model({
      kind: "language_model.generate",
      role: "task",
      prompt,
      request_id: "pi-chat",
      attempt: 1,
    });
    if (typeof result === "string")
      return {
        message: responseMessage(result),
        usage: usageFrom(),
        finish_reason: "stop",
      };
    return {
      message: responseMessage(result.text),
      usage: result.usage ?? usageFrom(),
      finish_reason: "stop",
    };
  };
}
export function createPiProvider(options: PiOptions = {}): ProviderCallbacks {
  return {
    ...providerCallbacks(createPiModel(options), createPiChat(options)),
  };
}
export const openai = createOpenAIProvider;
export const claude = createClaudeProvider;
export const google = createGoogleProvider;
export const pi = createPiProvider;
export const openaiModel = createOpenAIModel;
export const openaiChat = createOpenAIChat;
export const claudeModel = createClaudeModel;
export const claudeChat = createClaudeChat;
export const googleModel = createGoogleModel;
export const googleChat = createGoogleChat;
export const piModel = createPiModel;
export const piChat = createPiChat;
export const createOpenAI = createOpenAIProvider;
export const createClaude = createClaudeProvider;
export const createGoogle = createGoogleProvider;
export const createPi = createPiProvider;
