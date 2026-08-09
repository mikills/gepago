import type {
  AgentChat,
  AgentConfig,
  AgentTool,
  AsyncGepagoWasm,
  BuildPromptRequest,
  Candidate,
  CompileCapabilities,
  CompileConfig,
  CompileModel,
  Demo,
  JSONValue,
  MetricResult,
  OptimizeCapabilities,
  OptimizeConfig,
  PredictConfig,
  PredictModel,
  PredictResult,
  RunProgram,
  RunResult,
  Usage,
  ExactMatchOptions,
  ContainsOptions,
  NumericMatchOptions,
  ClassificationOptions,
  CompileResult,
} from "./public-types.ts";
import type { WireObject, WorkerMessage } from "./internal-types.ts";

type WorkerLike = unknown;
type Callback = (...args: never[]) => unknown;
type StructuredError = Error & { operation?: string; code?: string | number };
type Pending = {
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
  callbacks: Record<string, Callback>;
};
type ConnectOptions = {
  post: (message: WorkerMessage) => void;
  subscribe: (
    handler: (message: unknown) => void,
    onError: (error: unknown) => void,
  ) => void;
  unsubscribe?: (handler: (message: unknown) => void) => void;
  terminate?: () => void;
  init: unknown;
};

function asObject(value: unknown): WireObject | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as WireObject)
    : undefined;
}

function errorFrom(value: unknown): Error {
  const object = asObject(value);
  const error: StructuredError = new Error(
    typeof object?.message === "string"
      ? object.message
      : "Gepago WASM worker request failed",
  );
  error.name = typeof object?.name === "string" ? object.name : "Error";
  if (typeof object?.stack === "string") error.stack = object.stack;
  if (typeof object?.operation === "string") error.operation = object.operation;
  if (typeof object?.code === "string" || typeof object?.code === "number")
    error.code = object.code;
  return error;
}

function serializedError(error: unknown): WireObject {
  const object = asObject(error);
  const operation = object?.operation;
  const code = object?.code;
  return {
    name: error instanceof Error ? error.name : "Error",
    message: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack : undefined,
    ...(typeof operation === "string" ? { operation } : {}),
    ...(typeof code === "string" || typeof code === "number" ? { code } : {}),
  };
}

function assertCallback(
  callback: unknown,
  name: string,
): asserts callback is Callback {
  if (typeof callback !== "function")
    throw new TypeError(`${name} must be a function`);
}

/** Connects a Worker-like object to the shared Gepago worker protocol. */
export function connectWorker(
  _worker: WorkerLike,
  options: ConnectOptions,
): AsyncGepagoWasm {
  const { post, subscribe, unsubscribe, terminate, init } = options;
  let nextID = 0;
  let closed = false;
  let terminalError: Error | undefined;
  const pending = new Map<string, Pending>();
  let readyResolve!: (version: string) => void;
  let readyReject!: (error: unknown) => void;
  const ready = new Promise<string>((resolve, reject) => {
    readyResolve = resolve;
    readyReject = reject;
  });

  const failAll = (error: unknown): void => {
    if (closed) return;
    terminalError = error instanceof Error ? error : errorFrom(error);
    closed = true;
    for (const request of pending.values()) request.reject(terminalError);
    pending.clear();
    readyReject(terminalError);
    try {
      unsubscribe?.(onMessage);
    } catch {
      /* cleanup is best effort */
    }
    try {
      terminate?.();
    } catch {
      /* cleanup is best effort */
    }
  };

  const onMessage = (message: unknown): void => {
    const object = asObject(message);
    if (!object || typeof object.type !== "string") return;
    if (object.type === "ready") {
      if (typeof object.version === "string") readyResolve(object.version);
      return;
    }
    if (object.type === "init-error") {
      failAll(errorFrom(object.error));
      return;
    }
    if (object.type === "callback") {
      const callID = typeof object.callID === "string" ? object.callID : "";
      const request = pending.get(callID);
      if (!request) return;
      const kind = typeof object.kind === "string" ? object.kind : "";
      const callback = request.callbacks[kind];
      const callbackID =
        typeof object.callbackID === "string" ? object.callbackID : "";
      if (!callback) {
        post({
          type: "callback-result",
          callbackID,
          ok: false,
          error: serializedError(new Error(`No ${kind} callback was supplied`)),
        });
        return;
      }
      const reply = (value: unknown): void => {
        try {
          post({ type: "callback-result", callbackID, ok: true, value });
        } catch (error) {
          try {
            post({
              type: "callback-result",
              callbackID,
              ok: false,
              error: serializedError(
                new Error(
                  `callback result could not cross worker boundary: ${String(error)}`,
                ),
              ),
            });
          } catch {
            /* worker is already unavailable */
          }
        }
      };
      const reject = (error: unknown): void => {
        try {
          post({
            type: "callback-result",
            callbackID,
            ok: false,
            error: serializedError(error),
          });
        } catch {
          /* worker is already unavailable */
        }
      };
      Promise.resolve()
        .then(() => callback(object.payload as never))
        .then(reply, reject);
      return;
    }
    if (object.type === "result") {
      const id = typeof object.id === "string" ? object.id : "";
      const request = pending.get(id);
      if (!request) return;
      pending.delete(id);
      if (object.ok === true) request.resolve(object.value);
      else request.reject(errorFrom(object.error));
    }
  };

  subscribe(onMessage, (error) => failAll(errorFrom(error)));
  post({ type: "init", options: init });

  const call = (
    method: string,
    args: unknown[],
    callbacks: Record<string, Callback> = {},
  ): Promise<unknown> => {
    if (closed)
      return Promise.reject(
        terminalError ?? new Error("Gepago WASM worker is closed"),
      );
    const id = `request-${++nextID}`;
    return ready.then(() => {
      if (closed)
        return Promise.reject(
          terminalError ?? new Error("Gepago WASM worker is closed"),
        );
      return new Promise<unknown>((resolve, reject) => {
        pending.set(id, { resolve, reject, callbacks });
        try {
          post({ type: "call", id, method, args });
        } catch (error) {
          pending.delete(id);
          reject(error);
        }
      });
    });
  };

  const invoke = (operation: string, payload: JSONValue): Promise<JSONValue> =>
    call("invoke", [operation, payload]) as Promise<JSONValue>;
  const predict = (
    config: PredictConfig,
    model: PredictModel,
  ): Promise<PredictResult> => {
    assertCallback(model, "predict model");
    return call("predict", [config], { model }) as Promise<PredictResult>;
  };
  const runAgent = (
    config: AgentConfig,
    chat: AgentChat,
    tool: AgentTool,
  ): Promise<RunResult> => {
    assertCallback(chat, "runAgent chat");
    assertCallback(tool, "runAgent tool");
    return call("runAgent", [config], { chat, tool }) as Promise<RunResult>;
  };
  const runCompiled = (
    program: RunProgram,
    candidate: Candidate,
    inputs: Record<string, JSONValue>,
    model: PredictModel,
  ): Promise<PredictResult> => {
    assertCallback(model, "runCompiled model");
    return call("runCompiled", [program, candidate, inputs], {
      model,
    }) as Promise<PredictResult>;
  };
  const compile = (
    config: CompileConfig,
    model: CompileModel,
    capabilities: CompileCapabilities = {},
  ): Promise<CompileResult> => {
    assertCallback(model, "compile model");
    if (
      capabilities === null ||
      typeof capabilities !== "object" ||
      Array.isArray(capabilities)
    )
      throw new TypeError("compile capabilities must be an object");
    const callbacks: Record<string, Callback> = { model };
    for (const [name, callback] of Object.entries(capabilities)) {
      assertCallback(callback, `compile capabilities.${name}`);
      callbacks[name] = callback;
    }
    return call(
      "compile",
      [config, true, Object.keys(callbacks)],
      callbacks,
    ) as Promise<CompileResult>;
  };
  const optimize = (
    config: OptimizeConfig,
    callbacks: OptimizeCapabilities,
  ): Promise<{ state: import("./public-types.ts").OptimizationState }> => {
    if (
      callbacks === null ||
      typeof callbacks !== "object" ||
      Array.isArray(callbacks)
    )
      throw new TypeError("optimize callbacks must be an object");
    for (const [name, callback] of Object.entries(callbacks))
      assertCallback(callback, `optimize callbacks.${name}`);
    return call(
      "optimize",
      [config, Object.keys(callbacks)],
      callbacks as unknown as Record<string, Callback>,
    ) as Promise<{ state: import("./public-types.ts").OptimizationState }>;
  };

  const proxy: AsyncGepagoWasm = {
    ready,
    invoke,
    predict,
    runAgent,
    runCompiled,
    compile,
    optimize,
    htmlReport: (state, options = {}) =>
      call("htmlReport", [state, options]) as Promise<string>,
    version: () => call("version", []) as Promise<string>,
    parsePrediction: (raw) =>
      call("parsePrediction", [raw]) as Promise<Record<string, JSONValue>>,
    decodeDemos: (text) => call("decodeDemos", [text]) as Promise<Demo[]>,
    encodeDemos: (demos) => call("encodeDemos", [demos]) as Promise<string>,
    exactMatch: (expected, actual, options = {}) =>
      call("exactMatch", [expected, actual, options]) as Promise<MetricResult>,
    contains: (expected, actual, options = {}) =>
      call("contains", [expected, actual, options]) as Promise<MetricResult>,
    numericMatch: (expected, actual, options = {}) =>
      call("numericMatch", [
        expected,
        actual,
        options,
      ]) as Promise<MetricResult>,
    classification: (expected, actual, options = {}) =>
      call("classification", [
        expected,
        actual,
        options,
      ]) as Promise<MetricResult>,
    buildPrompt: ((...args: unknown[]) =>
      call(
        "buildPrompt",
        args,
      ) as Promise<string>) as AsyncGepagoWasm["buildPrompt"],
    close(): void {
      if (closed) return;
      terminalError = new Error("Gepago WASM worker is closed");
      closed = true;
      for (const request of pending.values()) request.reject(terminalError);
      pending.clear();
      readyReject(terminalError);
      try {
        unsubscribe?.(onMessage);
      } catch {
        /* cleanup is best effort */
      }
      try {
        terminate?.();
      } catch {
        /* cleanup is best effort */
      }
    },
  };
  return proxy;
}

export { errorFrom };
