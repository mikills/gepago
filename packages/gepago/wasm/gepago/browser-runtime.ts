import type {
  AgentChat,
  AgentConfig,
  AgentTool,
  BuildPromptRequest,
  Candidate,
  ChatRequest,
  CompileCapabilities,
  CompileConfig,
  CompileModel,
  Demo,
  JSONValue,
  OptimizeCapabilities,
  OptimizeConfig,
  PredictConfig,
  PredictModel,
  PredictResult,
  RunProgram,
  RunResult,
  Signature,
  MetricResult,
} from "./public-types.ts";
import type {
  LoadedRuntime,
  RuntimeNamespace,
  WireObject,
} from "./internal-types.ts";

type BrowserOptions = { wasmURL?: string | URL; wasmExecURL?: string | URL };
function object(value: unknown): WireObject {
  if (value === null || typeof value !== "object" || Array.isArray(value))
    throw new Error("expected object");
  return value as WireObject;
}
function runtimeString(value: unknown, operation: string): string {
  if (typeof value === "string") return value;
  const result = object(value);
  if (result.__gepago_wasm_error === true) {
    throw new Error(
      typeof result.message === "string"
        ? result.message
        : `${operation} failed in the WASM runtime`,
    );
  }
  throw new Error(`${operation} returned an invalid runtime result`);
}
function parseEnvelope(operation: string, encoded: string): unknown {
  if (typeof encoded !== "string")
    throw new Error(`${operation} returned a non-string response`);
  const response = object(JSON.parse(encoded));
  if (response.ok !== true && response.ok !== false)
    throw new Error(`${operation} returned an invalid JSON envelope`);
  if (response.ok === false)
    throw new Error(
      typeof response.error === "string"
        ? response.error
        : `${operation} failed`,
    );
  if (!Object.hasOwn(response, "value"))
    throw new Error(`${operation} returned an invalid JSON envelope`);
  return response.value;
}
function jsonPayload(operation: string, value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch (error) {
    throw new TypeError(
      `could not encode ${operation} payload: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}
function as<T>(value: unknown): T {
  return value as T;
}

function createBrowserRuntime(namespace: RuntimeNamespace): LoadedRuntime {
  const invoke = (operation: string, payload: unknown): JSONValue =>
    as<JSONValue>(
      parseEnvelope(
        operation,
        runtimeString(
          namespace.invoke(operation, jsonPayload(operation, payload)),
          operation,
        ),
      ),
    );
  async function predict(
    config: PredictConfig,
    model: PredictModel,
  ): Promise<PredictResult> {
    let pending = object(invoke("predict.start", config));
    const jobID = String(pending.job_id);
    while (true) {
      let result: unknown;
      try {
        result = await model(
          as<import("./public-types.ts").PredictHostRequest>(
            pending.host_request,
          ),
        );
      } catch (error) {
        try {
          invoke("predict.cancel", { job_id: jobID });
        } catch {
          /* best effort */
        }
        throw error;
      }
      pending = object(
        invoke("predict.resume", {
          job_id: jobID,
          request_id: object(pending.host_request).request_id as JSONValue,
          result: as<JSONValue>(result),
        }),
      );
      if (pending.status === "complete") return as<PredictResult>(pending);
    }
  }
  async function runCapabilities(
    operation: string,
    payload: JSONValue,
    callbacks: Record<string, (...args: never[]) => unknown> = {},
  ): Promise<unknown> {
    if (typeof namespace.startCapabilities !== "function")
      throw new Error(
        "Gepago WASM module does not support capability-backed operations",
      );
    const normalized: Record<string, (request: string) => Promise<unknown>> =
      {};
    for (const [name, callback] of Object.entries(callbacks)) {
      if (typeof callback !== "function")
        throw new TypeError(`${operation} callback ${name} must be a function`);
      normalized[name] = async (requestJSON) =>
        callback(JSON.parse(requestJSON) as never);
    }
    return new Promise((resolve, reject) => {
      let jobID = "";
      const complete = (completedJobID: string, resultJSON: string): void => {
        if (completedJobID !== jobID) {
          reject(new Error(`${operation} completed with an unexpected job id`));
          return;
        }
        try {
          resolve(parseEnvelope(operation, resultJSON));
        } catch (error) {
          reject(error);
        }
      };
      try {
        jobID = runtimeString(
          namespace.startCapabilities(
            operation,
            jsonPayload(operation, payload),
            normalized,
            complete,
          ),
          operation,
        );
      } catch (error) {
        reject(error);
      }
    });
  }
  async function compile(
    config: CompileConfig,
    model: CompileModel,
    capabilities: CompileCapabilities = {},
  ): Promise<import("./public-types.ts").CompileResult> {
    const callbackObject = capabilities ?? {};
    const result =
      Object.keys(callbackObject).length > 0 || config.metric.kind === "host"
        ? await runCapabilities("compile", config as unknown as JSONValue, {
            model: model as unknown as (...args: never[]) => unknown,
            ...(callbackObject as Record<
              string,
              (...args: never[]) => unknown
            >),
          })
        : await new Promise<unknown>((resolve, reject) => {
            let jobID = "";
            try {
              jobID = runtimeString(
                namespace.start(
                  "compile",
                  jsonPayload("compile", config),
                  (requestJSON) => model(JSON.parse(requestJSON) as never),
                  (completedJobID, resultJSON) => {
                    if (completedJobID !== jobID)
                      reject(
                        new Error(
                          "compile completed with an unexpected job id",
                        ),
                      );
                    else
                      try {
                        resolve(parseEnvelope("compile", resultJSON));
                      } catch (error) {
                        reject(error);
                      }
                  },
                ),
                "compile",
              );
            } catch (error) {
              reject(error);
            }
          });
    return as<import("./public-types.ts").CompileResult>(result);
  }
  async function runAgent(
    config: AgentConfig,
    chat: AgentChat,
    tool: AgentTool,
  ): Promise<RunResult> {
    return as<RunResult>(
      await new Promise<unknown>((resolve, reject) => {
        let jobID = "";
        try {
          jobID = runtimeString(
            namespace.startAgent(
              jsonPayload("agent", config),
              (requestJSON) => chat(JSON.parse(requestJSON) as never),
              (requestJSON) => tool(JSON.parse(requestJSON) as never),
              (completedJobID, resultJSON) => {
                if (completedJobID !== jobID)
                  reject(
                    new Error("agent completed with an unexpected job id"),
                  );
                else
                  try {
                    resolve(parseEnvelope("agent", resultJSON));
                  } catch (error) {
                    reject(error);
                  }
              },
            ),
            "agent.run",
          );
        } catch (error) {
          reject(error);
        }
      }),
    );
  }
  async function optimize(
    config: OptimizeConfig,
    callbacks: OptimizeCapabilities = {} as OptimizeCapabilities,
  ): Promise<{ state: import("./public-types.ts").OptimizationState }> {
    return as<{ state: import("./public-types.ts").OptimizationState }>(
      await runCapabilities(
        "optimize",
        config as unknown as JSONValue,
        callbacks as unknown as Record<string, (...args: never[]) => unknown>,
      ),
    );
  }
  function applyCandidate(
    program: RunProgram,
    candidate: Candidate,
  ): RunProgram {
    const result = { ...program } as RunProgram & {
      instruction?: string;
      demos?: Demo[];
    };
    if (candidate.instruction !== undefined && program.kind !== "pipeline")
      result.instruction = candidate.instruction;
    if (candidate.demos !== undefined && program.kind !== "pipeline")
      result.demos = as<Demo[]>(
        invoke("decodeDemos", { text: candidate.demos }),
      );
    return result;
  }
  async function runCompiled(
    program: RunProgram,
    candidate: Candidate,
    inputs: Record<string, JSONValue>,
    model: PredictModel,
  ): Promise<PredictResult> {
    if (program.kind !== "pipeline")
      return predict(
        {
          program: applyCandidate(
            program,
            candidate,
          ) as PredictConfig["program"],
          inputs,
        },
        model,
      );
    // A pipeline runs in the Go module exactly as compile runs it, so both accept and reject
    // the same programmes and inputs and send the same prompts.
    return as<PredictResult>(
      await new Promise<unknown>((resolve, reject) => {
        let jobID = "";
        try {
          jobID = runtimeString(
            namespace.start(
              "pipeline.run",
              jsonPayload("runCompiled", { program, candidate, inputs }),
              (requestJSON) => model(JSON.parse(requestJSON) as never),
              (completedJobID, resultJSON) => {
                if (completedJobID !== jobID)
                  reject(
                    new Error("runCompiled completed with an unexpected job id"),
                  );
                else
                  try {
                    resolve(parseEnvelope("runCompiled", resultJSON));
                  } catch (error) {
                    reject(error);
                  }
              },
            ),
            "runCompiled",
          );
        } catch (error) {
          reject(error);
        }
      }),
    );
  }
  return {
    invoke,
    predict,
    compile,
    optimize,
    runAgent,
    runCompiled,
    version: () => String(invoke("version", {})),
    parsePrediction: (raw) =>
      as<Record<string, JSONValue>>(invoke("parsePrediction", { raw })),
    decodeDemos: (text) => as<Demo[]>(invoke("decodeDemos", { text })),
    encodeDemos: (demos) => String(invoke("encodeDemos", { demos })),
    exactMatch: (expected, actual, options = {}) =>
      as<MetricResult>(
        invoke("metric", {
          kind: "exactMatch",
          expected,
          actual,
          options: Array.isArray(options) ? { fields: options } : options,
        }),
      ),
    contains: (expected, actual, options = {}) =>
      as<MetricResult>(
        invoke("metric", {
          kind: "contains",
          expected,
          actual,
          options: Array.isArray(options) ? { fields: options } : options,
        }),
      ),
    numericMatch: (expected, actual, options = {}) =>
      as<MetricResult>(
        invoke("metric", {
          kind: "numericMatch",
          expected,
          actual,
          options: Array.isArray(options) ? { fields: options } : options,
        }),
      ),
    classification: (expected, actual, options = {}) =>
      as<MetricResult>(
        invoke("metric", { kind: "classification", expected, actual, options }),
      ),
    buildPrompt: ((...args: unknown[]) => {
      let request: WireObject;
      if (args.length === 1) request = object(args[0]);
      else if (args.length === 3)
        request = {
          signature: args[0] as JSONValue,
          instruction: object(args[1]).instruction ?? "",
          demos: [],
          inputs: args[2] as JSONValue,
        };
      else if (args.length === 4)
        request = {
          signature: args[0] as JSONValue,
          instruction: args[1] as JSONValue,
          demos: args[2] as JSONValue,
          inputs: args[3] as JSONValue,
        };
      else
        throw new TypeError(
          "buildPrompt expects a request object or signature, instruction, demos, inputs",
        );
      return String(invoke("buildPrompt", request as JSONValue));
    }) as LoadedRuntime["buildPrompt"],
    htmlReport: (state, options = {}) =>
      String(invoke("htmlReport", { state, options })),
    close: () => {},
  };
}

export async function loadBrowserRuntime(
  options: BrowserOptions = {},
): Promise<LoadedRuntime> {
  const wasmURL =
    options.wasmURL ?? new URL("./gepago.wasm", import.meta.url).href;
  const wasmExecURL =
    options.wasmExecURL ?? new URL("./wasm_exec.js", import.meta.url).href;
  await import(String(wasmExecURL));
  const global = globalThis as typeof globalThis & {
    Go?: new () => {
      importObject: WebAssembly.Imports;
      run(instance: WebAssembly.Instance): Promise<void>;
    };
    gepagoWasm?: RuntimeNamespace;
  };
  if (typeof global.Go !== "function")
    throw new Error("wasm_exec.js did not install Go");
  const response = await fetch(wasmURL);
  if (!response.ok)
    throw new Error(`could not fetch Gepago WASM (${response.status})`);
  const bytes = await response.arrayBuffer();
  const go = new global.Go();
  const { instance } = await WebAssembly.instantiate(bytes, go.importObject);
  void go.run(instance);
  await Promise.resolve();
  if (!global.gepagoWasm)
    throw new Error("Gepago WASM module did not register the expected API");
  return createBrowserRuntime(global.gepagoWasm);
}
