import type {
  AsyncGepagoWasm,
  CreateGepagoWasmWorkerOptions,
} from "./public-types.ts";
import { connectWorker } from "./worker-client.ts";

function asURL(value: unknown, name: string): string | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "string" && !(value instanceof URL))
    throw new TypeError(`${name} must be a string or URL`);
  return String(value);
}
export function createGepagoWasmWorker(
  options: CreateGepagoWasmWorkerOptions = {},
): Promise<AsyncGepagoWasm> {
  if (options === null || typeof options !== "object" || Array.isArray(options))
    throw new TypeError("createGepagoWasmWorker options must be an object");
  if (typeof Worker !== "function")
    throw new Error("Web Workers are not available in this environment");
  const workerURL =
    asURL(options.workerURL, "workerURL") ??
    new URL("./browser-worker.js", import.meta.url).toString();
  const wasmURL = asURL(options.wasmURL, "wasmURL");
  const wasmExecURL = asURL(options.wasmExecURL, "wasmExecURL");
  const worker = new Worker(workerURL, { type: "module" });
  const proxy = connectWorker(worker, {
    init: {
      ...(wasmURL === undefined ? {} : { wasmURL }),
      ...(wasmExecURL === undefined ? {} : { wasmExecURL }),
    },
    post: (message) => worker.postMessage(message),
    subscribe: (handler, onError) => {
      worker.addEventListener("message", (event) => handler(event.data));
      worker.addEventListener("error", onError);
    },
    unsubscribe: () => undefined,
    terminate: () => worker.terminate(),
  });
  return proxy.ready.then(
    () => proxy,
    (error) => {
      proxy.close();
      throw error;
    },
  );
}
export const createGepagoWasm = createGepagoWasmWorker;
export * from "./providers-browser.ts";
export * from "./storage-browser.ts";
export * from "./artifacts-core.ts";
