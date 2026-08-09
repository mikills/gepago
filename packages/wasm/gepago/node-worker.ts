import { parentPort, type MessagePort } from "node:worker_threads";
import { createGepagoWasm } from "./index.ts";
import { installWorkerRuntime } from "./worker-runtime.ts";

if (!parentPort) throw new Error("Gepago WASM Node worker requires parentPort");
const port: MessagePort = parentPort;
installWorkerRuntime({
  post: (message) => port.postMessage(message),
  onMessage: (handler) => {
    port.on("message", handler);
  },
  load: (options) =>
    createGepagoWasm(options as Parameters<typeof createGepagoWasm>[0]),
});
