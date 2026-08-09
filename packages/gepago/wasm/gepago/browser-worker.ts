import { installWorkerRuntime } from "./worker-runtime.ts";
import { loadBrowserRuntime } from "./browser-runtime.ts";

installWorkerRuntime({
  post: (message) => globalThis.postMessage(message),
  onMessage: (handler) =>
    globalThis.addEventListener("message", (event) => handler(event.data)),
  load: (options) =>
    loadBrowserRuntime(options as Parameters<typeof loadBrowserRuntime>[0]),
});
