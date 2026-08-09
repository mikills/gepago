import type {
  LoadedRuntime,
  RuntimeNamespace,
  WireObject,
  WorkerMessage,
} from "./internal-types.ts";

type Post = (message: WorkerMessage) => void;
type MessageHandler = (message: unknown) => void | Promise<void>;
type Load = (options: unknown) => Promise<LoadedRuntime>;
type HostCallback = {
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
};
type StructuredError = Error & { operation?: string; code?: string | number };

function serializedError(error: unknown): WireObject {
  const object =
    error !== null && typeof error === "object" && !Array.isArray(error)
      ? (error as WireObject)
      : undefined;
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

function restoreError(value: unknown): Error {
  const object =
    value !== null && typeof value === "object" ? (value as WireObject) : {};
  const error: StructuredError = new Error(
    typeof object.message === "string"
      ? object.message
      : "worker request failed",
  );
  error.name = typeof object.name === "string" ? object.name : "Error";
  if (typeof object.stack === "string") error.stack = object.stack;
  if (typeof object.operation === "string") error.operation = object.operation;
  if (typeof object.code === "string" || typeof object.code === "number")
    error.code = object.code;
  return error;
}

function object(value: unknown): WireObject | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as WireObject)
    : undefined;
}

/** Installs the message protocol used by both Node and browser workers. */
export function installWorkerRuntime(options: {
  post: Post;
  onMessage: (handler: MessageHandler) => void;
  load: Load;
}): void {
  const { post, onMessage, load } = options;
  let instance: LoadedRuntime | undefined;
  let callbackSequence = 0;
  const hostCallbacks = new Map<string, HostCallback>();

  const askHost = (
    callID: string,
    kind: string,
    payload: unknown,
  ): Promise<unknown> =>
    new Promise((resolve, reject) => {
      const callbackID = `callback-${++callbackSequence}`;
      hostCallbacks.set(callbackID, { resolve, reject });
      post({ type: "callback", callID, callbackID, kind, payload });
    });

  const runCall = async (message: WireObject): Promise<void> => {
    const id = typeof message.id === "string" ? message.id : "";
    const method = typeof message.method === "string" ? message.method : "";
    const args = Array.isArray(message.args) ? message.args : [];
    try {
      if (!instance) throw new Error("Gepago WASM worker is not ready");
      let value: unknown;
      switch (method) {
        case "predict":
          value = await instance.predict(
            args[0] as never,
            (request) =>
              askHost(id, "model", request) as Promise<
                string | import("./public-types.ts").PredictModelResult
              >,
          );
          break;
        case "compile": {
          const callbacks: Record<
            string,
            (request: never) => Promise<unknown>
          > = {};
          for (const name of Array.isArray(args[2]) ? args[2] : [])
            if (typeof name === "string" && name !== "model")
              callbacks[name] = (request) => askHost(id, name, request);
          value = await instance.compile(
            args[0] as never,
            (request) => askHost(id, "model", request) as never,
            callbacks as never,
          );
          break;
        }
        case "optimize": {
          const callbacks: Record<
            string,
            (request: never) => Promise<unknown>
          > = {};
          for (const name of Array.isArray(args[1]) ? args[1] : [])
            if (typeof name === "string")
              callbacks[name] = (request) => askHost(id, name, request);
          value = await instance.optimize(args[0] as never, callbacks as never);
          break;
        }
        case "runCompiled":
          value = await instance.runCompiled(
            args[0] as never,
            args[1] as never,
            args[2] as never,
            (request) => askHost(id, "model", request) as never,
          );
          break;
        case "runAgent":
          value = await instance.runAgent(
            args[0] as never,
            (request) => askHost(id, "chat", request) as never,
            (request) => askHost(id, "tool", request) as never,
          );
          break;
        case "invoke":
          value = await instance.invoke(args[0] as string, args[1] as never);
          break;
        default: {
          const callable = instance[method];
          if (typeof callable !== "function")
            throw new Error(`Unknown Gepago WASM method: ${method}`);
          value = await (callable as (...values: unknown[]) => unknown)(
            ...args,
          );
        }
      }
      post({ type: "result", id, ok: true, value });
    } catch (error) {
      post({ type: "result", id, ok: false, error: serializedError(error) });
    }
  };

  onMessage(async (message) => {
    const objectMessage = object(message);
    if (!objectMessage || typeof objectMessage.type !== "string") return;
    if (objectMessage.type === "init") {
      try {
        instance = await load(objectMessage.options ?? {});
        post({ type: "ready", version: instance.version() });
      } catch (error) {
        post({ type: "init-error", error: serializedError(error) });
      }
      return;
    }
    if (objectMessage.type === "callback-result") {
      const callbackID =
        typeof objectMessage.callbackID === "string"
          ? objectMessage.callbackID
          : "";
      const callback = hostCallbacks.get(callbackID);
      if (!callback) return;
      hostCallbacks.delete(callbackID);
      if (objectMessage.ok === true) callback.resolve(objectMessage.value);
      else callback.reject(restoreError(objectMessage.error));
      return;
    }
    if (objectMessage.type === "call") await runCall(objectMessage);
  });
}

export { serializedError };
