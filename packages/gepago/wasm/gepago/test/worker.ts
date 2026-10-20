import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { createGepagoWasmWorker } from "@mikills/wasm/gepago";
import type { Signature } from "@mikills/wasm/gepago";
import { connectWorker } from "../worker-client.ts";
import { serializedError } from "../worker-runtime.ts";

const signature = {
  inputs: [{ name: "text" }],
  outputs: [{ name: "label" }],
} satisfies Signature;

test("worker factory provides independent WASM instances", async () => {
  const [first, second] = await Promise.all([
    createGepagoWasmWorker(),
    createGepagoWasmWorker(),
  ]);
  try {
    assert.deepEqual(await Promise.all([first.version(), second.version()]), [
      "3",
      "3",
    ]);
    const [a, b] = await Promise.all([
      first.predict(
        { program: { signature }, inputs: { text: "first" } },
        async (request) => ({
          text: JSON.stringify({
            label: request.prompt.includes("first") ? "first" : "wrong",
          }),
        }),
      ),
      second.predict(
        { program: { signature }, inputs: { text: "second" } },
        async (request) => ({
          text: JSON.stringify({
            label: request.prompt.includes("second") ? "second" : "wrong",
          }),
        }),
      ),
    ]);
    assert.equal(a.prediction.label, "first");
    assert.equal(b.prediction.label, "second");
    assert.equal(
      await first
        .parsePrediction('{"label":"one"}')
        .then((value) => value.label),
      "one",
    );
    const compiled = await first.compile(
      {
        program: { signature, instruction: "classify" },
        trainset: [
          { id: "one", input: { text: "first" }, expected: { label: "first" } },
        ],
        metric: { kind: "exactMatch", fields: ["label"] },
        components: ["instruction"],
        maxMetricCalls: 1,
        seed: 1,
      },
      async () => ({ text: '{"label":"first"}' }),
    );
    assert.equal(typeof compiled.candidate.instruction, "string");

    const agent = await first.runAgent(
      {
        name: "worker-agent",
        model: "host-model",
        messages: [{ role: "user", content: "hello" }],
        run_id: "worker-agent",
      },
      async () => ({
        message: { role: "assistant", content: "hello" },
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        finish_reason: "stop",
      }),
      async () => "unused",
    );
    assert.equal(agent.final.content, "hello");
  } finally {
    first.close();
    second.close();
  }

  // Terminating a worker releases its Go runtime, so a new isolated instance
  // can be created without relying on the legacy in-process close semantics.
  const replacement = await createGepagoWasmWorker();
  try {
    assert.equal(await replacement.version(), "3");
  } finally {
    replacement.close();
  }
});

test("worker rejects a module that reports another API version", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gepago-version-"));
  try {
    const wasmPath = join(directory, "gepago.wasm");
    const wasmExecPath = join(directory, "wasm_exec.js");
    // an empty WASM module, run by a Go shim that registers a version-2 API.
    await writeFile(
      wasmPath,
      new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]),
    );
    await writeFile(
      wasmExecPath,
      `const job = () => "";
globalThis.Go = class {
  importObject = {};
  run() {
    globalThis.gepagoWasm = { version: "2", invoke: job, start: job, startAgent: job, startCapabilities: job, cancel: job };
    return new Promise(() => {});
  }
};`,
    );
    await assert.rejects(
      createGepagoWasmWorker({ wasmPath, wasmExecPath }),
      /reports API version "2", but this wrapper needs version "3"/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("worker client makes every unexpected exit terminal, including code 0", async () => {
  type ConnectOptions = Parameters<typeof connectWorker>[1];
  type MessageHandler = Parameters<ConnectOptions["subscribe"]>[0];
  type ErrorHandler = Parameters<ConnectOptions["subscribe"]>[1];
  let onError: ErrorHandler | undefined;
  let initialized: MessageHandler | undefined;
  const client = connectWorker(
    {},
    {
      init: {},
      post(message) {
        if (message.type === "init")
          queueMicrotask(() => {
            assert.ok(initialized);
            initialized({ type: "ready", version: "test" });
          });
      },
      subscribe(handler, errorHandler) {
        initialized = handler;
        onError = errorHandler;
      },
      unsubscribe() {},
      terminate() {},
    },
  );
  await client.ready;
  const pending = client.version();
  assert.ok(onError);
  onError(new Error("Gepago WASM worker exited with code 0"));
  await assert.rejects(pending, /code 0/);
  await assert.rejects(client.version(), /code 0/);
  client.close();
});

test("worker client preserves error operation and code", async () => {
  type ConnectOptions = Parameters<typeof connectWorker>[1];
  type MessageHandler = Parameters<ConnectOptions["subscribe"]>[0];
  let onMessage: MessageHandler | undefined;
  const client = connectWorker(
    {},
    {
      init: {},
      post(message) {
        if (message.type === "init")
          queueMicrotask(() => {
            assert.ok(onMessage);
            onMessage({ type: "ready", version: "test" });
          });
        if (message.type === "call")
          queueMicrotask(() => {
            assert.ok(onMessage);
            onMessage({
              type: "result",
              id: message.id,
              ok: false,
              error: {
                name: "GepagoWasmError",
                message: "invalid prediction",
                operation: "parsePrediction",
                code: "INVALID_PREDICTION",
              },
            });
          });
      },
      subscribe(handler) {
        onMessage = handler;
      },
      terminate() {},
    },
  );

  await assert.rejects(client.parsePrediction("invalid"), (error: unknown) => {
    const structured = error as Error & { operation?: string; code?: string };
    assert.equal(structured.name, "GepagoWasmError");
    assert.equal(structured.operation, "parsePrediction");
    assert.equal(structured.code, "INVALID_PREDICTION");
    return true;
  });
  const workerError = Object.assign(new Error("worker failure"), {
    operation: "compile",
    code: "COMPILE_FAILED",
  });
  assert.equal(serializedError(workerError).operation, "compile");
  assert.equal(serializedError(workerError).code, "COMPILE_FAILED");
  client.close();
});

test("browser entry remains importable without Node-only imports", async () => {
  const browser = await import("@mikills/wasm/gepago/browser");
  assert.equal(typeof browser.createGepagoWasm, "function");
  assert.equal(typeof browser.createGepagoWasmWorker, "function");
});
