import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import type { PipelineProgram } from "@mikills/wasm/gepago";
import { loadBrowserRuntime } from "../browser-runtime.ts";

// the browser runtime runs the built module in this test's own process, apart from the Node
// runtime the other test files load, and fetch reads the built files as a browser would
// fetch them from a server.
test("browser runtime reports a failing model callback's message", async () => {
  const dist = new URL("../../dist/", import.meta.url);
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = new URL(String(input instanceof Request ? input.url : input));
    if (url.protocol !== "file:") return originalFetch(input);
    return new Response(await readFile(fileURLToPath(url)), {
      headers: { "content-type": "application/wasm" },
    });
  };
  try {
    const runtime = await loadBrowserRuntime({
      wasmURL: new URL("gepago.wasm", dist),
      wasmExecURL: new URL("wasm_exec.js", dist),
    });
    assert.equal(runtime.version(), "2");

    const program = {
      kind: "pipeline",
      steps: [
        {
          name: "extract",
          program: {
            kind: "predict",
            signature: {
              inputs: [{ name: "text" }],
              outputs: [{ name: "topic" }],
            },
            instruction: "Extract the topic.",
            outputPrefix: "extract.",
          },
        },
        {
          name: "decide",
          program: {
            kind: "predict",
            signature: {
              inputs: [{ name: "extract.topic" }],
              outputs: [{ name: "label" }],
            },
            instruction: "Classify the topic.",
            inputKeys: ["extract.topic"],
          },
        },
      ],
    } satisfies PipelineProgram;
    const hostFailure = /pipeline step "extract": .*host model exploded/;
    const failingModels = [
      (): never => {
        throw new Error("host model exploded");
      },
      async (): Promise<never> => {
        throw new Error("host model exploded");
      },
    ];
    for (const failingModel of failingModels) {
      await assert.rejects(
        runtime.runCompiled(program, {}, { text: "great" }, failingModel),
        hostFailure,
      );
      await assert.rejects(
        runtime.compile(
          {
            program,
            trainset: [
              {
                id: "pipeline",
                input: { text: "great" },
                expected: { label: "positive" },
              },
            ],
            metric: { kind: "exactMatch", fields: ["label"] },
            maxMetricCalls: 1,
            seed: 3,
          },
          failingModel,
        ),
        hostFailure,
      );
    }

    // the runtime still runs a pipeline after the failures.
    const result = await runtime.runCompiled(
      program,
      {},
      { text: "great" },
      async (request) => ({
        text: request.prompt.includes("output fields: topic")
          ? '{"topic":"service"}'
          : '{"label":"positive"}',
      }),
    );
    assert.equal(result.prediction.label, "positive");
  } finally {
    globalThis.fetch = originalFetch;
  }
});
