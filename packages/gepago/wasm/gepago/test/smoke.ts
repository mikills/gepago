import assert from "node:assert/strict";
import { test } from "node:test";
import { createGepagoWasm } from "@mikills/wasm/gepago";
import type {
  AgentChat,
  AgentToolRequest,
  CapabilityEvaluatorRequest,
  CapabilityMetricRequest,
  CapabilityObserverRequest,
  CapabilityProposerRequest,
  ChatRequest,
  ChatResponse,
  ChainOfThoughtProgram,
  CompileConfig,
  CompileModelRequest,
  GepagoWasm,
  JSONObject,
  JSONValue,
  PredictConfig,
  PredictHostRequest,
  PredictModelResult,
  PipelineProgram,
  RunResult,
  Signature,
} from "@mikills/wasm/gepago";

type GoGlobals = typeof globalThis & { Go?: unknown; gepagoWasm?: unknown };

test("loads generated Go WASM and exposes deterministic APIs", async () => {
  const globals = globalThis as GoGlobals;
  const hadGo = Object.hasOwn(globals, "Go");
  const previousGo = globals.Go;
  const sentinelGo = function SentinelGo() {};
  globals.Go = sentinelGo;
  let gepago: GepagoWasm | undefined;
  try {
    await assert.rejects(
      createGepagoWasm({ wasmPath: "/definitely/missing/gepago.wasm" }),
      /no such file|ENOENT/i,
    );
    assert.equal(globals.Go, sentinelGo);
    const api = (gepago = await createGepagoWasm({
      wasmPath: process.env.GEPAGO_WASM,
    }));
    assert.equal(
      globals.Go,
      sentinelGo,
      "wasm_exec Go constructor must be restored",
    );
    assert.equal(gepago.version(), "3");
    assert.equal(
      gepago.parsePrediction('```json\n{"label":"positive"}\n```').label,
      "positive",
    );
    assert.deepEqual(gepago.decodeDemos(""), []);
    const demos = [
      { inputs: { text: "great" }, outputs: { label: "positive" } },
    ];
    const encodedDemos = gepago.encodeDemos(demos);
    assert.equal(typeof encodedDemos, "string");
    assert.equal(
      gepago.exactMatch(
        { label: "ok" },
        { label: " OK " },
        { trimSpace: true, caseInsensitive: true },
      ).score,
      1,
    );
    assert.equal(
      gepago.contains({ text: "hello" }, { text: "hello world" }).score,
      1,
    );
    assert.equal(
      gepago.numericMatch({ count: 3 }, { count: "3.05" }, { tolerance: 0.1 })
        .score,
      1,
    );
    assert.equal(
      gepago.classification(
        { label: "positive" },
        { label: "POSITIVE" },
        { ignoreCase: true },
      ).score,
      1,
    );
    const signature = {
      inputs: [{ name: "text" }],
      outputs: [{ name: "label" }],
    } satisfies Signature;
    const request = {
      signature,
      instruction: "Classify the text.",
      demos,
      inputs: { text: "great" },
    };
    const prompt = gepago.buildPrompt(request);
    assert.match(prompt, /Classify the text/);
    assert.equal(
      gepago.buildPrompt(signature, request.instruction, demos, request.inputs),
      prompt,
    );
    assert.equal(
      gepago.buildPrompt(
        signature,
        { instruction: request.instruction, demos: encodedDemos },
        request.inputs,
      ),
      prompt,
    );

    const predictConfig = {
      program: { signature, instruction: request.instruction },
      inputs: request.inputs,
    } satisfies PredictConfig;
    const hostRequests: PredictHostRequest[] = [];
    const result = await gepago.predict(
      predictConfig,
      async (hostRequest: PredictHostRequest): Promise<PredictModelResult> => {
        hostRequests.push(hostRequest);
        assert.equal(hostRequest.kind, "language_model.generate");
        assert.equal(hostRequest.role, "task");
        return {
          text: '{"label":"positive"}',
          usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
        };
      },
    );
    assert.deepEqual(result, {
      prediction: { label: "positive" },
      usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
    });
    assert.equal(hostRequests.length, 1);

    const cotProgram = {
      kind: "chainOfThought",
      signature,
      instruction: "Classify the text.",
      reasoningField: "thoughts",
    } satisfies ChainOfThoughtProgram;
    const cotRequests: PredictHostRequest[] = [];
    const cotPrediction = await gepago.predict(
      { program: cotProgram, inputs: request.inputs },
      async (hostRequest: PredictHostRequest): Promise<PredictModelResult> => {
        cotRequests.push(hostRequest);
        await new Promise((resolve) => setTimeout(resolve, 1));
        return { text: '{"label":"positive","thoughts":"matched"}' };
      },
    );
    assert.deepEqual(cotPrediction.prediction, {
      label: "positive",
      thoughts: "matched",
    });
    assert.equal(
      (cotRequests[0].prompt.match(/Think step by step/g) ?? []).length,
      1,
    );
    assert.equal(
      (cotRequests[0].prompt.match(/thoughts field/g) ?? []).length,
      1,
    );
    assert.match(cotRequests[0].prompt, /output fields: thoughts, label\./);
    assert.equal(cotRequests[0].response_format?.strict, false);

    const typedCotRequests: PredictHostRequest[] = [];
    await gepago.predict(
      {
        program: {
          kind: "chainOfThought",
          signature: {
            inputs: [{ name: "text" }],
            outputs: [{ name: "label", type: "string" }],
          },
          reasoningField: "thoughts",
        },
        inputs: request.inputs,
      },
      async (hostRequest: PredictHostRequest): Promise<PredictModelResult> => {
        typedCotRequests.push(hostRequest);
        return { text: '{"thoughts":"matched","label":"positive"}' };
      },
    );
    const typedFormat = typedCotRequests[0].response_format;
    assert.ok(typedFormat);
    assert.equal(typedFormat.strict, true);
    const typedProperties = typedFormat.schema.properties as JSONObject;
    assert.deepEqual(Object.keys(typedProperties), ["thoughts", "label"]);
    assert.deepEqual(typedProperties.thoughts, {
      type: "string",
      description: "brief reasoning for the answer",
    });

    const cotCompiled = await gepago.compile(
      {
        program: cotProgram,
        trainset: [
          {
            id: "positive-cot",
            input: { text: "great" },
            expected: { label: "positive" },
          },
        ],
        metric: { kind: "exactMatch", fields: ["label"] },
        components: ["instruction"],
        maxMetricCalls: 1,
        seed: 9,
      },
      async () => ({ text: '{"label":"positive","thoughts":"matched"}' }),
    );
    assert.match(cotCompiled.candidate.instruction, /thoughts field/);
    const cotRunRequests: PredictHostRequest[] = [];
    await gepago.runCompiled(
      cotProgram,
      cotCompiled.candidate,
      { text: "great" },
      async (hostRequest: PredictHostRequest): Promise<string> => {
        cotRunRequests.push(hostRequest);
        return '{"label":"positive","thoughts":"matched"}';
      },
    );
    assert.equal(
      (cotRunRequests[0].prompt.match(/Think step by step/g) ?? []).length,
      1,
    );

    const repairRequests: PredictHostRequest[] = [];
    const repaired = await gepago.predict(
      {
        program: { signature, maxParseRetries: 1 },
        inputs: request.inputs,
      },
      async (hostRequest: PredictHostRequest): Promise<PredictModelResult> => {
        repairRequests.push(hostRequest);
        return hostRequest.attempt === 1
          ? {
              text: "not JSON",
              usage: {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
              },
            }
          : {
              text: '{"label":"repaired"}',
              usage: {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
              },
            };
      },
    );
    assert.equal(repairRequests.length, 2);
    assert.equal(repairRequests[1].attempt, 2);
    assert.match(
      repairRequests[1].prompt,
      /Your previous reply was not a valid JSON object/,
    );
    assert.ok(repairRequests[1].prompt.startsWith(repairRequests[0].prompt));
    assert.deepEqual(repaired, {
      prediction: { label: "repaired" },
      usage: { prompt_tokens: 4, completion_tokens: 3, total_tokens: 7 },
    });

    await assert.rejects(
      gepago.predict(predictConfig, async () => {
        throw new Error("model failed");
      }),
      /model failed/,
    );

    const truncatedRequests: PredictHostRequest[] = [];
    await assert.rejects(
      gepago.predict(
        {
          program: { signature, maxParseRetries: 2 },
          inputs: request.inputs,
        },
        async (hostRequest: PredictHostRequest): Promise<PredictModelResult> => {
          truncatedRequests.push(hostRequest);
          return {
            text: '{"label":"pos',
            finishReason: "length",
            model: "host-model",
            usage: {
              prompt_tokens: 4,
              completion_tokens: 8,
              total_tokens: 12,
              reasoning_tokens: 3,
            },
          };
        },
      ),
      /truncated/,
    );
    assert.equal(truncatedRequests.length, 1, "a truncated reply is not repaired");
    await assert.rejects(
      gepago.predict(predictConfig, async () => ({
        text: '{"label":"positive"}',
        finishReason: "end_turn" as never,
      })),
      /finishReason must be one of/,
    );
    for (const finishReason of [null, ""] as const) {
      const unreported = await gepago.predict(predictConfig, async () => ({
        text: '{"label":"positive"}',
        finishReason,
      }));
      assert.deepEqual(unreported.prediction, { label: "positive" });
    }

    const compileConfig = {
      program: { signature, instruction: "Classify the text." },
      trainset: [
        {
          id: "positive",
          input: { text: "great" },
          expected: { label: "positive" },
        },
      ],
      metric: { kind: "exactMatch", fields: ["label"] },
      components: ["instruction"],
      maxMetricCalls: 1,
      seed: 7,
    } satisfies CompileConfig;
    const compileRequests: CompileModelRequest[] = [];
    const compiled = await gepago.compile(
      compileConfig,
      async (hostRequest: CompileModelRequest) => {
        compileRequests.push(hostRequest);
        assert.equal(typeof hostRequest, "object");
        assert.equal(hostRequest.kind, "language_model.generate");
        assert.equal(hostRequest.role, "task");
        await new Promise((resolve) => setTimeout(resolve, 1));
        return '{"label":"positive"}';
      },
    );
    assert.equal(compiled.candidate.instruction, "Classify the text.");
    assert.equal(compiled.state.candidates.length > 0, true);
    assert.match(
      gepago.htmlReport(compiled.state, { title: "Smoke report" }),
      /Smoke report/,
    );
    assert.equal(compiled.state.ledger.total_tokens, 0);
    assert.equal(compileRequests.length, 1);

    const truncatedCompileRoles: string[] = [];
    const truncatedCompiled = await gepago.compile(
      { ...compileConfig, program: { ...compileConfig.program, maxParseRetries: 1 } },
      async (hostRequest: CompileModelRequest) => {
        truncatedCompileRoles.push(hostRequest.role);
        return {
          text: '{"label":"positive"}',
          finishReason: "length",
          model: "host-model",
          usage: { prompt_tokens: 2, completion_tokens: 3, total_tokens: 5 },
        };
      },
    );
    assert.deepEqual(truncatedCompileRoles, ["task"]);
    for (const finishReason of [null, ""] as const) {
      const unreportedCompiled = await gepago.compile(compileConfig, async () => ({
        text: '{"label":"positive"}',
        finishReason,
      }));
      assert.equal(unreportedCompiled.state.candidates[0].validation_score, 1);
    }
    assert.equal(truncatedCompiled.state.candidates[0].validation_score, 0);
    assert.equal(truncatedCompiled.state.ledger.total_tokens, 5);
    const compiledPrediction = await gepago.runCompiled(
      compileConfig.program,
      compiled.candidate,
      { text: "great" },
      async () => ({ text: '{"label":"positive"}' }),
    );
    assert.deepEqual(compiledPrediction.prediction, { label: "positive" });
    await assert.rejects(
      gepago.runCompiled(
        compileConfig.program,
        { "step.instruction": "unsupported" },
        { text: "great" },
        () => "never",
      ),
      /only supports Predict candidates.*step\.instruction/i,
    );

    const pipelineProgram = {
      kind: "pipeline",
      steps: [
        {
          name: "extract",
          program: {
            kind: "predict",
            signature: {
              inputs: [{ name: "text" }],
              outputs: [
                { name: "topic" },
                { name: "note", type: "string", optional: true },
              ],
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
      returnAll: true,
    } satisfies PipelineProgram;
    const pipelineCompiled = await gepago.compile(
      {
        program: pipelineProgram,
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
      async (hostRequest: CompileModelRequest) => {
        await new Promise((resolve) => setTimeout(resolve, 1));
        return hostRequest.prompt.includes("output fields: topic")
          ? { text: '{"topic":"service"}' }
          : { text: '{"label":"positive"}' };
      },
    );
    assert.equal(
      typeof pipelineCompiled.candidate["extract.instruction"],
      "string",
    );
    assert.equal(
      typeof pipelineCompiled.candidate["decide.instruction"],
      "string",
    );
    const pipelineRequests: PredictHostRequest[] = [];
    const pipelineResult = await gepago.runCompiled(
      pipelineProgram,
      pipelineCompiled.candidate,
      { text: "great", id: "not read" },
      async (hostRequest: PredictHostRequest): Promise<PredictModelResult> => {
        pipelineRequests.push(hostRequest);
        await new Promise((resolve) => setTimeout(resolve, 1));
        return hostRequest.prompt.includes(
          "Return only a JSON object with these output fields: topic",
        )
          ? {
              text: '{"topic":"service"}',
              usage: {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
              },
            }
          : {
              text: '{"label":"positive"}',
              usage: {
                prompt_tokens: 2,
                completion_tokens: 1,
                total_tokens: 3,
              },
            };
      },
    );
    // as in Compile, a step without inputKeys reads only its signature's inputs, and an
    // optional output the reply leaves out holds its type's zero value.
    assert.deepEqual(pipelineResult.prediction, {
      text: "great",
      id: "not read",
      "extract.topic": "service",
      "extract.note": "",
      label: "positive",
    });
    assert.deepEqual(pipelineResult.usage, {
      prompt_tokens: 3,
      completion_tokens: 2,
      total_tokens: 5,
    });
    assert.equal(pipelineRequests.length, 2);
    assert.ok(pipelineRequests[0].prompt.includes('"text": "great"'));
    assert.ok(!pipelineRequests[0].prompt.includes("not read"));

    // a model callback that fails, by throwing or by rejecting, returns its error to Go as
    // that call's error, so the step fails with the host's message and the job is not
    // reported as cancelled.
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
        gepago.runCompiled(
          pipelineProgram,
          pipelineCompiled.candidate,
          { text: "great" },
          failingModel,
        ),
        hostFailure,
      );
      await assert.rejects(
        gepago.compile(
          {
            program: pipelineProgram,
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

    await assert.rejects(
      gepago.compile(
        {
          program: {
            kind: "pipeline",
            steps: [
              { name: "duplicate", program: { signature } },
              { name: "duplicate", program: { signature } },
            ],
          },
          trainset: [
            { id: "bad", input: { text: "x" }, expected: { label: "x" } },
          ],
          metric: { kind: "exactMatch" },
        },
        () => "never",
      ),
      /duplicate/,
    );
    await assert.rejects(
      gepago.runCompiled(
        {
          kind: "pipeline",
          steps: [
            { name: "missing", program: { signature, inputKeys: ["absent"] } },
          ],
        },
        {},
        { text: "x" },
        () => "never",
      ),
      /pipeline input "absent" is missing/,
    );

    // compile reports a pipeline's key and input errors, including those found before any
    // model call, and runCompiled accepts, rejects and prompts as compile does.
    const countProgram = {
      kind: "pipeline",
      steps: [
        {
          name: "count",
          program: {
            signature: {
              name: "count",
              inputs: [
                { name: "count", type: "int" },
                { name: "weight", type: "float32" },
              ],
              outputs: [{ name: "label" }],
            },
            instruction: "Count.",
          },
        },
      ],
    } satisfies PipelineProgram;
    const compilePrompts: string[] = [];
    const compileCount = (input: JSONObject) =>
      api.compile(
        {
          program: countProgram,
          trainset: [{ id: "count", input, expected: { label: "three" } }],
          metric: { kind: "exactMatch", fields: ["label"] },
          maxMetricCalls: 1,
        },
        (hostRequest: CompileModelRequest) => {
          if (hostRequest.role === "task")
            compilePrompts.push(hostRequest.prompt);
          return '{"label":"three"}';
        },
      );
    const badCount = /invalid pipeline input: state key "count"/;
    await assert.rejects(compileCount({ count: "3", weight: 1 }), badCount);
    await assert.rejects(
      gepago.runCompiled(
        countProgram,
        {},
        { count: "3", weight: 1 },
        () => "never",
      ),
      badCount,
    );
    const rewriteProgram = {
      kind: "pipeline",
      steps: [
        {
          name: "rewrite",
          program: {
            signature: { inputs: [{ name: "draft" }], outputs: [{ name: "text" }] },
            inputKeys: ["text"],
          },
        },
      ],
    } satisfies PipelineProgram;
    const writesInput =
      /pipeline step "rewrite" writes pipeline input "text", which is read-only/;
    await assert.rejects(
      gepago.compile(
        {
          program: rewriteProgram,
          trainset: [
            { id: "rewrite", input: { text: "x" }, expected: { text: "y" } },
          ],
          metric: { kind: "exactMatch" },
        },
        () => "never",
      ),
      writesInput,
    );
    await assert.rejects(
      gepago.runCompiled(rewriteProgram, {}, { text: "x" }, () => "never"),
      writesInput,
    );
    await compileCount({ count: 3, weight: 16777217 });
    const runPrompts: string[] = [];
    const counted = await gepago.runCompiled(
      countProgram,
      { "count.instruction": "Count." },
      { count: 3, weight: 16777217 },
      (hostRequest: PredictHostRequest) => {
        runPrompts.push(hostRequest.prompt);
        return '{"label":"three"}';
      },
    );
    assert.deepEqual(counted.prediction, { label: "three" });
    assert.deepEqual(runPrompts, [compilePrompts[0]]);
    assert.ok(compilePrompts[0].includes('"weight": 16777216'));

    const judgeRoles: CompileModelRequest["role"][] = [];
    const judged = await gepago.compile(
      {
        ...compileConfig,
        metric: {
          kind: "llmJudge",
          rubric: "matching labels score one",
          role: "judge",
        },
      },
      async (hostRequest: CompileModelRequest) => {
        judgeRoles.push(hostRequest.role);
        await new Promise((resolve) => setTimeout(resolve, 1));
        return hostRequest.role === "task"
          ? {
              text: '{"label":"positive"}',
              usage: {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
              },
            }
          : {
              text: '{"score":1,"feedback":"match"}',
              usage: {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
              },
            };
      },
    );
    assert.deepEqual(judgeRoles, ["task", "judge"]);
    assert.equal(judged.state.ledger.total_tokens, 4);

    let customMetricCalls = 0;
    const customCompiled = await gepago.compile(
      {
        ...compileConfig,
        metric: { kind: "host" },
      },
      async () => ({ text: '{"label":"positive"}' }),
      {
        metric: async (request: CapabilityMetricRequest) => {
          customMetricCalls += 1;
          assert.equal(request.kind, "metric.score");
          return {
            score: request.expected.label === request.actual.label ? 1 : 0,
            feedback: "host metric",
          };
        },
        observer: async (request: CapabilityObserverRequest) => {
          assert.equal(request.kind, "optimizer.observe");
        },
      },
    );
    assert.equal(customMetricCalls, 1);
    assert.equal(customCompiled.state.candidates.length > 0, true);

    const optimized = await gepago.optimize(
      {
        seedCandidate: { instruction: "Classify." },
        trainset: [{ id: "one", input: { text: "great" } }],
        valset: [{ id: "one", input: { text: "great" } }],
        maxMetricCalls: 1,
        minibatchSize: 1,
        seed: 2,
      },
      {
        evaluator: async (request: CapabilityEvaluatorRequest) => {
          const example = request.examples[0];
          assert.ok(
            example && typeof example === "object" && !Array.isArray(example),
          );
          return {
            items: [
              {
                example_id: example.id,
                output: { label: "positive" },
                score: 1,
              },
            ],
            usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
          };
        },
        proposer: async (_request: CapabilityProposerRequest) => ({
          instruction: "Improved.",
        }),
      },
    );
    assert.equal(optimized.state.best_candidate_id.length > 0, true);

    const agentRequests: ChatRequest[] = [];
    const toolRequests: AgentToolRequest[] = [];
    const agentResult = await gepago.runAgent(
      {
        name: "assistant",
        model: "host-model",
        systemPrompt: "Be concise.",
        messages: [{ role: "user", content: "Look up the value." }],
        tools: [
          {
            name: "lookup",
            description: "Look up a value.",
            parameters: { type: "object" },
          },
        ],
        run_id: "agent-smoke",
        memory: { tenant: "demo", count: 0 },
      },
      async (request: ChatRequest): Promise<ChatResponse> => {
        agentRequests.push(request);
        await new Promise((resolve) => setTimeout(resolve, 1));
        if (agentRequests.length === 1) {
          return {
            message: {
              role: "assistant",
              tool_calls: [
                { id: "call-1", name: "lookup", arguments: '{"key":"value"}' },
              ],
            },
            usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
            finish_reason: "tool_calls",
          };
        }
        const lastMessage = request.messages.at(-1);
        assert.ok(lastMessage);
        assert.equal(lastMessage.role, "tool");
        return {
          message: { role: "assistant", content: "The value is ready." },
          usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
          finish_reason: "stop",
        };
      },
      async (request: AgentToolRequest) => {
        toolRequests.push(request);
        await new Promise((resolve) => setTimeout(resolve, 1));
        assert.equal(request.call_id, "call-1");
        assert.deepEqual(request.memory, { tenant: "demo", count: 0 });
        return {
          output: "value",
          memory: { tenant: "demo", count: 1, found: true },
        };
      },
    );
    assert.equal(agentRequests.length, 2);
    assert.equal(toolRequests.length, 1);
    assert.deepEqual(agentResult.final, {
      role: "assistant",
      content: "The value is ready.",
    });
    assert.equal(agentResult.stop_reason, "complete");
    assert.equal(agentResult.tool_calls.length, 1);
    assert.equal(agentResult.usage.total_tokens, 13);

    const noToolAgent = await gepago.runAgent(
      {
        name: "assistant",
        model: "host-model",
        messages: [{ role: "user", content: "Say hello." }],
        run_id: "agent-no-tool",
      },
      async () => ({
        message: { role: "assistant", content: "Hello." },
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        finish_reason: "stop",
      }),
      async () => "never",
    );
    assert.deepEqual(noToolAgent.tool_calls, []);

    for (const reason of [undefined, null, ""]) {
      const response: Record<string, unknown> = {
        message: { role: "assistant", content: "Hello." },
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      };
      if (reason !== undefined) response.finish_reason = reason;
      const unreportedAgent: RunResult = await gepago.runAgent(
        {
          name: "assistant",
          model: "host-model",
          messages: [{ role: "user", content: "Say hello." }],
          run_id: "agent-unreported-finish",
        },
        async (): Promise<ChatResponse> =>
          response as unknown as ChatResponse,
        async () => "never",
      );
      assert.equal(unreportedAgent.stop_reason, "complete");
    }

    await assert.rejects(
      gepago.runAgent(
        {
          name: "assistant",
          model: "host-model",
          messages: [{ role: "user", content: "hello" }],
          run_id: "bad-agent",
          tools: [{ name: "", description: "missing name" }],
        },
        async (): Promise<ChatResponse> => ({}) as unknown as ChatResponse,
        async () => "never",
      ),
      TypeError,
    );
    await assert.rejects(
      gepago.runAgent(
        {
          name: "assistant",
          model: "host-model",
          messages: [{ role: "user", content: "hello" }],
          run_id: "bad-response",
        },
        async () => {
          await new Promise((resolve) => setTimeout(resolve, 1));
          return {
            message: { role: "user" },
            usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
            finish_reason: "stop",
          };
        },
        async () => "never",
      ),
      /assistant|cancel/i,
    );

    await assert.rejects(
      api.compile(compileConfig, async () => {
        throw new Error("compile model failed");
      }),
      /compile model failed/,
    );
    await assert.rejects(
      api.compile({ ...compileConfig, trainset: [] }, () => "never"),
      TypeError,
    );

    assert.throws(() => api.parsePrediction("not JSON"), /parse prediction/i);
    assert.throws(
      () => api.exactMatch({ label: NaN }, { label: "x" }),
      TypeError,
    );
    assert.throws(
      () => api.encodeDemos(Object.assign([...demos], { extra: true })),
      TypeError,
    );
    assert.throws(
      () => api.numericMatch({}, {}, { tolerance: Infinity }),
      TypeError,
    );
    assert.throws(
      () => api.invoke("version", undefined as unknown as JSONValue),
      TypeError,
    );
    await assert.rejects(createGepagoWasm(), /one instance per process/);

    api.close();
    api.close();
    assert.equal(globals.gepagoWasm, undefined);
    assert.equal(globals.Go, sentinelGo);
    assert.throws(() => api.version(), /instance is closed/);
    await assert.rejects(
      api.compile(compileConfig, () => "never"),
      /instance is closed/,
    );
    await assert.rejects(createGepagoWasm(), /one instance per process/);
  } finally {
    gepago?.close();
    if (hadGo) globals.Go = previousGo;
    else delete globals.Go;
  }
});
