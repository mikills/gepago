import assert from "node:assert/strict";
import { chmod, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type {
  ChatResponse,
  JSONObject,
  PredictHostRequest,
  PredictModel,
  PredictModelResult,
} from "@mikills/wasm/gepago";
import {
  createClaudeChat,
  createClaudeModel,
  createGoogleChat,
  createGoogleModel,
  createOpenAIChat,
  createOpenAIModel,
  createPiModel,
  type ProviderFetch,
  type ProviderFetchResponse,
} from "@mikills/wasm/gepago/providers";

type FetchInit = NonNullable<Parameters<ProviderFetch>[1]>;
type FetchHandler<T extends JSONObject> = (
  url: URL,
  init: FetchInit,
  body: T,
) => ProviderFetchResponse | Promise<ProviderFetchResponse>;

type OpenAIToolFixture = JSONObject & {
  function: JSONObject & { name: string };
};
type OpenAIRequestFixture = JSONObject & {
  model: string;
  max_tokens?: number;
  tools?: OpenAIToolFixture[];
  response_format?: JSONObject & {
    json_schema: JSONObject & { strict: boolean };
  };
};
type GeminiRequestFixture = JSONObject & {
  tools: Array<
    JSONObject & { functionDeclarations: Array<JSONObject & { name: string }> }
  >;
  contents: Array<JSONObject & { parts: JSONObject[] }>;
  generationConfig?: JSONObject & {
    maxOutputTokens?: number;
    temperature?: number;
  };
};

type ModelFixture = PredictHostRequest;
type PredictCallback = (
  request: PredictHostRequest,
) => Promise<PredictModelResult>;

function fakeFetch<T extends JSONObject>(
  handler: FetchHandler<T>,
): ProviderFetch {
  return async (url, init = {}) => {
    const body = JSON.parse(init.body ?? "") as T;
    return handler(new URL(url), init, body);
  };
}

function jsonResponse(value: JSONObject, status = 200): ProviderFetchResponse {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(value),
  };
}

function modelRequest(prompt: string): ModelFixture {
  return {
    kind: "language_model.generate",
    role: "task",
    request_id: "test",
    attempt: 1,
    prompt,
  };
}

test("OpenAI model callback keeps credentials out of the request payload", async () => {
  let seen:
    { url: URL; init: FetchInit; body: OpenAIRequestFixture } | undefined;
  const model = createOpenAIModel({
    apiKey: "secret",
    model: "compatible",
    maxTokens: 64,
    fetch: fakeFetch<OpenAIRequestFixture>((url, init, body) => {
      seen = { url, init, body };
      return jsonResponse({
        choices: [
          {
            message: { role: "assistant", content: '{"label":"yes"}' },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 2, completion_tokens: 3, total_tokens: 5 },
      });
    }),
  });
  const result = (await (model as PredictModel)(
    modelRequest("classify"),
  )) as PredictModelResult;
  assert.equal(result.text, '{"label":"yes"}');
  assert.deepEqual(result.usage, {
    prompt_tokens: 2,
    completion_tokens: 3,
    total_tokens: 5,
  });
  assert.equal(result.finishReason, "stop");
  assert.ok(seen);
  assert.equal(seen.init.headers?.authorization, "Bearer secret");
  assert.equal("apiKey" in seen.body, false);
  assert.equal(seen.body.max_tokens, 64);
});

test("OpenAI and Claude model callbacks enforce a strict response format", async () => {
  const schema = {
    type: "object",
    properties: { label: { type: "string" } },
    required: ["label"],
    additionalProperties: false,
  };
  const strict = {
    ...modelRequest("classify"),
    response_format: { name: "sentiment", schema, strict: true },
  };
  const loose = {
    ...modelRequest("classify"),
    response_format: {
      name: "sentiment",
      schema: { type: "object" },
      strict: false,
    },
  };
  const bodies: JSONObject[] = [];
  const openAIFetch = fakeFetch<JSONObject>((_url, _init, body) => {
    bodies.push(body);
    return jsonResponse({
      choices: [
        {
          message: { role: "assistant", content: '{"label":"yes"}' },
          finish_reason: "stop",
        },
      ],
    });
  });
  const openAI = createOpenAIModel({
    apiKey: "k",
    model: "m",
    fetch: openAIFetch,
  }) as PredictModel;
  await openAI(strict);
  await openAI(loose);
  assert.deepEqual(bodies[0]?.response_format, {
    type: "json_schema",
    json_schema: { name: "sentiment", schema, strict: true },
  });
  assert.equal(
    "response_format" in (bodies[1] ?? {}),
    false,
    "a loose format is not enforced",
  );

  bodies.length = 0;
  const claudeFetch = fakeFetch<JSONObject>((_url, _init, body) => {
    bodies.push(body);
    return jsonResponse({
      content: [{ type: "text", text: '{"label":"yes"}' }],
      stop_reason: "end_turn",
    });
  });
  const claude = createClaudeModel({
    apiKey: "k",
    model: "m",
    fetch: claudeFetch,
  }) as PredictModel;
  await claude(strict);
  await claude(loose);
  assert.deepEqual(bodies[0]?.output_config, {
    format: { type: "json_schema", schema },
  });
  assert.equal(
    "output_config" in (bodies[1] ?? {}),
    false,
    "Claude has no loose schema mode",
  );

  bodies.length = 0;
  const googleFetch = fakeFetch<JSONObject>((_url, _init, body) => {
    bodies.push(body);
    return jsonResponse({
      candidates: [
        {
          content: { parts: [{ text: '{"label":"yes"}' }] },
          finishReason: "STOP",
        },
      ],
    });
  });
  const google = createGoogleModel({
    apiKey: "k",
    model: "m",
    fetch: googleFetch,
  }) as PredictModel;
  await google(strict);
  assert.equal(
    "responseSchema" in
      ((bodies[0]?.generationConfig as JSONObject | undefined) ?? {}),
    false,
    "Gemini model callbacks keep the prompt-only path, as the Go provider does",
  );
});

test("OpenAI refusals are content-filtered replies", async () => {
  const model = createOpenAIModel({
    apiKey: "k",
    model: "m",
    fetch: fakeFetch(() =>
      jsonResponse({
        choices: [
          {
            message: {
              role: "assistant",
              content: null,
              refusal: "I can't help with that.",
            },
            finish_reason: "stop",
          },
        ],
      }),
    ),
  }) as PredictModel;
  const result = (await model(modelRequest("classify"))) as PredictModelResult;
  assert.equal(result.text, "I can't help with that.");
  assert.equal(result.finishReason, "content_filter");
});

test("model callbacks report normalised finish reasons", async () => {
  const cases: Array<{
    create: (fetch: ProviderFetch) => PredictModel;
    body: JSONObject;
    want: PredictModelResult["finishReason"];
  }> = [
    {
      create: (fetch) =>
        createOpenAIModel({ apiKey: "k", model: "m", fetch }) as PredictModel,
      body: {
        choices: [{ message: { content: '{"cut' }, finish_reason: "length" }],
      },
      want: "length",
    },
    {
      create: (fetch) =>
        createOpenAIModel({ apiKey: "k", model: "m", fetch }) as PredictModel,
      body: {
        choices: [
          { message: { content: "x" }, finish_reason: "content_filter" },
        ],
      },
      want: "content_filter",
    },
    {
      create: (fetch) =>
        createClaudeModel({ apiKey: "k", model: "m", fetch }) as PredictModel,
      body: {
        content: [{ type: "text", text: '{"cut' }],
        stop_reason: "model_context_window_exceeded",
      },
      want: "length",
    },
    {
      create: (fetch) =>
        createClaudeModel({ apiKey: "k", model: "m", fetch }) as PredictModel,
      body: {
        content: [{ type: "text", text: "no" }],
        stop_reason: "refusal",
      },
      want: "content_filter",
    },
    {
      create: (fetch) =>
        createGoogleModel({ apiKey: "k", model: "m", fetch }) as PredictModel,
      body: {
        candidates: [
          { content: { parts: [{ text: "x" }] }, finishReason: "SAFETY" },
        ],
      },
      want: "content_filter",
    },
  ];
  for (const { create, body, want } of cases) {
    const model = create(fakeFetch(() => jsonResponse(body)));
    const result = (await model(modelRequest("p"))) as PredictModelResult;
    assert.equal(result.finishReason, want, JSON.stringify(body));
  }
});

test("unknown or missing finish reasons are not reported", async () => {
  const chats: Array<{
    chat: (fetch: ProviderFetch) => ReturnType<typeof createOpenAIChat>;
    body: JSONObject;
  }> = [
    {
      chat: (fetch) => createClaudeChat({ apiKey: "k", model: "m", fetch }),
      body: {
        content: [{ type: "text", text: "paused" }],
        stop_reason: "pause_turn",
      },
    },
    {
      chat: (fetch) => createGoogleChat({ apiKey: "k", model: "m", fetch }),
      body: {
        candidates: [
          { content: { parts: [{ text: "x" }] }, finishReason: "OTHER" },
        ],
      },
    },
    {
      chat: (fetch) => createOpenAIChat({ apiKey: "k", model: "m", fetch }),
      body: { choices: [{ message: { content: "x" } }] },
    },
  ];
  for (const { chat, body } of chats) {
    const fetch = fakeFetch(() => jsonResponse(body));
    const response = await chat(fetch)({
      model: "m",
      messages: [{ role: "user", content: "hi" }],
    });
    assert.equal("finish_reason" in response, false, JSON.stringify(body));
  }
  const model = createGoogleModel({
    apiKey: "k",
    model: "m",
    fetch: fakeFetch(() =>
      jsonResponse({
        candidates: [
          { content: { parts: [{ text: "x" }] }, finishReason: "OTHER" },
        ],
      }),
    ),
  }) as PredictModel;
  const result = (await model(modelRequest("p"))) as PredictModelResult;
  assert.equal("finishReason" in result, false);
});

test("OpenAI chat maps tool calls and response format", async () => {
  let body: OpenAIRequestFixture | undefined;
  const chat = createOpenAIChat({
    apiKey: "key",
    model: "gpt-test",
    fetch: fakeFetch<OpenAIRequestFixture>((_url, _init, request) => {
      body = request;
      return jsonResponse({
        choices: [
          {
            message: {
              role: "assistant",
              tool_calls: [
                {
                  id: "c1",
                  function: { name: "lookup", arguments: '{"q":"x"}' },
                },
              ],
            },
            finish_reason: "tool_calls",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 },
      });
    }),
  });
  const response = await chat({
    model: "gpt-override",
    messages: [{ role: "user", content: "find" }],
    tools: [
      { name: "lookup", description: "Find", parameters: { type: "object" } },
    ],
    response_format: {
      name: "answer",
      schema: { type: "object" },
      strict: true,
    },
  });
  assert.ok(body);
  assert.equal(body.model, "gpt-override");
  assert.ok(body.tools);
  assert.ok(body.response_format);
  assert.equal(body.tools[0]?.function.name, "lookup");
  assert.equal(body.response_format.json_schema.strict, true);
  assert.deepEqual(response.message.tool_calls?.[0], {
    id: "c1",
    name: "lookup",
    arguments: '{"q":"x"}',
  });
  assert.equal(response.finish_reason, "tool_calls");
});

test("Gemini assigns collision-free normalized tool names and restores originals", async () => {
  let body: GeminiRequestFixture | undefined;
  const gemini = createGoogleChat({
    apiKey: "google-secret",
    model: "gemini-test",
    fetch: fakeFetch<GeminiRequestFixture>((_url, _init, request) => {
      body = request;
      return jsonResponse({
        candidates: [
          {
            content: {
              parts: [
                { functionCall: { name: "foo_bar_2", args: { value: 1 } } },
              ],
            },
            finishReason: "STOP",
          },
        ],
      });
    }),
  });
  const response = await gemini({
    model: "gemini-test",
    messages: [
      {
        role: "assistant",
        content: "",
        tool_calls: [{ id: "call-1", name: "foo/bar", arguments: "{}" }],
      },
      { role: "tool", tool_call_id: "call-1", content: "done" },
    ],
    tools: [
      { name: "foo bar", description: "first", parameters: { type: "object" } },
      {
        name: "foo/bar",
        description: "second",
        parameters: { type: "object" },
      },
    ],
  });
  assert.ok(body);
  assert.deepEqual(
    body.tools[0]?.functionDeclarations.map((tool) => tool.name),
    ["foo_bar", "foo_bar_2"],
  );
  assert.equal(
    (
      body.contents[0]?.parts[0] as JSONObject & {
        functionCall: JSONObject & { name: string };
      }
    ).functionCall.name,
    "foo_bar_2",
  );
  assert.equal(
    (
      body.contents[1]?.parts[0] as JSONObject & {
        functionResponse: JSONObject & { name: string };
      }
    ).functionResponse.name,
    "foo_bar_2",
  );
  assert.equal(response.message.tool_calls?.[0]?.name, "foo/bar");
});

test("browser Gemini adapter also keeps normalized tool names unique", async () => {
  const browserProviders = await import("@mikills/wasm/gepago/browser");
  let body: GeminiRequestFixture | undefined;
  const chat = browserProviders.createGoogleChat({
    apiKey: "key",
    model: "gemini-test",
    fetch: fakeFetch<GeminiRequestFixture>((_url, _init, request) => {
      body = request;
      return jsonResponse({
        candidates: [
          { content: { parts: [{ text: "ok" }] }, finishReason: "STOP" },
        ],
      });
    }),
  });
  await chat({
    model: "gemini-test",
    messages: [{ role: "user", content: "use it" }],
    tools: [
      { name: "a b", description: "one" },
      { name: "a_b", description: "two" },
    ],
  });
  assert.ok(body);
  assert.deepEqual(
    body.tools[0]?.functionDeclarations.map((tool) => tool.name),
    ["a_b", "a_b_2"],
  );
});

test("Claude messages and Gemini generateContent are deterministic with injected fetch", async () => {
  let claudeBody: JSONObject | undefined;
  const claude = createClaudeChat({
    apiKey: "anthropic-secret",
    model: "claude-test",
    temperature: 0.25,
    fetch: fakeFetch<JSONObject>((_url, _init, body) => {
      claudeBody = body;
      return jsonResponse({
        content: [{ type: "text", text: "hello" }],
        usage: { input_tokens: 2, output_tokens: 1 },
        stop_reason: "end_turn",
      });
    }),
  });
  const claudeResponse = await claude({
    model: "claude-test",
    messages: [
      { role: "system", content: "system" },
      { role: "user", content: "hi" },
    ],
  });
  assert.ok(claudeBody);
  assert.equal(claudeBody.system, "system");
  assert.equal(claudeBody.temperature, 0.25);
  assert.equal(claudeResponse.message.content, "hello");
  assert.equal(claudeResponse.usage.total_tokens, 3);

  let geminiURL: URL | undefined;
  let geminiBody: GeminiRequestFixture | undefined;
  const gemini = createGoogleChat({
    apiKey: "google-secret",
    model: "gemini-test",
    maxTokens: 321,
    temperature: 0.4,
    fetch: fakeFetch<GeminiRequestFixture>((url, _init, body) => {
      geminiURL = url;
      geminiBody = body;
      return jsonResponse({
        candidates: [
          { content: { parts: [{ text: "hi" }] }, finishReason: "STOP" },
        ],
        usageMetadata: {
          promptTokenCount: 4,
          candidatesTokenCount: 2,
          totalTokenCount: 6,
        },
      });
    }),
  });
  const geminiResponse = await gemini({
    model: "gemini-test",
    messages: [{ role: "user", content: "hello" }],
  });
  assert.ok(geminiURL);
  assert.ok(geminiBody);
  assert.equal(geminiURL.searchParams.get("key"), "google-secret");
  assert.equal(geminiBody.contents[0].parts[0].text, "hello");
  assert.deepEqual(geminiBody.generationConfig, {
    maxOutputTokens: 321,
    temperature: 0.4,
  });
  assert.equal(geminiResponse.message.content, "hi");
  assert.equal(geminiResponse.usage.total_tokens, 6);
});

test("Node adapters can resolve explicit environment keys", async () => {
  const chat = createOpenAIChat({
    model: "test",
    apiKeyEnv: "MY_KEY",
    env: { MY_KEY: "from-env" },
    fetch: fakeFetch((_url, init) => {
      assert.equal(init.headers?.authorization, "Bearer from-env");
      return jsonResponse({
        choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
      });
    }),
  });
  await chat({ model: "test", messages: [{ role: "user", content: "hi" }] });
});

test("Pi model adapter drains pipes and supports timeout, abort, and forced kill", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gepago-pi-"));
  const command = join(directory, "fake-pi");
  await writeFile(
    command,
    "#!/bin/sh\ncase \"$PI_MODE\" in\n  drain) printf early; (sleep 0.03; printf late) & exit 0 ;;\n  hang) trap '' TERM; while :; do sleep 1; done ;;\n  *) for last do :; done; printf '%s' \"$last\" ;;\nesac\n",
  );
  await chmod(command, 0o755);

  const piModel = createPiModel({
    command,
    env: { ...process.env, PI_MODE: "drain" },
  }) as unknown as PredictCallback;
  const result = await piModel(modelRequest("pi-test"));
  assert.equal(result.text, "earlylate");

  await assert.rejects(
    (
      createPiModel({
        command,
        env: { ...process.env, PI_MODE: "hang" },
        timeoutMs: 20,
        killTimeoutMs: 20,
      }) as unknown as PredictCallback
    )(modelRequest("timeout")),
    /timed out/,
  );

  const controller = new AbortController();
  const aborted = (
    createPiModel({
      command,
      env: { ...process.env, PI_MODE: "hang" },
      killTimeoutMs: 20,
      signal: controller.signal,
    }) as unknown as PredictCallback
  )(modelRequest("abort"));
  setTimeout(() => controller.abort(), 20);
  await assert.rejects(aborted, /aborted/);
});
