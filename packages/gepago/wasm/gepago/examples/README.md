# Node flow examples

The SDK runtime supports Node 22.18+. These examples use native Node TypeScript
execution, which is available in current Node releases (including Node 25); no
`tsx` loader is required. Native execution strips types but does not type-check
or support every TypeScript feature.

## Extract a document into a knowledge graph

Build the package once before running the example so its compiled `dist/`
entrypoint exists:

```bash
(cd packages/gepago/wasm && npm ci && npm run build)
```

This flow uses the Node SDK to:

1. load a text or Markdown document;
2. split it into chunks;
3. call `gepago.predict` for structured entity and relationship extraction;
4. merge and de-duplicate facts into graph `nodes` and `edges`.

Run it with the configured Pi model:

```bash
node packages/gepago/wasm/gepago/examples/document-knowledge-graph.ts \
  --input packages/gepago/wasm/gepago/examples/fixtures/acme-robotics.md \
  --out /tmp/acme-graph.json
```

Or use an OpenAI-compatible model:

```bash
OPENAI_API_KEY=... node packages/gepago/wasm/gepago/examples/document-knowledge-graph.ts \
  --provider openai --model gpt-4.1-mini --out /tmp/acme-graph.json
```

## Run GEPA optimization first

The extraction instruction can be optimized with the same SDK compile bridge. The
example uses two labeled graph snippets, a host graph F1 metric, and then runs
the best candidate over the full document:

```bash
node packages/gepago/wasm/gepago/examples/document-knowledge-graph.ts \
  --compile --out /tmp/acme-graph-optimized.json
```

Compilation makes additional model calls. `--no-compile` (the default) runs the
seed extraction instruction only. For an offline smoke test:

```bash
node packages/gepago/wasm/gepago/examples/document-knowledge-graph.ts \
  --mock --no-compile --out /tmp/acme-graph.json
```
