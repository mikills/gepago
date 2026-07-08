# Crucible

Crucible is an independently installable GEPAgo program for building and running evaluations over model-like systems.

It evaluates subjects such as raw models, GEPA programs, compiled programs, agents, endpoints, or custom Go functions by turning each run into observable evidence and scoring that evidence with evaluators.

## Install CLI

```bash
go install github.com/mikills/gepago/programs/crucible/cmd/crucible@latest
```

## Use as a Go package

```go
import "github.com/mikills/gepago/programs/crucible"
```

## Built-ins

Built-ins are runnable evaluator programs that demonstrate reusable evaluation patterns.

```bash
crucible built-ins list
crucible built-ins info tool-choice
crucible models list -registry built-ins/model-registry.example.json
crucible dashboard -store .crucible
go run ./built-ins/tool-choice
go run ./built-ins/json-reliability
```

Current built-ins:

- `tool-choice`: evaluates whether an agent calls `Analyze file` with the right file arguments, gathers multi-file context when needed, avoids tools for trivial prompts, and returns grounded final answers.
- `json-reliability`: evaluates whether subjects reliably return valid JSON with required fields.

The tool-choice built-in uses declarative expectations over observable run evidence:

```go
Expectations: []crucible.Expectation{
    {Select: "tool_calls.names", Should: "contains_sequence", Value: []string{"Analyze file"}},
    {Select: "tool_calls.arguments.path", Should: "contains", Value: "payments.go"},
    {Select: "final", Should: "contains", Value: "idempotency"},
}
```

## Model registry, costs, and dashboard

Crucible can enrich runs with model metadata and estimated token cost from a user-owned registry:

```bash
crucible run \
  -suite built-ins/json-reliability/suite.json \
  -subjects built-ins/json-reliability/subjects.static.json \
  -registry built-ins/model-registry.example.json \
  -out .crucible
```

The run command writes JSON/CSV/HTML artifacts plus a local run index and dashboard:

```bash
crucible dashboard -store .crucible
open .crucible/dashboard.html
```

Inspect registry entries with:

```bash
crucible models list -registry built-ins/model-registry.example.json
crucible models info openai/gpt-4o-mini -registry built-ins/model-registry.example.json
```

## Evaluator packs

Crucible includes starter packs that are just helpers returning `[]WeightedEvaluator`:

```go
evals := []crucible.WeightedEvaluator{}
evals = append(evals, crucible.ToolUsePack(1)...)
evals = append(evals, crucible.StructuredOutputPack([]string{"answer", "confidence"}, 1)...)
evals = append(evals, crucible.CostLatencyPack(2*time.Second, 4096, 0.5)...)
```

Use packs as defaults, then add your own evaluators for domain-specific behavior.

## Build a custom evaluator

Custom evaluators implement one small interface:

```go
type Evaluator interface {
    Name() string
    Evaluate(context.Context, crucible.EvalInput) (crucible.Score, error)
}
```

Example:

```go
type GroundedAnswerEvaluator struct{}

func (GroundedAnswerEvaluator) Name() string { return "grounded_answer" }

func (GroundedAnswerEvaluator) Evaluate(
    ctx context.Context,
    input crucible.EvalInput,
) (crucible.Score, error) {
    final := fmt.Sprint(input.Output.Value["final"])
    toolOutputs := fmt.Sprint(input.Output.ToolCalls)
    if strings.Contains(toolOutputs, "idempotency") && strings.Contains(final, "idempotency") {
        return crucible.Score{Score: 1, Feedback: "answer is grounded in analyzed file context"}, nil
    }
    return crucible.Score{Score: 0, Feedback: "answer is not grounded in analyzed file context"}, nil
}
```

Then plug it into a run:

```go
result, err := crucible.Run(ctx, crucible.RunConfig{
    Suite: suite,
    Subjects: []crucible.Subject{subjectA, subjectB},
    Evaluators: []crucible.WeightedEvaluator{
        {Evaluator: crucible.ExpectationEvaluator{}, Weight: 1},
        {Evaluator: GroundedAnswerEvaluator{}, Weight: 1},
    },
})
```

For inline checks, use `FuncEvaluator`:

```go
custom := crucible.FuncEvaluator{
    EvaluatorName: "mentions_required_term",
    Func: func(ctx context.Context, input crucible.EvalInput) (crucible.Score, error) {
        final := fmt.Sprint(input.Output.Value["final"])
        if strings.Contains(final, "ledger") {
            return crucible.Score{Score: 1, Feedback: "mentions ledger"}, nil
        }
        return crucible.Score{Score: 0, Feedback: "missing ledger"}, nil
    },
}
```

Custom evaluators can inspect:

- `input.Case` for input, rubrics, expected output, metadata, and expectations
- `input.Output.Value` for structured final values
- `input.Output.Raw` for raw model text
- `input.Output.ToolCalls` for tool names, arguments, outputs, and errors
- `input.Output.Trace` for subject-specific traces
- `input.Output.Usage` and `input.Output.Latency` for operational scoring
- `input.Output.Metadata` for adapter-specific evidence

## Dynamic JSON evals

You can run evals from JSON without writing Go code.

A suite defines the cases and evaluators:

```json
{
  "name": "json-reliability",
  "cases": [
    {"id": "capital", "input": {"prompt": "What is the capital of France? Return JSON."}}
  ],
  "evaluators": [
    {"type": "json_validity", "required_fields": ["answer", "confidence"], "weight": 1}
  ]
}
```

Subjects define what to run. OpenAI-compatible model subject:

```json
{
  "subjects": [
    {
      "type": "openai",
      "name": "gpt-json-candidate",
      "model": "gpt-4o-mini",
      "api_key_env": "OPENAI_API_KEY",
      "parse_json": true,
      "prompt_template_lines": [
        "{{.Input.prompt}}",
        "Return only valid JSON with fields: answer (string), confidence (number)."
      ]
    }
  ]
}
```

Provider-backed agent subject. The built-in agent provider supports `openai`, `openai-compatible`, `anthropic`, and `google`; users can register their own subject type names in Go for custom agents. OpenAI-compatible agents default to Chat Completions (`provider_api: "chat_completions"`) and can opt into the Responses API with `provider_api: "responses"`.

```json
{
  "subjects": [
    {
      "type": "agent",
      "provider": "openai",
      "name": "gpt-tool-agent",
      "model": "gpt-4o-mini",
      "api_key_env": "OPENAI_API_KEY",
      "input_message_field": "prompt",
      "system_prompt": "Use Analyze file when source-file context is required.",
      "tools": [
        {
          "name": "Analyze file",
          "description": "Read and analyze a specific source file before answering.",
          "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
          },
          "output": "payments.go: ChargeCustomer validates idempotency keys before capture."
        }
      ]
    }
  ]
}
```

In Go, users can register their own JSON subject type names:

```go
crucible.RegisterSubjectType("my-agent", func(
    spec crucible.SubjectSpec,
    options crucible.SubjectBuildOptions,
) (crucible.Subject, error) {
    return MyAgentSubject{Spec: spec}, nil
})
```

Then JSON can use that name:

```json
{"subjects": [{"type": "my-agent", "name": "candidate-agent"}]}
```

For OpenAI-compatible Responses API providers, add:

```json
{
  "type": "agent",
  "provider": "openai-compatible",
  "provider_api": "responses",
  "name": "responses-agent"
}
```

Anthropic agent subject uses the same shape with `"provider": "anthropic"` and `ANTHROPIC_API_KEY`. Google/Gemini agent subjects use `"provider": "google"` and `GOOGLE_API_KEY`.

HTTP endpoint subject:

```json
{
  "subjects": [
    {
      "type": "http",
      "name": "local-app",
      "url": "http://localhost:8080/eval",
      "parse_json": true
    }
  ]
}
```

Command subject. Crucible sends case input as JSON on stdin and reads stdout:

```json
{
  "subjects": [
    {
      "type": "command",
      "name": "local-script",
      "command": "python3",
      "args": ["./answer.py"],
      "parse_json": true
    }
  ]
}
```

Run it:

```bash
crucible run \
  -suite built-ins/json-reliability/suite.json \
  -subjects built-ins/json-reliability/subjects.openai.example.json \
  -out .crucible \
  -repeat 3 \
  -concurrency 2 \
  -cache-dir .crucible/cache \
  -fail-below 0.9
```

For a no-API smoke test:

```bash
crucible run \
  -suite built-ins/json-reliability/suite.json \
  -subjects built-ins/json-reliability/subjects.static.json
```

## CLI

```bash
crucible validate-suite suite.json
crucible run -suite suite.json -subjects subjects.json -out .crucible -repeat 3 -concurrency 2 -cache-dir .crucible/cache -fail-below 0.9
crucible compare -baseline old.json -candidate new.json -fail-drop 0.05
crucible render-report -run run.json -html report.html -csv summary.csv
crucible built-ins list
crucible built-ins info tool-choice
```
