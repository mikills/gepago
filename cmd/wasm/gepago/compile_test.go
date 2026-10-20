package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
)

func TestCompileWireDecode(t *testing.T) {
	payload := `{
		"program": {"signature": {"inputs": [{"name": "text"}], "outputs": [{"name": "label"}]}, "maxParseRetries": 1},
		"trainset": [{"id": "train-1", "input": {"text": "great"}, "expected": {"label": "positive"}}],
		"valset": [{"id": "val-1", "input": {"text": "bad"}, "expected": {"label": "negative"}}],
		"metric": {"kind": "exactMatch", "fields": ["label"], "caseInsensitive": true},
		"components": ["instruction"], "maxMetricCalls": 4, "minibatchSize": 1, "seed": 7, "acceptance": "equal"
	}`

	wire, err := decodeCompileWireConfig(payload)
	if err != nil {
		t.Fatal(err)
	}
	if wire.Program == nil || wire.Program.MaxParseRetries == nil || *wire.Program.MaxParseRetries != 1 {
		t.Fatalf("program was not decoded: %+v", wire.Program)
	}
	if len(wire.Trainset) != 1 || len(wire.Valset) != 1 || wire.Metric.Kind != "exactMatch" {
		t.Fatalf("wire config was not decoded: %+v", wire)
	}

	examples := compileExamples(wire.Trainset)
	if len(examples) != 1 {
		t.Fatalf("got %d examples, want one", len(examples))
	}
	ioExample, ok := examples[0].Input.(gepa.IOExample)
	if !ok {
		t.Fatalf("example input has type %T, want gepa.IOExample", examples[0].Input)
	}
	if ioExample.Inputs["text"] != "great" || ioExample.Expected["label"] != "positive" {
		t.Fatalf("unexpected IOExample: %+v", ioExample)
	}
}

func TestCompileDefaultsMatchGoConfig(t *testing.T) {
	payload := map[string]any{
		"program": map[string]any{
			"signature": gepa.Signature{Inputs: []gepa.Field{{Name: "text"}}, Outputs: []gepa.Field{{Name: "label"}}},
		},
		"trainset": []map[string]any{
			{"id": "one", "input": map[string]any{"text": "a"}, "expected": map[string]any{"label": "x"}},
		},
		"metric": map[string]any{"kind": "exactMatch", "fields": []string{"label"}},
	}
	wire, err := decodeCompileWireConfig(mustJSON(t, payload))
	if err != nil {
		t.Fatal(err)
	}
	config, err := buildCompileConfig(
		wire,
		compileModelInvokerFunc(func(context.Context, compileModelRequest) (compileModelResult, error) {
			return compileModelResult{Text: `{"label":"x"}`}, nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if config.MinibatchSize != 1 || config.MaxMetricCalls != 25 {
		t.Fatalf("defaults = minibatch %d, metric calls %d; want 1, 25", config.MinibatchSize, config.MaxMetricCalls)
	}
}

func TestCompileWireRejects(t *testing.T) {
	base := `{"program":{"signature":{"inputs":[{"name":"text"}],"outputs":[{"name":"label"}]}},"trainset":[{"id":"x","input":{"text":"x"},"expected":{"label":"x"}}],"metric":{"kind":"%s"},"acceptance":"%s"}`
	for _, test := range []struct {
		name       string
		metricKind string
		acceptance string
		want       string
	}{
		{name: "metric", metricKind: "custom", acceptance: "strict", want: "unsupported compile metric"},
		{name: "acceptance", metricKind: "exactMatch", acceptance: "maybe", want: "acceptance"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := decodeCompileWireConfig(fmt.Sprintf(base, test.metricKind, test.acceptance))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want substring %q", err, test.want)
			}
		})
	}
}

func TestCompileWithFakeHost(t *testing.T) {
	var requests []compileModelRequest
	invoker := compileModelInvokerFunc(
		func(_ context.Context, request compileModelRequest) (compileModelResult, error) {
			requests = append(requests, request)
			response := compileModelResult{Usage: gepa.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2}}
			switch request.Role {
			case "task":
				response.Text = `{"label":"positive"}`
			case "reflector":
				response.Text = "Keep the instruction concise."
			default:
				return compileModelResult{}, fmt.Errorf("unexpected role %q", request.Role)
			}
			return response, nil
		},
	)
	payload := map[string]any{
		"program": map[string]any{
			"signature":   gepa.Signature{Inputs: []gepa.Field{{Name: "text"}}, Outputs: []gepa.Field{{Name: "label"}}},
			"instruction": "Classify the text.",
		},
		"trainset": []map[string]any{
			{
				"id":       "train-1",
				"input":    map[string]any{"text": "great"},
				"expected": map[string]any{"label": "positive"},
			},
		},
		"metric":         map[string]any{"kind": "exactMatch", "fields": []string{"label"}},
		"components":     []string{"instruction"},
		"maxMetricCalls": 3,
		"minibatchSize":  1,
		"seed":           11,
	}

	value, err := runCompileOperation(context.Background(), mustJSON(t, payload), invoker)
	if err != nil {
		t.Fatal(err)
	}
	if value.Candidate[gepa.InstructionComponent] != "Classify the text." {
		t.Fatalf("candidate = %+v", value.Candidate)
	}
	if value.State.MetricCalls != 3 || len(value.State.Candidates) == 0 {
		t.Fatalf(
			"unexpected optimizer state: metric_calls=%d candidates=%d",
			value.State.MetricCalls,
			len(value.State.Candidates),
		)
	}
	if value.State.Ledger.TotalTokens < 8 {
		t.Fatalf("host usage was not accounted for: %+v", value.State.Ledger)
	}
	if len(requests) != 4 {
		t.Fatalf("got %d host requests, want task seed/before/after plus reflector", len(requests))
	}
	for _, request := range requests {
		if request.Kind != compileModelRequestKind || request.Attempt < 1 || request.Prompt == "" {
			t.Fatalf("invalid host request: %+v", request)
		}
	}
	if requests[0].Role != "task" || requests[1].Role != "task" ||
		requests[2].Role != "reflector" || requests[3].Role != "task" {
		t.Fatalf("unexpected role tags: %+v", requests)
	}
}

func TestCompileDeclarativePrograms(t *testing.T) {
	signature := gepa.Signature{Inputs: []gepa.Field{{Name: "text"}}, Outputs: []gepa.Field{{Name: "label"}}}
	cotPayload := map[string]any{
		"program": map[string]any{
			"kind":           "chainOfThought",
			"signature":      signature,
			"reasoningField": "thoughts",
			"instruction":    "Classify.",
		},
		"trainset": []map[string]any{
			{"id": "cot", "input": map[string]any{"text": "great"}, "expected": map[string]any{"label": "positive"}},
		},
		"metric": map[string]any{"kind": "exactMatch", "fields": []string{"label"}}, "maxMetricCalls": 1,
	}
	cot, err := runCompileOperation(
		context.Background(),
		mustJSON(t, cotPayload),
		compileModelInvokerFunc(func(_ context.Context, request compileModelRequest) (compileModelResult, error) {
			if request.Role != "task" {
				return compileModelResult{}, fmt.Errorf("unexpected role %q", request.Role)
			}
			return compileModelResult{Text: `{"label":"positive","thoughts":"matched"}`}, nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(cot.Candidate[gepa.InstructionComponent], "thoughts field") {
		t.Fatalf("COT candidate did not use ToPredict instruction: %+v", cot.Candidate)
	}

	pipelinePayload := map[string]any{
		"program": map[string]any{
			"kind": "pipeline", "returnAll": true,
			"steps": []map[string]any{
				{"name": "extract", "program": map[string]any{
					"signature": signatureForTest(
						[]string{"text"},
						[]string{"topic"},
					), "instruction": "Extract.", "outputPrefix": "extract.",
				}},
				{"name": "decide", "program": map[string]any{
					"signature": signatureForTest(
						[]string{"extract.topic"},
						[]string{"label"},
					), "instruction": "Decide.", "inputKeys": []string{"extract.topic"},
				}},
			},
		},
		"trainset": []map[string]any{
			{
				"id":       "pipeline",
				"input":    map[string]any{"text": "great"},
				"expected": map[string]any{"label": "positive"},
			},
		},
		"metric": map[string]any{"kind": "exactMatch", "fields": []string{"label"}}, "maxMetricCalls": 1,
	}
	pipeline, err := runCompileOperation(
		context.Background(),
		mustJSON(t, pipelinePayload),
		compileModelInvokerFunc(func(_ context.Context, request compileModelRequest) (compileModelResult, error) {
			if strings.Contains(request.Prompt, "output fields: topic") {
				return compileModelResult{Text: `{"topic":"service"}`}, nil
			}
			return compileModelResult{Text: `{"label":"positive"}`}, nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if pipeline.Candidate["extract.instruction"] != "Extract." ||
		pipeline.Candidate["decide.instruction"] != "Decide." {
		t.Fatalf("pipeline candidate was not scoped: %+v", pipeline.Candidate)
	}

	for name, payload := range map[string]any{
		"kind":           map[string]any{"program": map[string]any{"kind": "unknown", "signature": signature}, "trainset": []any{map[string]any{"id": "x", "input": map[string]any{"text": "x"}, "expected": map[string]any{"label": "x"}}}, "metric": map[string]any{"kind": "exactMatch"}},
		"duplicate step": map[string]any{"program": map[string]any{"kind": "pipeline", "steps": []any{map[string]any{"name": "step", "program": map[string]any{"signature": signature}}, map[string]any{"name": "step", "program": map[string]any{"signature": signature}}}}, "trainset": []any{map[string]any{"id": "x", "input": map[string]any{"text": "x"}, "expected": map[string]any{"label": "x"}}}, "metric": map[string]any{"kind": "exactMatch"}},
	} {
		if _, err := decodeCompileWireConfig(mustJSON(t, payload)); err == nil {
			t.Errorf("%s: expected invalid configuration", name)
		}
	}
}

func signatureForTest(inputs, outputs []string) gepa.Signature {
	signature := gepa.Signature{}
	for _, name := range inputs {
		signature.Inputs = append(signature.Inputs, gepa.Field{Name: name})
	}
	for _, name := range outputs {
		signature.Outputs = append(signature.Outputs, gepa.Field{Name: name})
	}
	return signature
}

func TestCompileJudgeUsage(t *testing.T) {
	var roles []string
	invoker := compileModelInvokerFunc(
		func(_ context.Context, request compileModelRequest) (compileModelResult, error) {
			roles = append(roles, request.Role)
			response := compileModelResult{Usage: gepa.Usage{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3}}
			if request.Role == "task" {
				response.Text = `{"label":"positive"}`
			} else if request.Role == "judge" {
				response.Text = `{"score":1,"feedback":"matches"}`
			} else {
				return compileModelResult{}, fmt.Errorf("unexpected role %q", request.Role)
			}
			return response, nil
		},
	)
	payload := map[string]any{
		"program": map[string]any{
			"signature": gepa.Signature{Inputs: []gepa.Field{{Name: "text"}}, Outputs: []gepa.Field{{Name: "label"}}},
		},
		"trainset": []map[string]any{
			{
				"id":       "train-1",
				"input":    map[string]any{"text": "great"},
				"expected": map[string]any{"label": "positive"},
			},
		},
		"metric":         map[string]any{"kind": "llmJudge", "rubric": "score whether labels match", "role": "judge"},
		"maxMetricCalls": 1,
		"minibatchSize":  1,
	}

	value, err := runCompileOperation(context.Background(), mustJSON(t, payload), invoker)
	if err != nil {
		t.Fatal(err)
	}
	if len(roles) != 2 || roles[0] != "task" || roles[1] != "judge" {
		t.Fatalf("roles = %v, want task then judge", roles)
	}
	if value.State.Ledger.TotalTokens != 6 {
		t.Fatalf("judge usage was not included in optimizer accounting: %+v", value.State.Ledger)
	}
}

func TestCompileLanguageModelReportsHostUsage(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var requests []compileModelRequest
	model := newCompileLanguageModel(compileModelInvokerFunc(
		func(_ context.Context, request compileModelRequest) (compileModelResult, error) {
			requests = append(requests, request)
			if request.Attempt == 2 {
				cancel()
			}
			return compileModelResult{Text: "ok", Usage: gepa.Usage{TotalTokens: 5, ReasoningTokens: 2}}, nil
		},
	), "program")

	got, err := model.Generate(context.Background(), "prompt")
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	want := gepa.Generation{Text: "ok", Usage: gepa.Usage{TotalTokens: 5, ReasoningTokens: 2}}
	if got != want {
		t.Fatalf("Generate() = %+v, want %+v", got, want)
	}

	got, err = model.Generate(ctx, "prompt")
	if err != context.Canceled {
		t.Fatalf("Generate() error = %v, want context.Canceled", err)
	}
	if got != (gepa.Generation{Usage: want.Usage}) {
		t.Fatalf("Generate() after the host spent tokens = %+v, want only its usage", got)
	}

	got, err = model.Generate(ctx, "prompt")
	if err != context.Canceled || got != (gepa.Generation{}) || len(requests) != 2 {
		t.Fatalf("Generate() on a cancelled context = %+v, %v after %d host calls", got, err, len(requests))
	}
	if requests[0].Role != "program" || requests[1].Attempt != 2 {
		t.Fatalf("requests = %+v", requests)
	}
}

func TestCompileLanguageModelRequiresInvoker(t *testing.T) {
	var model *compileLanguageModel
	if _, err := model.Generate(context.Background(), "prompt"); err == nil {
		t.Fatal("Generate() error = nil, want missing host model error")
	}
	hostErr := errors.New("host failed")
	model = newCompileLanguageModel(compileModelInvokerFunc(
		func(context.Context, compileModelRequest) (compileModelResult, error) {
			return compileModelResult{}, hostErr
		},
	), "task")
	if _, err := model.Generate(context.Background(), "prompt"); !errors.Is(err, hostErr) {
		t.Fatalf("Generate() error = %v, want host error", err)
	}
}

func TestParseHostFinishReason(t *testing.T) {
	for _, value := range []string{"", "stop", "length", "tool_calls", "content_filter"} {
		got, err := parseHostFinishReason(value)
		if err != nil || string(got) != value {
			t.Fatalf("parseHostFinishReason(%q) = %q, %v", value, got, err)
		}
	}
	if _, err := parseHostFinishReason("MAX_TOKENS"); err == nil {
		t.Fatal("parseHostFinishReason accepted a provider-native reason")
	}
}

func TestCompileTreatsTruncatedHostRepliesAsInvalidOutput(t *testing.T) {
	var roles []string
	invoker := compileModelInvokerFunc(
		func(_ context.Context, request compileModelRequest) (compileModelResult, error) {
			roles = append(roles, request.Role)
			return compileModelResult{
				Text:         `{"label":"positive"}`,
				Usage:        gepa.Usage{TotalTokens: 4},
				FinishReason: gepa.FinishLength,
				Model:        "host-model",
			}, nil
		},
	)
	model := newCompileLanguageModel(invoker, "task")
	got, err := model.Generate(context.Background(), "prompt")
	if err != nil || got.FinishReason != gepa.FinishLength || got.Model != "host-model" {
		t.Fatalf("Generate() = %+v, %v; want the host finish reason and model", got, err)
	}

	payload := map[string]any{
		"program": map[string]any{
			"signature": map[string]any{
				"inputs":  []map[string]any{{"name": "text"}},
				"outputs": []map[string]any{{"name": "label"}},
			},
			"maxParseRetries": 1,
		},
		"trainset": []map[string]any{{
			"id": "one", "input": map[string]any{"text": "great"}, "expected": map[string]any{"label": "positive"},
		}},
		"metric":         map[string]any{"kind": "exactMatch", "fields": []string{"label"}},
		"maxMetricCalls": 1,
	}
	roles = nil

	value, err := runCompileOperation(context.Background(), mustJSON(t, payload), invoker)

	if err != nil {
		t.Fatal(err)
	}
	if len(roles) != 1 || roles[0] != "task" {
		t.Fatalf("roles = %v, want one task call and no repair", roles)
	}
	if score := value.State.Candidates[0].ValidationScore; score != 0 {
		t.Fatalf("seed score = %v, want 0 for a truncated reply", score)
	}
	if value.State.Ledger.TotalTokens != 4 {
		t.Fatalf("ledger = %+v, want the truncated call's usage", value.State.Ledger)
	}
}

func TestCompileLanguageModelPassesTheResponseFormatToTheHost(t *testing.T) {
	var requests []compileModelRequest
	model := newCompileLanguageModel(compileModelInvokerFunc(
		func(_ context.Context, request compileModelRequest) (compileModelResult, error) {
			requests = append(requests, request)
			return compileModelResult{Text: `{"label":"positive"}`}, nil
		},
	), "task")
	var _ gepa.StructuredGenerator = model
	format := gepa.ResponseFormat{Name: "sentiment", Schema: []byte(`{"type":"object"}`), Strict: true}

	if _, err := model.GenerateStructured(context.Background(), "prompt", format); err != nil {
		t.Fatal(err)
	}
	if _, err := model.Generate(context.Background(), "prompt"); err != nil {
		t.Fatal(err)
	}

	want := hostResponseFormat{Name: "sentiment", Schema: format.Schema, Strict: true}
	if got := requests[0].ResponseFormat; got == nil || got.Name != want.Name ||
		string(got.Schema) != string(want.Schema) ||
		!got.Strict {
		t.Fatalf("structured request response_format = %+v, want %+v", got, want)
	}
	if requests[1].ResponseFormat != nil {
		t.Fatalf("plain request response_format = %+v, want none", requests[1].ResponseFormat)
	}
	encoded := mustJSON(t, requests[0])
	if !strings.Contains(encoded, `"response_format":{"name":"sentiment","schema":{"type":"object"},"strict":true}`) {
		t.Fatalf("encoded request = %s", encoded)
	}
	if strings.Contains(mustJSON(t, requests[1]), "response_format") {
		t.Fatalf("plain request carries a response format: %s", mustJSON(t, requests[1]))
	}
}

func TestCompileSendsTheResponseFormatOnTaskRequests(t *testing.T) {
	var requests []compileModelRequest
	invoker := compileModelInvokerFunc(
		func(_ context.Context, request compileModelRequest) (compileModelResult, error) {
			requests = append(requests, request)
			return compileModelResult{Text: `{"label":"positive"}`}, nil
		},
	)
	payload := map[string]any{
		"program": map[string]any{
			"signature": map[string]any{
				"name":    "sentiment",
				"inputs":  []map[string]any{{"name": "text"}},
				"outputs": []map[string]any{{"name": "label", "type": "string"}},
			},
		},
		"trainset": []map[string]any{{
			"id": "one", "input": map[string]any{"text": "great"}, "expected": map[string]any{"label": "positive"},
		}},
		"metric":         map[string]any{"kind": "exactMatch", "fields": []string{"label"}},
		"maxMetricCalls": 1,
	}

	if _, err := runCompileOperation(context.Background(), mustJSON(t, payload), invoker); err != nil {
		t.Fatal(err)
	}
	if len(requests) == 0 || requests[0].Role != "task" || requests[0].ResponseFormat == nil ||
		!requests[0].ResponseFormat.Strict || requests[0].ResponseFormat.Name != "sentiment" {
		t.Fatalf("task requests = %+v, want a strict response format", requests)
	}
}
