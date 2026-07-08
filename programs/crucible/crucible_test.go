package crucible

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRun(t *testing.T) {
	t.Run("summarizes subject scores", func(t *testing.T) {
		suite := Suite{
			Name: "sentiment-fit",
			Cases: []EvalCase{
				{
					ID:       "positive",
					Input:    gepa.Prediction{"text": "great"},
					Expected: gepa.Prediction{"label": "positive"},
				},
				{
					ID:       "negative",
					Input:    gepa.Prediction{"text": "bad"},
					Expected: gepa.Prediction{"label": "negative"},
				},
			},
		}
		good := staticSubject("good", func(input gepa.Prediction) gepa.Prediction {
			if input["text"] == "great" {
				return gepa.Prediction{"label": "positive"}
			}
			return gepa.Prediction{"label": "negative"}
		})
		bad := staticSubject("bad", func(gepa.Prediction) gepa.Prediction {
			return gepa.Prediction{"label": "positive"}
		})
		result, err := Run(context.Background(), RunConfig{
			Suite:      suite,
			Subjects:   []Subject{good, bad},
			Evaluators: []WeightedEvaluator{Classification("label")},
			RunID:      "test-run",
		})
		require.NoError(t, err)
		require.Len(t, result.Results, 4)
		assert.Equal(t, float64(1), findSummary(result.Summary, "good").AverageScore)
		assert.Equal(t, 0.5, findSummary(result.Summary, "bad").AverageScore)
	})
}

func TestJudges(t *testing.T) {
	t.Run("rubric and pairwise", func(t *testing.T) {
		judge := gepa.LanguageModelFunc(func(_ context.Context, prompt string) (string, error) {
			if strings.Contains(prompt, "Compare two subject outputs") {
				return `{"winner":"a","score_a":0.9,"score_b":0.2,"feedback":"A is clearer"}`, nil
			}
			return `{"score":0.8,"feedback":"good"}`, nil
		})
		suite := Suite{
			Name: "open-ended",
			Cases: []EvalCase{{
				ID:     "explain",
				Input:  gepa.Prediction{"question": "why retries need idempotency"},
				Rubric: "Prefer correct, concise answers.",
			}},
		}
		result, err := Run(context.Background(), RunConfig{
			Suite: suite,
			Subjects: []Subject{
				valueSubject("a", gepa.Prediction{"answer": "clear"}),
				valueSubject("b", gepa.Prediction{"answer": "vague"}),
			},
			Evaluators: []WeightedEvaluator{{Evaluator: RubricJudgeEvaluator{LM: judge}, Weight: 1}},
			PairwiseEvaluators: []WeightedPairwiseEvaluator{{
				Evaluator: PairwiseJudgeEvaluator{LM: judge},
				Weight:    1,
			}},
		})
		require.NoError(t, err)
		require.Len(t, result.Pairwise, 1)
		assert.Equal(t, "a", result.Pairwise[0].Scores[0].Winner)
	})
}

func TestToolCalls(t *testing.T) {
	result, err := ToolCallEvaluator{RequireOrder: true}.Evaluate(context.Background(), EvalInput{
		Case: EvalCase{Metadata: map[string]any{
			expectedToolCallsKey: []string{"Analyze file", "Read dependency"},
		}},
		Output: SubjectOutput{Metadata: map[string]any{
			toolCallsKey: []ToolCallSummary{{Name: "Analyze file"}, {Name: "Read dependency"}},
		}},
	})
	require.NoError(t, err)
	assert.Equal(t, float64(1), result.Score)
}

func TestExpectations(t *testing.T) {
	output := SubjectOutput{
		Value: gepa.Prediction{"final": "ChargeCustomer uses idempotency."},
		Raw:   `{"answer":"ok","confidence":0.9}`,
		ToolCalls: []ToolCallSummary{{
			Name:           "Analyze file",
			ArgumentsValue: map[string]any{"path": "payments.go"},
		}},
		Latency: 10 * time.Millisecond,
	}
	cases := []Expectation{
		{Select: "final", Should: "regex", Value: "idempotency"},
		{Select: "final", Should: "starts_with", Value: "ChargeCustomer"},
		{Select: "tool_calls.names", Should: "contains_any", Value: []string{"Analyze file", "Search files"}},
		{Select: "raw", Should: "json_schema", Value: map[string]any{
			"type":     "object",
			"required": []string{"answer", "confidence"},
		}},
		{Select: "latency", Should: "latency_below", Value: "20ms"},
		{Select: toolCallsKey, Should: "tool_args_match", Value: map[string]any{
			"name":      "Analyze file",
			"arguments": map[string]any{"path": "payments.go"},
		}},
		{Select: "value", Should: "json_schema", Value: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"final": map[string]any{"type": "string", "pattern": "idempotency"},
			},
		}},
	}
	for _, expectation := range cases {
		t.Run(expectation.Select+" "+expectation.Should, func(t *testing.T) {
			actual := selectObservation(output, expectation.Select)
			assert.Equal(t, float64(1), matchExpectation(actual, expectation))
		})
	}
}

func TestBuiltIns(t *testing.T) {
	builtIns, err := BuiltIns()
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(builtIns), 2)
	_, ok, err := BuiltInInfo("tool-choice")
	require.NoError(t, err)
	assert.True(t, ok)
}

func TestLoadSuiteJSONAndBuildEvaluators(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "suite.json")
	data := `{
		"name":"json-suite",
		"cases":[{"id":"case-1","input":{"text":"hello"},"expected":{"text":"hello"}}],
		"evaluators":[{"type":"exact_match","fields":["text"],"weight":2}]
	}`
	require.NoError(t, os.WriteFile(path, []byte(data), publicFileMode))
	suite, err := LoadSuiteJSON(path)
	require.NoError(t, err)
	evaluators, err := BuildEvaluators(suite.Evaluators, EvaluatorFactoryConfig{})
	require.NoError(t, err)
	require.Len(t, evaluators, 1)
	assert.Equal(t, float64(2), evaluators[0].Weight)
}

func TestSecurityOptions(t *testing.T) {
	_, err := BuildSubjects(SubjectConfig{Subjects: []SubjectSpec{{Type: "command", Command: "echo"}}})
	require.Error(t, err)
	_, err = BuildSubjectsWithOptions(
		SubjectConfig{Subjects: []SubjectSpec{{Type: "command", Command: "echo"}}},
		SubjectBuildOptions{AllowCommand: true},
	)
	require.NoError(t, err)
	_, err = BuildEvaluators([]EvaluatorSpec{{Type: "command", Command: "echo"}}, EvaluatorFactoryConfig{})
	require.Error(t, err)
	_, err = BuildEvaluators(
		[]EvaluatorSpec{{Type: "command", Command: "echo"}},
		EvaluatorFactoryConfig{AllowCommand: true},
	)
	require.NoError(t, err)
}

func TestCustomSubjectRegistration(t *testing.T) {
	RegisterSubjectType("custom-test-subject", func(spec SubjectSpec, _ SubjectBuildOptions) (Subject, error) {
		return valueSubject(subjectName(spec), gepa.Prediction{"answer": "registered"}), nil
	})
	subject, err := BuildSubject(SubjectSpec{Type: "custom-test-subject", Name: "custom"})
	require.NoError(t, err)
	output, err := subject.Run(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, "registered", output.Value["answer"])
}

func TestPromptTemplateLines(t *testing.T) {
	spec := SubjectSpec{PromptTemplateLines: []string{"{{.Input.prompt}}", "", "Return JSON."}}
	prompt, err := renderPromptTemplate(subjectPromptTemplate(spec), gepa.Prediction{"prompt": "hello"})
	require.NoError(t, err)
	assert.Equal(t, "hello\n\nReturn JSON.", prompt)
}

func TestConfiguredSubjects(t *testing.T) {
	t.Run("http", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, `{"answer":"ok"}`)
		}))
		defer server.Close()
		subject, err := BuildSubject(SubjectSpec{Type: "http", Name: "http", URL: server.URL, ParseJSON: true})
		require.NoError(t, err)
		output, err := subject.Run(context.Background(), gepa.Prediction{"prompt": "hello"})
		require.NoError(t, err)
		assert.Equal(t, "ok", output.Value["answer"])
	})
	t.Run("command", func(t *testing.T) {
		subject, err := BuildSubject(SubjectSpec{
			Type:      "command",
			Name:      "command",
			Command:   "sh",
			Args:      []string{"-c", "printf '{\"answer\":\"ok\"}'"},
			ParseJSON: true,
		})
		require.NoError(t, err)
		output, err := subject.Run(context.Background(), gepa.Prediction{"prompt": "hello"})
		require.NoError(t, err)
		assert.Equal(t, "ok", output.Value["answer"])
	})
}

func TestProviderAgentSubjects(t *testing.T) {
	t.Run("openai chat completions", testOpenAIAgentSubject)
	t.Run("openai responses", testOpenAIResponsesAgentSubject)
	t.Run("anthropic", testAnthropicAgentSubject)
	t.Run("google", testGoogleAgentSubject)
}

func testOpenAIAgentSubject(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/chat/completions", r.URL.Path)
		var req map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		call := calls.Add(1)
		if call == 1 {
			fmt.Fprint(
				w,
				`{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call-1","type":"function","function":{"name":"Analyze file","arguments":"{\"path\":\"payments.go\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
			)
			return
		}
		messages, ok := req["messages"].([]any)
		require.True(t, ok)
		assert.Contains(t, fmt.Sprint(messages), "tool")
		fmt.Fprint(
			w,
			`{"choices":[{"message":{"role":"assistant","content":"ChargeCustomer uses idempotency."},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16}}`,
		)
	}))
	defer server.Close()
	subject, err := BuildSubject(SubjectSpec{
		Type:              "agent",
		Provider:          "openai",
		Name:              "agent",
		Model:             "gpt-test",
		BaseURL:           server.URL,
		InputMessageField: "prompt",
		Tools: []ToolSpec{{
			Name:        "Analyze file",
			Description: "Analyze a file",
			Output:      "payments.go mentions idempotency",
		}},
	})
	require.NoError(t, err)
	output, err := subject.Run(context.Background(), gepa.Prediction{"prompt": "Explain ChargeCustomer"})
	require.NoError(t, err)
	require.Len(t, output.ToolCalls, 1)
	assert.Equal(t, "Analyze file", output.ToolCalls[0].Name)
	assert.Equal(t, "payments.go", output.ToolCalls[0].ArgumentsValue["path"])
	assert.Equal(t, "ChargeCustomer uses idempotency.", output.Value["final"])
}

func testOpenAIResponsesAgentSubject(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/responses", r.URL.Path)
		var req map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		call := calls.Add(1)
		if call == 1 {
			tools, ok := req["tools"].([]any)
			require.True(t, ok)
			assert.Contains(t, fmt.Sprint(tools), "Analyze file")
			fmt.Fprint(
				w,
				`{"output":[{"type":"function_call","call_id":"call-1","name":"Analyze file","arguments":"{\"path\":\"payments.go\"}"}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`,
			)
			return
		}
		input, ok := req["input"].([]any)
		require.True(t, ok)
		assert.Contains(t, fmt.Sprint(input), "function_call_output")
		fmt.Fprint(
			w,
			`{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ChargeCustomer uses idempotency."}]}],"usage":{"input_tokens":12,"output_tokens":4,"total_tokens":16}}`,
		)
	}))
	defer server.Close()
	subject, err := BuildSubject(SubjectSpec{
		Type:              "agent",
		Provider:          "openai-compatible",
		ProviderAPI:       "responses",
		Name:              "agent",
		Model:             "gpt-test",
		BaseURL:           server.URL,
		InputMessageField: "prompt",
		Tools: []ToolSpec{{
			Name:        "Analyze file",
			Description: "Analyze a file",
			Output:      "payments.go mentions idempotency",
		}},
	})
	require.NoError(t, err)
	output, err := subject.Run(context.Background(), gepa.Prediction{"prompt": "Explain ChargeCustomer"})
	require.NoError(t, err)
	require.Len(t, output.ToolCalls, 1)
	assert.Equal(t, "Analyze file", output.ToolCalls[0].Name)
	assert.Equal(t, "payments.go", output.ToolCalls[0].ArgumentsValue["path"])
	assert.Equal(t, "ChargeCustomer uses idempotency.", output.Value["final"])
}

func testAnthropicAgentSubject(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		call := calls.Add(1)
		if call == 1 {
			fmt.Fprint(
				w,
				`{"content":[{"type":"tool_use","id":"toolu_1","name":"Analyze file","input":{"path":"payments.go"}}],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5}}`,
			)
			return
		}
		messages, ok := req["messages"].([]any)
		require.True(t, ok)
		assert.Contains(t, fmt.Sprint(messages), "tool_result")
		fmt.Fprint(
			w,
			`{"content":[{"type":"text","text":"ChargeCustomer uses idempotency."}],"stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":4}}`,
		)
	}))
	defer server.Close()
	subject, err := BuildSubject(SubjectSpec{
		Type:              "agent",
		Provider:          "anthropic",
		Name:              "agent",
		Model:             "claude-test",
		BaseURL:           server.URL,
		InputMessageField: "prompt",
		Tools: []ToolSpec{{
			Name:        "Analyze file",
			Description: "Analyze a file",
			Output:      "payments.go mentions idempotency",
		}},
	})
	require.NoError(t, err)
	output, err := subject.Run(context.Background(), gepa.Prediction{"prompt": "Explain ChargeCustomer"})
	require.NoError(t, err)
	require.Len(t, output.ToolCalls, 1)
	assert.Equal(t, "Analyze file", output.ToolCalls[0].Name)
	assert.Equal(t, "payments.go", output.ToolCalls[0].ArgumentsValue["path"])
	assert.Equal(t, "ChargeCustomer uses idempotency.", output.Value["final"])
}

func testGoogleAgentSubject(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1beta/models/gemini-test:generateContent", r.URL.Path)
		var req map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		call := calls.Add(1)
		if call == 1 {
			assert.Contains(t, fmt.Sprint(req["tools"]), "Analyze_file")
			fmt.Fprint(
				w,
				`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"Analyze_file","args":{"path":"payments.go"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`,
			)
			return
		}
		assert.Contains(t, fmt.Sprint(req["contents"]), "functionResponse")
		fmt.Fprint(
			w,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"ChargeCustomer uses idempotency."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":4,"totalTokenCount":16}}`,
		)
	}))
	defer server.Close()
	subject, err := BuildSubject(SubjectSpec{
		Type:              "agent",
		Provider:          "google",
		APIKey:            "test-key",
		Name:              "agent",
		Model:             "gemini-test",
		BaseURL:           server.URL,
		InputMessageField: "prompt",
		Tools: []ToolSpec{{
			Name:        "Analyze file",
			Description: "Analyze a file",
			Output:      "payments.go mentions idempotency",
		}},
	})
	require.NoError(t, err)
	output, err := subject.Run(context.Background(), gepa.Prediction{"prompt": "Explain ChargeCustomer"})
	require.NoError(t, err)
	require.Len(t, output.ToolCalls, 1)
	assert.Equal(t, "Analyze file", output.ToolCalls[0].Name)
	assert.Equal(t, "payments.go", output.ToolCalls[0].ArgumentsValue["path"])
	assert.Equal(t, "ChargeCustomer uses idempotency.", output.Value["final"])
}

func TestRegistryAndRunStore(t *testing.T) {
	registry := ModelRegistry{Models: []ModelInfo{{
		ID:                    "gpt-test",
		Provider:              "openai",
		InputPricePerMTokens:  1,
		OutputPricePerMTokens: 2,
	}}}
	result := RunResult{
		RunID:     "run-1",
		SuiteName: "suite",
		StartedAt: time.Now().UTC(),
		EndedAt:   time.Now().UTC(),
		Subjects:  []string{"subject"},
		Cases:     []EvalCase{{ID: "case"}},
		SubjectMetadata: map[string]SubjectMetadata{
			"subject": {Name: "subject", Provider: "openai", Model: "gpt-test"},
		},
		Summary: []SubjectSummary{
			{
				Subject:      "subject",
				Cases:        1,
				AverageScore: 1,
				Usage:        gepa.Usage{PromptTokens: 1000, CompletionTokens: 1000},
			},
		},
	}
	ApplyModelRegistry(&result, registry)
	assert.Equal(t, 0.003, result.Summary[0].EstimatedCostUSD)
	dir := t.TempDir()
	require.NoError(t, WriteRunJSON(filepath.Join(dir, "run-1.json"), result))
	require.NoError(t, WriteRunToStore(dir, result))
	index, err := LoadRunStore(dir)
	require.NoError(t, err)
	require.Len(t, index.Runs, 1)
	require.NoError(t, WriteDashboardHTML(filepath.Join(dir, "dashboard.html"), index))
}

func TestArtifactsRender(t *testing.T) {
	result := RunResult{
		RunID:     "run",
		SuiteName: "suite",
		Subjects:  []string{"subject"},
		Cases:     []EvalCase{{ID: "case", Input: gepa.Prediction{"x": "y"}}},
		Summary:   []SubjectSummary{{Subject: "subject", Cases: 1, AverageScore: 1}},
		Results: []SubjectCaseResult{{
			Subject:        "subject",
			CaseID:         "case",
			AggregateScore: 1,
			Output:         SubjectOutput{Value: gepa.Prediction{"answer": "ok"}, Latency: time.Millisecond},
		}},
	}
	dir := t.TempDir()
	require.NoError(t, WriteRunJSON(filepath.Join(dir, "run.json"), result))
	require.NoError(t, WriteCSVSummary(filepath.Join(dir, "summary.csv"), result))
	html, err := HTMLReport(result)
	require.NoError(t, err)
	assert.Contains(t, html, "Leaderboard")
	assert.Contains(t, html, "subject")
}

func staticSubject(name string, fn func(gepa.Prediction) gepa.Prediction) Subject {
	return FuncSubject{
		SubjectName: name,
		Func: func(_ context.Context, input gepa.Prediction) (SubjectOutput, error) {
			return SubjectOutput{Value: fn(input)}, nil
		},
	}
}

func valueSubject(name string, value gepa.Prediction) Subject {
	return staticSubject(name, func(gepa.Prediction) gepa.Prediction { return value })
}

func findSummary(summaries []SubjectSummary, subject string) SubjectSummary {
	for _, summary := range summaries {
		if summary.Subject == subject {
			return summary
		}
	}
	return SubjectSummary{}
}
