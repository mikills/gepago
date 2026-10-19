package main

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/mikills/gepago/agents"
)

func TestAgentProtocolValidation(t *testing.T) {
	valid := `{"name":"assistant","model":"host-model","run_id":"run-1","messages":[{"role":"user","content":"hi"}]}`
	if err := validateAgentRunPayload(valid); err != nil {
		t.Fatalf("valid payload rejected: %v", err)
	}
	for _, test := range []struct {
		name string
		json string
		want string
	}{
		{"missing run id", `{"name":"a","model":"m","messages":[{"role":"user"}]}`, "run_id is required"},
		{"missing messages", `{"name":"a","model":"m","run_id":"r"}`, "at least one message"},
		{"duplicate tool", `{"name":"a","model":"m","run_id":"r","messages":[{"role":"user"}],"tools":[{"name":"x","description":"one"},{"name":"x","description":"two"}]}`, "duplicate registration"},
		{"bad max tokens", `{"name":"a","model":"m","run_id":"r","maxTokens":-1,"messages":[{"role":"user"}]}`, "maxTokens must not be negative"},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := validateAgentRunPayload(test.json)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestAgentOperationWithToolAndMemory(t *testing.T) {
	payload := `{"name":"assistant","model":"host-model","systemPrompt":"be useful","run_id":"run-1","memory":{"count":1},"messages":[{"role":"user","content":"look it up"}],"tools":[{"name":"lookup","description":"look up a value","parameters":{"type":"object"}}]}`
	chatCalls := 0
	toolCalls := 0
	callbacks := agentHostCallbacks{
		chat: func(_ context.Context, requestJSON string) (string, error) {
			var request agents.ChatRequest
			if err := json.Unmarshal([]byte(requestJSON), &request); err != nil {
				t.Fatal(err)
			}
			chatCalls++
			if chatCalls == 1 {
				if request.Model != "host-model" || len(request.Tools) != 1 || len(request.Messages) != 2 ||
					request.Messages[0].Role != agents.RoleSystem {
					t.Fatalf("unexpected first chat request: %+v", request)
				}
				return mustAgentJSON(t, agents.ChatResponse{
					Message: agents.Message{
						Role:      agents.RoleAssistant,
						ToolCalls: []agents.ToolCall{{ID: "call-1", Name: "lookup", Arguments: `{"q":"x"}`}},
					},
					Usage: agents.Usage{PromptTokens: 2, CompletionTokens: 3}, FinishReason: agents.FinishToolCalls,
				}), nil
			}
			if request.Messages[len(request.Messages)-1].Role != agents.RoleTool ||
				request.Messages[len(request.Messages)-1].Content != "found" {
				t.Fatalf("tool result was not added to transcript: %+v", request.Messages)
			}
			return mustAgentJSON(t, agents.ChatResponse{
				Message: agents.Message{Role: agents.RoleAssistant, Content: "done"},
				Usage: agents.Usage{
					PromptTokens:     4,
					CompletionTokens: 1,
					TotalTokens:      5,
				}, FinishReason: agents.FinishStop,
			}), nil
		},
		tool: func(_ context.Context, requestJSON string) (agentToolHostResult, error) {
			var request struct {
				Name      string         `json:"name"`
				CallID    string         `json:"call_id"`
				Arguments string         `json:"arguments"`
				RunID     string         `json:"run_id"`
				Memory    map[string]any `json:"memory"`
			}
			if err := json.Unmarshal([]byte(requestJSON), &request); err != nil {
				t.Fatal(err)
			}
			if request.Name != "lookup" || request.CallID != "call-1" || request.Arguments != `{"q":"x"}` ||
				request.RunID != "run-1" ||
				request.Memory["count"] != float64(1) {
				t.Fatalf("unexpected tool request: %+v", request)
			}
			toolCalls++
			return agentToolHostResult{Output: "found", Memory: map[string]any{"count": 2}, HasMemory: true}, nil
		},
	}

	result, err := runAgentOperation(context.Background(), payload, callbacks)
	if err != nil {
		t.Fatalf("runAgentOperation error = %v", err)
	}
	if result.Final.Content != "done" || result.StopReason != agents.StopComplete || chatCalls != 2 || toolCalls != 1 {
		t.Fatalf("unexpected result: %+v", result)
	}
	if result.Usage.TotalTokens != 10 || result.Ledger.ToolCalls != 1 {
		t.Fatalf("usage/ledger = %+v / %+v", result.Usage, result.Ledger)
	}
}

func TestAgentHostChatRequestToolSchemas(t *testing.T) {
	payload := `{"name":"a","model":"m","run_id":"r","messages":[{"role":"user","content":"hi"}],"tools":[` +
		`{"name":"now","description":"current time","parameters":{}},` +
		`{"name":"find","description":"find","parameters":{"properties":{"q":{"type":"string"}},"required":["q"]}},` +
		`{"name":"typed","description":"typed","parameters":{"type":"object","properties":{}}},` +
		`{"name":"bare","description":"no parameters"},` +
		`{"name":"null","description":"null parameters","parameters":null},` +
		`{"name":"spaced","description":"spaced","parameters":{ }},` +
		`{"name":"any","description":"any of","parameters":{"anyOf":[{"type":"object"}]}}]}`
	var sent map[string]any
	callbacks := agentHostCallbacks{
		chat: func(_ context.Context, requestJSON string) (string, error) {
			if err := json.Unmarshal([]byte(requestJSON), &sent); err != nil {
				t.Fatal(err)
			}
			return `{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}`, nil
		},
	}

	if _, err := runAgentOperation(context.Background(), payload, callbacks); err != nil {
		t.Fatalf("runAgentOperation error = %v", err)
	}

	tools := sent["tools"].([]any)
	want := []any{
		map[string]any{"type": "object"},
		map[string]any{
			"type":       "object",
			"properties": map[string]any{"q": map[string]any{"type": "string"}},
			"required":   []any{"q"},
		},
		map[string]any{"type": "object", "properties": map[string]any{}},
		nil,
		nil,
		map[string]any{"type": "object"},
		map[string]any{"type": "object", "anyOf": []any{map[string]any{"type": "object"}}},
	}
	if len(tools) != len(want) {
		t.Fatalf("sent %d tools, want %d", len(tools), len(want))
	}
	for i, tool := range tools {
		if got := tool.(map[string]any)["parameters"]; !reflect.DeepEqual(got, want[i]) {
			t.Fatalf("tool %d parameters = %#v, want %#v", i, got, want[i])
		}
	}
}

func TestAgentUsageValidation(t *testing.T) {
	got, err := normalizeAgentUsage(agents.Usage{PromptTokens: 2, CompletionTokens: 3})
	if err != nil || got.TotalTokens != 5 {
		t.Fatalf("normalized usage = %+v, err = %v", got, err)
	}
	if _, err := normalizeAgentUsage(agents.Usage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 4}); err == nil {
		t.Fatal("inconsistent usage was accepted")
	}
}

func TestHostUsageRejectsNegativeBreakdown(t *testing.T) {
	for _, usage := range []agents.Usage{
		{ReasoningTokens: -1},
		{CachedInputTokens: -1},
		{CacheCreationTokens: -1},
		{PromptTokens: -1},
	} {
		if !negativeUsage(usage) {
			t.Fatalf("negativeUsage(%+v) = false", usage)
		}
		if err := validateUsage(usage); err == nil {
			t.Fatalf("validateUsage(%+v) accepted negative counts", usage)
		}
		if err := validatePredictUsage(usage); err == nil {
			t.Fatalf("validatePredictUsage(%+v) accepted negative counts", usage)
		}
		if _, err := normalizeAgentUsage(usage); err == nil {
			t.Fatalf("normalizeAgentUsage(%+v) accepted negative counts", usage)
		}
	}
	if negativeUsage(agents.Usage{PromptTokens: 1, ReasoningTokens: 1, CachedInputTokens: 1}) {
		t.Fatal("positive usage reported as negative")
	}
}

func TestAgentToolResponseValidation(t *testing.T) {
	result, err := parseAgentToolHostResultJSON(`{"output":"ok","memory":{"ready":true}}`)
	if err != nil || !result.HasMemory || result.Memory["ready"] != true {
		t.Fatalf("result = %+v, err = %v", result, err)
	}
	for _, response := range []string{`{}`, `{"output":null}`, `{"output":"ok","memory":[]}`} {
		if _, err := parseAgentToolHostResultJSON(response); err == nil {
			t.Fatalf("accepted invalid tool response %s", response)
		}
	}
}

func mustAgentJSON(t *testing.T, value any) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}
