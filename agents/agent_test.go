package agents

import (
	"context"
	"errors"
	"reflect"
	"testing"

	gepa "github.com/mikills/gepago"
)

type fakeChatClient struct {
	responses []ChatResponse
	requests  []ChatRequest
	err       error
}

func (c *fakeChatClient) Chat(_ context.Context, req ChatRequest) (ChatResponse, error) {
	c.requests = append(c.requests, req)
	if c.err != nil {
		return ChatResponse{}, c.err
	}
	if len(c.responses) == 0 {
		return ChatResponse{Message: Message{Role: RoleAssistant, Content: "done"}, FinishReason: FinishStop}, nil
	}
	resp := c.responses[0]
	c.responses = c.responses[1:]
	return resp, nil
}

func TestAgentRun(t *testing.T) {
	t.Run("returns final assistant message", func(t *testing.T) {
		client := &fakeChatClient{
			responses: []ChatResponse{
				{
					Message:      Message{Role: RoleAssistant, Content: "hello"},
					Usage:        Usage{TotalTokens: 3},
					FinishReason: FinishStop,
				},
			},
		}
		recorder := NewTraceRecorder()
		agent := Agent{
			Name:         "test",
			Client:       client,
			Model:        "fake",
			SystemPrompt: "system",
			Observers:    []Observer{recorder},
		}

		result, err := agent.Run(
			context.Background(),
			RunRequest{Memory: NewMemory("run-1"), Messages: []Message{{Role: RoleUser, Content: "hi"}}},
		)
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if result.Final.Content != "hello" {
			t.Fatalf("Final.Content = %q", result.Final.Content)
		}
		if result.StopReason != StopComplete {
			t.Fatalf("StopReason = %q", result.StopReason)
		}
		if result.Ledger.Runs != 1 || result.Ledger.ModelCalls != 1 || result.Ledger.TotalTokens != 3 {
			t.Fatalf("Ledger = %#v", result.Ledger)
		}
		if len(result.Spans) != 2 || result.Spans[0].Kind != gepa.UsageSpanLLMCall ||
			result.Spans[1].Kind != gepa.UsageSpanAgentRun {
			t.Fatalf("Spans = %#v", result.Spans)
		}
		if len(client.requests) != 1 {
			t.Fatalf("requests len = %d", len(client.requests))
		}
		if client.requests[0].Messages[0].Role != RoleSystem {
			t.Fatalf("first transcript role = %q", client.requests[0].Messages[0].Role)
		}
		trace, ok := recorder.Trace("run-1")
		if !ok {
			t.Fatal("trace was not recorded")
		}
		kinds := eventKinds(trace.Events)
		want := []EventKind{EventRunStart, EventTurnStart, EventLLMRequest, EventLLMResponse, EventTurnEnd, EventRunEnd}
		if !reflect.DeepEqual(kinds, want) {
			t.Fatalf("event kinds = %#v, want %#v", kinds, want)
		}
	})

	t.Run("dispatches tool calls and continues", func(t *testing.T) {
		client := &fakeChatClient{responses: []ChatResponse{
			{
				Message: Message{
					Role:      RoleAssistant,
					ToolCalls: []ToolCall{{ID: "call-1", Name: "lookup", Arguments: `{"q":"x"}`}},
				},
				FinishReason: FinishToolCalls,
			},
			{Message: Message{Role: RoleAssistant, Content: "used tool"}, FinishReason: FinishStop},
		}}
		agent := Agent{
			Name:   "test",
			Client: client,
			Model:  "fake",
			Tools: []ToolBinding{
				{
					Definition: Tool{Name: "lookup", Description: "look up data"},
					Handler: func(_ context.Context, _ *Memory, arguments string) (string, error) {
						if arguments != `{"q":"x"}` {
							t.Fatalf("arguments = %q", arguments)
						}
						return "tool output", nil
					},
				},
			},
		}

		result, err := agent.Run(
			context.Background(),
			RunRequest{Memory: NewMemory("run-2"), Messages: []Message{{Role: RoleUser, Content: "hi"}}},
		)
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if result.Final.Content != "used tool" {
			t.Fatalf("Final.Content = %q", result.Final.Content)
		}
		if len(result.ToolCalls) != 1 || result.ToolCalls[0].Output != "tool output" {
			t.Fatalf("ToolCalls = %#v", result.ToolCalls)
		}
		if result.Ledger.ToolCalls != 1 || result.Ledger.ModelCalls != 2 {
			t.Fatalf("Ledger = %#v", result.Ledger)
		}
		if got := client.requests[1].Messages[len(client.requests[1].Messages)-1]; got.Role != RoleTool ||
			got.Content != "tool output" {
			t.Fatalf("last second request message = %#v", got)
		}
	})

	t.Run("tool trace events are immutable snapshots", func(t *testing.T) {
		client := &fakeChatClient{responses: []ChatResponse{
			{
				Message: Message{
					Role:      RoleAssistant,
					ToolCalls: []ToolCall{{ID: "call-1", Name: "lookup", Arguments: `{}`}},
				},
				FinishReason: FinishToolCalls,
			},
			{Message: Message{Role: RoleAssistant, Content: "done"}, FinishReason: FinishStop},
		}}
		recorder := NewTraceRecorder()
		agent := Agent{
			Name:      "test",
			Client:    client,
			Model:     "fake",
			Observers: []Observer{recorder},
			Tools: []ToolBinding{
				{
					Definition: Tool{Name: "lookup", Description: "look up data"},
					Handler: func(context.Context, *Memory, string) (string, error) {
						return "tool output", nil
					},
				},
			},
		}

		_, err := agent.Run(
			context.Background(),
			RunRequest{Memory: NewMemory("run-snapshot"), Messages: []Message{{Role: RoleUser, Content: "hi"}}},
		)
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		trace, ok := recorder.Trace("run-snapshot")
		if !ok {
			t.Fatal("trace was not recorded")
		}
		var start, end *ToolCallRecord
		for _, event := range trace.Events {
			if event.Kind == EventToolCallStart {
				start = event.ToolCall
			}
			if event.Kind == EventToolCallEnd {
				end = event.ToolCall
			}
		}
		if start == nil || end == nil {
			t.Fatalf("tool events missing from trace: %#v", eventKinds(trace.Events))
		}
		if start.Output != "" {
			t.Fatalf("start event output = %q, want empty snapshot", start.Output)
		}
		if end.Output != "tool output" {
			t.Fatalf("end event output = %q", end.Output)
		}
	})

	t.Run("unknown tool returns tool error", func(t *testing.T) {
		client := &fakeChatClient{
			responses: []ChatResponse{
				{
					Message:      Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call-1", Name: "missing"}}},
					FinishReason: FinishToolCalls,
				},
			},
		}
		agent := Agent{Name: "test", Client: client, Model: "fake"}

		result, err := agent.Run(
			context.Background(),
			RunRequest{Memory: NewMemory("run-3"), Messages: []Message{{Role: RoleUser, Content: "hi"}}},
		)
		if !errors.Is(err, ErrUnknownTool) {
			t.Fatalf("Run() error = %v, want ErrUnknownTool", err)
		}
		if result.StopReason != StopToolError {
			t.Fatalf("StopReason = %q", result.StopReason)
		}
	})

	t.Run("observer panic is contained", func(t *testing.T) {
		client := &fakeChatClient{
			responses: []ChatResponse{{Message: Message{Role: RoleAssistant, Content: "ok"}, FinishReason: FinishStop}},
		}
		recorder := NewTraceRecorder()
		agent := Agent{
			Name:      "test",
			Client:    client,
			Model:     "fake",
			Observers: []Observer{ObserverFunc(func(context.Context, Event) { panic("boom") }), recorder},
		}

		_, err := agent.Run(
			context.Background(),
			RunRequest{Memory: NewMemory("run-4"), Messages: []Message{{Role: RoleUser, Content: "hi"}}},
		)
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if _, ok := recorder.Trace("run-4"); !ok {
			t.Fatal("recorder after panicking observer did not run")
		}
	})
}

func eventKinds(events []Event) []EventKind {
	kinds := make([]EventKind, 0, len(events))
	for _, event := range events {
		kinds = append(kinds, event.Kind)
	}
	return kinds
}
