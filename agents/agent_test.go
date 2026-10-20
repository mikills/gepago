package agents

import (
	"context"
	"errors"
	"reflect"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
)

// chatRequests returns the requests the fake model received, in order.
func chatRequests(fake *gepatest.FakeGenerator) []ChatRequest {
	var requests []ChatRequest
	for _, call := range fake.Calls() {
		requests = append(requests, *call.Request)
	}
	return requests
}

func TestAgentRun(t *testing.T) {
	t.Run("returns final assistant message", func(t *testing.T) {
		client := gepatest.NewFakeGenerator().Default(gepatest.Text("hello").WithUsage(Usage{TotalTokens: 3}))
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
		requests := chatRequests(client)
		if len(requests) != 1 {
			t.Fatalf("requests len = %d", len(requests))
		}
		if requests[0].Messages[0].Role != RoleSystem {
			t.Fatalf("first transcript role = %q", requests[0].Messages[0].Role)
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
		client := gepatest.NewFakeGenerator().Default(
			gepatest.ToolCall("call-1", "lookup", `{"q":"x"}`),
			gepatest.Text("used tool"),
		)
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
		second := chatRequests(client)[1]
		if got := second.Messages[len(second.Messages)-1]; got.Role != RoleTool || got.Content != "tool output" {
			t.Fatalf("last second request message = %#v", got)
		}
	})

	t.Run("tool trace events are immutable snapshots", func(t *testing.T) {
		client := gepatest.NewFakeGenerator().Default(gepatest.ToolCall("call-1", "lookup", `{}`), gepatest.Text("done"))
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
		client := gepatest.NewFakeGenerator().Default(gepatest.ToolCall("call-1", "missing", ""))
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
		client := gepatest.NewFakeGenerator().Default(gepatest.Text("ok"))
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

func TestAgentRunMapsFinishReasons(t *testing.T) {
	tests := []struct {
		finish FinishReason
		want   StopReason
	}{
		{finish: FinishStop, want: StopComplete},
		{finish: FinishLength, want: StopLength},
		{finish: FinishContentFilter, want: StopContentFilter},
		{finish: "", want: StopComplete},
	}
	for _, tt := range tests {
		t.Run(string(tt.finish), func(t *testing.T) {
			client := gepatest.NewFakeGenerator().Default(gepatest.Text("partial").
				WithUsage(Usage{TotalTokens: 4, ReasoningTokens: 2, CachedInputTokens: 1}).
				WithFinishReason(tt.finish))
			agent := Agent{Name: "test", Client: client, Model: "fake"}

			result, err := agent.Run(context.Background(), RunRequest{
				Memory:   NewMemory("run"),
				Messages: []Message{{Role: RoleUser, Content: "hi"}},
			})
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			if result.StopReason != tt.want {
				t.Fatalf("StopReason = %q, want %q", result.StopReason, tt.want)
			}
			if result.Ledger.ReasoningTokens != 2 || result.Ledger.CachedInputTokens != 1 {
				t.Fatalf("Ledger = %#v", result.Ledger)
			}
		})
	}
}

func TestFinishReasonAliasesGepa(t *testing.T) {
	var reason gepa.FinishReason = FinishContentFilter
	if reason != gepa.FinishContentFilter || FinishStop != gepa.FinishStop || FinishLength != gepa.FinishLength ||
		FinishToolCalls != gepa.FinishToolCalls {
		t.Fatalf("agents finish reasons diverge from gepa: %q", reason)
	}
}
