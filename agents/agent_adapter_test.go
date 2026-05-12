package agents

import (
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
)

func TestTraceDatasetBuilder(t *testing.T) {
	t.Run("adds summary for reflective records", func(t *testing.T) {
		trace := Trace{RunID: "run", AgentName: "agent", Events: []Event{
			{
				Kind:      EventLLMResponse,
				RunID:     "run",
				AgentName: "agent",
				Turn:      1,
				Timestamp: time.Now().UTC(),
				Usage:     Usage{PromptTokens: 3, CompletionTokens: 4},
				Response: &ChatResponse{
					Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call", Name: "lookup"}}},
				},
			},
			{
				Kind:      EventToolCallEnd,
				RunID:     "run",
				AgentName: "agent",
				Turn:      1,
				Timestamp: time.Now().UTC(),
				ToolCall:  &ToolCallRecord{Name: "lookup", Output: "data"},
			},
			{
				Kind:      EventLLMResponse,
				RunID:     "run",
				AgentName: "agent",
				Turn:      2,
				Timestamp: time.Now().UTC(),
				Usage:     Usage{PromptTokens: 5, CompletionTokens: 6},
				Response:  &ChatResponse{Message: Message{Role: RoleAssistant, Content: "final"}},
			},
			{
				Kind:       EventRunEnd,
				RunID:      "run",
				AgentName:  "agent",
				Turn:       2,
				Timestamp:  time.Now().UTC(),
				StopReason: StopComplete,
			},
		}}
		dataset := TraceDatasetBuilder{}.BuildReflectiveDataset(
			gepa.Candidate{"prompt": "old"},
			gepa.EvaluationResult{Items: []gepa.EvaluationItem{{ExampleID: "ex", Score: 1, Trace: &trace}}},
			[]string{"prompt"},
		)
		summary, ok := dataset["prompt"][0]["summary"].(TraceSummary)
		if !ok {
			t.Fatalf("summary missing: %#v", dataset)
		}
		if summary.FinalOutput != "final" || len(summary.ToolCalls) != 1 || summary.TotalTurns != 2 {
			t.Fatalf("summary = %#v", summary)
		}
	})
}
