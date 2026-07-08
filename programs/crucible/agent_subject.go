package crucible

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/agents"
)

// AgentSubject adapts a tool-calling agent into an evaluation subject.
type AgentSubject struct {
	SubjectName       string
	Agent             *agents.Agent
	InputMessageField string
	MemoryFactory     func(gepa.Prediction) *agents.Memory
}

func (s AgentSubject) Name() string { return s.SubjectName }

func (s AgentSubject) Run(ctx context.Context, input gepa.Prediction) (SubjectOutput, error) {
	if s.Agent == nil {
		return SubjectOutput{}, errors.New("eval agent subject requires agent")
	}
	memory := s.memory(input)
	result, err := s.Agent.Run(ctx, agents.RunRequest{
		Memory:   memory,
		Messages: []agents.Message{{Role: agents.RoleUser, Content: s.message(input)}},
	})
	return agentOutput(result), err
}

func (s AgentSubject) memory(input gepa.Prediction) *agents.Memory {
	if s.MemoryFactory != nil {
		return s.MemoryFactory(input)
	}
	if runID := strings.TrimSpace(fmt.Sprint(input["run_id"])); runID != "" && runID != "<nil>" {
		return agents.NewMemory(runID)
	}
	name := strings.ReplaceAll(strings.TrimSpace(s.SubjectName), " ", "-")
	if name == "" {
		name = "agent"
	}
	return agents.NewMemory("crucible-" + name)
}

func (s AgentSubject) message(input gepa.Prediction) string {
	field := strings.TrimSpace(s.InputMessageField)
	if field == "" {
		field = "prompt"
	}
	if value := strings.TrimSpace(fmt.Sprint(input[field])); value != "" && value != "<nil>" {
		return value
	}
	data, err := json.MarshalIndent(input, "", "  ")
	if err != nil {
		return fmt.Sprint(input)
	}
	return string(data)
}

func agentOutput(result agents.RunResult) SubjectOutput {
	toolCalls := summarizeToolCalls(result.ToolCalls)
	return SubjectOutput{
		Value: gepa.Prediction{
			"final":       result.Final.Content,
			"stop_reason": string(result.StopReason),
			"turns":       result.Turns,
		},
		ToolCalls: toolCalls,
		Usage:     result.Usage,
		Trace:     result.Transcript,
		Latency:   result.Ledger.TotalDuration,
		Metadata: map[string]any{
			toolCallsKey: toolCalls,
			"spans":      result.Spans,
		},
	}
}

func summarizeToolCalls(records []agents.ToolCallRecord) []ToolCallSummary {
	summaries := make([]ToolCallSummary, 0, len(records))
	for _, record := range records {
		summaries = append(summaries, ToolCallSummary{
			Name:           record.Name,
			Arguments:      record.Arguments,
			ArgumentsValue: decodeArguments(record.Arguments),
			Output:         record.Output,
			Error:          record.Err,
		})
	}
	return summaries
}

func decodeArguments(arguments string) map[string]any {
	if strings.TrimSpace(arguments) == "" {
		return nil
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(arguments), &decoded); err != nil {
		return nil
	}
	return decoded
}
