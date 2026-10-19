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

// AgentSubject adapts a tool-calling agent into an evaluation subject. A run whose final reply
// stopped at the output token limit fails the case with gepa.ErrOutputTruncated, and one a
// content policy withheld or the model refused with gepa.ErrContentFiltered, so neither is
// scored as the answer; the failed output still carries the transcript, tool calls and usage.
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
	if err == nil {
		err = agentStopError(result)
	}
	return agentOutput(result), err
}

// agentStopError returns gepa.FinishReasonError for the run's final reply: an error when the
// run stopped at the output token limit or under a content policy, and nil otherwise. The
// completion tokens it reports are the whole run's.
func agentStopError(result agents.RunResult) error {
	generation := gepa.Generation{Text: result.Final.Content, Usage: result.Usage}
	switch result.StopReason {
	case agents.StopLength:
		generation.FinishReason = gepa.FinishLength
	case agents.StopContentFilter:
		generation.FinishReason = gepa.FinishContentFilter
	}
	return gepa.FinishReasonError(generation, "agent run")
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
	metadata := map[string]any{
		toolCallsKey: toolCalls,
		"spans":      result.Spans,
	}
	if len(result.ResolvedModels) > 0 {
		metadata["resolved_models"] = append([]string(nil), result.ResolvedModels...)
		metadata["resolved_model"] = result.ResolvedModels[len(result.ResolvedModels)-1]
	}
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
		Metadata:  metadata,
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
