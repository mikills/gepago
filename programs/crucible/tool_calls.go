package crucible

import (
	"context"
	"fmt"
	"reflect"
	"strings"
)

const (
	expectedToolCallsKey = "expected_tool_calls"
	toolCallsKey         = "tool_calls"
)

// ToolCallSummary is the stable tool-call shape used by Crucible reports and evaluators.
type ToolCallSummary struct {
	Name           string         `json:"name"`
	Arguments      string         `json:"arguments,omitempty"`
	ArgumentsValue map[string]any `json:"arguments_value,omitempty"`
	Output         string         `json:"output,omitempty"`
	Error          string         `json:"error,omitempty"`
}

// ToolCallEvaluator scores whether a subject called the expected tools.
type ToolCallEvaluator struct {
	ExpectedNames []string
	RequireOrder  bool
	AllowExtra    bool
}

func (e ToolCallEvaluator) Name() string { return toolCallsKey }

func (e ToolCallEvaluator) Evaluate(_ context.Context, input EvalInput) (Score, error) {
	expected := e.expected(input)
	if len(expected) == 0 {
		return Score{Skipped: true, Feedback: "expected tool calls not provided"}, nil
	}
	actual := outputToolCallNames(input.Output)
	matched := toolCallMatchScore(expected, actual, e.RequireOrder, e.AllowExtra)
	feedback := toolCallFeedback(expected, actual, matched)
	return Score{
		Score:    matched,
		Feedback: feedback,
		Details: map[string]any{
			"expected": expected,
			"actual":   actual,
		},
	}, nil
}

func (e ToolCallEvaluator) expected(input EvalInput) []string {
	if len(e.ExpectedNames) > 0 {
		return append([]string(nil), e.ExpectedNames...)
	}
	return stringList(input.Case.Metadata[expectedToolCallsKey])
}

func outputToolCallNames(output SubjectOutput) []string {
	if len(output.ToolCalls) > 0 {
		return namesFromToolSummaries(output.ToolCalls)
	}
	return toolCallNames(output.Metadata[toolCallsKey])
}

func toolCallNames(value any) []string {
	switch calls := value.(type) {
	case []ToolCallSummary:
		return namesFromToolSummaries(calls)
	case []map[string]any:
		return namesFromMaps(calls)
	case []any:
		return namesFromAny(calls)
	default:
		return namesFromStructSlice(value)
	}
}

func namesFromToolSummaries(calls []ToolCallSummary) []string {
	names := make([]string, 0, len(calls))
	for _, call := range calls {
		names = append(names, call.Name)
	}
	return names
}

func namesFromMaps(calls []map[string]any) []string {
	names := make([]string, 0, len(calls))
	for _, call := range calls {
		names = append(names, fmt.Sprint(call["name"]))
	}
	return names
}

func namesFromAny(calls []any) []string {
	names := make([]string, 0, len(calls))
	for _, call := range calls {
		if mapped, ok := call.(map[string]any); ok {
			names = append(names, fmt.Sprint(mapped["name"]))
		}
	}
	return names
}

func namesFromStructSlice(value any) []string {
	reflected := reflect.ValueOf(value)
	if reflected.Kind() != reflect.Slice {
		return nil
	}
	names := make([]string, 0, reflected.Len())
	for i := 0; i < reflected.Len(); i++ {
		name := reflected.Index(i).FieldByName("Name")
		if name.IsValid() && name.Kind() == reflect.String {
			names = append(names, name.String())
		}
	}
	return names
}

func toolCallMatchScore(expected []string, actual []string, requireOrder bool, allowExtra bool) float64 {
	if requireOrder {
		return boolFloat(matchesOrdered(expected, actual, allowExtra))
	}
	return unorderedMatchScore(expected, actual, allowExtra)
}

func matchesOrdered(expected []string, actual []string, allowExtra bool) bool {
	if !allowExtra && len(expected) != len(actual) {
		return false
	}
	position := 0
	for _, name := range actual {
		if position < len(expected) && name == expected[position] {
			position++
		}
	}
	return position == len(expected)
}

func unorderedMatchScore(expected []string, actual []string, allowExtra bool) float64 {
	if !allowExtra && len(expected) != len(actual) {
		return 0
	}
	remaining := append([]string(nil), actual...)
	matched := 0
	for _, want := range expected {
		for i, got := range remaining {
			if got == want {
				matched++
				remaining = append(remaining[:i], remaining[i+1:]...)
				break
			}
		}
	}
	return float64(matched) / float64(len(expected))
}

func toolCallFeedback(expected []string, actual []string, score float64) string {
	if score == 1 {
		return "tool calls matched"
	}
	return "expected " + strings.Join(expected, ", ") + "; got " + strings.Join(actual, ", ")
}

func stringList(value any) []string {
	switch typed := value.(type) {
	case []string:
		return typed
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			out = append(out, fmt.Sprint(item))
		}
		return out
	default:
		return nil
	}
}

func boolFloat(ok bool) float64 {
	if ok {
		return 1
	}
	return 0
}
