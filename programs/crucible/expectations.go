package crucible

import (
	"context"
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

// ExpectationEvaluator scores case expectations against observable subject output.
type ExpectationEvaluator struct{}

func (ExpectationEvaluator) Name() string { return "expectations" }

func (ExpectationEvaluator) Evaluate(_ context.Context, input EvalInput) (Score, error) {
	if len(input.Case.Expectations) == 0 {
		return Score{Skipped: true, Feedback: "expectations not provided"}, nil
	}
	results := make([]map[string]any, 0, len(input.Case.Expectations))
	var weightedSum float64
	var weightSum float64
	for _, expectation := range input.Case.Expectations {
		actual := selectObservation(input.Output, expectation.Select)
		matched := matchExpectation(actual, expectation)
		weight := normalizedWeight(expectation.Weight)
		weightedSum += matched * weight
		weightSum += weight
		results = append(results, expectationDetails(expectation, actual, matched))
	}
	return Score{
		Score:    weightedSum / weightSum,
		Feedback: expectationFeedback(results),
		Details:  map[string]any{"expectations": results},
	}, nil
}

func selectObservation(output SubjectOutput, selector string) any {
	parts := strings.Split(strings.TrimSpace(selector), ".")
	if len(parts) == 0 {
		return nil
	}
	switch parts[0] {
	case "final":
		return output.Value["final"]
	case "raw":
		return output.Raw
	case "value":
		return selectMapPath(map[string]any(output.Value), parts[1:])
	case "metadata":
		return selectMapPath(output.Metadata, parts[1:])
	case "latency":
		return output.Latency
	case toolCallsKey:
		return selectToolCalls(output.ToolCalls, parts[1:])
	default:
		return nil
	}
}

func selectMapPath(values map[string]any, parts []string) any {
	var current any = values
	for _, part := range parts {
		mapped, ok := current.(map[string]any)
		if !ok {
			return nil
		}
		current = mapped[part]
	}
	return current
}

func selectToolCalls(calls []ToolCallSummary, parts []string) any {
	if len(parts) == 0 {
		return calls
	}
	switch parts[0] {
	case "count":
		return len(calls)
	case "names":
		return namesFromToolSummaries(calls)
	case "arguments":
		return selectToolArguments(calls, parts[1:])
	case "outputs":
		return toolCallOutputs(calls)
	default:
		return nil
	}
}

func selectToolArguments(calls []ToolCallSummary, parts []string) any {
	if len(parts) == 0 {
		out := make([]map[string]any, 0, len(calls))
		for _, call := range calls {
			out = append(out, call.ArgumentsValue)
		}
		return out
	}
	values := make([]string, 0, len(calls))
	for _, call := range calls {
		if value, ok := call.ArgumentsValue[parts[0]]; ok {
			values = append(values, fmt.Sprint(value))
		}
	}
	return values
}

func toolCallOutputs(calls []ToolCallSummary) []string {
	out := make([]string, 0, len(calls))
	for _, call := range calls {
		out = append(out, call.Output)
	}
	return out
}

func matchExpectation(actual any, expectation Expectation) float64 {
	matcher := expectationMatchers()[strings.TrimSpace(expectation.Should)]
	if matcher == nil {
		return 0
	}
	return matcher(actual, expectation)
}

type expectationMatcher func(actual any, expectation Expectation) float64

func expectationMatchers() map[string]expectationMatcher {
	return map[string]expectationMatcher{
		"equals":            matchEquals,
		"contains":          matchContains,
		"not_contains":      matchNotContains,
		"contains_any":      matchContainsAny,
		"contains_all":      matchContainsAll,
		"contains_sequence": matchContainsSequence,
		"count_equals":      matchCountEquals,
		"count_at_least":    matchCountAtLeast,
		"regex":             matchRegex,
		"matches_regex":     matchRegex,
		"starts_with":       matchStartsWith,
		"json_schema":       matchJSONSchema,
		"latency_below":     matchLatencyBelow,
		"tool_args_match":   matchToolArgs,
	}
}

func matchEquals(actual any, expectation Expectation) float64 {
	return boolFloat(equalValues(actual, expectation.Value))
}

func matchContains(actual any, expectation Expectation) float64 {
	return boolFloat(containsValue(actual, expectation.Value))
}

func matchNotContains(actual any, expectation Expectation) float64 {
	return boolFloat(!containsValue(actual, expectation.Value))
}

func matchContainsAll(actual any, expectation Expectation) float64 {
	return boolFloat(containsAll(actual, stringList(expectation.Value)))
}

func matchContainsSequence(actual any, expectation Expectation) float64 {
	return boolFloat(containsSequence(stringList(actual), stringList(expectation.Value)))
}

func matchCountEquals(actual any, expectation Expectation) float64 {
	return boolFloat(asInt(actual) == asInt(expectation.Value))
}

func matchCountAtLeast(actual any, expectation Expectation) float64 {
	return boolFloat(asInt(actual) >= asInt(expectation.Value))
}

func equalValues(actual any, expected any) bool {
	if reflect.DeepEqual(actual, expected) {
		return true
	}
	return fmt.Sprint(actual) == fmt.Sprint(expected)
}

func containsValue(actual any, expected any) bool {
	needle := fmt.Sprint(expected)
	if strings.Contains(fmt.Sprint(actual), needle) {
		return true
	}
	for _, value := range stringList(actual) {
		if value == needle || strings.Contains(value, needle) {
			return true
		}
	}
	return false
}

func containsAll(actual any, expected []string) bool {
	for _, value := range expected {
		if !containsValue(actual, value) {
			return false
		}
	}
	return true
}

func containsSequence(actual []string, expected []string) bool {
	position := 0
	for _, value := range actual {
		if position < len(expected) && value == expected[position] {
			position++
		}
	}
	return position == len(expected)
}

func asInt(value any) int {
	switch typed := value.(type) {
	case int:
		return typed
	case float64:
		return int(typed)
	case jsonNumber:
		parsed, _ := strconv.Atoi(typed.String())
		return parsed
	default:
		parsed, _ := strconv.Atoi(fmt.Sprint(value))
		return parsed
	}
}

type jsonNumber interface{ String() string }

func expectationDetails(expectation Expectation, actual any, matched float64) map[string]any {
	return map[string]any{
		"name":   expectation.Name,
		"select": expectation.Select,
		"should": expectation.Should,
		"want":   expectation.Value,
		"got":    actual,
		"score":  matched,
	}
}

func expectationFeedback(results []map[string]any) string {
	failed := []string{}
	for _, result := range results {
		if result["score"] != float64(1) {
			failed = append(failed, fmt.Sprint(result["select"]))
		}
	}
	if len(failed) == 0 {
		return "expectations matched"
	}
	return "failed expectations: " + strings.Join(failed, ", ")
}
