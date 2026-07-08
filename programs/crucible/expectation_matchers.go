package crucible

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

func matchContainsAny(actual any, expectation Expectation) float64 {
	for _, value := range stringList(expectation.Value) {
		if containsValue(actual, value) {
			return 1
		}
	}
	return 0
}

func matchRegex(actual any, expectation Expectation) float64 {
	pattern := fmt.Sprint(expectation.Value)
	matched, err := regexp.MatchString(pattern, fmt.Sprint(actual))
	if err != nil {
		return 0
	}
	return boolFloat(matched)
}

func matchStartsWith(actual any, expectation Expectation) float64 {
	return boolFloat(strings.HasPrefix(fmt.Sprint(actual), fmt.Sprint(expectation.Value)))
}

func matchLatencyBelow(actual any, expectation Expectation) float64 {
	latency, ok := durationValue(actual)
	if !ok {
		return 0
	}
	limit, ok := durationValue(expectation.Value)
	if !ok {
		return 0
	}
	return boolFloat(latency <= limit)
}

func durationValue(value any) (time.Duration, bool) {
	switch typed := value.(type) {
	case time.Duration:
		return typed, true
	case string:
		duration, err := time.ParseDuration(typed)
		return duration, err == nil
	case int:
		return time.Duration(typed) * time.Millisecond, true
	case float64:
		return time.Duration(typed) * time.Millisecond, true
	default:
		return 0, false
	}
}

func matchJSONSchema(actual any, expectation Expectation) float64 {
	schema, ok := mapValue(expectation.Value)
	if !ok {
		return 0
	}
	candidate, ok := jsonCandidate(actual)
	if !ok {
		return 0
	}
	compiled, err := compileJSONSchema(schema)
	if err != nil {
		return 0
	}
	return boolFloat(compiled.Validate(candidate) == nil)
}

func compileJSONSchema(schema map[string]any) (*jsonschema.Schema, error) {
	document, err := jsonCompatibleValue(schema)
	if err != nil {
		return nil, err
	}
	compiler := jsonschema.NewCompiler()
	if err := compiler.AddResource("schema.json", document); err != nil {
		return nil, err
	}
	return compiler.Compile("schema.json")
}

func jsonCompatibleValue(value any) (any, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	return jsonschema.UnmarshalJSON(bytes.NewReader(data))
}

func jsonCandidate(actual any) (any, bool) {
	if text, ok := actual.(string); ok {
		decoded, err := jsonschema.UnmarshalJSON(bytes.NewBufferString(text))
		return decoded, err == nil
	}
	return actual, actual != nil
}

func matchToolArgs(actual any, expectation Expectation) float64 {
	calls := toolCallsValue(actual)
	if len(calls) == 0 {
		return 0
	}
	expected := expectedToolArgSpecs(expectation.Value)
	if len(expected) == 0 {
		return 0
	}
	for _, spec := range expected {
		if !toolArgSpecMatched(calls, spec) {
			return 0
		}
	}
	return 1
}

type toolArgSpec struct {
	Name      string
	Arguments map[string]any
}

func expectedToolArgSpecs(value any) []toolArgSpec {
	if list, ok := value.([]any); ok {
		return toolArgSpecList(list)
	}
	if mapped, ok := mapValue(value); ok {
		return []toolArgSpec{toolArgSpecFromMap(mapped)}
	}
	return nil
}

func toolArgSpecList(values []any) []toolArgSpec {
	out := make([]toolArgSpec, 0, len(values))
	for _, value := range values {
		mapped, ok := mapValue(value)
		if ok {
			out = append(out, toolArgSpecFromMap(mapped))
		}
	}
	return out
}

func toolArgSpecFromMap(value map[string]any) toolArgSpec {
	arguments, _ := mapValue(value["arguments"])
	return toolArgSpec{Name: fmt.Sprint(value["name"]), Arguments: arguments}
}

func toolArgSpecMatched(calls []ToolCallSummary, spec toolArgSpec) bool {
	for _, call := range calls {
		if spec.Name != "" && call.Name != spec.Name {
			continue
		}
		if containsArgumentMap(call.ArgumentsValue, spec.Arguments) {
			return true
		}
	}
	return false
}

func containsArgumentMap(actual map[string]any, expected map[string]any) bool {
	for key, want := range expected {
		got, ok := actual[key]
		if !ok || !equalValues(got, want) {
			return false
		}
	}
	return true
}

func toolCallsValue(value any) []ToolCallSummary {
	if calls, ok := value.([]ToolCallSummary); ok {
		return calls
	}
	return nil
}

func mapValue(value any) (map[string]any, bool) {
	if mapped, ok := value.(map[string]any); ok {
		return mapped, true
	}
	return nil, false
}
