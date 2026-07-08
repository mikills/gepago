package crucible

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	gepa "github.com/mikills/gepago"
)

// EvaluatorSpec is the serialized form for built-in evaluators.
type EvaluatorSpec struct {
	Type           string            `json:"type"`
	Name           string            `json:"name,omitempty"`
	Weight         float64           `json:"weight,omitempty"`
	Fields         []string          `json:"fields,omitempty"`
	RequiredFields []string          `json:"required_fields,omitempty"`
	Rubric         string            `json:"rubric,omitempty"`
	Tolerance      float64           `json:"tolerance,omitempty"`
	MaxLatency     string            `json:"max_latency,omitempty"`
	MaxTokens      int               `json:"max_tokens,omitempty"`
	URL            string            `json:"url,omitempty"`
	Method         string            `json:"method,omitempty"`
	Headers        map[string]string `json:"headers,omitempty"`
	Command        string            `json:"command,omitempty"`
	Args           []string          `json:"args,omitempty"`
	Env            []string          `json:"env,omitempty"`
}

// EvaluatorFactoryConfig provides runtime dependencies for serialized evaluator specs.
type EvaluatorFactoryConfig struct {
	JudgeLM      gepa.LanguageModel
	AllowCommand bool
}

// LoadSuiteJSON reads an Crucible suite from JSON.
func LoadSuiteJSON(path string) (Suite, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Suite{}, err
	}
	var suite Suite
	if err := json.Unmarshal(data, &suite); err != nil {
		return Suite{}, err
	}
	return suite, validateSuite(suite)
}

// BuildEvaluators constructs built-in evaluators from suite specs.
func BuildEvaluators(specs []EvaluatorSpec, config EvaluatorFactoryConfig) ([]WeightedEvaluator, error) {
	evaluators := make([]WeightedEvaluator, 0, len(specs))
	for _, spec := range specs {
		evaluator, err := buildEvaluator(spec, config)
		if err != nil {
			return nil, err
		}
		evaluators = append(evaluators, WeightedEvaluator{Evaluator: evaluator, Weight: normalizedWeight(spec.Weight)})
	}
	return evaluators, nil
}

func validateSuite(suite Suite) error {
	if strings.TrimSpace(suite.Name) == "" {
		return fmt.Errorf("suite name is required")
	}
	if len(suite.Cases) == 0 {
		return fmt.Errorf("suite requires at least one case")
	}
	for i, evalCase := range suite.Cases {
		if strings.TrimSpace(evalCase.ID) == "" {
			return fmt.Errorf("case %d id is required", i)
		}
	}
	return nil
}

func buildEvaluator(spec EvaluatorSpec, config EvaluatorFactoryConfig) (Evaluator, error) {
	evaluatorType := strings.TrimSpace(spec.Type)
	if evaluator, ok := buildMetricEvaluator(evaluatorType, spec); ok {
		return evaluator, nil
	}
	if evaluator, ok := buildSimpleEvaluator(evaluatorType, spec, config); ok {
		return evaluator, nil
	}
	if evaluatorType == "budget" || evaluatorType == "cost_latency" {
		return buildBudgetEvaluator(spec)
	}
	return nil, fmt.Errorf("unknown evaluator type %q", spec.Type)
}

func buildMetricEvaluator(evaluatorType string, spec EvaluatorSpec) (Evaluator, bool) {
	switch evaluatorType {
	case "exact_match":
		return namedMetric(spec, gepa.ExactMatchMetric{Fields: spec.Fields, TrimSpace: true}), true
	case "contains":
		return namedMetric(spec, gepa.ContainsMetric{Fields: spec.Fields, CaseInsensitive: true}), true
	case "numeric_tolerance":
		return namedMetric(spec, gepa.NumericMatchMetric{Fields: spec.Fields, Tolerance: spec.Tolerance}), true
	case "classification":
		metric := gepa.ClassificationMetric{Field: firstField(spec.Fields), TrimSpace: true, IgnoreCase: true}
		return namedMetric(spec, metric), true
	default:
		return nil, false
	}
}

func buildSimpleEvaluator(
	evaluatorType string,
	spec EvaluatorSpec,
	config EvaluatorFactoryConfig,
) (Evaluator, bool) {
	switch evaluatorType {
	case "json_validity", "schema_validity":
		return JSONValidityEvaluator{RequiredFields: spec.RequiredFields}, true
	case "rubric_judge":
		return RubricJudgeEvaluator{LM: config.JudgeLM, Rubric: spec.Rubric}, true
	case toolCallsKey:
		return ToolCallEvaluator{ExpectedNames: spec.Fields, RequireOrder: true}, true
	case "expectations":
		return ExpectationEvaluator{}, true
	case "command", "command_evaluator":
		if !config.AllowCommand {
			return nil, false
		}
		return CommandEvaluator{EvaluatorName: spec.Name, Command: spec.Command, Args: spec.Args, Env: spec.Env}, true
	case "webhook", "webhook_evaluator":
		evaluator := WebhookEvaluator{EvaluatorName: spec.Name, URL: spec.URL, Method: spec.Method, Headers: spec.Headers}
		return evaluator, true
	default:
		return nil, false
	}
}

func buildBudgetEvaluator(spec EvaluatorSpec) (Evaluator, error) {
	latency, err := parseOptionalDuration(spec.MaxLatency)
	if err != nil {
		return nil, err
	}
	return BudgetEvaluator{MaxLatency: latency, MaxTokens: spec.MaxTokens}, nil
}

func namedMetric(spec EvaluatorSpec, metric gepa.Metric) MetricEvaluator {
	name := strings.TrimSpace(spec.Name)
	if name == "" {
		name = spec.Type
	}
	return MetricEvaluator{EvaluatorName: name, Metric: metric}
}

func firstField(fields []string) string {
	if len(fields) == 0 {
		return ""
	}
	return fields[0]
}

func parseOptionalDuration(value string) (time.Duration, error) {
	if strings.TrimSpace(value) == "" {
		return 0, nil
	}
	latency, err := time.ParseDuration(value)
	if err != nil {
		return 0, fmt.Errorf("parse max latency %q: %w", value, err)
	}
	return latency, nil
}
