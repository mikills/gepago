package crucible

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	gepa "github.com/mikills/gepago"
)

// MetricEvaluator adapts a GEPA metric for expected-vs-actual evaluation.
type MetricEvaluator struct {
	EvaluatorName string
	Metric        gepa.Metric
}

func (e MetricEvaluator) Name() string {
	if strings.TrimSpace(e.EvaluatorName) != "" {
		return e.EvaluatorName
	}
	return "metric"
}

func (e MetricEvaluator) Evaluate(ctx context.Context, input EvalInput) (Score, error) {
	if input.Case.Expected == nil {
		return Score{Skipped: true, Feedback: "expected output not provided"}, nil
	}
	if e.Metric == nil {
		return Score{}, fmt.Errorf("%s metric is required", e.Name())
	}
	result, err := e.Metric.Score(ctx, input.Case.Expected, input.Output.Value)
	if err != nil {
		return Score{}, err
	}
	return Score{Score: result.Score, Feedback: result.Feedback, Details: result.Details}, nil
}

// ExactMatch returns an evaluator for exact field equality.
func ExactMatch(fields ...string) WeightedEvaluator {
	return WeightedEvaluator{
		Evaluator: MetricEvaluator{
			EvaluatorName: "exact_match",
			Metric: gepa.ExactMatchMetric{
				Fields:          fields,
				TrimSpace:       true,
				CaseInsensitive: false,
			},
		},
		Weight: 1,
	}
}

// Contains returns an evaluator for substring checks.
func Contains(fields ...string) WeightedEvaluator {
	return WeightedEvaluator{
		Evaluator: MetricEvaluator{
			EvaluatorName: "contains",
			Metric:        gepa.ContainsMetric{Fields: fields, CaseInsensitive: true},
		},
		Weight: 1,
	}
}

// NumericTolerance returns an evaluator for numeric fields.
func NumericTolerance(tolerance float64, fields ...string) WeightedEvaluator {
	return WeightedEvaluator{
		Evaluator: MetricEvaluator{
			EvaluatorName: "numeric_tolerance",
			Metric:        gepa.NumericMatchMetric{Fields: fields, Tolerance: tolerance},
		},
		Weight: 1,
	}
}

// Classification returns an evaluator for label classification.
func Classification(field string) WeightedEvaluator {
	return WeightedEvaluator{
		Evaluator: MetricEvaluator{
			EvaluatorName: "classification",
			Metric: gepa.ClassificationMetric{
				Field:      field,
				TrimSpace:  true,
				IgnoreCase: true,
			},
		},
		Weight: 1,
	}
}

// JSONValidityEvaluator scores parseability and required field presence.
type JSONValidityEvaluator struct {
	RequiredFields []string
}

func (e JSONValidityEvaluator) Name() string { return "json_validity" }

func (e JSONValidityEvaluator) Evaluate(_ context.Context, input EvalInput) (Score, error) {
	value := input.Output.Value
	if value == nil && strings.TrimSpace(input.Output.Raw) != "" {
		parsed, err := gepa.ParsePrediction(input.Output.Raw)
		if err != nil {
			return Score{Score: 0, Feedback: err.Error()}, nil
		}
		value = parsed
	}
	if value == nil {
		return Score{Score: 0, Feedback: "output is not structured JSON"}, nil
	}
	missing := missingFields(value, e.RequiredFields)
	if len(missing) > 0 {
		return Score{
			Score:    0,
			Feedback: "missing required fields: " + strings.Join(missing, ", "),
			Details:  map[string]any{"missing_fields": missing},
		}, nil
	}
	return Score{Score: 1}, nil
}

// BudgetEvaluator scores operational fit against latency and token budgets.
type BudgetEvaluator struct {
	MaxLatency time.Duration
	MaxTokens  int
}

func (e BudgetEvaluator) Name() string { return "budget" }

func (e BudgetEvaluator) Evaluate(_ context.Context, input EvalInput) (Score, error) {
	parts := []float64{}
	details := map[string]any{}
	if e.MaxLatency > 0 {
		parts = append(parts, budgetScore(float64(input.Output.Latency), float64(e.MaxLatency)))
		details["latency"] = input.Output.Latency.String()
		details["max_latency"] = e.MaxLatency.String()
	}
	if e.MaxTokens > 0 {
		parts = append(parts, budgetScore(float64(input.Output.Usage.TotalTokens), float64(e.MaxTokens)))
		details["total_tokens"] = input.Output.Usage.TotalTokens
		details["max_tokens"] = e.MaxTokens
	}
	if len(parts) == 0 {
		return Score{Skipped: true, Feedback: "no budget configured"}, nil
	}
	return Score{Score: average(parts), Details: details}, nil
}

// FuncEvaluator adapts custom scoring logic.
type FuncEvaluator struct {
	EvaluatorName string
	Func          func(context.Context, EvalInput) (Score, error)
}

func (e FuncEvaluator) Name() string { return e.EvaluatorName }

func (e FuncEvaluator) Evaluate(ctx context.Context, input EvalInput) (Score, error) {
	if e.Func == nil {
		return Score{}, fmt.Errorf("%s evaluator function is required", e.Name())
	}
	return e.Func(ctx, input)
}

func missingFields(value gepa.Prediction, fields []string) []string {
	missing := []string{}
	for _, field := range fields {
		if _, ok := value[field]; !ok {
			missing = append(missing, field)
		}
	}
	return missing
}

func budgetScore(actual float64, limit float64) float64 {
	if limit <= 0 || actual <= limit {
		return 1
	}
	return math.Max(0, limit/actual)
}

func average(values []float64) float64 {
	var sum float64
	for _, value := range values {
		sum += value
	}
	if len(values) == 0 {
		return 0
	}
	return sum / float64(len(values))
}

func marshalJSONMap(value any) string {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Sprint(value)
	}
	return string(data)
}
