package gepa

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Prediction is a structured model output keyed by signature output field name.
type Prediction map[string]any

// MetricResult is a score plus optional natural-language feedback for reflection.
type MetricResult struct {
	Score           float64            `json:"score"`
	ObjectiveScores map[string]float64 `json:"objective_scores,omitempty"`
	Feedback        string             `json:"feedback,omitempty"`
	Details         map[string]any     `json:"details,omitempty"`
}

// Metric scores a prediction against expected output and returns feedback for reflection.
type Metric interface {
	Score(ctx context.Context, expected Prediction, actual Prediction) (MetricResult, error)
}

// MetricFunc adapts a function to the Metric interface.
type MetricFunc func(ctx context.Context, expected Prediction, actual Prediction) (MetricResult, error)

func (fn MetricFunc) Score(ctx context.Context, expected Prediction, actual Prediction) (MetricResult, error) {
	return fn(ctx, expected, actual)
}

// ExactMatchMetric scores fields by exact string equality.
type ExactMatchMetric struct {
	Fields          []string
	CaseInsensitive bool
	TrimSpace       bool
}

func (m ExactMatchMetric) Score(_ context.Context, expected Prediction, actual Prediction) (MetricResult, error) {
	fields := metricFields(m.Fields, expected)
	if len(fields) == 0 {
		return MetricResult{}, nil
	}
	matched := 0
	missing := make([]string, 0)
	for _, field := range fields {
		want := fmt.Sprint(expected[field])
		got := fmt.Sprint(actual[field])
		if m.TrimSpace {
			want = strings.TrimSpace(want)
			got = strings.TrimSpace(got)
		}
		if m.CaseInsensitive {
			want = strings.ToLower(want)
			got = strings.ToLower(got)
		}
		if want == got {
			matched++
		} else {
			missing = append(missing, field)
		}
	}
	return metricResult(float64(matched)/float64(len(fields)), missing), nil
}

// ContainsMetric scores fields when the actual string contains the expected string.
type ContainsMetric struct {
	Fields          []string
	CaseInsensitive bool
}

func (m ContainsMetric) Score(_ context.Context, expected Prediction, actual Prediction) (MetricResult, error) {
	fields := metricFields(m.Fields, expected)
	if len(fields) == 0 {
		return MetricResult{}, nil
	}
	matched := 0
	missing := make([]string, 0)
	for _, field := range fields {
		want := fmt.Sprint(expected[field])
		got := fmt.Sprint(actual[field])
		if m.CaseInsensitive {
			want = strings.ToLower(want)
			got = strings.ToLower(got)
		}
		if strings.Contains(got, want) {
			matched++
		} else {
			missing = append(missing, field)
		}
	}
	return metricResult(float64(matched)/float64(len(fields)), missing), nil
}

// NumericMatchMetric scores numeric fields within a tolerance.
type NumericMatchMetric struct {
	Fields    []string
	Tolerance float64
}

// ClassificationMetric scores binary classifications and exposes Pareto-safe objectives.
type ClassificationMetric struct {
	Field         string
	PositiveLabel string
	NegativeLabel string
	TrimSpace     bool
	IgnoreCase    bool
}

func (m ClassificationMetric) Score(_ context.Context, expected Prediction, actual Prediction) (MetricResult, error) {
	field := m.Field
	if field == "" {
		field = "label"
	}
	positive := m.normalise(m.PositiveLabel)
	if positive == "" {
		positive = "positive"
	}
	negative := m.normalise(m.NegativeLabel)
	if negative == "" {
		negative = "negative"
	}
	want := m.normaliseValue(expected[field])
	got := m.normaliseValue(actual[field])
	kind := classificationKind(want, got, positive, negative)
	result := MetricResult{
		Score:           boolScore(want == got),
		ObjectiveScores: classificationObjectiveScores(kind),
		Details: map[string]any{
			"field":          field,
			"expected":       want,
			"actual":         got,
			"kind":           kind,
			"false_positive": kind == "false_positive",
			"false_negative": kind == "false_negative",
		},
	}
	if want != got {
		result.Feedback = "classification mismatch: " + kind
	}
	return result, nil
}

func (m ClassificationMetric) normaliseValue(value any) string {
	return m.normalise(fmt.Sprint(value))
}

func (m ClassificationMetric) normalise(value string) string {
	if m.TrimSpace {
		value = strings.TrimSpace(value)
	}
	if m.IgnoreCase {
		value = strings.ToLower(value)
	}
	return value
}

func (m NumericMatchMetric) Score(_ context.Context, expected Prediction, actual Prediction) (MetricResult, error) {
	fields := metricFields(m.Fields, expected)
	if len(fields) == 0 {
		return MetricResult{}, nil
	}
	matched := 0
	missing := make([]string, 0)
	for _, field := range fields {
		want, wantOK := numericValue(expected[field])
		got, gotOK := numericValue(actual[field])
		if wantOK && gotOK && math.Abs(want-got) <= m.Tolerance {
			matched++
		} else {
			missing = append(missing, field)
		}
	}
	return metricResult(float64(matched)/float64(len(fields)), missing), nil
}

func classificationKind(expected string, actual string, positive string, negative string) string {
	if kind, ok := binaryClassificationKind(expected, actual, positive, negative); ok {
		return kind
	}
	if expected == actual {
		return "correct"
	}
	return "incorrect"
}

func binaryClassificationKind(expected string, actual string, positive string, negative string) (string, bool) {
	key := expected + "\x00" + actual
	kinds := map[string]string{
		positive + "\x00" + positive: "true_positive",
		negative + "\x00" + negative: "true_negative",
		negative + "\x00" + positive: "false_positive",
		positive + "\x00" + negative: "false_negative",
	}
	kind, ok := kinds[key]
	return kind, ok
}

func classificationObjectiveScores(kind string) map[string]float64 {
	return map[string]float64{
		"accuracy":             boolScore(kind == "true_positive" || kind == "true_negative" || kind == "correct"),
		"avoid_false_positive": boolScore(kind != "false_positive"),
		"avoid_false_negative": boolScore(kind != "false_negative"),
		"true_positive":        boolScore(kind == "true_positive"),
		"true_negative":        boolScore(kind == "true_negative"),
	}
}

func boolScore(ok bool) float64 {
	if ok {
		return 1
	}
	return 0
}

func metricFields(configured []string, expected Prediction) []string {
	if len(configured) > 0 {
		return append([]string(nil), configured...)
	}
	fields := make([]string, 0, len(expected))
	for field := range expected {
		fields = append(fields, field)
	}
	return fields
}

func metricResult(score float64, missing []string) MetricResult {
	result := MetricResult{Score: score}
	if len(missing) > 0 {
		result.Feedback = "mismatched fields: " + strings.Join(missing, ", ")
		result.Details = map[string]any{"mismatched_fields": missing}
	}
	return result
}

func numericValue(value any) (float64, bool) {
	switch v := value.(type) {
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case json.Number:
		parsed, err := strconv.ParseFloat(string(v), 64)
		return parsed, err == nil
	case string:
		parsed, err := strconv.ParseFloat(strings.ReplaceAll(strings.TrimSpace(v), ",", ""), 64)
		return parsed, err == nil
	default:
		return 0, false
	}
}
