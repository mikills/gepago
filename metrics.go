package gepa

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
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

// ExactMatchMetric scores fields by exact equality of their canonical text forms. A string
// is compared as it is; any other value, typed Go values from TypedPredict or a pipeline
// included, by its JSON encoding with object keys sorted, unquoted when that encoding is a
// JSON string. A missing or nil value, a nil pointer included, is null.
type ExactMatchMetric struct {
	Fields          []string
	CaseInsensitive bool
	TrimSpace       bool
}

// Score returns the fraction of fields whose canonical text forms are equal.
func (m ExactMatchMetric) Score(_ context.Context, expected Prediction, actual Prediction) (MetricResult, error) {
	fields := metricFields(m.Fields, expected)
	if len(fields) == 0 {
		return MetricResult{}, nil
	}
	matched := 0
	missing := make([]string, 0)
	for _, field := range fields {
		want := metricText(expected[field])
		got := metricText(actual[field])
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

// ContainsMetric scores fields when the actual value's canonical text form, as described on
// ExactMatchMetric, contains the expected value's.
type ContainsMetric struct {
	Fields          []string
	CaseInsensitive bool
}

// Score returns the fraction of fields whose actual text contains the expected text.
func (m ContainsMetric) Score(_ context.Context, expected Prediction, actual Prediction) (MetricResult, error) {
	fields := metricFields(m.Fields, expected)
	if len(fields) == 0 {
		return MetricResult{}, nil
	}
	matched := 0
	missing := make([]string, 0)
	for _, field := range fields {
		want := metricText(expected[field])
		got := metricText(actual[field])
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

// NumericMatchMetric scores numeric fields within a tolerance. A value is numeric when it is
// any Go integer or floating-point kind, a json.Number or a string holding a number (commas
// are ignored), or a pointer to one of these.
type NumericMatchMetric struct {
	Fields    []string
	Tolerance float64
}

// ClassificationMetric scores binary classifications and exposes Pareto-safe objectives. Labels
// are compared by their canonical text forms, as described on ExactMatchMetric, so a typed
// label such as a named string type matches its plain string.
type ClassificationMetric struct {
	Field         string
	PositiveLabel string
	NegativeLabel string
	TrimSpace     bool
	IgnoreCase    bool
}

// Score compares the expected and actual labels and classifies the outcome.
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
	return m.normalise(metricText(value))
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

// Score returns the fraction of fields whose numeric values are within the tolerance.
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

// numericValue reads value as a float64, following pointers and interfaces. It accepts every
// integer and floating-point kind and a string kind, json.Number included, holding a number.
func numericValue(value any) (float64, bool) {
	v := reflect.ValueOf(value)
	for v.Kind() == reflect.Pointer || v.Kind() == reflect.Interface {
		if v.IsNil() {
			return 0, false
		}
		v = v.Elem()
	}
	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return float64(v.Int()), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return float64(v.Uint()), true
	case reflect.Float32, reflect.Float64:
		return v.Float(), true
	case reflect.String:
		parsed, err := strconv.ParseFloat(strings.ReplaceAll(strings.TrimSpace(v.String()), ",", ""), 64)
		return parsed, err == nil
	default:
		return 0, false
	}
}

// metricText returns the canonical text form the built-in text metrics compare: a string as
// it is, and any other value its canonical JSON encoding from metricJSON, unquoted when that
// encoding is a JSON string. A missing or nil value is therefore "null".
func metricText(value any) string {
	if text, ok := value.(string); ok {
		return text
	}
	encoded := metricJSON(value)
	var text string
	if encoded[0] == '"' && json.Unmarshal(encoded, &text) == nil {
		return text
	}
	return string(encoded)
}

// metricJSON returns the canonical JSON encoding of value: pointers followed, object keys
// sorted whether value is a struct or a map, numbers kept as encoded and HTML left unescaped.
// A value JSON cannot encode, such as NaN or a channel, is a JSON string of its fmt form.
func metricJSON(value any) json.RawMessage {
	encoded, err := encodeMetricJSON(value)
	if err != nil {
		// a string always encodes.
		encoded, _ = encodeMetricJSON(fmt.Sprint(value))
		return encoded
	}
	var decoded any
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	if decoder.Decode(&decoded) != nil {
		// only nesting deeper than the decoder allows lands here; keep the first encoding.
		return encoded
	}
	// decoded values are maps, slices, strings, numbers, booleans and nil, which always encode.
	canonical, _ := encodeMetricJSON(decoded)
	return canonical
}

// encodeMetricJSON encodes value as compact JSON without escaping HTML.
func encodeMetricJSON(value any) (json.RawMessage, error) {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return bytes.TrimSpace(buffer.Bytes()), nil
}
