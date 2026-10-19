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
// JSON string. A struct whose fields JSON cannot see, such as one with only unexported fields,
// is compared by its Go syntax (%#v) instead, since its JSON is always {}. A missing or nil
// value, a nil pointer included, has no text form and matches only another nil value.
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
		if matchMetricText(expected[field], actual[field], m.normalise, func(got, want string) bool {
			return got == want
		}) {
			matched++
		} else {
			missing = append(missing, field)
		}
	}
	return metricResult(float64(matched)/float64(len(fields)), missing), nil
}

// normalise applies the TrimSpace and CaseInsensitive options to a canonical text form.
func (m ExactMatchMetric) normalise(text string) string {
	if m.TrimSpace {
		text = strings.TrimSpace(text)
	}
	if m.CaseInsensitive {
		text = strings.ToLower(text)
	}
	return text
}

// ContainsMetric scores fields when the actual value's canonical text form, as described on
// ExactMatchMetric, contains the expected value's. A nil value matches only another nil value.
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
		if matchMetricText(expected[field], actual[field], m.normalise, strings.Contains) {
			matched++
		} else {
			missing = append(missing, field)
		}
	}
	return metricResult(float64(matched)/float64(len(fields)), missing), nil
}

// normalise applies the CaseInsensitive option to a canonical text form.
func (m ContainsMetric) normalise(text string) string {
	if m.CaseInsensitive {
		return strings.ToLower(text)
	}
	return text
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
// label such as a named string type matches its plain string. A nil label matches only another
// nil label, and is null in the result details.
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
	want, wantOK := m.normaliseValue(expected[field])
	got, gotOK := m.normaliseValue(actual[field])
	correct := wantOK == gotOK && want == got
	kind := boolKind(correct)
	if wantOK && gotOK {
		kind = classificationKind(want, got, positive, negative)
	}
	result := MetricResult{
		Score:           boolScore(correct),
		ObjectiveScores: classificationObjectiveScores(kind),
		Details: map[string]any{
			"field":          field,
			"expected":       detailLabel(want, wantOK),
			"actual":         detailLabel(got, gotOK),
			"kind":           kind,
			"false_positive": kind == "false_positive",
			"false_negative": kind == "false_negative",
		},
	}
	if !correct {
		result.Feedback = "classification mismatch: " + kind
	}
	return result, nil
}

// normaliseValue returns the normalised label of value, and false when value is nil.
func (m ClassificationMetric) normaliseValue(value any) (string, bool) {
	text, ok := metricText(value)
	return m.normalise(text), ok
}

// boolKind is the classification kind of a pair that is not two labels.
func boolKind(correct bool) string {
	if correct {
		return "correct"
	}
	return "incorrect"
}

// detailLabel is a label as the result details show it: nil for a nil value.
func detailLabel(label string, ok bool) any {
	if !ok {
		return nil
	}
	return label
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

// matchMetricText reports whether match holds for the normalised canonical text forms of got
// and want. A nil value has no text form: it matches only another nil value.
func matchMetricText(want, got any, normalise func(string) string, match func(got, want string) bool) bool {
	wantText, wantOK := metricText(want)
	gotText, gotOK := metricText(got)
	if !wantOK || !gotOK {
		return wantOK == gotOK
	}
	return match(normalise(gotText), normalise(wantText))
}

// metricText returns the canonical text form the built-in text metrics compare: a string as
// it is, and any other value its canonical JSON encoding from metricJSON, unquoted when that
// encoding is a JSON string. It returns false for a value whose encoding is null, such as a
// missing value or a nil pointer, which has no text form, so that it differs from the string
// "null".
func metricText(value any) (string, bool) {
	if text, ok := value.(string); ok {
		return text, true
	}
	encoded := metricJSON(value)
	if string(encoded) == "null" {
		return "", false
	}
	var text string
	if encoded[0] == '"' && json.Unmarshal(encoded, &text) == nil {
		return text, true
	}
	return string(encoded), true
}

// metricJSON returns the canonical JSON encoding of value: pointers followed, object keys
// sorted whether value is a struct or a map, numbers kept as encoded and HTML left unescaped.
// A value JSON cannot encode, such as NaN or a channel, is a JSON string of its fmt form, and a
// struct whose fields JSON cannot see, which encodes as {} whatever it holds, a JSON string of
// its Go syntax (%#v).
func metricJSON(value any) json.RawMessage {
	encoded, err := encodeMetricJSON(value)
	if err != nil {
		// a string always encodes.
		encoded, _ = encodeMetricJSON(fmt.Sprint(value))
		return encoded
	}
	if string(encoded) == "{}" {
		if hidden, ok := hiddenStruct(value); ok {
			encoded, _ = encodeMetricJSON(fmt.Sprintf("%#v", hidden.Interface()))
			return encoded
		}
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

// hiddenStruct returns the struct value points to, following pointers and interfaces, when it
// has fields but none that JSON can see, such as only unexported fields or fields tagged "-".
func hiddenStruct(value any) (reflect.Value, bool) {
	v := reflect.ValueOf(value)
	for v.Kind() == reflect.Pointer || v.Kind() == reflect.Interface {
		if v.IsNil() {
			return reflect.Value{}, false
		}
		v = v.Elem()
	}
	return v, v.Kind() == reflect.Struct && v.NumField() > 0 && len(jsonFields(v.Type())) == 0
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
