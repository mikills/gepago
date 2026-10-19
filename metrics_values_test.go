package gepa

import (
	"context"
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type metricLabel string

type metricScore float32

// metricObligation declares its fields out of alphabetical order, so its JSON encoding
// differs from that of an equal map until keys are sorted.
type metricObligation struct {
	Deadline string `json:"deadline"`
	Amount   int    `json:"amount"`
}

type metricIn struct {
	Q string `json:"q"`
}

type metricScoreOut struct {
	Count int32   `json:"count"`
	Ref   *string `json:"ref"`
}

type metricTypedOut struct {
	Count      uint16           `json:"count"`
	Ref        *string          `json:"ref"`
	Missing    *string          `json:"missing"`
	Label      metricLabel      `json:"label"`
	Tags       []string         `json:"tags"`
	Obligation metricObligation `json:"obligation"`
}

// datasetExpected decodes expected outputs as a JSON dataset would hold them.
func datasetExpected(t *testing.T, text string) Prediction {
	t.Helper()
	var expected Prediction
	require.NoError(t, json.Unmarshal([]byte(text), &expected))
	return expected
}

const metricExpectedJSON = `{"count": 3, "ref": "r-1", "missing": null, "label": "positive",
	"tags": ["a", "b"], "obligation": {"amount": 10, "deadline": "2026-06-30"}}`

func TestTypedPredictScoresWithMetrics(t *testing.T) {
	program := TypedPredict[metricIn, metricScoreOut]{
		LM: textModel(func(context.Context, string) (string, error) { return `{"count": 3, "ref": "r-1"}`, nil }),
	}
	examples := []Example{NewIOExample("one", Prediction{"q": "x"}, Prediction{"count": 3, "ref": "r-1"})}
	got, err := program.RunCandidate(context.Background(), program.SeedCandidate(), Prediction{"q": "x"})
	require.NoError(t, err)
	require.Equal(t, Prediction{"count": int32(3), "ref": pointerTo("r-1")}, got)

	numeric, err := ProgramEvaluator{Program: program, Metric: NumericMatchMetric{Fields: []string{"count"}}}.
		Evaluate(context.Background(), program.SeedCandidate(), examples, false)
	require.NoError(t, err)
	exact, err := ProgramEvaluator{Program: program, Metric: ExactMatchMetric{Fields: []string{"ref"}}}.
		Evaluate(context.Background(), program.SeedCandidate(), examples, false)
	require.NoError(t, err)
	require.Equal(t, 1.0, numeric.Items[0].Score, "a correct int32 count scores 1")
	require.Equal(t, 1.0, exact.Items[0].Score, "a correct ref scores 1")
}

// typedMetrics are the built-in metrics over every field of metricTypedOut.
func typedMetrics() map[string]Metric {
	return map[string]Metric{
		"exact":          ExactMatchMetric{},
		"contains":       ContainsMetric{Fields: []string{"ref", "label", "tags", "obligation"}},
		"numeric":        NumericMatchMetric{Fields: []string{"count"}},
		"classification": ClassificationMetric{Field: "label"},
	}
}

func TestTypedPredictOutputsScoreWithEveryMetric(t *testing.T) {
	reply := `{"count": 3, "ref": "r-1", "missing": null, "label": "positive", "tags": ["a", "b"],
		"obligation": {"deadline": "2026-06-30", "amount": 10}}`
	program := TypedPredict[metricIn, metricTypedOut]{
		LM: textModel(func(context.Context, string) (string, error) { return reply, nil }),
	}
	examples := []Example{NewIOExample("one", Prediction{"q": "x"}, datasetExpected(t, metricExpectedJSON))}
	for name, metric := range typedMetrics() {
		result, err := ProgramEvaluator{Program: program, Metric: metric}.
			Evaluate(context.Background(), program.SeedCandidate(), examples, false)
		require.NoError(t, err, name)
		require.Equal(t, 1.0, result.Items[0].Score, "%s: %v", name, result.Items[0].SideInfo)
	}
}

func TestPipelineTypedOutputsScoreWithEveryMetric(t *testing.T) {
	count := Key[uint16]{Name: "count"}
	ref := Key[*string]{Name: "ref"}
	missing := Key[*string]{Name: "missing"}
	label := Key[metricLabel]{Name: "label"}
	tags := Key[[]string]{Name: "tags"}
	obligation := Key[metricObligation]{Name: "obligation"}
	write := NewStep(
		"write",
		refs(docKey.Ref()),
		refs(count.Ref(), ref.Ref(), missing.Ref(), label.Ref(), tags.Ref(), obligation.Ref()),
		func(_ context.Context, s State) error {
			count.Set(s, 3)
			ref.Set(s, pointerTo("r-1"))
			missing.Set(s, nil)
			label.Set(s, "positive")
			tags.Set(s, []string{"a", "b"})
			obligation.Set(s, metricObligation{Deadline: "2026-06-30", Amount: 10})
			return nil
		},
	)
	pipeline, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{write}}.Build()
	require.NoError(t, err)
	examples := []Example{NewIOExample("one", Prediction{"document": "d"}, datasetExpected(t, metricExpectedJSON))}
	for name, metric := range typedMetrics() {
		result, err := ProgramEvaluator{Program: pipeline, Metric: metric}.
			Evaluate(context.Background(), pipeline.SeedCandidate(), examples, false)
		require.NoError(t, err, name)
		require.Equal(t, 1.0, result.Items[0].Score, "%s: %v", name, result.Items[0].SideInfo)
	}
}

func TestMetricsReportTypedMismatches(t *testing.T) {
	expected := datasetExpected(t, metricExpectedJSON)
	actual := Prediction{
		"count":      int64(4),
		"ref":        (*string)(nil),
		"missing":    pointerTo("m"),
		"label":      metricLabel("negative"),
		"tags":       []string{"b", "a"},
		"obligation": metricObligation{Deadline: "2026-06-30", Amount: 11},
	}
	exact, err := ExactMatchMetric{}.Score(context.Background(), expected, actual)
	require.NoError(t, err)
	require.Zero(t, exact.Score)
	require.ElementsMatch(
		t,
		[]string{"count", "ref", "missing", "label", "tags", "obligation"},
		exact.Details["mismatched_fields"],
	)

	numeric, err := NumericMatchMetric{Fields: []string{"count"}}.Score(context.Background(), expected, actual)
	require.NoError(t, err)
	require.Zero(t, numeric.Score)

	classification, err := ClassificationMetric{Field: "label"}.Score(context.Background(), expected, actual)
	require.NoError(t, err)
	require.Equal(t, "false_negative", classification.Details["kind"])
	require.Equal(t, "negative", classification.Details["actual"])
}

func TestContainsMetricMatchesTypedText(t *testing.T) {
	result, err := ContainsMetric{Fields: []string{"ref", "tags"}, CaseInsensitive: true}.Score(
		context.Background(),
		Prediction{"ref": "R-1", "tags": "b"},
		Prediction{"ref": pointerTo("the r-1 ref"), "tags": []metricLabel{"a", "b"}},
	)
	require.NoError(t, err)
	require.Equal(t, 1.0, result.Score)
}

func TestNumericValueKinds(t *testing.T) {
	score := metricScore(1.5)
	number := json.Number("7")
	var empty any
	cases := []struct {
		name  string
		value any
		want  float64
		ok    bool
	}{
		{"int", 1, 1, true},
		{"int8", int8(-8), -8, true},
		{"int16", int16(16), 16, true},
		{"int32", int32(32), 32, true},
		{"int64", int64(64), 64, true},
		{"uint", uint(1), 1, true},
		{"uint8", uint8(8), 8, true},
		{"uint16", uint16(16), 16, true},
		{"uint32", uint32(32), 32, true},
		{"uint64", uint64(64), 64, true},
		{"uintptr", uintptr(9), 9, true},
		{"float32", float32(0.5), 0.5, true},
		{"float64", 2.5, 2.5, true},
		{"named float", score, 1.5, true},
		{"pointer to named float", &score, 1.5, true},
		{"pointer to pointer", pointerTo(pointerTo(int16(3))), 3, true},
		{"pointer to interface", &[]any{4}[0], 4, true},
		{"json number", number, 7, true},
		{"pointer to json number", &number, 7, true},
		{"string with commas", " 1,234.5 ", 1234.5, true},
		{"named string", metricLabel("12"), 12, true},
		{"nil", nil, 0, false},
		{"nil pointer", (*int)(nil), 0, false},
		{"pointer to nil interface", &empty, 0, false},
		{"non-numeric string", "twelve", 0, false},
		{"bool", true, 0, false},
		{"slice", []int{1}, 0, false},
	}
	for _, c := range cases {
		got, ok := numericValue(c.value)
		require.Equal(t, c.ok, ok, c.name)
		require.Equal(t, c.want, got, c.name)
	}
}

func TestMetricTextCanonicalForm(t *testing.T) {
	deep := any("x")
	for range 10001 {
		deep = []any{deep}
	}
	cases := []struct {
		name  string
		value any
		want  string
	}{
		{"string", " a ", " a "},
		{"nil", nil, "null"},
		{"nil pointer", (*string)(nil), "null"},
		{"nil slice", []string(nil), "null"},
		{"pointer to string", pointerTo("r-1"), "r-1"},
		{"named string", metricLabel("positive"), "positive"},
		{"time", time.Date(2026, 6, 30, 0, 0, 0, 0, time.UTC), "2026-06-30T00:00:00Z"},
		{"int32", int32(3), "3"},
		{"whole float", 3.0, "3"},
		{"large float", 1e8, "100000000"},
		{"json number", json.Number("3.50"), "3.50"},
		{"bool", true, "true"},
		{"slice", []string{"<a>", "b"}, `["<a>","b"]`},
		{"struct", metricObligation{Deadline: "d", Amount: 1}, `{"amount":1,"deadline":"d"}`},
		{"map", map[string]any{"b": 1, "a": []any{}}, `{"a":[],"b":1}`},
		{"raw message", json.RawMessage(`{"b": 1, "a": 2}`), `{"a":2,"b":1}`},
		{"NaN", math.NaN(), "NaN"},
		{"NaN in a struct", struct{ V float64 }{math.NaN()}, "{NaN}"},
		{"deeper than the decoder allows", deep, strings.Repeat("[", 10001) + `"x"` + strings.Repeat("]", 10001)},
	}
	for _, c := range cases {
		require.Equal(t, c.want, metricText(c.value), c.name)
	}
	require.True(t, strings.HasPrefix(metricText(make(chan int)), "0x"), "a channel falls back to its fmt form")
}

func TestLLMJudgeMetricRendersTypedValues(t *testing.T) {
	var prompt string
	metric := LLMJudgeMetric{
		LM: textModel(func(_ context.Context, text string) (string, error) {
			prompt = text
			return `{"score": 1, "feedback": "ok"}`, nil
		}),
	}
	actual := Prediction{
		"count":      int32(3),
		"ref":        pointerTo("<r-1>"),
		"missing":    (*string)(nil),
		"label":      metricLabel("positive"),
		"obligation": metricObligation{Deadline: "2026-06-30", Amount: 10},
		"ratio":      math.NaN(),
	}
	result, err := metric.Score(context.Background(), Prediction{"count": 3}, actual)
	require.NoError(t, err)
	require.Equal(t, 1.0, result.Score)
	require.Contains(t, prompt, "Expected JSON:\n```json\n{\n  \"count\": 3\n}\n```")
	require.Contains(t, prompt, strings.Join([]string{
		"Actual JSON:\n```json\n{",
		`  "count": 3,`,
		`  "label": "positive",`,
		`  "missing": null,`,
		`  "obligation": {`,
		`    "amount": 10,`,
		`    "deadline": "2026-06-30"`,
		`  },`,
		`  "ratio": "NaN",`,
		`  "ref": "<r-1>"`,
		"}\n```",
	}, "\n"))
}
