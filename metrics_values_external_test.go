package gepa_test

import (
	"context"
	"encoding/json"
	"math"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/stretchr/testify/require"
)

type metricLabel string

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
func datasetExpected(t *testing.T, text string) gepa.Prediction {
	t.Helper()
	var expected gepa.Prediction
	require.NoError(t, json.Unmarshal([]byte(text), &expected))
	return expected
}

const metricExpectedJSON = `{"count": 3, "ref": "r-1", "missing": null, "label": "positive",
	"tags": ["a", "b"], "obligation": {"amount": 10, "deadline": "2026-06-30"}}`

func TestTypedPredictScoresWithMetrics(t *testing.T) {
	program := gepa.TypedPredict[metricIn, metricScoreOut]{
		LM: replyModel(`{"count": 3, "ref": "r-1"}`),
	}
	examples := []gepa.Example{
		gepa.NewIOExample("one", gepa.Prediction{"q": "x"}, gepa.Prediction{"count": 3, "ref": "r-1"}),
	}
	got, err := program.RunCandidate(context.Background(), program.SeedCandidate(), gepa.Prediction{"q": "x"})
	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"count": int32(3), "ref": pointerTo("r-1")}, got)

	numeric, err := gepa.ProgramEvaluator{Program: program, Metric: gepa.NumericMatchMetric{Fields: []string{"count"}}}.
		Evaluate(context.Background(), program.SeedCandidate(), examples, false)
	require.NoError(t, err)
	exact, err := gepa.ProgramEvaluator{Program: program, Metric: gepa.ExactMatchMetric{Fields: []string{"ref"}}}.
		Evaluate(context.Background(), program.SeedCandidate(), examples, false)
	require.NoError(t, err)
	require.Equal(t, 1.0, numeric.Items[0].Score, "a correct int32 count scores 1")
	require.Equal(t, 1.0, exact.Items[0].Score, "a correct ref scores 1")
}

func TestTypedPredictOutputsScoreWithEveryMetric(t *testing.T) {
	reply := `{"count": 3, "ref": "r-1", "missing": null, "label": "positive", "tags": ["a", "b"],
		"obligation": {"deadline": "2026-06-30", "amount": 10}}`
	program := gepa.TypedPredict[metricIn, metricTypedOut]{
		LM: replyModel(reply),
	}
	examples := []gepa.Example{
		gepa.NewIOExample("one", gepa.Prediction{"q": "x"}, datasetExpected(t, metricExpectedJSON)),
	}
	for name, metric := range typedMetrics() {
		result, err := gepa.ProgramEvaluator{Program: program, Metric: metric}.
			Evaluate(context.Background(), program.SeedCandidate(), examples, false)
		require.NoError(t, err, name)
		require.Equal(t, 1.0, result.Items[0].Score, "%s: %v", name, result.Items[0].SideInfo)
	}
}

func TestLLMJudgeMetricRendersTypedValues(t *testing.T) {
	lm := replyModel(`{"score": 1, "feedback": "ok"}`)
	metric := gepa.LLMJudgeMetric{LM: lm}
	actual := gepa.Prediction{
		"count":      int32(3),
		"ref":        pointerTo("<r-1>"),
		"missing":    (*string)(nil),
		"label":      metricLabel("positive"),
		"obligation": metricObligation{Deadline: "2026-06-30", Amount: 10},
		"ratio":      math.NaN(),
	}
	result, err := metric.Score(context.Background(), gepa.Prediction{"count": 3}, actual)
	require.NoError(t, err)
	require.Equal(t, 1.0, result.Score)
	prompt := prompts(lm)[0]
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

// typedMetrics are the built-in metrics over every field of metricTypedOut.
func typedMetrics() map[string]gepa.Metric {
	return map[string]gepa.Metric{
		"exact":          gepa.ExactMatchMetric{},
		"contains":       gepa.ContainsMetric{Fields: []string{"ref", "label", "tags", "obligation"}},
		"numeric":        gepa.NumericMatchMetric{Fields: []string{"count"}},
		"classification": gepa.ClassificationMetric{Field: "label"},
	}
}
