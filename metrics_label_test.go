package gepa_test

import (
	"context"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/stretchr/testify/require"
)

func TestLabelMetric(t *testing.T) {
	support := gepa.LabelMetric{Labels: []string{"billing", "refund", "shipping"}, Critical: []string{"refund"}}
	tests := []struct {
		name             string
		metric           gepa.LabelMetric
		expected, actual any
		score            float64
		kind             string
		feedback         string
		missed, alarm    float64
	}{
		{
			name:     "correct",
			metric:   support,
			expected: "billing",
			actual:   "billing",
			score:    1,
			kind:     "correct",
			missed:   1,
			alarm:    1,
		},
		{
			name:     "critical correct",
			metric:   support,
			expected: "refund",
			actual:   "refund",
			score:    1,
			kind:     "correct",
			missed:   1,
			alarm:    1,
		},
		{
			name:     "wrong",
			metric:   support,
			expected: "billing",
			actual:   "shipping",
			kind:     "incorrect",
			feedback: `expected "billing", got "shipping"`,
			missed:   1,
			alarm:    1,
		},
		{
			name:     "missed critical",
			metric:   support,
			expected: "refund",
			actual:   "billing",
			kind:     "incorrect",
			feedback: `expected "refund", got "billing"`,
			alarm:    1,
		},
		{
			name:     "false alarm",
			metric:   support,
			expected: "billing",
			actual:   "refund",
			kind:     "incorrect",
			feedback: `expected "billing", got "refund"`,
			missed:   1,
		},
		{
			name:     "unknown label",
			metric:   support,
			expected: "refund",
			actual:   "cancel",
			kind:     "unknown_label",
			feedback: `expected "refund", got "cancel", which is not one of the labels`,
			alarm:    1,
		},
		{
			name:     "no labels to check",
			metric:   gepa.LabelMetric{Critical: []string{"refund"}},
			expected: "billing",
			actual:   "cancel",
			kind:     "incorrect",
			feedback: `expected "billing", got "cancel"`,
			missed:   1,
			alarm:    1,
		},
		{
			name:     "no actual label",
			metric:   support,
			expected: "refund",
			kind:     "incorrect",
			feedback: `expected "refund", got no label`,
			alarm:    1,
		},
		{
			name:     "no expected label",
			metric:   support,
			actual:   "refund",
			kind:     "incorrect",
			feedback: `expected no label, got "refund"`,
			missed:   1,
		},
		{
			name:   "neither label",
			metric: support,
			actual: (*string)(nil),
			score:  1,
			kind:   "correct",
			missed: 1,
			alarm:  1,
		},
		{
			name:     "typed label",
			metric:   support,
			expected: "refund",
			actual:   metricLabel("refund"),
			score:    1,
			kind:     "correct",
			missed:   1,
			alarm:    1,
		},
		{
			name:     "case sensitive",
			metric:   support,
			expected: "refund",
			actual:   "Refund",
			kind:     "unknown_label",
			feedback: `expected "refund", got "Refund", which is not one of the labels`,
			alarm:    1,
		},
		{
			name: "normalised",
			metric: gepa.LabelMetric{
				Labels:     []string{"Billing", " Refund"},
				Critical:   []string{"refund"},
				TrimSpace:  true,
				IgnoreCase: true,
			},
			expected: "refund",
			actual:   " REFUND\n",
			score:    1,
			kind:     "correct",
			missed:   1,
			alarm:    1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := tt.metric.Score(
				context.Background(),
				gepa.Prediction{"label": tt.expected},
				gepa.Prediction{"label": tt.actual},
			)
			require.NoError(t, err)
			require.Equal(t, tt.score, result.Score)
			require.Equal(t, tt.feedback, result.Feedback)
			require.Equal(t, map[string]float64{
				"accuracy":           tt.score,
				"missed:refund":      tt.missed,
				"false_alarm:refund": tt.alarm,
			}, result.ObjectiveScores)
			require.Equal(t, "label", result.Details["field"])
			require.Equal(t, tt.kind, result.Details["kind"])
		})
	}

	t.Run("details", func(t *testing.T) {
		result, err := gepa.LabelMetric{
			Field:      "category",
			Critical:   []string{"Refund", "fraud"},
			IgnoreCase: true,
		}.Score(context.Background(), gepa.Prediction{"category": "REFUND"}, gepa.Prediction{"category": nil})
		require.NoError(t, err)
		require.Equal(t, map[string]any{
			"field":    "category",
			"expected": "refund",
			"actual":   nil,
			"kind":     "incorrect",
		}, result.Details)
		require.Equal(t, map[string]float64{
			"accuracy":           0,
			"missed:Refund":      0,
			"false_alarm:Refund": 1,
			"missed:fraud":       1,
			"false_alarm:fraud":  1,
		}, result.ObjectiveScores)
	})
}
