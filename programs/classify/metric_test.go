package classify_test

import (
	"context"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/programs/classify"
	"github.com/stretchr/testify/require"
)

func supportMetric() classify.Metric {
	return classify.Metric{
		Labels: []classify.Label{
			{Name: "billing", Description: "Charges and invoices."},
			{Name: "refund", Description: "Requests for money back.", Critical: true},
			{Name: "shipping", Description: "Deliveries."},
		},
		AbstainScore: 0.5,
	}
}

func TestMetric(t *testing.T) {
	t.Run("score", func(t *testing.T) {
		tests := []struct {
			name          string
			expected      string
			decision      any
			score         float64
			accuracy      float64
			missed, alarm float64
			feedback      string
			kind          string
			abstain       any
		}{
			{
				name:     "answered",
				expected: "refund",
				decision: classify.Decision{Label: "refund", Proposed: "refund", Confidence: 0.9},
				score:    1,
				accuracy: 1,
				missed:   1,
				alarm:    1,
				kind:     "correct",
			},
			{
				name:     "answered as JSON",
				expected: "refund",
				decision: map[string]any{
					"label":      "billing",
					"proposed":   "billing",
					"confidence": 0.8,
					"route":      "classify",
				},
				alarm:    1,
				feedback: `expected "refund", got "billing"`,
				kind:     "incorrect",
			},
			{
				name:     "outputs not scored",
				expected: "refund",
				decision: map[string]any{
					"label":      "refund",
					"proposed":   "refund",
					"confidence": 0.9,
					"outputs":    map[string]any{"urgency": "low"},
				},
				score:    1,
				accuracy: 1,
				missed:   1,
				alarm:    1,
				kind:     "correct",
			},
			{
				name:     "false alarm",
				expected: "shipping",
				decision: classify.Decision{Label: "refund", Proposed: "refund", Confidence: 0.7},
				missed:   1,
				feedback: `expected "shipping", got "refund"`,
				kind:     "incorrect",
			},
			{
				name:     "low confidence",
				expected: "refund",
				decision: classify.Decision{Proposed: "billing", Confidence: 0.4, Abstain: classify.LowConfidence},
				score:    0.5,
				accuracy: 0.5,
				missed:   1,
				alarm:    1,
				feedback: `abstained with low confidence 0.4 in the proposed label "billing"`,
				kind:     "abstained",
				abstain:  "low_confidence",
			},
			{
				name:     "low confidence as JSON",
				expected: "billing",
				decision: map[string]any{"proposed": "refund", "confidence": 0.3, "abstain": "low_confidence"},
				score:    0.5,
				accuracy: 0.5,
				missed:   1,
				alarm:    1,
				feedback: `abstained with low confidence 0.3 in the proposed label "refund"`,
				kind:     "abstained",
				abstain:  "low_confidence",
			},
			{
				name:     "refused critical",
				expected: "refund",
				decision: classify.Decision{Abstain: classify.Refused},
				alarm:    1,
				feedback: "abstained: refused",
				kind:     "abstained",
				abstain:  "refused",
			},
			{
				name:     "disputed as JSON",
				expected: "billing",
				decision: map[string]any{
					"proposed": "refund", "confidence": .9, "abstain": "disputed",
					"outputs": map[string]any{"urgency": "high"},
				},
				missed:   1,
				alarm:    1,
				feedback: "abstained: disputed",
				kind:     "abstained",
				abstain:  "disputed",
			},
			{
				name:     "disputed critical",
				expected: "refund",
				decision: classify.Decision{Proposed: "refund", Confidence: .9, Abstain: classify.Disputed},
				alarm:    1,
				feedback: "abstained: disputed",
				kind:     "abstained",
				abstain:  "disputed",
			},
			{
				name:     "invalid output",
				expected: "billing",
				decision: classify.Decision{Abstain: classify.InvalidOutput},
				missed:   1,
				alarm:    1,
				feedback: "abstained: invalid_output",
				kind:     "abstained",
				abstain:  "invalid_output",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result, err := supportMetric().Score(
					context.Background(),
					gepa.Prediction{"label": tt.expected},
					gepa.Prediction{"decision": tt.decision},
				)
				require.NoError(t, err)
				require.Equal(t, tt.score, result.Score)
				require.Equal(t, map[string]float64{
					"accuracy":           tt.accuracy,
					"missed:refund":      tt.missed,
					"false_alarm:refund": tt.alarm,
				}, result.ObjectiveScores)
				require.Equal(t, tt.feedback, result.Feedback)
				require.Equal(t, tt.kind, result.Details["kind"])
				require.Equal(t, tt.abstain, result.Details["abstain"])
				require.Equal(t, tt.expected, result.Details["expected"])
			})
		}
	})

	t.Run("errors", func(t *testing.T) {
		refused := gepa.Prediction{"decision": classify.Decision{Abstain: classify.Refused}}
		tests := []struct {
			name     string
			expected gepa.Prediction
			actual   gepa.Prediction
			err      string
		}{
			{
				name:     "no expected label",
				expected: gepa.Prediction{},
				actual:   refused,
				err:      `expected prediction has no "label"`,
			},
			{
				name:     "null expected label",
				expected: gepa.Prediction{"label": nil},
				actual:   refused,
				err:      `expected prediction has no "label"`,
			},
			{
				name:     "misnamed expected label",
				expected: gepa.Prediction{"Label": "refund"},
				actual:   refused,
				err:      `expected prediction has no "label"`,
			},
			{
				name:     "no decision",
				expected: gepa.Prediction{"label": "refund"},
				actual:   gepa.Prediction{"label": "refund"},
				err:      `state key "decision" is missing`,
			},
			{
				name:     "unknown decision field",
				expected: gepa.Prediction{"label": "refund"},
				actual:   gepa.Prediction{"decision": map[string]any{"label": "refund", "confidence": 0.9, "score": 1}},
				err:      `state key "decision": decode as classify.Decision`,
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				_, err := supportMetric().Score(context.Background(), tt.expected, tt.actual)
				require.ErrorContains(t, err, "classify metric: "+tt.err)
			})
		}
	})
}
