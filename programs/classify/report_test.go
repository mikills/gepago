package classify_test

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/mikills/gepago/programs/classify"
	"github.com/stretchr/testify/require"
)

func answered(label string, confidence float64) classify.Decision {
	return classify.Decision{Label: label, Proposed: label, Confidence: confidence}
}

func lowConfidence(proposed string, confidence float64) classify.Decision {
	return classify.Decision{Proposed: proposed, Confidence: confidence, Abstain: classify.LowConfidence}
}

func emptyBins() []classify.CalibrationBin {
	bins := make([]classify.CalibrationBin, 10)
	for i := range bins {
		bins[i].Lower, bins[i].Upper = float64(i)/10, float64(i+1)/10
	}
	return bins
}

func TestReport(t *testing.T) {
	t.Run("disputed", func(t *testing.T) {
		outcomes := []classify.Outcome{
			{Expected: "refund", Decision: answered("refund", .8)},
			{Expected: "refund", Decision: classify.Decision{
				Proposed: "refund", Confidence: .9, Abstain: classify.Disputed,
			}},
			{Expected: "refund", Decision: classify.Decision{
				Proposed: "billing", Confidence: .9, Abstain: classify.Disputed,
			}},
		}
		report := classify.NewReport(outcomes)
		require.Equal(t, 3, report.Total)
		require.Equal(t, 1, report.Answered)
		require.Equal(t, 1, report.Correct)
		require.Equal(t, 1.0, report.Accuracy)
		require.Equal(t, 1.0/3, report.Coverage)
		require.Equal(t, map[classify.AbstainReason]int{classify.Disputed: 2}, report.Abstentions)
		require.Equal(t, map[string]classify.LabelReport{
			"refund": {Support: 3, Answered: 1, Precision: 1, Recall: 1.0 / 3, F1: .5},
		}, report.Labels)
		require.Equal(t, .5, report.MacroF1)
		require.Equal(t, map[string]map[string]int{"refund": {"refund": 1}}, report.Confusion)
		require.Equal(t, classify.CalibrationBin{
			Lower: .8, Upper: .9, Count: 1, MeanConfidence: .8, Accuracy: 1,
		}, report.Calibration[8])
		require.Equal(t, classify.CalibrationBin{
			Lower: .9, Upper: 1, Count: 2, MeanConfidence: .9, Accuracy: .5,
		}, report.Calibration[9])
		require.InDelta(t, 1.0/3, report.ExpectedCalibrationError, 1e-12)
		require.Equal(t, outcomes, report.Outcomes)

		report = classify.NewReport(outcomes[1:])
		require.Zero(t, report.Answered)
		require.Zero(t, report.Coverage)
		require.Zero(t, report.Accuracy)
		require.Empty(t, report.Confusion)
		require.InDelta(t, .4, report.ExpectedCalibrationError, 1e-12)
	})

	t.Run("metrics", func(t *testing.T) {
		outcomes := []classify.Outcome{
			{ID: "a", Expected: "billing", Decision: answered("billing", 0.95)},
			{ID: "b", Expected: "billing", Decision: answered("refund", 0.85)},
			{ID: "c", Expected: "refund", Decision: answered("refund", 0.9)},
			{ID: "d", Expected: "refund", Decision: lowConfidence("billing", 0.3)},
			{ID: "e", Expected: "shipping", Decision: classify.Decision{Abstain: classify.Refused}},
			{ID: "f", Expected: "shipping", Decision: answered("shipping", 1)},
		}
		report := classify.NewReport(outcomes)

		require.Equal(t, 6, report.Total)
		require.Equal(t, 4, report.Answered)
		require.Equal(t, 3, report.Correct)
		require.Equal(t, 0.75, report.Accuracy)
		require.Equal(t, 4.0/6, report.Coverage)
		require.Equal(t, map[string]classify.LabelReport{
			"billing":  {Support: 2, Answered: 1, Precision: 1, Recall: 0.5, F1: 2.0 / 3},
			"refund":   {Support: 2, Answered: 2, Precision: 0.5, Recall: 0.5, F1: 0.5},
			"shipping": {Support: 2, Answered: 1, Precision: 1, Recall: 0.5, F1: 2.0 / 3},
		}, report.Labels)
		require.InDelta(t, 11.0/18, report.MacroF1, 1e-12)
		require.Equal(t, map[string]map[string]int{
			"billing":  {"billing": 1, "refund": 1},
			"refund":   {"refund": 1},
			"shipping": {"shipping": 1},
		}, report.Confusion)
		require.Equal(
			t,
			map[classify.AbstainReason]int{classify.LowConfidence: 1, classify.Refused: 1},
			report.Abstentions,
		)

		want := emptyBins()
		want[3].Count, want[3].MeanConfidence = 1, 0.3
		want[8].Count, want[8].MeanConfidence = 1, 0.85
		want[9].Count, want[9].MeanConfidence, want[9].Accuracy = 3, 0.95, 1
		require.Len(t, report.Calibration, 10)
		for i, bin := range report.Calibration {
			require.Equal(t, want[i].Lower, bin.Lower, "bin %d", i)
			require.Equal(t, want[i].Upper, bin.Upper, "bin %d", i)
			require.Equal(t, want[i].Count, bin.Count, "bin %d", i)
			require.InDelta(t, want[i].MeanConfidence, bin.MeanConfidence, 1e-12, "bin %d", i)
			require.Equal(t, want[i].Accuracy, bin.Accuracy, "bin %d", i)
		}
		require.InDelta(t, (0.3+0.85+3*0.05)/5, report.ExpectedCalibrationError, 1e-12)
		require.Equal(t, outcomes, report.Outcomes)
	})

	t.Run("zero denominators", func(t *testing.T) {
		empty := classify.NewReport(nil)
		require.Equal(t, classify.Report{
			Labels:      map[string]classify.LabelReport{},
			Confusion:   map[string]map[string]int{},
			Abstentions: map[classify.AbstainReason]int{},
			Calibration: emptyBins(),
			Outcomes:    []classify.Outcome{},
		}, empty)
		data, err := json.Marshal(empty)
		require.NoError(t, err)
		require.Contains(t, string(data), `"labels":{},"macro_f1":0,"confusion":{},"abstentions":{}`)
		require.Contains(t, string(data), `"outcomes":[]`)

		abstained := classify.NewReport([]classify.Outcome{
			{ID: "a", Expected: "refund", Decision: classify.Decision{Abstain: classify.ModelFailure}},
			{ID: "b", Expected: "billing", Decision: answered("fraud", 0.6)},
		})
		require.Equal(t, 1, abstained.Answered)
		require.Zero(t, abstained.Accuracy)
		require.Equal(t, 0.5, abstained.Coverage)
		require.Equal(t, map[string]classify.LabelReport{
			"billing": {Support: 1},
			"fraud":   {Answered: 1},
			"refund":  {Support: 1},
		}, abstained.Labels)
		require.Zero(t, abstained.MacroF1)
		require.Equal(t, 0.6, abstained.ExpectedCalibrationError)
		_, err = json.Marshal(abstained)
		require.NoError(t, err)
	})

	t.Run("calibration bounds", func(t *testing.T) {
		x := 0.3
		tests := []struct {
			confidence float64
			bin        int
		}{
			{0, 0}, {0.1, 1}, {0.2, 2}, {0.3, 3}, {0.4, 4}, {0.5, 5}, {0.6, 6}, {0.7, 7}, {0.8, 8}, {0.9, 9}, {1, 9},
			{x * 3, 8}, // 0.8999999999999999, though x*3*10 rounds to 9
		}
		for _, tt := range tests {
			report := classify.NewReport(
				[]classify.Outcome{{Expected: "billing", Decision: answered("billing", tt.confidence)}},
			)
			require.Equal(t, 1, report.Calibration[tt.bin].Count, "confidence %v", tt.confidence)
			require.Equal(t, 1.0, report.Calibration[tt.bin].Accuracy, "confidence %v", tt.confidence)
		}
	})
}

func TestChooseThreshold(t *testing.T) {
	t.Run("hard abstentions", func(t *testing.T) {
		for _, reason := range []classify.AbstainReason{
			classify.Disputed, classify.InvalidOutput, classify.Refused, classify.ModelFailure,
		} {
			t.Run(string(reason), func(t *testing.T) {
				outcomes := []classify.Outcome{
					{Expected: "refund", Decision: answered("refund", .8)},
					{Expected: "refund", Decision: classify.Decision{
						Proposed: "refund", Confidence: .9, Abstain: reason,
					}},
					{Expected: "refund", Decision: classify.Decision{
						Proposed: "billing", Confidence: .8, Abstain: reason,
					}},
				}
				threshold, coverage, err := classify.ChooseThreshold(outcomes, 1)
				require.NoError(t, err)
				require.Equal(t, .8, threshold)
				require.Equal(t, 1.0/3, coverage)
				_, _, err = classify.ChooseThreshold(outcomes[1:], 0)
				require.EqualError(t, err, "classify threshold: no threshold reaches accuracy 0; "+
					"the best is 0 over 0 outcomes with a proposed label")
			})
		}
	})

	outcomes := []classify.Outcome{
		{ID: "a", Expected: "billing", Decision: answered("billing", 0.95)},
		{ID: "b", Expected: "refund", Decision: answered("billing", 0.9)},
		{ID: "c", Expected: "refund", Decision: answered("refund", 0.9)},
		{ID: "d", Expected: "shipping", Decision: answered("shipping", 0.8)},
		{ID: "e", Expected: "billing", Decision: answered("refund", 0.7)},
		{ID: "f", Expected: "refund", Decision: lowConfidence("refund", 0.6)},
		{ID: "g", Expected: "refund", Decision: classify.Decision{Abstain: classify.Refused}},
	}
	// Accuracy at or above each confidence: 0.6 → 4/6, 0.7 → 3/5, 0.8 → 3/4, 0.9 → 2/3, 0.95 → 1.
	tests := []struct {
		name      string
		target    float64
		threshold float64
		coverage  float64
	}{
		{name: "zero target", target: 0, threshold: 0.6, coverage: 6.0 / 7},
		{name: "at the lowest", target: 2.0 / 3, threshold: 0.6, coverage: 6.0 / 7},
		{name: "skips a dip", target: 0.7, threshold: 0.8, coverage: 4.0 / 7},
		{name: "exactly reached", target: 0.75, threshold: 0.8, coverage: 4.0 / 7},
		{name: "ties kept together", target: 0.76, threshold: 0.95, coverage: 1.0 / 7},
		{name: "perfect", target: 1, threshold: 0.95, coverage: 1.0 / 7},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			threshold, coverage, err := classify.ChooseThreshold(outcomes, tt.target)
			require.NoError(t, err)
			require.Equal(t, tt.threshold, threshold)
			require.Equal(t, tt.coverage, coverage)
		})
	}

	t.Run("errors", func(t *testing.T) {
		tests := []struct {
			name     string
			outcomes []classify.Outcome
			target   float64
			err      string
		}{
			{
				name: "unreached",
				outcomes: []classify.Outcome{
					{Expected: "billing", Decision: answered("refund", 0.9)},
					{Expected: "refund", Decision: answered("refund", 0.5)},
				},
				target: 1,
				err: "classify threshold: no threshold reaches accuracy 1; " +
					"the best is 0.5 over 2 outcomes with a proposed label",
			},
			{
				name: "no proposals",
				outcomes: []classify.Outcome{
					{Expected: "refund", Decision: classify.Decision{Abstain: classify.Refused}},
				},
				target: 0,
				err: "classify threshold: no threshold reaches accuracy 0; " +
					"the best is 0 over 0 outcomes with a proposed label",
			},
			{
				name:   "negative",
				target: -0.1,
				err:    "classify threshold: target accuracy must be between 0 and 1, got -0.1",
			},
			{
				name:   "above one",
				target: 1.1,
				err:    "classify threshold: target accuracy must be between 0 and 1, got 1.1",
			},
			{
				name:   "NaN",
				target: math.NaN(),
				err:    "classify threshold: target accuracy must be between 0 and 1, got NaN",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				_, _, err := classify.ChooseThreshold(tt.outcomes, tt.target)
				require.EqualError(t, err, tt.err)
			})
		}
	})
}
