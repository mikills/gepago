package classify

import (
	"context"
	"errors"
	"fmt"

	gepa "github.com/mikills/gepago"
)

// Metric scores a classifier's output for gepa.Compile. The expected prediction must hold the
// label under "label". The actual one holds the Decision under "decision", as a Decision or its JSON
// object, which is converted strictly, as gepa.Key converts a value.
//
// An answered decision is scored by gepa.LabelMetric over Labels, with each critical label's
// misses and false alarms as objectives. A LowConfidence abstention scores AbstainScore, and
// its "accuracy" objective is AbstainScore too: it neither misses nor falsely alarms a critical
// label. Other abstentions, including Disputed, score 0 and miss the expected label when it
// is critical. A disputed proposal earns no credit even when its first label was correct.
type Metric struct {
	Labels []Label
	// AbstainScore is the score of a LowConfidence abstention, usually between 0 for a wrong
	// label and 1 for a right one.
	AbstainScore float64
}

// Score compares the decision with the expected label.
func (m Metric) Score(ctx context.Context, expected, actual gepa.Prediction) (gepa.MetricResult, error) {
	if expected[labelKey.Name] == nil {
		return gepa.MetricResult{}, errors.New(`classify metric: expected prediction has no "label"`)
	}
	decision, err := decisionKey.Get(gepa.NewState(actual))
	if err != nil {
		return gepa.MetricResult{}, fmt.Errorf("classify metric: %w", err)
	}
	var metric gepa.LabelMetric
	for _, label := range m.Labels {
		metric.Labels = append(metric.Labels, label.Name)
		if label.Critical {
			metric.Critical = append(metric.Critical, label.Name)
		}
	}
	var label any
	if !decision.Abstained() {
		label = decision.Label
	}
	result, err := metric.Score(ctx, expected, gepa.Prediction{labelKey.Name: label})
	if err != nil || !decision.Abstained() {
		return result, err
	}
	result.Details["kind"] = "abstained"
	result.Details["abstain"] = string(decision.Abstain)
	if decision.Abstain != LowConfidence {
		result.Feedback = "abstained: " + string(decision.Abstain)
		return result, nil
	}
	result.Score = m.AbstainScore
	for name := range result.ObjectiveScores {
		result.ObjectiveScores[name] = 1
	}
	result.ObjectiveScores["accuracy"] = m.AbstainScore
	result.Feedback = fmt.Sprintf(
		"abstained with low confidence %v in the proposed label %q",
		decision.Confidence,
		decision.Proposed,
	)
	return result, nil
}
