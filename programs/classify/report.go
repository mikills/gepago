package classify

import (
	"cmp"
	"fmt"
	"maps"
	"math"
	"slices"
)

// Outcome is a classifier's Decision for one example, with the example's expected label.
type Outcome struct {
	ID       string   `json:"id"`
	Expected string   `json:"expected"`
	Decision Decision `json:"decision"`
}

// Report summarises a classifier's outcomes. A rate whose denominator is zero is 0, and the
// counts beside it show why.
type Report struct {
	Total    int `json:"total"`
	Answered int `json:"answered"`
	// Correct counts the answered outcomes whose label is the expected one.
	Correct int `json:"correct"`
	// Accuracy is Correct over Answered.
	Accuracy float64 `json:"accuracy"`
	// Coverage is Answered over Total.
	Coverage float64 `json:"coverage"`
	// Labels holds the figures of every label that an outcome expected or answered.
	Labels map[string]LabelReport `json:"labels"`
	// MacroF1 is the mean F1 of Labels.
	MacroF1 float64 `json:"macro_f1"`
	// Confusion counts the answered outcomes by expected label, then by answered label.
	Confusion   map[string]map[string]int `json:"confusion"`
	Abstentions map[AbstainReason]int     `json:"abstentions"`
	// Calibration sorts the outcomes with a Proposed label, abstentions included, into ten bins
	// of equal width by Confidence. Disputed contributes its first label and confidence.
	Calibration []CalibrationBin `json:"calibration"`
	// ExpectedCalibrationError is the gap between each bin's MeanConfidence and Accuracy,
	// averaged over the bins weighted by their counts.
	ExpectedCalibrationError float64 `json:"expected_calibration_error"`
	// Outcomes are the outcomes summarised, in their given order, ready for ChooseThreshold.
	Outcomes []Outcome `json:"outcomes"`
}

// LabelReport is one label's figures. Support counts every outcome that expected the label, so
// an abstention lowers the label's Recall but no label's Precision.
type LabelReport struct {
	Support int `json:"support"`
	// Answered counts the outcomes answered with the label.
	Answered  int     `json:"answered"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
}

// CalibrationBin holds the outcomes whose confidence is in [Lower, Upper), or [Lower, Upper]
// for the last bin.
type CalibrationBin struct {
	Lower          float64 `json:"lower"`
	Upper          float64 `json:"upper"`
	Count          int     `json:"count"`
	MeanConfidence float64 `json:"mean_confidence"`
	// Accuracy is the share of the bin's outcomes whose Proposed label is the expected one.
	Accuracy float64 `json:"accuracy"`
}

// NewReport summarises outcomes.
func NewReport(outcomes []Outcome) Report {
	report := Report{
		Total:       len(outcomes),
		Labels:      map[string]LabelReport{},
		Confusion:   map[string]map[string]int{},
		Abstentions: map[AbstainReason]int{},
		Calibration: make([]CalibrationBin, 10),
		Outcomes:    append([]Outcome{}, outcomes...),
	}
	proposed := 0
	for _, outcome := range outcomes {
		decision := outcome.Decision
		expected := report.Labels[outcome.Expected]
		expected.Support++
		report.Labels[outcome.Expected] = expected
		if decision.Proposed != "" {
			proposed++
			i := min(max(int(decision.Confidence*10), 0), 9)
			if i > 0 && decision.Confidence < float64(i)/10 {
				i--
			}
			bin := &report.Calibration[i]
			bin.Count++
			bin.MeanConfidence += decision.Confidence
			if decision.Proposed == outcome.Expected {
				bin.Accuracy++
			}
		}
		if decision.Abstained() {
			report.Abstentions[decision.Abstain]++
			continue
		}
		report.Answered++
		answered := report.Labels[decision.Label]
		answered.Answered++
		report.Labels[decision.Label] = answered
		if report.Confusion[outcome.Expected] == nil {
			report.Confusion[outcome.Expected] = map[string]int{}
		}
		report.Confusion[outcome.Expected][decision.Label]++
		if decision.Label == outcome.Expected {
			report.Correct++
		}
	}
	report.Accuracy = rate(report.Correct, report.Answered)
	report.Coverage = rate(report.Answered, report.Total)
	for _, name := range slices.Sorted(maps.Keys(report.Labels)) {
		label := report.Labels[name]
		correct := report.Confusion[name][name]
		label.Precision = rate(correct, label.Answered)
		label.Recall = rate(correct, label.Support)
		label.F1 = rate(2*correct, label.Support+label.Answered)
		report.Labels[name] = label
		report.MacroF1 += label.F1 / float64(len(report.Labels))
	}
	for i := range report.Calibration {
		bin := &report.Calibration[i]
		bin.Lower, bin.Upper = float64(i)/10, float64(i+1)/10
		if bin.Count > 0 {
			bin.MeanConfidence /= float64(bin.Count)
			bin.Accuracy /= float64(bin.Count)
			gap := math.Abs(bin.MeanConfidence - bin.Accuracy)
			report.ExpectedCalibrationError += gap * float64(bin.Count) / float64(proposed)
		}
	}
	return report
}

// ChooseThreshold returns the lowest confidence threshold at which eligible outcomes with a
// Confidence at or above it are at least targetAccuracy correct, and the coverage it gives:
// the share of all outcomes it keeps. An outcome is eligible when it has a Proposed label and
// either answered or abstained with LowConfidence. Other abstentions, including Disputed, stay
// in the coverage denominator but cannot be kept. Only eligible confidences are tried, and
// the lowest that reaches the target keeps the most outcomes; outcomes with equal confidence
// are kept or dropped together. It returns an error when no threshold reaches targetAccuracy.
// Classifier.WithMinConfidence applies the threshold.
func ChooseThreshold(outcomes []Outcome, targetAccuracy float64) (threshold, coverage float64, err error) {
	if math.IsNaN(targetAccuracy) || targetAccuracy < 0 || targetAccuracy > 1 {
		return 0, 0, fmt.Errorf("classify threshold: target accuracy must be between 0 and 1, got %v", targetAccuracy)
	}
	var proposed []Outcome
	correct := 0
	for _, outcome := range outcomes {
		decision := outcome.Decision
		if decision.Proposed != "" && (!decision.Abstained() || decision.Abstain == LowConfidence) {
			proposed = append(proposed, outcome)
			if outcome.Decision.Proposed == outcome.Expected {
				correct++
			}
		}
	}
	slices.SortFunc(proposed, func(a, b Outcome) int {
		return cmp.Compare(a.Decision.Confidence, b.Decision.Confidence)
	})
	best := 0.0
	for i, outcome := range proposed {
		if i == 0 || proposed[i-1].Decision.Confidence != outcome.Decision.Confidence {
			kept := len(proposed) - i
			accuracy := float64(correct) / float64(kept)
			if accuracy >= targetAccuracy {
				return outcome.Decision.Confidence, rate(kept, len(outcomes)), nil
			}
			best = max(best, accuracy)
		}
		if outcome.Decision.Proposed == outcome.Expected {
			correct--
		}
	}
	return 0, 0, fmt.Errorf(
		"classify threshold: no threshold reaches accuracy %v; the best is %v over %d outcomes with a proposed label",
		targetAccuracy,
		best,
		len(proposed),
	)
}

func rate(count, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(count) / float64(total)
}
