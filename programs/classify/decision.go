package classify

import (
	"fmt"

	gepa "github.com/mikills/gepago"
)

// AbstainReason says why a Decision holds no label.
type AbstainReason string

// The reasons a Decision abstains.
const (
	// LowConfidence means the confidence was below Spec.MinConfidence.
	LowConfidence AbstainReason = "low_confidence"
	// Disputed means a second look for Spec.Confirm chose a different label. The decision
	// keeps the first answer's Proposed, Confidence, Reason, Route and Outputs.
	Disputed AbstainReason = "disputed"
	// InvalidOutput means every model's reply broke the output contract.
	InvalidOutput AbstainReason = "invalid_output"
	// Refused means a content policy withheld or refused the reply (see gepa.ErrContentFiltered).
	Refused AbstainReason = "refused"
	// ModelFailure means the last model failed with any other Transient or unclassified error.
	ModelFailure AbstainReason = "model_failure"
)

// Decision is a classifier's answer for one text: a label, or an abstention with its reason.
type Decision struct {
	// Label is the chosen label; empty when the decision abstains.
	Label string `json:"label,omitempty"`
	// Proposed is the label a model answered, if any, even when the decision abstains.
	Proposed   string  `json:"proposed,omitempty"`
	Confidence float64 `json:"confidence"`
	// Reason is the model's short justification.
	Reason  string        `json:"reason,omitempty"`
	Abstain AbstainReason `json:"abstain,omitempty"`
	// Route names the step that produced Proposed.
	Route string `json:"route,omitempty"`
	// Outputs holds the values of the Spec's Outputs in the answer Proposed comes from: the
	// answer kept after a second look, the Backup's when it answered, or under Votes the first
	// sample that voted for Proposed. Disputed keeps the first answer's outputs. It leaves out
	// an Optional output the reply left out or set to null, and is empty when no model answered.
	// Read a value with Output.
	Outputs map[string]any `json:"outputs,omitempty"`
}

// Abstained reports whether d holds no label.
func (d Decision) Abstained() bool { return d.Abstain != "" }

// Output returns the output name of d as T. It converts the value as gepa.Key converts a state
// value, strictly and through T's Validate method if T has one. It fails when d has no such
// output.
func Output[T any](d Decision, name string) (T, error) {
	value, ok := d.Outputs[name]
	if !ok {
		var zero T
		return zero, fmt.Errorf("classify decision has no output %q", name)
	}
	out, err := gepa.Key[T]{Name: name}.Get(gepa.NewState(gepa.Prediction{name: value}))
	if err != nil {
		return out, fmt.Errorf("classify decision output: %w", err)
	}
	return out, nil
}
