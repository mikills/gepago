package classify

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"unicode"

	gepa "github.com/mikills/gepago"
)

// Label is one class a classifier can choose.
type Label struct {
	// Name is the value the model returns for the label.
	Name string
	// Description defines the label. It seeds the "label.<Name>" component that GEPA optimises.
	Description string
	// Critical scores the label's misses and false alarms as objectives of their own.
	Critical bool
}

// Spec describes a classifier.
type Spec struct {
	Name string
	// Instruction seeds the "instruction" component that GEPA optimises.
	Instruction string
	Labels      []Label
	// Outputs are fields the model returns in the same reply as the label, its confidence and
	// reason. The reply's JSON schema holds them, and each is checked against its Type or Schema
	// as gepa.Predict checks an output, so MaxParseRetries repairs a value that breaks it. Strict
	// providers enforce them when the reply schema is strict, that is when each output stays
	// within the subset gepa.StrictJSONSchema describes. An output with no Type or Schema, or
	// with Type "map" or "slice", makes the whole reply non-strict, the label included. A
	// decision holds them in Decision.Outputs. Metric and Train score only the label, not the
	// outputs. An output cannot take a name the classifier uses: label, confidence, reason,
	// text, hint, route, decision, vote, votes, sample or samples; with Confirm, previous too.
	Outputs []gepa.Field
	// Demos seed the "demos" component. Each demo's Inputs hold the "text", and its Outputs the
	// "label", with the "confidence", "reason" and Outputs when you have them;
	// gepa.DemosFromExamples turns the examples Example builds into demos. Under Votes the
	// samples' demos leave out "confidence".
	Demos   []gepa.Demo
	Primary gepa.Generator
	// Backup is an optional second model that answers when Primary fails with an error that is
	// not Permanent, as gepa.Fallback falls back.
	Backup gepa.Generator
	// MinConfidence makes a decision whose confidence is below it abstain with LowConfidence.
	// Zero disables LowConfidence. Under SelfReported the classifier first asks again once, with a hint
	// that names the earlier answer and its confidence, and keeps the new answer only when its
	// confidence is higher. When that second try fails, on the Backup too when there is one, the
	// decision keeps the first answer and abstains with LowConfidence, unless the error fails
	// the run. When Confirm also applies, its agreement rules take precedence.
	// ChooseThreshold picks a MinConfidence for a target accuracy.
	MinConfidence float64
	// Confirm names labels that get a second look even above MinConfidence. The hint names the
	// first label, confidence and reason. Agreement keeps the second answer with the higher
	// confidence; disagreement abstains with Disputed and retains the first answer's fields.
	// A failed second look restores the first answer unless the error fails the run.
	// MinConfidence still applies to the kept answer. There is at most one second look, even
	// when both triggers apply. Confirm is not supported with Votes.
	Confirm []string
	// Confidence chooses how a decision's confidence is measured; nil means SelfReported.
	Confidence Confidence
	// MaxParseRetries is how often each model's reply is repaired (see gepa.Predict).
	MaxParseRetries int
	// Policy runs each step that asks a model under gepa.WithPolicy: classify_backup, and
	// classify under SelfReported or each vote's ask step under Votes. A vote whose last attempt
	// fails is a failed vote.
	Policy gepa.StepPolicy
	// Budget, MetadataOnly, Redactor, Observers and Finalizer are passed to the classifier's
	// gepa.PipelineSpec.
	Budget       gepa.RunBudget
	MetadataOnly bool
	Redactor     gepa.Redactor
	Observers    []gepa.PipelineObserver
	Finalizer    *gepa.PipelineFinalizer
}

// Validate checks that the spec describes a usable classifier.
func (s Spec) Validate() error {
	if strings.TrimSpace(s.Name) == "" {
		return errors.New("classify spec Name is required")
	}
	if strings.ContainsFunc(s.Name, unicode.IsSpace) {
		return fmt.Errorf("classify spec Name %q cannot contain whitespace", s.Name)
	}
	if strings.TrimSpace(s.Instruction) == "" {
		return errors.New("classify spec Instruction is required")
	}
	if len(s.Labels) < 2 {
		return fmt.Errorf("classify spec Labels needs at least two labels, got %d", len(s.Labels))
	}
	seen := map[string]bool{}
	for i, label := range s.Labels {
		switch {
		case label.Name == "":
			return fmt.Errorf("classify spec Labels[%d] Name is required", i)
		case strings.ContainsFunc(label.Name, unicode.IsSpace):
			return fmt.Errorf("classify spec label %q cannot contain whitespace", label.Name)
		case seen[label.Name]:
			return fmt.Errorf("classify spec label %q is repeated", label.Name)
		case strings.TrimSpace(label.Description) == "":
			return fmt.Errorf("classify spec label %q Description is required", label.Name)
		}
		seen[label.Name] = true
	}
	confirm := map[string]bool{}
	for _, name := range s.Confirm {
		switch {
		case !seen[name]:
			return fmt.Errorf("classify spec Confirm label %q is not in Labels", name)
		case confirm[name]:
			return fmt.Errorf("classify spec Confirm label %q is repeated", name)
		}
		confirm[name] = true
	}
	outputs := map[string]bool{}
	for _, output := range s.Outputs {
		switch {
		case slices.Contains(reservedNames, output.Name) || len(s.Confirm) > 0 && output.Name == previousKey.Name:
			return fmt.Errorf("classify spec output %q is reserved: the classifier uses that name", output.Name)
		case outputs[output.Name]:
			return fmt.Errorf("classify spec output %q is repeated", output.Name)
		}
		outputs[output.Name] = true
	}
	if len(s.Outputs) > 0 {
		signature := gepa.Signature{Inputs: []gepa.Field{{Name: textKey.Name}}, Outputs: s.Outputs}
		if err := signature.Validate(); err != nil {
			return fmt.Errorf("classify spec Outputs: %w", err)
		}
	}
	if s.Primary == nil {
		return errors.New("classify spec Primary is required")
	}
	if math.IsNaN(s.MinConfidence) || s.MinConfidence < 0 || s.MinConfidence > 1 {
		return fmt.Errorf("classify spec MinConfidence must be between 0 and 1, got %v", s.MinConfidence)
	}
	switch c := s.Confidence.(type) {
	case nil, SelfReported:
	case Votes:
		if len(s.Confirm) > 0 {
			return errors.New("classify spec Confirm is not supported with Votes")
		}
		if c.N < 2 {
			return fmt.Errorf("classify spec Votes.N must be at least 2, got %d", c.N)
		}
		if c.StopAt != 0 && (c.StopAt <= c.N/2 || c.StopAt > c.N) {
			return fmt.Errorf(
				"classify spec Votes.StopAt must be 0 or a majority of N (%d), from %d to %d, got %d",
				c.N, c.N/2+1, c.N, c.StopAt,
			)
		}
	default:
		return fmt.Errorf("classify spec Confidence %T is not supported; use SelfReported or Votes", c)
	}
	return nil
}

// Confidence chooses how a classifier measures its confidence in a label: SelfReported or
// Votes.
type Confidence interface{ confidence() }

// SelfReported has the model state its confidence, in [0, 1], with its label.
type SelfReported struct{}

// Votes asks Primary for up to N samples. The confidence is the share of the answering samples
// that agree with the majority label; fewer than N answer when StopAt stops early. A tie goes
// to the label voted first. The samples state no confidence of their own. When no sample
// answers, Backup answers once and states its confidence; without a Backup, the decision
// abstains for the last sample's failure. Run Primary at a temperature above 0, or the samples
// agree whatever the text.
type Votes struct {
	// N is the number of samples, at least 2.
	N int
	// StopAt stops sampling once this many samples agree, a strict majority of N; 0 runs all N
	// concurrently.
	StopAt int
}

func (SelfReported) confidence() {}

func (Votes) confidence() {}
