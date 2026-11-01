package classify

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	gepa "github.com/mikills/gepago"
)

// program asks a model for one of the labels. Its candidate holds the "instruction", a
// "label.<name>" definition of each label and, when the spec has demos, the "demos". It gives
// its Predict the instruction followed by the labels and their definitions. It adds its step's
// name to the reply under "route", so the decision names the step that answered.
type program struct {
	predict gepa.Predict
	labels  []Label
	confirm []string
	route   string
	seed    gepa.Candidate
}

// newProgram returns the programme of the step that route names, which asks lm with the
// signature inputs and outputs. Its seed is the spec's text, replaced by what candidate holds
// under route. Its demos leave out "confidence" when its outputs do.
func newProgram(
	spec Spec,
	candidate gepa.Candidate,
	lm gepa.Generator,
	route string,
	inputs, outputs []gepa.Field,
) program {
	predict := gepa.Predict{
		Signature:       gepa.Signature{Name: spec.Name, Inputs: inputs, Outputs: outputs},
		LM:              lm,
		Instruction:     spec.Instruction,
		Demos:           spec.Demos,
		MaxParseRetries: spec.MaxParseRetries,
	}
	if !slices.ContainsFunc(outputs, func(field gepa.Field) bool { return field.Name == confidenceKey.Name }) {
		predict.Demos = nil
		for _, demo := range spec.Demos {
			demo.Outputs = maps.Clone(demo.Outputs)
			delete(demo.Outputs, confidenceKey.Name)
			predict.Demos = append(predict.Demos, demo)
		}
	}
	seed := predict.SeedCandidate()
	for _, label := range spec.Labels {
		seed[labelComponent(label.Name)] = label.Description
	}
	for key, text := range candidate {
		if component, ok := strings.CutPrefix(key, route+"."); ok {
			seed[component] = text
		}
	}
	return program{predict: predict, labels: spec.Labels, confirm: spec.Confirm, route: route, seed: seed}
}

// labelComponent returns the candidate key of a label's definition.
func labelComponent(name string) string {
	return "label." + name
}

func (p program) Validate() error {
	return p.predict.Validate()
}

func (p program) SeedCandidate() gepa.Candidate {
	return gepa.CloneCandidate(p.seed)
}

// RunCandidate fails with gepa.ErrInvalidOutput when the reply's confidence is outside [0, 1].
func (p program) RunCandidate(ctx context.Context, candidate gepa.Candidate, inputs gepa.Prediction) (
	gepa.Prediction,
	error,
) {
	var instruction strings.Builder
	instruction.WriteString(strings.TrimSpace(candidate[gepa.InstructionComponent]))
	instruction.WriteString("\n\nLabels:")
	for _, label := range p.labels {
		fmt.Fprintf(&instruction, "\n- %s: %s", label.Name, strings.TrimSpace(candidate[labelComponent(label.Name)]))
	}
	outputs, err := p.predict.RunCandidate(ctx, gepa.Candidate{
		gepa.InstructionComponent: instruction.String(),
		gepa.DemosComponent:       candidate[gepa.DemosComponent],
	}, inputs)
	if err != nil {
		return outputs, err
	}
	if _, ok := outputs[confidenceKey.Name]; ok {
		// Predict has checked that the confidence is a float64.
		confidence, _ := confidenceKey.Get(gepa.NewState(outputs))
		if confidence < 0 || confidence > 1 {
			err := fmt.Errorf(
				"%w: signature %q: output %q is %v, not between 0 and 1",
				gepa.ErrInvalidOutput,
				p.predict.Signature.Name,
				confidenceKey.Name,
				confidence,
			)
			return outputs, gepa.WithStepDetails(err, map[string]any{"output": maps.Clone(outputs)})
		}
	}
	outputs[routeKey.Name] = p.route
	return outputs, nil
}

// ProgramSignature is the Predict's signature with the "route" output the programme adds.
func (p program) ProgramSignature() (gepa.ProgramSignature, error) {
	signature, err := p.predict.ProgramSignature()
	signature.Signature.Outputs = append(
		slices.Clone(signature.Signature.Outputs),
		gepa.Field{Name: routeKey.Name, Type: "string"},
	)
	return signature, err
}

// Fingerprint covers the Predict's signature and schemas, and the labels that need confirmation.
func (p program) Fingerprint() string {
	predict := p.predict.Fingerprint()
	if len(p.confirm) == 0 {
		return predict
	}
	data, _ := json.Marshal(struct {
		Version int      `json:"version"`
		Predict string   `json:"predict"`
		Confirm []string `json:"confirm"`
	}{Version: 1, Predict: predict, Confirm: p.confirm})
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// vote is one sample of a Votes classifier: the label it answered, why, and the values of the
// Spec's Outputs it gave, or the reason the decision would abstain for its failure and the
// failure's message.
type vote struct {
	Label   string         `json:"label,omitempty"`
	Reason  string         `json:"reason,omitempty"`
	Outputs map[string]any `json:"outputs,omitempty"`
	Failed  AbstainReason  `json:"failed,omitempty"`
	Error   string         `json:"error,omitempty"`
}

// voter runs one sample of a Votes classifier, the pipeline whose ask step asks Primary under
// the spec's Policy, and returns its vote under "vote", with the values the sample wrote under
// outputs. A failure that would make a decision abstain, after the policy's last attempt, is a
// failed vote, so the other samples still count; any other failure, or one after the context
// ended, fails the sample. With carry it reads the votes so far and returns them with its own
// under "votes", as a Loop's body.
type voter struct {
	sample  gepa.Pipeline
	outputs []gepa.Key[any]
	carry   bool
}

func (v voter) Validate() error {
	return v.sample.Validate()
}

func (v voter) SeedCandidate() gepa.Candidate {
	return v.sample.SeedCandidate()
}

func (v voter) RunCandidate(ctx context.Context, candidate gepa.Candidate, inputs gepa.Prediction) (
	gepa.Prediction,
	error,
) {
	var cast vote
	answer, err := v.sample.RunCandidate(ctx, candidate, gepa.Prediction{textKey.Name: inputs[textKey.Name]})
	if err != nil {
		reason, ok := abstention(err)
		if !ok || ctx.Err() != nil {
			return nil, err
		}
		cast = vote{Failed: reason, Error: err.Error()}
	} else {
		state := gepa.NewState(answer)
		var labelErr, reasonErr, outputsErr error
		cast.Label, labelErr = labelKey.Get(state)
		cast.Reason, reasonErr = reasonKey.Get(state)
		cast.Outputs, outputsErr = outputValues(state, v.outputs)
		if err := errors.Join(labelErr, reasonErr, outputsErr); err != nil {
			return nil, err
		}
	}
	if !v.carry {
		return gepa.Prediction{voteKey.Name: cast}, nil
	}
	votes, err := votesKey.Get(gepa.NewState(inputs))
	if err != nil {
		return nil, err
	}
	return gepa.Prediction{votesKey.Name: append(votes, cast)}, nil
}

// Fingerprint is the sample pipeline's, which covers the label names.
func (v voter) Fingerprint() string {
	return v.sample.Fingerprint()
}

// majority returns the first vote for the label that most answered votes agree on, how many
// agree, and how many votes answered. A tie goes to the label voted first.
func majority(votes []vote) (winner vote, agree, answered int) {
	counts := map[string]int{}
	for _, v := range votes {
		if v.Failed == "" {
			counts[v.Label]++
			answered++
		}
	}
	for _, v := range votes {
		if v.Failed == "" && counts[v.Label] > agree {
			winner, agree = v, counts[v.Label]
		}
	}
	return winner, agree, answered
}

// unanswered is the error of a tally whose samples all failed. It matches what made the last
// sample fail, gepa.ErrContentFiltered or gepa.ErrInvalidOutput, so the decision abstains for
// the same reason.
type unanswered struct{ last vote }

func (e unanswered) Error() string {
	return "no sample answered; the last failed: " + e.last.Error
}

func (e unanswered) Unwrap() error {
	switch e.last.Failed {
	case Refused:
		return gepa.ErrContentFiltered
	case InvalidOutput:
		return gepa.ErrInvalidOutput
	}
	return nil
}
