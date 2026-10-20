package gepa

import (
	"encoding/json"
	"errors"
	"fmt"
)

// failStep wraps err as the failure of node, as newStepError does, and records copies of the
// values of node's declared reads found in values.
func failStep(node *stepNode, values map[string]any, err error) *StepError {
	stepErr := newStepError(node.name, err)
	stepErr.Reads = stepReads(node, values)
	return stepErr
}

// failure returns err as the failure of the step p wraps. With a fallback step, the StepError
// the pipeline gives p holds the fallback's reads as well, so every failure of the wrapped step
// gets a StepError of its own, with only its own reads and details.
func (p *policyStep) failure(s State, err error) error {
	if p.fallback == nil {
		return err
	}
	return &namedStepError{step: failStep(p.child, s.values, err)}
}

// namedStepError holds the StepError of a step that the enclosing StepError already names, as
// the step a WithPolicy wraps, and prints only that step's error, so the message names the
// step once.
type namedStepError struct {
	step *StepError
}

func (e *namedStepError) Error() string {
	return e.step.Err.Error()
}

func (e *namedStepError) Unwrap() error {
	return e.step
}

// stepReads returns copies of the values of node's declared reads found in values, or nil when
// there are none.
func stepReads(node *stepNode, values map[string]any) map[string]any {
	var reads map[string]any
	for _, key := range node.reads {
		value, ok := values[key.Name]
		if !ok {
			continue
		}
		if reads == nil {
			reads = make(map[string]any, len(node.reads))
		}
		reads[key.Name] = deepCopy(value)
	}
	return reads
}

// invalidOutputStep returns the innermost StepError in err's tree whose error is an invalid
// output, which names the step that produced it, or nil when there is none. It walks the tree
// in the order errors.Is does, so when two branches hold an invalid output, as a policy whose
// step and fallback step both failed can, the one errors.Is reaches first wins.
func invalidOutputStep(err error) *StepError {
	switch e := err.(type) {
	case nil:
		return nil
	case *StepError:
		if inner := invalidOutputStep(e.Err); inner != nil {
			return inner
		}
		if errors.Is(e.Err, ErrInvalidOutput) {
			return e
		}
		return nil
	case interface{ Unwrap() error }:
		return invalidOutputStep(e.Unwrap())
	case interface{ Unwrap() []error }:
		for _, inner := range e.Unwrap() {
			if found := invalidOutputStep(inner); found != nil {
				return found
			}
		}
	}
	return nil
}

// reflectionValues returns values with each value that encoding/json cannot encode, such as a
// function or NaN, replaced by a note naming its type, so one such value cannot stop the
// reflection prompt from being built.
func reflectionValues(values map[string]any) map[string]any {
	if values == nil {
		return nil
	}
	out := make(map[string]any, len(values))
	for key, value := range values {
		if _, err := json.Marshal(value); err != nil {
			out[key] = fmt.Sprintf("value of type %T cannot be encoded as JSON", value)
			continue
		}
		out[key] = value
	}
	return out
}
