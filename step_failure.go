package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
)

// failStep wraps err as the failure of node, as newStepError does, and records copies of the
// values of node's declared reads found in values. ctx is the context that node ran with. So the
// StepError keeps the iteration numbers that were in effect there. Without any, numbers stays nil,
// so a StepError compares as it did before Loops had iteration keys.
func failStep(ctx context.Context, node *stepNode, values map[string]any, err error) *StepError {
	stepErr := newStepError(node.name, err)
	stepErr.Reads = stepReads(node, values)
	if numbers := iterationFrom(ctx).numbers; len(numbers) > 0 {
		stepErr.numbers = numbers
	}
	return stepErr
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
// in the order errors.Is does, so when two branches hold an invalid output, as a Parallel whose
// steps both failed can, the one errors.Is reaches first wins.
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
