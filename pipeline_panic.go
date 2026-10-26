package gepa

import (
	"fmt"
	"runtime/debug"
)

// PanicError records a recovered panic from a step, tool, merge, transition hook or pipeline finalizer.
// Value is the original recovered value. Stack is the stack at recovery.
// A run's returned error retains both fields, so errors.As can find them.
// Redacted event errors expose only their message and error kind.
type PanicError struct {
	Value   any
	Stack   []byte
	message string
}

func (e *PanicError) Error() string {
	if e.message != "" {
		return e.message
	}
	return fmt.Sprintf("step panicked: %v", e.Value)
}

// recoveredPanic retains the panic and attaches its stack to the existing step details.
func recoveredPanic(value any, prefix string) error {
	stack := debug.Stack()
	return WithStepDetails(
		&PanicError{Value: value, Stack: stack, message: fmt.Sprintf("%s: %v", prefix, value)},
		map[string]any{
			"stack": string(stack),
		},
	)
}
