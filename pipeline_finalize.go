package gepa

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"time"
)

// ErrFinalization marks a failed finalizer, independently of the error that ended the steps.
var ErrFinalization = errors.New("pipeline finalizer failed")

// PipelineFinalizer completes host bookkeeping after the steps of a run end.
// Run receives the committed state, including inputs and defaults, and the run's error.
// It runs once before RunFinished, on success, failure or cancellation.
// Invalid inputs and an unbuilt pipeline do not start a run or call Run.
//
// The state is read-only. The callback cannot change the pipeline's result or suppress its error.
// Its own error matches ErrFinalization and is joined to the run's error. A panic becomes a PanicError.
//
// The context keeps caller values, but ignores caller cancellation and has its own positive Timeout.
// It carries no pipeline observation, budget, usage collector, evaluation capture or tool session.
// Work started by the callback is independent of the completed steps.
// Like a step timeout, Timeout requires cooperation: the pipeline waits for Run to return.
// A callback that returns after its deadline fails even when it returns nil.
type PipelineFinalizer struct {
	Timeout time.Duration
	Run     func(context.Context, State, error) error
}

func (f *PipelineFinalizer) validate() error {
	if f == nil {
		return nil
	}
	if f.Run == nil {
		return errors.New("pipeline finalizer requires Run")
	}
	if f.Timeout <= 0 {
		return errors.New("pipeline finalizer Timeout must be positive")
	}
	return nil
}

func (f *PipelineFinalizer) clone() *PipelineFinalizer {
	if f == nil {
		return nil
	}
	copy := *f
	return &copy
}

func (p Pipeline) finalize(caller context.Context, state State, runErr error) (err error) {
	if p.finalizer == nil {
		return nil
	}
	base := observerCtx{Context: context.WithoutCancel(caller), values: caller}
	ctx, cancel := context.WithTimeout(base, p.finalizer.Timeout)
	defer cancel()
	defer func() {
		if value := recover(); value != nil {
			err = recoveredPanic(value, "pipeline finalizer panicked")
		}
		if expired := ctx.Err(); expired != nil && !errors.Is(err, expired) {
			err = errors.Join(err, expired)
		}
		if err != nil {
			err = fmt.Errorf("%w: %w", ErrFinalization, err)
		}
	}()
	reads := make(map[string]reflect.Type, len(state.values))
	for key, value := range state.values {
		reads[key] = reflect.TypeOf(value)
	}
	view := readOnlyView(state.values, reads)
	err = p.finalizer.Run(ctx, view, runErr)
	if view.scope.err != nil {
		err = errors.Join(err, view.scope.err)
	}
	return err
}
