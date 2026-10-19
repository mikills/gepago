package gepa

import (
	"context"
	"errors"
	"fmt"
	"runtime/debug"
	"sync"
	"time"
)

type pipelineRunKey struct{}

// pipelineRun is what the steps of one Pipeline run share: the defaults a skipped step
// commits, the step cache, and the outcome of every step execution. It is safe for
// concurrent use, and a nil run, outside a pipeline, records and caches nothing.
type pipelineRun struct {
	defaults map[string]any
	mu       sync.Mutex
	cache    map[stepCacheKey]stepResult
	attempts map[string]int
	outcomes []stepOutcome
}

// stepCacheKey identifies one cached step result: the step and the key WithCache computed from
// its reads.
type stepCacheKey struct {
	step string
	key  string
}

// stepOutcome records one execution of a step, top-level or nested: each retry attempt, each
// cache hit and each skip on error is an execution of its own. Attempt counts the step's
// executions in the run from one. Usage is what the execution's model calls recorded, nested
// steps included. Err is the step's own error, before any enclosing step wraps it.
type stepOutcome struct {
	Step     string
	Attempt  int
	Duration time.Duration
	Usage    Usage
	Cached   bool
	Skipped  bool
	Err      error
}

func newPipelineRun(defaults map[string]any) *pipelineRun {
	return &pipelineRun{defaults: defaults, cache: map[stepCacheKey]stepResult{}, attempts: map[string]int{}}
}

func pipelineRunFrom(ctx context.Context) *pipelineRun {
	run, _ := ctx.Value(pipelineRunKey{}).(*pipelineRun)
	return run
}

func (r *pipelineRun) nextAttempt(step string) int {
	if r == nil {
		return 1
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.attempts[step]++
	return r.attempts[step]
}

func (r *pipelineRun) record(outcome stepOutcome) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.outcomes = append(r.outcomes, outcome)
}

func (r *pipelineRun) defaultFor(key string) (any, bool) {
	if r == nil {
		return nil, false
	}
	value, ok := r.defaults[key]
	return value, ok
}

// cached returns a copy of the result cached for step under key.
func (r *pipelineRun) cached(step, key string) (stepResult, bool) {
	if r == nil {
		return stepResult{}, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	result, ok := r.cache[stepCacheKey{step: step, key: key}]
	return result.clone(), ok
}

func (r *pipelineRun) storeCached(step, key string, result stepResult) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.cache[stepCacheKey{step: step, key: key}] = result.clone()
}

// stepResult is what a successful step leaves to commit: its pending writes, already copies
// of their declared types, and the declared writes it left unwritten because it was skipped.
type stepResult struct {
	writes  map[string]any
	skipped map[string]bool
}

func (r stepResult) clone() stepResult {
	out := stepResult{writes: make(map[string]any, len(r.writes))}
	for key, value := range r.writes {
		out.writes[key] = deepCopy(value)
	}
	for key := range r.skipped {
		if out.skipped == nil {
			out.skipped = map[string]bool{}
		}
		out.skipped[key] = true
	}
	return out
}

// commitTo copies a child's result into the view of the composite step that ran it.
func (r stepResult) commitTo(s State, child *stepNode) {
	for key, value := range r.writes {
		s.store(key, child.writeTypes[key], value, true)
	}
	for key := range r.skipped {
		s.skip(key)
	}
}

// runChild runs a step nested in the composite step named parent, on a fresh view over the
// values the composite sees, and returns its result for the composite to commit. candidate is
// the composite's candidate namespace. The child's error is wrapped in a StepError naming it,
// unless the child is the step a WithPolicy or WithCache wrapper wraps, which shares the
// wrapper's name.
func runChild(ctx context.Context, candidate Candidate, parent string, child *stepNode, values map[string]any) (
	stepResult,
	error,
) {
	result, err := runNode(ctx, childCandidate(candidate, child), child, values, nil)
	if err != nil && child.name != parent {
		return stepResult{}, newStepError(child.name, err)
	}
	return result, err
}

// runNode runs one step on a fresh view of values limited to its declared keys and returns its
// writes when it succeeds. The candidate is the step's, as childCandidate selects it. Each
// execution of a step other than a WithPolicy or WithCache wrapper gets its own usage frame,
// observer events and outcome record. A panic becomes an error with the stack in its details.
// When the step's context ended because a budget ran out or an attempt timed out, that cause
// replaces the step's result, unless the step's error already carries it, so a step that
// finishes late still fails and its writes are dropped.
//
// commit, when set, checks the result before the execution ends, so an error it returns fails
// the execution; a wrapper, which has no execution of its own, records the failure as an
// execution that commits nothing.
func runNode(
	ctx context.Context,
	candidate Candidate,
	node *stepNode,
	values map[string]any,
	commit func(stepResult) error,
) (result stepResult, err error) {
	if node.step == nil {
		return stepResult{}, errors.New("step is nil")
	}
	var scope *stepScope
	if !isWrapper(node.step) {
		var finish func(stepResult, error, []stateWrite)
		ctx, finish = beginExecution(ctx, node, values)
		defer func() { finish(result, err, scope.attempted()) }()
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			result = stepResult{}
			err = WithStepDetails(
				fmt.Errorf("step panicked: %v", recovered),
				map[string]any{"stack": string(debug.Stack())},
			)
		}
	}()
	if err := stepCanStart(ctx); err != nil {
		return stepResult{}, err
	}
	scope = &stepScope{reads: node.readTypes, writes: node.writeTypes, pending: map[string]any{}}
	view := State{values: values, scope: scope}
	var runErr error
	if componentStep, ok := node.step.(ComponentStep); ok {
		runErr = componentStep.RunCandidate(ctx, candidate, view)
	} else {
		runErr = node.step.Run(ctx, view)
	}
	if cause := stopCause(ctx); cause != nil && !errors.Is(runErr, cause) {
		return stepResult{}, cause
	}
	switch {
	case runErr != nil && scope.err != nil:
		return stepResult{}, errors.Join(runErr, scope.err)
	case runErr != nil:
		return stepResult{}, runErr
	case scope.err != nil:
		return stepResult{}, scope.err
	}
	for _, key := range node.writeNames {
		if _, ok := scope.pending[key]; !ok && !scope.skipped[key] {
			return stepResult{}, fmt.Errorf("step did not write declared state key %q", key)
		}
	}
	result = stepResult{writes: scope.pending, skipped: scope.skipped}
	if commit == nil {
		return result, nil
	}
	if err := commit(result); err != nil {
		if isWrapper(node.step) {
			beginStandIn(ctx, commitFailed, node, values)(stepResult{}, err)
		}
		return stepResult{}, err
	}
	return result, nil
}

// beginExecution starts one execution of node over values, the state its view reads: it
// numbers the attempt, reports the start to the run's observers, opens a usage frame and
// returns a function that records the outcome when the execution ends, given the values the
// step wrote or tried to write.
func beginExecution(ctx context.Context, node *stepNode, values map[string]any) (
	context.Context,
	func(stepResult, error, []stateWrite),
) {
	run := pipelineRunFrom(ctx)
	attempt := run.nextAttempt(node.name)
	ctx, observed := observeStep(ctx, node, attempt, values)
	ctx, frame, release := usageFrameFrom(ctx).enterStep(ctx, node.name)
	start := time.Now()
	return ctx, func(result stepResult, err error, attempted []stateWrite) {
		release()
		outcome := stepOutcome{
			Step:     node.name,
			Attempt:  attempt,
			Duration: time.Since(start),
			Usage:    frame.total(),
			Err:      err,
		}
		run.record(outcome)
		observed.finish(outcome, result, attempted)
	}
}

// standIn is an execution that does not run its step.
type standIn int

const (
	// cacheHit commits a WithCache step's cached writes.
	cacheHit standIn = iota
	// skippedOnError commits the defaults of a step that skips on error.
	skippedOnError
	// commitFailed fails a top-level WithPolicy or WithCache step whose writes cannot be merged
	// into the state.
	commitFailed
)

// beginStandIn starts an execution of node over values that does not run the step, and
// returns a function that records it with the writes it committed or its error.
func beginStandIn(ctx context.Context, kind standIn, node *stepNode, values map[string]any) func(stepResult, error) {
	run := pipelineRunFrom(ctx)
	attempt := run.nextAttempt(node.name)
	_, observed := observeStep(ctx, node, attempt, values)
	start := time.Now()
	return func(result stepResult, err error) {
		outcome := stepOutcome{
			Step:     node.name,
			Attempt:  attempt,
			Duration: time.Since(start),
			Cached:   kind == cacheHit,
			Skipped:  kind == skippedOnError,
			Err:      err,
		}
		run.record(outcome)
		observed.finish(outcome, result, nil)
	}
}

// stepCanStart reports why a step must not start: its context has ended or a budget that
// covers it is spent.
func stepCanStart(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		if cause := stopCause(ctx); cause != nil {
			return cause
		}
		return err
	}
	return usageFrameFrom(ctx).check()
}

// stopCause returns the cause of ctx ending when the pipeline reports it in place of a step's
// own error: a spent budget or a timed-out attempt.
func stopCause(ctx context.Context) error {
	if ctx.Err() == nil {
		return nil
	}
	cause := context.Cause(ctx)
	var timeout *stepTimeoutError
	if IsBudget(cause) || errors.As(cause, &timeout) {
		return cause
	}
	return nil
}

// stepTimeoutError reports an attempt that ran past its policy's timeout. It matches
// context.DeadlineExceeded through errors.Is.
type stepTimeoutError struct {
	attempt int
	timeout time.Duration
}

func (e *stepTimeoutError) Error() string {
	return fmt.Sprintf("attempt %d timed out after %s: %v", e.attempt, e.timeout, context.DeadlineExceeded)
}

func (e *stepTimeoutError) Unwrap() error {
	return context.DeadlineExceeded
}
