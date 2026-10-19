package gepa

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"time"
)

// StepPolicy sets how a pipeline runs one step: a timeout for each attempt, retries, and what
// to do when the step still fails. Apply it with WithPolicy.
type StepPolicy struct {
	// Timeout bounds each attempt; zero means no timeout. An attempt that runs past it fails
	// with a Transient error that matches context.DeadlineExceeded, and its writes are
	// dropped. Timeout does not bound a step that ignores its context: the pipeline waits for
	// the step to return rather than abandon a goroutine that still uses the run's state, and
	// only then reports the timeout.
	Timeout time.Duration
	Retry   RetryPolicy
	OnError ErrorAction
}

// RetryPolicy retries a failed step. Each attempt runs on a fresh view of the state as it was
// before the step, so a failed attempt's writes never reach the next attempt.
type RetryPolicy struct {
	// MaxAttempts is the total number of attempts; zero means one.
	MaxAttempts int
	// Backoff returns the wait after failed attempt n, counting from one, before the next.
	// Nil retries without waiting. The wait ends early when the run's context ends.
	Backoff func(attempt int) time.Duration
	// RetryIf reports whether an error is worth another attempt. Nil retries Transient errors
	// only (see IsTransient), which include timed-out attempts.
	RetryIf func(error) bool
}

type errorActionKind int

const (
	errorActionFail errorActionKind = iota
	errorActionSkip
	errorActionFallback
)

// ErrorAction says what a policy does when its step still fails after its last attempt. The
// zero value is OnErrorFail. An action applies to any error the step returns, permanent ones
// included, but not once the context the policy runs under has ended: when the run was
// cancelled, spent a RunBudget limit or reached MaxDuration, or a sibling in a Parallel failed,
// the step fails with its error. The same holds for retries.
type ErrorAction struct {
	kind     errorActionKind
	fallback Step
}

var (
	// OnErrorFail fails the step with its error. It is the zero ErrorAction.
	OnErrorFail = ErrorAction{kind: errorActionFail}
	// OnErrorSkip drops the failed step's writes and lets the pipeline continue: each write
	// with a default in PipelineSpec.Defaults commits the default, and each other write is left
	// unwritten. Build requires every write to have a default or a later step that writes it.
	OnErrorSkip = ErrorAction{kind: errorActionSkip}
)

// OnErrorFallback runs step in place of the failed step, on a fresh view of the state as it
// was before the failed step, and commits its writes. step must write the same keys with the
// same types as the failed step and may read other keys. Its name must be unique in the
// pipeline, and its components are scoped beside the failed step's, as "<step>.<component>"
// or "<outer>.<step>.<component>" inside a composite step, so they never collide with the
// failed step's own. It runs as part of the failed step: its tokens count against that step's
// PerStepTokens limit. When step fails too, the error keeps both errors for errors.Is and
// errors.As.
//
// Unlike Fallback, which falls back only on a non-permanent error, OnErrorFallback falls back
// on any error left after the policy's retries.
func OnErrorFallback(step Step) ErrorAction {
	return ErrorAction{kind: errorActionFallback, fallback: step}
}

// WithPolicy returns step run under policy. The returned step has step's name and writes, and
// in a pipeline step keeps its candidate keys, so a policy changes neither the pipeline's
// candidate keys nor how budgets and other steps refer to step; its reads add those of an
// OnErrorFallback step. It is a ComponentStep when step or the fallback step is one. Its own
// SeedCandidate and RunCandidate use keys already scoped by step's name and the fallback
// step's, because it shares the candidate namespace of the steps around it. Build rejects a
// negative Timeout or MaxAttempts.
func WithPolicy(step Step, policy StepPolicy) Step {
	p := &policyStep{child: newStepNode(step), policy: policy}
	nodes := []*stepNode{p.child}
	if policy.OnError.kind == errorActionFallback {
		p.fallback = newStepNode(policy.OnError.fallback)
		nodes = append(nodes, p.fallback)
	}
	for _, node := range nodes {
		p.reads = unionKeys(p.reads, node.reads)
	}
	return asComponentStep(p, nodes...)
}

type policyStep struct {
	child    *stepNode
	fallback *stepNode
	policy   StepPolicy
	reads    []KeyRef
}

func (p *policyStep) Name() string     { return p.child.name }
func (p *policyStep) Reads() []KeyRef  { return slices.Clone(p.reads) }
func (p *policyStep) Writes() []KeyRef { return slices.Clone(p.child.writes) }
func (p *policyStep) wrapper() bool    { return true }

func (p *policyStep) skipsOnError() bool {
	return p.policy.OnError.kind == errorActionSkip
}

func (p *policyStep) Validate() error {
	switch {
	case p.policy.Timeout < 0:
		return errors.New("policy timeout must not be negative")
	case p.policy.Retry.MaxAttempts < 0:
		return errors.New("policy retry max attempts must not be negative")
	}
	return nil
}

func (p *policyStep) children() []childStep {
	children := []childStep{{node: p.child}}
	if p.fallback != nil {
		children = append(children, childStep{label: "fallback step", node: p.fallback})
	}
	return children
}

func (p *policyStep) checkChildren() error {
	if p.fallback == nil {
		return nil
	}
	return sameWrites(
		fmt.Sprintf("step %q", p.child.name),
		p.child,
		fmt.Sprintf("fallback step %q", p.fallback.name),
		p.fallback,
	)
}

func (p *policyStep) seedCandidate() Candidate {
	nodes := []*stepNode{p.child}
	if p.fallback != nil {
		nodes = append(nodes, p.fallback)
	}
	return seedChildren(nodes...)
}

func (p *policyStep) Run(ctx context.Context, s State) error {
	return p.runCandidate(ctx, nil, s)
}

func (p *policyStep) runCandidate(ctx context.Context, candidate Candidate, s State) error {
	retryIf := p.policy.Retry.RetryIf
	if retryIf == nil {
		retryIf = IsTransient
	}
	attempts := max(p.policy.Retry.MaxAttempts, 1)
	var err error
	for attempt := 1; ; attempt++ {
		var result stepResult
		result, err = p.attempt(ctx, candidate, s.values, attempt)
		if err == nil {
			result.commitTo(s, p.child)
			return nil
		}
		if ctx.Err() != nil {
			return err
		}
		if attempt >= attempts || !retryIf(err) {
			if attempt > 1 {
				err = fmt.Errorf("after %d attempts: %w", attempt, err)
			}
			break
		}
		if !sleepContext(ctx, p.backoff(attempt)) {
			return errors.Join(err, ctx.Err())
		}
	}
	return p.onError(ctx, candidate, s, err)
}

// attempt runs the wrapped step once on a fresh view, under the policy's timeout. When the
// timeout fires, the step's run reports the timeout in place of its own result.
func (p *policyStep) attempt(ctx context.Context, candidate Candidate, values map[string]any, attempt int) (
	stepResult,
	error,
) {
	if p.policy.Timeout > 0 {
		timeout := TransientError(&stepTimeoutError{attempt: attempt, timeout: p.policy.Timeout})
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeoutCause(ctx, p.policy.Timeout, timeout)
		defer cancel()
	}
	return runChild(ctx, candidate, p.child.name, p.child, values)
}

func (p *policyStep) backoff(attempt int) time.Duration {
	if p.policy.Retry.Backoff == nil {
		return 0
	}
	return p.policy.Retry.Backoff(attempt)
}

func (p *policyStep) onError(ctx context.Context, candidate Candidate, s State, err error) error {
	switch p.policy.OnError.kind {
	case errorActionSkip:
		run := pipelineRunFrom(ctx)
		for _, key := range p.child.writes {
			if value, ok := run.defaultFor(key.Name); ok {
				s.store(key.Name, key.Type, deepCopy(value), true)
			} else {
				s.skip(key.Name)
			}
		}
		return nil
	case errorActionFallback:
		// the fallback runs as part of this step: its tokens count against the step's
		// PerStepTokens limit, which BudgetFrom inside the fallback reports.
		ctx, _, release := usageFrameFrom(ctx).enterStep(ctx, p.child.name)
		defer release()
		result, fallbackErr := runChild(ctx, candidate, p.child.name, p.fallback, s.values)
		if fallbackErr != nil {
			return &fallbackError{primary: err, fallback: fallbackErr}
		}
		result.commitTo(s, p.fallback)
		return nil
	}
	return err
}

// KeyDefault is the value a step that skips on error commits for one of its writes. Build it
// with DefaultValue.
type KeyDefault struct {
	Key   KeyRef
	Value any
}

// DefaultValue builds a KeyDefault for a typed key.
func DefaultValue[T any](k Key[T], value T) KeyDefault {
	return KeyDefault{Key: k.Ref(), Value: value}
}

// fallbackError keeps both errors of a step that failed, fell back and failed again. The
// fallback's error comes first in Unwrap, so when it has a kind, that kind is the kind of
// the whole.
type fallbackError struct {
	primary  error
	fallback error
}

func (e *fallbackError) Error() string {
	return fmt.Sprintf("%v; fallback failed: %v", e.primary, e.fallback)
}

func (e *fallbackError) Unwrap() []error {
	return []error{e.fallback, e.primary}
}
