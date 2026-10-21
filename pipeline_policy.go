package gepa

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"time"
)

// StepPolicy sets how a pipeline runs one step: a timeout for each attempt and retries of
// Transient errors. Apply it with WithPolicy. A step that still fails after its last attempt
// fails; an OnError Transition (see Transition and Fallback) decides what the run does next.
type StepPolicy struct {
	// Timeout bounds each attempt; zero means no timeout. An attempt that runs past it fails
	// with a Transient error that matches context.DeadlineExceeded, and its writes are
	// dropped. Timeout does not bound a step that ignores its context: the pipeline waits for
	// the step to return rather than abandon a goroutine that still uses the run's state, and
	// only then reports the timeout.
	Timeout time.Duration
	Retry   RetryPolicy
}

// RetryPolicy retries a step that failed with a Transient error (see IsTransient), which
// includes an attempt that timed out. Any other error fails the step at once, and nothing is
// retried once the context the policy runs under has ended: when the run was cancelled, spent
// a RunBudget limit or reached MaxDuration, or a sibling in a Parallel or a Map failed. Each
// attempt runs on a fresh view of the state as it was before the step, so a failed attempt's
// writes never reach the next attempt.
type RetryPolicy struct {
	// MaxAttempts is the total number of attempts; zero means one.
	MaxAttempts int
	// Backoff returns the wait after failed attempt n, counting from one, before the next.
	// Nil retries without waiting. The wait ends early when the run's context ends.
	Backoff func(attempt int) time.Duration
}

// WithPolicy returns step run under policy. The returned step has step's name, reads and
// writes, and in a pipeline step keeps its candidate keys, so a policy changes neither the
// pipeline's candidate keys nor how budgets, transitions and other steps refer to step. It is
// a ComponentStep when step is one. Its own SeedCandidate and RunCandidate use keys already
// scoped by step's name, because it shares the candidate namespace of the steps around it.
// Build rejects a negative Timeout or MaxAttempts.
func WithPolicy(step Step, policy StepPolicy) Step {
	p := &policyStep{child: newStepNode(step), policy: policy}
	return asComponentStep(p, p.child)
}

type policyStep struct {
	child  *stepNode
	policy StepPolicy
}

func (p *policyStep) Name() string     { return p.child.name }
func (p *policyStep) Reads() []KeyRef  { return slices.Clone(p.child.reads) }
func (p *policyStep) Writes() []KeyRef { return slices.Clone(p.child.writes) }
func (p *policyStep) wrapper() bool    { return true }

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
	return []childStep{{node: p.child}}
}

func (p *policyStep) checkChildren() error { return nil }

func (p *policyStep) seedCandidate() Candidate {
	return seedChildren(p.child)
}

func (p *policyStep) Run(ctx context.Context, s State) error {
	return p.runCandidate(ctx, nil, s)
}

func (p *policyStep) runCandidate(ctx context.Context, candidate Candidate, s State) error {
	attempts := max(p.policy.Retry.MaxAttempts, 1)
	for attempt := 1; ; attempt++ {
		result, err := p.attempt(ctx, candidate, s.values, attempt)
		if err == nil {
			result.commitTo(s, p.child)
			return nil
		}
		if ctx.Err() != nil {
			return err
		}
		if attempt >= attempts || !IsTransient(err) {
			if attempt > 1 {
				err = fmt.Errorf("after %d attempts: %w", attempt, err)
			}
			return err
		}
		if !sleepContext(ctx, p.backoff(attempt)) {
			return errors.Join(err, ctx.Err())
		}
	}
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
