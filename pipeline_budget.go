package gepa

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"sync"
	"time"
)

// RunBudget limits one Pipeline run. A zero field sets no limit.
//
// MaxModelCalls counts the calls AdmitModelCall admits: a run with MaxModelCalls N makes N
// calls and the next is refused before it reaches the model. MaxTokens and PerStepTokens count
// the tokens RecordUsage records, as Usage.TotalTokens or, when a provider leaves that zero,
// PromptTokens plus CompletionTokens; reaching a token limit is allowed, and the run or step
// fails as soon as recorded tokens exceed it. MaxDuration fails the run once the time since it
// started reaches the limit.
//
// Exceeding a limit fails with a Budget error (see IsBudget) that names the limit and its
// value. A run limit ends the run: it cancels the context of every running step, and a step
// that finishes after that fails with the budget error, its writes dropped. A PerStepTokens
// limit fails only the steps it names, which a policy or Fallback around them may handle.
type RunBudget struct {
	MaxDuration   time.Duration
	MaxModelCalls int
	MaxTokens     int
	// PerStepTokens limits the tokens recorded while the named step runs, nested steps
	// included, summed over all its executions in the run. A name may be a nested step's.
	PerStepTokens map[string]int
}

func (b RunBudget) validate() error {
	switch {
	case b.MaxDuration < 0:
		return errors.New("run budget MaxDuration must not be negative")
	case b.MaxModelCalls < 0:
		return errors.New("run budget MaxModelCalls must not be negative")
	case b.MaxTokens < 0:
		return errors.New("run budget MaxTokens must not be negative")
	}
	for _, name := range slices.Sorted(maps.Keys(b.PerStepTokens)) {
		if b.PerStepTokens[name] <= 0 {
			return fmt.Errorf("run budget PerStepTokens[%q] must be positive", name)
		}
	}
	return nil
}

func (b RunBudget) clone() RunBudget {
	b.PerStepTokens = maps.Clone(b.PerStepTokens)
	return b
}

// BudgetMeter reports what is left of the budgets that cover a running step: the RunBudget of
// its run, those of any pipelines that run that pipeline as a step, and the PerStepTokens
// limits of the step and the steps around it. It is safe for concurrent use.
type BudgetMeter struct {
	frame *usageFrame
}

// BudgetFrom returns the budget meter of the pipeline step running with ctx. Outside a
// pipeline run the meter reports no limits.
func BudgetFrom(ctx context.Context) BudgetMeter {
	return BudgetMeter{frame: usageFrameFrom(ctx)}
}

// BudgetRemaining is what is left of each budget limit: the least that is left where more than
// one limit applies, and never less than zero. A limit that is not set reports the largest
// value of its type.
type BudgetRemaining struct {
	Duration   time.Duration
	ModelCalls int
	Tokens     int
	// StepTokens is what is left of the PerStepTokens limits of the step and its enclosing steps.
	StepTokens int
}

// Remaining reports what is left of each limit now.
func (m BudgetMeter) Remaining() BudgetRemaining {
	remaining := BudgetRemaining{
		Duration:   math.MaxInt64,
		ModelCalls: math.MaxInt,
		Tokens:     math.MaxInt,
		StepTokens: math.MaxInt,
	}
	if m.frame == nil {
		return remaining
	}
	m.frame.budget.mu.Lock()
	defer m.frame.budget.mu.Unlock()
	for frame := m.frame; frame != nil; frame = frame.parent {
		budget := frame.budget
		limits := budget.limits
		if frame.step != "" {
			if limit := limits.PerStepTokens[frame.step]; limit > 0 {
				remaining.StepTokens = min(remaining.StepTokens, max(limit-budget.stepTokens[frame.step], 0))
			}
			continue
		}
		if limits.MaxDuration > 0 {
			left := limits.MaxDuration - budget.now().Sub(budget.start)
			remaining.Duration = min(remaining.Duration, max(left, 0))
		}
		if limits.MaxModelCalls > 0 {
			remaining.ModelCalls = min(remaining.ModelCalls, max(limits.MaxModelCalls-budget.calls, 0))
		}
		if limits.MaxTokens > 0 {
			remaining.Tokens = min(remaining.Tokens, max(limits.MaxTokens-budget.tokens, 0))
		}
	}
	return remaining
}

type usageFrameKey struct{}

// budgetClockKey carries a func() time.Time that replaces the clock a run measures
// MaxDuration with, so tests can place a run exactly at its limit.
type budgetClockKey struct{}

// usageFrame attributes model usage to one step execution, or to a whole run when step is
// empty, and enforces their limits. Frames chain from the innermost running step out through
// its enclosing steps and runs, including the runs of pipelines that run this one as a step.
type usageFrame struct {
	parent *usageFrame
	budget *runBudget
	step   string
	cancel context.CancelCauseFunc
	usage  Usage
}

// runBudget is the spending of one run against its RunBudget. Its fields, and the usage of
// every frame, are guarded by mu, which a nested run shares with the run around it.
type runBudget struct {
	mu          *sync.Mutex
	limits      RunBudget
	now         func() time.Time
	start       time.Time
	calls       int
	tokens      int
	stepTokens  map[string]int
	stepErrs    map[string]error
	err         error
	durationErr error
	cancel      context.CancelCauseFunc
}

func usageFrameFrom(ctx context.Context) *usageFrame {
	frame, _ := ctx.Value(usageFrameKey{}).(*usageFrame)
	return frame
}

// startRunBudget returns ctx carrying a run frame for limits under any frame ctx already
// carries, and a function that releases the run's context. A run limit cancels that context
// with its budget error as the cause.
func startRunBudget(ctx context.Context, limits RunBudget) (context.Context, func()) {
	parent := usageFrameFrom(ctx)
	mu := &sync.Mutex{}
	if parent != nil {
		mu = parent.budget.mu
	}
	now, _ := ctx.Value(budgetClockKey{}).(func() time.Time)
	if now == nil {
		now = time.Now
	}
	budget := &runBudget{
		mu:         mu,
		limits:     limits,
		now:        now,
		start:      now(),
		stepTokens: map[string]int{},
		stepErrs:   map[string]error{},
		durationErr: BudgetError(
			fmt.Errorf("run budget MaxDuration %s exceeded: %w", limits.MaxDuration, context.DeadlineExceeded),
		),
	}
	ctx, cancel := context.WithCancelCause(ctx)
	budget.cancel = cancel
	release := func() { cancel(nil) }
	if limits.MaxDuration > 0 {
		var cancelDeadline context.CancelFunc
		ctx, cancelDeadline = context.WithDeadlineCause(ctx, time.Now().Add(limits.MaxDuration), budget.durationErr)
		release = func() {
			cancelDeadline()
			cancel(nil)
		}
	}
	return context.WithValue(ctx, usageFrameKey{}, &usageFrame{parent: parent, budget: budget}), release
}

// enterStep opens a frame for one execution of step under f. Its context is cancelled with
// the step's budget error when the step exceeds its PerStepTokens limit.
func (f *usageFrame) enterStep(ctx context.Context, step string) (context.Context, *usageFrame, func()) {
	if f == nil {
		return ctx, nil, func() {}
	}
	ctx, cancel := context.WithCancelCause(ctx)
	frame := &usageFrame{parent: f, budget: f.budget, step: step, cancel: cancel}
	return context.WithValue(ctx, usageFrameKey{}, frame), frame, func() { cancel(nil) }
}

// add records usage against f and every frame around it, failing each limit it exceeds.
func (f *usageFrame) add(usage Usage) {
	if f == nil {
		return
	}
	tokens := usageTokens(usage)
	f.budget.mu.Lock()
	defer f.budget.mu.Unlock()
	for frame := f; frame != nil; frame = frame.parent {
		frame.usage = frame.usage.Add(usage)
		budget := frame.budget
		if frame.step == "" {
			budget.tokens += tokens
			if limit := budget.limits.MaxTokens; limit > 0 && budget.tokens > limit {
				budget.fail(BudgetError(
					fmt.Errorf("run budget MaxTokens %d exceeded: %d tokens recorded", limit, budget.tokens),
				))
			}
			continue
		}
		budget.stepTokens[frame.step] += tokens
		limit := budget.limits.PerStepTokens[frame.step]
		if limit <= 0 || budget.stepTokens[frame.step] <= limit {
			continue
		}
		if budget.stepErrs[frame.step] == nil {
			budget.stepErrs[frame.step] = BudgetError(fmt.Errorf(
				"run budget PerStepTokens[%q] %d exceeded: %d tokens recorded",
				frame.step,
				limit,
				budget.stepTokens[frame.step],
			))
		}
		frame.cancel(budget.stepErrs[frame.step])
	}
}

// admit reserves one model call against every run around f, or returns the budget error that
// refuses it without reserving anything.
func (f *usageFrame) admit() error {
	if f == nil {
		return nil
	}
	f.budget.mu.Lock()
	defer f.budget.mu.Unlock()
	if err := f.checkLocked(); err != nil {
		return err
	}
	for frame := f; frame != nil; frame = frame.parent {
		budget := frame.budget
		if frame.step != "" {
			continue
		}
		if limit := budget.limits.MaxModelCalls; limit > 0 && budget.calls >= limit {
			return budget.fail(BudgetError(
				fmt.Errorf("run budget MaxModelCalls %d exceeded: model call %d refused", limit, budget.calls+1),
			))
		}
	}
	for frame := f; frame != nil; frame = frame.parent {
		if frame.step == "" {
			frame.budget.calls++
		}
	}
	return nil
}

// check returns the budget error that stops work under f, if there is one.
func (f *usageFrame) check() error {
	if f == nil {
		return nil
	}
	f.budget.mu.Lock()
	defer f.budget.mu.Unlock()
	return f.checkLocked()
}

// checkLocked returns the first budget error that applies under f: a step over its token
// limit, a spent run budget or a run that has reached MaxDuration. The caller holds mu.
func (f *usageFrame) checkLocked() error {
	for frame := f; frame != nil; frame = frame.parent {
		budget := frame.budget
		if frame.step != "" {
			if err := budget.stepErrs[frame.step]; err != nil {
				return err
			}
			continue
		}
		if budget.err != nil {
			return budget.err
		}
		if limit := budget.limits.MaxDuration; limit > 0 && budget.now().Sub(budget.start) >= limit {
			return budget.fail(budget.durationErr)
		}
	}
	return nil
}

// total returns the usage recorded against f.
func (f *usageFrame) total() Usage {
	if f == nil {
		return Usage{}
	}
	f.budget.mu.Lock()
	defer f.budget.mu.Unlock()
	return f.usage
}

// fail records the run's first budget error and cancels the run with it. It returns the error
// that ended the run. The caller holds mu.
func (b *runBudget) fail(err error) error {
	if b.err == nil {
		b.err = err
		b.cancel(err)
	}
	return b.err
}

func usageTokens(usage Usage) int {
	if usage.TotalTokens != 0 {
		return usage.TotalTokens
	}
	return usage.PromptTokens + usage.CompletionTokens
}
