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
// MaxModelCalls counts the provider calls AdmitModelCall admits, each retry of a
// RetryingLanguageModel included: a run with MaxModelCalls N makes N calls and the next is
// refused before it reaches the model. MaxTokens and PerStepTokens count every token spent:
// those RecordUsage records, as Usage.TotalTokens or, when a provider leaves that zero,
// PromptTokens plus CompletionTokens, and those a retrying model has spent on earlier attempts
// of a call still in progress. Reaching a token limit is allowed; the run or step fails as soon
// as spent tokens exceed it, and no call or retry starts after that. MaxDuration fails the run
// once the time since it started reaches the limit.
//
// Exceeding a limit fails with a Budget error (see IsBudget) that names the limit and its
// value. A run limit ends the run: it cancels the context of every running step, and a step
// that finishes after that fails with the budget error, its writes dropped. A PerStepTokens
// limit fails only the steps it names, which a policy or Fallback around them may handle.
//
// MaxDuration does not bound a step that ignores its context: the pipeline waits for every
// step to return rather than abandon a goroutine that still uses the run's state, and only then
// reports the budget error.
type RunBudget struct {
	MaxDuration   time.Duration
	MaxModelCalls int
	MaxTokens     int
	// PerStepTokens limits the tokens spent while the named step runs, nested steps included,
	// summed over all its executions in the run. A name may be a nested step's. A step's
	// OnErrorFallback step is not part of it: the fallback is limited by its own entry, under
	// its own name, and by the run limits.
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
				remaining.StepTokens = min(remaining.StepTokens, max(limit-frame.spent(), 0))
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
			remaining.Tokens = min(remaining.Tokens, max(limits.MaxTokens-frame.spent(), 0))
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
// returned lists the retried calls made with this frame's context that have returned with
// pending tokens their callers have yet to record.
type usageFrame struct {
	parent   *usageFrame
	budget   *runBudget
	step     string
	cancel   context.CancelCauseFunc
	usage    Usage
	returned []returnedCall
}

// returnedCall is a retried call that has returned: the tokens of the usage it returned, which
// its caller records, and the part of them still pending.
type returnedCall struct {
	tokens  int
	pending int
}

// runBudget is the spending of one run against its RunBudget. Tokens are spent once recorded
// (tokens, stepTokens) or once a retrying call reports a failed attempt whose usage its caller
// has yet to record (pending, stepPending). Its fields, and those of every frame, are guarded
// by mu, which a nested run shares with the run around it.
type runBudget struct {
	mu          *sync.Mutex
	limits      RunBudget
	now         func() time.Time
	start       time.Time
	calls       int
	tokens      int
	pending     int
	stepTokens  map[string]int
	stepPending map[string]int
	stepErrs    map[string]error
	err         error
	durationErr error
	cancel      context.CancelCauseFunc
}

// spent returns the tokens spent against f's limit: the run's for a run frame, the step's for
// a step frame. The caller holds mu.
func (f *usageFrame) spent() int {
	if f.step == "" {
		return f.budget.tokens + f.budget.pending
	}
	return f.budget.stepTokens[f.step] + f.budget.stepPending[f.step]
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
		mu:          mu,
		limits:      limits,
		now:         now,
		start:       now(),
		stepTokens:  map[string]int{},
		stepPending: map[string]int{},
		stepErrs:    map[string]error{},
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

// add records usage against f and every frame around it, failing each limit it exceeds. When
// usage is what a retried call made with f's context returned, recording it settles that call's
// pending tokens, so its failed attempts, counted when the call reported them, are not counted
// again; a call still in progress is never settled. Two returned calls with the same token
// total whose callers record them at the same moment may settle each other's pending tokens;
// the totals are right again once both have recorded.
func (f *usageFrame) add(usage Usage) {
	if f == nil {
		return
	}
	tokens := usageTokens(usage)
	f.budget.mu.Lock()
	defer f.budget.mu.Unlock()
	settled := 0
	if i := slices.IndexFunc(f.returned, func(call returnedCall) bool { return call.tokens == tokens }); i >= 0 {
		settled = f.returned[i].pending
		f.returned = slices.Delete(f.returned, i, i+1)
	}
	for frame := f; frame != nil; frame = frame.parent {
		frame.usage = frame.usage.Add(usage)
		budget := frame.budget
		if frame.step == "" {
			budget.pending -= settled
			budget.tokens += tokens
			budget.checkTokens(frame.spent())
			continue
		}
		budget.stepPending[frame.step] -= settled
		budget.stepTokens[frame.step] += tokens
		frame.checkStepTokens(frame.spent())
	}
}

// checkTokens fails the run when spent, the tokens it has spent, exceeds MaxTokens, and
// returns the error that ended the run. The caller holds mu.
func (b *runBudget) checkTokens(spent int) error {
	if limit := b.limits.MaxTokens; limit > 0 && spent > limit {
		return b.fail(BudgetError(fmt.Errorf("run budget MaxTokens %d exceeded: %d tokens spent", limit, spent)))
	}
	return nil
}

// checkStepTokens fails the step of frame f when spent, the tokens the step has spent,
// exceeds its PerStepTokens limit: it cancels the step's context with the budget error and
// returns that error. The caller holds mu.
func (f *usageFrame) checkStepTokens(spent int) error {
	budget := f.budget
	limit := budget.limits.PerStepTokens[f.step]
	if limit <= 0 || spent <= limit {
		return nil
	}
	if budget.stepErrs[f.step] == nil {
		budget.stepErrs[f.step] = BudgetError(fmt.Errorf(
			"run budget PerStepTokens[%q] %d exceeded: %d tokens spent",
			f.step,
			limit,
			spent,
		))
	}
	f.cancel(budget.stepErrs[f.step])
	return budget.stepErrs[f.step]
}

type retryAdmissionKey struct{}

// retryAdmission admits the attempts after the first of one call of a retrying model, such as
// RetryingLanguageModel, against the budgets of the pipeline run in its context.
//
// Tokens of failed attempts are pending from the moment the call reports them until its caller
// records the usage the call returns: the run and its steps count them as spent, for every
// call's admissions. reported is the number of tokens of the call's attempts already pending.
// A retrying model called by another with the same usage frame is nested in it: the tokens it
// reports count as reported by every call it is nested in, so the outer call does not report
// them again, and only the outermost call leaves its pending tokens to settle on return.
type retryAdmission struct {
	frame    *usageFrame
	parent   *retryAdmission
	reported int
}

// beginRetryAdmission starts the admission of one call and returns the context its attempts
// run with. Outside a pipeline run it admits everything.
func beginRetryAdmission(ctx context.Context) (context.Context, *retryAdmission) {
	frame := usageFrameFrom(ctx)
	if frame == nil {
		return ctx, &retryAdmission{}
	}
	call := &retryAdmission{frame: frame}
	if parent, _ := ctx.Value(retryAdmissionKey{}).(*retryAdmission); parent != nil && parent.frame == frame {
		call.parent = parent
	}
	return context.WithValue(ctx, retryAdmissionKey{}, call), call
}

// next reserves the next attempt of the call. spent is the summed usage of its attempts so far;
// the tokens of it that are not yet reported become pending.
func (r *retryAdmission) next(spent Usage) error {
	f := r.frame
	if f == nil {
		return nil
	}
	f.budget.mu.Lock()
	defer f.budget.mu.Unlock()
	unreported := max(usageTokens(spent)-r.reported, 0)
	for call := r; call != nil; call = call.parent {
		call.reported += unreported
	}
	for frame := f; frame != nil; frame = frame.parent {
		if frame.step == "" {
			frame.budget.pending += unreported
		} else {
			frame.budget.stepPending[frame.step] += unreported
		}
	}
	return f.admitLocked()
}

// finish ends the call, which returns spent. An outermost call leaves its pending tokens on its
// frame for the caller's RecordUsage of spent to settle; a nested call's pending tokens belong
// to the call around it.
func (r *retryAdmission) finish(spent *Usage) {
	if r.frame == nil || r.parent != nil {
		return
	}
	r.frame.budget.mu.Lock()
	defer r.frame.budget.mu.Unlock()
	if r.reported > 0 {
		r.frame.returned = append(r.frame.returned, returnedCall{tokens: usageTokens(*spent), pending: r.reported})
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
	return f.admitLocked()
}

// admitLocked is admit for a caller that holds mu. A token limit that the spent tokens,
// pending ones included, exceed refuses the call.
func (f *usageFrame) admitLocked() error {
	if err := f.checkLocked(); err != nil {
		return err
	}
	for frame := f; frame != nil; frame = frame.parent {
		budget := frame.budget
		if frame.step != "" {
			if err := frame.checkStepTokens(frame.spent()); err != nil {
				return err
			}
			continue
		}
		if err := budget.checkTokens(frame.spent()); err != nil {
			return err
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
