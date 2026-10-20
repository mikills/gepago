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

// pipelineRun is what the steps of one Pipeline run share: the step cache, the outcome of every
// step execution and the run's tool session, which records every tool call. It is safe for
// concurrent use, and a nil run, outside a pipeline, records and caches nothing.
type pipelineRun struct {
	optional map[string]map[string]bool
	tools    *ToolSession
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
// steps included, and Unbudgeted what RecordUnbudgetedUsage recorded. Err is the step's own
// error, before any enclosing step wraps it. ToolCalls holds the tool calls the execution made
// itself, not those of the executions nested in it. Attributes holds what the step reported
// about its execution, such as a Map's failed elements.
type stepOutcome struct {
	Step       string
	Attempt    int
	Duration   time.Duration
	Usage      Usage
	Unbudgeted Usage
	Cached     bool
	Skipped    bool
	Err        error
	ToolCalls  []ToolCallRecord
	Attributes map[string]int
}

func newPipelineRun(optional map[string]map[string]bool) *pipelineRun {
	return &pipelineRun{
		optional: optional,
		cache:    map[stepCacheKey]stepResult{},
		attempts: map[string]int{},
	}
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

// optionalOutputs returns the optional outputs Build recorded for the programme step named step,
// and false when Build recorded none, as outside a pipeline.
func (r *pipelineRun) optionalOutputs(step string) (map[string]bool, bool) {
	if r == nil {
		return nil, false
	}
	optional, ok := r.optional[step]
	return optional, ok
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
// of their declared types, the declared writes it left unwritten because it was skipped, and
// the tool memory writes and cached tool results it staged.
type stepResult struct {
	writes  map[string]any
	skipped map[string]bool
	staged  stagedTools
}

func (r stepResult) clone() stepResult {
	out := stepResult{writes: make(map[string]any, len(r.writes)), staged: r.staged.clone()}
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

// commitTo copies a child's result into the view of the composite step that ran it, and what
// it staged for tools into the composite's stage.
func (r stepResult) commitTo(s State, child *stepNode) {
	for key, value := range r.writes {
		s.store(key, child.writeTypes[key], value, true)
	}
	for key := range r.skipped {
		s.skip(key)
	}
	if s.scope != nil && s.scope.stage != nil {
		s.scope.stage.add(r.staged)
	}
}

// commitStaged applies what a top-level step of a pipeline run staged for tools, once its
// state writes have committed: its memory writes to the run's memory, results cached by the
// run's own session to that session's cache, and results cached by an enclosing run's session
// to the stage of the step this run is nested in, or to that session when there is none.
func commitStaged(ctx context.Context, staged stagedTools) {
	if memory := ToolMemoryFrom(ctx); !isNil(memory) {
		applyMemoryWrites(memory, staged.memory)
	}
	var own *ToolSession
	if run := pipelineRunFrom(ctx); run != nil {
		own = run.tools
	}
	enclosing := stepStageFrom(ctx)
	for key, result := range staged.cache {
		switch {
		case key.session == own || enclosing == nil:
			key.session.store(key.key, result)
		default:
			enclosing.storeCached(key.session, key.key, result)
		}
	}
}

// stagedTools is what one step execution staged for tools: the memory writes and the cached
// results of the tool calls it made through a pipeline run's session.
type stagedTools struct {
	memory map[string]any
	cache  map[stagedCacheKey]ToolResult
}

// stagedCacheKey identifies a staged cached result: the session whose cache it belongs to and
// the key in that cache.
type stagedCacheKey struct {
	session *ToolSession
	key     toolCacheKey
}

func (t stagedTools) clone() stagedTools {
	var out stagedTools
	if t.memory != nil {
		out.memory = deepCopy(t.memory).(map[string]any)
	}
	if t.cache != nil {
		out.cache = make(map[stagedCacheKey]ToolResult, len(t.cache))
		for key, result := range t.cache {
			out.cache[key] = result.clone()
		}
	}
	return out
}

// stepStage holds what one step execution stages for tools until the step commits, so the
// memory writes and cached results of a step that fails, times out or is skipped never reach
// the run, and a retry, a fallback or a later step calls the tool again. As the step's
// ToolMemory it reads through to the memory the step runs with; a pipeline session's call
// first looks in the stages of the step and of the steps it runs inside for a cached result.
// It is safe for concurrent use.
type stepStage struct {
	base   ToolMemory
	parent *stepStage
	mu     sync.RWMutex
	staged stagedTools
}

type stepStageKey struct{}

// beginStepStage returns ctx with a new stage for one step execution, inside the stage ctx
// carries. The stage is also the tool memory of ctx when ctx carries one.
func beginStepStage(ctx context.Context) (context.Context, *stepStage) {
	stage := &stepStage{parent: stepStageFrom(ctx)}
	if base := ToolMemoryFrom(ctx); !isNil(base) {
		stage.base = base
		ctx = WithToolMemory(ctx, stage)
	}
	return context.WithValue(ctx, stepStageKey{}, stage), stage
}

func stepStageFrom(ctx context.Context) *stepStage {
	stage, _ := ctx.Value(stepStageKey{}).(*stepStage)
	return stage
}

// Get returns a copy of the value the step wrote under key, or else the base memory's value.
func (m *stepStage) Get(key string) (any, bool) {
	m.mu.RLock()
	value, ok := m.staged.memory[key]
	m.mu.RUnlock()
	if ok {
		return deepCopy(value), true
	}
	if m.base == nil {
		return nil, false
	}
	return m.base.Get(key)
}

// Set stages a copy of value under key.
func (m *stepStage) Set(key string, value any) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.staged.memory == nil {
		m.staged.memory = map[string]any{}
	}
	m.staged.memory[key] = deepCopy(value)
}

// cachedResult returns a copy of the result the stage holds for session under key.
func (m *stepStage) cachedResult(session *ToolSession, key toolCacheKey) (ToolResult, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result, ok := m.staged.cache[stagedCacheKey{session: session, key: key}]
	return result.clone(), ok
}

// storeCached stages a copy of result as session's cached result under key.
func (m *stepStage) storeCached(session *ToolSession, key toolCacheKey, result ToolResult) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.staged.cache == nil {
		m.staged.cache = map[stagedCacheKey]ToolResult{}
	}
	m.staged.cache[stagedCacheKey{session: session, key: key}] = result.clone()
}

// add stages what a child step staged, as if this step had staged it.
func (m *stepStage) add(staged stagedTools) {
	applyMemoryWrites(m, staged.memory)
	for key, result := range staged.cache {
		m.storeCached(key.session, key.key, result)
	}
}

// snapshot returns a copy of what the stage holds.
func (m *stepStage) snapshot() stagedTools {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.staged.clone()
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
		return stepResult{}, failStep(child, values, err)
	}
	return result, err
}

// runNode runs one step on a fresh view of values limited to its declared keys and returns its
// writes when it succeeds, with the tool memory writes and cached tool results it staged;
// those of a step that fails are dropped with its state writes. The candidate is the step's, as
// childCandidate selects it. Each execution of a step other than a WithPolicy or WithCache
// wrapper gets its own usage frame, observer events and outcome record. A panic becomes an
// error with the stack in its details. When the step's context ended because a budget ran out
// or an attempt timed out, that cause replaces the step's result, unless the step's error
// already carries it, so a step that finishes late still fails and its writes are dropped.
//
// commit, when set, checks the result before the execution ends, so an error it returns fails
// the execution and drops what it staged. A wrapper has no execution of its own, so the last
// execution inside it that succeeds is held until commit has checked the result: it fails with
// an error commit marks with failsExecution, and otherwise ends as it did and the failure is
// recorded as an execution that commits nothing.
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
	var hold *finishHold
	if commit != nil && isWrapper(node.step) {
		hold = &finishHold{}
		ctx = context.WithValue(ctx, finishHoldKey{}, hold)
		defer hold.release(nil)
	}
	var scope *stepScope
	if !isWrapper(node.step) {
		var finish func(stepResult, error, *stepScope)
		ctx, finish = beginExecution(ctx, node, values)
		defer func() { finish(result, err, scope) }()
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
	ctx, scope.stage = beginStepStage(ctx)
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
	result = stepResult{writes: scope.pending, skipped: scope.skipped, staged: scope.stage.snapshot()}
	if commit == nil {
		return result, nil
	}
	if err := commit(result); err != nil {
		own, failsOwn := err.(failsExecution)
		if failsOwn {
			err = own.err
		}
		if hold != nil && !(failsOwn && hold.release(err)) {
			hold.release(nil)
			beginStandIn(ctx, commitFailed, node, values)(stepResult{}, err)
		}
		return stepResult{}, err
	}
	return result, nil
}

// failsExecution marks an error of runNode's commit that fails the execution whose result it
// checks, such as a transition's When that panics: inside a wrapper, the last execution that
// succeeded fails with it. Any other commit error, such as a failed merge, fails a wrapper as
// an execution of its own after that one.
type failsExecution struct {
	err error
}

func (e failsExecution) Error() string { return e.err.Error() }

type finishHoldKey struct{}

// finishHold holds the report of the latest successful execution directly inside a top-level
// wrapper, so the execution ends with the pipeline's check of the wrapper's result and a failed
// transition fails it instead of adding an execution. It is safe for concurrent use.
type finishHold struct {
	mu     sync.Mutex
	report func(stepOutcome, stepResult)
	held   stepOutcome
	result stepResult
}

// finishHoldFrom returns the hold of the wrapper that ctx runs directly inside, if any.
func finishHoldFrom(ctx context.Context) *finishHold {
	hold, _ := ctx.Value(finishHoldKey{}).(*finishHold)
	return hold
}

// finish reports an execution's outcome, or holds it when h is not nil and the execution
// succeeded, reporting any execution held before it.
func (h *finishHold) finish(outcome stepOutcome, result stepResult, report func(stepOutcome, stepResult)) {
	if h == nil || outcome.Err != nil {
		report(outcome, result)
		return
	}
	h.release(nil)
	h.mu.Lock()
	defer h.mu.Unlock()
	h.report, h.held, h.result = report, outcome, result
}

// release reports the held execution, failed with err when err is not nil, and returns whether
// one was held.
func (h *finishHold) release(err error) bool {
	h.mu.Lock()
	report, outcome, result := h.report, h.held, h.result
	h.report = nil
	h.mu.Unlock()
	if report == nil {
		return false
	}
	if err != nil {
		outcome.Err, result = err, stepResult{}
	}
	report(outcome, result)
	return true
}

// beginExecution starts one execution of node over values, the state its view reads: it
// numbers the attempt, reports the start to the run's observers, opens a usage frame, which also
// collects the execution's tool calls, and returns a function that records the outcome when the
// execution ends, given the step's scope, which holds the values the step wrote or tried to
// write and the attributes it reported. An execution directly inside a top-level wrapper
// reports a success through the wrapper's hold (see finishHold); the executions inside it
// report their own. An execution that an element of a Map runs itself adds its usage to the
// element's when it ends, whether or not a hold delays its report, so the usage counts once.
func beginExecution(ctx context.Context, node *stepNode, values map[string]any) (
	context.Context,
	func(stepResult, error, *stepScope),
) {
	run := pipelineRunFrom(ctx)
	attempt := run.nextAttempt(node.name)
	capture := evaluationCaptureFrom(ctx)
	hold := finishHoldFrom(ctx)
	if hold != nil {
		ctx = context.WithValue(ctx, finishHoldKey{}, (*finishHold)(nil))
	}
	ctx, observed := observeStep(ctx, node, attempt, visitFrom(ctx), values)
	ctx, element := claimElement(ctx)
	ctx, frame, release := usageFrameFrom(ctx).enterStep(ctx, node.name)
	start := time.Now()
	return ctx, func(result stepResult, err error, scope *stepScope) {
		release()
		attempted := scope.attempted()
		outcome := stepOutcome{
			Step:       node.name,
			Attempt:    attempt,
			Duration:   time.Since(start),
			Usage:      frame.total(),
			Unbudgeted: frame.unbudgetedTotal(),
			Err:        err,
			ToolCalls:  frame.toolCallRecords(),
			Attributes: scope.reported(),
		}
		element.add(outcome.Usage)
		hold.finish(outcome, result, func(outcome stepOutcome, result stepResult) {
			run.record(outcome)
			capture.offerStep(attempted, outcome.Err)
			capture.offerToolCalls(outcome.ToolCalls)
			observed.finish(outcome, result, attempted)
		})
	}
}

// standIn is an execution that does not run its step.
type standIn int

const (
	// cacheHit commits a WithCache step's cached writes.
	cacheHit standIn = iota
	// skippedOnError leaves the writes of a step that skips on error unwritten.
	skippedOnError
	// commitFailed fails a top-level WithPolicy or WithCache step whose writes cannot be merged
	// into the state.
	commitFailed
)

// beginStandIn starts an execution of node over values that does not run the step, and
// returns a function that records it with the writes it committed or its error. Like
// beginExecution, it reports a success directly inside a top-level wrapper through the
// wrapper's hold.
func beginStandIn(ctx context.Context, kind standIn, node *stepNode, values map[string]any) func(stepResult, error) {
	run := pipelineRunFrom(ctx)
	attempt := run.nextAttempt(node.name)
	capture := evaluationCaptureFrom(ctx)
	hold := finishHoldFrom(ctx)
	_, observed := observeStep(ctx, node, attempt, visitFrom(ctx), values)
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
		hold.finish(outcome, result, func(outcome stepOutcome, result stepResult) {
			run.record(outcome)
			capture.offer(result.writes)
			capture.offerStep(nil, outcome.Err)
			observed.finish(outcome, result, nil)
		})
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
