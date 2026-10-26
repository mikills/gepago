package gepa

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"sync"
	"time"
)

type pipelineRunKey struct{}

// pipelineRun is what the steps of one Pipeline run share: the step cache, the outcome of every
// step execution and the run's tool session, which records every tool call. It is safe for
// concurrent use, and a nil run, outside a pipeline, records and caches nothing.
//
// replaced, while a rewind's range runs again, maps each cache entry stored since the rewind
// began to the entry it replaced, nil for none, so a rejected rewind can take them back.
type pipelineRun struct {
	optional map[string]map[string]bool
	tools    *ToolSession
	mu       sync.Mutex
	cache    map[stepCacheKey]stepResult
	replaced map[stepCacheKey]*stepResult
	attempts map[string]int
	outcomes []stepOutcome
}

// stepCacheKey identifies one cached step result: the step and the key WithCache computed from
// its reads.
type stepCacheKey struct {
	step string
	key  string
}

// stepOutcome records one execution of a step, top-level or nested: each retry attempt and
// each cache hit is an execution of its own. Attempt counts the step's
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
	Err        error
	ToolCalls  []ToolCallRecord
	Attributes map[string]int
	// loopKey is the iteration key under which the run's Redactors see the "iterations"
	// attribute.
	loopKey string
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

// storeCached caches result, a copy that nothing else holds, for step under key.
func (r *pipelineRun) storeCached(step, key string, result stepResult) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	entry := stepCacheKey{step: step, key: key}
	if _, recorded := r.replaced[entry]; r.replaced != nil && !recorded {
		var prev *stepResult
		if old, ok := r.cache[entry]; ok {
			prev = &old
		}
		r.replaced[entry] = prev
	}
	r.cache[entry] = result
}

// startCacheJournal records the cache entries stored from now on, until endCacheJournal.
func (r *pipelineRun) startCacheJournal() {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.replaced = map[stepCacheKey]*stepResult{}
}

// endCacheJournal stops recording the cache entries stored and, unless keep is set, takes back
// those recorded, putting back the entries they replaced.
func (r *pipelineRun) endCacheJournal(keep bool) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	replaced := r.replaced
	r.replaced = nil
	if keep {
		return
	}
	for entry, prev := range replaced {
		if prev == nil {
			delete(r.cache, entry)
		} else {
			r.cache[entry] = *prev
		}
	}
}

// stepResult holds a successful step's pending writes and the tool results it staged.
// The preserved set holds unchanged outputs. Those keys do not enter merges or replace defaults.
type stepResult struct {
	writes    map[string]any
	preserved map[string]bool
	staged    stagedTools
}

func (r stepResult) clone() stepResult {
	out := stepResult{
		writes:    make(map[string]any, len(r.writes)),
		preserved: maps.Clone(r.preserved),
		staged:    r.staged.clone(),
	}
	for key, value := range r.writes {
		out.writes[key] = deepCopy(value)
	}
	return out
}

// commitTo copies a child's result into the view of the composite step that ran it, and what
// it staged for tools into the composite's stage.
func (r stepResult) commitTo(s State, child *stepNode) error {
	for key, value := range r.writes {
		s.store(key, child.writeTypes[key], value, true)
	}
	for key := range r.preserved {
		if err := s.preserve(key, child.writeTypes[key]); err != nil {
			return err
		}
	}
	if s.scope != nil && s.scope.stage != nil {
		s.scope.stage.add(r.staged)
	}
	return nil
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
// memory writes and cached results of a step that fails or times out never reach the run, and
// a retry or a later step calls the tool again. As the step's
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
// the composite's candidate namespace. runChild wraps the child's error in a StepError naming it.
// It does not when a WithPolicy, WithCache or WithAmendable wrapper wraps the child, which shares
// the wrapper's name.
func runChild(ctx context.Context, candidate Candidate, parent string, child *stepNode, values map[string]any) (
	stepResult,
	error,
) {
	result, err := runNode(ctx, childCandidate(candidate, child), child, values, nil)
	if err != nil && child.name != parent {
		return stepResult{}, failStep(ctx, child, values, err)
	}
	return result, err
}

// nameWrapped returns err, the error of child, the step that a wrapper ran on s. It returns it as a
// StepError naming child when no execution with child's name will name it. That is so when a caller
// runs the wrapper outside such an execution. A pipeline names the wrapped step through the
// wrapper's execution, which shares its name.
func nameWrapped(ctx context.Context, s State, child *stepNode, err error) error {
	if err == nil || s.scope.named(child.name) {
		return err
	}
	return failStep(ctx, child, s.visibleValues(child.reads), err)
}

// nameItself returns err, the error of a composite step named name and reading reads that ran on s.
// It returns it as a StepError naming the step when no execution with that name will name it. That
// is so when a caller runs the composite outside such an execution.
func nameItself(ctx context.Context, name string, reads []KeyRef, s State, err error) error {
	if err == nil || s.scope.named(name) {
		return err
	}
	return failStep(ctx, &stepNode{name: name, reads: reads}, s.visibleValues(reads), err)
}

// runNode runs one step on a fresh view of values. The view holds only the step's declared keys.
// When the step succeeds, runNode returns its writes with the tool memory writes and cached tool
// results it staged. When the step fails, runNode drops what it staged with its state writes. The
// candidate is the step's, as childCandidate selects it. Each execution of a step other than a
// WithPolicy, WithCache or WithAmendable wrapper gets its own usage frame, observer events and
// outcome record. A panic becomes an error with the stack in its details. A budget that runs out
// or an attempt that times out can end the step's context. That cause then replaces the step's
// result, unless the step's error already carries it. So a step that finishes late still fails,
// and runNode drops its writes.
//
// commit, when set, checks the result before the execution ends. An error that it returns fails
// the execution and drops what the execution staged. A wrapper has no execution of its own. So
// runNode holds the last execution inside the wrapper that succeeds until commit has checked the
// result. That execution fails with an error that commit marks with failsExecution. With any
// other error it ends as it did, but without the work that waits for its result to commit.
// runNode then records the failure as an execution that commits nothing.
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
			err = recoveredPanic(recovered, "step panicked")
		}
	}()
	if err := stepCanStart(ctx); err != nil {
		return stepResult{}, err
	}
	scope = &stepScope{
		reads:   node.readTypes,
		writes:  node.writeTypes,
		pending: map[string]any{},
		step:    node.step,
		name:    node.name,
	}
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
		if _, ok := scope.pending[key]; !ok && !scope.preserved[key] {
			return stepResult{}, fmt.Errorf("step did not write declared state key %q", key)
		}
	}
	result = stepResult{writes: scope.pending, preserved: scope.preserved, staged: scope.stage.snapshot()}
	if commit == nil {
		return result, nil
	}
	if err := commit(result); err != nil {
		own, failsOwn := err.(failsExecution)
		if failsOwn {
			err = own.err
		}
		if hold != nil && !(failsOwn && hold.release(err)) {
			// the held execution succeeded, but its result does not commit, so WithCache does not
			// store it.
			hold.releaseUncommitted()
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
// transition fails it instead of adding an execution. succeeded holds what waits for the held
// execution to end as a success whose result commits, such as WithCache storing its result. It
// is safe for concurrent use.
type finishHold struct {
	mu        sync.Mutex
	report    func(stepOutcome, stepResult)
	held      stepOutcome
	result    stepResult
	succeeded []func()
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
// one was held. What waits for the execution to succeed runs when it ends as a success and is
// dropped otherwise.
func (h *finishHold) release(err error) bool {
	return h.end(err, err == nil)
}

// releaseUncommitted reports the held execution as the success it was, but drops what waits for
// it to succeed, since the pipeline could not commit its result.
func (h *finishHold) releaseUncommitted() {
	h.end(nil, false)
}

// end reports the held execution, failed with err when err is not nil, runs what waits for it to
// succeed when commits is set and drops it otherwise, and returns whether one was held.
func (h *finishHold) end(err error, commits bool) bool {
	h.mu.Lock()
	report, outcome, result, succeeded := h.report, h.held, h.result, h.succeeded
	h.report, h.succeeded = nil, nil
	h.mu.Unlock()
	if report == nil {
		return false
	}
	if err != nil {
		outcome.Err, result = err, stepResult{}
	}
	report(outcome, result)
	if commits {
		for _, do := range succeeded {
			do()
		}
	}
	return true
}

// whenSucceeded runs do once the execution that h holds ends as a success whose result commits,
// or at once when h is nil, as outside a top-level wrapper, where an execution that succeeded
// has ended.
func (h *finishHold) whenSucceeded(do func()) {
	if h == nil {
		do()
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	h.succeeded = append(h.succeeded, do)
}

// beginExecution starts one execution of node over values, the state its view reads: it
// numbers the attempt, reports the start to the run's observers, opens a usage frame, which also
// collects the execution's tool calls, and returns a function that records the outcome when the
// execution ends, given the step's scope, which holds the values the step wrote or tried to
// write and the attributes it reported. An execution directly inside a top-level wrapper
// reports a success through the wrapper's hold (see finishHold); the executions inside it
// report their own. An execution that an element of a Map runs itself adds its usage, budgeted
// and unbudgeted, to the element's when it ends, whether or not a hold delays its report, so
// the usage counts once.
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
			loopKey:    scope.iterationKey(),
		}
		element.add(outcome.Usage.Add(outcome.Unbudgeted))
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
	// commitFailed fails a top-level WithPolicy, WithCache or WithAmendable step whose writes the
	// pipeline cannot merge into the state.
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
	if ctx.Err() != nil {
		return contextEnd(ctx)
	}
	return usageFrameFrom(ctx).check()
}

// contextEnd returns the error a step reports because ctx has ended: the cause the pipeline
// reports in place of a step's own error (see stopCause), or else ctx.Err().
func contextEnd(ctx context.Context) error {
	if cause := stopCause(ctx); cause != nil {
		return cause
	}
	return ctx.Err()
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
