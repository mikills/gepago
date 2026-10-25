package gepa

import (
	"context"
	"errors"
	"maps"
	"reflect"
	"slices"
	"sync"
	"time"
)

// PipelineObserver receives the events of the runs of a Pipeline built with it in
// PipelineSpec.Observers.
//
// Each run has a span whose ID is the RunID of the run's RunContext, and each execution of a
// step has a span whose parent is the span of the run or of the execution around it. Every span
// of a run has the TraceID of the run's RunContext, which the runs nested in it share (see
// RunContext), and every event carries the run's RunContext, with its attributes redacted (see
// Redactor). An execution is one run of a step: each retry attempt and each cache hit is an
// execution of its own, numbered by Attempt. WithPolicy, WithCache and WithAmendable are not
// executions of their own. The executions of the steps they wrap are, and so are the cache hits of
// WithCache. A top-level step's writes merge into the state as part of its execution, so a
// MergeRule that fails fails that execution. For a top-level WithPolicy, WithCache or WithAmendable
// step, that failure adds a failed execution after the one that wrote. The same holds for the
// step's transitions on success, evaluated before its execution ends: a When, Amend or Accept that
// fails or panics fails that execution. Each execution reports the visit of its top-level step in
// Attributes.
//
// The events of a run are ordered:
//   - RunStarted comes before every other event of the run and RunFinished after them. A run
//     that fails before its first step, because the pipeline is not built or its inputs are
//     invalid, emits no events.
//   - Every execution emits StepStarted and later StepFinished with the same span.
//   - A top-level step finishes before the next one starts, and a top-level step a transition
//     skips over emits nothing (see Transition). An execution nested in a composite step starts
//     after the composite starts and finishes before it finishes. Such an execution is a step of
//     a Parallel, an element of a Map or an iteration of a Loop.
//   - The attempts of a step follow each other in order, except for the steps of a Map's body
//     above a MaxConcurrency of one (see below).
//   - With a MaxConcurrency of zero or one, the elements of a Map run in input order, each
//     finishing before the next starts, so the attempts of its body follow each other in order.
//     Above one, its elements start in any order, and the attempts of its body and of the steps
//     nested in it interleave in any order, those of one element still following each other in
//     order. The events of an element, and of the executions nested in it, carry its index in
//     the list under Attributes["index"].
//   - The iterations of a Loop run in order, each finishing before the next starts. The events
//     of an iteration, and of the executions nested in it, carry its number, from one, under
//     Attributes["iteration"].
//   - A pipeline run by a step, through NewProgramStep or a step that runs it with its
//     context, joins the observation of the run: its RunStarted and RunFinished come between
//     that step's StepStarted and StepFinished, and its run span's parent is the step's span.
//     Its events go to its own observers and then to those of every run around it. A run that
//     has a Redactor but no observers, of its own or of a run around it, is observed with no
//     observers, so the pipelines nested in it redact their events with its Redactor too (see
//     Redactor); their run spans then have the span of a step no observer saw as their parent.
//
// Under Parallel and Map the events of steps that run side by side interleave, but the observer
// and redactor calls of one observed run, the runs nested in it included, are serialised: no two
// of them run at once, and every observer of the run sees its events in the same order. Calls
// are synchronous, so an observer that blocks stalls the run, and it must not wait for another
// step of the same run. Separate top-level runs call a shared observer concurrently, even when
// they share a TraceID because their callers gave them the same RunContext, so an observer of a
// pipeline that runs concurrently must be safe for concurrent use.
//
// Observers are called in the order of PipelineSpec.Observers. An observer of a comparable
// type, such as a pointer, is called once for each event even when it is listed more than
// once, or by both a pipeline and a pipeline nested in it. An observer that cannot be compared,
// such as a struct value with a func, map or slice field, is called once for each listing, so
// prefer pointer observers. An observer that panics is recovered: the run continues, the other
// observers still receive the event, and the observer keeps receiving later events.
//
// ctx carries the values of the context of the run or step the event belongs to. It ends when
// the context the caller passed to the trace's top-level run ends, when the MaxDuration of the
// event's run or of a run around it expires, or once the event's run has finished, but not when
// a step's timeout, a token or model call limit or a failed Parallel sibling ends the step's
// own context. It carries neither the run's observation, RunContext, budget, usage collection,
// tool session nor tool memory, so an observer's model calls, tool calls and pipeline runs are
// independent of the observed run: they spend none of its budget or tool call limits, do not
// count in its usage or tool records, are not logged under its RunContext, and a pipeline run
// with ctx starts a trace of its own. An observer that wants its work correlated with the run
// can pass the event's RunContext to WithRunContext.
//
// ctx does not carry where the event's execution runs, such as its Map element or its Loop
// iteration. So a run that an observer starts takes none of the iteration numbers of the Loop it
// observes. ctx keeps the other values of the context, such as the tool scope.
//
// The values an event carries pass through PipelineSpec.Redactor; see Redactor. Each observer
// receives its own copy of them, its RunContext's Attributes included, so changing them changes
// neither the run nor what other observers receive.
//
// An observer that also implements TransitionObserver receives the run's transitions too.
type PipelineObserver interface {
	RunStarted(ctx context.Context, event RunStartedEvent)
	StepStarted(ctx context.Context, event StepStartedEvent)
	StepFinished(ctx context.Context, event StepFinishedEvent)
	RunFinished(ctx context.Context, event RunFinishedEvent)
}

// TransitionObserver is a PipelineObserver that also receives the transitions of the runs it
// observes (see Transition), with the same ordering, serialisation, context and panic rules as
// the other events: TransitionTaken follows the StepFinished of the last execution of the step
// the transition leaves, after any RewindRejected for it, and comes before the next StepStarted
// of the run. RecordingObserver is one.
type TransitionObserver interface {
	PipelineObserver
	TransitionTaken(ctx context.Context, event TransitionTakenEvent)
	RewindRejected(ctx context.Context, event RewindRejectedEvent)
}

// TransitionTakenEvent reports that a transition matched after its From step finished.
type TransitionTakenEvent struct {
	// RunContext is the RunContext of the run, as RunStarted reported it.
	RunContext RunContext
	// From and To name the steps the run moves between; To is End or Fail for those.
	From string
	To   string
	// Rule is the transition's index in PipelineSpec.Transitions.
	Rule int
	// Rewind reports a transition to From itself or an earlier step.
	Rewind bool
}

// RewindRejectedEvent reports that a rewind's Accept kept the visit before it: the values the
// rewind's range committed before it are restored and the run goes on past Step.
type RewindRejectedEvent struct {
	// RunContext is the RunContext of the run, as RunStarted reported it.
	RunContext RunContext
	// Step is the From step of the rejected rewind.
	Step string
	// Rule is the rewind's index in PipelineSpec.Transitions.
	Rule int
}

// RunStartedEvent reports the start of a pipeline run.
type RunStartedEvent struct {
	// Span is the run's span, of kind UsageSpanPipelineRun, whose ID is RunContext.RunID and
	// TraceID RunContext.TraceID. ParentID is the span of the step that started a nested run,
	// and empty for a top-level run or one nested in a run that has neither observers nor a
	// Redactor; see PipelineObserver.
	Span UsageSpan
	// RunContext is the run's, with its Attributes redacted.
	RunContext RunContext
	// Inputs holds the run's inputs after redaction.
	Inputs map[string]any
}

// RunFinishedEvent reports the end of a pipeline run.
type RunFinishedEvent struct {
	// Span is the run's finished span; its Duration, Usage and Unbudgeted are the event's, and
	// its Err is the message of the event's Err.
	Span UsageSpan
	// RunContext is the run's, as RunStarted reported it.
	RunContext RunContext
	Duration   time.Duration
	// Usage is what the model calls of the run recorded, nested steps and pipelines included.
	Usage Usage
	// Unbudgeted is the usage the run recorded with RecordUnbudgetedUsage, such as its tool
	// calls' ToolResult.UnbudgetedUsage, nested steps and pipelines included. Usage does not
	// include it, and no budget counts it.
	Unbudgeted Usage
	// Err is the run's error, a *StepError without a Redactor; see Redactor.
	Err error
}

// StepStartedEvent reports the start of one execution of a step.
type StepStartedEvent struct {
	// Span is the execution's span, of kind UsageSpanPipelineStep and named after the step.
	// ParentID is the span of the enclosing execution, or of the run for a top-level step.
	Span UsageSpan
	// RunContext is the RunContext of the run the step belongs to, as RunStarted reported it.
	RunContext RunContext
	Step       string
	// Attempt counts the step's executions in the run from one, cache hits included.
	Attempt int
	// Reads holds the values of the step's declared reads after redaction.
	Reads map[string]any
	// Lane names the step of a Parallel the execution belongs to when the execution is that step
	// or one its step policy runs, such as a retry; it is empty otherwise, for the executions
	// nested in them too. The executions of different lanes of a
	// Parallel run side by side, so they may start in any order.
	Lane string
	// Attributes describe where the execution runs. "visit" numbers, from one, the visits of the
	// top-level step it belongs to: a rewind visits a step again (see Transition). "index" is set
	// for the executions of an element of a Map, those nested in them included, to the element's
	// index in the list, from zero; above a MaxConcurrency of one, elements start in any order.
	// "iteration" gives the executions of an iteration of a Loop, those nested in them included,
	// the number of the iteration, from one. Where Loops nest, the nearest Loop gives it. The
	// event leaves it out when the run's Redactors change it under the Loop's iteration key.
	Attributes map[string]int
}

// StepFinishedEvent reports the end of one execution of a step.
type StepFinishedEvent struct {
	// Span is the execution's finished span; its Duration, Usage and Unbudgeted are the event's,
	// and its Err is the message of the event's Err.
	Span UsageSpan
	// RunContext is the RunContext of the run the step belongs to, as RunStarted reported it.
	RunContext RunContext
	Step       string
	Attempt    int
	Duration   time.Duration
	// Usage is what the execution's model calls recorded, nested steps included.
	Usage Usage
	// Unbudgeted is the usage the execution recorded with RecordUnbudgetedUsage, nested steps
	// included. Usage does not include it, and no budget counts it.
	Unbudgeted Usage
	// Err is the step's own error, before any enclosing step wraps it; see Redactor.
	Err error
	// Details holds the diagnostics the step attached to Err with WithStepDetails, after
	// redaction.
	Details map[string]any
	// Reads holds the values of the step's declared reads after redaction, as StepStarted
	// reported them.
	Reads map[string]any
	// Writes holds the values the execution wrote after redaction, before any MergeRule
	// combines them with earlier values. A failed execution has none.
	Writes map[string]any
	// Cached reports a WithCache hit: the step did not run and its cached writes were committed.
	Cached bool
	// Attributes are those StepStarted reported. A Map or MapCollect adds "failed", the number of
	// its elements that failed. A Loop adds "iterations", the number of its iterations that ran,
	// the failed one included. Only the nearest execution with the Map's or Loop's name carries the
	// count. That is its own execution, or else that of a step with its name that runs it. A
	// decorator that embeds the Map or Loop and keeps its name is such a step. The event leaves
	// "iterations" out when the run's Redactors change it under the Loop's iteration key.
	Attributes map[string]int
	// ToolCalls holds the tool calls the execution made, through the run's ToolSession or any
	// other, such as an agent's, in the order they finished, after redaction (see Redactor). The
	// calls of the executions nested in it are in their own events. Each call's usage is part of
	// Usage already and its unbudgeted usage part of Unbudgeted, and a call answered from a
	// ToolPolicy cache is marked Cached and has neither.
	ToolCalls []ToolCallRecord
}

type observeKey struct{}

// observeScope is what an observed run's context carries: the run and the span of the
// innermost execution, which becomes the parent of the next span. While ProgramEvaluator
// records the trace of a run that nothing else observes, recording and recordingSpan observe
// it for the evaluation's RecordingObserver, and run, the run as its observers see it outside
// an evaluation, is nil until a nested pipeline with observers of its own starts a trace.
type observeScope struct {
	run           *runObservation
	span          string
	recording     *runObservation
	recordingSpan string
	// lane is the step of a Parallel the executions started in the scope belong to, or empty.
	lane string
}

func observeScopeFrom(ctx context.Context) *observeScope {
	scope, _ := ctx.Value(observeKey{}).(*observeScope)
	return scope
}

// withLane returns ctx for the step named lane of a Parallel, so that the StepStarted events of
// its executions report Lane. The executions nested in them do not inherit it.
func withLane(ctx context.Context, lane string) context.Context {
	scope := observeScopeFrom(ctx)
	if scope == nil {
		return ctx
	}
	laned := *scope
	laned.lane = lane
	return context.WithValue(ctx, observeKey{}, &laned)
}

// observerContext returns ctx, a context of the run, as observers receive it. It keeps the
// values of ctx, but it ends only when the run's life ends (see runObservation). It leaves out the
// run that ctx belongs to, so nothing an observer does with it counts against the run. numbers
// are the traced numbers of the run or the execution that the event belongs to (see
// tracedNumbersKey).
func (r *runObservation) observerContext(ctx context.Context, numbers iterationNumbers) context.Context {
	return observerCtx{Context: r.life, values: context.WithoutCancel(ctx), numbers: numbers}
}

// observerCtx takes its deadline and cancellation from a run's life and its values from a
// run's context. It hides the values that tie it to a pipeline run. These are the trace, the
// step frame, the run context, the budget, usage collection and the step cache. They are also
// evaluation capture, the tool session, tool memory, staged tool results, and the tool call
// and usage tally of a wrapped tool. It also hides where the execution runs. That is the visit
// of its top-level step, its Map element, and its Loop iteration with the iteration numbers. It
// also hides the hold of a top-level wrapper. So a run that an observer starts with it is a run
// of its own. Under tracedNumbersKey it holds numbers, the traced numbers of its event's run or
// execution, or nil. So it never passes on the numbers of another event. It keeps the other
// values of the context, such as the tool scope.
type observerCtx struct {
	context.Context
	values  context.Context
	numbers iterationNumbers
}

func (c observerCtx) Value(key any) any {
	switch key.(type) {
	case tracedNumbersKey:
		if len(c.numbers) == 0 {
			return nil
		}
		return c.numbers
	case observeKey, evaluationKey, pipelineRunKey, runContextKey, usageFrameKey, usageCollectorKey, retryAdmissionKey,
		budgetClockKey, toolEnvKey, toolMemoryKey, stepStageKey, toolAttemptsKey, recordedUsageKey, visitKey,
		mapElementKey, loopIterationKey, finishHoldKey:
		return nil
	}
	return c.values.Value(key)
}

// appendObservers appends to list each observer it does not already hold.
func appendObservers(list []PipelineObserver, observers ...PipelineObserver) []PipelineObserver {
	for _, observer := range observers {
		if !slices.ContainsFunc(list, func(listed PipelineObserver) bool { return sameObserver(listed, observer) }) {
			list = append(list, observer)
		}
	}
	return list
}

// sameObserver reports whether a and b are the same observer: equal values of one comparable
// type, such as the same pointer.
func sameObserver(a, b PipelineObserver) (same bool) {
	defer func() {
		if recover() != nil {
			same = false
		}
	}()
	return reflect.TypeOf(a) == reflect.TypeOf(b) && reflect.TypeOf(a).Comparable() && a == b
}

// traceObservation is shared by the runs of one observation: a top-level observed run and the
// runs nested in it. Its lock serialises their observer and redactor calls and guards the text
// it removes from errors; see Redactor. An observation started inside a recorded one shares
// that observation's lock.
type traceObservation struct {
	mu *sync.Mutex
	redactedText
}

// runObservation reports one run to its observers. These are its own observers and those of the
// runs around it. Its own redactor comes first, and then theirs. runContext is the run's
// RunContext as its events carry it, with the attributes redacted. numbers are the run's traced
// numbers (see tracedNumbersKey). observeRun sets both when it emits RunStarted. life ends when
// the context that the caller passed to the observation's top-level run ends. It also ends when
// the MaxDuration of the run or of a run around it expires. At the latest, endLife ends it after
// the run has finished. recording, when set, reports the same run to the RecordingObserver of an
// evaluation that records traces, in an observation of its own (see observeScope).
type runObservation struct {
	trace      *traceObservation
	observers  []PipelineObserver
	redactors  []Redactor
	span       UsageSpan
	runContext RunContext
	numbers    iterationNumbers
	start      time.Time
	life       context.Context
	endLife    context.CancelFunc
	recording  *runObservation
}

// observeRun starts observing a run of p over its converted inputs, with rc as its RunContext,
// as part of the observation in ctx if there is one, and emits RunStarted. caller is the context
// the caller passed to the run. It returns ctx carrying the run, and nil when the run is not
// observed, not even silently (see startRun).
//
// A top-level run of an example that ProgramEvaluator records traces for reports to the
// evaluation's RecordingObserver as well as to its own observers. When it has no observers of
// its own, the recorder observes it apart from its observers' observation: the runs inside it
// report to the recorder as part of the recorded trace and, separately, to their own observers
// as they would outside an evaluation, which for a run with a Redactor means within its silent
// observation.
func (p Pipeline) observeRun(ctx, caller context.Context, inputs map[string]any, rc RunContext) (
	context.Context,
	*runObservation,
) {
	parent := observeScopeFrom(ctx)
	capture := evaluationCaptureFrom(ctx)
	recorder := capture.join(p, parent == nil)
	numbers := iterationFrom(ctx).numbers
	capture.offer(numbers.ordinary(inputs))
	capture.offer(attributeValues(rc))
	if parent == nil {
		parent = &observeScope{}
	}
	var run, recording *runObservation
	switch {
	case recorder != nil && len(p.observers) == 0:
		recording = p.startRun(nil, "", caller, []PipelineObserver{recorder}, nil, rc)
		run = p.startRun(nil, "", caller, nil, recording.trace.mu, rc)
	case recorder != nil:
		run = p.startRun(nil, "", caller, append(slices.Clone(p.observers), recorder), nil, rc)
	default:
		var lock *sync.Mutex
		if parent.recording != nil {
			recording = p.startRun(parent.recording, parent.recordingSpan, caller, nil, nil, rc)
			lock = recording.trace.mu
		}
		run = p.startRun(parent.run, parent.span, caller, p.observers, lock, rc)
	}
	scope := &observeScope{}
	if recording != nil {
		scope.recording, scope.recordingSpan = recording, recording.span.ID
	}
	if run != nil {
		scope.run, scope.span = run, run.span.ID
		run.recording = recording
	} else {
		run = recording
	}
	if run == nil {
		return ctx, nil
	}
	ctx = context.WithValue(ctx, observeKey{}, scope)

	run.trace.mu.Lock()
	defer run.trace.mu.Unlock()
	for lane := run; lane != nil; lane = lane.recording {
		lane.runContext = lane.redactRunContext(rc)
		event := RunStartedEvent{
			Span:       lane.span,
			RunContext: lane.runContext,
			Inputs:     lane.redactStateValues(inputs, numbers),
		}
		lane.numbers = numbers.inputs(inputs)
		observerCtx := lane.observerContext(ctx, lane.numbers)
		lane.deliver(func(observer PipelineObserver) {
			event := event
			event.RunContext = event.RunContext.clone()
			event.Inputs = copyValues(event.Inputs)
			observer.RunStarted(observerCtx, event)
		})
	}
	return ctx, run
}

// startRun returns the observation of a run of p with the RunContext rc, with observers added
// to those of parent: in parent's observation, under the span parentSpan, when parent is set,
// and otherwise in a new observation serialised by lock, or by a lock of its own when lock is
// nil. The run's span has rc's RunID as its ID and rc's TraceID. It returns nil when there is
// neither a parent, an observer nor a Redactor. A run with a Redactor and nothing else is
// observed silently, by no observer, so that the runs nested in it, which join its
// observation, redact their events with its Redactor and the text it removed (see Redactor).
// caller is the context the caller passed to the run.
func (p Pipeline) startRun(
	parent *runObservation,
	parentSpan string,
	caller context.Context,
	observers []PipelineObserver,
	lock *sync.Mutex,
	rc RunContext,
) *runObservation {
	if parent == nil && len(observers) == 0 && p.redactor == nil {
		return nil
	}
	run := &runObservation{observers: appendObservers(nil, observers...), start: time.Now(), life: caller}
	if p.redactor != nil {
		run.redactors = []Redactor{p.redactor}
	}
	if parent != nil {
		run.trace = parent.trace
		run.observers = appendObservers(run.observers, parent.observers...)
		run.redactors = append(run.redactors, parent.redactors...)
		run.life = parent.life
	} else {
		if lock == nil {
			lock = &sync.Mutex{}
		}
		run.trace = &traceObservation{mu: lock}
	}
	if limit := p.budget.MaxDuration; limit > 0 {
		run.life, run.endLife = context.WithDeadline(run.life, run.start.Add(limit))
	} else {
		run.life, run.endLife = context.WithCancel(run.life)
	}
	run.span = NewUsageSpan(rc.TraceID, parentSpan, UsageSpanPipelineRun, "pipeline")
	run.span.ID = rc.RunID
	return run
}

// finish emits RunFinished with err, the run's error. ctx is the run's context.
func (r *runObservation) finish(ctx context.Context, err error) {
	if r == nil {
		return
	}
	duration := time.Since(r.start)
	frame := usageFrameFrom(ctx)
	usage, unbudgeted := frame.total(), frame.unbudgetedTotal()
	r.trace.mu.Lock()
	defer r.trace.mu.Unlock()
	for lane := r; lane != nil; lane = lane.recording {
		laneErr := lane.redactError(err)
		event := RunFinishedEvent{
			Span:       finishSpan(lane.span, duration, usage, unbudgeted, laneErr),
			RunContext: lane.runContext,
			Duration:   duration,
			Usage:      usage,
			Unbudgeted: unbudgeted,
			Err:        laneErr,
		}
		observerCtx := lane.observerContext(ctx, lane.numbers)
		lane.deliver(func(observer PipelineObserver) {
			event := event
			event.RunContext = event.RunContext.clone()
			observer.RunFinished(observerCtx, event)
		})
		lane.endLife()
	}
}

// stepObservation reports one execution of a step. recording, when set, reports it to an
// evaluation's RecordingObserver (see runObservation). numbers are the execution's traced numbers
// (see tracedNumbersKey).
type stepObservation struct {
	run        *runObservation
	ctx        context.Context
	span       UsageSpan
	step       string
	reads      map[string]any
	attributes map[string]int
	numbers    iterationNumbers
	recording  *stepObservation
}

// observeStep emits StepStarted for an execution of node over values, the state its view
// reads, when the run in ctx is observed. visit is the visit of the top-level step the execution
// belongs to. The events of a Map's executions, and of those nested in them, hide its elements'
// items and outputs (see hideElements). The "iteration" attribute of an execution in a Loop's
// iteration passes through the redactors under the Loop's iteration key. So does the "iterations"
// attribute of a Loop. The events leave out one that the redactors change. It returns ctx carrying
// the execution's span, and nil when nothing observes the run.
func observeStep(ctx context.Context, node *stepNode, attempt, visit int, values map[string]any) (
	context.Context,
	*stepObservation,
) {
	scope := observeScopeFrom(ctx)
	if scope == nil {
		return ctx, nil
	}
	next := &observeScope{}
	var lanes []*stepObservation
	if scope.run != nil {
		run := hideElements(node, scope.run)
		span := NewUsageSpan(run.span.TraceID, scope.span, UsageSpanPipelineStep, node.name)
		next.run, next.span = run, span.ID
		lanes = append(lanes, &stepObservation{run: run, span: span, step: node.name})
	}
	if scope.recording != nil {
		recording := hideElements(node, scope.recording)
		span := NewUsageSpan(recording.span.TraceID, scope.recordingSpan, UsageSpanPipelineStep, node.name)
		next.recording, next.recordingSpan = recording, span.ID
		lanes = append(lanes, &stepObservation{run: recording, span: span, step: node.name})
	}
	if len(lanes) == 0 {
		return ctx, nil
	}
	ctx = context.WithValue(ctx, observeKey{}, next)
	reads := make(map[string]any, len(node.reads))
	for _, key := range node.reads {
		if value, ok := values[key.Name]; ok {
			reads[key.Name] = value
		}
	}

	observed := lanes[0]
	observed.run.trace.mu.Lock()
	defer observed.run.trace.mu.Unlock()
	for i, lane := range lanes {
		if i > 0 {
			lanes[i-1].recording = lane
		}
		lane.ctx = ctx
		iteration := iterationFrom(ctx)
		lane.reads = lane.run.redactStateValues(reads, iteration.numbers)
		lane.numbers = iteration.numbers.inputs(reads)
		lane.attributes = executionAttributes(ctx, visit)
		if iteration.number > 0 && !lane.run.keepsAttribute(iteration.key, iteration.number) {
			delete(lane.attributes, "iteration")
		}
		event := StepStartedEvent{
			Span:       lane.span,
			RunContext: lane.run.runContext,
			Step:       node.name,
			Attempt:    attempt,
			Reads:      lane.reads,
			Lane:       scope.lane,
			Attributes: lane.attributes,
		}
		observerCtx := lane.run.observerContext(ctx, lane.numbers)
		lane.run.deliver(func(observer PipelineObserver) {
			event := event
			event.RunContext = event.RunContext.clone()
			event.Reads = copyValues(event.Reads)
			event.Attributes = maps.Clone(event.Attributes)
			observer.StepStarted(observerCtx, event)
		})
	}
	return ctx, observed
}

// finish emits StepFinished for the execution that ended with outcome, writing result's
// writes when it succeeded. When it failed, attempted holds the values the step wrote or tried
// to write: they leave in no event, but pass through the redactors so that their text is
// removed from the error. The execution's tool calls are redacted first, so that their text is
// removed from everything else the event carries.
func (s *stepObservation) finish(outcome stepOutcome, result stepResult, attempted []stateWrite) {
	if s == nil {
		return
	}
	var writes map[string]any
	if outcome.Err == nil {
		writes = result.writes
	}
	var details map[string]any
	var detailed *stepDetailsError
	if errors.As(outcome.Err, &detailed) {
		details = detailed.details
	}

	s.run.trace.mu.Lock()
	defer s.run.trace.mu.Unlock()
	for lane := s; lane != nil; lane = lane.recording {
		run := lane.run
		if outcome.Err != nil {
			for _, write := range attempted {
				run.redact(write.key, write.value)
			}
		}
		toolCalls := run.redactToolCalls(outcome.ToolCalls)
		attributes := lane.attributes
		if len(outcome.Attributes) > 0 {
			attributes = maps.Clone(attributes)
			maps.Copy(attributes, outcome.Attributes)
			if count, ok := outcome.Attributes["iterations"]; ok && !run.keepsAttribute(outcome.loopKey, count) {
				delete(attributes, "iterations")
			}
		}
		event := StepFinishedEvent{
			RunContext: run.runContext,
			Step:       lane.step,
			Attempt:    outcome.Attempt,
			Duration:   outcome.Duration,
			Usage:      outcome.Usage,
			Unbudgeted: outcome.Unbudgeted,
			Writes:     run.redactValues(writes),
			Details:    run.redactDetails(details),
			Reads:      lane.reads,
			Cached:     outcome.Cached,
			Attributes: attributes,
		}
		event.ToolCalls = run.scrubToolCalls(toolCalls)
		event.Err = run.redactError(outcome.Err)
		event.Span = finishSpan(lane.span, outcome.Duration, outcome.Usage, outcome.Unbudgeted, event.Err)
		observerCtx := run.observerContext(lane.ctx, lane.numbers)
		run.deliver(func(observer PipelineObserver) {
			event := event
			event.RunContext = event.RunContext.clone()
			event.Reads = copyValues(event.Reads)
			event.Writes = copyValues(event.Writes)
			event.Details = copyValues(event.Details)
			event.ToolCalls = cloneRecords(event.ToolCalls)
			event.Attributes = maps.Clone(event.Attributes)
			observer.StepFinished(observerCtx, event)
		})
	}
}

// transitionTaken emits TransitionTaken to the observers that are TransitionObservers. ctx is
// the run's context.
func (r *runObservation) transitionTaken(ctx context.Context, event TransitionTakenEvent) {
	r.deliverTransition(ctx, func(observer TransitionObserver, rc RunContext, observerCtx context.Context) {
		event := event
		event.RunContext = rc
		observer.TransitionTaken(observerCtx, event)
	})
}

// rewindRejected emits RewindRejected to the observers that are TransitionObservers. ctx is
// the run's context.
func (r *runObservation) rewindRejected(ctx context.Context, event RewindRejectedEvent) {
	r.deliverTransition(ctx, func(observer TransitionObserver, rc RunContext, observerCtx context.Context) {
		event := event
		event.RunContext = rc
		observer.RewindRejected(observerCtx, event)
	})
}

// deliverTransition calls call for each TransitionObserver of the run and of its recording, with
// a copy of the run's RunContext as its events carry it.
func (r *runObservation) deliverTransition(
	ctx context.Context,
	call func(TransitionObserver, RunContext, context.Context),
) {
	if r == nil {
		return
	}
	r.trace.mu.Lock()
	defer r.trace.mu.Unlock()
	for lane := r; lane != nil; lane = lane.recording {
		observerCtx := lane.observerContext(ctx, lane.numbers)
		lane.deliver(func(observer PipelineObserver) {
			if transitions, ok := observer.(TransitionObserver); ok {
				call(transitions, lane.runContext.clone(), observerCtx)
			}
		})
	}
}

// deliver calls each observer, recovering any panic so it neither fails the run nor keeps the
// event from the other observers. The caller holds the trace lock.
func (r *runObservation) deliver(call func(PipelineObserver)) {
	for _, observer := range r.observers {
		func() {
			defer func() { _ = recover() }()
			call(observer)
		}()
	}
}

func finishSpan(span UsageSpan, duration time.Duration, usage, unbudgeted Usage, err error) UsageSpan {
	span.EndedAt = span.StartedAt.Add(duration)
	span.Duration = duration
	span.Usage = usage
	span.Unbudgeted = unbudgeted
	span.Err = errorString(err)
	return span
}

// copyValues returns a copy of values for one observer.
func copyValues(values map[string]any) map[string]any {
	if values == nil {
		return nil
	}
	out := make(map[string]any, len(values))
	for key, value := range values {
		out[key] = deepCopy(value)
	}
	return out
}
