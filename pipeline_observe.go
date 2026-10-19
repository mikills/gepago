package gepa

import (
	"context"
	"errors"
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
// Redactor). An execution is one run of a step: each retry attempt, each fallback step, each
// cache hit and each skip on error is an execution of its own, numbered by Attempt. WithPolicy
// and WithCache are not executions of their own; their attempts, fallbacks, cache hits and skips
// are. A top-level step's writes are merged into the state as part of its execution, so a
// MergeRule that fails fails that execution; for a top-level WithPolicy or WithCache step it
// adds a failed execution after the one that wrote.
//
// The events of a run are ordered:
//   - RunStarted comes before every other event of the run and RunFinished after them. A run
//     that fails before its first step, because the pipeline is not built or its inputs are
//     invalid, emits no events.
//   - Every execution emits StepStarted and later StepFinished with the same span.
//   - A top-level step finishes before the next one starts. An execution nested in a composite
//     step (Branch, Parallel, Fallback, or the fallback step of an OnErrorFallback policy
//     nested in one) starts after the composite starts and finishes before it finishes.
//   - The attempts of a step follow each other in order. A skip, or the fallback step of an
//     OnErrorFallback policy, follows the step's last failed attempt.
//   - A pipeline run by a step, through NewProgramStep or a step that runs it with its
//     context, joins the observation of the run: its RunStarted and RunFinished come between
//     that step's StepStarted and StepFinished, and its run span's parent is the step's span.
//     Its events go to its own observers and then to those of every run around it.
//
// Under Parallel the events of steps that run side by side interleave, but the observer and
// redactor calls of one observed run, the runs nested in it included, are serialised: no two of
// them run at once, and every observer of the run sees its events in the same order. Calls are
// synchronous, so an observer that blocks stalls the run, and it must not wait for another step
// of the same run. Separate top-level runs call a shared observer concurrently, even when they
// share a TraceID because their callers gave them the same RunContext, so an observer of a
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
// The values an event carries pass through PipelineSpec.Redactor; see Redactor. Each observer
// receives its own copy of them, its RunContext's Attributes included, so changing them changes
// neither the run nor what other observers receive.
type PipelineObserver interface {
	RunStarted(ctx context.Context, event RunStartedEvent)
	StepStarted(ctx context.Context, event StepStartedEvent)
	StepFinished(ctx context.Context, event StepFinishedEvent)
	RunFinished(ctx context.Context, event RunFinishedEvent)
}

// RunStartedEvent reports the start of a pipeline run.
type RunStartedEvent struct {
	// Span is the run's span, of kind UsageSpanPipelineRun, whose ID is RunContext.RunID and
	// TraceID RunContext.TraceID. ParentID is the span of the step that started a nested run in
	// the same observation, and empty otherwise.
	Span UsageSpan
	// RunContext is the run's, with its Attributes redacted.
	RunContext RunContext
	// Inputs holds the run's inputs after redaction.
	Inputs map[string]any
}

// RunFinishedEvent reports the end of a pipeline run.
type RunFinishedEvent struct {
	// Span is the run's finished span; its Duration and Usage are the event's, and its Err is
	// the message of the event's Err.
	Span UsageSpan
	// RunContext is the run's, as RunStarted reported it.
	RunContext RunContext
	Duration   time.Duration
	// Usage is what the model calls of the run recorded, nested steps and pipelines included.
	Usage Usage
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
	// Attempt counts the step's executions in the run from one, cache hits and skips included.
	Attempt int
	// Reads holds the values of the step's declared reads after redaction.
	Reads map[string]any
}

// StepFinishedEvent reports the end of one execution of a step.
type StepFinishedEvent struct {
	// Span is the execution's finished span; its Duration and Usage are the event's, and its
	// Err is the message of the event's Err.
	Span UsageSpan
	// RunContext is the RunContext of the run the step belongs to, as RunStarted reported it.
	RunContext RunContext
	Step       string
	Attempt    int
	Duration   time.Duration
	// Usage is what the execution's model calls recorded, nested steps included.
	Usage Usage
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
	// Skipped reports a step that skipped on error under OnErrorSkip: the step did not run
	// again, and Writes holds the defaults it committed.
	Skipped bool
	// ToolCalls holds the tool calls the execution made, through the run's ToolSession or any
	// other, such as an agent's, in the order they finished, after redaction (see Redactor). The
	// calls of the executions nested in it are in their own events. Each call's usage is part of
	// Usage already, and a call answered from a ToolPolicy cache is marked Cached and has none.
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
}

func observeScopeFrom(ctx context.Context) *observeScope {
	scope, _ := ctx.Value(observeKey{}).(*observeScope)
	return scope
}

// observerContext returns ctx, a context of the run, as observers receive it: with its values,
// but ending only when the run's life ends (see runObservation), and without the run it
// belongs to, so nothing an observer does with it counts against the run.
func (r *runObservation) observerContext(ctx context.Context) context.Context {
	return observerCtx{Context: r.life, values: context.WithoutCancel(ctx)}
}

// observerCtx takes its deadline and cancellation from a run's life and its values from a
// run's context, hiding those that tie it to a pipeline run: its trace, step frame, run
// context, budget, usage collection, step cache, evaluation capture, tool session, tool memory
// and staged tool results.
type observerCtx struct {
	context.Context
	values context.Context
}

func (c observerCtx) Value(key any) any {
	switch key.(type) {
	case observeKey, evaluationKey, pipelineRunKey, runContextKey, usageFrameKey, usageCollectorKey, retryAdmissionKey,
		budgetClockKey, toolEnvKey, toolMemoryKey, stepStageKey:
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

// runObservation reports one run to its observers: its own and those of the runs around it,
// with its own redactor followed by theirs. runContext is the run's RunContext as its events
// carry it, with the attributes redacted; it is set when RunStarted is emitted. life ends when
// the context the caller passed to the observation's top-level run ends, when the MaxDuration of
// the run or a run around it expires, or when endLife ends it once the run has finished.
// recording, when set, reports the same run to the RecordingObserver of an evaluation that
// records traces, in an observation of its own; see observeScope.
type runObservation struct {
	trace      *traceObservation
	observers  []PipelineObserver
	redactors  []Redactor
	span       UsageSpan
	runContext RunContext
	start      time.Time
	life       context.Context
	endLife    context.CancelFunc
	recording  *runObservation
}

// observeRun starts observing a run of p over its converted inputs, with rc as its RunContext,
// as part of the observation in ctx if there is one, and emits RunStarted. caller is the context
// the caller passed to the run. It returns ctx carrying the run, and nil when nothing observes
// the run.
//
// A top-level run of an example that ProgramEvaluator records traces for reports to the
// evaluation's RecordingObserver as well as to its own observers. When it has no observers of
// its own, the recorder alone observes it, apart from its observers' trace: the runs inside it
// report to the recorder as part of the recorded trace and, separately, to their own observers
// as they would outside an evaluation.
func (p Pipeline) observeRun(ctx, caller context.Context, inputs map[string]any, rc RunContext) (
	context.Context,
	*runObservation,
) {
	parent := observeScopeFrom(ctx)
	capture := evaluationCaptureFrom(ctx)
	recorder := capture.join(p, parent == nil)
	capture.offer(inputs)
	capture.offer(attributeValues(rc))
	if parent == nil {
		parent = &observeScope{}
	}
	var run, recording *runObservation
	switch {
	case recorder != nil && len(p.observers) == 0:
		recording = p.startRun(nil, "", caller, []PipelineObserver{recorder}, nil, rc)
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
		event := RunStartedEvent{Span: lane.span, RunContext: lane.runContext, Inputs: lane.redactValues(inputs)}
		observerCtx := lane.observerContext(ctx)
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
// neither a parent nor an observer. caller is the context the caller passed to the run.
func (p Pipeline) startRun(
	parent *runObservation,
	parentSpan string,
	caller context.Context,
	observers []PipelineObserver,
	lock *sync.Mutex,
	rc RunContext,
) *runObservation {
	if parent == nil && len(observers) == 0 {
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
	usage := usageFrameFrom(ctx).total()
	r.trace.mu.Lock()
	defer r.trace.mu.Unlock()
	for lane := r; lane != nil; lane = lane.recording {
		laneErr := lane.redactError(err)
		event := RunFinishedEvent{
			Span:       finishSpan(lane.span, duration, usage, laneErr),
			RunContext: lane.runContext,
			Duration:   duration,
			Usage:      usage,
			Err:        laneErr,
		}
		observerCtx := lane.observerContext(ctx)
		lane.deliver(func(observer PipelineObserver) {
			event := event
			event.RunContext = event.RunContext.clone()
			observer.RunFinished(observerCtx, event)
		})
		lane.endLife()
	}
}

// stepObservation reports one execution of a step, and recording, when set, reports it to an
// evaluation's RecordingObserver; see runObservation.
type stepObservation struct {
	run       *runObservation
	ctx       context.Context
	span      UsageSpan
	step      string
	reads     map[string]any
	recording *stepObservation
}

// observeStep emits StepStarted for an execution of node over values, the state its view
// reads, when the run in ctx is observed. It returns ctx carrying the execution's span, and
// nil when nothing observes the run.
func observeStep(ctx context.Context, node *stepNode, attempt int, values map[string]any) (
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
		span := NewUsageSpan(scope.run.span.TraceID, scope.span, UsageSpanPipelineStep, node.name)
		next.run, next.span = scope.run, span.ID
		lanes = append(lanes, &stepObservation{run: scope.run, span: span, step: node.name})
	}
	if scope.recording != nil {
		span := NewUsageSpan(scope.recording.span.TraceID, scope.recordingSpan, UsageSpanPipelineStep, node.name)
		next.recording, next.recordingSpan = scope.recording, span.ID
		lanes = append(lanes, &stepObservation{run: scope.recording, span: span, step: node.name})
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
		lane.reads = lane.run.redactValues(reads)
		event := StepStartedEvent{
			Span:       lane.span,
			RunContext: lane.run.runContext,
			Step:       node.name,
			Attempt:    attempt,
			Reads:      lane.reads,
		}
		observerCtx := lane.run.observerContext(ctx)
		lane.run.deliver(func(observer PipelineObserver) {
			event := event
			event.RunContext = event.RunContext.clone()
			event.Reads = copyValues(event.Reads)
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
		event := StepFinishedEvent{
			RunContext: run.runContext,
			Step:       lane.step,
			Attempt:    outcome.Attempt,
			Duration:   outcome.Duration,
			Usage:      outcome.Usage,
			Writes:     run.redactValues(writes),
			Details:    run.redactDetails(details),
			Reads:      lane.reads,
			Cached:     outcome.Cached,
			Skipped:    outcome.Skipped,
		}
		event.ToolCalls = run.scrubToolCalls(toolCalls)
		event.Err = run.redactError(outcome.Err)
		event.Span = finishSpan(lane.span, outcome.Duration, outcome.Usage, event.Err)
		observerCtx := run.observerContext(lane.ctx)
		run.deliver(func(observer PipelineObserver) {
			event := event
			event.RunContext = event.RunContext.clone()
			event.Reads = copyValues(event.Reads)
			event.Writes = copyValues(event.Writes)
			event.Details = copyValues(event.Details)
			event.ToolCalls = cloneRecords(event.ToolCalls)
			observer.StepFinished(observerCtx, event)
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

func finishSpan(span UsageSpan, duration time.Duration, usage Usage, err error) UsageSpan {
	span.EndedAt = span.StartedAt.Add(duration)
	span.Duration = duration
	span.Usage = usage
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
