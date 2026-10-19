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
// Every run is one trace: each run has a span, each execution of a step has a span whose parent
// is the span of the run or of the execution around it, and every span of the run, those of
// pipelines nested in it included, shares its TraceID. An execution is one run of a step: each
// retry attempt, each fallback step, each cache hit and each skip on error is an execution of
// its own, numbered by Attempt. WithPolicy and WithCache are not executions of their own; their
// attempts, fallbacks, cache hits and skips are. A top-level step's writes are merged into the
// state as part of its execution, so a MergeRule that fails fails that execution; for a
// top-level WithPolicy or WithCache step it adds a failed execution after the one that wrote.
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
//     context, joins the run's trace: its RunStarted and RunFinished come between that step's
//     StepStarted and StepFinished, and its run span's parent is the step's span. Its events
//     go to its own observers and then to those of every run around it.
//
// Under Parallel the events of steps that run side by side interleave, but the observer and
// redactor calls of one trace are serialised: no two of them run at once, and every observer
// of the trace sees its events in the same order. Calls are synchronous, so an observer that
// blocks stalls the run, and it must not wait for another step of the same run. Runs of
// different traces call a shared observer concurrently, so an observer of a pipeline that runs
// concurrently must be safe for concurrent use.
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
// own context. It
// carries neither the run's trace, budget nor usage collection, so an observer's model calls
// and pipeline runs are independent of the observed run: they spend none of its budget, do not
// count in its usage, and a pipeline run with ctx starts a trace of its own.
//
// The values an event carries pass through PipelineSpec.Redactor; see Redactor. Each observer
// receives its own copy of them, so changing them changes neither the run nor what other
// observers receive.
type PipelineObserver interface {
	RunStarted(ctx context.Context, event RunStartedEvent)
	StepStarted(ctx context.Context, event StepStartedEvent)
	StepFinished(ctx context.Context, event StepFinishedEvent)
	RunFinished(ctx context.Context, event RunFinishedEvent)
}

// RunStartedEvent reports the start of a pipeline run.
type RunStartedEvent struct {
	// Span is the run's span, of kind UsageSpanPipelineRun. ParentID is the span of the step
	// that started a nested run, empty for a top-level run.
	Span UsageSpan
	// Inputs holds the run's inputs after redaction.
	Inputs map[string]any
}

// RunFinishedEvent reports the end of a pipeline run.
type RunFinishedEvent struct {
	// Span is the run's finished span; its Duration and Usage are the event's, and its Err is
	// the message of the event's Err.
	Span     UsageSpan
	Duration time.Duration
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
	Step string
	// Attempt counts the step's executions in the run from one, cache hits and skips included.
	Attempt int
	// Reads holds the values of the step's declared reads after redaction.
	Reads map[string]any
}

// StepFinishedEvent reports the end of one execution of a step.
type StepFinishedEvent struct {
	// Span is the execution's finished span; its Duration and Usage are the event's, and its
	// Err is the message of the event's Err.
	Span     UsageSpan
	Step     string
	Attempt  int
	Duration time.Duration
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
}

type observeKey struct{}

// observeScope is what an observed run's context carries: the run and the span of the
// innermost execution, which becomes the parent of the next span.
type observeScope struct {
	run  *runObservation
	span string
}

func observeScopeFrom(ctx context.Context) *observeScope {
	scope, _ := ctx.Value(observeKey{}).(*observeScope)
	return scope
}

// evaluationKey carries an evaluationCapture, which every pipeline run started with the
// context reports to; see ProgramEvaluator.
type evaluationKey struct{}

// evaluationCapture collects what ProgramEvaluator needs from the pipeline runs of one
// evaluated example: the Redactors of every run, nested ones included, the types of the keys
// those pipelines declare, the text the runs' redaction removed from errors, failed writes
// included, and, when recorder is set, the traces of the top-level runs.
type evaluationCapture struct {
	recorder  *RecordingObserver
	mu        sync.Mutex
	redactors []Redactor
	types     map[string]reflect.Type
	forms     map[string]string
}

func withEvaluationCapture(ctx context.Context, capture *evaluationCapture) context.Context {
	return context.WithValue(ctx, evaluationKey{}, capture)
}

// join records a run of p and returns the recorder a top-level run reports to, if any. The
// first pipeline to declare a key gives its type.
func (c *evaluationCapture) join(p Pipeline, topLevel bool) *RecordingObserver {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	if p.redactor != nil {
		c.redactors = append(c.redactors, p.redactor)
	}
	if c.types == nil {
		c.types = map[string]reflect.Type{}
	}
	for name, typ := range p.inputTypes {
		if c.types[name] == nil {
			c.types[name] = typ
		}
	}
	for _, node := range p.steps {
		for name, typ := range node.writeTypes {
			if c.types[name] == nil {
				c.types[name] = typ
			}
		}
	}
	c.mu.Unlock()
	if !topLevel {
		return nil
	}
	return c.recorder
}

// remember keeps the text a run's redaction removed. The caller holds the run's trace lock.
func (c *evaluationCapture) remember(text *redactedText) {
	if c == nil || len(text.forms) == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.forms == nil {
		c.forms = map[string]string{}
	}
	maps.Copy(c.forms, text.forms)
}

// apply returns the evaluation item of the example the capture watched, with the traces it
// recorded and its reflection values redacted by the Redactors of the pipelines it ran.
func (c *evaluationCapture) apply(item EvaluationItem) EvaluationItem {
	if trace := evaluationTrace(c.recorder); trace != nil {
		item.Trace = trace
	}
	c.mu.Lock()
	redaction := evaluationRedaction{
		redactors: slices.Clone(c.redactors),
		types:     maps.Clone(c.types),
		forms:     maps.Clone(c.forms),
	}
	c.mu.Unlock()
	return redaction.apply(item)
}

// observerContext returns ctx, a context of the run, as observers receive it: with its values,
// but ending only when the run's life ends (see runObservation), and without the run it
// belongs to, so nothing an observer does with it counts against the run.
func (r *runObservation) observerContext(ctx context.Context) context.Context {
	return observerCtx{Context: r.life, values: context.WithoutCancel(ctx)}
}

// observerCtx takes its deadline and cancellation from a run's life and its values from a
// run's context, hiding those that tie it to a pipeline run: its trace, step frame, budget,
// usage collection, step cache and evaluation capture.
type observerCtx struct {
	context.Context
	values context.Context
}

func (c observerCtx) Value(key any) any {
	switch key.(type) {
	case observeKey, evaluationKey, pipelineRunKey, usageFrameKey, usageCollectorKey, retryAdmissionKey, budgetClockKey:
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

// traceObservation is shared by the runs of one trace. Its lock serialises their observer and
// redactor calls and guards the text it removes from errors; see Redactor.
type traceObservation struct {
	id string
	mu sync.Mutex
	redactedText
}

// runObservation reports one run to its observers: its own and those of the runs around it,
// with its own redactor followed by theirs. life ends when the context the caller passed to the
// trace's top-level run ends, when the MaxDuration of the run or a run around it expires, or
// when endLife ends it once the run has finished.
type runObservation struct {
	trace     *traceObservation
	observers []PipelineObserver
	redactors []Redactor
	span      UsageSpan
	start     time.Time
	life      context.Context
	endLife   context.CancelFunc
	capture   *evaluationCapture
}

// observeRun starts observing a run of p over its converted inputs, as part of the trace in ctx
// if there is one, and emits RunStarted. caller is the context the caller passed to the run.
// It returns ctx carrying the run, and nil when nothing observes the run.
func (p Pipeline) observeRun(ctx, caller context.Context, inputs map[string]any) (
	context.Context,
	*runObservation,
) {
	parent := observeScopeFrom(ctx)
	capture, _ := ctx.Value(evaluationKey{}).(*evaluationCapture)
	sink := capture.join(p, parent == nil)
	// an evaluated run with a Redactor is observed even without observers, so that the text
	// its redaction removes reaches the evaluation
	evaluated := capture != nil && p.redactor != nil
	if parent == nil && len(p.observers) == 0 && sink == nil && !evaluated {
		return ctx, nil
	}
	run := &runObservation{
		observers: appendObservers(nil, p.observers...),
		start:     time.Now(),
		life:      caller,
		capture:   capture,
	}
	if p.redactor != nil {
		run.redactors = []Redactor{p.redactor}
	}
	parentSpan := ""
	if parent != nil {
		run.trace = parent.run.trace
		run.observers = appendObservers(run.observers, parent.run.observers...)
		run.redactors = append(run.redactors, parent.run.redactors...)
		run.life = parent.run.life
		parentSpan = parent.span
	} else {
		if sink != nil {
			run.observers = appendObservers(run.observers, sink)
		}
		run.trace = &traceObservation{id: NewID()}
	}
	if limit := p.budget.MaxDuration; limit > 0 {
		run.life, run.endLife = context.WithDeadline(run.life, run.start.Add(limit))
	} else {
		run.life, run.endLife = context.WithCancel(run.life)
	}
	run.span = NewUsageSpan(run.trace.id, parentSpan, UsageSpanPipelineRun, "pipeline")
	ctx = context.WithValue(ctx, observeKey{}, &observeScope{run: run, span: run.span.ID})

	run.trace.mu.Lock()
	defer run.trace.mu.Unlock()
	event := RunStartedEvent{Span: run.span, Inputs: run.redactValues(inputs)}
	observerCtx := run.observerContext(ctx)
	run.deliver(func(observer PipelineObserver) {
		event := event
		event.Inputs = copyValues(event.Inputs)
		observer.RunStarted(observerCtx, event)
	})
	return ctx, run
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
	err = r.redactError(err)
	event := RunFinishedEvent{Span: finishSpan(r.span, duration, usage, err), Duration: duration, Usage: usage, Err: err}
	observerCtx := r.observerContext(ctx)
	r.deliver(func(observer PipelineObserver) {
		observer.RunFinished(observerCtx, event)
	})
	r.capture.remember(&r.trace.redactedText)
	r.endLife()
}

// stepObservation reports one execution of a step.
type stepObservation struct {
	run   *runObservation
	ctx   context.Context
	span  UsageSpan
	step  string
	reads map[string]any
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
	run := scope.run
	span := NewUsageSpan(run.trace.id, scope.span, UsageSpanPipelineStep, node.name)
	ctx = context.WithValue(ctx, observeKey{}, &observeScope{run: run, span: span.ID})
	reads := make(map[string]any, len(node.reads))
	for _, key := range node.reads {
		if value, ok := values[key.Name]; ok {
			reads[key.Name] = value
		}
	}

	run.trace.mu.Lock()
	defer run.trace.mu.Unlock()
	observed := &stepObservation{run: run, ctx: ctx, span: span, step: node.name, reads: run.redactValues(reads)}
	event := StepStartedEvent{Span: span, Step: node.name, Attempt: attempt, Reads: observed.reads}
	observerCtx := run.observerContext(ctx)
	run.deliver(func(observer PipelineObserver) {
		event := event
		event.Reads = copyValues(event.Reads)
		observer.StepStarted(observerCtx, event)
	})
	return ctx, observed
}

// finish emits StepFinished for the execution that ended with outcome, writing result's
// writes when it succeeded. When it failed, attempted holds the values the step wrote or tried
// to write: they leave in no event, but pass through the redactors so that their text is
// removed from the error.
func (s *stepObservation) finish(outcome stepOutcome, result stepResult, attempted []stateWrite) {
	if s == nil {
		return
	}
	run := s.run
	var writes map[string]any
	if outcome.Err == nil {
		writes = result.writes
	}
	var details map[string]any
	var detailed *stepDetailsError
	if errors.As(outcome.Err, &detailed) {
		details = detailed.details
	}

	run.trace.mu.Lock()
	defer run.trace.mu.Unlock()
	if outcome.Err != nil {
		for _, write := range attempted {
			run.redact(write.key, write.value)
		}
	}
	event := StepFinishedEvent{
		Step:     s.step,
		Attempt:  outcome.Attempt,
		Duration: outcome.Duration,
		Usage:    outcome.Usage,
		Writes:   run.redactValues(writes),
		Details:  run.redactDetails(details),
		Reads:    s.reads,
		Cached:   outcome.Cached,
		Skipped:  outcome.Skipped,
	}
	event.Err = run.redactError(outcome.Err)
	event.Span = finishSpan(s.span, outcome.Duration, outcome.Usage, event.Err)
	observerCtx := run.observerContext(s.ctx)
	run.deliver(func(observer PipelineObserver) {
		event := event
		event.Reads = copyValues(event.Reads)
		event.Writes = copyValues(event.Writes)
		event.Details = copyValues(event.Details)
		observer.StepFinished(observerCtx, event)
	})
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
