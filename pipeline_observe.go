package gepa

import (
	"context"
	"errors"
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
// attempts, fallbacks, cache hits and skips are.
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
// Observers are called in the order of PipelineSpec.Observers. An observer that panics is
// recovered: the run continues, the other observers still receive the event, and the observer
// keeps receiving later events. ctx is the context of the run or step the event belongs to,
// without its trace: a pipeline run with it from inside an observer starts a trace of its own.
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

// withoutTrace returns ctx without the trace it carries, for observers.
func withoutTrace(ctx context.Context) context.Context {
	return context.WithValue(ctx, observeKey{}, (*observeScope)(nil))
}

// traceObservation is shared by the runs of one trace. Its lock serialises their observer and
// redactor calls and guards the text it removes from errors; see Redactor.
type traceObservation struct {
	id string
	mu sync.Mutex
	redactedText
}

// runObservation reports one run to its observers: its own and those of the runs around it,
// with its own redactor followed by theirs.
type runObservation struct {
	trace     *traceObservation
	observers []PipelineObserver
	redactors []Redactor
	span      UsageSpan
	start     time.Time
}

// observeRun starts observing a run of p over its converted inputs, as part of the trace in ctx
// if there is one, and emits RunStarted. It returns ctx carrying the run, and nil when nothing
// observes the run.
func (p Pipeline) observeRun(ctx context.Context, inputs map[string]any) (context.Context, *runObservation) {
	parent := observeScopeFrom(ctx)
	if parent == nil && len(p.observers) == 0 {
		return ctx, nil
	}
	run := &runObservation{observers: slices.Clone(p.observers), start: time.Now()}
	if p.redactor != nil {
		run.redactors = []Redactor{p.redactor}
	}
	parentSpan := ""
	if parent != nil {
		run.trace = parent.run.trace
		run.observers = append(run.observers, parent.run.observers...)
		run.redactors = append(run.redactors, parent.run.redactors...)
		parentSpan = parent.span
	} else {
		run.trace = &traceObservation{id: NewID()}
	}
	run.span = NewUsageSpan(run.trace.id, parentSpan, UsageSpanPipelineRun, "pipeline")
	ctx = context.WithValue(ctx, observeKey{}, &observeScope{run: run, span: run.span.ID})

	run.trace.mu.Lock()
	defer run.trace.mu.Unlock()
	event := RunStartedEvent{Span: run.span, Inputs: run.redactValues(inputs)}
	observerCtx := withoutTrace(ctx)
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
	observerCtx := withoutTrace(ctx)
	r.deliver(func(observer PipelineObserver) {
		observer.RunFinished(observerCtx, event)
	})
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
	observerCtx := withoutTrace(ctx)
	run.deliver(func(observer PipelineObserver) {
		event := event
		event.Reads = copyValues(event.Reads)
		observer.StepStarted(observerCtx, event)
	})
	return ctx, observed
}

// finish emits StepFinished for the execution that ended with outcome, writing result's
// writes when it succeeded.
func (s *stepObservation) finish(outcome stepOutcome, result stepResult) {
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
	observerCtx := withoutTrace(s.ctx)
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
