package gepa

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// eventLog records the events it observes as lines that name each span by its step and
// attempt, so a test can compare whole event sequences.
type eventLog struct {
	mu           sync.Mutex
	labels       map[string]string
	lines        []string
	runsStarted  []RunStartedEvent
	stepsStarted []StepStartedEvent
	finished     []StepFinishedEvent
	runsFinished []RunFinishedEvent
	onFinish     func(StepFinishedEvent)
}

func (l *eventLog) label(span string) string {
	if label, ok := l.labels[span]; ok {
		return label
	}
	return "?"
}

func (l *eventLog) RunStarted(_ context.Context, event RunStartedEvent) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.labels == nil {
		l.labels = map[string]string{}
	}
	label := "run"
	if event.Span.ParentID != "" {
		label = "run in " + l.label(event.Span.ParentID)
	}
	l.labels[event.Span.ID] = label
	l.lines = append(l.lines, "start "+label)
	l.runsStarted = append(l.runsStarted, event)
}

func (l *eventLog) StepStarted(_ context.Context, event StepStartedEvent) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.labels == nil {
		l.labels = map[string]string{}
	}
	label := fmt.Sprintf("%s#%d", event.Step, event.Attempt)
	l.lines = append(l.lines, fmt.Sprintf("start %s in %s", label, l.label(event.Span.ParentID)))
	l.labels[event.Span.ID] = label
	l.stepsStarted = append(l.stepsStarted, event)
}

func (l *eventLog) StepFinished(_ context.Context, event StepFinishedEvent) {
	l.mu.Lock()
	line := "finish " + l.label(event.Span.ID)
	if event.Cached {
		line += " cached"
	}
	if event.Skipped {
		line += " skipped"
	}
	if event.Err != nil {
		line += " err: " + event.Err.Error()
	}
	l.lines = append(l.lines, line)
	l.finished = append(l.finished, event)
	onFinish := l.onFinish
	l.mu.Unlock()
	if onFinish != nil {
		onFinish(event)
	}
}

func (l *eventLog) RunFinished(_ context.Context, event RunFinishedEvent) {
	l.mu.Lock()
	defer l.mu.Unlock()
	line := "finish " + l.label(event.Span.ID)
	if event.Err != nil {
		line += " err: " + event.Err.Error()
	}
	l.lines = append(l.lines, line)
	l.runsFinished = append(l.runsFinished, event)
}

func (l *eventLog) Lines() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return slices.Clone(l.lines)
}

// finishedStep returns the StepFinished event of the named step's attempt.
func (l *eventLog) finishedStep(t *testing.T, step string, attempt int) StepFinishedEvent {
	t.Helper()
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, event := range l.finished {
		if event.Step == step && event.Attempt == attempt {
			return event
		}
	}
	t.Fatalf("no StepFinished for %s#%d", step, attempt)
	return StepFinishedEvent{}
}

// traceIDs returns every trace ID the log's events carry.
func (l *eventLog) traceIDs() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	var ids []string
	for _, event := range l.runsStarted {
		ids = append(ids, event.Span.TraceID)
	}
	for _, event := range l.stepsStarted {
		ids = append(ids, event.Span.TraceID)
	}
	for _, event := range l.finished {
		ids = append(ids, event.Span.TraceID)
	}
	for _, event := range l.runsFinished {
		ids = append(ids, event.Span.TraceID)
	}
	return slices.Compact(slices.Sorted(slices.Values(ids)))
}

func lengthStep() Step {
	return NewStep("parse", refs(docKey.Ref()), refs(amountKey.Ref()), func(_ context.Context, s State) error {
		document, err := docKey.Get(s)
		if err != nil {
			return err
		}
		amountKey.Set(s, len(document))
		return nil
	})
}

func tenTokenModel(text string) Generator {
	return GeneratorFunc(func(context.Context, string) (Generation, error) {
		return Generation{Text: text, Usage: tenTokens}, nil
	})
}

func TestPipelineObserverSuccessSequence(t *testing.T) {
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			lengthStep(),
			predictStep("summarise", tenTokenModel(`{"summary": "short"}`), docKey.Ref(), summaryKey.Ref()),
		},
		Observers: []PipelineObserver{log},
	}.Build()
	require.NoError(t, err)

	got, err := pipeline.Run(context.Background(), NewState(Prediction{"document": "cash 10"}))
	require.NoError(t, err)
	require.Equal(t, Prediction{"amount": 7, "summary": "short"}, got.Prediction())
	require.Equal(t, []string{
		"start run",
		"start parse#1 in run",
		"finish parse#1",
		"start summarise#1 in run",
		"finish summarise#1",
		"finish run",
	}, log.Lines())

	require.Len(t, log.traceIDs(), 1)
	started := log.runsStarted[0]
	require.Equal(t, UsageSpanPipelineRun, started.Span.Kind)
	require.Empty(t, started.Span.ParentID)
	require.Equal(t, map[string]any{"document": "cash 10"}, started.Inputs)

	parse := log.finishedStep(t, "parse", 1)
	require.Equal(t, UsageSpanPipelineStep, parse.Span.Kind)
	require.Equal(t, "parse", parse.Span.Name)
	require.Equal(t, started.Span.ID, parse.Span.ParentID)
	require.Equal(t, map[string]any{"document": "cash 10"}, parse.Reads)
	require.Equal(t, map[string]any{"amount": 7}, parse.Writes)
	require.Equal(t, Usage{}, parse.Usage)

	summarise := log.finishedStep(t, "summarise", 1)
	require.Equal(t, tenTokens, summarise.Usage)
	require.Equal(t, tenTokens, summarise.Span.Usage)
	require.Positive(t, summarise.Duration)
	require.Equal(t, summarise.Duration, summarise.Span.Duration)
	require.Equal(t, summarise.Span.StartedAt.Add(summarise.Duration), summarise.Span.EndedAt)
	require.Equal(t, map[string]any{"summary": "short"}, summarise.Writes)
	require.Equal(t, log.stepsStarted[1].Span.ID, summarise.Span.ID)
	require.Equal(t, map[string]any{"document": "cash 10"}, log.stepsStarted[1].Reads)

	finished := log.runsFinished[0]
	require.Equal(t, started.Span.ID, finished.Span.ID)
	require.Equal(t, tenTokens, finished.Usage)
	require.NoError(t, finished.Err)
	require.Positive(t, finished.Duration)
}

func TestPipelineObserverFailureSequence(t *testing.T) {
	stepErr := PermanentError(errors.New("bad amount"))
	check := NewStep("check", refs(amountKey.Ref()), refs(decisionKey.Ref()), func(_ context.Context, s State) error {
		decisionKey.Set(s, "never committed")
		return WithStepDetails(stepErr, map[string]any{"limit": 5})
	})
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Steps:     []Step{lengthStep(), check},
		Observers: []PipelineObserver{log},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(Prediction{"document": "cash 10"}))
	require.EqualError(t, err, `pipeline step "check": bad amount`)
	require.Equal(t, []string{
		"start run",
		"start parse#1 in run",
		"finish parse#1",
		"start check#1 in run",
		"finish check#1 err: bad amount",
		`finish run err: pipeline step "check": bad amount`,
	}, log.Lines())

	failed := log.finishedStep(t, "check", 1)
	require.ErrorIs(t, failed.Err, stepErr)
	require.True(t, IsPermanent(failed.Err))
	require.Nil(t, failed.Writes)
	require.Equal(t, map[string]any{"amount": 7}, failed.Reads)
	require.Equal(t, map[string]any{"limit": 5}, failed.Details)
	require.Equal(t, "bad amount", failed.Span.Err)

	var runErr *StepError
	require.ErrorAs(t, log.runsFinished[0].Err, &runErr)
	require.Equal(t, "check", runErr.Step)
}

func TestPipelineObserverRetrySequence(t *testing.T) {
	attempts := 0
	flaky := NewStep("flaky", nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
		attempts++
		if attempts == 1 {
			return errRateLimited
		}
		amountKey.Set(s, attempts)
		return nil
	})
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Steps:     []Step{WithPolicy(flaky, StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}})},
		Observers: []PipelineObserver{log},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(nil))
	require.NoError(t, err)
	require.Equal(t, []string{
		"start run",
		"start flaky#1 in run",
		"finish flaky#1 err: rate limited",
		"start flaky#2 in run",
		"finish flaky#2",
		"finish run",
	}, log.Lines())
	require.True(t, IsTransient(log.finishedStep(t, "flaky", 1).Err))
	require.Equal(t, map[string]any{"amount": 2}, log.finishedStep(t, "flaky", 2).Writes)
}

func TestPipelineObserverCacheHitSequence(t *testing.T) {
	calls := &atomic.Int32{}
	seen, see := closer()
	log := &eventLog{onFinish: func(event StepFinishedEvent) {
		if event.Step == "summarise" {
			see()
		}
	}}
	attempts := 0
	flaky := NewStep("flaky", nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
		attempts++
		if attempts == 1 {
			<-seen
			return errRateLimited
		}
		amountKey.Set(s, attempts)
		return nil
	})
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{WithPolicy(
			Parallel("both", WithCache(summariseStep(calls, nil), nil), flaky),
			StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}},
		)},
		Observers: []PipelineObserver{log},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(Prediction{"document": "cash 10"}))
	require.NoError(t, err)
	require.Equal(t, int32(1), calls.Load())

	// the steps of a Parallel start in either order, so the sequence is compared per sibling
	var flakyLines, others []string
	for _, line := range log.Lines() {
		if strings.Contains(line, "flaky#") {
			flakyLines = append(flakyLines, line)
		} else {
			others = append(others, line)
		}
	}
	require.Equal(t, []string{
		"start run",
		"start both#1 in run",
		"start summarise#1 in both#1",
		"finish summarise#1",
		`finish both#1 err: pipeline step "flaky": rate limited`,
		"start both#2 in run",
		"start summarise#2 in both#2",
		"finish summarise#2 cached",
		"finish both#2",
		"finish run",
	}, others)
	require.Equal(t, []string{
		"start flaky#1 in both#1",
		"finish flaky#1 err: rate limited",
		"start flaky#2 in both#2",
		"finish flaky#2",
	}, flakyLines)

	hit := log.finishedStep(t, "summarise", 2)
	require.True(t, hit.Cached)
	require.Equal(t, Usage{}, hit.Usage)
	require.Equal(t, map[string]any{"document": "cash 10"}, hit.Reads)
	require.Equal(t, map[string]any{"summary": "short"}, hit.Writes)
}

func TestPipelineObserverNestedCompositeSequence(t *testing.T) {
	verdictKey := Key[string]{Name: "verdict"}
	brief := NewStep("brief", refs(docKey.Ref()), refs(decisionKey.Ref()), func(_ context.Context, s State) error {
		decisionKey.Set(s, "brief")
		return nil
	})
	full := NewStep("full", refs(docKey.Ref()), refs(decisionKey.Ref()), func(_ context.Context, s State) error {
		decisionKey.Set(s, "full")
		return nil
	})
	route := Branch("route", func(State) (string, error) { return "short", nil }, map[string]Step{
		"short": brief,
		"long":  full,
	})
	verdict := func(name, value string, err error) Step {
		return NewStep(name, nil, refs(verdictKey.Ref()), func(_ context.Context, s State) error {
			verdictKey.Set(s, value)
			return err
		})
	}
	guess := NewStep("guess", nil, refs(notesKey.Ref()), func(context.Context, State) error {
		return errors.New("no notes")
	})
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			route,
			Fallback("fb", failAmount("primary", TransientError(errors.New("down"))), writeAmount("secondary", 7)),
			WithPolicy(guess, StepPolicy{OnError: OnErrorSkip}),
			WithPolicy(verdict("estimate", "x", errBadRequest), StepPolicy{OnError: OnErrorFallback(verdict("spare", "ok", nil))}),
		},
		Defaults:  []KeyDefault{DefaultValue(notesKey, []string{"none"})},
		Observers: []PipelineObserver{log},
	}.Build()
	require.NoError(t, err)

	got, err := pipeline.Run(context.Background(), NewState(Prediction{"document": "d"}))
	require.NoError(t, err)
	require.Equal(
		t,
		Prediction{"decision": "brief", "amount": 7, "notes": []string{"none"}, "verdict": "ok"},
		got.Prediction(),
	)
	require.Equal(t, []string{
		"start run",
		"start route#1 in run",
		"start brief#1 in route#1",
		"finish brief#1",
		"finish route#1",
		"start fb#1 in run",
		"start primary#1 in fb#1",
		"finish primary#1 err: down",
		"start secondary#1 in fb#1",
		"finish secondary#1",
		"finish fb#1",
		"start guess#1 in run",
		"finish guess#1 err: no notes",
		"start guess#2 in run",
		"finish guess#2 skipped",
		"start estimate#1 in run",
		"finish estimate#1 err: bad request",
		"start spare#1 in run",
		"finish spare#1",
		"finish run",
	}, log.Lines())
	require.Equal(t, map[string]any{"decision": "brief"}, log.finishedStep(t, "route", 1).Writes)
	require.Equal(t, map[string]any{"amount": 7}, log.finishedStep(t, "fb", 1).Writes)
	skipped := log.finishedStep(t, "guess", 2)
	require.True(t, skipped.Skipped)
	require.NoError(t, skipped.Err)
	require.Equal(t, map[string]any{"notes": []string{"none"}}, skipped.Writes)
}

func TestPipelineObserverNestedPipelineSequence(t *testing.T) {
	innerLog := &eventLog{}
	inner, err := PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Steps:     []Step{predictStep("summarise", tenTokenModel(`{"summary": "short"}`), docKey.Ref(), summaryKey.Ref())},
		Observers: []PipelineObserver{innerLog},
	}.Build()
	require.NoError(t, err)
	outerLog := &eventLog{}
	outer, err := PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Steps:     []Step{NewProgramStep("inner", inner, refs(docKey.Ref()), refs(summaryKey.Ref()))},
		Observers: []PipelineObserver{outerLog},
	}.Build()
	require.NoError(t, err)

	_, err = outer.Run(context.Background(), NewState(Prediction{"document": "cash 10"}))
	require.NoError(t, err)
	require.Equal(t, []string{
		"start run",
		"start inner#1 in run",
		"start run in inner#1",
		"start summarise#1 in run in inner#1",
		"finish summarise#1",
		"finish run in inner#1",
		"finish inner#1",
		"finish run",
	}, outerLog.Lines())
	require.Equal(t, []string{
		"start run in ?",
		"start summarise#1 in run in ?",
		"finish summarise#1",
		"finish run in ?",
	}, innerLog.Lines())

	traceIDs := outerLog.traceIDs()
	require.Len(t, traceIDs, 1)
	require.Equal(t, traceIDs, innerLog.traceIDs())
	innerStep := outerLog.finishedStep(t, "inner", 1)
	require.Equal(t, innerStep.Span.ID, innerLog.runsStarted[0].Span.ParentID)
	require.Equal(t, tenTokens, innerStep.Usage)
	require.Equal(t, tenTokens, innerLog.runsFinished[0].Usage)
	require.Equal(t, tenTokens, outerLog.runsFinished[0].Usage)

	// run on its own, the inner pipeline starts a trace of its own
	_, err = inner.Run(context.Background(), NewState(Prediction{"document": "cash 10"}))
	require.NoError(t, err)
	require.Len(t, innerLog.traceIDs(), 2)
	require.Empty(t, innerLog.runsStarted[1].Span.ParentID)
}

// panickyObserver panics in every call after counting it.
type panickyObserver struct{ calls atomic.Int32 }

func (o *panickyObserver) RunStarted(context.Context, RunStartedEvent) {
	o.calls.Add(1)
	panic("observer panicked")
}

func (o *panickyObserver) StepStarted(context.Context, StepStartedEvent) {
	o.calls.Add(1)
	panic("observer panicked")
}

func (o *panickyObserver) StepFinished(context.Context, StepFinishedEvent) {
	o.calls.Add(1)
	panic(errors.New("observer panicked"))
}

func (o *panickyObserver) RunFinished(context.Context, RunFinishedEvent) {
	o.calls.Add(1)
	panic("observer panicked")
}

func TestPipelineObserverPanicIsContained(t *testing.T) {
	first, second := &panickyObserver{}, &panickyObserver{}
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Steps:     []Step{lengthStep(), readAmount("decide")},
		Observers: []PipelineObserver{first, log, second},
	}.Build()
	require.NoError(t, err)

	got, err := pipeline.Run(context.Background(), NewState(Prediction{"document": "cash 10"}))
	require.NoError(t, err)
	require.Equal(t, Prediction{"amount": 7, "decision": "amount 7"}, got.Prediction())
	require.Equal(t, []string{
		"start run",
		"start parse#1 in run",
		"finish parse#1",
		"start decide#1 in run",
		"finish decide#1",
		"finish run",
	}, log.Lines())
	require.Equal(t, int32(6), first.calls.Load())
	require.Equal(t, int32(6), second.calls.Load())
}

// overlapDetector counts the observer calls in progress for each trace and records any that
// overlap.
type overlapDetector struct {
	mu       sync.Mutex
	inFlight map[string]int
	overlaps atomic.Int32
	calls    atomic.Int32
}

func (d *overlapDetector) enter(trace string) {
	d.calls.Add(1)
	d.mu.Lock()
	if d.inFlight == nil {
		d.inFlight = map[string]int{}
	}
	d.inFlight[trace]++
	if d.inFlight[trace] > 1 {
		d.overlaps.Add(1)
	}
	d.mu.Unlock()
	runtime.Gosched()
	d.mu.Lock()
	d.inFlight[trace]--
	d.mu.Unlock()
}

func (d *overlapDetector) RunStarted(_ context.Context, event RunStartedEvent) {
	d.enter(event.Span.TraceID)
}
func (d *overlapDetector) StepStarted(_ context.Context, event StepStartedEvent) {
	d.enter(event.Span.TraceID)
}

func (d *overlapDetector) StepFinished(_ context.Context, event StepFinishedEvent) {
	d.enter(event.Span.TraceID)
}
func (d *overlapDetector) RunFinished(_ context.Context, event RunFinishedEvent) {
	d.enter(event.Span.TraceID)
}

// TestPipelineObserverParallelCallsAreSerialised runs a Parallel with a redactor, a recorder
// and an overlap detector from many goroutines; run it with -race.
func TestPipelineObserverParallelCallsAreSerialised(t *testing.T) {
	const width, runs = 8, 12
	var steps []Step
	for i := range width {
		key := Key[string]{Name: fmt.Sprintf("out%d", i)}
		steps = append(steps, predictStep(key.Name, tenTokenModel(fmt.Sprintf(`{%q: "v"}`, key.Name)), docKey.Ref(), key.Ref()))
	}
	detector := &overlapDetector{}
	recorder := &RecordingObserver{}
	var redactions atomic.Int32
	pipeline, err := PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Steps:     []Step{Parallel("fan", steps...)},
		Observers: []PipelineObserver{detector, recorder},
		Redactor: func(_ string, value any) any {
			redactions.Add(1)
			return value
		},
	}.Build()
	require.NoError(t, err)

	var wg sync.WaitGroup
	for range runs {
		wg.Go(func() {
			_, err := pipeline.Run(context.Background(), NewState(Prediction{"document": "d"}))
			require.NoError(t, err)
		})
	}
	wg.Wait()

	require.Zero(t, detector.overlaps.Load())
	require.Equal(t, int32(runs*(2+2+2*width)), detector.calls.Load())
	require.Positive(t, redactions.Load())
	traces := recorder.Traces()
	require.Len(t, traces, runs)
	for _, trace := range traces {
		require.Len(t, trace.Steps, 1)
		require.Len(t, trace.Steps[0].Steps, width)
		require.Equal(t, Usage{PromptTokens: 4 * width, CompletionTokens: 6 * width, TotalTokens: 10 * width}, trace.Usage)
	}
}

func TestPipelineObserverRejectsNilObservers(t *testing.T) {
	for name, observer := range map[string]PipelineObserver{"nil": nil, "typed nil": (*eventLog)(nil)} {
		t.Run(name, func(t *testing.T) {
			_, err := PipelineSpec{
				Steps:     []Step{writeAmount("write", 1)},
				Observers: []PipelineObserver{&eventLog{}, observer},
			}.Build()
			require.EqualError(t, err, "pipeline observer 1 is nil")
		})
	}
}

// pipelineInObserver runs a pipeline with the context of each StepFinished it observes.
type pipelineInObserver struct {
	eventLog
	pipeline Pipeline
	errs     []error
}

func (o *pipelineInObserver) StepFinished(ctx context.Context, event StepFinishedEvent) {
	o.eventLog.StepFinished(ctx, event)
	_, err := o.pipeline.Run(ctx, NewState(nil))
	o.errs = append(o.errs, err)
}

func TestPipelineObserverContextStartsANewTrace(t *testing.T) {
	otherLog := &eventLog{}
	other, err := PipelineSpec{Steps: []Step{writeAmount("other", 1)}, Observers: []PipelineObserver{otherLog}}.Build()
	require.NoError(t, err)
	observer := &pipelineInObserver{pipeline: other}
	pipeline, err := PipelineSpec{
		Steps:     []Step{writeAmount("write", 2)},
		Observers: []PipelineObserver{observer},
	}.Build()
	require.NoError(t, err)

	done := make(chan error, 1)
	go func() {
		_, err := pipeline.Run(context.Background(), NewState(nil))
		done <- err
	}()
	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("a pipeline run inside an observer deadlocked")
	}
	require.Equal(t, []error{nil}, observer.errs)
	require.Equal(t, []string{"start run", "start write#1 in run", "finish write#1", "finish run"}, observer.Lines())
	require.Equal(t, []string{"start run", "start other#1 in run", "finish other#1", "finish run"}, otherLog.Lines())
	require.NotEqual(t, observer.traceIDs(), otherLog.traceIDs())
}

func TestPipelineObserverEventsAreCopies(t *testing.T) {
	notes := []string{"a"}
	mutating := &mutatingObserver{}
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(notesKey.Ref()),
		Steps:     []Step{readNotesStep()},
		Observers: []PipelineObserver{mutating, log},
	}.Build()
	require.NoError(t, err)

	state, err := pipeline.RunState(context.Background(), NewState(Prediction{"notes": notes}))
	require.NoError(t, err)
	notesAfter, err := notesKey.Get(state)
	require.NoError(t, err)
	require.Equal(t, []string{"a"}, notesAfter)
	require.Equal(t, map[string]any{"notes": []string{"a"}}, log.runsStarted[0].Inputs)
	require.Equal(t, map[string]any{"notes": []string{"a"}}, log.finished[0].Reads)
	require.Equal(t, map[string]any{"decision": "a"}, log.finished[0].Writes)
}

// mutatingObserver changes every value it receives.
type mutatingObserver struct{}

func (mutatingObserver) RunStarted(_ context.Context, event RunStartedEvent) { mutate(event.Inputs) }
func (mutatingObserver) StepStarted(_ context.Context, event StepStartedEvent) {
	mutate(event.Reads)
}

func (mutatingObserver) StepFinished(_ context.Context, event StepFinishedEvent) {
	mutate(event.Reads)
	mutate(event.Writes)
}
func (mutatingObserver) RunFinished(context.Context, RunFinishedEvent) {}

func mutate(values map[string]any) {
	for key, value := range values {
		if notes, ok := value.([]string); ok && len(notes) > 0 {
			notes[0] = "changed"
		}
		values[key] = "changed"
	}
}

func readNotesStep() Step {
	return NewStep("first", refs(notesKey.Ref()), refs(decisionKey.Ref()), func(_ context.Context, s State) error {
		notes, err := notesKey.Get(s)
		if err != nil {
			return err
		}
		decisionKey.Set(s, notes[0])
		return nil
	})
}
