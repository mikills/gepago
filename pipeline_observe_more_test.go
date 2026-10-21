package gepa

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestPipelineObserverSharedWithANestedPipelineGetsEachEventOnce(t *testing.T) {
	recorder := &RecordingObserver{}
	log := &eventLog{}
	inner, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			predictStep("summarise", tenTokenModel(`{"summary": "short"}`), docKey.Ref(), summaryKey.Ref()),
		},
		Observers: []PipelineObserver{recorder, log, recorder},
	}.Build()
	require.NoError(t, err)
	outer, err := PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Steps:     []Step{NewProgramStep("inner", inner, refs(docKey.Ref()), refs(summaryKey.Ref()))},
		Observers: []PipelineObserver{log, recorder},
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
	}, log.Lines())
	traces := recorder.Traces()
	require.Len(t, traces, 1)
	require.Len(t, traces[0].Steps, 1)
	require.Len(t, traces[0].Steps[0].Runs, 1)
	require.False(t, traces[0].Steps[0].Runs[0].EndedAt.IsZero())
	require.Equal(t, tenTokens, traces[0].Usage)
}

// sliceObserver is an observer of a type that cannot be compared.
type sliceObserver []*eventLog

func (o sliceObserver) RunStarted(ctx context.Context, event RunStartedEvent) {
	o[0].RunStarted(ctx, event)
}
func (o sliceObserver) StepStarted(ctx context.Context, event StepStartedEvent) {
	o[0].StepStarted(ctx, event)
}

func (o sliceObserver) StepFinished(ctx context.Context, event StepFinishedEvent) {
	o[0].StepFinished(ctx, event)
}
func (o sliceObserver) RunFinished(ctx context.Context, event RunFinishedEvent) {
	o[0].RunFinished(ctx, event)
}

// boxedObserver is comparable by type, but == panics when it holds an uncomparable value.
type boxedObserver struct {
	*eventLog
	box any
}

func TestSameObserver(t *testing.T) {
	log := &eventLog{}
	require.True(t, sameObserver(log, log))
	require.False(t, sameObserver(log, &eventLog{}))
	require.False(t, sameObserver(sliceObserver{log}, sliceObserver{log}))
	require.False(t, sameObserver(boxedObserver{log, []int{1}}, boxedObserver{log, []int{1}}))
	require.True(t, sameObserver(boxedObserver{log, 1}, boxedObserver{log, 1}))
	require.False(t, sameObserver(log, boxedObserver{log, 1}))

	// an observer that cannot be compared is called as often as it is listed
	pipeline, err := PipelineSpec{
		Steps:     []Step{writeAmount("write", 1)},
		Observers: []PipelineObserver{sliceObserver{log}, sliceObserver{log}},
	}.Build()
	require.NoError(t, err)
	_, err = pipeline.Run(context.Background(), NewState(nil))
	require.NoError(t, err)
	require.Len(t, log.Lines(), 8)
}

// hookedObserver is an observer value that cannot be compared, because it has a func field.
type hookedObserver struct {
	log  *eventLog
	hook func()
}

func (o hookedObserver) RunStarted(ctx context.Context, event RunStartedEvent) {
	o.hook()
	o.log.RunStarted(ctx, event)
}

func (o hookedObserver) StepStarted(ctx context.Context, event StepStartedEvent) {
	o.hook()
	o.log.StepStarted(ctx, event)
}

func (o hookedObserver) StepFinished(ctx context.Context, event StepFinishedEvent) {
	o.hook()
	o.log.StepFinished(ctx, event)
}

func (o hookedObserver) RunFinished(ctx context.Context, event RunFinishedEvent) {
	o.hook()
	o.log.RunFinished(ctx, event)
}

// TestPipelineObserverThatCannotBeComparedIsCalledForEachListing pins the documented
// behaviour: only comparable observers are de-duplicated.
func TestPipelineObserverThatCannotBeComparedIsCalledForEachListing(t *testing.T) {
	log := &eventLog{}
	observer := hookedObserver{log: log, hook: func() {}}
	inner, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			predictStep("summarise", tenTokenModel(`{"summary": "short"}`), docKey.Ref(), summaryKey.Ref()),
		},
		Observers: []PipelineObserver{observer},
	}.Build()
	require.NoError(t, err)
	outer, err := PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Steps:     []Step{NewProgramStep("inner", inner, refs(docKey.Ref()), refs(summaryKey.Ref()))},
		Observers: []PipelineObserver{observer},
	}.Build()
	require.NoError(t, err)

	_, err = outer.Run(context.Background(), NewState(Prediction{"document": "cash 10"}))
	require.NoError(t, err)
	require.Equal(t, []string{
		"start run",
		"start inner#1 in run",
		"start run in inner#1",
		"start run in inner#1",
		"start summarise#1 in run in inner#1",
		"start summarise#1 in run in inner#1",
		"finish summarise#1",
		"finish summarise#1",
		"finish run in inner#1",
		"finish run in inner#1",
		"finish inner#1",
		"finish run",
	}, log.Lines())
}

func TestPipelineObserverMergeFailureFailsTheStep(t *testing.T) {
	merge := func(err error, panicked bool) MergeRule {
		return MergeWith(amountKey, func(int, int) (int, error) {
			if panicked {
				panic("merge exploded")
			}
			return 0, err
		})
	}
	tests := []struct {
		name    string
		second  Step
		rule    MergeRule
		want    []string
		wantErr string
	}{
		{
			name:   "a plain step",
			second: writeAmount("second", 2),
			rule:   merge(errors.New("amounts disagree"), false),
			want: []string{
				"start run",
				"start first#1 in run",
				"finish first#1",
				"start second#1 in run",
				`finish second#1 err: merge state key "amount": amounts disagree`,
				`finish run err: pipeline step "second": merge state key "amount": amounts disagree`,
			},
			wantErr: `merge state key "amount": amounts disagree`,
		},
		{
			name:   "a merge that panics",
			second: writeAmount("second", 2),
			rule:   merge(nil, true),
			want: []string{
				"start run",
				"start first#1 in run",
				"finish first#1",
				"start second#1 in run",
				"finish second#1 err: step panicked: merge exploded",
				`finish run err: pipeline step "second": step panicked: merge exploded`,
			},
			wantErr: "step panicked: merge exploded",
		},
		{
			name:   "a step under a policy adds a failed execution",
			second: WithPolicy(writeAmount("second", 2), StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}}),
			rule:   merge(errors.New("amounts disagree"), false),
			want: []string{
				"start run",
				"start first#1 in run",
				"finish first#1",
				"start second#1 in run",
				"finish second#1",
				"start second#2 in run",
				`finish second#2 err: merge state key "amount": amounts disagree`,
				`finish run err: pipeline step "second": merge state key "amount": amounts disagree`,
			},
			wantErr: `merge state key "amount": amounts disagree`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := &RecordingObserver{}
			log := &eventLog{}
			pipeline, err := PipelineSpec{
				Steps:     []Step{writeAmount("first", 1), tt.second},
				Merges:    []MergeRule{tt.rule},
				Observers: []PipelineObserver{recorder, log},
			}.Build()
			require.NoError(t, err)

			got, err := pipeline.Run(context.Background(), NewState(nil))
			require.EqualError(t, err, `pipeline step "second": `+tt.wantErr)
			require.Equal(t, Prediction{"amount": 1}, got.Prediction())
			require.Equal(t, tt.want, log.Lines())
			trace := recorder.Traces()[0]
			last := trace.Steps[len(trace.Steps)-1]
			require.Equal(t, "second", last.Step)
			require.Equal(t, tt.wantErr, last.Err)
			require.Nil(t, last.Writes)
		})
	}
}

func TestPipelineObserverUsageOfARetriedFailedCall(t *testing.T) {
	var calls atomic.Int32
	lm := GeneratorFunc(func(context.Context, string) (Generation, error) {
		if calls.Add(1) == 1 {
			return Generation{Usage: tenTokens}, TransientError(errors.New("provider broke"))
		}
		return Generation{Text: `{"summary": "short"}`, Usage: tenTokens}, nil
	})
	recorder := &RecordingObserver{}
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{WithPolicy(
			predictStep("summarise", lm, docKey.Ref(), summaryKey.Ref()),
			StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}},
		)},
		Observers: []PipelineObserver{recorder, log},
	}.Build()
	require.NoError(t, err)

	ctx, collector := WithUsageCollector(context.Background())
	_, err = pipeline.Run(ctx, NewState(Prediction{"document": "cash 10"}))
	require.NoError(t, err)
	failed := log.finishedStep(t, "summarise", 1)
	require.ErrorContains(t, failed.Err, "provider broke")
	require.Equal(t, tenTokens, failed.Usage)
	require.Equal(t, tenTokens, log.finishedStep(t, "summarise", 2).Usage)
	require.Equal(t, tenTokens.Add(tenTokens), collector.Usage())
	require.Equal(t, collector.Usage(), log.runsFinished[0].Usage)
	trace := recorder.Traces()[0]
	require.Equal(t, collector.Usage(), trace.Usage)
	require.Equal(t, collector.Usage(), trace.Steps[0].Usage.Add(trace.Steps[1].Usage))
}

type ctxValueKey struct{}

// spendingObserver records usage and starts a pipeline with the context of every
// StepFinished, and notes what that context carries.
type spendingObserver struct {
	eventLog
	nested Pipeline
	errs   []error
	values []any
}

func (o *spendingObserver) StepFinished(ctx context.Context, event StepFinishedEvent) {
	RecordUsage(ctx, tenTokens)
	_, err := o.nested.Run(ctx, NewState(nil))
	o.errs = append(o.errs, err, ctx.Err())
	o.values = append(o.values, ctx.Value(ctxValueKey{}))
	o.eventLog.StepFinished(ctx, event)
}

func TestPipelineObserverSpendingIsIndependentOfTheRun(t *testing.T) {
	spendStep := func(name string) Step {
		return NewStep(name, nil, refs(Key[int]{Name: name}.Ref()), func(ctx context.Context, s State) error {
			RecordUsage(ctx, tenTokens)
			Key[int]{Name: name}.Set(s, 1)
			return nil
		})
	}
	nested, err := PipelineSpec{Steps: []Step{spendStep("inside")}, Budget: RunBudget{MaxTokens: 15}}.Build()
	require.NoError(t, err)
	observer := &spendingObserver{nested: nested}
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			predictStep("summarise", tenTokenModel(`{"summary": "short"}`), docKey.Ref(), summaryKey.Ref()),
			spendStep("spend"),
		},
		Observers: []PipelineObserver{observer},
		Budget:    RunBudget{MaxTokens: 15},
	}.Build()
	require.NoError(t, err)

	ctx, collector := WithUsageCollector(context.WithValue(context.Background(), ctxValueKey{}, "kept"))
	_, err = pipeline.Run(ctx, NewState(Prediction{"document": "cash 10"}))
	// had the observer's 10 tokens after summarise counted, summarise would have failed
	require.ErrorContains(t, err, `pipeline step "spend": run budget MaxTokens 15 exceeded`)
	require.True(t, IsBudget(err))
	require.Equal(t, tenTokens.Add(tenTokens), collector.Usage(), "the observer's usage is not the run's")
	require.Equal(t, tenTokens.Add(tenTokens), observer.runsFinished[0].Usage)
	require.Equal(t, []error{nil, nil, nil, nil}, observer.errs, "the run's budget does not end the observer's context")
	require.Equal(t, []any{"kept", "kept"}, observer.values)
}

// blockingObserver waits in StepStarted until its context ends, and records the error its
// context reports in each StepFinished.
type blockingObserver struct {
	eventLog
	block    bool
	finished []error
	last     context.Context
}

func (o *blockingObserver) StepStarted(ctx context.Context, event StepStartedEvent) {
	if o.block {
		<-ctx.Done()
	}
	o.eventLog.StepStarted(ctx, event)
}

func (o *blockingObserver) StepFinished(ctx context.Context, event StepFinishedEvent) {
	o.finished = append(o.finished, ctx.Err())
	o.last = ctx
	o.eventLog.StepFinished(ctx, event)
}

func TestPipelineObserverContextEndsWithTheRunsMaxDuration(t *testing.T) {
	observer := &blockingObserver{block: true}
	pipeline, err := PipelineSpec{
		Steps:     []Step{writeAmount("write", 1)},
		Observers: []PipelineObserver{observer},
		Budget:    RunBudget{MaxDuration: 100 * time.Millisecond},
	}.Build()
	require.NoError(t, err)

	done := make(chan error, 1)
	go func() {
		_, err := pipeline.Run(context.Background(), NewState(nil))
		done <- err
	}()
	select {
	case err := <-done:
		require.True(t, IsBudget(err), "%v", err)
		require.ErrorContains(t, err, "MaxDuration")
	case <-time.After(5 * time.Second):
		t.Fatal("the run is still blocked after its MaxDuration")
	}
	require.Equal(t, []error{context.DeadlineExceeded}, observer.finished)
	deadline, ok := observer.last.Deadline()
	require.True(t, ok, "the observer's context carries the run's deadline")
	require.False(t, deadline.IsZero())
}

// wrappedContext is a context type of its own, which the context package cannot link to its
// children without a goroutine.
type wrappedContext struct{ context.Context }

func TestPipelineObserverContextsLeakNoGoroutines(t *testing.T) {
	inner, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []Step{lengthStep()},
		Budget: RunBudget{MaxDuration: time.Hour},
	}.Build()
	require.NoError(t, err)
	outer, err := PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Steps:     []Step{NewProgramStep("inner", inner, refs(docKey.Ref()), refs(amountKey.Ref()))},
		Observers: []PipelineObserver{&RecordingObserver{}},
		Budget:    RunBudget{MaxDuration: time.Hour},
	}.Build()
	require.NoError(t, err)
	parent, cancel := context.WithCancel(context.Background())
	defer cancel()
	caller := wrappedContext{parent}

	runtime.GC()
	before := runtime.NumGoroutine()
	for range 500 {
		_, err := outer.Run(caller, NewState(Prediction{"document": "cash 10"}))
		require.NoError(t, err)
	}
	require.Eventually(t, func() bool {
		runtime.GC()
		return runtime.NumGoroutine() <= before+2
	}, 2*time.Second, 20*time.Millisecond)
}

func TestPipelineObserverContextEndsWithItsRun(t *testing.T) {
	observer := &blockingObserver{}
	pipeline, err := PipelineSpec{
		Steps:     []Step{writeAmount("write", 1)},
		Observers: []PipelineObserver{observer},
	}.Build()
	require.NoError(t, err)
	_, err = pipeline.Run(context.Background(), NewState(nil))
	require.NoError(t, err)
	require.Equal(t, []error{nil}, observer.finished)
	require.ErrorIs(t, observer.last.Err(), context.Canceled, "the context ends once the run has finished")
}

func TestPipelineObserverContextEndsWithTheCallersContext(t *testing.T) {
	observer := &blockingObserver{block: true}
	pipeline, err := PipelineSpec{
		Steps:     []Step{writeAmount("write", 1)},
		Observers: []PipelineObserver{observer},
	}.Build()
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		_, err := pipeline.Run(ctx, NewState(nil))
		done <- err
	}()
	select {
	case err := <-done:
		require.ErrorIs(t, err, context.DeadlineExceeded)
	case <-time.After(5 * time.Second):
		t.Fatal("the run is still blocked after its context expired")
	}
	require.Equal(t, []error{context.DeadlineExceeded}, observer.finished)
}

func TestPipelineObserverContextOutlivesStepTimeouts(t *testing.T) {
	observer := &blockingObserver{}
	slow := NewStep("slow", nil, refs(amountKey.Ref()), func(ctx context.Context, _ State) error {
		<-ctx.Done()
		return ctx.Err()
	})
	pipeline, err := PipelineSpec{
		Steps: []Step{
			WithPolicy(slow, StepPolicy{Timeout: 5 * time.Millisecond}),
			writeNotes("after", "x"),
		},
		Transitions: []Transition{{From: "slow", OnError: true, To: "after"}},
		Defaults:    []Default{DefaultValue(amountKey, 0)},
		Observers:   []PipelineObserver{observer},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(nil))
	require.NoError(t, err)
	require.ErrorIs(t, observer.finishedStep(t, "slow", 1).Err, context.DeadlineExceeded)
	require.Equal(t, []error{nil, nil}, observer.finished, "a step's timeout does not end the observer's context")
}

// TestPipelineObserverStress runs many runs of a Parallel of retried, cached steps, each failing
// its first attempt in a run, with a redactor and a recorder at once; run it with -race.
func TestPipelineObserverStress(t *testing.T) {
	var failed sync.Map
	branch := func(i int) Step {
		key := Key[string]{Name: fmt.Sprintf("out%d", i)}
		flaky := NewStep(
			fmt.Sprintf("s%d", i),
			refs(docKey.Ref()),
			refs(key.Ref()),
			func(ctx context.Context, s State) error {
				RecordUsage(ctx, tenTokens)
				rc, _ := RunContextFrom(ctx)
				if _, again := failed.LoadOrStore(fmt.Sprintf("%s/%d", rc.RunID, i), true); !again {
					return errRateLimited
				}
				document, err := docKey.Get(s)
				if err != nil {
					return err
				}
				key.Set(s, document)
				return nil
			},
		)
		return WithPolicy(WithCache(flaky, nil), StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}})
	}
	var branches []Step
	for i := range 6 {
		branches = append(branches, branch(i))
	}
	recorder := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Steps:     []Step{Parallel("fan", branches...)},
		Observers: []PipelineObserver{recorder, &overlapDetector{}},
		Redactor: func(key string, value any) any {
			if key == "document" {
				return "[doc]"
			}
			return value
		},
	}.Build()
	require.NoError(t, err)

	const runs = 8
	var wg sync.WaitGroup
	for range runs {
		wg.Go(func() {
			_, err := pipeline.Run(context.Background(), NewState(Prediction{"document": "secret"}))
			require.NoError(t, err)
		})
	}
	wg.Wait()

	traces := recorder.Traces()
	require.Len(t, traces, runs)
	for _, trace := range traces {
		require.False(t, trace.EndedAt.IsZero())
		require.Equal(t, trace.Usage, trace.Steps[0].Usage)
		walkStepTraces(trace.Steps, "", func(_ string, step StepTrace) {
			require.False(t, step.EndedAt.IsZero(), step.Step)
			require.Equal(t, map[string]any{"document": "[doc]"}, step.Reads, step.Step)
		})
	}
}
