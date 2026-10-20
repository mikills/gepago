package gepatest_test

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

var epoch = time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)

// timed returns step with times start and end milliseconds after epoch; an end of -1 leaves it
// unfinished.
func timed(step gepa.StepTrace, start, end int) gepa.StepTrace {
	step.StartedAt = epoch.Add(time.Duration(start) * time.Millisecond)
	if end >= 0 {
		step.EndedAt = epoch.Add(time.Duration(end) * time.Millisecond)
	}
	return step
}

// timedRun returns a nested run with the given steps and times.
func timedRun(start, end int, steps ...gepa.StepTrace) gepa.PipelineTrace {
	return gepa.PipelineTrace{
		StartedAt: epoch.Add(time.Duration(start) * time.Millisecond),
		EndedAt:   epoch.Add(time.Duration(end) * time.Millisecond),
		Steps:     steps,
	}
}

// orderedTrace has sequential steps, a Parallel whose steps did not overlap and nest steps of
// their own, a Fallback, a retried step, two nested runs that overlap and an unfinished step.
func orderedTrace() gepa.PipelineTrace {
	step := func(name string, attempt int) gepa.StepTrace { return gepa.StepTrace{Step: name, Attempt: attempt} }
	concurrent := func(step gepa.StepTrace) gepa.StepTrace {
		step.Concurrent = true
		return step
	}
	failed := func(step gepa.StepTrace) gepa.StepTrace {
		step.Failed = true
		return step
	}
	fan := timed(step("fan", 1), 20, 50)
	branchA := concurrent(timed(step("a", 1), 21, 30))
	branchA.Steps = []gepa.StepTrace{timed(step("a1", 1), 22, 25), timed(step("a2", 1), 26, 29)}
	branchB := concurrent(timed(step("b", 1), 31, 40))
	branchB.Steps = []gepa.StepTrace{timed(step("b1", 1), 32, 35)}
	fan.Steps = []gepa.StepTrace{branchA, branchB}
	fb := timed(step("fb", 1), 50, 60)
	fb.Steps = []gepa.StepTrace{failed(timed(step("primary", 1), 51, 54)), timed(step("secondary", 1), 55, 59)}
	custom := timed(step("custom", 1), 70, 90)
	custom.Runs = []gepa.PipelineTrace{
		timedRun(71, 80, timed(step("x", 1), 72, 79)),
		timedRun(75, 85, timed(step("y", 1), 76, 84)),
	}
	return gepa.PipelineTrace{Steps: []gepa.StepTrace{
		timed(step("first", 1), 0, 10),
		timed(step("second", 1), 10, 20),
		fan,
		fb,
		failed(timed(step("flaky", 1), 60, 61)),
		timed(step("flaky", 2), 62, 63),
		custom,
		timed(step("open", 1), 95, -1),
		timed(step("last", 1), 96, 97),
	}}
}

// orderedSteps lists the executions of orderedTrace in the order they ran.
func orderedSteps() []gepatest.ExpectedStep {
	return []gepatest.ExpectedStep{
		{Step: "first"},
		{Step: "second"},
		{Step: "fan"},
		{Step: "fan.a"},
		{Step: "fan.a.a1"},
		{Step: "fan.a.a2"},
		{Step: "fan.b"},
		{Step: "fan.b.b1"},
		{Step: "fb"},
		{Step: "fb.primary", Failed: true},
		{Step: "fb.secondary"},
		{Step: "flaky", Attempt: 1, Failed: true},
		{Step: "flaky", Attempt: 2},
		{Step: "custom"},
		{Step: "custom.x"},
		{Step: "custom.y"},
		{Step: "open"},
		{Step: "last"},
	}
}

// moved returns steps with the execution labelled from moved to just before the one labelled
// before.
func moved(t *testing.T, steps []gepatest.ExpectedStep, from, before string) []gepatest.ExpectedStep {
	t.Helper()
	labelOf := func(step gepatest.ExpectedStep) string { return fmt.Sprintf("%s #%d", step.Step, max(step.Attempt, 1)) }
	i := slices.IndexFunc(steps, func(step gepatest.ExpectedStep) bool { return labelOf(step) == from })
	require.GreaterOrEqual(t, i, 0, from)
	step := steps[i]
	steps = slices.Delete(slices.Clone(steps), i, i+1)
	j := slices.IndexFunc(steps, func(step gepatest.ExpectedStep) bool { return labelOf(step) == before })
	require.GreaterOrEqual(t, j, 0, before)
	return slices.Insert(steps, j, step)
}

func TestAssertTraceChecksTheOrderExecutionsRanIn(t *testing.T) {
	tests := []struct {
		name         string
		from, before string
		match        bool
	}{
		{"a step that started as the one before it ended", "second #1", "first #1", false},
		{"an execution before the one it is nested in", "fan.a #1", "fan #1", true},
		{"a step nested before a later step", "fan.a.a1 #1", "second #1", false},
		{"steps of a Parallel", "fan.b #1", "fan.a #1", true},
		{"steps nested in different steps of a Parallel", "fan.b.b1 #1", "fan.a.a1 #1", true},
		{"sequential steps in one step of a Parallel", "fan.a.a2 #1", "fan.a.a1 #1", false},
		{"a Fallback's second step before its first", "fb.secondary #1", "fb.primary #1", false},
		{"a retry before its first attempt", "flaky #2", "flaky #1", false},
		{"steps of overlapping runs", "custom.y #1", "custom.x #1", true},
		{"an unfinished step", "last #1", "open #1", true},
		{"a step before an unfinished one", "open #1", "custom #1", false},
	}
	recorder := &recordingT{}
	require.True(t, gepatest.AssertTrace(recorder, orderedTrace(), gepatest.ExpectedTrace{Steps: orderedSteps()}),
		recorder.errors)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := &recordingT{}
			want := gepatest.ExpectedTrace{Steps: moved(t, orderedSteps(), tt.from, tt.before)}

			got := gepatest.AssertTrace(recorder, orderedTrace(), want)

			require.Equal(t, tt.match, got, recorder.errors)
			if !tt.match {
				require.Len(t, recorder.errors, 1)
				require.Contains(t, recorder.errors[0], "order:\n")
			}
		})
	}
}

func TestAssertTraceReportsAnExecutionListedOutOfOrder(t *testing.T) {
	trace := gepa.PipelineTrace{Steps: []gepa.StepTrace{
		timed(gepa.StepTrace{Step: "first", Attempt: 1}, 0, 1),
		timed(gepa.StepTrace{Step: "second", Attempt: 1}, 2, 3),
		timed(gepa.StepTrace{Step: "third", Attempt: 1}, 4, 5),
	}}
	recorder := &plainT{}

	require.False(t, gepatest.AssertTrace(recorder, trace, gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{
		{Step: "third"},
		{Step: "first"},
		{Step: "second"},
	}}))

	require.Equal(t, []string{`gepatest: trace does not match (- want, + got)
step executions, in the order they ran:
  - third #1 ok
    first #1 ok
    second #1 ok
  + third #1 ok
order:
    want lists third #1 before first #1, but first #1 finished before third #1 started
    want lists third #1 before second #1, but second #1 finished before third #1 started
run:
    run ok`}, recorder.errors)
}

func TestAssertTraceCapsTheOrderViolationsItNames(t *testing.T) {
	var trace gepa.PipelineTrace
	var want []gepatest.ExpectedStep
	for i := range 6 {
		name := fmt.Sprintf("s%d", i)
		trace.Steps = append(trace.Steps, timed(gepa.StepTrace{Step: name, Attempt: 1}, 2*i, 2*i+1))
		want = append([]gepatest.ExpectedStep{{Step: name}}, want...)
	}
	recorder := &plainT{}

	require.False(t, gepatest.AssertTrace(recorder, trace, gepatest.ExpectedTrace{Steps: want}))

	require.Len(t, recorder.errors, 1)
	require.Equal(t, 10, strings.Count(recorder.errors[0], "\n    want lists "), "15 violations, 10 named")
	require.Contains(t, recorder.errors[0], "\n    and 5 more\n")
}

func TestAssertTraceReportsTimesThatContradictEachOther(t *testing.T) {
	// each step ended before it started, so each finished before the other started.
	trace := gepa.PipelineTrace{Steps: []gepa.StepTrace{
		timed(gepa.StepTrace{Step: "late", Attempt: 1}, 10, 0),
		timed(gepa.StepTrace{Step: "early", Attempt: 1}, 5, 0),
	}}
	recorder := &plainT{}

	require.False(t, gepatest.AssertTrace(recorder, trace, gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{
		{Step: "late"},
		{Step: "early"},
	}}))

	require.Equal(t, []string{`gepatest: trace does not match (- want, + got)
step executions, in the order they ran:
    late #1 ok
    early #1 ok
order:
    want lists late #1 before early #1, but early #1 finished before late #1 started
run:
    run ok`}, recorder.errors)
}

// orderPipeline runs a step, a Fallback whose primary fails, a step retried once and a Parallel
// of two steps, and records its runs with recorder.
func orderPipeline(t *testing.T, recorder *gepa.RecordingObserver) gepa.Pipeline {
	t.Helper()
	noop := func(name string) gepa.Step {
		return gepa.NewStep(name, nil, nil, func(context.Context, gepa.State) error { return nil })
	}
	var mu sync.Mutex
	attempts := 0
	flaky := gepa.NewStep("flaky", nil, nil, func(context.Context, gepa.State) error {
		mu.Lock()
		defer mu.Unlock()
		if attempts++; attempts%2 == 1 {
			return gepa.TransientError(errors.New("busy"))
		}
		return nil
	})
	primary := gepa.NewStep("primary", nil, nil, func(context.Context, gepa.State) error {
		return gepa.TransientError(errors.New("primary down"))
	})
	pipeline, err := gepa.PipelineSpec{
		Steps: []gepa.Step{
			noop("first"),
			gepa.Fallback("fb", primary, noop("secondary")),
			gepa.WithPolicy(flaky, gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 2}}),
			gepa.Parallel("fan", noop("left"), noop("right")),
		},
		Observers: []gepa.PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)
	return pipeline
}

func TestAssertTraceChecksTheOrderOfARecordedRun(t *testing.T) {
	recorder := &gepa.RecordingObserver{}
	pipeline := orderPipeline(t, recorder)
	for range 20 {
		_, err := pipeline.Run(context.Background(), gepa.NewState(nil))
		require.NoError(t, err)
	}
	inOrder := []gepatest.ExpectedStep{
		{Step: "first"},
		{Step: "fb"},
		{Step: "fb.primary", Err: "primary down"},
		{Step: "fb.secondary"},
		{Step: "flaky", Attempt: 1, Err: "busy"},
		{Step: "flaky", Attempt: 2},
		{Step: "fan"},
		{Step: "fan.left"},
		{Step: "fan.right"},
	}

	for _, trace := range recorder.Traces() {
		require.True(t, gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: inOrder}))
		require.True(t, gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{
			Steps: moved(t, inOrder, "fan.right #1", "fan.left #1"),
		}), "the steps of a Parallel pass in either order, whichever started first")
		for _, swap := range [][2]string{
			{"fb.secondary #1", "fb.primary #1"},
			{"flaky #2", "flaky #1"},
			{"fb #1", "first #1"},
		} {
			recorder := &recordingT{}
			require.False(t, gepatest.AssertTrace(recorder, trace, gepatest.ExpectedTrace{
				Steps: moved(t, inOrder, swap[0], swap[1]),
			}), "%s listed before %s", swap[0], swap[1])
		}
	}
}

func TestAssertTraceSeesAFailureWithoutAMessage(t *testing.T) {
	recorder := &gepa.RecordingObserver{}
	silent := gepa.NewStep("silent", nil, nil, func(context.Context, gepa.State) error { return errors.New("") })
	pipeline, err := gepa.PipelineSpec{Steps: []gepa.Step{silent}, Observers: []gepa.PipelineObserver{recorder}}.Build()
	require.NoError(t, err)
	_, err = pipeline.Run(context.Background(), gepa.NewState(nil))
	require.Error(t, err)
	trace := recorder.Traces()[0]
	require.True(t, trace.Steps[0].Failed)
	require.Empty(t, trace.Steps[0].Err)

	for _, tt := range []struct {
		name  string
		want  gepatest.ExpectedTrace
		match bool
	}{
		{"failed", gepatest.ExpectedTrace{Failed: true, Steps: []gepatest.ExpectedStep{{Step: "silent", Failed: true}}}, true},
		{"step ok", gepatest.ExpectedTrace{Failed: true, Steps: []gepatest.ExpectedStep{{Step: "silent"}}}, false},
		{"run ok", gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{{Step: "silent", Failed: true}}}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			recorder := &recordingT{}
			require.Equal(t, tt.match, gepatest.AssertTrace(recorder, trace, tt.want), recorder.errors)
		})
	}
}
