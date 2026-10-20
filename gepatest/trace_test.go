package gepatest_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// recordingT records what AssertTrace reports.
type recordingT struct {
	helpers int
	errors  []string
}

func (r *recordingT) Helper() { r.helpers++ }

func (r *recordingT) Errorf(format string, args ...any) {
	r.errors = append(r.errors, fmt.Sprintf(format, args...))
}

// plainT is a TestingT without a Helper method.
type plainT struct{ errors []string }

func (p *plainT) Errorf(format string, args ...any) {
	p.errors = append(p.errors, fmt.Sprintf(format, args...))
}

var (
	small = gepa.Usage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3}
	large = gepa.Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30, CachedInputTokens: 4}
)

// sampleTrace has a retried step, a Parallel whose second step started first, and a step that
// ran a nested pipeline twice. It has no times, so AssertTrace fixes the order of none of its
// executions.
func sampleTrace() gepa.PipelineTrace {
	return gepa.PipelineTrace{
		Usage: large,
		Steps: []gepa.StepTrace{
			{Step: "extract", Attempt: 1, Err: "busy", ErrKind: gepa.Transient, Usage: small},
			{Step: "extract", Attempt: 2, Usage: small},
			{Step: "fan", Attempt: 1, Usage: small, Steps: []gepa.StepTrace{
				{Step: "right", Attempt: 1, Cached: true},
				{Step: "left", Attempt: 1, Usage: small, ToolCalls: []gepa.ToolCallRecord{{Name: "a"}, {Name: "b"}}},
			}},
			{Step: "outer", Attempt: 1, Runs: []gepa.PipelineTrace{
				{Steps: []gepa.StepTrace{{Step: "inner", Attempt: 1, Err: "first run"}}},
				{Steps: []gepa.StepTrace{{Step: "inner", Attempt: 1}}},
			}},
			{Step: "optional", Attempt: 1, Skipped: true},
		},
	}
}

func sampleExpectation() gepatest.ExpectedTrace {
	return gepatest.ExpectedTrace{
		Usage: large,
		Steps: []gepatest.ExpectedStep{
			{Step: "optional", Skipped: true},
			{Step: "fan.right", Cached: true},
			{Step: "fan.left", Usage: small, ToolCalls: []string{"a", "b"}},
			{Step: "fan", Usage: small},
			{Step: "extract", Attempt: 2, Usage: small},
			{Step: "extract", Attempt: 1, Err: "busy", ErrKind: gepa.Transient, Usage: small},
			{Step: "outer"},
			{Step: "outer.inner", Failed: true},
			{Step: "outer.inner"},
		},
	}
}

func TestAssertTraceDoesNotOrderExecutionsWithoutTimes(t *testing.T) {
	recorder := &recordingT{}

	ok := gepatest.AssertTrace(recorder, sampleTrace(), sampleExpectation())

	require.True(t, ok)
	require.Empty(t, recorder.errors)
	require.Equal(t, 1, recorder.helpers, "AssertTrace marks itself as a helper")
}

func TestAssertTraceKeepsTheStartOrderOfEqualExecutions(t *testing.T) {
	want := sampleExpectation()
	for i, step := range want.Steps {
		if step.Step == "outer.inner" {
			want.Steps[i].Failed = !step.Failed
		}
	}
	recorder := &recordingT{}

	require.False(t, gepatest.AssertTrace(recorder, sampleTrace(), want),
		"the nested run that failed started first")
	require.Len(t, recorder.errors, 1)
}

func TestAssertTraceReportsADiff(t *testing.T) {
	trace := gepa.PipelineTrace{
		Err:     `pipeline step "decide": boom`,
		ErrKind: gepa.Permanent,
		Usage:   small,
		Steps: []gepa.StepTrace{
			{Step: "extract", Attempt: 1, Usage: small},
			{Step: "decide", Attempt: 1, Err: "boom", ErrKind: gepa.Permanent},
			{Step: "extra", Attempt: 1},
		},
	}
	want := gepatest.ExpectedTrace{
		Usage: large,
		Steps: []gepatest.ExpectedStep{
			{Step: "extract", Usage: small},
			{Step: "decide", Err: "kaboom"},
			{Step: "missing", Attempt: 2, Failed: true, Cached: true, Skipped: true, ToolCalls: []string{"x"}},
		},
	}
	recorder := &recordingT{}

	require.False(t, gepatest.AssertTrace(recorder, trace, want))

	require.Equal(t, []string{`gepatest: trace does not match (- want, + got)
step executions, in the order they ran:
    extract #1 ok usage{prompt=1 completion=2 total=3}
  - decide #1 failed, error containing "kaboom"
  + decide #1 failed: "boom" (permanent)
  - missing #2 failed cached skipped tools[x]
  + extra #1 ok
run:
  - run ok usage{prompt=10 completion=20 total=30 cached_input=4}
  + run failed: "pipeline step \"decide\": boom" (permanent) usage{prompt=1 completion=2 total=3}`},
		recorder.errors)
}

func TestAssertTraceReportsAMatchingRunAndNoSteps(t *testing.T) {
	recorder := &plainT{}

	require.False(t, gepatest.AssertTrace(recorder, gepa.PipelineTrace{Err: "x"}, gepatest.ExpectedTrace{Failed: true,
		ErrKind: gepa.Budget}))

	require.Equal(t, []string{`gepatest: trace does not match (- want, + got)
step executions, in the order they ran:
    (none)
run:
  - run failed (budget)
  + run failed: "x"`}, recorder.errors)

	recorder = &plainT{}
	require.False(t, gepatest.AssertTrace(recorder, gepa.PipelineTrace{
		Steps: []gepa.StepTrace{{Step: "a", Attempt: 1}},
	}, gepatest.ExpectedTrace{}))
	require.Equal(t, []string{`gepatest: trace does not match (- want, + got)
step executions, in the order they ran:
  + a #1 ok
run:
    run ok`}, recorder.errors)
}

func TestAssertTraceComparesEveryField(t *testing.T) {
	failedStep := gepa.StepTrace{Step: "s", Attempt: 1, Err: "model said no", ErrKind: gepa.Transient}
	tests := []struct {
		name  string
		got   gepa.StepTrace
		want  gepatest.ExpectedStep
		match bool
	}{
		{"zero attempt means one", gepa.StepTrace{Step: "s", Attempt: 1}, gepatest.ExpectedStep{Step: "s"}, true},
		{"step", gepa.StepTrace{Step: "s", Attempt: 1}, gepatest.ExpectedStep{Step: "t"}, false},
		{"attempt", gepa.StepTrace{Step: "s", Attempt: 2}, gepatest.ExpectedStep{Step: "s", Attempt: 1}, false},
		{
			"failed when ok",
			gepa.StepTrace{Step: "s", Attempt: 1},
			gepatest.ExpectedStep{Step: "s", Failed: true},
			false,
		},
		{"error when ok", gepa.StepTrace{Step: "s", Attempt: 1}, gepatest.ExpectedStep{Step: "s", Err: "no"}, false},
		{"ok when failed", failedStep, gepatest.ExpectedStep{Step: "s"}, false},
		{"failed", failedStep, gepatest.ExpectedStep{Step: "s", Failed: true}, true},
		{"error substring", failedStep, gepatest.ExpectedStep{Step: "s", Err: "said no"}, true},
		{"other error", failedStep, gepatest.ExpectedStep{Step: "s", Err: "said yes"}, false},
		{"kind", failedStep, gepatest.ExpectedStep{Step: "s", Err: "no", ErrKind: gepa.Transient}, true},
		{"other kind", failedStep, gepatest.ExpectedStep{Step: "s", Err: "no", ErrKind: gepa.Permanent}, false},
		{
			"kind without text",
			gepa.StepTrace{Step: "s", Attempt: 1, ErrKind: gepa.Budget},
			gepatest.ExpectedStep{Step: "s", ErrKind: gepa.Budget},
			true,
		},
		{
			"kind when ok",
			gepa.StepTrace{Step: "s", Attempt: 1},
			gepatest.ExpectedStep{Step: "s", ErrKind: gepa.Budget},
			false,
		},
		{"cached", gepa.StepTrace{Step: "s", Attempt: 1, Cached: true}, gepatest.ExpectedStep{Step: "s"}, false},
		{"skipped", gepa.StepTrace{Step: "s", Attempt: 1}, gepatest.ExpectedStep{Step: "s", Skipped: true}, false},
		{"usage", gepa.StepTrace{Step: "s", Attempt: 1, Usage: small}, gepatest.ExpectedStep{Step: "s"}, false},
		{
			"one usage count",
			gepa.StepTrace{Step: "s", Attempt: 1, Usage: gepa.Usage{CacheCreationTokens: 1}},
			gepatest.ExpectedStep{Step: "s"},
			false,
		},
		{
			"tool call order",
			gepa.StepTrace{Step: "s", Attempt: 1, ToolCalls: []gepa.ToolCallRecord{{Name: "a"}, {Name: "b"}}},
			gepatest.ExpectedStep{Step: "s", ToolCalls: []string{"b", "a"}},
			false,
		},
		{
			"missing tool calls",
			gepa.StepTrace{Step: "s", Attempt: 1, ToolCalls: []gepa.ToolCallRecord{{Name: "a"}}},
			gepatest.ExpectedStep{Step: "s"},
			false,
		},
		{
			"no tool calls, empty or nil",
			gepa.StepTrace{Step: "s", Attempt: 1, ToolCalls: []gepa.ToolCallRecord{}},
			gepatest.ExpectedStep{Step: "s", ToolCalls: []string{}},
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := &recordingT{}
			trace := gepa.PipelineTrace{Steps: []gepa.StepTrace{tt.got}}

			got := gepatest.AssertTrace(
				recorder,
				trace,
				gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{tt.want}},
			)

			require.Equal(t, tt.match, got)
			require.Equal(t, !tt.match, len(recorder.errors) == 1)
		})
	}
}

func TestAssertTraceComparesTheRun(t *testing.T) {
	failed := gepa.PipelineTrace{Err: "budget spent", ErrKind: gepa.Budget, Usage: small}
	tests := []struct {
		name  string
		got   gepa.PipelineTrace
		want  gepatest.ExpectedTrace
		match bool
	}{
		{"ok", gepa.PipelineTrace{}, gepatest.ExpectedTrace{}, true},
		{"failed", failed, gepatest.ExpectedTrace{Failed: true, Usage: small}, true},
		{"error", failed, gepatest.ExpectedTrace{Err: "spent", ErrKind: gepa.Budget, Usage: small}, true},
		{"ok when failed", failed, gepatest.ExpectedTrace{Usage: small}, false},
		{"failed when ok", gepa.PipelineTrace{}, gepatest.ExpectedTrace{Failed: true}, false},
		{"other error", failed, gepatest.ExpectedTrace{Err: "cancelled", Usage: small}, false},
		{"other kind", failed, gepatest.ExpectedTrace{Err: "spent", ErrKind: gepa.Transient, Usage: small}, false},
		{"usage", failed, gepatest.ExpectedTrace{Err: "spent", Usage: large}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := &recordingT{}

			require.Equal(t, tt.match, gepatest.AssertTrace(recorder, tt.got, tt.want))
			require.Equal(t, !tt.match, len(recorder.errors) == 1)
		})
	}
}

func TestExpectedTraceJSONRoundTrip(t *testing.T) {
	want := sampleExpectation()
	want.Err, want.ErrKind = "x", gepa.Budget

	data, err := json.Marshal(want)
	require.NoError(t, err)
	var got gepatest.ExpectedTrace
	require.NoError(t, json.Unmarshal(data, &got))

	require.Equal(t, want, got)
}

// TestAssertTraceOnARecordedParallelRun runs steps side by side many times: whatever order they
// start in, the trace matches the same expectation.
func TestAssertTraceOnARecordedParallelRun(t *testing.T) {
	fake := gepatest.NewFakeGenerator()
	task := gepa.Key[string]{Name: "task"}
	var steps []gepa.Step
	var want []gepatest.ExpectedStep
	total := gepa.Usage{}
	for i := range 6 {
		name := fmt.Sprintf("worker%d", i)
		out := gepa.Key[string]{Name: name + "_out"}
		spent := gepa.Usage{PromptTokens: i + 1, CompletionTokens: 1, TotalTokens: i + 2}
		total = total.Add(spent)
		fake.OnStep(name, gepatest.Text(`{"`+out.Name+`":"done"}`).WithUsage(spent))
		steps = append(steps, gepa.NewProgramStep(name, gepa.Predict{
			Signature: gepa.Signature{
				Name:    name,
				Inputs:  []gepa.Field{{Name: task.Name}},
				Outputs: []gepa.Field{{Name: out.Name}},
			},
			LM: fake,
		}, []gepa.KeyRef{task.Ref()}, []gepa.KeyRef{out.Ref()}))
		want = append(want, gepatest.ExpectedStep{Step: "fan." + name, Usage: spent})
	}
	fail := gepa.NewStep("fail", nil, nil, func(context.Context, gepa.State) error {
		return gepa.PermanentError(errors.New("stop here"))
	})
	want = append(want,
		gepatest.ExpectedStep{Step: "fan", Usage: total},
		gepatest.ExpectedStep{Step: "fail", Err: "stop here", ErrKind: gepa.Permanent},
	)
	recorder := &gepa.RecordingObserver{}
	pipeline, err := gepa.PipelineSpec{
		Inputs:    []gepa.KeyRef{task.Ref()},
		Steps:     []gepa.Step{gepa.Parallel("fan", steps...), fail},
		Observers: []gepa.PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)

	for range 10 {
		_, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"task": "work"}))
		require.ErrorContains(t, err, "stop here")
	}

	traces := recorder.Traces()
	require.Len(t, traces, 10)
	for _, trace := range traces {
		require.True(t, gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{
			Steps:   want,
			Err:     `pipeline step "fail": stop here`,
			ErrKind: gepa.Permanent,
			Usage:   total,
		}))
	}
}
