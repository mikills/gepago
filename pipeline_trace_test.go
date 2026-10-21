package gepa

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func stepNames(steps []StepTrace) []string {
	names := make([]string, 0, len(steps))
	for _, step := range steps {
		names = append(names, step.Step)
	}
	return names
}

func TestRecordingObserverTracesAreCopies(t *testing.T) {
	recorder := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(notesKey.Ref()),
		Steps:     []Step{readNotesStep()},
		Observers: []PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)
	_, err = pipeline.Run(context.Background(), NewState(Prediction{"notes": []string{"a"}}))
	require.NoError(t, err)

	first := recorder.Traces()
	first[0].Inputs["notes"].([]any)[0] = "changed"
	first[0].Steps[0].Reads["notes"] = "changed"
	first[0].Steps[0].Step = "changed"
	second := recorder.Traces()
	require.Equal(t, map[string]any{"notes": []any{"a"}}, second[0].Inputs)
	require.Equal(t, map[string]any{"notes": []any{"a"}}, second[0].Steps[0].Reads)
	require.Equal(t, "first", second[0].Steps[0].Step)
}

func TestRecordingObserverIgnoresUnknownSpans(t *testing.T) {
	var recorder RecordingObserver
	ctx := context.Background()
	recorder.StepStarted(ctx, StepStartedEvent{Span: UsageSpan{ID: "s1", ParentID: "nowhere"}, Step: "lost"})
	recorder.StepFinished(ctx, StepFinishedEvent{Span: UsageSpan{ID: "s1"}, Step: "lost"})
	recorder.RunFinished(ctx, RunFinishedEvent{Span: UsageSpan{ID: "r1"}})
	require.Empty(t, recorder.Traces())

	recorder.RunStarted(ctx, RunStartedEvent{Span: UsageSpan{ID: "r1", TraceID: "t1", ParentID: "elsewhere"}})
	recorder.StepStarted(ctx, StepStartedEvent{Span: UsageSpan{ID: "s1", ParentID: "nowhere"}, Step: "lost"})
	recorder.StepStarted(ctx, StepStartedEvent{Span: UsageSpan{ID: "s2", ParentID: "r1"}, Step: "kept", Attempt: 1})
	traces := recorder.Traces()
	require.Len(t, traces, 1)
	require.Equal(t, "elsewhere", traces[0].ParentID)
	require.Equal(t, []string{"kept"}, stepNames(traces[0].Steps))
	require.True(t, traces[0].EndedAt.IsZero(), "a run in progress has no end")

	recorder.RunFinished(ctx, RunFinishedEvent{Span: UsageSpan{ID: "r1", EndedAt: time.Unix(1, 0).UTC()}})
	// once its run has finished, the recorder stops tracking the trace's spans
	recorder.StepFinished(ctx, StepFinishedEvent{Span: UsageSpan{ID: "s2"}, Err: errors.New("late")})
	traces = recorder.Traces()
	require.Equal(t, time.Unix(1, 0).UTC(), traces[0].EndedAt)
	require.Empty(t, traces[0].Steps[0].Err)
}

func TestTraceValueFallsBackToText(t *testing.T) {
	require.Equal(t, "NaN", traceValue(math.NaN()))
	require.Equal(t, map[string]any{"a": float64(1)}, traceValue(map[string]int{"a": 1}))
	require.Equal(t, "panicked", traceValue(panickyJSON{}))
	require.Nil(t, traceValues(map[string]any{}))
}

// panickyJSON panics when encoded as JSON.
type panickyJSON struct{}

func (panickyJSON) MarshalJSON() ([]byte, error) { panic("cannot encode") }
func (panickyJSON) String() string               { return "panicked" }

// programFunc is a Program without components that runs a function.
type programFunc func(ctx context.Context, inputs Prediction) (Prediction, error)

func (programFunc) Validate() error          { return nil }
func (programFunc) SeedCandidate() Candidate { return Candidate{} }
func (fn programFunc) RunCandidate(ctx context.Context, _ Candidate, inputs Prediction) (Prediction, error) {
	return fn(ctx, inputs)
}

func TestTraceDatasetBuilderMatchesNestedComponents(t *testing.T) {
	trace := PipelineTrace{TraceID: "t1", Steps: []StepTrace{
		{Step: "fan", Steps: []StepTrace{
			{Step: "summarise", Attempt: 1, Writes: map[string]any{"summary": "a"}},
			{Step: "summarise", Attempt: 2, Cached: true, Writes: map[string]any{"summary": "a"}},
		}},
		{Step: "inner", Runs: []PipelineTrace{{Steps: []StepTrace{{Step: "classify", Attempt: 1}}}}},
		{Step: "route", Err: "branch condition: no label"},
		{Step: "guess", Attempt: 2},
	}}
	candidate := Candidate{"fan.summarise.instruction": "be brief", "inner.classify.instruction": "classify"}
	eval := EvaluationResult{Items: []EvaluationItem{
		{ExampleID: "value", Trace: trace},
		{ExampleID: "pointer", Trace: &trace},
		{ExampleID: "list", Trace: []PipelineTrace{trace, trace}},
		{ExampleID: "nil pointer", Trace: (*PipelineTrace)(nil)},
		{ExampleID: "other", Trace: "not a pipeline trace"},
	}}
	dataset := TraceDatasetBuilder{}.BuildReflectiveDataset(candidate, eval, []string{
		"fan.summarise.instruction",
		"inner.classify.instruction",
		"route.left.instruction",
		"guess.instruction",
		"fan.instruction",
		"summarise.instruction",
	})

	stepsOf := func(component string, item int) []ReflectiveRecord {
		t.Helper()
		record := dataset[component][item]
		require.Equal(t, eval.Items[item].ExampleID, record["example_id"])
		steps, _ := record["steps"].([]ReflectiveRecord)
		return steps
	}
	summarise := stepsOf("fan.summarise.instruction", 0)
	require.Len(t, summarise, 2)
	require.Equal(t, "summarise", summarise[0]["step"])
	require.Equal(t, "t1", summarise[0]["trace_id"])
	require.NotContains(t, summarise[0], "cached")
	require.Equal(t, true, summarise[1]["cached"])
	require.Equal(t, "be brief", dataset["fan.summarise.instruction"][0]["current_value"])
	require.Equal(t, trace, dataset["fan.summarise.instruction"][1]["trace"])
	require.Len(t, stepsOf("fan.summarise.instruction", 1), 2)
	require.Len(t, stepsOf("fan.summarise.instruction", 2), 4)
	require.Equal(t, []PipelineTrace{trace, trace}, dataset["fan.summarise.instruction"][2]["trace"])
	require.NotContains(t, dataset["fan.summarise.instruction"][3], "steps")
	require.Equal(t, (*PipelineTrace)(nil), dataset["fan.summarise.instruction"][3]["trace"])
	require.NotContains(t, dataset["fan.summarise.instruction"][4], "steps")

	require.Equal(t, "classify", stepsOf("inner.classify.instruction", 0)[0]["step"])
	route := stepsOf("route.left.instruction", 0)
	require.Len(t, route, 1)
	require.Equal(t, "branch condition: no label", route[0]["error"])
	require.NotContains(t, route[0], "details")
	guess := stepsOf("guess.instruction", 0)
	require.Len(t, guess, 1)
	require.Equal(t, 2, guess[0]["attempt"])
	require.NotContains(t, guess[0], "error")
	require.Empty(t, stepsOf("fan.instruction", 0), "a composite with nested executions owns no component")
	require.Empty(t, stepsOf("summarise.instruction", 0), "a nested step owns only keys scoped by its path")
	require.Equal(
		t,
		ReflectiveDataset{"x": {}},
		TraceDatasetBuilder{}.BuildReflectiveDataset(nil, EvaluationResult{}, []string{"x"}),
	)
}
