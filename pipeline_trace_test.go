package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// tracedPipeline runs a pipeline with a Parallel, a Fallback, a nested pipeline and a final
// step that fails, and returns its recorded trace.
func tracedPipeline(t *testing.T) PipelineTrace {
	t.Helper()
	type score struct {
		Value float64 `json:"value"`
		Label string  `json:"label"`
	}
	scoreKey := Key[score]{Name: "score"}
	ratioKey := Key[float64]{Name: "ratio"}
	inner, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			predictStep("classify", tenTokenModel(`{"decision": "approve"}`), docKey.Ref(), decisionKey.Ref()),
		},
	}.Build()
	require.NoError(t, err)
	measure := NewStep("measure", nil, refs(scoreKey.Ref(), ratioKey.Ref()), func(_ context.Context, s State) error {
		scoreKey.Set(s, score{Value: 0.5, Label: "half"})
		ratioKey.Set(s, math.NaN())
		return nil
	})
	primary := NewStep("primary", nil, refs(amountKey.Ref()), func(context.Context, State) error {
		return WithStepDetails(TransientError(errors.New("primary down")), map[string]any{"attempted": []string{"a"}})
	})
	check := NewStep("check", refs(amountKey.Ref()), refs(notesKey.Ref()), func(context.Context, State) error {
		return PermanentError(errors.New("amount too small"))
	})
	recorder := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			Parallel("fan", predictStep("summarise", tenTokenModel(`{"summary": "short"}`), docKey.Ref(), summaryKey.Ref()), measure),
			Fallback("fb", primary, writeAmount("secondary", 3)),
			NewProgramStep("inner", inner, refs(docKey.Ref()), refs(decisionKey.Ref())),
			check,
		},
		Observers: []PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)
	_, err = pipeline.Run(context.Background(), NewState(Prediction{"document": "cash 10"}))
	require.EqualError(t, err, `pipeline step "check": amount too small`)
	traces := recorder.Traces()
	require.Len(t, traces, 1)
	return traces[0]
}

func TestRecordingObserverRecordsNestedSteps(t *testing.T) {
	trace := tracedPipeline(t)
	require.NotEmpty(t, trace.TraceID)
	require.Empty(t, trace.ParentID)
	require.Equal(t, map[string]any{"document": "cash 10"}, trace.Inputs)
	require.Equal(t, `pipeline step "check": amount too small`, trace.Err)
	require.Equal(t, Permanent, trace.ErrKind)
	require.Equal(t, tenTokens.Add(tenTokens), trace.Usage)
	require.False(t, trace.EndedAt.IsZero())

	require.Equal(t, []string{"fan", "fb", "inner", "check"}, stepNames(trace.Steps))
	fan := trace.Steps[0]
	require.Equal(t, trace.SpanID, fan.ParentID)
	require.ElementsMatch(t, []string{"summarise", "measure"}, stepNames(fan.Steps))
	require.Equal(t, tenTokens, fan.Usage)
	require.Equal(t, map[string]any{
		"summary": "short",
		"score":   map[string]any{"value": 0.5, "label": "half"},
		"ratio":   "NaN",
	}, fan.Writes)

	fb := trace.Steps[1]
	require.Equal(t, []string{"primary", "secondary"}, stepNames(fb.Steps))
	require.Equal(t, "primary down", fb.Steps[0].Err)
	require.Equal(t, Transient, fb.Steps[0].ErrKind)
	require.Equal(t, map[string]any{"attempted": []any{"a"}}, fb.Steps[0].Details)
	require.Equal(t, map[string]any{"amount": float64(3)}, fb.Writes)

	nested := trace.Steps[2]
	require.Empty(t, nested.Steps)
	require.Len(t, nested.Runs, 1)
	require.Equal(t, nested.SpanID, nested.Runs[0].ParentID)
	require.Equal(t, trace.TraceID, nested.Runs[0].TraceID)
	require.Equal(t, []string{"classify"}, stepNames(nested.Runs[0].Steps))
	require.Equal(t, tenTokens, nested.Runs[0].Usage)

	check := trace.Steps[3]
	require.Equal(t, "amount too small", check.Err)
	require.Equal(t, map[string]any{"amount": float64(3)}, check.Reads)
	require.Nil(t, check.Writes)
}

func stepNames(steps []StepTrace) []string {
	names := make([]string, 0, len(steps))
	for _, step := range steps {
		names = append(names, step.Step)
	}
	return names
}

func TestPipelineTraceSurvivesAJSONRoundTrip(t *testing.T) {
	trace := tracedPipeline(t)
	data, err := json.Marshal(trace)
	require.NoError(t, err)
	var decoded PipelineTrace
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, trace, decoded)
	again, err := json.Marshal(decoded)
	require.NoError(t, err)
	require.JSONEq(t, string(data), string(again))
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

func TestTraceDatasetBuilderRecordsAFailedStepWithItsReads(t *testing.T) {
	invalid := GeneratorFunc(func(context.Context, string) (Generation, error) {
		return Generation{Text: `{"other": "x"}`, Usage: tenTokens}, nil
	})
	recorder := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			lengthStep(),
			predictStep("summarise", invalid, docKey.Ref(), summaryKey.Ref()),
		},
		Observers: []PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)
	_, err = pipeline.Run(context.Background(), NewState(Prediction{"document": "cash 10"}))
	require.ErrorIs(t, err, ErrInvalidOutput)

	candidate := pipeline.SeedCandidate()
	component := "summarise." + InstructionComponent
	require.Contains(t, candidate, component)
	traces := recorder.Traces()
	dataset := TraceDatasetBuilder{}.BuildReflectiveDataset(candidate, traces, []string{component, "missing.instruction"})

	require.Equal(t, []ReflectiveRecord{}, dataset["missing.instruction"])
	records := dataset[component]
	require.Len(t, records, 1)
	record := records[0]
	require.Equal(t, component, record["component"])
	require.Equal(t, candidate[component], record["current_value"])
	require.Equal(t, traces[0].TraceID, record["trace_id"])
	require.Equal(t, "summarise", record["step"])
	require.Equal(t, 1, record["attempt"])
	require.Equal(t, map[string]any{"document": "cash 10"}, record["reads"])
	require.Contains(t, record["error"], "invalid programme output")
	require.Equal(t, tenTokens, record["usage"])
	require.NotNil(t, record["details"])
	require.NotContains(t, record, "cached")
}

func TestTraceDatasetBuilderMatchesNestedComponents(t *testing.T) {
	trace := PipelineTrace{TraceID: "t1", Steps: []StepTrace{
		{Step: "fan", Steps: []StepTrace{
			{Step: "summarise", Attempt: 1, Writes: map[string]any{"summary": "a"}},
			{Step: "summarise", Attempt: 2, Cached: true, Writes: map[string]any{"summary": "a"}},
		}},
		{Step: "inner", Runs: []PipelineTrace{{Steps: []StepTrace{{Step: "classify", Attempt: 1}}}}},
		{Step: "route", Err: "branch condition: no label"},
		{Step: "guess", Attempt: 2, Skipped: true},
	}}
	candidate := Candidate{"fan.summarise.instruction": "be brief", "inner.classify.instruction": "classify"}
	dataset := TraceDatasetBuilder{}.BuildReflectiveDataset(candidate, []PipelineTrace{trace, trace}, []string{
		"fan.summarise.instruction",
		"inner.classify.instruction",
		"route.left.instruction",
		"guess.instruction",
		"fan.instruction",
		"summarise.instruction",
	})

	summarise := dataset["fan.summarise.instruction"]
	require.Len(t, summarise, 4)
	require.Equal(t, "summarise", summarise[0]["step"])
	require.Equal(t, "be brief", summarise[0]["current_value"])
	require.NotContains(t, summarise[0], "cached")
	require.Equal(t, true, summarise[1]["cached"])
	require.Equal(t, "classify", dataset["inner.classify.instruction"][0]["step"])
	require.Len(t, dataset["inner.classify.instruction"], 2)
	route := dataset["route.left.instruction"]
	require.Len(t, route, 2)
	require.Equal(t, "branch condition: no label", route[0]["error"])
	require.NotContains(t, route[0], "details")
	require.Equal(t, true, dataset["guess.instruction"][0]["skipped"])
	require.Empty(t, dataset["fan.instruction"], "a composite with nested executions owns no component")
	require.Empty(t, dataset["summarise.instruction"], "a nested step owns only keys scoped by its path")
}
