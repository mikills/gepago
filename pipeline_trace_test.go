package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
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

// TestTraceDatasetBuilderOverProgramEvaluator evaluates a pipeline whose model step fails with
// an invalid output, and checks that the reflection record names the failed step and carries
// its reads as the pipeline's Redactor redacts them.
func TestTraceDatasetBuilderOverProgramEvaluator(t *testing.T) {
	model := GeneratorFunc(func(_ context.Context, prompt string) (Generation, error) {
		if strings.Contains(prompt, "broken") {
			return Generation{Text: `{"other": "x"}`, Usage: tenTokens}, nil
		}
		return Generation{Text: `{"summary": "short"}`, Usage: tenTokens}, nil
	})
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []Step{lengthStep(), predictStep("summarise", model, docKey.Ref(), summaryKey.Ref())},
		Redactor: func(key string, value any) any {
			if key == "document" {
				return "[doc]"
			}
			return value
		},
	}.Build()
	require.NoError(t, err)
	evaluator := ProgramEvaluator{
		Program:      pipeline,
		Metric:       ExactMatchMetric{Fields: []string{"summary"}},
		RecordTraces: true,
	}
	examples := []Example{
		NewIOExample("ok", Prediction{"document": "fine text"}, Prediction{"summary": "short"}),
		NewIOExample("bad", Prediction{"document": "broken text"}, Prediction{"summary": "short"}),
	}
	candidate := pipeline.SeedCandidate()
	component := "summarise." + InstructionComponent
	require.Contains(t, candidate, component)

	result, err := evaluator.Evaluate(context.Background(), candidate, examples, true)
	require.NoError(t, err)
	require.Len(t, result.Items, 2)
	trace, ok := result.Items[1].Trace.(PipelineTrace)
	require.True(t, ok, "the item holds the run's trace")
	require.Equal(t, map[string]any{"document": "[doc]"}, trace.Inputs)
	require.NotContains(t, fmt.Sprint(trace), "broken text")

	var builder ReflectiveDatasetBuilder = TraceDatasetBuilder{}
	dataset := builder.BuildReflectiveDataset(candidate, result, []string{component, "missing.instruction"})
	require.Len(t, dataset["missing.instruction"], 2)
	require.Empty(t, dataset["missing.instruction"][0]["steps"])
	records := dataset[component]
	require.Len(t, records, 2)
	require.Equal(t, "ok", records[0]["example_id"])
	require.Equal(t, 1.0, records[0]["score"])
	require.Equal(t, "bad", records[1]["example_id"])
	require.Equal(t, 0.0, records[1]["score"])
	require.Equal(t, candidate[component], records[1]["current_value"])
	require.Equal(t, trace, records[1]["trace"])

	steps := records[1]["steps"].([]ReflectiveRecord)
	require.Len(t, steps, 1)
	failed := steps[0]
	require.Equal(t, "summarise", failed["step"])
	require.Equal(t, 1, failed["attempt"])
	require.Equal(t, trace.TraceID, failed["trace_id"])
	require.Equal(t, map[string]any{"document": "[doc]"}, failed["reads"])
	require.Contains(t, failed["error"], "invalid programme output")
	require.NotNil(t, failed["details"])
	require.NotContains(t, failed, "cached")
	succeeded := records[0]["steps"].([]ReflectiveRecord)[0]
	require.Equal(t, map[string]any{"summary": "short"}, succeeded["writes"])
	require.NotContains(t, succeeded, "error")

	untraced, err := evaluator.Evaluate(context.Background(), candidate, examples, false)
	require.NoError(t, err)
	require.Nil(t, untraced.Items[0].Trace, "without captureTraces")
	require.Nil(t, untraced.Items[1].Trace, "without captureTraces")
	evaluator.RecordTraces = false
	untraced, err = evaluator.Evaluate(context.Background(), candidate, examples, true)
	require.NoError(t, err)
	require.Nil(t, untraced.Items[0].Trace, "without RecordTraces")
	require.Nil(t, untraced.Items[1].Trace, "without RecordTraces")
}

func TestProgramEvaluatorTracesOnlyPipelines(t *testing.T) {
	predict := Predict{
		Signature: Signature{Name: "summarise", Inputs: []Field{{Name: "document"}}, Outputs: []Field{{Name: "summary"}}},
		LM:        tenTokenModel(`{"summary": "short"}`),
	}
	examples := []Example{NewIOExample("one", Prediction{"document": "d"}, Prediction{"summary": "short"})}
	evaluator := ProgramEvaluator{Program: predict, Metric: ExactMatchMetric{Fields: []string{"summary"}}, RecordTraces: true}
	result, err := evaluator.Evaluate(context.Background(), predict.SeedCandidate(), examples, true)
	require.NoError(t, err)
	require.Nil(t, result.Items[0].Trace)
	dataset := TraceDatasetBuilder{}.BuildReflectiveDataset(predict.SeedCandidate(), result, []string{InstructionComponent})
	require.Len(t, dataset[InstructionComponent], 1)
	require.NotContains(t, dataset[InstructionComponent][0], "steps")
	require.Equal(t, "one", dataset[InstructionComponent][0]["example_id"])

	// a programme that runs two pipelines yields both traces
	first, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{lengthStep()}}.Build()
	require.NoError(t, err)
	twice := programFunc(func(ctx context.Context, inputs Prediction) (Prediction, error) {
		if _, err := first.RunCandidate(ctx, nil, inputs); err != nil {
			return nil, err
		}
		return first.RunCandidate(ctx, nil, inputs)
	})
	result, err = ProgramEvaluator{Program: twice, Metric: ExactMatchMetric{Fields: []string{"amount"}}, RecordTraces: true}.
		Evaluate(context.Background(), nil, []Example{NewIOExample("one", Prediction{"document": "d"}, Prediction{"amount": 1})}, true)
	require.NoError(t, err)
	traces, ok := result.Items[0].Trace.([]PipelineTrace)
	require.True(t, ok)
	require.Len(t, traces, 2)
}

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
		{Step: "guess", Attempt: 2, Skipped: true},
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
	require.Equal(t, true, stepsOf("guess.instruction", 0)[0]["skipped"])
	require.Empty(t, stepsOf("fan.instruction", 0), "a composite with nested executions owns no component")
	require.Empty(t, stepsOf("summarise.instruction", 0), "a nested step owns only keys scoped by its path")
	require.Equal(t, ReflectiveDataset{"x": {}}, TraceDatasetBuilder{}.BuildReflectiveDataset(nil, EvaluationResult{}, []string{"x"}))
}
