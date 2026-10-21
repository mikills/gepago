package gepa_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// tracedPipeline runs a pipeline with a Parallel, a Fallback, a nested pipeline and a final
// step that fails, and returns its recorded trace.
func tracedPipeline(t *testing.T) gepa.PipelineTrace {
	t.Helper()
	type score struct {
		Value float64 `json:"value"`
		Label string  `json:"label"`
	}
	scoreKey := gepa.Key[score]{Name: "score"}
	ratioKey := gepa.Key[float64]{Name: "ratio"}
	inner, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{
			modelStep("classify", tenTokenModel(`{"decision": "approve"}`), docKey.Ref(), decisionKey.Ref()),
		},
	}.Build()
	require.NoError(t, err)
	measure := gepa.NewStep(
		"measure",
		nil,
		refs(scoreKey.Ref(), ratioKey.Ref()),
		func(_ context.Context, s gepa.State) error {
			scoreKey.Set(s, score{Value: 0.5, Label: "half"})
			ratioKey.Set(s, math.NaN())
			return nil
		},
	)
	primary := gepa.NewStep("primary", nil, refs(amountKey.Ref()), func(context.Context, gepa.State) error {
		return gepa.WithStepDetails(
			gepa.TransientError(errors.New("primary down")),
			map[string]any{"attempted": []string{"a"}},
		)
	})
	check := gepa.NewStep(
		"check",
		refs(amountKey.Ref()),
		refs(notesKey.Ref()),
		func(context.Context, gepa.State) error {
			return gepa.PermanentError(errors.New("amount too small"))
		},
	)
	recorder := &gepa.RecordingObserver{}
	// v0.9.0: check reads amount, which the fallback's steps may not write, so it has a default.
	pipeline, err := withFlows(
		gepa.PipelineSpec{
			Inputs:    refs(docKey.Ref()),
			Defaults:  []gepa.Default{gepa.DefaultValue(amountKey, 0)},
			Observers: []gepa.PipelineObserver{recorder},
		},
		gepa.Parallel(
			"fan",
			modelStep("summarise", tenTokenModel(`{"summary": "short"}`), docKey.Ref(), summaryKey.Ref()),
			measure,
		),
		gepa.Fallback("fb", primary, writeAmount("secondary", 3)),
		gepa.NewProgramStep("inner", inner, refs(docKey.Ref()), refs(decisionKey.Ref())),
		check,
	).Build()
	require.NoError(t, err)
	_, err = pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"document": "cash 10"}))
	require.EqualError(t, err, `pipeline step "check": amount too small`)
	return onlyTrace(t, recorder)
}

func TestRecordingObserverRecordsNestedSteps(t *testing.T) {
	trace := tracedPipeline(t)
	require.Empty(t, trace.ParentID)
	require.False(t, trace.EndedAt.IsZero())

	// the Fallback's steps are TestRecordingObserverRecordsFlowSteps's.
	require.Equal(t, []string{"fan", "primary", "secondary", "fb", "inner", "check"}, stepNames(trace.Steps))
	fan := trace.Steps[0]
	require.Equal(t, trace.SpanID, fan.ParentID)
	require.ElementsMatch(t, []string{"summarise", "measure"}, stepNames(fan.Steps))
	require.Equal(t, tenTokens, fan.Usage)
	require.Equal(t, map[string]any{
		"summary": "short",
		"score":   map[string]any{"value": 0.5, "label": "half"},
		"ratio":   "NaN",
	}, fan.Writes)

	nested := trace.Steps[4]
	require.Empty(t, nested.Steps)
	require.Len(t, nested.Runs, 1)
	require.Equal(t, nested.SpanID, nested.Runs[0].ParentID)
	require.Equal(t, trace.TraceID, nested.Runs[0].TraceID)
	require.Equal(t, []string{"classify"}, stepNames(nested.Runs[0].Steps))
	require.Equal(t, tenTokens, nested.Runs[0].Usage)
}

func stepNames(steps []gepa.StepTrace) []string {
	names := make([]string, 0, len(steps))
	for _, step := range steps {
		names = append(names, step.Step)
	}
	return names
}

// TestTraceDatasetBuilderOverProgramEvaluator evaluates a pipeline whose model step fails with
// an invalid output, and checks that the reflection record names the failed step and carries
// its reads as the pipeline's Redactor redacts them.
func TestTraceDatasetBuilderOverProgramEvaluator(t *testing.T) {
	model := gepatest.NewFakeGenerator().
		OnPrompt("broken", gepatest.Text(`{"other": "x"}`).WithUsage(tenTokens)).
		Default(gepatest.Text(`{"summary": "short"}`).WithUsage(tenTokens))
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{lengthStep(), modelStep("summarise", model, docKey.Ref(), summaryKey.Ref())},
		Redactor: func(key string, value any) any {
			if key == "document" {
				return "[doc]"
			}
			return value
		},
	}.Build()
	require.NoError(t, err)
	evaluator := gepa.ProgramEvaluator{
		Program:      pipeline,
		Metric:       gepa.ExactMatchMetric{Fields: []string{"summary"}},
		RecordTraces: true,
	}
	examples := []gepa.Example{
		gepa.NewIOExample("ok", gepa.Prediction{"document": "fine text"}, gepa.Prediction{"summary": "short"}),
		gepa.NewIOExample("bad", gepa.Prediction{"document": "broken text"}, gepa.Prediction{"summary": "short"}),
	}
	candidate := pipeline.SeedCandidate()
	component := "summarise." + gepa.InstructionComponent
	require.Contains(t, candidate, component)

	result, err := evaluator.Evaluate(context.Background(), candidate, examples, true)
	require.NoError(t, err)
	require.Len(t, result.Items, 2)
	trace, ok := result.Items[1].Trace.(gepa.PipelineTrace)
	require.True(t, ok, "the item holds the run's trace")
	require.Equal(t, map[string]any{"document": "[doc]"}, trace.Inputs)
	require.NotContains(t, fmt.Sprint(trace), "broken text")

	var builder gepa.ReflectiveDatasetBuilder = gepa.TraceDatasetBuilder{}
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

	steps := records[1]["steps"].([]gepa.ReflectiveRecord)
	require.Len(t, steps, 1)
	failed := steps[0]
	require.Equal(t, "summarise", failed["step"])
	require.Equal(t, 1, failed["attempt"])
	require.Equal(t, trace.TraceID, failed["trace_id"])
	require.Equal(t, map[string]any{"document": "[doc]"}, failed["reads"])
	require.Contains(t, failed["error"], "invalid programme output")
	require.NotNil(t, failed["details"])
	require.NotContains(t, failed, "cached")
	succeeded := records[0]["steps"].([]gepa.ReflectiveRecord)[0]
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
	predict := gepa.Predict{
		Signature: gepa.Signature{
			Name:    "summarise",
			Inputs:  []gepa.Field{{Name: "document"}},
			Outputs: []gepa.Field{{Name: "summary"}},
		},
		LM: tenTokenModel(`{"summary": "short"}`),
	}
	examples := []gepa.Example{
		gepa.NewIOExample("one", gepa.Prediction{"document": "d"}, gepa.Prediction{"summary": "short"}),
	}
	evaluator := gepa.ProgramEvaluator{
		Program:      predict,
		Metric:       gepa.ExactMatchMetric{Fields: []string{"summary"}},
		RecordTraces: true,
	}
	result, err := evaluator.Evaluate(context.Background(), predict.SeedCandidate(), examples, true)
	require.NoError(t, err)
	require.Nil(t, result.Items[0].Trace)
	dataset := gepa.TraceDatasetBuilder{}.BuildReflectiveDataset(
		predict.SeedCandidate(),
		result,
		[]string{gepa.InstructionComponent},
	)
	require.Len(t, dataset[gepa.InstructionComponent], 1)
	require.NotContains(t, dataset[gepa.InstructionComponent][0], "steps")
	require.Equal(t, "one", dataset[gepa.InstructionComponent][0]["example_id"])

	// a programme that runs two pipelines yields both traces
	first, err := gepa.PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []gepa.Step{lengthStep()}}.Build()
	require.NoError(t, err)
	twice := programFunc(func(ctx context.Context, inputs gepa.Prediction) (gepa.Prediction, error) {
		if _, err := first.RunCandidate(ctx, nil, inputs); err != nil {
			return nil, err
		}
		return first.RunCandidate(ctx, nil, inputs)
	})
	result, err = gepa.ProgramEvaluator{
		Program:      twice,
		Metric:       gepa.ExactMatchMetric{Fields: []string{"amount"}},
		RecordTraces: true,
	}.
		Evaluate(
			context.Background(),
			nil,
			[]gepa.Example{gepa.NewIOExample("one", gepa.Prediction{"document": "d"}, gepa.Prediction{"amount": 1})},
			true,
		)
	require.NoError(t, err)
	traces, ok := result.Items[0].Trace.([]gepa.PipelineTrace)
	require.True(t, ok)
	require.Len(t, traces, 2)
}

func TestHTMLRunReportShowsEveryStepAndTheFailure(t *testing.T) {
	trace := tracedPipeline(t)
	report, err := gepa.HTMLRunReport(trace, gepa.HTMLReportOptions{})
	require.NoError(t, err)

	require.Contains(t, report, "<title>Pipeline Run Report</title>")
	for _, step := range []string{"fan", "summarise", "measure", "fb", "primary", "secondary", "inner", "classify", "check"} {
		require.Contains(t, report, "<strong>"+step+"</strong>")
	}
	require.Contains(t, report, "amount too small")
	require.Contains(t, report, "pipeline step &#34;check&#34;: amount too small")
	require.Contains(t, report, "primary down")
	require.Contains(t, report, ">failed</strong>")
	require.Contains(t, report, "Nested run "+trace.Steps[4].Runs[0].RunContext.RunID+" ·")
	// a nested step is rendered inside the step that ran it
	fan := strings.Index(report, "<strong>fan</strong>")
	summarise := strings.Index(report, "<strong>summarise</strong>")
	fb := strings.Index(report, "<strong>fb</strong>")
	require.Less(t, fan, summarise)
	require.Less(t, summarise, fb)
	require.Equal(t, 3, strings.Count(report, `<ol class="steps">`))
}
