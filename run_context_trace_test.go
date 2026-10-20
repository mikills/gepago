package gepa_test

import (
	"context"
	"fmt"
	"sync"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// researchPipeline runs a pipeline with one model step "ask" as its step "research", recording
// its runs with recorder.
func researchPipeline(t *testing.T, lm gepa.Generator, recorder *gepa.RecordingObserver) gepa.Pipeline {
	t.Helper()
	answer := gepa.Key[string]{Name: "answer"}
	inner, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{modelStep("ask", lm, docKey.Ref(), answer.Ref())},
	}.Build()
	require.NoError(t, err)
	outer, err := gepa.PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Steps:     []gepa.Step{gepa.NewProgramStep("research", inner, refs(docKey.Ref()), refs(answer.Ref()))},
		Observers: []gepa.PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)
	return outer
}

func TestNestedRunsCarryTheRunContextToTheModelAndTheTrace(t *testing.T) {
	lm := gepatest.NewFakeGenerator().OnStep("ask", gepatest.Text(`{"answer": "42"}`).WithUsage(tokens(6)))
	recorder := &gepa.RecordingObserver{}
	pipeline := researchPipeline(t, lm, recorder)
	ctx := gepa.WithRunContext(context.Background(), gepa.RunContext{
		TraceID:    "trace-1",
		Attributes: map[string]string{"job": "nightly"},
	})

	_, err := pipeline.Run(ctx, gepa.NewState(gepa.Prediction{"document": "question"}))

	require.NoError(t, err)
	trace := onlyTrace(t, recorder)
	gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{{Step: "research", Usage: tokens(6)}, {Step: "research.ask", Usage: tokens(6)}},
		Usage: tokens(6),
	})
	require.Equal(t, "trace-1", trace.TraceID)
	require.Empty(t, trace.RunContext.ParentRunID)
	require.Len(t, trace.Steps[0].Runs, 1)
	nested := trace.Steps[0].Runs[0]
	require.Equal(t, trace.Steps[0].SpanID, nested.ParentID)
	want := gepa.RunContext{
		RunID:       nested.SpanID,
		ParentRunID: trace.SpanID,
		TraceID:     "trace-1",
		Attributes:  map[string]string{"job": "nightly"},
	}
	require.Equal(t, want, nested.RunContext)
	calls := lm.Calls()
	require.Len(t, calls, 1)
	require.Equal(t, "research.ask", calls[0].Step)
	require.Equal(t, want, calls[0].RunContext, "the model call runs in the nested run")
}

func TestConcurrentRunsGiveEachModelCallItsOwnRunContext(t *testing.T) {
	const runs = 8
	lm := gepatest.NewFakeGenerator().OnStep("ask", gepatest.Text(`{"answer": "42"}`))
	recorder := &gepa.RecordingObserver{}
	pipeline := researchPipeline(t, lm, recorder)

	errs := make([]error, runs)
	var wg sync.WaitGroup
	for i := range runs {
		wg.Go(func() {
			ctx := gepa.WithRunContext(context.Background(), gepa.RunContext{
				Attributes: map[string]string{"request": fmt.Sprint(i)},
			})
			_, errs[i] = pipeline.Run(ctx, gepa.NewState(gepa.Prediction{"document": "question"}))
		})
	}
	wg.Wait()
	for _, err := range errs {
		require.NoError(t, err)
	}

	traces := recorder.Traces()
	require.Len(t, traces, runs)
	nestedByRequest := map[string]gepa.RunContext{}
	for _, trace := range traces {
		gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{
			Steps: []gepatest.ExpectedStep{{Step: "research"}, {Step: "research.ask"}},
		})
		nested := trace.Steps[0].Runs[0].RunContext
		require.Equal(t, trace.SpanID, nested.ParentRunID)
		require.Equal(t, trace.TraceID, nested.TraceID)
		nestedByRequest[nested.Attributes["request"]] = nested
	}
	require.Len(t, nestedByRequest, runs)
	calls := lm.Calls()
	require.Len(t, calls, runs)
	for _, call := range calls {
		require.Equal(t, nestedByRequest[call.RunContext.Attributes["request"]], call.RunContext)
	}
}
