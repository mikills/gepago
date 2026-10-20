package gepa_test

import (
	"context"
	"sync/atomic"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

var (
	orderIDKey  = gepa.Key[string]{Name: "order_id"}
	carrierKey  = gepa.Key[string]{Name: "carrier"}
	questionKey = gepa.Key[string]{Name: "question"}
	statusKey   = gepa.Key[string]{Name: "status"}
)

// orderPipeline looks the order up with a ToolStep and then answers the question with a Predict
// step that may call the same tool through lm. Its tool policy caches tool results.
func orderPipeline(t *testing.T, lm gepa.Generator, calls *atomic.Int32, recorder *gepa.RecordingObserver) gepa.Pipeline {
	t.Helper()
	registry, err := gepa.NewToolRegistry(lookupTool(calls))
	require.NoError(t, err)
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(orderIDKey.Ref(), questionKey.Ref()),
		Steps: []gepa.Step{
			gepa.ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), refs(carrierKey.Ref())),
			gepa.NewProgramStep(
				"answer",
				gepa.Predict{Signature: orderSignature, LM: lm, Tools: []string{"lookup_order"}},
				refs(questionKey.Ref()),
				refs(statusKey.Ref(), summaryKey.Ref()),
			),
		},
		Tools:      registry,
		ToolPolicy: gepa.ToolPolicy{Cache: true},
		Observers:  []gepa.PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)
	return pipeline
}

func TestToolStepAndPredictToolLoopAsTheTraceShows(t *testing.T) {
	lm := gepatest.NewFakeGenerator().OnStep(
		"answer",
		gepatest.ToolCall("call-1", "lookup_order", `{"order_id":"A-1"}`).WithUsage(tokens(5)),
		gepatest.Text(orderAnswer).WithUsage(tokens(6)),
	)
	calls := &atomic.Int32{}
	recorder := &gepa.RecordingObserver{}
	pipeline := orderPipeline(t, lm, calls, recorder)

	state, err := pipeline.Run(
		context.Background(),
		gepa.NewState(gepa.Prediction{"order_id": "A-1", "question": "Where is order A-1?"}),
	)

	require.NoError(t, err)
	carrier, err := carrierKey.Get(state)
	require.NoError(t, err)
	require.Equal(t, "DHL", carrier)
	status, err := statusKey.Get(state)
	require.NoError(t, err)
	require.Equal(t, "shipped", status)
	require.Equal(t, int32(1), calls.Load(), "the model's call repeats the lookup step's and is answered from the cache")
	requests := chatRequests(lm)
	require.Len(t, requests, 2)
	require.Equal(t, []string{"lookup_order"}, toolNamesOf(requests[0].Tools))
	last := requests[1].Messages[len(requests[1].Messages)-1]
	require.Equal(t, gepa.RoleTool, last.Role)
	require.Equal(t, "call-1", last.ToolCallID)
	require.JSONEq(t, `{"status":"shipped","carrier":"DHL"}`, last.Content)
	trace := onlyTrace(t, recorder)
	gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{
			{Step: "lookup", ToolCalls: []string{"lookup_order"}},
			{Step: "answer", Usage: tokens(11), ToolCalls: []string{"lookup_order"}},
		},
		Usage: tokens(11),
	})
	require.False(t, trace.Steps[0].ToolCalls[0].Cached)
	require.True(t, trace.Steps[1].ToolCalls[0].Cached)
	require.Equal(t, "answer", trace.Steps[1].ToolCalls[0].Step)
}

func TestFailingToolStepStopsTheRunBeforeTheModelAsTheTraceShows(t *testing.T) {
	lm := gepatest.NewFakeGenerator()
	calls := &atomic.Int32{}
	recorder := &gepa.RecordingObserver{}
	pipeline := orderPipeline(t, lm, calls, recorder)

	_, err := pipeline.Run(
		context.Background(),
		gepa.NewState(gepa.Prediction{"order_id": "missing", "question": "Where is my order?"}),
	)

	require.ErrorContains(t, err, "order not found")
	require.Equal(t, int32(1), calls.Load())
	require.Empty(t, lm.Calls(), "the answer step never runs")
	trace := onlyTrace(t, recorder)
	gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{{Step: "lookup", Err: "order not found", ToolCalls: []string{"lookup_order"}}},
		Err:   "order not found",
	})
	require.Equal(t, `tool "lookup_order": order not found`, trace.Steps[0].ToolCalls[0].Err)
}

// toolNamesOf returns the names of tools, in order.
func toolNamesOf(tools []gepa.Tool) []string {
	names := make([]string, len(tools))
	for i, tool := range tools {
		names[i] = tool.Name
	}
	return names
}
