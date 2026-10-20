package gepa

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

// tenantScope is a scope a caller gives a run.
type tenantScope struct {
	Tenant string `json:"tenant"`
}

func (s tenantScope) String() string { return "tenant " + s.Tenant }

// scopeMarker is a tenant no transcript, record, trace, report or artefact may hold.
const scopeMarker = "SCOPE-MARKER-7F3A"

// callRecorder is a tool that records every call it receives and answers with an empty object.
type callRecorder struct {
	mu    sync.Mutex
	calls []ToolCall
}

func (r *callRecorder) tool(name string) ToolProvider {
	return NewToolFunc(
		Tool{Name: name, Description: "record " + name, Parameters: json.RawMessage(`{"type":"object"}`)},
		func(_ context.Context, call ToolCall) (ToolResult, error) {
			r.mu.Lock()
			defer r.mu.Unlock()
			r.calls = append(r.calls, call)
			return ToolResult{Output: json.RawMessage(`{"status":"shipped"}`)}, nil
		},
	)
}

func (r *callRecorder) received() []ToolCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]ToolCall(nil), r.calls...)
}

// scopedPipeline runs a ToolStep and then a Predict whose model calls the same tool once.
func scopedPipeline(t *testing.T, recorder *callRecorder, lm *scriptedChat, observers ...PipelineObserver) Pipeline {
	t.Helper()
	registry, err := NewToolRegistry(recorder.tool("lookup_order"))
	require.NoError(t, err)
	explain := Predict{
		Signature: Signature{
			Name:    "explain",
			Inputs:  []Field{{Name: "status"}},
			Outputs: []Field{{Name: "summary", Type: "string"}},
		},
		LM:    lm,
		Tools: []string{"lookup_order"},
	}
	pipeline, err := PipelineSpec{
		Inputs: refs(orderIDKey.Ref()),
		Steps: []Step{
			ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), refs(statusKey.Ref())),
			NewProgramStep("explain", explain, refs(statusKey.Ref()), refs(summaryKey.Ref())),
		},
		Tools:     registry,
		Observers: observers,
	}.Build()
	require.NoError(t, err)
	return pipeline
}

func scopedChat() *scriptedChat {
	return newScriptedChat(
		toolCallReply("call-1", "lookup_order", `{"order_id":"B-2"}`, Usage{TotalTokens: 10}),
		textReply(`{"summary":"shipped"}`, Usage{TotalTokens: 20}),
	)
}

func TestToolScopeReachesEveryCallOfARun(t *testing.T) {
	recorder := &callRecorder{}
	lm := scopedChat()
	pipeline := scopedPipeline(t, recorder, lm)
	ctx := WithToolScope(context.Background(), tenantScope{Tenant: "acme"})

	_, err := pipeline.Run(ctx, NewState(Prediction{"order_id": "A-1"}))

	require.NoError(t, err)
	calls := recorder.received()
	require.Len(t, calls, 2, "the ToolStep's call and the Predict tool loop's call")
	for _, call := range calls {
		require.Equal(t, tenantScope{Tenant: "acme"}, call.Scope)
		scope, ok := call.ScopeAs[tenantScope]()
		require.True(t, ok)
		require.Equal(t, "acme", scope.Tenant)
	}
	require.JSONEq(t, `{"order_id":"A-1"}`, calls[0].Arguments)
	require.Equal(t, "call-1", calls[1].ID)
	require.JSONEq(t, `{"order_id":"B-2"}`, calls[1].Arguments)
}

func TestToolScopeIsNilWithoutOne(t *testing.T) {
	tests := []struct {
		name string
		ctx  context.Context
	}{
		{name: "a run without a scope", ctx: context.Background()},
		{name: "a nil scope set over another", ctx: WithToolScope(
			WithToolScope(context.Background(), tenantScope{Tenant: "acme"}),
			nil,
		)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := &callRecorder{}
			pipeline := scopedPipeline(t, recorder, scopedChat())

			_, err := pipeline.Run(tt.ctx, NewState(Prediction{"order_id": "A-1"}))

			require.NoError(t, err)
			calls := recorder.received()
			require.Len(t, calls, 2)
			for _, call := range calls {
				require.Nil(t, call.Scope)
				_, ok := call.ScopeAs[any]()
				require.False(t, ok, "a nil scope is no scope")
			}
		})
	}
}

func TestToolScopeReachesAStandalonePredictToolLoop(t *testing.T) {
	recorder := &callRecorder{}
	registry, err := NewToolRegistry(recorder.tool("lookup_order"))
	require.NoError(t, err)
	lm := newScriptedChat(
		toolCallReply("call-1", "lookup_order", `{"order_id":"A-1"}`, Usage{}),
		textReply(orderAnswer, Usage{}),
	)
	predict := Predict{Signature: orderSignature, LM: lm, Tools: []string{"lookup_order"}, ToolRegistry: registry}

	_, err = predict.Run(WithToolScope(context.Background(), "tenant-7"), Prediction{"question": "Where is A-1?"})

	require.NoError(t, err)
	calls := recorder.received()
	require.Len(t, calls, 1)
	scope, ok := calls[0].ScopeAs[string]()
	require.True(t, ok)
	require.Equal(t, "tenant-7", scope)
}

func TestToolScopeReplacesAScopeTheModelGave(t *testing.T) {
	for _, scope := range []any{nil, "caller"} {
		t.Run(fmt.Sprint(scope), func(t *testing.T) {
			recorder := &callRecorder{}
			registry, err := NewToolRegistry(recorder.tool("lookup_order"))
			require.NoError(t, err)
			reply := toolCallReply("call-1", "lookup_order", `{}`, Usage{})
			reply.resp.Message.ToolCalls[0].Scope = "model"
			lm := newScriptedChat(reply, textReply(orderAnswer, Usage{}))
			predict := Predict{
				Signature:    orderSignature,
				LM:           lm,
				Tools:        []string{"lookup_order"},
				ToolRegistry: registry,
			}

			_, err = predict.Run(WithToolScope(context.Background(), scope), Prediction{"question": "?"})

			require.NoError(t, err)
			require.Len(t, recorder.received(), 1)
			require.Equal(t, scope, recorder.received()[0].Scope, "a tool acts for the caller, never for the model")
		})
	}
}

func TestToolScopeIsNotPartOfTheCacheKey(t *testing.T) {
	recorder := &callRecorder{}
	session := newSession(t, ToolPolicy{Cache: true}, recorder.tool("lookup_order"))
	call := ToolCall{ID: "1", Name: "lookup_order", Arguments: `{"order_id":"A-1"}`}

	_, first, err := session.Call(WithToolScope(context.Background(), "tenant-a"), call)
	require.NoError(t, err)
	_, second, err := session.Call(WithToolScope(context.Background(), "tenant-b"), call)
	require.NoError(t, err)

	require.False(t, first.Cached)
	require.True(t, second.Cached, "the scope does not key the cache, so give each run one scope")
	require.Len(t, recorder.received(), 1)
	require.Equal(t, "tenant-a", recorder.received()[0].Scope)
	require.Nil(t, call.Scope, "the session sets the scope on its own copy of the call")
}

func TestToolScopeNeverLeavesTheRun(t *testing.T) {
	recorder := &callRecorder{}
	lm := scopedChat()
	traces := &RecordingObserver{}
	log := &eventLog{}
	pipeline := scopedPipeline(t, recorder, lm, traces, log)
	ctx := WithToolScope(context.Background(), tenantScope{Tenant: scopeMarker})

	_, run, err := pipeline.execute(ctx, pipeline.SeedCandidate(), Prediction{"order_id": "A-1"})

	require.NoError(t, err)
	require.Len(t, recorder.received(), 2)
	require.Equal(t, tenantScope{Tenant: scopeMarker}, recorder.received()[1].Scope, "the tool sees the scope")
	leaves := map[string]any{
		"chat requests": lm.requests,
		"tool records":  run.tools.Records(),
		"events":        log.finished,
		"trace":         traces.Traces(),
	}
	for name, value := range leaves {
		data, err := json.Marshal(value)
		require.NoError(t, err, name)
		require.NotContains(t, string(data), scopeMarker, "%s hold the scope", name)
	}
	for _, request := range lm.requests {
		for _, message := range request.Messages {
			for _, call := range message.ToolCalls {
				require.Nil(t, call.Scope, "the transcript's tool calls carry no scope")
			}
		}
	}
	report, err := HTMLRunReport(traces.Traces()[0], HTMLReportOptions{})
	require.NoError(t, err)
	require.NotContains(t, report, scopeMarker)
	require.Contains(t, report, "lookup_order")
}

func TestToolScopeNeverReachesAnEvaluationOrAnArtefact(t *testing.T) {
	recorder := &callRecorder{}
	pipeline := scopedPipeline(t, recorder, newScriptedChat(
		toolCallReply("call-1", "lookup_order", `{"order_id":"B-2"}`, Usage{}),
		textReply(`{"summary":"shipped"}`, Usage{}),
	))
	evaluator := ProgramEvaluator{
		Program:      pipeline,
		Metric:       ExactMatchMetric{Fields: []string{"summary"}},
		RecordTraces: true,
	}
	examples := []Example{
		NewIOExample("1", Prediction{"order_id": "A-1"}, Prediction{"summary": "shipped"}),
	}
	ctx := WithToolScope(context.Background(), tenantScope{Tenant: scopeMarker})

	eval, err := evaluator.Evaluate(ctx, pipeline.SeedCandidate(), examples, true)

	require.NoError(t, err)
	require.Len(t, recorder.received(), 2)
	_, ok := eval.Items[0].Trace.(PipelineTrace)
	require.True(t, ok, "the item holds the run's trace")
	dataset := TraceDatasetBuilder{}.BuildReflectiveDataset(
		pipeline.SeedCandidate(),
		eval,
		[]string{"explain.instruction"},
	)
	artefact := CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}.Artifact("orders")
	for name, value := range map[string]any{"evaluation": eval, "dataset": dataset, "artefact": artefact} {
		data, err := json.Marshal(value)
		require.NoError(t, err, name)
		require.NotContains(t, string(data), scopeMarker, "the %s holds the scope", name)
	}
}

func TestToolScopeThatIsNotJSONWorksInProcess(t *testing.T) {
	tenant := func() string { return "acme" }
	channel := make(chan int)
	for name, scope := range map[string]any{"a func": tenant, "a channel": channel} {
		t.Run(name, func(t *testing.T) {
			recorder := &callRecorder{}
			traces := &RecordingObserver{}
			pipeline := scopedPipeline(t, recorder, scopedChat(), traces)

			_, err := pipeline.Run(WithToolScope(context.Background(), scope), NewState(Prediction{"order_id": "A-1"}))

			require.NoError(t, err)
			require.Len(t, recorder.received(), 2)
			_, err = HTMLRunReport(traces.Traces()[0], HTMLReportOptions{})
			require.NoError(t, err)
			_, err = json.Marshal(recorder.received()[0])
			require.Error(t, err, "the scope cannot cross a process boundary, which marshals the call")
		})
	}

	recorder := &callRecorder{}
	pipeline := scopedPipeline(t, recorder, scopedChat())
	_, err := pipeline.Run(WithToolScope(context.Background(), tenant), NewState(Prediction{"order_id": "A-1"}))
	require.NoError(t, err)
	got, ok := recorder.received()[0].ScopeAs[func() string]()
	require.True(t, ok)
	require.Equal(t, "acme", got())
}

func TestToolCallScopeAs(t *testing.T) {
	tenant := &tenantScope{Tenant: "acme"}
	t.Run("no scope", func(t *testing.T) {
		call := ToolCall{}
		value, ok := call.ScopeAs[string]()
		require.False(t, ok)
		require.Empty(t, value)
		_, ok = call.ScopeAs[any]()
		require.False(t, ok)
		_, ok = call.ScopeAs[*tenantScope]()
		require.False(t, ok)
	})
	t.Run("a scope of another type", func(t *testing.T) {
		value, ok := ToolCall{Scope: "acme"}.ScopeAs[tenantScope]()
		require.False(t, ok)
		require.Equal(t, tenantScope{}, value, "a wrong type returns the zero value")
		pointer, ok := ToolCall{Scope: tenantScope{Tenant: "acme"}}.ScopeAs[*tenantScope]()
		require.False(t, ok)
		require.Nil(t, pointer)
	})
	t.Run("a scope of the type", func(t *testing.T) {
		value, ok := ToolCall{Scope: tenant}.ScopeAs[*tenantScope]()
		require.True(t, ok)
		require.Same(t, tenant, value, "the scope is not copied")
	})
	t.Run("an interface the scope implements", func(t *testing.T) {
		value, ok := ToolCall{Scope: tenantScope{Tenant: "acme"}}.ScopeAs[fmt.Stringer]()
		require.True(t, ok)
		require.Equal(t, "tenant acme", value.String())
	})
	t.Run("a call decoded from JSON", func(t *testing.T) {
		var call ToolCall
		require.NoError(
			t,
			json.Unmarshal([]byte(`{"id":"1","name":"t","arguments":"{}","scope":{"tenant":"acme"}}`), &call),
		)
		_, ok := call.ScopeAs[tenantScope]()
		require.False(t, ok, "a remote host receives the scope's JSON, not the caller's type")
		value, ok := call.ScopeAs[map[string]any]()
		require.True(t, ok)
		require.Equal(t, map[string]any{"tenant": "acme"}, value)
	})
}

func TestToolScopeJSONForm(t *testing.T) {
	data, err := json.Marshal(ToolCall{ID: "1", Name: "t", Arguments: "{}"})
	require.NoError(t, err)
	require.JSONEq(t, `{"id":"1","name":"t","arguments":"{}"}`, string(data), "a call without a scope is unchanged")
	require.False(t, strings.Contains(string(data), "scope"))

	data, err = json.Marshal(ToolCall{ID: "1", Name: "t", Arguments: "{}", Scope: tenantScope{Tenant: "acme"}})
	require.NoError(t, err)
	require.JSONEq(t, `{"id":"1","name":"t","arguments":"{}","scope":{"tenant":"acme"}}`, string(data))
}
