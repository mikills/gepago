package gepa

import (
	"context"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestProvenanceValidate(t *testing.T) {
	tests := []struct {
		name       string
		provenance Provenance
		wantErr    string
	}{
		{name: "a resource", provenance: Provenance{Kind: "document", ID: "doc-1"}},
		{name: "a source", provenance: Provenance{Source: "https://example.com/a"}},
		{name: "a source and its detail", provenance: Provenance{Source: "file.go", Detail: "lines 3-9"}},
		{name: "a resource and a source", provenance: Provenance{Kind: "document", ID: "doc-1", Source: "file.go"}},
		{
			name:       "both pairs",
			provenance: Provenance{Kind: "document", ID: "doc-1", Source: "file.go", Detail: "lines 3-9"},
		},
		{name: "nothing", wantErr: "provenance is empty: set Kind and ID, or Source"},
		{
			name:       "only white space",
			provenance: Provenance{Kind: " ", ID: "\t", Source: "  "},
			wantErr:    "provenance is empty: set Kind and ID, or Source",
		},
		{
			name:       "a kind without an id",
			provenance: Provenance{Kind: "document"},
			wantErr:    "provenance has Kind but no ID",
		},
		{
			name:       "a kind with a blank id beside a source",
			provenance: Provenance{Kind: "document", ID: " ", Source: "file.go"},
			wantErr:    "provenance has Kind but no ID",
		},
		{name: "an id without a kind", provenance: Provenance{ID: "doc-1"}, wantErr: "provenance has ID but no Kind"},
		{
			name:       "an id without a kind beside a source",
			provenance: Provenance{ID: "doc-1", Source: "file.go"},
			wantErr:    "provenance has ID but no Kind",
		},
		{
			name:       "a detail alone",
			provenance: Provenance{Detail: "lines 3-9"},
			wantErr:    "provenance has Detail but no Source",
		},
		{
			name:       "a detail without a source beside a resource",
			provenance: Provenance{Kind: "document", ID: "doc-1", Detail: "lines 3-9"},
			wantErr:    "provenance has Detail but no Source",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.provenance.Validate()
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
			for _, value := range []string{"document", "doc-1", "file.go", "lines 3-9"} {
				require.NotContains(t, err.Error(), value, "the error names fields, never values")
			}
		})
	}
}

// citingTool returns a tool that answers with provenance and spends usage both ways.
func citingTool(calls *atomic.Int32, provenance ...Provenance) ToolProvider {
	return NewToolFunc(
		Tool{Name: "cite", Description: "cite sources", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, ToolCall) (ToolResult, error) {
			calls.Add(1)
			return ToolResult{
				Output:          json.RawMessage(`{"status":"cited"}`),
				MemoryWrites:    map[string]any{"cited": true},
				Provenance:      provenance,
				Usage:           Usage{TotalTokens: 7},
				UnbudgetedUsage: Usage{TotalTokens: 5},
			}, nil
		},
	)
}

func TestProvenanceThatIsInvalidFailsTheCall(t *testing.T) {
	calls := &atomic.Int32{}
	memory := NewToolMemory()
	session, err := NewToolSession(
		[]ToolProvider{citingTool(calls, Provenance{Source: "file.go"}, Provenance{Kind: "document"})},
		ToolPolicy{Cache: true},
		memory,
	)
	require.NoError(t, err)
	ctx, collector := WithUsageCollector(context.Background())
	call := ToolCall{ID: "1", Name: "cite", Arguments: `{}`}

	result, record, err := session.Call(ctx, call)

	require.EqualError(t, err, `tool "cite": invalid provenance 1: provenance has Kind but no ID`)
	require.Equal(t, ToolResult{}, result)
	require.Equal(t, err.Error(), record.Err)
	require.Equal(t, []Provenance{{Source: "file.go"}, {Kind: "document"}}, record.Provenance,
		"the record keeps what the tool returned")
	require.Equal(t, Usage{TotalTokens: 12}, collector.Usage(), "its usage is recorded once")
	require.Equal(t, Usage{TotalTokens: 7}, record.Usage)
	require.Equal(t, Usage{TotalTokens: 5}, record.UnbudgetedUsage)
	_, ok := memory.Get("cited")
	require.False(t, ok, "a failed call's memory writes are dropped")

	_, record, err = session.Call(ctx, call)
	require.Error(t, err)
	require.False(t, record.Cached, "a failed call is not cached")
	require.EqualValues(t, 2, calls.Load())
}

func TestProvenanceThatIsInvalidFailsAToolStep(t *testing.T) {
	calls := &atomic.Int32{}
	registry, err := NewToolRegistry(citingTool(calls, Provenance{Detail: "lines 3-9"}))
	require.NoError(t, err)
	pipeline, err := toolPipeline(registry, ToolPolicy{}, ToolStep("cite", "cite", nil, refs(statusKey.Ref())))
	require.NoError(t, err)

	got, err := pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

	require.EqualError(
		t,
		err,
		`pipeline step "cite": tool "cite": invalid provenance 0: provenance has Detail but no Source`,
	)
	require.Empty(t, got.Prediction(), "the step writes nothing")
}

// bothProvenanceShapes holds a resource, a citation and an entry with both.
var bothProvenanceShapes = []Provenance{
	{Kind: "document", ID: "doc-1"},
	{Source: "https://example.com/a", Detail: "paragraph 2"},
	{Kind: "document", ID: "doc-2", Source: "file.go", Detail: "lines 3-9"},
}

func TestProvenanceShapesRoundTrip(t *testing.T) {
	result := ToolResult{Output: json.RawMessage(`{"status":"cited"}`), Provenance: bothProvenanceShapes}
	data, err := json.Marshal(result)
	require.NoError(t, err)
	require.Contains(t, string(data), `"provenance":[{"kind":"document","id":"doc-1"},`+
		`{"source":"https://example.com/a","detail":"paragraph 2"},`+
		`{"kind":"document","id":"doc-2","source":"file.go","detail":"lines 3-9"}]`)
	var decoded ToolResult
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, result, decoded)

	calls := &atomic.Int32{}
	registry, err := NewToolRegistry(citingTool(calls, bothProvenanceShapes...))
	require.NoError(t, err)
	traces := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(orderIDKey.Ref()),
		Steps:     []Step{ToolStep("cite", "cite", nil, refs(statusKey.Ref()))},
		Tools:     registry,
		Observers: []PipelineObserver{traces},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

	require.NoError(t, err)
	trace := traces.Traces()[0]
	require.Equal(t, bothProvenanceShapes, trace.Steps[0].ToolCalls[0].Provenance)
	requireTraceRoundTrip(t, trace)
}

func TestProvenanceOfAV080TraceLoadsUnchanged(t *testing.T) {
	const v080 = `{"trace_id":"t","span_id":"r","run_context":{"run_id":"r","trace_id":"t"},` +
		`"started_at":"2026-01-02T03:04:05Z","ended_at":"2026-01-02T03:04:06Z","duration":1000000000,` +
		`"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3},"steps":[{"span_id":"s","parent_id":"r",` +
		`"step":"cite","attempt":1,"started_at":"2026-01-02T03:04:05Z","ended_at":"2026-01-02T03:04:06Z",` +
		`"duration":1000000000,"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3},` +
		`"tool_calls":[{"step":"cite","call_id":"c","name":"cite","arguments":"{}","output":{"status":"cited"},` +
		`"text":"{\"status\":\"cited\"}","provenance":[{"source":"https://example.com/a","detail":"paragraph 2"},` +
		`{"source":"file.go"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0},"duration":5}]}]}`

	var trace PipelineTrace
	require.NoError(t, json.Unmarshal([]byte(v080), &trace))

	require.Equal(t, []Provenance{
		{Source: "https://example.com/a", Detail: "paragraph 2"},
		{Source: "file.go"},
	}, trace.Steps[0].ToolCalls[0].Provenance)
	require.Equal(t, Usage{}, trace.Unbudgeted)
	require.Equal(t, Usage{}, trace.Steps[0].ToolCalls[0].UnbudgetedUsage)
	data, err := json.Marshal(trace)
	require.NoError(t, err)
	require.JSONEq(t, v080, string(data), "the trace writes back as it was")
}

func TestProvenanceIsRedactedLikeTheRestOfAToolCall(t *testing.T) {
	const secret = "ACME-7731"
	cite := NewToolFunc(
		Tool{Name: "cite", Description: "cite", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, ToolCall) (ToolResult, error) {
			return ToolResult{Output: json.RawMessage(`{"status":"cited"}`), Provenance: []Provenance{
				{Kind: "order " + secret, ID: secret, Source: "db://orders/" + secret, Detail: "row " + secret},
				{Kind: "audit", ID: "masked-by-the-redactor", Source: "log"},
			}}, nil
		},
	)
	registry, err := NewToolRegistry(cite)
	require.NoError(t, err)
	traces := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Inputs: refs(orderIDKey.Ref()),
		Steps:  []Step{ToolStep("cite", "cite", refs(orderIDKey.Ref()), refs(statusKey.Ref()))},
		Tools:  registry,
		Redactor: func(key string, value any) any {
			switch {
			case key == orderIDKey.Name:
				return RedactedText
			case strings.HasPrefix(key, "tool:"):
				if provenance, ok := value.([]Provenance); ok {
					for i := range provenance {
						if provenance[i].Kind == "audit" {
							provenance[i].ID = "[id]"
						}
					}
					return provenance
				}
			}
			return value
		},
		Observers: []PipelineObserver{traces},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": secret}))

	require.NoError(t, err)
	trace := traces.Traces()[0]
	require.Equal(t, []Provenance{
		{
			Kind:   "order " + RedactedText,
			ID:     RedactedText,
			Source: "db://orders/" + RedactedText,
			Detail: "row " + RedactedText,
		},
		{Kind: "audit", ID: "[id]", Source: "log"},
	}, trace.Steps[0].ToolCalls[0].Provenance, "Kind and ID are scrubbed as Source and Detail are, and a Redactor may rewrite them")
	data := requireTraceRoundTrip(t, trace)
	require.NotContains(t, data, secret)
	require.NotContains(t, data, "masked-by-the-redactor")
}
