package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
)

// billedTool returns a tool named name that answers with an empty object, spends usage against
// the run's budget and unbudgeted outside it, and fails with err. It counts its invocations.
func billedTool(name string, calls *atomic.Int32, usage, unbudgeted Usage, err error) ToolProvider {
	return NewToolFunc(
		Tool{Name: name, Description: "bill " + name},
		func(context.Context, ToolCall) (ToolResult, error) {
			calls.Add(1)
			return ToolResult{Output: json.RawMessage(`{}`), Usage: usage, UnbudgetedUsage: unbudgeted}, err
		},
	)
}

func TestUnbudgetedUsageDoesNotTripTheBudget(t *testing.T) {
	calls := &atomic.Int32{}
	registry, err := NewToolRegistry(billedTool("search", calls, Usage{}, Usage{TotalTokens: 50}, nil))
	require.NoError(t, err)
	var remaining BudgetRemaining
	log := &eventLog{}
	traces := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Steps: []Step{
			ToolStep("search", "search", nil, nil),
			NewStep("after", nil, nil, func(ctx context.Context, _ State) error {
				remaining = BudgetFrom(ctx).Remaining()
				return nil
			}),
		},
		Tools:     registry,
		Budget:    RunBudget{MaxTokens: 40, PerStepTokens: map[string]int{"search": 1}},
		Observers: []PipelineObserver{log, traces},
	}.Build()
	require.NoError(t, err)

	_, usage, err := CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}.RunWithUsage(
		context.Background(),
		Prediction{},
	)

	require.NoError(t, err, "50 unbudgeted tokens trip neither the 40-token run budget nor the step's")
	require.Equal(t, Usage{TotalTokens: 50}, usage, "RunWithUsage reports budgeted and unbudgeted usage")
	require.Equal(t, 40, remaining.Tokens, "the budget counted none of it")
	step := log.finishedStep(t, "search", 1)
	require.Equal(t, Usage{TotalTokens: 50}, step.Span.Unbudgeted)
	require.Equal(t, Usage{}, step.Span.Usage)
	require.Equal(t, Usage{TotalTokens: 50}, step.Unbudgeted)
	require.Equal(t, Usage{}, step.Usage)
	require.Len(t, step.ToolCalls, 1)
	require.Equal(t, Usage{TotalTokens: 50}, step.ToolCalls[0].UnbudgetedUsage)
	require.Equal(t, Usage{}, log.finishedStep(t, "after", 1).Unbudgeted)
	run := log.runsFinished[0]
	require.Equal(t, Usage{TotalTokens: 50}, run.Unbudgeted)
	require.Equal(t, Usage{TotalTokens: 50}, run.Span.Unbudgeted)
	require.Equal(t, Usage{}, run.Usage)

	trace := traces.Traces()[0]
	require.Equal(t, Usage{TotalTokens: 50}, trace.Unbudgeted)
	require.Equal(t, Usage{TotalTokens: 50}, trace.Steps[0].Unbudgeted)
	require.Equal(t, Usage{TotalTokens: 50}, trace.Steps[0].ToolCalls[0].UnbudgetedUsage)
	data := requireTraceRoundTrip(t, trace)
	require.Contains(t, data, `"unbudgeted":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":50}`)
	report, err := HTMLRunReport(trace, HTMLReportOptions{})
	require.NoError(t, err)
	require.Contains(t, report, "0 tokens (0 in, 0 out) · 50 unbudgeted tokens")
	require.Contains(t, report, `<span class="muted">Unbudgeted tokens</span><strong>50</strong>`)
}

func TestUnbudgetedUsageBesideBudgetedUsage(t *testing.T) {
	tests := []struct {
		name     string
		budgeted int
		wantErr  string
	}{
		{name: "budgeted usage one below the budget", budgeted: 39},
		{name: "budgeted usage at the budget", budgeted: 40},
		{
			name:     "budgeted usage one above the budget",
			budgeted: 41,
			wantErr:  `pipeline step "search": tool "search": run budget MaxTokens 40 exceeded: 41 tokens spent`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &atomic.Int32{}
			tool := billedTool("search", calls, Usage{TotalTokens: tt.budgeted}, Usage{TotalTokens: 50}, nil)
			registry, err := NewToolRegistry(tool)
			require.NoError(t, err)
			log := &eventLog{}
			pipeline, err := PipelineSpec{
				Steps:     []Step{ToolStep("search", "search", nil, nil)},
				Tools:     registry,
				Budget:    RunBudget{MaxTokens: 40},
				Observers: []PipelineObserver{log},
			}.Build()
			require.NoError(t, err)
			ctx, collector := WithUsageCollector(context.Background())

			_, err = pipeline.Run(ctx, NewState(nil))

			require.Equal(t, Usage{TotalTokens: tt.budgeted + 50}, collector.Usage(), "each usage is recorded once")
			step := log.finishedStep(t, "search", 1)
			require.Equal(t, Usage{TotalTokens: tt.budgeted}, step.Usage)
			require.Equal(t, Usage{TotalTokens: 50}, step.Unbudgeted)
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
			require.True(t, IsBudget(err), "the budgeted usage still trips the budget")
		})
	}
}

func TestUnbudgetedUsageOfAFailedCallIsRecordedOnce(t *testing.T) {
	failing := NewToolFunc(
		Tool{Name: "failing", Description: "fails"},
		func(context.Context, ToolCall) (ToolResult, error) {
			return ToolResult{Usage: Usage{TotalTokens: 3}, UnbudgetedUsage: Usage{TotalTokens: 50}}, errors.New("boom")
		},
	)
	notJSON := NewToolFunc(
		Tool{Name: "not_json", Description: "bad output"},
		func(context.Context, ToolCall) (ToolResult, error) {
			return ToolResult{Output: json.RawMessage(`{`), UnbudgetedUsage: Usage{TotalTokens: 50}}, nil
		},
	)
	badProvenance := NewToolFunc(
		Tool{Name: "uncited", Description: "bad provenance"},
		func(context.Context, ToolCall) (ToolResult, error) {
			return ToolResult{Provenance: []Provenance{{Kind: "doc"}}, UnbudgetedUsage: Usage{TotalTokens: 50}}, nil
		},
	)
	tests := []struct {
		tool     ToolProvider
		budgeted Usage
		wantErr  string
	}{
		{tool: failing, budgeted: Usage{TotalTokens: 3}, wantErr: `tool "failing": boom`},
		{tool: notJSON, wantErr: `tool "not_json": tool output is not valid JSON`},
		{tool: badProvenance, wantErr: `tool "uncited": invalid provenance 0: provenance has Kind but no ID`},
	}
	for _, tt := range tests {
		name := tt.tool.Schema().Name
		t.Run(name, func(t *testing.T) {
			registry, err := NewToolRegistry(tt.tool)
			require.NoError(t, err)
			log := &eventLog{}
			pipeline, err := PipelineSpec{
				Steps:     []Step{ToolStep("call", name, nil, nil)},
				Tools:     registry,
				Observers: []PipelineObserver{log},
			}.Build()
			require.NoError(t, err)
			ctx, collector := WithUsageCollector(context.Background())

			_, err = pipeline.Run(ctx, NewState(nil))

			require.EqualError(t, err, `pipeline step "call": `+tt.wantErr)
			require.Equal(t, tt.budgeted.Add(Usage{TotalTokens: 50}), collector.Usage())
			step := log.finishedStep(t, "call", 1)
			require.Equal(t, tt.budgeted, step.Usage)
			require.Equal(t, Usage{TotalTokens: 50}, step.Unbudgeted)
			require.Len(t, step.ToolCalls, 1)
			require.Equal(t, Usage{TotalTokens: 50}, step.ToolCalls[0].UnbudgetedUsage)
			require.Equal(t, tt.wantErr, step.ToolCalls[0].Err)
			require.Equal(t, Usage{TotalTokens: 50}, log.runsFinished[0].Unbudgeted)
		})
	}
}

func TestUnbudgetedUsageOfAResultThatOutlivesItsContextIsRecordedOnce(t *testing.T) {
	ctx, cancel := context.WithCancelCause(context.Background())
	stop := errors.New("run stopped")
	late := NewToolFunc(Tool{Name: "late", Description: "late"}, func(context.Context, ToolCall) (ToolResult, error) {
		cancel(stop)
		return ToolResult{UnbudgetedUsage: Usage{TotalTokens: 50}}, nil
	})
	session := newSession(t, ToolPolicy{}, late)
	ctx, collector := WithUsageCollector(ctx)

	_, record, err := session.Call(ctx, ToolCall{Name: "late", Arguments: `{}`})

	require.ErrorIs(t, err, stop)
	require.Equal(t, Usage{TotalTokens: 50}, record.UnbudgetedUsage)
	require.Equal(t, Usage{TotalTokens: 50}, collector.Usage())
}

func TestUnbudgetedUsageOfACachedCallIsZero(t *testing.T) {
	calls := &atomic.Int32{}
	session := newSession(t, ToolPolicy{Cache: true}, billedTool("search", calls, Usage{}, Usage{TotalTokens: 50}, nil))
	ctx, collector := WithUsageCollector(context.Background())
	call := ToolCall{ID: "1", Name: "search", Arguments: `{}`}

	_, first, err := session.Call(ctx, call)
	require.NoError(t, err)
	_, second, err := session.Call(ctx, call)
	require.NoError(t, err)

	require.EqualValues(t, 1, calls.Load())
	require.Equal(t, Usage{TotalTokens: 50}, first.UnbudgetedUsage)
	require.True(t, second.Cached)
	require.Equal(t, Usage{}, second.UnbudgetedUsage, "a cached call spends nothing")
	require.Equal(t, Usage{TotalTokens: 50}, collector.Usage())
}

func TestUnbudgetedUsageRollsUpThroughNestedStepsAndRuns(t *testing.T) {
	calls := &atomic.Int32{}
	registry, err := NewToolRegistry(billedTool("search", calls, Usage{TotalTokens: 2}, Usage{TotalTokens: 50}, nil))
	require.NoError(t, err)
	inner, err := PipelineSpec{
		Steps: []Step{Parallel("fan",
			ToolStep("left", "search", nil, nil),
			ToolStep("right", "search", nil, nil),
		)},
		Tools: registry,
	}.Build()
	require.NoError(t, err)
	log := &eventLog{}
	traces := &RecordingObserver{}
	outer, err := PipelineSpec{
		Steps: []Step{
			WithPolicy(NewProgramStep("nested", inner, nil, nil), StepPolicy{}),
			ToolStep("own", "search", nil, nil),
		},
		Tools:     registry,
		Budget:    RunBudget{MaxTokens: 6, PerStepTokens: map[string]int{"nested": 4}},
		Observers: []PipelineObserver{log, traces},
	}.Build()
	require.NoError(t, err)
	ctx, collector := WithUsageCollector(context.Background())

	_, err = outer.Run(ctx, NewState(nil))

	require.NoError(t, err, "150 unbudgeted tokens count against neither limit")
	require.EqualValues(t, 3, calls.Load())
	require.Equal(t, Usage{TotalTokens: 156}, collector.Usage())
	require.Equal(t, Usage{TotalTokens: 50}, log.finishedStep(t, "left", 1).Unbudgeted)
	require.Equal(t, Usage{TotalTokens: 50}, log.finishedStep(t, "right", 1).Unbudgeted)
	require.Equal(t, Usage{TotalTokens: 100}, log.finishedStep(t, "fan", 1).Unbudgeted)
	nested := log.finishedStep(t, "nested", 1)
	require.Equal(t, Usage{TotalTokens: 100}, nested.Unbudgeted)
	require.Equal(t, Usage{TotalTokens: 4}, nested.Usage)
	require.Len(t, log.runsFinished, 2)
	require.Equal(t, Usage{TotalTokens: 100}, log.runsFinished[0].Unbudgeted, "the nested run finishes first")
	require.Equal(t, Usage{TotalTokens: 150}, log.runsFinished[1].Unbudgeted)
	require.Equal(t, Usage{TotalTokens: 6}, log.runsFinished[1].Usage)

	trace := traces.Traces()[0]
	require.Equal(t, Usage{TotalTokens: 150}, trace.Unbudgeted)
	require.Equal(t, Usage{TotalTokens: 100}, trace.Steps[0].Unbudgeted)
	require.Len(t, trace.Steps[0].Runs, 1)
	require.Equal(t, Usage{TotalTokens: 100}, trace.Steps[0].Runs[0].Unbudgeted)
	require.Equal(t, Usage{TotalTokens: 100}, trace.Steps[0].Runs[0].Steps[0].Unbudgeted)
	require.Equal(t, Usage{TotalTokens: 50}, trace.Steps[1].Unbudgeted)
	requireTraceRoundTrip(t, trace)
	report, err := HTMLRunReport(trace, HTMLReportOptions{})
	require.NoError(t, err)
	require.Contains(t, report, "4 tokens · 100 unbudgeted tokens")
}

func TestUnbudgetedUsageInAPredictToolLoop(t *testing.T) {
	calls := &atomic.Int32{}
	tool := billedTool("lookup_order", calls, Usage{TotalTokens: 1}, Usage{TotalTokens: 50}, nil)
	registry, err := NewToolRegistry(tool)
	require.NoError(t, err)
	lm := newScriptedChat(
		toolCallReply("call-1", "lookup_order", `{}`, Usage{TotalTokens: 10}),
		textReply(orderAnswer, Usage{TotalTokens: 20}),
	)
	predict := Predict{Signature: orderSignature, LM: lm, Tools: []string{"lookup_order"}, ToolRegistry: registry}

	_, usage, err := CompiledProgram{Program: predict, Candidate: predict.SeedCandidate()}.RunWithUsage(
		context.Background(),
		Prediction{"question": "?"},
	)

	require.NoError(t, err)
	require.EqualValues(t, 1, calls.Load())
	require.Equal(t, Usage{TotalTokens: 81}, usage)
}

func TestProgramEvaluatorUsageIncludesUnbudgetedUsage(t *testing.T) {
	calls := &atomic.Int32{}
	registry, err := NewToolRegistry(
		billedTool("lookup_order", calls, Usage{TotalTokens: 1}, Usage{TotalTokens: 50}, nil),
	)
	require.NoError(t, err)
	predict := Predict{
		Signature: orderSignature,
		LM: newScriptedChat(
			toolCallReply("call-1", "lookup_order", `{}`, Usage{TotalTokens: 10}),
			textReply(orderAnswer, Usage{TotalTokens: 20}),
		),
		Tools:        []string{"lookup_order"},
		ToolRegistry: registry,
	}
	evaluator := ProgramEvaluator{Program: predict, Metric: ExactMatchMetric{Fields: []string{"status"}}}
	examples := []Example{NewIOExample("1", Prediction{"question": "?"}, Prediction{"status": "shipped"})}

	eval, err := evaluator.Evaluate(context.Background(), predict.SeedCandidate(), examples, false)

	require.NoError(t, err)
	require.Equal(t, Usage{TotalTokens: 81}, eval.Usage, "budgeted and unbudgeted together")
}

func TestRecordUnbudgetedUsage(t *testing.T) {
	t.Run("without a collector or a run", func(t *testing.T) {
		require.NotPanics(t, func() { RecordUnbudgetedUsage(context.Background(), Usage{TotalTokens: 5}) })
	})

	t.Run("into a collector", func(t *testing.T) {
		ctx, collector := WithUsageCollector(context.Background())
		RecordUsage(ctx, Usage{PromptTokens: 1, TotalTokens: 1})
		RecordUnbudgetedUsage(ctx, Usage{CompletionTokens: 2, TotalTokens: 2})
		require.Equal(t, Usage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3}, collector.Usage())
	})

	t.Run("in a run past its limits", func(t *testing.T) {
		ctx, release := startRunBudget(context.Background(), RunBudget{MaxTokens: 1, MaxModelCalls: 1})
		defer release()
		stepCtx, frame, leave := usageFrameFrom(ctx).enterStep(ctx, "s")
		defer leave()

		RecordUnbudgetedUsage(stepCtx, Usage{TotalTokens: 1000})

		require.NoError(t, frame.check())
		require.NoError(t, stepCtx.Err())
		require.NoError(t, AdmitModelCall(stepCtx), "unbudgeted usage leaves every limit open")
		require.Equal(t, Usage{TotalTokens: 1000}, frame.unbudgetedTotal())
		require.Equal(t, Usage{TotalTokens: 1000}, usageFrameFrom(ctx).unbudgetedTotal())
		require.Equal(t, Usage{}, frame.total())
		require.Equal(t, 1, BudgetFrom(stepCtx).Remaining().Tokens)
	})
}

func TestUnbudgetedUsageJSONForm(t *testing.T) {
	data, err := json.Marshal(ToolResult{Text: "ok"})
	require.NoError(t, err)
	require.JSONEq(t, `{"text":"ok","usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`, string(data),
		"a result without unbudgeted usage keeps its v0.8.0 form")

	result := ToolResult{Text: "ok", UnbudgetedUsage: Usage{TotalTokens: 50}}
	data, err = json.Marshal(result)
	require.NoError(t, err)
	require.Contains(t, string(data), `"unbudgeted_usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":50}`)
	var decoded ToolResult
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, result, decoded)

	data, err = json.Marshal(NewUsageSpan("trace", "", UsageSpanToolCall, "t"))
	require.NoError(t, err)
	require.NotContains(t, string(data), "unbudgeted", "a span without unbudgeted usage keeps its v0.8.0 form")
}
