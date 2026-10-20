package gepa_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// countingTool returns result and err for every call and counts the calls.
func countingTool(name string, calls *atomic.Int32, result gepa.ToolResult, err error) gepa.ToolProvider {
	return gepa.NewToolFunc(
		gepa.Tool{Name: name, Description: "tool " + name},
		func(context.Context, gepa.ToolCall) (gepa.ToolResult, error) {
			calls.Add(1)
			return result, err
		},
	)
}

func echoTool(name string) gepa.ToolProvider {
	return gepa.NewToolFunc(
		gepa.Tool{Name: name, Description: "echo " + name, Parameters: json.RawMessage(`{"type":"object"}`)},
		func(_ context.Context, call gepa.ToolCall) (gepa.ToolResult, error) {
			return gepa.ToolResult{Text: call.Arguments}, nil
		},
	)
}

func tracingMiddleware(name string, trace *[]string, mu *sync.Mutex) gepa.ChatMiddleware {
	return func(next gepa.ChatClient) gepa.ChatClient {
		return gepa.ChatClientFunc(func(ctx context.Context, req gepa.ChatRequest) (gepa.ChatResponse, error) {
			mu.Lock()
			*trace = append(*trace, name+">")
			mu.Unlock()
			resp, err := next.Chat(ctx, req)
			mu.Lock()
			*trace = append(*trace, "<"+name)
			mu.Unlock()
			return resp, err
		})
	}
}

func TestPredictToolLoop(t *testing.T) {
	calls := &atomic.Int32{}
	lm := script(
		gepatest.ToolCall("call-1", "lookup_order", `{"order_id":"A-1"}`).WithUsage(gepa.Usage{TotalTokens: 10}),
		gepatest.Text(orderAnswer).WithUsage(gepa.Usage{TotalTokens: 20}),
	)
	predict := orderPredict(t, lm, lookupTool(calls))
	ctx, collector := gepa.WithUsageCollector(context.Background())

	got, err := predict.Run(ctx, gepa.Prediction{"question": "Where is A-1?"})

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"status": "shipped", "summary": "Order A-1 shipped with DHL."}, got)
	require.EqualValues(t, 1, calls.Load())
	require.Equal(t, gepa.Usage{TotalTokens: 30}, collector.Usage(), "each chat turn's usage is recorded once")
	require.Len(t, chatRequests(lm), 2)
	first := chatRequests(lm)[0]
	require.Len(t, first.Tools, 1)
	require.Equal(t, "lookup_order", first.Tools[0].Name)
	require.NotNil(t, first.ResponseFormat, "a strict signature format is sent")
	require.True(t, first.ResponseFormat.Strict)
	require.Len(t, first.Messages, 1)
	require.Equal(t, gepa.RoleUser, first.Messages[0].Role)
	require.Contains(t, first.Messages[0].Content, "Where is A-1?")
	second := chatRequests(lm)[1].Messages
	require.Len(t, second, 3)
	require.Equal(t, gepa.RoleAssistant, second[1].Role)
	require.Equal(t, "call-1", second[1].ToolCalls[0].ID)
	require.Equal(t, gepa.Message{
		Role:       gepa.RoleTool,
		Content:    `{"status":"shipped","carrier":"DHL"}`,
		ToolCallID: "call-1",
	}, second[2])
}

func TestPredictToolLoopIsBounded(t *testing.T) {
	tests := []struct {
		name      string
		maxTurns  int
		toolTurns int
		wantErr   bool
	}{
		{name: "limit one below the tool turns", maxTurns: 2, toolTurns: 3, wantErr: true},
		{name: "limit at the tool turns", maxTurns: 3, toolTurns: 3},
		{name: "limit one above the tool turns", maxTurns: 4, toolTurns: 3},
		{name: "default limit of 8 reached", toolTurns: 8},
		{name: "default limit of 8 exceeded", toolTurns: 9, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &atomic.Int32{}
			var replies []gepatest.Reply
			for i := range tt.toolTurns {
				replies = append(replies, gepatest.ToolCall(fmt.Sprint(i), "lookup_order", `{"order_id":"A-1"}`))
			}
			lm := script(append(replies, gepatest.Text(orderAnswer))...)
			predict := orderPredict(t, lm, lookupTool(calls))
			predict.MaxToolTurns = tt.maxTurns

			_, err := predict.Run(context.Background(), gepa.Prediction{"question": "?"})

			if !tt.wantErr {
				require.NoError(t, err)
				require.Len(t, lm.Calls(), tt.toolTurns+1)
				require.EqualValues(t, tt.toolTurns, calls.Load())
				return
			}
			limit := tt.maxTurns
			if limit == 0 {
				limit = 8 // the documented default
			}
			require.ErrorIs(t, err, gepa.ErrToolTurnLimit)
			require.True(t, gepa.IsBudget(err))
			require.EqualError(t, err, fmt.Sprintf(
				`tool turn limit reached: signature "order_status": the model called tools in more than %d replies`,
				limit,
			))
			require.Len(t, lm.Calls(), limit+1)
			require.EqualValues(t, limit, calls.Load(), "the refused reply's tool calls do not run")
		})
	}
}

func TestPredictToolLoopAdmitsEveryTurn(t *testing.T) {
	tests := []struct {
		name      string
		maxCalls  int
		wantErr   string
		wantChats int
	}{
		{
			name:      "budget one below the turns",
			maxCalls:  2,
			wantErr:   `pipeline step "answer": run budget MaxModelCalls 2 exceeded: model call 3 refused`,
			wantChats: 2,
		},
		{name: "budget at the turns", maxCalls: 3, wantChats: 3},
		{name: "budget one above the turns", maxCalls: 4, wantChats: 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &atomic.Int32{}
			lm := script(
				gepatest.ToolCall("1", "lookup_order", `{"order_id":"A-1"}`).WithUsage(gepa.Usage{TotalTokens: 1}),
				gepatest.ToolCall("2", "lookup_order", `{"order_id":"A-2"}`).WithUsage(gepa.Usage{TotalTokens: 1}),
				gepatest.Text(orderAnswer).WithUsage(gepa.Usage{TotalTokens: 1}),
			)
			registry, err := gepa.NewToolRegistry(lookupTool(calls))
			require.NoError(t, err)
			predict := gepa.Predict{Signature: orderSignature, LM: lm, Tools: []string{"lookup_order"}}
			pipeline, err := gepa.PipelineSpec{
				Inputs: refs(gepa.Key[string]{Name: "question"}.Ref()),
				Steps: []gepa.Step{
					gepa.NewProgramStep("answer", predict, refs(gepa.Key[string]{Name: "question"}.Ref()),
						refs(gepa.Key[string]{Name: "status"}.Ref(), gepa.Key[string]{Name: "summary"}.Ref())),
				},
				Budget: gepa.RunBudget{MaxModelCalls: tt.maxCalls},
				Tools:  registry,
			}.Build()
			require.NoError(t, err)

			_, err = pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"question": "?"}))

			require.Len(t, lm.Calls(), tt.wantChats)
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
			require.True(t, gepa.IsBudget(err))
		})
	}
}

func TestPredictToolLoopValidatesTheFinalReply(t *testing.T) {
	t.Run("an invalid reply fails with ErrInvalidOutput", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := script(gepatest.Text(`{"status":"shipped"}`))

		got, err := orderPredict(t, lm, lookupTool(calls)).Run(context.Background(), gepa.Prediction{"question": "?"})

		require.ErrorIs(t, err, gepa.ErrInvalidOutput)
		require.ErrorContains(t, err, `output "summary" is missing`)
		require.Equal(t, gepa.Prediction{"status": "shipped"}, got)
	})

	t.Run("a repair continues the chat", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := script(
			gepatest.Text(`{"status":"shipped","extra":1}`).WithUsage(gepa.Usage{TotalTokens: 2}),
			gepatest.ToolCall("1", "lookup_order", `{"order_id":"A-1"}`).WithUsage(gepa.Usage{TotalTokens: 3}),
			gepatest.Text(orderAnswer).WithUsage(gepa.Usage{TotalTokens: 4}),
		)
		predict := orderPredict(t, lm, lookupTool(calls))
		predict.MaxParseRetries = 1
		ctx, collector := gepa.WithUsageCollector(context.Background())

		got, err := predict.Run(ctx, gepa.Prediction{"question": "?"})

		require.NoError(t, err)
		require.Equal(t, "shipped", got["status"])
		require.Equal(t, gepa.Usage{TotalTokens: 9}, collector.Usage())
		repair := chatRequests(lm)[1].Messages
		require.Len(t, repair, 3)
		require.Equal(t, gepa.RoleAssistant, repair[1].Role)
		require.Equal(t, `{"status":"shipped","extra":1}`, repair[1].Content)
		require.Equal(t, gepa.RoleUser, repair[2].Role)
		require.Contains(t, repair[2].Content, `field "extra" is not an output of the signature`)
		require.Contains(t, repair[2].Content, "The JSON object must match this JSON schema")
		require.NotContains(t, repair[2].Content, "Input JSON", "the original request is already in the chat")
	})

	t.Run("a truncated reply is not repaired", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := script(gepatest.Reply{
			Text:         `{"status":`,
			FinishReason: gepa.FinishLength,
			Usage:        gepa.Usage{CompletionTokens: 9},
		})
		predict := orderPredict(t, lm, lookupTool(calls))
		predict.MaxParseRetries = 3

		_, err := predict.Run(context.Background(), gepa.Prediction{"question": "?"})

		require.ErrorIs(t, err, gepa.ErrOutputTruncated)
		require.Len(t, lm.Calls(), 1)
	})

	t.Run("a truncated tool call does not run", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := script(gepatest.ToolCall("1", "lookup_order", `{"order_id":`).WithFinishReason(gepa.FinishLength))

		_, err := orderPredict(t, lm, lookupTool(calls)).Run(context.Background(), gepa.Prediction{"question": "?"})

		require.ErrorIs(t, err, gepa.ErrOutputTruncated)
		require.Zero(t, calls.Load())
	})

	t.Run("a refusal fails with ErrContentFiltered", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := script(gepatest.Reply{Text: "I can't help with that.", FinishReason: gepa.FinishContentFilter})

		_, err := orderPredict(t, lm, lookupTool(calls)).Run(context.Background(), gepa.Prediction{"question": "?"})

		require.ErrorIs(t, err, gepa.ErrContentFiltered)
		require.ErrorContains(t, err, `"I can't help with that."`)
	})
}

func TestPredictToolLoopFailures(t *testing.T) {
	t.Run("a chat error is returned with its usage recorded", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := script(gepatest.Fail(gepa.TransientError(errors.New("busy"))).WithUsage(gepa.Usage{TotalTokens: 6}))
		ctx, collector := gepa.WithUsageCollector(context.Background())

		_, err := orderPredict(t, lm, lookupTool(calls)).Run(ctx, gepa.Prediction{"question": "?"})

		require.EqualError(t, err, "busy")
		require.True(t, gepa.IsTransient(err))
		require.Equal(t, gepa.Usage{TotalTokens: 6}, collector.Usage())
	})

	t.Run("a tool error fails the call", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := script(gepatest.ToolCall("1", "lookup_order", `{"order_id":"missing"}`))

		_, err := orderPredict(t, lm, lookupTool(calls)).Run(context.Background(), gepa.Prediction{"question": "?"})

		require.EqualError(t, err, `tool "lookup_order": order not found`)
	})

	t.Run("a call to a registered tool the Predict does not name is refused", func(t *testing.T) {
		calls := &atomic.Int32{}
		echoCalls := &atomic.Int32{}
		lm := script(gepatest.ToolCall("1", "echo", `{}`))
		predict := orderPredict(t, lm, lookupTool(calls), countingTool("echo", echoCalls, gepa.ToolResult{}, nil))

		_, err := predict.Run(context.Background(), gepa.Prediction{"question": "?"})

		require.ErrorIs(t, err, gepa.ErrUnknownTool)
		require.EqualError(t, err, `unknown tool "echo": signature "order_status" offers only ["lookup_order"]`)
		require.True(t, gepa.IsPermanent(err))
		require.Zero(t, echoCalls.Load())
	})

	t.Run("a cancelled context makes no chat call", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := script(gepatest.Text(orderAnswer))
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		_, err := orderPredict(t, lm, lookupTool(calls)).Run(ctx, gepa.Prediction{"question": "?"})

		require.ErrorIs(t, err, context.Canceled)
		require.Empty(t, lm.Calls())
	})

	t.Run("no registry", func(t *testing.T) {
		predict := gepa.Predict{
			Signature: orderSignature,
			LM:        gepatest.NewFakeGenerator(),
			Tools:     []string{"lookup_order"},
		}

		_, err := predict.Run(context.Background(), gepa.Prediction{"question": "?"})

		require.EqualError(t, err, `predict tools ["lookup_order"] and chat middleware [] need a tool registry: `+
			"set Predict.ToolRegistry or run the Predict in a pipeline with PipelineSpec.Tools")
	})

	t.Run("a non-strict format is not sent", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := script(gepatest.Text(`{"answer":"x"}`))
		predict := orderPredict(t, lm, lookupTool(calls))
		predict.Signature = gepa.Signature{
			Inputs:  []gepa.Field{{Name: "question"}},
			Outputs: []gepa.Field{{Name: "answer"}},
		}

		_, err := predict.Run(context.Background(), gepa.Prediction{"question": "?"})

		require.NoError(t, err)
		require.Nil(t, chatRequests(lm)[0].ResponseFormat)
	})
}

func TestPredictValidateWithTools(t *testing.T) {
	calls := &atomic.Int32{}
	registry, err := gepa.NewToolRegistry(lookupTool(calls), echoTool("echo"))
	require.NoError(t, err)
	require.NoError(t, registry.RegisterMiddleware("log", func(next gepa.ChatClient) gepa.ChatClient { return next }))
	chat := gepatest.NewFakeGenerator()
	generator := chat.Plain()
	tests := []struct {
		name    string
		predict gepa.Predict
		wantErr string
	}{
		{
			name: "valid",
			predict: gepa.Predict{
				LM:           chat,
				Tools:        []string{"echo"},
				Middleware:   []string{"log"},
				ToolRegistry: registry,
			},
		},
		{name: "valid without a registry", predict: gepa.Predict{LM: chat, Tools: []string{"anything"}}},
		{
			name:    "model without Chat",
			predict: gepa.Predict{LM: generator, Tools: []string{"echo"}},
			wantErr: "predict with tools or chat middleware needs a language model that implements ChatClient; " +
				"gepatest.plainGenerator does not",
		},
		{
			name:    "middleware needs Chat too",
			predict: gepa.Predict{LM: generator, Middleware: []string{"log"}},
			wantErr: "predict with tools or chat middleware needs a language model that implements ChatClient; " +
				"gepatest.plainGenerator does not",
		},
		{
			name:    "negative MaxToolTurns",
			predict: gepa.Predict{LM: chat, Tools: []string{"echo"}, MaxToolTurns: -1},
			wantErr: "predict MaxToolTurns must not be negative",
		},
		{
			name:    "empty tool name",
			predict: gepa.Predict{LM: chat, Tools: []string{""}},
			wantErr: "tool name must not be empty",
		},
		{
			name:    "repeated tool",
			predict: gepa.Predict{LM: chat, Tools: []string{"echo", "echo"}},
			wantErr: `tool "echo" is listed twice`,
		},
		{
			name:    "repeated middleware",
			predict: gepa.Predict{LM: chat, Middleware: []string{"log", "log"}},
			wantErr: `chat middleware "log" is listed twice`,
		},
		{
			name:    "unknown tool",
			predict: gepa.Predict{LM: chat, Tools: []string{"missing"}, ToolRegistry: registry},
			wantErr: `unknown tool "missing": register it on the tool registry`,
		},
		{
			name: "disallowed tool",
			predict: gepa.Predict{LM: chat, Tools: []string{"echo"}, ToolRegistry: registry,
				ToolPolicy: gepa.ToolPolicy{Allow: []string{"lookup_order"}}},
			wantErr: `tool not allowed: tool "echo" is not in the tool policy's Allow list ["lookup_order"]`,
		},
		{
			name:    "unknown middleware",
			predict: gepa.Predict{LM: chat, Middleware: []string{"trace"}, ToolRegistry: registry},
			wantErr: `chat middleware "trace" is not registered`,
		},
		{
			name: "invalid policy",
			predict: gepa.Predict{LM: chat, Tools: []string{"echo"}, ToolRegistry: registry,
				ToolPolicy: gepa.ToolPolicy{MaxCallsPerTool: map[string]int{"nope": 1}}},
			wantErr: `tool policy MaxCallsPerTool names unknown tool "nope"`,
		},
		{
			name:    "policy without a registry",
			predict: gepa.Predict{LM: chat, Tools: []string{"echo"}, ToolPolicy: gepa.ToolPolicy{MaxCalls: 1}},
			wantErr: "predict ToolPolicy is set without a ToolRegistry; set ToolRegistry, or in a pipeline set " +
				"PipelineSpec.ToolPolicy instead",
		},
		{
			name:    "policy without a registry or tools",
			predict: gepa.Predict{LM: generator, ToolPolicy: gepa.ToolPolicy{Allow: []string{}}},
			wantErr: "predict ToolPolicy is set without a ToolRegistry; set ToolRegistry, or in a pipeline set " +
				"PipelineSpec.ToolPolicy instead",
		},
		{
			name:    "empty per-tool limits are no policy",
			predict: gepa.Predict{LM: generator, ToolPolicy: gepa.ToolPolicy{MaxCallsPerTool: map[string]int{}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.predict.Signature = orderSignature
			err := tt.predict.Validate()
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

type orderQuestion struct {
	Question string `json:"question"`
}

type orderAnswerFields struct {
	Status  string `json:"status"`
	Summary string `json:"summary"`
}

func TestBuildAcceptsPredictStepsThatUseThePipelinesTools(t *testing.T) {
	question := gepa.Key[string]{Name: "question"}
	status := gepa.Key[string]{Name: "status"}
	summary := gepa.Key[string]{Name: "summary"}
	registry, err := gepa.NewToolRegistry(lookupTool(&atomic.Int32{}))
	require.NoError(t, err)
	lm := gepatest.NewFakeGenerator()
	same := gepa.Predict{Signature: orderSignature, LM: lm, Tools: []string{"lookup_order"}, ToolRegistry: registry}
	typed := gepa.TypedPredict[orderQuestion, orderAnswerFields]{
		LM: lm, Tools: []string{"lookup_order"}, ToolRegistry: registry,
	}
	plain := gepa.Predict{Signature: orderSignature, LM: lm}
	for name, step := range map[string]gepa.Step{
		"the pipeline's registry":               gepa.NewProgramStep("answer", same, refs(question.Ref()), refs(status.Ref(), summary.Ref())),
		"a TypedPredict with the same":          gepa.NewProgramStep("answer", typed, refs(question.Ref()), refs(status.Ref(), summary.Ref())),
		"a Predict without tools":               gepa.NewProgramStep("answer", plain, refs(question.Ref()), refs(status.Ref(), summary.Ref())),
		"the pipeline's registry in WithPolicy": gepa.WithPolicy(gepa.NewProgramStep("answer", same, refs(question.Ref()), refs(status.Ref(), summary.Ref())), gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 2}}),
	} {
		_, err := gepa.PipelineSpec{
			Inputs:     refs(question.Ref()),
			Steps:      []gepa.Step{step},
			Tools:      registry,
			ToolPolicy: gepa.ToolPolicy{MaxCalls: 3},
		}.Build()
		require.NoError(t, err, name)
	}
}

func TestPredictMiddlewareWrapsEveryTurnOutermostFirst(t *testing.T) {
	var trace []string
	var mu sync.Mutex
	calls := &atomic.Int32{}
	lm := script(
		gepatest.ToolCall("1", "lookup_order", `{"order_id":"A-1"}`),
		gepatest.Text(orderAnswer),
	)
	predict := orderPredict(t, lm, lookupTool(calls))
	require.NoError(t, predict.ToolRegistry.RegisterMiddleware("outer", tracingMiddleware("outer", &trace, &mu)))
	require.NoError(t, predict.ToolRegistry.RegisterMiddleware("inner", tracingMiddleware("inner", &trace, &mu)))
	predict.Middleware = []string{"outer", "inner"}

	_, err := predict.Run(context.Background(), gepa.Prediction{"question": "?"})

	require.NoError(t, err)
	turn := []string{"outer>", "inner>", "<inner", "<outer"}
	require.Equal(t, append(append([]string{}, turn...), turn...), trace)
}
