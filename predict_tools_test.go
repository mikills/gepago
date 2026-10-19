package gepa

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
)

// chatReply is one scripted chat answer.
type chatReply struct {
	resp ChatResponse
	err  error
}

// scriptedChat is a Generator and ChatClient that answers chat calls with replies in order
// and records every request. Generate is never expected.
type scriptedChat struct {
	mu       sync.Mutex
	replies  []chatReply
	requests []ChatRequest
}

func newScriptedChat(replies ...chatReply) *scriptedChat {
	return &scriptedChat{replies: replies}
}

func (c *scriptedChat) Generate(context.Context, string) (Generation, error) {
	return Generation{}, errors.New("scripted chat: Generate is not expected")
}

func (c *scriptedChat) Chat(_ context.Context, req ChatRequest) (ChatResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.requests = append(c.requests, req)
	if len(c.replies) == 0 {
		return ChatResponse{}, errors.New("scripted chat: no reply left")
	}
	reply := c.replies[0]
	c.replies = c.replies[1:]
	return reply.resp, reply.err
}

func (c *scriptedChat) calls() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.requests)
}

func toolCallReply(id, name, arguments string, usage Usage) chatReply {
	return chatReply{resp: ChatResponse{
		Message: Message{
			Role:      RoleAssistant,
			ToolCalls: []ToolCall{{ID: id, Name: name, Arguments: arguments}},
		},
		Usage:        usage,
		FinishReason: FinishToolCalls,
	}}
}

func textReply(text string, usage Usage) chatReply {
	return chatReply{resp: ChatResponse{
		Message:      Message{Role: RoleAssistant, Content: text},
		Usage:        usage,
		FinishReason: FinishStop,
	}}
}

var orderSignature = Signature{
	Name:    "order_status",
	Inputs:  []Field{{Name: "question"}},
	Outputs: []Field{{Name: "status", Type: "string"}, {Name: "summary", Type: "string"}},
}

const orderAnswer = `{"status":"shipped","summary":"Order A-1 shipped with DHL."}`

func orderPredict(t *testing.T, lm Generator, tools ...ToolProvider) Predict {
	t.Helper()
	registry, err := NewToolRegistry(tools...)
	require.NoError(t, err)
	return Predict{Signature: orderSignature, LM: lm, Tools: []string{"lookup_order"}, ToolRegistry: registry}
}

func TestPredictToolLoop(t *testing.T) {
	calls := &atomic.Int32{}
	lm := newScriptedChat(
		toolCallReply("call-1", "lookup_order", `{"order_id":"A-1"}`, Usage{TotalTokens: 10}),
		textReply(orderAnswer, Usage{TotalTokens: 20}),
	)
	predict := orderPredict(t, lm, lookupTool(calls))
	ctx, collector := WithUsageCollector(context.Background())

	got, err := predict.Run(ctx, Prediction{"question": "Where is A-1?"})

	require.NoError(t, err)
	require.Equal(t, Prediction{"status": "shipped", "summary": "Order A-1 shipped with DHL."}, got)
	require.EqualValues(t, 1, calls.Load())
	require.Equal(t, Usage{TotalTokens: 30}, collector.Usage(), "each chat turn's usage is recorded once")
	require.Len(t, lm.requests, 2)
	first := lm.requests[0]
	require.Len(t, first.Tools, 1)
	require.Equal(t, "lookup_order", first.Tools[0].Name)
	require.NotNil(t, first.ResponseFormat, "a strict signature format is sent")
	require.True(t, first.ResponseFormat.Strict)
	require.Len(t, first.Messages, 1)
	require.Equal(t, RoleUser, first.Messages[0].Role)
	require.Contains(t, first.Messages[0].Content, "Where is A-1?")
	second := lm.requests[1].Messages
	require.Len(t, second, 3)
	require.Equal(t, RoleAssistant, second[1].Role)
	require.Equal(t, "call-1", second[1].ToolCalls[0].ID)
	require.Equal(t, Message{
		Role:       RoleTool,
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
			var replies []chatReply
			for i := range tt.toolTurns {
				replies = append(replies, toolCallReply(fmt.Sprint(i), "lookup_order", `{"order_id":"A-1"}`, Usage{}))
			}
			lm := newScriptedChat(append(replies, textReply(orderAnswer, Usage{}))...)
			predict := orderPredict(t, lm, lookupTool(calls))
			predict.MaxToolTurns = tt.maxTurns

			_, err := predict.Run(context.Background(), Prediction{"question": "?"})

			if !tt.wantErr {
				require.NoError(t, err)
				require.Equal(t, tt.toolTurns+1, lm.calls())
				require.EqualValues(t, tt.toolTurns, calls.Load())
				return
			}
			limit := tt.maxTurns
			if limit == 0 {
				limit = defaultMaxToolTurns
			}
			require.ErrorIs(t, err, ErrToolTurnLimit)
			require.True(t, IsBudget(err))
			require.EqualError(t, err, fmt.Sprintf(
				`tool turn limit reached: signature "order_status": the model called tools in more than %d replies`,
				limit,
			))
			require.Equal(t, limit+1, lm.calls())
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
			lm := newScriptedChat(
				toolCallReply("1", "lookup_order", `{"order_id":"A-1"}`, Usage{TotalTokens: 1}),
				toolCallReply("2", "lookup_order", `{"order_id":"A-2"}`, Usage{TotalTokens: 1}),
				textReply(orderAnswer, Usage{TotalTokens: 1}),
			)
			registry, err := NewToolRegistry(lookupTool(calls))
			require.NoError(t, err)
			predict := Predict{Signature: orderSignature, LM: lm, Tools: []string{"lookup_order"}}
			pipeline, err := PipelineSpec{
				Inputs: refs(Key[string]{Name: "question"}.Ref()),
				Steps: []Step{NewProgramStep("answer", predict, refs(Key[string]{Name: "question"}.Ref()),
					refs(Key[string]{Name: "status"}.Ref(), Key[string]{Name: "summary"}.Ref()))},
				Budget: RunBudget{MaxModelCalls: tt.maxCalls},
				Tools:  registry,
			}.Build()
			require.NoError(t, err)

			_, err = pipeline.Run(context.Background(), NewState(Prediction{"question": "?"}))

			require.Equal(t, tt.wantChats, lm.calls())
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
			require.True(t, IsBudget(err))
		})
	}
}

func TestPredictToolLoopValidatesTheFinalReply(t *testing.T) {
	t.Run("an invalid reply fails with ErrInvalidOutput", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := newScriptedChat(textReply(`{"status":"shipped"}`, Usage{}))

		got, err := orderPredict(t, lm, lookupTool(calls)).Run(context.Background(), Prediction{"question": "?"})

		require.ErrorIs(t, err, ErrInvalidOutput)
		require.ErrorContains(t, err, `output "summary" is missing`)
		require.Equal(t, Prediction{"status": "shipped"}, got)
	})

	t.Run("a repair continues the chat", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := newScriptedChat(
			textReply(`{"status":"shipped","extra":1}`, Usage{TotalTokens: 2}),
			toolCallReply("1", "lookup_order", `{"order_id":"A-1"}`, Usage{TotalTokens: 3}),
			textReply(orderAnswer, Usage{TotalTokens: 4}),
		)
		predict := orderPredict(t, lm, lookupTool(calls))
		predict.MaxParseRetries = 1
		ctx, collector := WithUsageCollector(context.Background())

		got, err := predict.Run(ctx, Prediction{"question": "?"})

		require.NoError(t, err)
		require.Equal(t, "shipped", got["status"])
		require.Equal(t, Usage{TotalTokens: 9}, collector.Usage())
		repair := lm.requests[1].Messages
		require.Len(t, repair, 3)
		require.Equal(t, RoleAssistant, repair[1].Role)
		require.Equal(t, `{"status":"shipped","extra":1}`, repair[1].Content)
		require.Equal(t, RoleUser, repair[2].Role)
		require.Contains(t, repair[2].Content, `field "extra" is not an output of the signature`)
		require.Contains(t, repair[2].Content, "The JSON object must match this JSON schema")
		require.NotContains(t, repair[2].Content, "Input JSON", "the original request is already in the chat")
	})

	t.Run("a truncated reply is not repaired", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := newScriptedChat(chatReply{resp: ChatResponse{
			Message:      Message{Content: `{"status":`},
			FinishReason: FinishLength,
			Usage:        Usage{CompletionTokens: 9},
		}})
		predict := orderPredict(t, lm, lookupTool(calls))
		predict.MaxParseRetries = 3

		_, err := predict.Run(context.Background(), Prediction{"question": "?"})

		require.ErrorIs(t, err, ErrOutputTruncated)
		require.Equal(t, 1, lm.calls())
	})

	t.Run("a truncated tool call does not run", func(t *testing.T) {
		calls := &atomic.Int32{}
		reply := toolCallReply("1", "lookup_order", `{"order_id":`, Usage{})
		reply.resp.FinishReason = FinishLength
		lm := newScriptedChat(reply)

		_, err := orderPredict(t, lm, lookupTool(calls)).Run(context.Background(), Prediction{"question": "?"})

		require.ErrorIs(t, err, ErrOutputTruncated)
		require.Zero(t, calls.Load())
	})

	t.Run("a refusal fails with ErrContentFiltered", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := newScriptedChat(chatReply{resp: ChatResponse{
			Message:      Message{Content: "I can't help with that."},
			FinishReason: FinishContentFilter,
		}})

		_, err := orderPredict(t, lm, lookupTool(calls)).Run(context.Background(), Prediction{"question": "?"})

		require.ErrorIs(t, err, ErrContentFiltered)
		require.ErrorContains(t, err, `"I can't help with that."`)
	})
}

func TestPredictToolLoopFailures(t *testing.T) {
	t.Run("a chat error is returned with its usage recorded", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := newScriptedChat(chatReply{resp: ChatResponse{Usage: Usage{TotalTokens: 6}}, err: TransientError(errors.New("busy"))})
		ctx, collector := WithUsageCollector(context.Background())

		_, err := orderPredict(t, lm, lookupTool(calls)).Run(ctx, Prediction{"question": "?"})

		require.EqualError(t, err, "busy")
		require.True(t, IsTransient(err))
		require.Equal(t, Usage{TotalTokens: 6}, collector.Usage())
	})

	t.Run("a tool error fails the call", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := newScriptedChat(toolCallReply("1", "lookup_order", `{"order_id":"missing"}`, Usage{}))

		_, err := orderPredict(t, lm, lookupTool(calls)).Run(context.Background(), Prediction{"question": "?"})

		require.EqualError(t, err, `tool "lookup_order": order not found`)
	})

	t.Run("a call to a registered tool the Predict does not name is refused", func(t *testing.T) {
		calls := &atomic.Int32{}
		echoCalls := &atomic.Int32{}
		lm := newScriptedChat(toolCallReply("1", "echo", `{}`, Usage{}))
		predict := orderPredict(t, lm, lookupTool(calls), countingTool("echo", echoCalls, ToolResult{}, nil))

		_, err := predict.Run(context.Background(), Prediction{"question": "?"})

		require.ErrorIs(t, err, ErrUnknownTool)
		require.EqualError(t, err, `unknown tool "echo": signature "order_status" offers only ["lookup_order"]`)
		require.True(t, IsPermanent(err))
		require.Zero(t, echoCalls.Load())
	})

	t.Run("a cancelled context makes no chat call", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := newScriptedChat(textReply(orderAnswer, Usage{}))
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		_, err := orderPredict(t, lm, lookupTool(calls)).Run(ctx, Prediction{"question": "?"})

		require.ErrorIs(t, err, context.Canceled)
		require.Zero(t, lm.calls())
	})

	t.Run("no registry", func(t *testing.T) {
		predict := Predict{Signature: orderSignature, LM: newScriptedChat(), Tools: []string{"lookup_order"}}

		_, err := predict.Run(context.Background(), Prediction{"question": "?"})

		require.EqualError(t, err, `predict tools ["lookup_order"] and chat middleware [] need a tool registry: `+
			"set Predict.ToolRegistry or run the Predict in a pipeline with PipelineSpec.Tools")
	})

	t.Run("a non-strict format is not sent", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := newScriptedChat(textReply(`{"answer":"x"}`, Usage{}))
		predict := orderPredict(t, lm, lookupTool(calls))
		predict.Signature = Signature{Inputs: []Field{{Name: "question"}}, Outputs: []Field{{Name: "answer"}}}

		_, err := predict.Run(context.Background(), Prediction{"question": "?"})

		require.NoError(t, err)
		require.Nil(t, lm.requests[0].ResponseFormat)
	})
}

func TestPredictValidateWithTools(t *testing.T) {
	calls := &atomic.Int32{}
	registry, err := NewToolRegistry(lookupTool(calls), echoTool("echo"))
	require.NoError(t, err)
	require.NoError(t, registry.RegisterMiddleware("log", func(next ChatClient) ChatClient { return next }))
	chat := newScriptedChat()
	generator := GeneratorFunc(func(context.Context, string) (Generation, error) { return Generation{}, nil })
	tests := []struct {
		name    string
		predict Predict
		wantErr string
	}{
		{name: "valid", predict: Predict{LM: chat, Tools: []string{"echo"}, Middleware: []string{"log"}, ToolRegistry: registry}},
		{name: "valid without a registry", predict: Predict{LM: chat, Tools: []string{"anything"}}},
		{
			name:    "model without Chat",
			predict: Predict{LM: generator, Tools: []string{"echo"}},
			wantErr: "predict with tools or chat middleware needs a language model that implements ChatClient; " +
				"gepa.GeneratorFunc does not",
		},
		{
			name:    "middleware needs Chat too",
			predict: Predict{LM: generator, Middleware: []string{"log"}},
			wantErr: "predict with tools or chat middleware needs a language model that implements ChatClient; " +
				"gepa.GeneratorFunc does not",
		},
		{
			name:    "negative MaxToolTurns",
			predict: Predict{LM: chat, Tools: []string{"echo"}, MaxToolTurns: -1},
			wantErr: "predict MaxToolTurns must not be negative",
		},
		{name: "empty tool name", predict: Predict{LM: chat, Tools: []string{""}}, wantErr: "tool name must not be empty"},
		{
			name:    "repeated tool",
			predict: Predict{LM: chat, Tools: []string{"echo", "echo"}},
			wantErr: `tool "echo" is listed twice`,
		},
		{
			name:    "repeated middleware",
			predict: Predict{LM: chat, Middleware: []string{"log", "log"}},
			wantErr: `chat middleware "log" is listed twice`,
		},
		{
			name:    "unknown tool",
			predict: Predict{LM: chat, Tools: []string{"missing"}, ToolRegistry: registry},
			wantErr: `unknown tool "missing": register it on the tool registry`,
		},
		{
			name: "disallowed tool",
			predict: Predict{LM: chat, Tools: []string{"echo"}, ToolRegistry: registry,
				ToolPolicy: ToolPolicy{Allow: []string{"lookup_order"}}},
			wantErr: `tool not allowed: tool "echo" is not in the tool policy's Allow list ["lookup_order"]`,
		},
		{
			name:    "unknown middleware",
			predict: Predict{LM: chat, Middleware: []string{"trace"}, ToolRegistry: registry},
			wantErr: `chat middleware "trace" is not registered`,
		},
		{
			name: "invalid policy",
			predict: Predict{LM: chat, Tools: []string{"echo"}, ToolRegistry: registry,
				ToolPolicy: ToolPolicy{MaxCallsPerTool: map[string]int{"nope": 1}}},
			wantErr: `tool policy MaxCallsPerTool names unknown tool "nope"`,
		},
		{
			name:    "policy without a registry",
			predict: Predict{LM: chat, Tools: []string{"echo"}, ToolPolicy: ToolPolicy{MaxCalls: 1}},
			wantErr: "predict ToolPolicy is set without a ToolRegistry; set ToolRegistry, or in a pipeline set " +
				"PipelineSpec.ToolPolicy instead",
		},
		{
			name:    "policy without a registry or tools",
			predict: Predict{LM: generator, ToolPolicy: ToolPolicy{Allow: []string{}}},
			wantErr: "predict ToolPolicy is set without a ToolRegistry; set ToolRegistry, or in a pipeline set " +
				"PipelineSpec.ToolPolicy instead",
		},
		{name: "empty per-tool limits are no policy", predict: Predict{LM: generator, ToolPolicy: ToolPolicy{MaxCallsPerTool: map[string]int{}}}},
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

func TestTypedPredictWithTools(t *testing.T) {
	calls := &atomic.Int32{}
	registry, err := NewToolRegistry(lookupTool(calls))
	require.NoError(t, err)
	lm := newScriptedChat(toolCallReply("1", "lookup_order", `{"order_id":"A-1"}`, Usage{}), textReply(orderAnswer, Usage{}))
	predict := TypedPredict[orderQuestion, orderAnswerFields]{
		Signature:    Signature{Name: "order_status"},
		LM:           lm,
		Tools:        []string{"lookup_order"},
		ToolRegistry: registry,
	}

	got, err := predict.Run(context.Background(), orderQuestion{Question: "Where is A-1?"})

	require.NoError(t, err)
	require.Equal(t, orderAnswerFields{Status: "shipped", Summary: "Order A-1 shipped with DHL."}, got)
	require.EqualValues(t, 1, calls.Load())

	t.Run("Build binds its tools", func(t *testing.T) {
		question := Key[string]{Name: "question"}
		step := NewProgramStep("answer", TypedPredict[orderQuestion, orderAnswerFields]{
			Signature: Signature{Name: "order_status"},
			LM:        lm,
			Tools:     []string{"missing"},
		}, refs(question.Ref()), refs(Key[string]{Name: "status"}.Ref(), Key[string]{Name: "summary"}.Ref()))

		_, err := PipelineSpec{Inputs: refs(question.Ref()), Steps: []Step{step}, Tools: registry}.Build()

		require.EqualError(t, err, `pipeline step "answer": unknown tool "missing": register it on the tool registry`)
	})

	t.Run("a model without Chat fails Validate", func(t *testing.T) {
		predict := TypedPredict[orderQuestion, orderAnswerFields]{
			LM:    GeneratorFunc(func(context.Context, string) (Generation, error) { return Generation{}, nil }),
			Tools: []string{"lookup_order"},
		}
		require.ErrorContains(t, predict.Validate(), "needs a language model that implements ChatClient")
		require.ErrorContains(t, predict.bindTools(registry, ToolPolicy{}), "implements ChatClient")
	})
}

func TestBuildAcceptsPredictStepsThatUseThePipelinesTools(t *testing.T) {
	question := Key[string]{Name: "question"}
	status := Key[string]{Name: "status"}
	summary := Key[string]{Name: "summary"}
	registry, err := NewToolRegistry(lookupTool(&atomic.Int32{}))
	require.NoError(t, err)
	lm := newScriptedChat()
	same := Predict{Signature: orderSignature, LM: lm, Tools: []string{"lookup_order"}, ToolRegistry: registry}
	typed := TypedPredict[orderQuestion, orderAnswerFields]{
		LM: lm, Tools: []string{"lookup_order"}, ToolRegistry: registry,
	}
	plain := Predict{Signature: orderSignature, LM: lm}
	for name, step := range map[string]Step{
		"the pipeline's registry":               NewProgramStep("answer", same, refs(question.Ref()), refs(status.Ref(), summary.Ref())),
		"a TypedPredict with the same":          NewProgramStep("answer", typed, refs(question.Ref()), refs(status.Ref(), summary.Ref())),
		"a Predict without tools":               NewProgramStep("answer", plain, refs(question.Ref()), refs(status.Ref(), summary.Ref())),
		"the pipeline's registry in WithPolicy": WithPolicy(NewProgramStep("answer", same, refs(question.Ref()), refs(status.Ref(), summary.Ref())), StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}}),
	} {
		_, err := PipelineSpec{
			Inputs:     refs(question.Ref()),
			Steps:      []Step{step},
			Tools:      registry,
			ToolPolicy: ToolPolicy{MaxCalls: 3},
		}.Build()
		require.NoError(t, err, name)
	}
}

func TestPredictMiddlewareWrapsEveryTurnOutermostFirst(t *testing.T) {
	var trace []string
	var mu sync.Mutex
	calls := &atomic.Int32{}
	lm := newScriptedChat(
		toolCallReply("1", "lookup_order", `{"order_id":"A-1"}`, Usage{}),
		textReply(orderAnswer, Usage{}),
	)
	predict := orderPredict(t, lm, lookupTool(calls))
	require.NoError(t, predict.ToolRegistry.RegisterMiddleware("outer", tracingMiddleware("outer", &trace, &mu)))
	require.NoError(t, predict.ToolRegistry.RegisterMiddleware("inner", tracingMiddleware("inner", &trace, &mu)))
	predict.Middleware = []string{"outer", "inner"}

	_, err := predict.Run(context.Background(), Prediction{"question": "?"})

	require.NoError(t, err)
	turn := []string{"outer>", "inner>", "<inner", "<outer"}
	require.Equal(t, append(append([]string{}, turn...), turn...), trace)
}

func TestPredictToolsInAPipeline(t *testing.T) {
	question := Key[string]{Name: "question"}
	status := Key[string]{Name: "status"}
	summary := Key[string]{Name: "summary"}
	newSpec := func(lm Generator, predict Predict, registry *ToolRegistry, policy ToolPolicy) PipelineSpec {
		predict.Signature, predict.LM = orderSignature, lm
		return PipelineSpec{
			Inputs:     refs(question.Ref()),
			Steps:      []Step{NewProgramStep("answer", predict, refs(question.Ref()), refs(status.Ref(), summary.Ref()))},
			Tools:      registry,
			ToolPolicy: policy,
		}
	}
	calls := &atomic.Int32{}
	registry, err := NewToolRegistry(lookupTool(calls), echoTool("echo"))
	require.NoError(t, err)

	t.Run("the pipeline's registry and policy apply", func(t *testing.T) {
		lm := newScriptedChat(
			toolCallReply("1", "lookup_order", `{"order_id":"A-1"}`, Usage{}),
			toolCallReply("2", "lookup_order", `{"order_id":"A-1"}`, Usage{}),
			textReply(orderAnswer, Usage{}),
		)
		before := calls.Load()
		pipeline, err := newSpec(lm, Predict{Tools: []string{"lookup_order"}}, registry, ToolPolicy{Cache: true}).Build()
		require.NoError(t, err)

		state, run, err := pipeline.execute(context.Background(), nil, Prediction{"question": "?"})

		require.NoError(t, err)
		require.Equal(t, "shipped", state.values["status"])
		require.Equal(t, before+1, calls.Load(), "the second identical call is cached")
		records := run.tools.Records()
		require.Len(t, records, 2)
		require.Equal(t, "answer", records[0].Step)
		require.False(t, records[0].Cached)
		require.True(t, records[1].Cached)
	})

	for _, tt := range []struct {
		name     string
		predict  Predict
		registry *ToolRegistry
		policy   ToolPolicy
		wantErr  string
	}{
		{
			name:     "an unknown tool fails Build",
			predict:  Predict{Tools: []string{"missing"}},
			registry: registry,
			wantErr:  `pipeline step "answer": unknown tool "missing": register it on the tool registry`,
		},
		{
			name:    "tools without a registry fail Build",
			predict: Predict{Tools: []string{"lookup_order"}},
			wantErr: `pipeline step "answer": calls tools ["lookup_order"] and chat middleware [], but the pipeline ` +
				"has no tool registry; set PipelineSpec.Tools",
		},
		{
			name:     "a disallowed tool fails Build",
			predict:  Predict{Tools: []string{"echo"}},
			registry: registry,
			policy:   ToolPolicy{Allow: []string{"lookup_order"}},
			wantErr: `pipeline step "answer": tool not allowed: tool "echo" is not in the tool policy's Allow ` +
				`list ["lookup_order"]`,
		},
		{
			name:     "unknown middleware fails Build",
			predict:  Predict{Middleware: []string{"trace"}},
			registry: registry,
			wantErr:  `pipeline step "answer": chat middleware "trace" is not registered`,
		},
		{
			name:     "a policy naming an unknown tool fails Build",
			predict:  Predict{Tools: []string{"echo"}},
			registry: registry,
			policy:   ToolPolicy{Allow: []string{"nope"}},
			wantErr:  `pipeline tool policy Allow names unknown tool "nope"`,
		},
		{
			name:    "a policy without a registry fails Build",
			predict: Predict{},
			policy:  ToolPolicy{MaxCallsPerTool: map[string]int{"echo": 1}},
			wantErr: `pipeline tool policy MaxCallsPerTool names unknown tool "echo"`,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := newSpec(newScriptedChat(), tt.predict, tt.registry, tt.policy).Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}

	t.Run("a Predict that Build did not bind meets the policy at run time", func(t *testing.T) {
		predict := Predict{Signature: orderSignature, LM: newScriptedChat(), Tools: []string{"echo"}}
		answer := NewStep("answer", refs(question.Ref()), nil, func(ctx context.Context, s State) error {
			_, err := predict.Run(ctx, Prediction{"question": "?"})
			return err
		})
		pipeline, err := PipelineSpec{
			Inputs:     refs(question.Ref()),
			Steps:      []Step{answer},
			Tools:      registry,
			ToolPolicy: ToolPolicy{Allow: []string{"lookup_order"}},
		}.Build()
		require.NoError(t, err)

		_, err = pipeline.Run(context.Background(), NewState(Prediction{"question": "?"}))

		require.ErrorIs(t, err, ErrToolNotAllowed)
	})

	t.Run("a Predict that names the pipeline's registry uses the run's session", func(t *testing.T) {
		lm := newScriptedChat(
			toolCallReply("1", "lookup_order", `{"order_id":"A-1"}`, Usage{}),
			toolCallReply("2", "lookup_order", `{"order_id":"A-1"}`, Usage{}),
			textReply(orderAnswer, Usage{}),
		)
		before := calls.Load()
		predict := Predict{Tools: []string{"lookup_order"}, ToolRegistry: registry}
		pipeline, err := newSpec(lm, predict, registry, ToolPolicy{Cache: true}).Build()
		require.NoError(t, err)

		_, run, err := pipeline.execute(context.Background(), nil, Prediction{"question": "?"})

		require.NoError(t, err)
		require.Equal(t, before+1, calls.Load(), "the run's cache answers the second call")
		require.Len(t, run.tools.Records(), 2)
	})

	own, err := NewToolRegistry(lookupTool(calls))
	require.NoError(t, err)
	for _, tt := range []struct {
		name    string
		step    Step
		wantErr string
	}{
		{
			name:    "a Predict with a registry of its own fails Build",
			step:    NewProgramStep("answer", Predict{Signature: orderSignature, LM: newScriptedChat(), Tools: []string{"lookup_order"}, ToolRegistry: own}, refs(question.Ref()), refs(status.Ref(), summary.Ref())),
			wantErr: `pipeline step "answer": predict sets a ToolRegistry other than the pipeline's`,
		},
		{
			name:    "a Predict with a registry of its own but no tools fails Build",
			step:    NewProgramStep("answer", Predict{Signature: orderSignature, LM: newScriptedChat(), ToolRegistry: own}, refs(question.Ref()), refs(status.Ref(), summary.Ref())),
			wantErr: `pipeline step "answer": predict sets a ToolRegistry other than the pipeline's`,
		},
		{
			name: "a Predict with a policy of its own fails Build",
			step: NewProgramStep("answer", Predict{
				Signature: orderSignature, LM: newScriptedChat(), Tools: []string{"lookup_order"},
				ToolRegistry: registry, ToolPolicy: ToolPolicy{MaxCalls: 9},
			}, refs(question.Ref()), refs(status.Ref(), summary.Ref())),
			wantErr: `pipeline step "answer": predict sets its own ToolPolicy, but a pipeline run's tool calls all ` +
				"go through the run's session; set PipelineSpec.ToolPolicy instead",
		},
		{
			name: "a policy without a registry fails Validate",
			step: NewProgramStep("answer", Predict{
				Signature: orderSignature, LM: newScriptedChat(), Tools: []string{"lookup_order"},
				ToolPolicy: ToolPolicy{Cache: true},
			}, refs(question.Ref()), refs(status.Ref(), summary.Ref())),
			wantErr: `pipeline step "answer": predict ToolPolicy is set without a ToolRegistry; set ToolRegistry, ` +
				"or in a pipeline set PipelineSpec.ToolPolicy instead",
		},
		{
			name: "a TypedPredict with a policy of its own fails Build",
			step: NewProgramStep("answer", TypedPredict[orderQuestion, orderAnswerFields]{
				Signature: Signature{Name: "order_status"}, LM: newScriptedChat(), Tools: []string{"lookup_order"},
				ToolRegistry: registry, ToolPolicy: ToolPolicy{Allow: []string{"lookup_order"}},
			}, refs(question.Ref()), refs(status.Ref(), summary.Ref())),
			wantErr: `pipeline step "answer": predict sets its own ToolPolicy`,
		},
		{
			name: "a nested Predict with a registry of its own fails Build",
			step: WithPolicy(NewProgramStep("answer", Predict{
				Signature: orderSignature, LM: newScriptedChat(), Tools: []string{"lookup_order"}, ToolRegistry: own,
			}, refs(question.Ref()), refs(status.Ref(), summary.Ref())), StepPolicy{}),
			wantErr: `pipeline step "answer": predict sets a ToolRegistry other than the pipeline's`,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PipelineSpec{
				Inputs: refs(question.Ref()),
				Steps:  []Step{tt.step},
				Tools:  registry,
			}.Build()
			require.ErrorContains(t, err, tt.wantErr)
			require.ErrorContains(t, err, "PipelineSpec.Tool")
		})
	}

	t.Run("a Predict whose registry escapes the run's policy fails Build", func(t *testing.T) {
		lm := newScriptedChat(
			toolCallReply("1", "lookup_order", `{"order_id":"A-1"}`, Usage{}),
			toolCallReply("2", "lookup_order", `{"order_id":"A-2"}`, Usage{}),
			textReply(orderAnswer, Usage{}),
		)
		predict := Predict{Tools: []string{"lookup_order"}, ToolRegistry: own}
		_, err := newSpec(lm, predict, registry, ToolPolicy{Allow: []string{"echo"}, MaxCalls: 1}).Build()

		require.ErrorContains(t, err, `pipeline step "answer": predict sets a ToolRegistry other than the pipeline's`)
		require.Zero(t, lm.calls())
	})

	t.Run("a Predict with its own registry in a step's code meets the run's policy and records", func(t *testing.T) {
		lm := newScriptedChat(
			toolCallReply("1", "lookup_order", `{"order_id":"A-1"}`, Usage{}),
			toolCallReply("2", "lookup_order", `{"order_id":"A-2"}`, Usage{}),
			textReply(orderAnswer, Usage{}),
		)
		predict := Predict{Signature: orderSignature, LM: lm, Tools: []string{"lookup_order"}, ToolRegistry: own}
		answer := NewStep("answer", refs(question.Ref()), nil, func(ctx context.Context, s State) error {
			_, err := predict.Run(ctx, Prediction{"question": "?"})
			return err
		})
		pipeline, err := PipelineSpec{
			Inputs:     refs(question.Ref()),
			Steps:      []Step{answer},
			Tools:      registry,
			ToolPolicy: ToolPolicy{MaxCalls: 1},
		}.Build()
		require.NoError(t, err)
		before := calls.Load()

		_, run, err := pipeline.execute(context.Background(), nil, Prediction{"question": "?"})

		require.ErrorIs(t, err, ErrToolCallLimit)
		require.ErrorContains(t, err, `tool "lookup_order" refused: the enclosing pipeline run reached its MaxCalls limit of 1`)
		require.Equal(t, before+1, calls.Load())
		records := run.tools.Records()
		require.Len(t, records, 1, "the run records the call its own session did not make")
		require.Equal(t, "answer", records[0].Step)
		require.Equal(t, `{"order_id":"A-1"}`, records[0].Arguments)
	})
}
