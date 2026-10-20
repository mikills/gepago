package gepa

import (
	"context"
	"errors"
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
	lm := newScriptedChat(
		toolCallReply("1", "lookup_order", `{"order_id":"A-1"}`, Usage{}),
		textReply(orderAnswer, Usage{}),
	)
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

func TestPredictToolsInAPipeline(t *testing.T) {
	question := Key[string]{Name: "question"}
	status := Key[string]{Name: "status"}
	summary := Key[string]{Name: "summary"}
	newSpec := func(lm Generator, predict Predict, registry *ToolRegistry, policy ToolPolicy) PipelineSpec {
		predict.Signature, predict.LM = orderSignature, lm
		return PipelineSpec{
			Inputs: refs(question.Ref()),
			Steps: []Step{
				NewProgramStep("answer", predict, refs(question.Ref()), refs(status.Ref(), summary.Ref())),
			},
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
		pipeline, err := newSpec(
			lm,
			Predict{Tools: []string{"lookup_order"}},
			registry,
			ToolPolicy{Cache: true},
		).Build()
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
		require.ErrorContains(
			t,
			err,
			`tool "lookup_order" refused: the enclosing pipeline run reached its MaxCalls limit of 1`,
		)
		require.Equal(t, before+1, calls.Load())
		records := run.tools.Records()
		require.Len(t, records, 1, "the run records the call its own session did not make")
		require.Equal(t, "answer", records[0].Step)
		require.Equal(t, `{"order_id":"A-1"}`, records[0].Arguments)
	})
}
