package agents

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// agentPipeline runs agent as the only step of a pipeline under budget, writing its final reply.
func agentPipeline(t *testing.T, agent *Agent, budget gepa.RunBudget) gepa.Pipeline {
	t.Helper()
	return agentPipelineSpec(t, agent, gepa.PipelineSpec{Budget: budget})
}

// agentPipelineSpec builds spec with agent as its only step, reading question and writing the
// agent's final reply as answer.
func agentPipelineSpec(t *testing.T, agent *Agent, spec gepa.PipelineSpec) gepa.Pipeline {
	t.Helper()
	question := gepa.Key[string]{Name: "question"}
	answer := gepa.Key[string]{Name: "answer"}
	step := gepa.NewStep(
		"agent",
		[]gepa.KeyRef{question.Ref()},
		[]gepa.KeyRef{answer.Ref()},
		func(ctx context.Context, s gepa.State) error {
			text, err := question.Get(s)
			if err != nil {
				return err
			}
			result, err := agent.Run(ctx, RunRequest{
				Memory:   NewMemory("run"),
				Messages: []Message{{Role: RoleUser, Content: text}},
			})
			if err != nil {
				return err
			}
			answer.Set(s, result.Final.Content)
			return nil
		},
	)
	spec.Inputs = []gepa.KeyRef{question.Ref()}
	spec.Steps = []gepa.Step{step}
	pipeline, err := spec.Build()
	require.NoError(t, err)
	return pipeline
}

func TestAgentInAPipelineMeetsTheRunsToolPolicy(t *testing.T) {
	other := gepa.NewToolFunc(
		gepa.Tool{Name: "other", Description: "other"},
		func(context.Context, ToolCall) (gepa.ToolResult, error) {
			return gepa.ToolResult{}, nil
		},
	)
	echo := gepa.NewToolFunc(
		gepa.Tool{Name: "echo", Description: "echo"},
		func(context.Context, ToolCall) (gepa.ToolResult, error) {
			return gepa.ToolResult{}, nil
		},
	)
	tools, err := gepa.NewToolRegistry(other, echo)
	require.NoError(t, err)
	tests := []struct {
		name      string
		policy    gepa.ToolPolicy
		wantChats int
		wantIs    error
		wantErr   string
	}{
		{
			name:      "the run's MaxCalls counts the agent's calls",
			policy:    gepa.ToolPolicy{MaxCalls: 1},
			wantChats: 2,
			wantIs:    gepa.ErrToolCallLimit,
			wantErr: `pipeline step "agent": agent "budgeted" tool call limit reached: tool "echo" refused: the ` +
				"enclosing pipeline run reached its MaxCalls limit of 1 tool calls",
		},
		{
			name:      "the run's per-tool limit counts the agent's calls",
			policy:    gepa.ToolPolicy{MaxCallsPerTool: map[string]int{"echo": 1}},
			wantChats: 2,
			wantIs:    gepa.ErrToolCallLimit,
			wantErr: `pipeline step "agent": agent "budgeted" tool call limit reached: tool "echo" reached the ` +
				"enclosing pipeline run's MaxCallsPerTool limit of 1; call 2 refused",
		},
		{
			name:      "the run's Allow list covers the agent's tools",
			policy:    gepa.ToolPolicy{Allow: []string{"other"}},
			wantChats: 1,
			wantIs:    gepa.ErrToolNotAllowed,
			wantErr: `pipeline step "agent": agent "budgeted" tool not allowed: tool "echo" is not in the Allow ` +
				`list ["other"] of the enclosing pipeline run's tool policy`,
		},
		{name: "calls within the run's limits", policy: gepa.ToolPolicy{MaxCalls: 2}, wantChats: 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent, client := threeTurnAgent()
			pipeline := agentPipelineSpec(t, agent, gepa.PipelineSpec{Tools: tools, ToolPolicy: tt.policy})

			_, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"question": "hi"}))

			require.Len(t, client.Calls(), tt.wantChats)
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.ErrorIs(t, err, tt.wantIs)
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestAgentInAPipelineWithAPolicyButNoTools(t *testing.T) {
	tests := []struct {
		name      string
		spec      gepa.PipelineSpec
		wantChats int
		wantIs    error
		wantErr   string
	}{
		{name: "no policy governs nothing", wantChats: 3},
		{
			name:      "MaxCalls counts the agent's calls",
			spec:      gepa.PipelineSpec{ToolPolicy: gepa.ToolPolicy{MaxCalls: 1}},
			wantChats: 2,
			wantIs:    gepa.ErrToolCallLimit,
			wantErr: `pipeline step "agent": agent "budgeted" tool call limit reached: tool "echo" refused: the ` +
				"enclosing pipeline run reached its MaxCalls limit of 1 tool calls",
		},
		{
			name:      "an empty Allow list refuses every call",
			spec:      gepa.PipelineSpec{ToolPolicy: gepa.ToolPolicy{Allow: []string{}}},
			wantChats: 1,
			wantIs:    gepa.ErrToolNotAllowed,
			wantErr: `pipeline step "agent": agent "budgeted" tool not allowed: tool "echo" is not in the Allow ` +
				"list [] of the enclosing pipeline run's tool policy",
		},
		{
			name:      "an empty registry is the same",
			spec:      gepa.PipelineSpec{Tools: &gepa.ToolRegistry{}, ToolPolicy: gepa.ToolPolicy{MaxCalls: 1}},
			wantChats: 2,
			wantIs:    gepa.ErrToolCallLimit,
			wantErr: `pipeline step "agent": agent "budgeted" tool call limit reached: tool "echo" refused: the ` +
				"enclosing pipeline run reached its MaxCalls limit of 1 tool calls",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent, client := threeTurnAgent()
			pipeline := agentPipelineSpec(t, agent, tt.spec)

			_, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"question": "hi"}))

			require.Len(t, client.Calls(), tt.wantChats)
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.ErrorIs(t, err, tt.wantIs)
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestAgentWithARetryingClientInABudgetedPipeline(t *testing.T) {
	tests := []struct {
		maxCalls     int
		wantAttempts int
		wantTokens   int
		wantErr      string
	}{
		{
			maxCalls:     1,
			wantAttempts: 1,
			wantTokens:   3,
			wantErr: `pipeline step "agent": agent "a" chat turn 1: run budget MaxModelCalls 1 exceeded: model ` +
				"call 2 refused",
		},
		{maxCalls: 2, wantAttempts: 2, wantTokens: 8},
		{maxCalls: 3, wantAttempts: 2, wantTokens: 8},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("MaxModelCalls %d", tt.maxCalls), func(t *testing.T) {
			// the model fails its first chat call transiently after spending three tokens, then answers.
			inner := gepatest.NewFakeGenerator().Default(
				gepatest.Fail(gepa.TransientError(errors.New("busy"))).WithUsage(Usage{TotalTokens: 3}),
				gepatest.Text("done").WithUsage(Usage{TotalTokens: 5}),
			)
			agent := &Agent{Name: "a", Client: &gepa.RetryingLanguageModel{LM: inner, MaxAttempts: 2}, Model: "m"}
			pipeline := agentPipeline(t, agent, gepa.RunBudget{MaxModelCalls: tt.maxCalls})
			ctx, collector := gepa.WithUsageCollector(context.Background())

			_, err := pipeline.Run(ctx, gepa.NewState(gepa.Prediction{"question": "hi"}))

			require.Len(t, inner.Calls(), tt.wantAttempts, "each attempt is a model call")
			require.Equal(t, tt.wantTokens, collector.Usage().TotalTokens, "each attempt's usage is recorded once")
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
			require.True(t, gepa.IsBudget(err))
		})
	}
}

func TestAgentInAPipelineAppliesItsMemoryWritesAtOnce(t *testing.T) {
	client := gepatest.NewFakeGenerator().Default(
		gepatest.ToolCall("call-1", "lookup_order", `{"order_id":"A-1"}`),
		gepatest.Text("shipped"),
	)
	agent := Agent{
		Name:         "orders",
		Client:       client,
		Model:        "fake",
		ToolRegistry: orderRegistry(t, &atomic.Int32{}),
		ToolNames:    []string{"lookup_order"},
	}
	memory := NewMemory("run")
	step := gepa.NewStep("agent", nil, nil, func(ctx context.Context, _ gepa.State) error {
		if _, err := agent.Run(ctx, RunRequest{Memory: memory, Messages: []Message{{Role: RoleUser, Content: "hi"}}}); err != nil {
			return err
		}
		return errors.New("step failed after the agent ran")
	})
	pipeline, err := gepa.PipelineSpec{Steps: []gepa.Step{step}, Tools: orderRegistry(t, &atomic.Int32{})}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), gepa.NewState(nil))

	require.EqualError(t, err, `pipeline step "agent": step failed after the agent ran`)
	last, ok := memory.Get("last_order")
	require.True(t, ok, "an agent's memory is its own, written as each tool call succeeds")
	require.Equal(t, `{"order_id":"A-1"}`, last)
}

// threeTurnAgent calls a tool twice and then answers: three chat calls of ten tokens each.
func threeTurnAgent() (*Agent, *gepatest.FakeGenerator) {
	usage := Usage{PromptTokens: 4, CompletionTokens: 6, TotalTokens: 10}
	client := gepatest.NewFakeGenerator().Default(
		gepatest.ToolCall("c1", "echo", `{}`).WithUsage(usage),
		gepatest.ToolCall("c2", "echo", `{}`).WithUsage(usage),
		gepatest.Text("done").WithUsage(usage),
	)
	return &Agent{
		Name:   "budgeted",
		Client: client,
		Model:  "fake",
		Tools: []ToolBinding{{
			Definition: Tool{Name: "echo", Description: "echo"},
			Handler:    func(context.Context, *Memory, string) (string, error) { return "ok", nil },
		}},
	}, client
}

func TestAgentInABudgetedPipeline(t *testing.T) {
	tests := []struct {
		name      string
		budget    gepa.RunBudget
		wantChats int
		wantErr   string
	}{
		{
			name:      "MaxModelCalls one below the chat calls",
			budget:    gepa.RunBudget{MaxModelCalls: 2},
			wantChats: 2,
			wantErr: `pipeline step "agent": agent "budgeted" chat turn 3: run budget MaxModelCalls 2 exceeded: ` +
				"model call 3 refused",
		},
		{name: "MaxModelCalls at the chat calls", budget: gepa.RunBudget{MaxModelCalls: 3}, wantChats: 3},
		{name: "MaxModelCalls one above the chat calls", budget: gepa.RunBudget{MaxModelCalls: 4}, wantChats: 3},
		{
			name:      "MaxTokens one below the tokens spent",
			budget:    gepa.RunBudget{MaxTokens: 29},
			wantChats: 3,
			wantErr:   `pipeline step "agent": run budget MaxTokens 29 exceeded: 30 tokens spent`,
		},
		{name: "MaxTokens at the tokens spent", budget: gepa.RunBudget{MaxTokens: 30}, wantChats: 3},
		{
			name:      "PerStepTokens one below the tokens spent",
			budget:    gepa.RunBudget{PerStepTokens: map[string]int{"agent": 19}},
			wantChats: 2,
			wantErr:   `pipeline step "agent": run budget PerStepTokens["agent"] 19 exceeded: 20 tokens spent`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent, client := threeTurnAgent()
			pipeline := agentPipeline(t, agent, tt.budget)
			ctx, collector := gepa.WithUsageCollector(context.Background())

			got, err := pipeline.Run(ctx, gepa.NewState(gepa.Prediction{"question": "hi"}))

			require.Len(t, client.Calls(), tt.wantChats)
			require.Equal(t, 10*tt.wantChats, collector.Usage().TotalTokens, "each chat call's usage is recorded once")
			if tt.wantErr == "" {
				require.NoError(t, err)
				require.Equal(t, gepa.Prediction{"answer": "done"}, got.Prediction())
				return
			}
			require.EqualError(t, err, tt.wantErr)
			require.True(t, gepa.IsBudget(err))
		})
	}
}

func TestAgentRecordsTheUsageOfAFailedChatCallOnce(t *testing.T) {
	spent := Usage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7}
	client := gepatest.NewFakeGenerator().Default(
		gepatest.Fail(gepa.TransientError(errors.New("connection reset mid-reply"))).WithUsage(spent),
	)
	agent := Agent{Name: "failing", Model: "fake", Client: client}
	ctx, collector := gepa.WithUsageCollector(context.Background())

	result, err := agent.Run(
		ctx,
		RunRequest{Memory: NewMemory("run"), Messages: []Message{{Role: RoleUser, Content: "hi"}}},
	)

	require.EqualError(t, err, `agent "failing" chat turn 1: connection reset mid-reply`)
	require.True(t, gepa.IsTransient(err))
	require.Len(t, client.Calls(), 1)
	require.Equal(t, spent, collector.Usage())
	require.Equal(t, 7, result.Ledger.TotalTokens)
	require.Equal(t, 1, result.Ledger.ModelCalls)
}

func TestAgentRefusedChatCallMakesNoCall(t *testing.T) {
	agent, client := threeTurnAgent()
	pipeline := agentPipeline(t, agent, gepa.RunBudget{MaxModelCalls: 1})

	_, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"question": "hi"}))

	require.ErrorContains(t, err, "model call 2 refused")
	require.Len(t, client.Calls(), 1)
}
