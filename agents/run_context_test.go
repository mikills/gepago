package agents

import (
	"context"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// TestAgentInAPipelineStepSeesTheStepsRunContext runs an agent inside a pipeline step: its
// model calls and tool handlers see the RunContext of the step's run.
func TestAgentInAPipelineStepSeesTheStepsRunContext(t *testing.T) {
	var seen []gepa.RunContext
	client := gepatest.NewFakeGenerator().Default(gepatest.ToolCall("call-1", "lookup", `{}`), gepatest.Text("done"))
	agent := Agent{
		Name:   "agent",
		Model:  "fake",
		Client: client,
		Tools: []ToolBinding{{
			Definition: Tool{Name: "lookup", Description: "looks the answer up"},
			Handler: func(ctx context.Context, _ *Memory, _ string) (string, error) {
				rc, ok := gepa.RunContextFrom(ctx)
				require.True(t, ok)
				seen = append(seen, rc)
				return "found", nil
			},
		}},
	}
	answer := gepa.Key[string]{Name: "answer"}
	var step gepa.RunContext
	pipeline, err := gepa.PipelineSpec{Steps: []gepa.Step{
		gepa.NewStep("ask", nil, []gepa.KeyRef{answer.Ref()}, func(ctx context.Context, s gepa.State) error {
			step, _ = gepa.RunContextFrom(ctx)
			result, err := agent.Run(ctx, RunRequest{Memory: NewMemory("agent-run"), Messages: []Message{{Role: RoleUser, Content: "hi"}}})
			if err != nil {
				return err
			}
			answer.Set(s, result.Final.Content)
			return nil
		}),
	}}.Build()
	require.NoError(t, err)
	ctx := gepa.WithRunContext(context.Background(), gepa.RunContext{TraceID: "trace-1", Attributes: map[string]string{"job": "nightly"}})

	_, err = pipeline.Run(ctx, gepa.NewState(nil))

	require.NoError(t, err)
	require.Equal(t, "trace-1", step.TraceID)
	calls := client.Calls()
	require.Len(t, calls, 2)
	for _, call := range calls {
		require.Equal(t, step, call.RunContext, "each model call sees the step's RunContext")
		require.Equal(t, "ask", call.Step)
	}
	require.Equal(t, []gepa.RunContext{step}, seen, "the tool call sees the step's RunContext")
}
