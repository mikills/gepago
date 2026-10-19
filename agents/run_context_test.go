package agents

import (
	"context"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/stretchr/testify/require"
)

// TestAgentInAPipelineStepSeesTheStepsRunContext runs an agent inside a pipeline step: its
// model calls and tool handlers see the RunContext of the step's run.
func TestAgentInAPipelineStepSeesTheStepsRunContext(t *testing.T) {
	var seen []gepa.RunContext
	record := func(ctx context.Context) {
		rc, ok := gepa.RunContextFrom(ctx)
		require.True(t, ok)
		seen = append(seen, rc)
	}
	replies := []ChatResponse{
		callReply("call-1", "lookup", `{}`),
		{Message: Message{Role: RoleAssistant, Content: "done"}, FinishReason: FinishStop},
	}
	agent := Agent{
		Name:  "agent",
		Model: "fake",
		Client: gepa.ChatClientFunc(func(ctx context.Context, _ ChatRequest) (ChatResponse, error) {
			record(ctx)
			reply := replies[0]
			replies = replies[1:]
			return reply, nil
		}),
		Tools: []ToolBinding{{
			Definition: Tool{Name: "lookup", Description: "looks the answer up"},
			Handler: func(ctx context.Context, _ *Memory, _ string) (string, error) {
				record(ctx)
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
	require.Equal(t, []gepa.RunContext{step, step, step}, seen, "two model calls and one tool call")
}
