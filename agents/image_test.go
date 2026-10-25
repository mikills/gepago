package agents

import (
	"context"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

func TestAgentRunSendsImagePartsToTheClient(t *testing.T) {
	figure := gepatest.PNG("figure")
	client := gepatest.NewFakeGenerator().OnPrompt("Name the chart type.", gepatest.Text("A bar chart."))
	agent := Agent{Name: "vision", Client: client, Model: "fake", SystemPrompt: "Answer in three words."}
	question := Message{
		Role:    RoleUser,
		Content: "Look at the figure.",
		Parts:   []gepa.Part{{Image: figure}, {Text: "Name the chart type."}},
	}

	result, err := agent.Run(
		context.Background(),
		RunRequest{Memory: NewMemory("run-1"), Messages: []Message{question}},
	)

	require.NoError(t, err)
	require.Equal(t, "A bar chart.", result.Final.Content)
	calls := client.Calls()
	require.Len(t, calls, 1)
	require.Equal(t, []gepa.Image{figure}, calls[0].Images)
	require.Equal(
		t,
		[]Message{{Role: RoleSystem, Content: "Answer in three words."}, question},
		calls[0].Request.Messages,
	)
	require.Equal(t, question, result.Transcript[1], "the transcript keeps the parts")
}
