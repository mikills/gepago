package crucible

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/agents"
	openailm "github.com/mikills/gepago/providers/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRawModelSubjectRefusalFailsCase(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","created":0,"model":"m",
			"choices":[{"index":0,"message":{"role":"assistant","content":null,"refusal":"I can't help with that."},
			"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`))
	}))
	defer server.Close()
	lm, err := openailm.NewLanguageModel(openailm.Config{APIKey: "k", BaseURL: server.URL, Model: "m"})
	require.NoError(t, err)
	subject := RawModelSubject{SubjectName: "raw", LM: lm}

	out, err := subject.Run(context.Background(), gepa.Prediction{"q": "x"})

	require.ErrorIs(t, err, gepa.ErrContentFiltered, "a refusal must not be scored as the subject's answer")
	assert.Contains(t, err.Error(), `subject reply: "I can't help with that."`)
	assert.Nil(t, out.Value)
	assert.Equal(t, "I can't help with that.", out.Raw)
	assert.Equal(t, 8, out.Usage.TotalTokens)
	assert.Equal(t, map[string]any{"resolved_model": "m"}, out.Metadata)
}

func TestRawModelSubjectTruncationFailsCase(t *testing.T) {
	usage := gepa.Usage{PromptTokens: 4, CompletionTokens: 16, TotalTokens: 20}
	subject := RawModelSubject{
		SubjectName: "raw",
		ParseJSON:   true,
		LM: gepa.GeneratorFunc(func(context.Context, string) (gepa.Generation, error) {
			return gepa.Generation{Text: `{"answer": "par`, Usage: usage, FinishReason: gepa.FinishLength}, nil
		}),
	}

	out, err := subject.Run(context.Background(), gepa.Prediction{"q": "?"})

	require.ErrorIs(t, err, gepa.ErrOutputTruncated)
	assert.Contains(t, err.Error(), "subject reply, 16 completion tokens")
	assert.Nil(t, out.Value)
	assert.Equal(t, `{"answer": "par`, out.Raw)
	assert.Equal(t, usage, out.Usage)
}

// stopChatClient answers every chat with the same response.
type stopChatClient struct {
	response agents.ChatResponse
}

func (c stopChatClient) Chat(context.Context, agents.ChatRequest) (agents.ChatResponse, error) {
	return c.response, nil
}

func TestAgentSubjectFailsTruncatedOrRefusedRun(t *testing.T) {
	usage := gepa.Usage{PromptTokens: 5, CompletionTokens: 7, TotalTokens: 12}
	tests := []struct {
		name      string
		finish    agents.FinishReason
		wantErr   error
		wantStop  agents.StopReason
		wantInErr string
	}{
		{"complete", agents.FinishStop, nil, agents.StopComplete, ""},
		{
			"length",
			agents.FinishLength,
			gepa.ErrOutputTruncated,
			agents.StopLength,
			"agent run, 7 completion tokens",
		},
		{
			"refusal",
			agents.FinishContentFilter,
			gepa.ErrContentFiltered,
			agents.StopContentFilter,
			`agent run: "final text"`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := stopChatClient{response: agents.ChatResponse{
				Message:      agents.Message{Role: agents.RoleAssistant, Content: "final text"},
				Usage:        usage,
				FinishReason: tt.finish,
			}}
			subject := AgentSubject{
				SubjectName: "agent",
				Agent:       &agents.Agent{Name: "agent", Model: "m", Client: client},
			}

			out, err := subject.Run(context.Background(), gepa.Prediction{"prompt": "hi"})

			if tt.wantErr == nil {
				require.NoError(t, err)
			} else {
				require.ErrorIs(t, err, tt.wantErr)
				assert.Contains(t, err.Error(), tt.wantInErr)
			}
			assert.Equal(t, "final text", out.Value["final"])
			assert.Equal(t, string(tt.wantStop), out.Value["stop_reason"])
			assert.Equal(t, usage, out.Usage)
		})
	}
}

func TestOpenAIAgentSubjectMapsRefusals(t *testing.T) {
	chat, err := decodeOpenAIChatResponse([]byte(`{"model":"m","choices":[{"message":{"role":"assistant",` +
		`"content":null,"refusal":"I can't help with that."},"finish_reason":"stop"}]}`))
	require.NoError(t, err)
	assert.Equal(t, agents.FinishContentFilter, chat.FinishReason)
	assert.Equal(t, "I can't help with that.", chat.Message.Content)

	chat, err = decodeOpenAIChatResponse([]byte(`{"model":"m","choices":[{"message":{"role":"assistant",` +
		`"content":"partial","refusal":"no"},"finish_reason":"stop"}]}`))
	require.NoError(t, err)
	assert.Equal(t, agents.FinishContentFilter, chat.FinishReason)
	assert.Equal(t, "partial", chat.Message.Content, "a refusal does not replace reply text")

	responses, err := decodeOpenAIResponsesResponse([]byte(`{"model":"m","output":[{"type":"message",` +
		`"content":[{"type":"refusal","refusal":"I can't help with that."}]}]}`))
	require.NoError(t, err)
	assert.Equal(t, agents.FinishContentFilter, responses.FinishReason)
	assert.Equal(t, "I can't help with that.", responses.Message.Content)

	responses, err = decodeOpenAIResponsesResponse([]byte(`{"model":"m","output":[{"type":"message",` +
		`"content":[{"type":"output_text","text":"fine"}]}]}`))
	require.NoError(t, err)
	assert.Equal(t, agents.FinishStop, responses.FinishReason)
	assert.Equal(t, "fine", responses.Message.Content)
}
