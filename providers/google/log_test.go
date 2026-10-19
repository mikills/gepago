package google

import (
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/internal/providertest"
	"github.com/stretchr/testify/require"
)

func loggedConfig(baseURL string, recorder *providertest.LogRecorder) Config {
	return Config{APIKey: "secret-key", BaseURL: baseURL, Model: "gemini-requested", Logger: slog.New(recorder)}
}

func TestModelCallsLogTheRunContext(t *testing.T) {
	server := googleServer(t, `{
		"modelVersion":"gemini-resolved",
		"candidates":[{"content":{"role":"model","parts":[{"text":"secret reply"}]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":3,"thoughtsTokenCount":2,"totalTokenCount":12}
	}`)
	recorder := &providertest.LogRecorder{}
	lm, err := NewLanguageModel(loggedConfig(server.URL, recorder))
	require.NoError(t, err)
	client, err := NewChatClient(loggedConfig(server.URL, recorder))
	require.NoError(t, err)
	ctx := gepa.WithRunContext(context.Background(), providertest.LogRunContext())
	chat := gepa.ChatRequest{Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "secret prompt"}}}

	_, err = lm.Generate(ctx, "secret prompt")
	require.NoError(t, err)
	_, err = lm.Chat(ctx, chat)
	require.NoError(t, err)
	chat.Model = "gemini-other"
	_, err = client.Chat(ctx, chat)
	require.NoError(t, err)

	usage := gepa.Usage{PromptTokens: 7, CompletionTokens: 5, TotalTokens: 12, ReasoningTokens: 2}
	call := func(operation, model string) map[string]any {
		return providertest.WantModelCall(slog.LevelInfo, usage, map[string]any{
			"provider":       "google",
			"operation":      operation,
			"model":          model,
			"resolved_model": "gemini-resolved",
			"finish_reason":  "stop",
		})
	}
	require.Equal(t, []map[string]any{
		call("generate", "gemini-requested"),
		call("chat", "gemini-requested"),
		call("chat", "gemini-other"),
	}, recorder.ModelCalls(t, "secret prompt", "secret reply"), "each call logs one record")
}

func TestFailedModelCallsAreLoggedAtWarn(t *testing.T) {
	t.Run("after spending tokens", func(t *testing.T) {
		server := googleServer(t, `{
			"modelVersion":"gemini-resolved",
			"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":2,"totalTokenCount":11}
		}`)
		recorder := &providertest.LogRecorder{}
		lm, err := NewLanguageModel(loggedConfig(server.URL, recorder))
		require.NoError(t, err)
		ctx := gepa.WithRunContext(context.Background(), providertest.LogRunContext())

		_, err = lm.Generate(ctx, "secret prompt")

		require.EqualError(t, err, "google reflection completion: empty content")
		require.Equal(t, []map[string]any{providertest.WantModelCall(
			slog.LevelWarn,
			gepa.Usage{PromptTokens: 9, CompletionTokens: 2, TotalTokens: 11},
			map[string]any{
				"provider":       "google",
				"operation":      "generate",
				"model":          "gemini-requested",
				"resolved_model": "gemini-resolved",
				"finish_reason":  "stop",
				"error":          err.Error(),
			},
		)}, recorder.ModelCalls(t, "secret prompt"))
	})

	t.Run("with an error status, the API key redacted", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"bad request for ` + r.URL.String() + `"}}`))
		}))
		t.Cleanup(server.Close)
		recorder := &providertest.LogRecorder{}
		client, err := NewChatClient(loggedConfig(server.URL, recorder))
		require.NoError(t, err)
		ctx := gepa.WithRunContext(context.Background(), providertest.LogRunContext())

		_, err = client.Chat(ctx, gepa.ChatRequest{Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "secret prompt"}}})

		require.Error(t, err)
		records := recorder.ModelCalls(t, "secret prompt", "secret-key")
		require.Equal(t, []map[string]any{providertest.WantModelCall(slog.LevelWarn, gepa.Usage{}, map[string]any{
			"provider":  "google",
			"operation": "chat",
			"model":     "gemini-requested",
			"error":     err.Error(),
		})}, records)
		require.True(t, strings.Contains(err.Error(), "key=REDACTED"), err.Error())
	})
}
