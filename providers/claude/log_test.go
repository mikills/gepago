package claude

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/internal/providertest"
	"github.com/stretchr/testify/require"
)

func loggedModel(t *testing.T, baseURL string, recorder *providertest.LogRecorder) *LanguageModel {
	t.Helper()
	lm, err := NewLanguageModel(Config{
		APIKey:  "test-key",
		BaseURL: baseURL,
		Model:   "claude-requested",
		Logger:  slog.New(recorder),
	})
	require.NoError(t, err)
	return lm
}

func TestModelCallsLogTheRunContext(t *testing.T) {
	server := messageServer(t, `{
		"model":"claude-resolved","stop_reason":"end_turn",
		"content":[{"type":"text","text":"secret reply"}],
		"usage":{"input_tokens":3,"output_tokens":2,"cache_read_input_tokens":4}
	}`)
	recorder := &providertest.LogRecorder{}
	lm := loggedModel(t, server.URL, recorder)
	ctx := gepa.WithRunContext(context.Background(), providertest.LogRunContext())

	_, err := lm.Generate(ctx, "secret prompt")
	require.NoError(t, err)
	_, err = lm.GenerateStructured(ctx, "secret prompt", gepa.ResponseFormat{
		Schema: json.RawMessage(`{"type":"object"}`),
		Strict: true,
	})
	require.NoError(t, err)
	_, err = lm.GenerateStructured(ctx, "secret prompt", gepa.ResponseFormat{Schema: json.RawMessage(`{}`)})
	require.NoError(t, err)
	_, err = lm.Chat(ctx, gepa.ChatRequest{Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "secret prompt"}}})
	require.NoError(t, err)

	usage := gepa.Usage{PromptTokens: 7, CompletionTokens: 2, TotalTokens: 9, CachedInputTokens: 4}
	call := func(operation string) map[string]any {
		return providertest.WantModelCall(slog.LevelInfo, usage, map[string]any{
			"provider":       "claude",
			"operation":      operation,
			"model":          "claude-requested",
			"resolved_model": "claude-resolved",
			"finish_reason":  "stop",
		})
	}
	require.Equal(t, []map[string]any{
		call("generate"),
		call("generate_structured"),
		call("generate_structured"),
		call("chat"),
	}, recorder.ModelCalls(t, "secret prompt", "secret reply"))
}

func TestFailedModelCallsAreLoggedAtWarn(t *testing.T) {
	t.Run("after spending tokens", func(t *testing.T) {
		server := messageServer(t, `{
			"model":"claude-resolved","stop_reason":"end_turn",
			"content":[{"type":"text","text":"   "}],
			"usage":{"input_tokens":8,"output_tokens":3}
		}`)
		recorder := &providertest.LogRecorder{}
		ctx := gepa.WithRunContext(context.Background(), providertest.LogRunContext())

		_, err := loggedModel(t, server.URL, recorder).Generate(ctx, "secret prompt")

		require.EqualError(t, err, "claude reflection completion: no text returned")
		require.Equal(t, []map[string]any{providertest.WantModelCall(
			slog.LevelWarn,
			gepa.Usage{PromptTokens: 8, CompletionTokens: 3, TotalTokens: 11},
			map[string]any{
				"provider":       "claude",
				"operation":      "generate",
				"model":          "claude-requested",
				"resolved_model": "claude-resolved",
				"finish_reason":  "stop",
				"error":          err.Error(),
			},
		)}, recorder.ModelCalls(t, "secret prompt"))
	})

	t.Run("with an error status", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("content-type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}`))
		}))
		t.Cleanup(server.Close)
		recorder := &providertest.LogRecorder{}
		ctx := gepa.WithRunContext(context.Background(), providertest.LogRunContext())

		_, err := loggedModel(t, server.URL, recorder).Chat(ctx, gepa.ChatRequest{
			Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "secret prompt"}},
		})

		require.Error(t, err)
		require.Equal(t, []map[string]any{providertest.WantModelCall(slog.LevelWarn, gepa.Usage{}, map[string]any{
			"provider":  "claude",
			"operation": "chat",
			"model":     "claude-requested",
			"error":     err.Error(),
		})}, recorder.ModelCalls(t, "secret prompt"))
	})
}

func TestRequestsRejectedBeforeSendingAreNotLogged(t *testing.T) {
	recorder := &providertest.LogRecorder{}
	lm := loggedModel(t, "http://127.0.0.1:1", recorder)

	_, err := lm.GenerateStructured(context.Background(), "prompt", gepa.ResponseFormat{
		Schema: json.RawMessage(`{`),
		Strict: true,
	})
	require.Error(t, err)
	_, err = lm.Chat(context.Background(), gepa.ChatRequest{Messages: []gepa.Message{{Role: "narrator"}}})
	require.Error(t, err)
	require.Empty(t, recorder.Records())
}
