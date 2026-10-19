package claude

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/internal/providertest"
	"github.com/stretchr/testify/require"
)

// messageServer answers every request with one messages body.
func messageServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("content-type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	return server
}

func testModel(t *testing.T, baseURL string) *LanguageModel {
	t.Helper()
	lm, err := NewLanguageModel(Config{APIKey: "test-key", BaseURL: baseURL, Model: "claude-requested", MaxTokens: 64})
	require.NoError(t, err)
	return lm
}

func TestClaudeLanguageModel(t *testing.T) {
	t.Run("validates config", func(t *testing.T) {
		_, err := NewLanguageModel(Config{Model: "claude-test"})
		require.EqualError(t, err, "claude api key is required")
		_, err = NewLanguageModel(Config{APIKey: "key"})
		require.EqualError(t, err, "claude model is required")
	})

	t.Run("generates text", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			require.Equal(t, "/v1/messages", r.URL.Path)
			require.Equal(t, "test-key", r.Header.Get("x-api-key"))
			require.Equal(t, "2023-06-01", r.Header.Get("anthropic-version"))
			var request map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
			require.Equal(t, "claude-test", request["model"])
			require.Equal(t, float64(64), request["max_tokens"])
			w.Header().Set("content-type", "application/json")
			_, _ = w.Write([]byte(`{
				"model":"claude-resolved","stop_reason":"end_turn",
				"content":[{"type":"text","text":"improved prompt"}],
				"usage":{"input_tokens":2,"output_tokens":1}
			}`))
		}))
		defer server.Close()

		lm, err := NewLanguageModel(Config{
			APIKey:    "test-key",
			BaseURL:   server.URL,
			Model:     "claude-test",
			MaxTokens: 64,
		})
		require.NoError(t, err)

		got, err := lm.Generate(context.Background(), "make this better")

		require.NoError(t, err)
		require.Equal(t, gepa.Generation{
			Text:         "improved prompt",
			Usage:        gepa.Usage{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3},
			Model:        "claude-resolved",
			FinishReason: gepa.FinishStop,
		}, got)
	})
}

func TestGenerateMapsUsageBreakdown(t *testing.T) {
	server := messageServer(t, `{
		"model":"claude-resolved","stop_reason":"end_turn",
		"content":[{"type":"text","text":"ok"}],
		"usage":{
			"input_tokens":10,"output_tokens":30,
			"cache_read_input_tokens":200,"cache_creation_input_tokens":50
		}
	}`)

	got, err := testModel(t, server.URL).Generate(context.Background(), "prompt")

	require.NoError(t, err)
	// ReasoningTokens stays zero: Claude counts extended-thinking tokens inside output_tokens
	// and does not report them separately.
	require.Equal(t, gepa.Usage{
		PromptTokens:        260,
		CompletionTokens:    30,
		TotalTokens:         290,
		CachedInputTokens:   200,
		CacheCreationTokens: 50,
	}, got.Usage)
}

func TestGenerateReturnsUsageOfFailedCall(t *testing.T) {
	server := messageServer(t, `{
		"model":"claude-resolved","stop_reason":"end_turn",
		"content":[{"type":"text","text":"   "}],
		"usage":{"input_tokens":8,"output_tokens":3}
	}`)

	got, err := testModel(t, server.URL).Generate(context.Background(), "prompt")

	require.EqualError(t, err, "claude reflection completion: no text returned")
	require.Equal(t, gepa.Usage{PromptTokens: 8, CompletionTokens: 3, TotalTokens: 11}, got.Usage)
	require.Equal(t, "claude-resolved", got.Model)
}

func TestGenerateFinishReasons(t *testing.T) {
	tests := []struct {
		stopReason string
		text       string
		want       gepa.FinishReason
	}{
		{stopReason: "end_turn", text: "ok", want: gepa.FinishStop},
		{stopReason: "stop_sequence", text: "ok", want: gepa.FinishStop},
		{stopReason: "max_tokens", text: "", want: gepa.FinishLength},
		{stopReason: "max_tokens", text: `{"cut`, want: gepa.FinishLength},
		{stopReason: "tool_use", text: "calling", want: gepa.FinishToolCalls},
		{stopReason: "refusal", text: "declined", want: gepa.FinishContentFilter},
		{stopReason: "model_context_window_exceeded", text: `{"cut`, want: gepa.FinishLength},
		{stopReason: "pause_turn", text: "paused", want: ""},
		{stopReason: "", text: "ok", want: ""},
	}
	for _, tt := range tests {
		t.Run(tt.stopReason+"/"+tt.text, func(t *testing.T) {
			server := messageServer(t, fmt.Sprintf(`{
				"model":"m","stop_reason":%q,
				"content":[{"type":"text","text":%q}],
				"usage":{"input_tokens":1,"output_tokens":1}
			}`, tt.stopReason, tt.text))

			got, err := testModel(t, server.URL).Generate(context.Background(), "prompt")

			require.NoError(t, err, "a reply cut at the token limit is returned, not rejected")
			require.Equal(t, tt.want, got.FinishReason)
			require.Equal(t, tt.text, got.Text)
		})
	}
}

func TestContextWindowTruncationFailsPredict(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.Header().Set("content-type", "application/json")
		_, _ = w.Write([]byte(`{"model":"m","stop_reason":"model_context_window_exceeded",
			"content":[{"type":"text","text":"{\"answer\": \"the reply was cut"}],
			"usage":{"input_tokens":199000,"output_tokens":1000}}`))
	}))
	defer server.Close()
	program := gepa.Predict{
		Signature: gepa.Signature{
			Name:    "qa",
			Inputs:  []gepa.Field{{Name: "q"}},
			Outputs: []gepa.Field{{Name: "answer"}},
		},
		LM:              testModel(t, server.URL),
		MaxParseRetries: 1,
	}

	_, err := program.Run(context.Background(), gepa.Prediction{"q": "?"})

	require.ErrorIs(t, err, gepa.ErrOutputTruncated)
	require.Equal(t, int32(1), requests.Load(), "a truncated reply must not be sent for repair")
}

func TestGenerateClassifiesErrors(t *testing.T) {
	generate := func(ctx context.Context, baseURL string, client *http.Client) error {
		lm, err := NewLanguageModel(Config{APIKey: "test-key", BaseURL: baseURL, Model: "m", Client: client})
		require.NoError(t, err)
		generation, err := lm.Generate(ctx, "prompt")
		require.Equal(t, gepa.Generation{}, generation)
		return err
	}
	t.Run("status", func(t *testing.T) {
		providertest.RunStatusCases(t, `{"type":"error","error":{"type":"api_error","message":"failed"}}`, generate)
	})
	t.Run("transport", func(t *testing.T) {
		providertest.RunTransportCases(t, generate)
	})
	t.Run("errors name the provider", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, `{"type":"error","error":{"type":"invalid_request_error"}}`, http.StatusBadRequest)
		}))
		defer server.Close()
		require.ErrorContains(t, generate(context.Background(), server.URL, nil), "claude reflection completion")
	})
}

func TestRetryingLanguageModelIsTheOnlyRetryLayer(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.Header().Set("content-type", "application/json")
		w.Header().Set("retry-after-ms", "1")
		w.WriteHeader(529)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"x"}}`))
	}))
	defer server.Close()
	retrying := &gepa.RetryingLanguageModel{LM: testModel(t, server.URL), MaxAttempts: 2}

	_, err := retrying.Generate(context.Background(), "prompt")

	require.True(t, gepa.IsTransient(err), "err = %v", err)
	require.Equal(t, int32(2), requests.Load())
}

func TestLanguageModelIsSafeForConcurrentUse(t *testing.T) {
	server := messageServer(t, `{
		"model":"m","stop_reason":"end_turn",
		"content":[{"type":"text","text":"ok"}],
		"usage":{"input_tokens":1,"output_tokens":1}
	}`)
	lm := testModel(t, server.URL)
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			got, err := lm.Generate(context.Background(), "prompt")
			require.NoError(t, err)
			require.Equal(t, 2, got.Usage.TotalTokens)
		})
	}
	wg.Wait()
}
