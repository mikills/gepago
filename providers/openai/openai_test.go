package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/openai/openai-go/v3/option"
	"github.com/stretchr/testify/require"
)

func floatPtr(value float64) *float64 { return &value }

// completionServer answers every request with one chat completion body.
func completionServer(t *testing.T, body string) *httptest.Server {
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
	lm, err := newLanguageModel(Config{BaseURL: baseURL, Model: "requested-model"}, option.WithMaxRetries(0))
	require.NoError(t, err)
	return lm
}

func TestLanguageModel(t *testing.T) {
	t.Run("requires api key for default OpenAI endpoint", func(t *testing.T) {
		_, err := NewLanguageModel(Config{Model: "gpt-test"})
		require.EqualError(t, err, "openai api key is required for the default endpoint")
	})

	t.Run("requires model", func(t *testing.T) {
		_, err := NewLanguageModel(Config{APIKey: "key"})
		require.EqualError(t, err, "openai model is required")
	})

	t.Run("allows OpenAI-compatible endpoint with custom headers", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			require.Equal(t, "/chat/completions", r.URL.Path)
			require.Equal(t, "Bearer compatible-token", r.Header.Get("Authorization"))
			require.Equal(t, "east", r.Header.Get("X-Provider-Route"))
			var request map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
			require.Equal(t, "compatible-model", request["model"])
			require.Equal(t, float64(0), request["temperature"])
			w.Header().Set("content-type", "application/json")
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-test",
				"object":"chat.completion",
				"created":0,
				"model":"compatible-model-2026",
				"choices":[{
					"index":0,
					"message":{"role":"assistant","content":"improved prompt"},
					"finish_reason":"stop"
				}],
				"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}
			}`))
		}))
		defer server.Close()

		lm, err := NewLanguageModel(Config{
			BaseURL:     server.URL,
			Model:       "compatible-model",
			Temperature: floatPtr(0),
			Headers: map[string]string{
				"Authorization":    "Bearer compatible-token",
				"X-Provider-Route": "east",
			},
		})
		require.NoError(t, err)

		got, err := lm.Generate(context.Background(), "improve this")

		require.NoError(t, err)
		require.Equal(t, gepa.Generation{
			Text:         "improved prompt",
			Usage:        gepa.Usage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3},
			Model:        "compatible-model-2026",
			FinishReason: gepa.FinishStop,
		}, got)
	})
}

func TestGenerateMapsUsageBreakdown(t *testing.T) {
	server := completionServer(t, `{
		"id":"chatcmpl-test","object":"chat.completion","created":0,"model":"o-test",
		"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
		"usage":{
			"prompt_tokens":120,"completion_tokens":40,"total_tokens":160,
			"prompt_tokens_details":{"cached_tokens":64},
			"completion_tokens_details":{"reasoning_tokens":25}
		}
	}`)

	got, err := testModel(t, server.URL).Generate(context.Background(), "prompt")

	require.NoError(t, err)
	// CacheCreationTokens stays zero: OpenAI caches prompts automatically and reports no
	// cache writes.
	require.Equal(t, gepa.Usage{
		PromptTokens:      120,
		CompletionTokens:  40,
		TotalTokens:       160,
		ReasoningTokens:   25,
		CachedInputTokens: 64,
	}, got.Usage)
}

func TestGenerateReturnsUsageOfFailedCall(t *testing.T) {
	usage := gepa.Usage{PromptTokens: 7, CompletionTokens: 5, TotalTokens: 12}
	tests := []struct {
		name    string
		choices string
		wantErr string
	}{
		{
			name:    "empty content",
			choices: `[{"index":0,"message":{"role":"assistant","content":"  "},"finish_reason":"stop"}]`,
			wantErr: "openai reflection completion: empty content",
		},
		{
			name:    "no choices",
			choices: `[]`,
			wantErr: "openai reflection completion: no choices returned",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := completionServer(t, fmt.Sprintf(`{
				"id":"chatcmpl-test","object":"chat.completion","created":0,"model":"resolved-model",
				"choices":%s,
				"usage":{"prompt_tokens":7,"completion_tokens":5,"total_tokens":12}
			}`, tt.choices))

			got, err := testModel(t, server.URL).Generate(context.Background(), "prompt")

			require.EqualError(t, err, tt.wantErr)
			require.Equal(t, usage, got.Usage)
			require.Equal(t, "resolved-model", got.Model)
		})
	}
}

func TestGenerateFinishReasons(t *testing.T) {
	tests := []struct {
		reason  string
		content string
		want    gepa.FinishReason
	}{
		{reason: "stop", content: "ok", want: gepa.FinishStop},
		{reason: "length", content: "", want: gepa.FinishLength},
		{reason: "length", content: `{"cut`, want: gepa.FinishLength},
		{reason: "content_filter", content: "withheld", want: gepa.FinishContentFilter},
		{reason: "tool_calls", content: "calling", want: gepa.FinishToolCalls},
		{reason: "function_call", content: "calling", want: gepa.FinishToolCalls},
		{reason: "something_new", content: "ok", want: gepa.FinishStop},
	}
	for _, tt := range tests {
		t.Run(tt.reason+"/"+tt.content, func(t *testing.T) {
			server := completionServer(t, fmt.Sprintf(`{
				"id":"chatcmpl-test","object":"chat.completion","created":0,"model":"m",
				"choices":[{"index":0,"message":{"role":"assistant","content":%q},"finish_reason":%q}],
				"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
			}`, tt.content, tt.reason))

			got, err := testModel(t, server.URL).Generate(context.Background(), "prompt")

			require.NoError(t, err, "a reply cut at the token limit is returned, not rejected")
			require.Equal(t, tt.want, got.FinishReason)
			require.Equal(t, tt.content, got.Text)
		})
	}
}

func TestGenerateClassifiesErrors(t *testing.T) {
	for _, tt := range []struct {
		status int
		want   gepa.ErrorKind
	}{
		{400, gepa.Permanent},
		{401, gepa.Permanent},
		{403, gepa.Permanent},
		{404, gepa.Permanent},
		{422, gepa.Permanent},
		{429, gepa.Transient},
		{500, gepa.Transient},
		{502, gepa.Transient},
		{503, gepa.Transient},
	} {
		t.Run(fmt.Sprint(tt.status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("content-type", "application/json")
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(`{"error":{"message":"failed","type":"error"}}`))
			}))
			defer server.Close()

			got, err := testModel(t, server.URL).Generate(context.Background(), "prompt")

			require.Error(t, err)
			require.Equal(t, tt.want, gepa.ErrorKindOf(err))
			require.ErrorContains(t, err, "openai reflection completion")
			require.Equal(t, gepa.Generation{}, got)
		})
	}

	t.Run("network error", func(t *testing.T) {
		server := httptest.NewServer(http.NotFoundHandler())
		url := server.URL
		server.Close()

		_, err := testModel(t, url).Generate(context.Background(), "prompt")

		require.Error(t, err)
		require.True(t, gepa.IsTransient(err), "err = %v", err)
	})

	t.Run("cancelled context", func(t *testing.T) {
		server := completionServer(t, `{}`)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		_, err := testModel(t, server.URL).Generate(ctx, "prompt")

		require.Same(t, context.Canceled, err)
		require.Empty(t, gepa.ErrorKindOf(err))
	})
}

func TestLanguageModelIsSafeForConcurrentUse(t *testing.T) {
	server := completionServer(t, `{
		"id":"chatcmpl-test","object":"chat.completion","created":0,"model":"m",
		"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
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
