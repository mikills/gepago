package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/internal/providertest"
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
	lm, err := NewLanguageModel(Config{BaseURL: baseURL, Model: "requested-model"})
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
		{reason: "something_new", content: "ok", want: ""},
		{reason: "", content: "ok", want: ""},
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
	generate := func(ctx context.Context, baseURL string, client *http.Client) error {
		lm, err := NewLanguageModel(Config{BaseURL: baseURL, Model: "m", Client: client})
		require.NoError(t, err)
		generation, err := lm.Generate(ctx, "prompt")
		require.Equal(t, gepa.Generation{}, generation)
		return err
	}
	t.Run("status", func(t *testing.T) {
		providertest.RunStatusCases(t, `{"error":{"message":"failed","type":"error"}}`, generate)
	})
	t.Run("transport", func(t *testing.T) {
		providertest.RunTransportCases(t, generate)
	})
	t.Run("errors name the provider", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, `{"error":{"message":"bad"}}`, http.StatusBadRequest)
		}))
		defer server.Close()
		require.ErrorContains(t, generate(context.Background(), server.URL, nil), "openai reflection completion")
	})
}

func TestRetryingLanguageModelIsTheOnlyRetryLayer(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.Header().Set("Retry-After-Ms", "1")
		http.Error(w, `{"error":{"message":"overloaded"}}`, http.StatusServiceUnavailable)
	}))
	defer server.Close()
	retrying := &gepa.RetryingLanguageModel{LM: testModel(t, server.URL), MaxAttempts: 2}

	_, err := retrying.Generate(context.Background(), "prompt")

	require.True(t, gepa.IsTransient(err), "err = %v", err)
	require.Equal(t, int32(2), requests.Load())
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

// recordingServer answers every request with body and records each request body.
func recordingServer(t *testing.T, body string) (*httptest.Server, *[]string) {
	t.Helper()
	var mu sync.Mutex
	var requests []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		mu.Lock()
		requests = append(requests, string(data))
		mu.Unlock()
		w.Header().Set("content-type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	return server, &requests
}

const structuredReply = `{
	"id":"chatcmpl-test","object":"chat.completion","created":0,"model":"m",
	"choices":[{"index":0,"message":{"role":"assistant","content":"{\"answer\":\"4\",\"confidence\":1}"},
		"finish_reason":"stop"}],
	"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}
}`

func typedSignature() gepa.Signature {
	return gepa.Signature{
		Name:    "qa",
		Inputs:  []gepa.Field{{Name: "question"}},
		Outputs: []gepa.Field{{Name: "answer", Type: "string"}, {Name: "confidence", Type: "float64"}},
	}
}

func TestGenerateStructuredSendsAStrictJSONSchema(t *testing.T) {
	server, requests := recordingServer(t, structuredReply)
	format, err := typedSignature().ResponseFormat()
	require.NoError(t, err)

	got, err := testModel(t, server.URL).GenerateStructured(context.Background(), "prompt", format)

	require.NoError(t, err)
	require.Equal(t, `{"answer":"4","confidence":1}`, got.Text)
	require.Len(t, *requests, 1)
	var request struct {
		ResponseFormat struct {
			Type       string `json:"type"`
			JSONSchema struct {
				Name   string          `json:"name"`
				Strict bool            `json:"strict"`
				Schema json.RawMessage `json:"schema"`
			} `json:"json_schema"`
		} `json:"response_format"`
	}
	require.NoError(t, json.Unmarshal([]byte((*requests)[0]), &request))
	require.Equal(t, "json_schema", request.ResponseFormat.Type)
	require.Equal(t, "qa", request.ResponseFormat.JSONSchema.Name)
	require.True(t, request.ResponseFormat.JSONSchema.Strict)
	require.JSONEq(t, string(format.Schema), string(request.ResponseFormat.JSONSchema.Schema))
	require.Less(
		t,
		strings.Index((*requests)[0], `"answer"`),
		strings.Index((*requests)[0], `"confidence"`),
		"properties keep signature order",
	)
}

func TestGenerateStructuredSendsThePromptAloneForANonStrictFormat(t *testing.T) {
	server, requests := recordingServer(t, structuredReply)
	format := gepa.ResponseFormat{Name: "qa", Schema: json.RawMessage(`{"type":"object"}`)}

	_, err := testModel(t, server.URL).GenerateStructured(context.Background(), "prompt", format)

	require.NoError(t, err)
	require.NotContains(t, (*requests)[0], "response_format")
}

func TestGenerateStructuredDefaultsTheSchemaName(t *testing.T) {
	server, requests := recordingServer(t, structuredReply)
	format := gepa.ResponseFormat{
		Schema: json.RawMessage(`{"type":"object","properties":{},"required":[],"additionalProperties":false}`),
		Strict: true,
	}

	_, err := testModel(t, server.URL).GenerateStructured(context.Background(), "prompt", format)

	require.NoError(t, err)
	require.Contains(t, (*requests)[0], `"name":"output"`)
}

func TestGenerateStructuredRejectsAnInvalidSchema(t *testing.T) {
	server, requests := recordingServer(t, structuredReply)
	format := gepa.ResponseFormat{Name: "qa", Schema: json.RawMessage(`{"type":`), Strict: true}

	_, err := testModel(t, server.URL).GenerateStructured(context.Background(), "prompt", format)

	require.EqualError(t, err, "openai reflection completion: response format schema is not valid JSON")
	require.True(t, gepa.IsPermanent(err))
	require.Empty(t, *requests)
}

func TestGenerateStructuredReportsARefusal(t *testing.T) {
	server := completionServer(t, `{
		"id":"chatcmpl-test","object":"chat.completion","created":0,"model":"m",
		"choices":[{"index":0,"message":{"role":"assistant","content":null,"refusal":"I can't help with that."},
			"finish_reason":"stop"}],
		"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}
	}`)
	format, err := typedSignature().ResponseFormat()
	require.NoError(t, err)

	got, err := testModel(t, server.URL).GenerateStructured(context.Background(), "prompt", format)

	require.EqualError(t, err, "openai reflection completion: model refused: I can't help with that.")
	require.Equal(t, gepa.FinishContentFilter, got.FinishReason)
	require.Equal(t, 8, got.Usage.TotalTokens)
}

func TestPredictSendsTheSchemaThroughRetryingLanguageModel(t *testing.T) {
	server, requests := recordingServer(t, structuredReply)
	program := gepa.Predict{
		Signature: typedSignature(),
		LM:        &gepa.RetryingLanguageModel{LM: testModel(t, server.URL), MaxAttempts: 2},
	}

	got, err := program.Run(context.Background(), gepa.Prediction{"question": "2+2?"})

	require.NoError(t, err)
	require.Equal(t, "4", got["answer"])
	require.Len(t, *requests, 1)
	require.Contains(t, (*requests)[0], `"response_format":{"json_schema":{"name":"qa"`)
	require.Contains(t, (*requests)[0], `"strict":true`)
}
