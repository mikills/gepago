package google

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/agents"
	"github.com/mikills/gepago/internal/providertest"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatClient(t *testing.T) {
	t.Run("text response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "/v1beta/models/gemini-test:generateContent", r.URL.Path)
			assert.Equal(t, "secret", r.URL.Query().Get("key"))
			var req map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
			assert.Contains(t, fmt.Sprint(req["systemInstruction"]), "system")
			assert.Contains(t, fmt.Sprint(req["contents"]), "hello")
			fmt.Fprint(w, `{
				"candidates":[{"content":{"role":"model","parts":[{"text":"hi there"}]},"finishReason":"STOP"}],
				"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}
			}`)
		}))
		defer server.Close()
		client, err := NewChatClient(Config{APIKey: "secret", BaseURL: server.URL, Model: "gemini-test"})
		require.NoError(t, err)
		resp, err := client.Chat(context.Background(), agents.ChatRequest{
			Model: "gemini-test",
			Messages: []agents.Message{
				{Role: agents.RoleSystem, Content: "system"},
				{Role: agents.RoleUser, Content: "hello"},
			},
		})
		require.NoError(t, err)
		assert.Equal(t, "hi there", resp.Message.Content)
		assert.Equal(t, agents.FinishStop, resp.FinishReason)
		assert.Equal(t, 5, resp.Usage.TotalTokens)
	})

	t.Run("tool call loop", func(t *testing.T) {
		var calls atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
			call := calls.Add(1)
			if call == 1 {
				assert.Contains(t, fmt.Sprint(req["tools"]), "Analyze_file")
				fmt.Fprint(w, `{
					"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"Analyze_file","args":{"path":"payments.go"}}}]},"finishReason":"STOP"}],
					"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}
				}`)
				return
			}
			contents := fmt.Sprint(req["contents"])
			assert.Contains(t, contents, "functionResponse")
			assert.Contains(t, contents, "payments.go mentions idempotency")
			fmt.Fprint(w, `{
				"candidates":[{"content":{"role":"model","parts":[{"text":"ChargeCustomer uses idempotency."}]},"finishReason":"STOP"}],
				"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":4,"totalTokenCount":16}
			}`)
		}))
		defer server.Close()
		client, err := NewChatClient(Config{APIKey: "secret", BaseURL: server.URL, Model: "gemini-test"})
		require.NoError(t, err)
		agent := agents.Agent{
			Name:   "google-agent",
			Client: client,
			Model:  "gemini-test",
			Tools: []agents.ToolBinding{{
				Definition: agents.Tool{Name: "Analyze file", Description: "Analyze a source file"},
				Handler: func(context.Context, *agents.Memory, string) (string, error) {
					return "payments.go mentions idempotency", nil
				},
			}},
		}
		result, err := agent.Run(context.Background(), agents.RunRequest{
			Memory:   agents.NewMemory("run"),
			Messages: []agents.Message{{Role: agents.RoleUser, Content: "Explain ChargeCustomer"}},
		})
		require.NoError(t, err)
		require.Len(t, result.ToolCalls, 1)
		assert.Equal(t, "Analyze file", result.ToolCalls[0].Name)
		assert.True(t, strings.Contains(result.ToolCalls[0].Arguments, "payments.go"))
		assert.Equal(t, "ChargeCustomer uses idempotency.", result.Final.Content)
	})
}

func TestLanguageModel(t *testing.T) {
	server := googleServer(t, `{
		"modelVersion":"gemini-resolved",
		"candidates":[{"content":{"role":"model","parts":[{"text":"reflection"}]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":3,"totalTokenCount":10}
	}`)

	got, err := testModel(t, server.URL).Generate(context.Background(), "prompt")

	require.NoError(t, err)
	assert.Equal(t, gepa.Generation{
		Text:         "reflection",
		Usage:        gepa.Usage{PromptTokens: 7, CompletionTokens: 3, TotalTokens: 10},
		Model:        "gemini-resolved",
		FinishReason: gepa.FinishStop,
	}, got)
}

// googleServer answers every request with one generateContent body.
func googleServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, body)
	}))
	t.Cleanup(server.Close)
	return server
}

func testModel(t *testing.T, baseURL string) *LanguageModel {
	t.Helper()
	lm, err := NewLanguageModel(Config{APIKey: "secret", BaseURL: baseURL, Model: "gemini-requested"})
	require.NoError(t, err)
	return lm
}

func TestGenerateMapsUsageBreakdown(t *testing.T) {
	server := googleServer(t, `{
		"modelVersion":"gemini-resolved",
		"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],
		"usageMetadata":{
			"promptTokenCount":100,"candidatesTokenCount":20,"thoughtsTokenCount":30,
			"cachedContentTokenCount":60,"totalTokenCount":150
		}
	}`)

	got, err := testModel(t, server.URL).Generate(context.Background(), "prompt")

	require.NoError(t, err)
	// Gemini counts thinking apart from candidate tokens, so CompletionTokens adds them back.
	// CacheCreationTokens stays zero: Gemini writes caches through the cachedContents API and
	// generateContent reports only cache reads.
	assert.Equal(t, gepa.Usage{
		PromptTokens:      100,
		CompletionTokens:  50,
		TotalTokens:       150,
		ReasoningTokens:   30,
		CachedInputTokens: 60,
	}, got.Usage)
}

func TestGenerateReturnsUsageOfFailedCall(t *testing.T) {
	usage := gepa.Usage{PromptTokens: 9, CompletionTokens: 2, TotalTokens: 11}
	tests := []struct {
		name       string
		body       string
		wantErr    string
		wantFinish gepa.FinishReason
	}{
		{
			name: "empty content",
			body: `{
				"modelVersion":"gemini-resolved",
				"candidates":[{"content":{"role":"model","parts":[{"text":" "}]},"finishReason":"STOP"}],
				"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":2,"totalTokenCount":11}
			}`,
			wantErr:    "google reflection completion: empty content",
			wantFinish: gepa.FinishStop,
		},
		{
			name: "no candidates",
			body: `{
				"modelVersion":"gemini-resolved",
				"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":2,"totalTokenCount":11}
			}`,
			wantErr: "google reflection completion: google generateContent returned no candidates",
		},
		{
			name: "blocked prompt",
			body: `{
				"modelVersion":"gemini-resolved",
				"promptFeedback":{"blockReason":"SAFETY"},
				"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":2,"totalTokenCount":11}
			}`,
			wantErr:    "google reflection completion: google generateContent blocked the prompt: SAFETY",
			wantFinish: gepa.FinishContentFilter,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := googleServer(t, tt.body)

			got, err := testModel(t, server.URL).Generate(context.Background(), "prompt")

			require.EqualError(t, err, tt.wantErr)
			assert.Equal(t, usage, got.Usage)
			assert.Equal(t, "gemini-resolved", got.Model)
			assert.Equal(t, tt.wantFinish, got.FinishReason)
		})
	}
}

func TestGenerateFinishReasons(t *testing.T) {
	tests := []struct {
		reason string
		text   string
		want   gepa.FinishReason
	}{
		{reason: "STOP", text: "ok", want: gepa.FinishStop},
		{reason: "MAX_TOKENS", text: "", want: gepa.FinishLength},
		{reason: "MAX_TOKENS", text: `{"cut`, want: gepa.FinishLength},
		{reason: "SAFETY", text: "partial", want: gepa.FinishContentFilter},
		{reason: "SAFETY", text: "", want: gepa.FinishContentFilter},
		{reason: "RECITATION", text: "partial", want: gepa.FinishContentFilter},
		{reason: "PROHIBITED_CONTENT", text: "partial", want: gepa.FinishContentFilter},
		{reason: "OTHER", text: "ok", want: ""},
		{reason: "", text: "ok", want: ""},
	}
	for _, tt := range tests {
		t.Run(tt.reason+"/"+tt.text, func(t *testing.T) {
			server := googleServer(t, fmt.Sprintf(`{
				"candidates":[{"content":{"role":"model","parts":[{"text":%q}]},"finishReason":%q}],
				"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}
			}`, tt.text, tt.reason))

			got, err := testModel(t, server.URL).Generate(context.Background(), "prompt")

			require.NoError(t, err, "a reply cut at the token limit is returned, not rejected")
			assert.Equal(t, tt.want, got.FinishReason)
			assert.Equal(t, tt.text, got.Text)
		})
	}
}

func TestGenerateClassifiesErrors(t *testing.T) {
	generate := func(ctx context.Context, baseURL string, client *http.Client) error {
		lm, err := NewLanguageModel(Config{APIKey: "secret", BaseURL: baseURL, Model: "m", Client: client})
		require.NoError(t, err)
		generation, err := lm.Generate(ctx, "prompt")
		assert.Equal(t, gepa.Generation{}, generation)
		return err
	}
	t.Run("status", func(t *testing.T) {
		providertest.RunStatusCases(t, `{"error":{"message":"failed"}}`, generate)
	})
	t.Run("transport", func(t *testing.T) {
		providertest.RunTransportCases(t, generate)
	})
	t.Run("errors name the status", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, `{"error":{"message":"failed"}}`, http.StatusBadRequest)
		}))
		defer server.Close()
		assert.ErrorContains(t, generate(context.Background(), server.URL, nil), "google generateContent status 400")
	})
	t.Run("chat client classifies too", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "slow down", http.StatusTooManyRequests)
		}))
		defer server.Close()
		client, err := NewChatClient(Config{APIKey: "secret", BaseURL: server.URL, Model: "gemini-test"})
		require.NoError(t, err)

		_, err = client.Chat(context.Background(), agents.ChatRequest{
			Messages: []agents.Message{{Role: agents.RoleUser, Content: "hi"}},
		})

		assert.True(t, gepa.IsTransient(err), "err = %v", err)
	})
}

func TestErrorsRedactTheAPIKey(t *testing.T) {
	const key = "SECRET-KEY-123"
	tlsServer := httptest.NewTLSServer(http.NotFoundHandler())
	defer tlsServer.Close()

	for name, base := range map[string]string{
		"tls failure":        tlsServer.URL,
		"unsupported scheme": "ftp://example.invalid",
	} {
		t.Run(name, func(t *testing.T) {
			lm, err := NewLanguageModel(Config{APIKey: key, BaseURL: base, Model: "m"})
			require.NoError(t, err)

			_, err = lm.Generate(context.Background(), "p")

			require.Error(t, err)
			assert.NotContains(t, err.Error(), key)
			assert.Contains(t, err.Error(), "?key=REDACTED")
			assert.True(t, gepa.IsPermanent(err), "redaction keeps the kind: %v", err)
			var urlErr *url.Error
			require.ErrorAs(t, err, &urlErr)
			assert.NotContains(t, urlErr.URL, key)
			assert.NotContains(t, urlErr.Error(), key)
		})
	}

	t.Run("a short key redacts only the parameter", func(t *testing.T) {
		lm, err := NewLanguageModel(Config{APIKey: "k", BaseURL: tlsServer.URL, Model: "m"})
		require.NoError(t, err)

		_, err = lm.Generate(context.Background(), "p")

		require.Error(t, err)
		assert.Contains(t, err.Error(), "m:generateContent?key=REDACTED")
		assert.Contains(t, err.Error(), "signed by unknown authority")
		assert.NotContains(t, err.Error(), "key=k")
	})

	t.Run("context errors stay bare", func(t *testing.T) {
		lm, err := NewLanguageModel(Config{APIKey: key, BaseURL: tlsServer.URL, Model: "m"})
		require.NoError(t, err)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		_, err = lm.Generate(ctx, "p")

		assert.Same(t, context.Canceled, err)
	})
}

func TestRedactKeyParam(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		{`Post "https://h/m:generateContent?key=abc": EOF`, `Post "https://h/m:generateContent?key=REDACTED": EOF`},
		{`https://h/p?alt=sse&key=a%2Bb&x=1`, `https://h/p?alt=sse&key=REDACTED&x=1`},
		{`look up key k in the monkey`, `look up key k in the monkey`},
		{`https://h/p?apikey=keep`, `https://h/p?apikey=keep`},
	} {
		assert.Equal(t, tt.want, redactKeyParam(tt.in))
	}
}

func TestLanguageModelIsSafeForConcurrentUse(t *testing.T) {
	server := googleServer(t, `{
		"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}
	}`)
	lm := testModel(t, server.URL)
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			got, err := lm.Generate(context.Background(), "prompt")
			assert.NoError(t, err)
			assert.Equal(t, 2, got.Usage.TotalTokens)
		})
	}
	wg.Wait()
}
