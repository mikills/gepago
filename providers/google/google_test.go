package google

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mikills/gepago/agents"
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
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{
			"candidates":[{"content":{"role":"model","parts":[{"text":"reflection"}]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":3,"totalTokenCount":10}
		}`)
	}))
	defer server.Close()
	lm, err := NewLanguageModel(Config{APIKey: "secret", BaseURL: server.URL, Model: "gemini-test"})
	require.NoError(t, err)
	out, err := lm.Generate(context.Background(), "prompt")
	require.NoError(t, err)
	assert.Equal(t, "reflection", out)
	assert.Equal(t, 10, lm.LastUsage().TotalTokens)
}
