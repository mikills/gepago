package google

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// cachingServer answers every generateContent request with a reply, reporting a prompt cache
// read from the second request on, and counts the requests.
func cachingServer(t *testing.T) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		cached := 0
		if requests.Add(1) > 1 {
			cached = 8
		}
		fmt.Fprintf(w, `{
			"candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":2,"thoughtsTokenCount":1,
				"totalTokenCount":15,"cachedContentTokenCount":%d},
			"modelVersion":"gemini-test"
		}`, cached)
	}))
	t.Cleanup(server.Close)
	return server, &requests
}

func TestChatClientMeetsTheChatContract(t *testing.T) {
	server, requests := cachingServer(t)
	client, err := NewChatClient(Config{APIKey: "secret", BaseURL: server.URL, Model: "gemini-test"})
	require.NoError(t, err)

	require.True(t, gepatest.CheckChatClient(t, client))
	require.True(t, gepatest.CheckChatClient(t, testModel(t, server.URL)))
	require.Greater(t, requests.Load(), int32(2), "the checks reached the server")
}

func TestChatRefusesARequestGeminiCannotTake(t *testing.T) {
	server, requests := cachingServer(t)
	client, err := NewChatClient(Config{APIKey: "secret", BaseURL: server.URL, Model: "gemini-test"})
	require.NoError(t, err)
	user := gepa.Message{Role: gepa.RoleUser, Content: "hi"}

	for name, req := range map[string]gepa.ChatRequest{
		"an unknown role": {Messages: []gepa.Message{user, {Role: "narrator", Content: "Once"}}},
		"tool parameters that are not JSON": {
			Messages: []gepa.Message{user},
			Tools:    []gepa.Tool{{Name: "lookup", Description: "Look up.", Parameters: json.RawMessage(`{"type":`)}},
		},
		"tool parameters that are not an object": {
			Messages: []gepa.Message{user},
			Tools:    []gepa.Tool{{Name: "lookup", Description: "Look up.", Parameters: json.RawMessage(`[1]`)}},
		},
		"a format schema that is not JSON": {
			Messages:       []gepa.Message{user},
			ResponseFormat: &gepa.ResponseFormat{Name: "answer", Schema: json.RawMessage(`{"type":`), Strict: true},
		},
	} {
		t.Run(name, func(t *testing.T) {
			resp, err := client.Chat(context.Background(), req)

			require.Error(t, err)
			require.True(t, gepa.IsPermanent(err), "%v", err)
			require.ErrorContains(t, err, "google chat: ")
			require.Equal(t, gepa.ChatResponse{}, resp)
			require.Zero(t, requests.Load(), "the request was not sent")
		})
	}

	_, err = client.Chat(context.Background(), gepa.ChatRequest{
		Messages:       []gepa.Message{user},
		Tools:          []gepa.Tool{{Name: "lookup", Description: "Look up."}},
		ResponseFormat: &gepa.ResponseFormat{Name: "answer", Schema: json.RawMessage(`[1]`)},
	})
	require.NoError(t, err, "a request with tools is sent without its response format")
	_, err = client.Chat(context.Background(), gepa.ChatRequest{
		Messages:       []gepa.Message{user},
		ResponseFormat: &gepa.ResponseFormat{Name: "answer"},
	})
	require.NoError(t, err, "a format without a schema asks for JSON alone")
	require.Equal(t, int32(2), requests.Load())
}
