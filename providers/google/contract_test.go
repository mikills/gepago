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
	fake := gepatest.NewFakeGenerator().Default(gepatest.Text("ok"))
	user := gepa.Message{Role: gepa.RoleUser, Content: "hi"}
	tool := func(parameters string) []gepa.Tool {
		return []gepa.Tool{{Name: "lookup", Description: "Look up.", Parameters: json.RawMessage(parameters)}}
	}
	strict := func(schema string) *gepa.ResponseFormat {
		return &gepa.ResponseFormat{Name: "answer", Schema: json.RawMessage(schema), Strict: true}
	}

	// each request the fake refuses too, since no provider accepts it.
	for name, req := range map[string]gepa.ChatRequest{
		"an unknown role":                   {Messages: []gepa.Message{user, {Role: "narrator", Content: "Once"}}},
		"tool parameters that are not JSON": {Messages: []gepa.Message{user}, Tools: tool(`{"type":`)},
		"tool parameters that are an array": {Messages: []gepa.Message{user}, Tools: tool(`[1]`)},
		"tool parameters that are a string": {Messages: []gepa.Message{user}, Tools: tool(`"object"`)},
		"a strict format that is not JSON":  {Messages: []gepa.Message{user}, ResponseFormat: strict(`{"type":`)},
		"a strict format of plain text":     {Messages: []gepa.Message{user}, ResponseFormat: strict(`not json`)},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := fake.Chat(context.Background(), req)
			require.ErrorIs(t, err, gepatest.ErrInvalidChatRequest)

			resp, err := client.Chat(context.Background(), req)

			require.Error(t, err)
			require.True(t, gepa.IsPermanent(err), "%v", err)
			require.ErrorContains(t, err, "google chat: ")
			require.Equal(t, gepa.ChatResponse{}, resp)
			require.Zero(t, requests.Load(), "the request was not sent")
		})
	}

	// requests the fake accepts because Gemini sends them.
	for name, req := range map[string]gepa.ChatRequest{
		"a request with tools, sent without its response format": {
			Messages:       []gepa.Message{user},
			Tools:          tool(``),
			ResponseFormat: &gepa.ResponseFormat{Name: "answer", Schema: json.RawMessage(`[1]`)},
		},
		"a strict format that is not JSON, with tools": {
			Messages:       []gepa.Message{user},
			Tools:          tool(``),
			ResponseFormat: strict(`{"type":`),
		},
		"a format without a schema, which asks for JSON alone": {
			Messages:       []gepa.Message{user},
			ResponseFormat: &gepa.ResponseFormat{Name: "answer"},
		},
		"a number no float64 holds": {Messages: []gepa.Message{user}, Tools: tool(
			`{"type":"object","properties":{"n":{"type":"number","maximum":1e400}}}`,
		)},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := fake.Chat(context.Background(), req)
			require.NoError(t, err)
			sent := requests.Load()

			_, err = client.Chat(context.Background(), req)

			require.NoError(t, err)
			require.Equal(t, sent+1, requests.Load(), "the request was sent")
		})
	}
}
