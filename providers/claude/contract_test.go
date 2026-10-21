package claude

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

// helloServer answers every Messages API request with a reply, reporting a prompt cache read
// from the second request on, and counts the requests.
func helloServer(t *testing.T) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		cached := 0
		if requests.Add(1) > 1 {
			cached = 8
		}
		w.Header().Set("Content-Type", "application/json")
		// the Messages API counts cache reads apart from input_tokens; the client adds them back.
		fmt.Fprintf(w, `{
			"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
			"content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn",
			"usage":{"input_tokens":4,"output_tokens":2,"cache_read_input_tokens":%d,"cache_creation_input_tokens":1}
		}`, cached)
	}))
	t.Cleanup(server.Close)
	return server, &requests
}

func TestLanguageModelMeetsTheChatContract(t *testing.T) {
	server, requests := helloServer(t)
	lm, err := NewLanguageModel(Config{APIKey: "secret", BaseURL: server.URL, Model: "claude-test"})
	require.NoError(t, err)

	require.True(t, gepatest.CheckChatClient(t, lm))
	require.Greater(t, requests.Load(), int32(2), "the check reached the server")
}

func TestChatRefusesEveryRequestTheFakeGeneratorRefuses(t *testing.T) {
	server, requests := helloServer(t)
	lm, err := NewLanguageModel(Config{APIKey: "secret", BaseURL: server.URL, Model: "claude-test"})
	require.NoError(t, err)
	fake := gepatest.NewFakeGenerator().Default(gepatest.Text("ok"))
	user := []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}}
	tool := func(parameters string) []gepa.Tool {
		return []gepa.Tool{{Name: "lookup", Description: "Look up.", Parameters: json.RawMessage(parameters)}}
	}
	format := func(schema string, strict bool) *gepa.ResponseFormat {
		return &gepa.ResponseFormat{Name: "answer", Schema: json.RawMessage(schema), Strict: strict}
	}

	for name, req := range map[string]gepa.ChatRequest{
		"an unknown role":                   {Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}, {Role: "narrator"}}},
		"tool parameters that are not JSON": {Messages: user, Tools: tool(`{"type":`)},
		"tool parameters that are an array": {Messages: user, Tools: tool(`[1]`)},
		"tool parameters that are a string": {Messages: user, Tools: tool(`"object"`)},
		"a strict format that is not JSON":  {Messages: user, ResponseFormat: format(`{"type":`, true)},
		"a strict format of plain text":     {Messages: user, ResponseFormat: format(`not json`, true)},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := fake.Chat(context.Background(), req)
			require.ErrorIs(t, err, gepatest.ErrInvalidChatRequest)

			resp, err := lm.Chat(context.Background(), req)

			require.Error(t, err)
			require.True(t, gepa.IsPermanent(err), "%v", err)
			require.Equal(t, gepa.ChatResponse{}, resp)
			require.Zero(t, requests.Load(), "the request was not sent")
		})
	}

	// requests the fake accepts because Claude sends them: a format that is not strict is left
	// out of the request.
	for name, req := range map[string]gepa.ChatRequest{
		"a format that is not strict, of an array": {Messages: user, ResponseFormat: format(`[1]`, false)},
		"a format that is not strict, of text":     {Messages: user, ResponseFormat: format(`{"type":`, false)},
		"a number no float64 holds": {Messages: user, Tools: tool(
			`{"type":"object","properties":{"n":{"type":"number","maximum":1e400}}}`,
		)},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := fake.Chat(context.Background(), req)
			require.NoError(t, err)
			sent := requests.Load()

			_, err = lm.Chat(context.Background(), req)

			require.NoError(t, err)
			require.Equal(t, sent+1, requests.Load(), "the request was sent")
		})
	}
}
