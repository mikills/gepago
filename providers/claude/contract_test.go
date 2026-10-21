package claude

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

func TestLanguageModelMeetsTheChatContract(t *testing.T) {
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
	lm, err := NewLanguageModel(Config{APIKey: "secret", BaseURL: server.URL, Model: "claude-test"})
	require.NoError(t, err)

	require.True(t, gepatest.CheckChatClient(t, lm))
	require.Greater(t, requests.Load(), int32(2), "the check reached the server")
}
