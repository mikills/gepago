package openai

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
		fmt.Fprintf(w, `{
			"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-test",
			"choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14,
				"prompt_tokens_details":{"cached_tokens":%d},"completion_tokens_details":{"reasoning_tokens":1}}
		}`, cached)
	}))
	t.Cleanup(server.Close)
	lm, err := NewLanguageModel(Config{BaseURL: server.URL, Model: "gpt-test"})
	require.NoError(t, err)

	require.True(t, gepatest.CheckChatClient(t, lm))
	require.Greater(t, requests.Load(), int32(2), "the check reached the server")
}
