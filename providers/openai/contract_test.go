package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/mikills/gepago/internal/providertest"
	"github.com/stretchr/testify/require"
)

// helloServer answers every chat completion with a reply, reporting a prompt cache read of 8 of
// its 12 prompt tokens when the same request came before, and counts the requests.
func helloServer(t *testing.T) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	return providertest.PromptCacheServer(t, func(hit bool) string {
		cached := 0
		if hit {
			cached = 8
		}
		return fmt.Sprintf(`{
			"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-test",
			"choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14,
				"prompt_tokens_details":{"cached_tokens":%d},"completion_tokens_details":{"reasoning_tokens":1}}
		}`, cached)
	})
}

func TestLanguageModelMeetsTheChatContract(t *testing.T) {
	server, requests := helloServer(t)
	lm, err := NewLanguageModel(Config{BaseURL: server.URL, Model: "gpt-test"})
	require.NoError(t, err)

	require.True(t, gepatest.CheckChatClient(t, lm))
	require.Greater(t, requests.Load(), int32(2), "the check reached the server")
}

func TestChatRefusesEveryRequestTheFakeGeneratorRefuses(t *testing.T) {
	server, requests := helloServer(t)
	lm, err := NewLanguageModel(Config{BaseURL: server.URL, Model: "gpt-test"})
	require.NoError(t, err)
	fake := gepatest.NewFakeGenerator().Default(gepatest.Text("ok"))
	user := []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}}
	tool := func(parameters string) []gepa.Tool {
		return []gepa.Tool{{Name: "lookup", Description: "Look up.", Parameters: json.RawMessage(parameters)}}
	}
	strict := func(schema string) *gepa.ResponseFormat {
		return &gepa.ResponseFormat{Name: "answer", Schema: json.RawMessage(schema), Strict: true}
	}

	for name, req := range map[string]gepa.ChatRequest{
		"an unknown role":                   {Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}, {Role: "narrator"}}},
		"tool parameters that are not JSON": {Messages: user, Tools: tool(`{"type":`)},
		"tool parameters that are an array": {Messages: user, Tools: tool(`[1]`)},
		"tool parameters that are a string": {Messages: user, Tools: tool(`"object"`)},
		"a strict format that is not JSON":  {Messages: user, ResponseFormat: strict(`{"type":`)},
		"a strict format of plain text":     {Messages: user, ResponseFormat: strict(`not json`)},
		"a strict format of spaces":         {Messages: user, ResponseFormat: strict(" \n")},
		"a NaN temperature":                 {Messages: user, Temperature: floatPtr(math.NaN())},
		"an infinite temperature":           {Messages: user, Temperature: floatPtr(math.Inf(1))},
		"a temperature of minus infinity":   {Messages: user, Temperature: floatPtr(math.Inf(-1))},
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

	// requests the fake accepts because OpenAI sends them; its API may still refuse them.
	for name, req := range map[string]gepa.ChatRequest{
		"a strict format that is an array": {Messages: user, ResponseFormat: strict(`[1]`)},
		"a strict format of null":          {Messages: user, ResponseFormat: strict(`null`)},
		"a number no float64 holds": {Messages: user, Tools: tool(
			`{"type":"object","properties":{"n":{"type":"number","maximum":1e400}}}`,
		)},
		"the largest finite temperature": {Messages: user, Temperature: floatPtr(math.MaxFloat64)},
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
