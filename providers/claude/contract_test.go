package claude

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

// helloServer answers every Messages API request with a reply, reporting a prompt cache write
// of 8 of its 12 input tokens, or a read of them when the same request came before, and counts
// the requests.
func helloServer(t *testing.T) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	return providertest.PromptCacheServer(t, func(hit bool) string {
		read, written := 0, 8
		if hit {
			read, written = 8, 0
		}
		// the Messages API counts cache reads and writes apart from input_tokens; the client adds
		// them back.
		return fmt.Sprintf(`{
			"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
			"content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn",
			"usage":{"input_tokens":4,"output_tokens":2,"cache_read_input_tokens":%d,"cache_creation_input_tokens":%d}
		}`, read, written)
	})
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
	temperature := func(value float64) *float64 { return &value }

	for name, req := range map[string]gepa.ChatRequest{
		"an unknown role":                   {Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}, {Role: "narrator"}}},
		"tool parameters that are not JSON": {Messages: user, Tools: tool(`{"type":`)},
		"tool parameters that are an array": {Messages: user, Tools: tool(`[1]`)},
		"tool parameters that are a string": {Messages: user, Tools: tool(`"object"`)},
		"a strict format that is not JSON":  {Messages: user, ResponseFormat: format(`{"type":`, true)},
		"a strict format of plain text":     {Messages: user, ResponseFormat: format(`not json`, true)},
		"a strict format of spaces":         {Messages: user, ResponseFormat: format(" \n", true)},
		"a NaN temperature":                 {Messages: user, Temperature: temperature(math.NaN())},
		"an infinite temperature":           {Messages: user, Temperature: temperature(math.Inf(1))},
		"a temperature of minus infinity":   {Messages: user, Temperature: temperature(math.Inf(-1))},
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
		"a format that is not strict, of spaces":   {Messages: user, ResponseFormat: format(" \n", false)},
		"the largest finite temperature":           {Messages: user, Temperature: temperature(math.MaxFloat64)},
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
